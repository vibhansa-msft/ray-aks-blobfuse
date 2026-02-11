import os
from typing import Any, Dict, List, Optional

import click
import tzlocal
import yaml

from anyscale.anyscale_pydantic import (
    BaseModel,
    Field,
    validator,
)
from anyscale.authenticate import get_auth_api_client
from anyscale.cli_logger import LogsLogger
from anyscale.client.openapi_client.api.default_api import DefaultApi
from anyscale.client.openapi_client.models.create_schedule import CreateSchedule
from anyscale.client.openapi_client.models.production_job_config import (
    ProductionJobConfig,
)
from anyscale.client.openapi_client.models.schedule_config import (
    ScheduleConfig as APIScheduleConfig,
)
from anyscale.controllers.base_controller import BaseController
from anyscale.models.job_model import JobConfig
from anyscale.project_utils import infer_project_id
from anyscale.tables import SchedulesTable
from anyscale.util import (
    AnyscaleEndpointFormatter,
    populate_unspecified_cluster_configs_from_current_workspace,
    validate_job_config_dict,
)
from anyscale.utils.runtime_env import override_runtime_env_config
from anyscale.utils.workload_types import Workload


log = LogsLogger()


class ScheduleConfig(BaseModel):
    cron_expression: str = Field(
        ...,
        description="A cron expression to define the frequency at which to run this cron job, for example '0 0 * * *' is a cron expression that means 'run at midnight'. Visit crontab.guru to construct a precise cron_expression.",
    )
    timezone: Optional[str] = Field(
        None,
        description="The timezone in which to interpret the cron_expression. Default is Universal time (UTC). In the CLI, you can specify 'local' to use the local timezone",
    )

    @validator("timezone")
    def rewrite_timezone(cls, timezone):
        if timezone == "local":
            local_tz = get_local_timezone()
            log.info(f"Inferred local timezone to be `{local_tz}`")
            return local_tz
        else:
            return timezone


class CreateScheduleConfig(JobConfig):
    schedule: ScheduleConfig


def load_yaml_file_as_dict(config_file: str) -> Dict[str, Any]:
    if not os.path.exists(config_file):
        raise click.ClickException(f"Config file {config_file} not found.")

    with open(config_file) as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def apply_overrides_to_dict(start_dict, **overrides):
    for key, value in overrides.items():
        if value is not None:
            start_dict[key] = value


def load_yaml_file_with_overrides(config_file: str, **overrides):
    d = load_yaml_file_as_dict(config_file)
    apply_overrides_to_dict(d, **overrides)
    return d


def load_schedule_pydantic(config: Dict[str, Any]):
    schedule_config = CreateScheduleConfig.parse_obj(config)
    return schedule_config


def get_local_timezone() -> str:
    return tzlocal.get_localzone_name()


class ScheduleController(BaseController):
    def __init__(
        self, log: Optional[LogsLogger] = None,
    ):
        if log is None:
            log = LogsLogger()

        super().__init__()
        self.log = log
        self.log.open_block("Output")
        self.schedule_api = ScheduleApi(self.api_client)
        self.endpoint_formatter = AnyscaleEndpointFormatter()

    def apply(
        self,
        schedule_config_file: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        schedule_config = self._resolve_config(
            schedule_config_file, name=name, description=description
        )
        id = self.schedule_api.apply(schedule_config)  # noqa: A001
        self.url(id)
        self.log.info(f"Schedule {id} has been updated successfully.")
        return id

    def _resolve_config(
        self, schedule_config_file: str, **overrides
    ) -> CreateScheduleConfig:
        schedule_config_dict = load_yaml_file_with_overrides(
            schedule_config_file, **overrides
        )
        validate_job_config_dict(schedule_config_dict, self.api_client)

        # If running in a workspace, auto-populate unspecified fields.
        schedule_config_dict = populate_unspecified_cluster_configs_from_current_workspace(
            schedule_config_dict, self.anyscale_api_client,
        )

        schedule_config: CreateScheduleConfig = CreateScheduleConfig.parse_obj(
            schedule_config_dict
        )

        schedule_config.project_id = infer_project_id(
            self.anyscale_api_client,
            self.api_client,
            self.log,
            project_id=schedule_config.project_id,
            cluster_compute_id=schedule_config.compute_config_id,
            cluster_compute=schedule_config.compute_config,
            cloud=schedule_config.cloud,
        )
        schedule_config.runtime_env = override_runtime_env_config(
            runtime_env=schedule_config.runtime_env,
            anyscale_api_client=self.anyscale_api_client,
            api_client=self.api_client,
            workload_type=Workload.SCHEDULED_JOBS,
            compute_config_id=schedule_config.compute_config_id,
            log=self.log,
        )

        return schedule_config

    def resolve_file_name_or_id(
        self,
        *,
        schedule_config_file: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
    ) -> str:
        """Try to resolve a Schedule Config File, Name, or Id to a Schedule Id

        At least one of the inputs must be non null.

        Args:
            schedule_config_file (Optional[str]): A file name
            id (Optional[str]): An id
            name (Optional[str]): a name

        Raises:
            click.ClickException: If the id could not be uniquely determined.

        Returns:
            str: the id of the schedule
        """

        assert_single_parameter(
            schedule_config_file=schedule_config_file, id=id, name=name
        )

        if id:
            try:
                result = self.schedule_api.get(id)
                return result.id
            except Exception as e:  # noqa: BLE001
                raise click.ClickException(f"No schedule found with id `{id}`") from e
        elif name:
            schedules = self.schedule_api.list(name=name)
            schedules = [schedule for schedule in schedules if schedule.name == name]

            if len(schedules) == 1:
                return schedules[0].id
            elif len(schedules) > 1:
                self.log.error(str(SchedulesTable(schedules)))
                raise click.ClickException(
                    f"Found multiple schedules matching the name {name}"
                )
            else:
                raise click.ClickException(
                    f"Found no schedules matching the name {name}"
                )
        elif schedule_config_file:
            config = self._resolve_config(schedule_config_file)
            name = config.name
            project_id = config.project_id

            # Schedules are unique by name and project_id
            schedules = self.schedule_api.list(name=name, project_id=project_id)
            if len(schedules) == 1:
                return schedules[0].id
            elif len(schedules) > 1:
                self.log.error(str(SchedulesTable(schedules)))
                raise click.ClickException(
                    f"Unable to determine which of the following schedules this file `{schedule_config_file}` is referring to. Please make sure `name` and `project_id` are specified in your schedule yaml."
                )
            else:
                raise click.ClickException(
                    f"Found no schedules matching the file `{schedule_config_file}"
                )

        raise click.ClickException(
            "Please specify a file, a name, or an id to resolve your schedule"
        )

    def list(self, project_id=None, name=None, creator_id=None, id=None):  # noqa: A002
        if id:
            if any([project_id, name, creator_id]):
                raise click.ClickException(
                    "If `id` is specified, no other filters can be specified."
                )
            schedules = [self.schedule_api.get(id)]
        else:
            schedules = self.schedule_api.list(
                project_id=project_id, name=name, creator_id=creator_id
            )
        print(SchedulesTable(schedules))

    def pause(self, id, is_paused=True):  # noqa: A002
        schedule = self.schedule_api.pause(id, is_paused)
        text = "paused" if is_paused else "resumed"
        self.url(id)
        self.log.info(f"Scheduled Job {id} has been {text}.")
        if not is_paused:
            self.log.info(f"Next trigger at {schedule.next_trigger_at}")

    def trigger(self, id):  # noqa: A002
        created_job = self.schedule_api.trigger(id)
        self.url(id)
        self.log.info(f"Successfully created Job with id {created_job.id}")
        url = self.endpoint_formatter.get_job_endpoint(created_job.id)
        self.log.info(f"View your Production Job at {url}")

    def url(self, id) -> str:  # noqa: A002
        url = self.endpoint_formatter.get_schedule_endpoint(id)
        self.log.info(f"View your schedule at {url}")
        return url


class ScheduleApi:
    def __init__(self, api_client: DefaultApi = None):
        self.api_client = api_client or get_auth_api_client().api_client

    def apply(self, schedule_config: CreateScheduleConfig) -> str:
        formatted = format_create_schedule_config(schedule_config)
        job = self.api_client.create_or_update_job_api_v2_experimental_cron_jobs_put(
            formatted
        ).result
        return job.id

    def list(self, project_id=None, creator_id=None, name=None) -> List[Any]:
        return self.api_client.list_cron_jobs_api_v2_experimental_cron_jobs_get(
            project_id=project_id, creator_id=creator_id, name=name
        ).results

    def get(self, id: str) -> Any:  # noqa: A002
        return self.api_client.get_cron_job_api_v2_experimental_cron_jobs_cron_job_id_get(
            id
        ).result

    def pause(self, cron_job_id: str, is_paused=True):
        return self.api_client.pause_cron_job_api_v2_experimental_cron_jobs_cron_job_id_pause_post(
            cron_job_id, {"is_paused": is_paused}
        ).result

    def trigger(self, cron_job_id: str):
        return self.api_client.trigger_cron_job_api_v2_experimental_cron_jobs_cron_job_id_trigger_post(
            cron_job_id
        ).result

    def get_current_user(self) -> str:
        user = self.api_client.get_user_info_api_v2_userinfo_get().result
        return user.id


def format_schedule_config(schedule: ScheduleConfig):
    return APIScheduleConfig(
        cron_expression=schedule.cron_expression, timezone=schedule.timezone,
    )


def format_api_job_config(job_config: JobConfig):
    return ProductionJobConfig(
        entrypoint=job_config.entrypoint,
        runtime_env=job_config.runtime_env,
        build_id=job_config.build_id,
        compute_config_id=job_config.compute_config_id,
        max_retries=job_config.max_retries,
    )


def format_create_schedule_config(schedule_config: CreateScheduleConfig):
    return CreateSchedule(
        name=schedule_config.name,
        description=schedule_config.description,
        project_id=schedule_config.project_id,
        config=format_api_job_config(schedule_config),
        schedule=format_schedule_config(schedule_config.schedule),
    )


def assert_single_parameter(**kwargs) -> None:
    keys_not_none = [a for a in kwargs if kwargs[a] is not None]
    if len(keys_not_none) == 0:
        raise click.ClickException(
            f"At least one of the following flags must be set: {kwargs.keys()}"
        )
    if len(keys_not_none) > 1:
        raise click.ClickException(
            f"Only one of the following flags can be set: {keys_not_none}"
        )
