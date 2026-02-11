from dataclasses import dataclass, field
from typing import ClassVar, Dict, Union

import tzlocal

from anyscale._private.models import ModelBase, ModelEnum
from anyscale.job.models import JobConfig


UTC_TZ_NAME = "UTC"


@dataclass(frozen=True)
class ScheduleConfig(ModelBase):
    """Configuration options for a schedule."""

    __doc_py_example__ = """\
from anyscale.schedule.models import ScheduleConfig
from anyscale.job.models import JobConfig

config = ScheduleConfig(
    job_config=JobConfig(
        entrypoint="python main.py"
    ),
    timezone="local",
    cron_expression="0 0 * * *"
)
"""

    __doc_yaml_example__ = """\
timezone: local
cron_expression: 0 0 * * * *
job_config:
    name: my-schedule-job
    entrypoint: python main.py
    max_retries: 5
"""

    job_config: Union[Dict, JobConfig] = field(
        metadata={"docstring": "Configuration of the job."}
    )

    def _validate_job_config(self, job_config: Union[Dict, JobConfig]) -> JobConfig:
        if isinstance(job_config, dict):
            job_config = JobConfig.from_dict(job_config)

        if not isinstance(job_config, JobConfig):
            raise TypeError("'job_config' must be a JobConfig.")

        return job_config

    cron_expression: str = field(
        metadata={
            "docstring": "A cron expression to define the frequency at which to run this cron job, for example '0 0 * * *' is a cron expression that means 'run at midnight'. Visit crontab.guru to construct a precise cron_expression."
        }
    )

    def _validate_cron_expression(self, cron_expression: str):
        # validation of cron expression is primarily handled by the backend when
        # applying the schedule. simply check if it is a non-empty string here.
        if not isinstance(cron_expression, str):
            raise TypeError("'cron_expression' must be a string.")

        if not cron_expression:
            raise ValueError("'cron_expression' cannot be empty.")

    timezone: str = field(
        default=UTC_TZ_NAME,
        metadata={
            "docstring": "The timezone in which to interpret the cron_expression. Default is Universal time (UTC). You can specify 'local' to use the local timezone"
        },
    )

    def _validate_timezone(self, timezone: str) -> str:
        # validation of timezone is primarily handled by the backend when
        # applying the schedule. simply check if it is a non-empty string here.
        if not isinstance(timezone, str):
            raise TypeError("'timezone' must be a string.")
        if not timezone:
            raise ValueError("'timezone' cannot be empty.")
        elif timezone == "local":
            return tzlocal.get_localzone_name()
        else:
            return timezone


class ScheduleState(ModelEnum):
    """Current state of a schedule."""

    ENABLED = "ENABLED"
    DISABLED = "DISABLED"

    __docstrings__: ClassVar[Dict[str, str]] = {
        ENABLED: "The schedule is enabled. Jobs will be started according to this jobs cron expression.",
        DISABLED: "The schedule is disabled. No jobs will be started until the schedule is reenabled.",
    }


@dataclass(frozen=True)
class ScheduleStatus(ModelBase):
    """Current status of a schedule."""

    __doc_py_example__ = """\
import anyscale
from anyscale.schedule.models import ScheduleStatus
status: ScheduleStatus = anyscale.schedule.status(name="my-schedule")
"""

    __doc_cli_example__ = """\
$ anyscale schedule status -n my-schedule
id: cronjob_dfhqufws6s3issltpjgrdzcgyc
name: my-schedule
state: ENABLED
"""

    config: ScheduleConfig = field(
        metadata={"docstring": "Configuration of the schedule."}
    )

    def _validate_config(self, config: ScheduleConfig):
        if not isinstance(config, ScheduleConfig):
            raise TypeError("'config' must be a ScheduleConfig.")

    id: str = field(
        metadata={
            "docstring": "Unique ID of the schedule (generated when the schedule is first submitted)."
        }
    )

    def _validate_id(self, id: str):  # noqa: A002
        if not isinstance(id, str):
            raise TypeError("'id' must be a string.")

    name: str = field(metadata={"docstring": "Name of the schedule."},)

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")

    state: Union[str, ScheduleState] = field(
        metadata={"docstring": "Current state of the schedule."}
    )

    def _validate_state(self, state: Union[str, ScheduleState]) -> ScheduleState:
        return ScheduleState.validate(state)
