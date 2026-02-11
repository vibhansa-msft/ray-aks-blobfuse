from typing import Optional

from anyscale._private.anyscale_client.common import AnyscaleClientInterface
from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk.base_sdk import BaseSDK, Timer
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models.create_schedule import CreateSchedule
from anyscale.client.openapi_client.models.decorated_schedule import DecoratedSchedule
from anyscale.client.openapi_client.models.list_response_metadata import (
    ListResponseMetadata,
)
from anyscale.client.openapi_client.models.production_job_config import (
    ProductionJobConfig,
)
from anyscale.client.openapi_client.models.schedule_config import (
    ScheduleConfig as BackendScheduleConfig,
)
from anyscale.job._private.job_sdk import PrivateJobSDK
from anyscale.job.models import JobConfig
from anyscale.schedule.models import ScheduleConfig, ScheduleState, ScheduleStatus
from anyscale.util import get_endpoint


logger = BlockLogger()

MAX_PAGE_SIZE = 50


class PrivateScheduleSDK(BaseSDK):
    def __init__(
        self,
        *,
        logger: Optional[BlockLogger] = None,
        client: Optional[AnyscaleClientInterface] = None,
        timer: Optional[Timer] = None,
    ):
        super().__init__(logger=logger, client=client, timer=timer)
        self._job_sdk = PrivateJobSDK(logger=self.logger, client=self.client)

    def apply(self, config: ScheduleConfig) -> str:
        job_config = config.job_config
        assert isinstance(job_config, JobConfig)
        name = job_config.name or self._job_sdk.get_default_name()

        compute_config_id, cloud_id = self._job_sdk.resolve_compute_config_and_cloud_id(
            compute_config=job_config.compute_config, cloud=job_config.cloud
        )

        project_id = self.client.get_project_id(
            parent_cloud_id=cloud_id, name=job_config.project
        )

        job_queue_config = None
        if job_config.job_queue_config is not None:
            job_queue_config = self._job_sdk.create_job_queue_config(
                job_config.job_queue_config
            )

        schedule: DecoratedSchedule = self.client.apply_schedule(
            CreateSchedule(
                name=name,
                project_id=project_id,
                config=self._job_sdk.job_config_to_internal_prod_job_conf(
                    config=job_config,
                    name=name,
                    cloud_id=cloud_id,
                    compute_config_id=compute_config_id,
                ),
                job_queue_config=job_queue_config,
                schedule=BackendScheduleConfig(
                    cron_expression=config.cron_expression, timezone=config.timezone,
                ),
            )
        )

        self.logger.info(f"Schedule '{name}' submitted, ID: '{schedule.id}'.")

        return schedule.id

    def _resolve_to_schedule_model(
        self,
        *,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
    ) -> DecoratedSchedule:
        if name is None and id is None:
            raise ValueError("One of 'name' or 'id' must be provided.")

        if name is not None and id is not None:
            raise ValueError("Only one of 'name' or 'id' can be provided.")

        if id is not None and (cloud is not None or project is not None):
            raise ValueError("'cloud' and 'project' should only be used with 'name'.")

        model: Optional[DecoratedSchedule] = self.client.get_schedule(
            name=name, id=id, cloud=cloud, project=project,
        )

        if model is None:
            if name is not None:
                raise RuntimeError(f"Schedule with name '{name}' was not found.")
            else:
                raise RuntimeError(f"Schedule with ID '{id}' was not found.")

        return model

    def set_state(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        state: ScheduleState,
    ) -> str:
        schedule_model = self._resolve_to_schedule_model(
            name=name, id=id, cloud=cloud, project=project
        )
        is_paused = state == ScheduleState.DISABLED
        self.client.set_schedule_state(id=schedule_model.id, is_paused=is_paused)
        self.logger.info(f"Set schedule '{schedule_model.name}' to state {state}")
        return schedule_model.id

    def _schedule_model_to_status(self, model: DecoratedSchedule) -> ScheduleStatus:
        # TODO: Consider batching get_project calls once the backend API supports
        # fetching multiple projects in a single request.
        project_model = self.client.get_project(model.project_id)
        project = (
            project_model.name
            if project_model is not None and project_model.name != "default"
            else None
        )

        prod_job_config: ProductionJobConfig = model.config
        job_config = self._job_sdk.prod_job_config_to_job_config(
            prod_job_config=prod_job_config, name=model.name, project=project
        )

        config = ScheduleConfig(
            job_config=job_config,
            cron_expression=model.schedule.cron_expression,
            timezone=model.schedule.timezone,
        )

        state = (
            ScheduleState.ENABLED
            if model.next_trigger_at is not None
            else ScheduleState.DISABLED
        )

        return ScheduleStatus(id=model.id, name=model.name, config=config, state=state)

    def status(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> ScheduleStatus:
        schedule_model = self._resolve_to_schedule_model(
            name=name, id=id, cloud=cloud, project=project
        )
        return self._schedule_model_to_status(model=schedule_model)

    def trigger(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        schedule_model = self._resolve_to_schedule_model(
            name=name, id=id, cloud=cloud, project=project
        )
        self.client.trigger_schedule(id=schedule_model.id)
        self.logger.info(f"Triggered job for schedule '{schedule_model.name}'.")
        return schedule_model.id

    def delete(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        """Delete a schedule.

        If the schedule is active, it will be automatically paused before deletion.
        The schedule must have no active triggered jobs.

        Args:
            id: The schedule ID.
            name: The schedule name (requires cloud and project).
            cloud: Cloud name (required with name).
            project: Project name (required with name).

        Returns:
            The ID of the deleted schedule.
        """
        schedule_model = self._resolve_to_schedule_model(
            name=name, id=id, cloud=cloud, project=project
        )
        self.client.delete_schedule(schedule_id=schedule_model.id)
        self.logger.info(f"Schedule '{schedule_model.name}' deleted.")
        return schedule_model.id

    def url(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        """Get the web UI URL for a schedule.

        Args:
            id: The schedule ID.
            name: The schedule name (requires cloud and project).
            cloud: Cloud name (required with name).
            project: Project name (required with name).

        Returns:
            The URL string for viewing the schedule in the Anyscale console.
        """
        schedule_model = self._resolve_to_schedule_model(
            id=id, name=name, cloud=cloud, project=project
        )
        return get_endpoint(f"/scheduled-jobs/{schedule_model.id}")

    def list(  # noqa: PLR0913
        self,
        *,
        name: Optional[str] = None,
        schedule_id: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        creator_id: Optional[str] = None,
        include_all_users: bool = False,
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
    ) -> ResultIterator[ScheduleStatus]:
        """List schedules with filtering and pagination.

        Args:
            name: Filter by schedule name.
            schedule_id: Fetch a specific schedule by ID.
            project: Filter by project name.
            cloud: Filter by cloud name.
            creator_id: Filter by creator ID.
            include_all_users: Include schedules from all users.
            page_size: Number of items per page.
            max_items: Maximum total items to return.

        Returns a ResultIterator that lazily fetches pages of schedules.
        """
        # Validate page_size
        if page_size is not None and (page_size <= 0 or page_size > MAX_PAGE_SIZE):
            raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}.")

        # If schedule_id provided, fetch single schedule
        if schedule_id is not None:
            schedule_model = self._resolve_to_schedule_model(id=schedule_id)
            status = self._schedule_model_to_status(model=schedule_model)

            def _fetch_single_page(token: Optional[str]):
                class SingleItemResponse:
                    results = [status] if token is None else []
                    metadata = ListResponseMetadata(total=1, next_paging_token=None)

                return SingleItemResponse()

            return ResultIterator(
                page_token=None,
                max_items=1,
                fetch_page=_fetch_single_page,
                parse_fn=lambda x: x,
            )

        # Resolve cloud and project IDs
        cloud_id = self.client.get_cloud_id(cloud_name=cloud) if cloud else None
        project_id = None
        if project:
            project_id = self.client.get_project_id(
                parent_cloud_id=cloud_id, name=project
            )

        # Auto-populate creator_id if not include_all_users and creator_id not specified
        resolved_creator_id = creator_id
        if not include_all_users and creator_id is None:
            user = self.client.get_user_info()
            resolved_creator_id = user.id if user else None

        def _fetch_page(token: Optional[str]):
            return self.client.list_schedules(
                name=name,
                project_id=project_id,
                cloud_id=cloud_id,
                creator_id=resolved_creator_id,
                count=page_size,
                paging_token=token,
            )

        def _parse_schedule(decorated_schedule: DecoratedSchedule) -> ScheduleStatus:
            return self._schedule_model_to_status(model=decorated_schedule)

        return ResultIterator(
            page_token=None,
            max_items=max_items,
            fetch_page=_fetch_page,
            parse_fn=_parse_schedule,
        )
