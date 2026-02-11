from typing import Any, cast, ClassVar, Dict, List, Optional, Union
import uuid

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.workload import WorkloadSDK
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models import (
    CreateInternalProductionJob,
    InternalProductionJob,
    ProductionJobConfig,
)
from anyscale.client.openapi_client.models.archive_status import ArchiveStatus
from anyscale.client.openapi_client.models.create_job_queue_config import (
    CreateJobQueueConfig,
)
from anyscale.client.openapi_client.models.decorated_production_job import (
    DecoratedProductionJob,
)
from anyscale.client.openapi_client.models.job_queue_spec import JobQueueSpec
from anyscale.client.openapi_client.models.list_response_metadata import (
    ListResponseMetadata,
)
from anyscale.client.openapi_client.models.production_job import ProductionJob
from anyscale.client.openapi_client.models.ray_runtime_env_config import (
    RayRuntimeEnvConfig,
)
from anyscale.client.openapi_client.models.resource_tag_resource_type import (
    ResourceTagResourceType,
)
from anyscale.commands.util import flatten_tag_dict_to_api_list
from anyscale.compute_config.models import (
    ComputeConfig,
    ComputeConfigType,
    MultiResourceComputeConfig,
)
from anyscale.job.models import (
    JobConfig,
    JobLogMode,
    JobQueueConfig,
    JobRunState,
    JobRunStatus,
    JobState,
    JobStatus,
)
from anyscale.sdk.anyscale_client.models import Job
from anyscale.sdk.anyscale_client.models.ha_job_states import HaJobStates
from anyscale.sdk.anyscale_client.models.job_status import JobStatus as BackendJobStatus
from anyscale.utils.runtime_env import parse_requirements_file


logger = BlockLogger()

HA_JOB_STATE_TO_JOB_STATE = {
    HaJobStates.UPDATING: JobState.RUNNING,
    HaJobStates.RUNNING: JobState.RUNNING,
    HaJobStates.RESTARTING: JobState.RUNNING,
    HaJobStates.CLEANING_UP: JobState.RUNNING,
    HaJobStates.PENDING: JobState.STARTING,
    HaJobStates.AWAITING_CLUSTER_START: JobState.STARTING,
    HaJobStates.SUCCESS: JobState.SUCCEEDED,
    # ERRORED is a transient state that can transition to RESTARTING when retries remain.
    HaJobStates.ERRORED: JobState.RUNNING,
    HaJobStates.TERMINATED: JobState.FAILED,
    HaJobStates.BROKEN: JobState.FAILED,
    HaJobStates.OUT_OF_RETRIES: JobState.FAILED,
}

TERMINAL_HA_JOB_STATES = [
    HaJobStates.SUCCESS,
    HaJobStates.TERMINATED,
    HaJobStates.OUT_OF_RETRIES,
]

# TODO(praneethkaturi): This is a temporary mapping. The backend should accept
# user-facing JobState values directly instead of requiring conversion to
# HaJobStates. Once the backend API is updated, this mapping can be removed.
# Reverse mapping from JobState to HaJobStates for filtering in list operations
JOB_STATE_TO_HA_JOB_STATES: Dict[str, List[str]] = {
    JobState.SUCCEEDED: [HaJobStates.SUCCESS],
    JobState.FAILED: [
        HaJobStates.TERMINATED,
        HaJobStates.BROKEN,
        HaJobStates.OUT_OF_RETRIES,
    ],
    JobState.RUNNING: [
        HaJobStates.UPDATING,
        HaJobStates.RUNNING,
        HaJobStates.RESTARTING,
        HaJobStates.CLEANING_UP,
        HaJobStates.ERRORED,
    ],
    JobState.STARTING: [HaJobStates.PENDING, HaJobStates.AWAITING_CLUSTER_START],
}


def _normalize_state_filter(
    states: Optional[List[Union[JobState, str]]]
) -> Optional[List[str]]:
    """Normalize state filter to list of HaJobStates strings.

    Converts JobState enums or string values to backend HaJobStates format.
    """
    if states is None:
        return None

    ha_job_states_filter: List[str] = []
    for s in states:
        if isinstance(s, JobState):
            state_key = s.value
        elif isinstance(s, str):
            state_key = s.upper()
        else:
            raise TypeError(
                "'state_filter' entries must be JobState or str, "
                f"got {type(s).__name__}"
            )
        ha_states = JOB_STATE_TO_HA_JOB_STATES.get(state_key, [])
        ha_job_states_filter.extend(ha_states)

    return ha_job_states_filter if ha_job_states_filter else None


class PrivateJobSDK(WorkloadSDK):
    _POLLING_INTERVAL_SECONDS = 10.0

    def _populate_runtime_env(
        self,
        config: JobConfig,
        *,
        autopopulate_in_workspace: bool = True,
        cloud_id: str,
        workspace_requirements_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Populates a runtime_env from the config.

        Local directories specified in the 'working_dir' will be uploaded and
        replaced with the resulting remote URIs.

        Requirements files will be loaded and populated into the 'pip' field.

        If autopopulate_from_workspace is passed and this code is running inside a
        workspace, the following defaults will be applied:
            - 'working_dir' will be set to '.'.
            - 'pip' will be set to the workspace-managed requirements file.
        """
        cloud_resource_names = self._get_compute_config_cloud_resources(
            compute_config=config.compute_config, cloud=config.cloud
        )
        assert len(cloud_resource_names) > 0

        runtime_env: Dict[str, Any] = {}
        if len(cloud_resource_names) == 1:
            [runtime_env] = self.override_and_upload_local_dirs_single_deployment(
                [runtime_env],
                working_dir_override=config.working_dir,
                excludes_override=config.excludes,
                cloud_id=cloud_id,
                autopopulate_in_workspace=autopopulate_in_workspace,
                additional_py_modules=config.py_modules,
                py_executable_override=config.py_executable,
                cloud_resource_name=cloud_resource_names[0],
            )
        else:
            [runtime_env] = self.override_and_upload_local_dirs_multi_cloud_resource(
                [runtime_env],
                working_dir_override=config.working_dir,
                excludes_override=config.excludes,
                cloud_id=cloud_id,
                autopopulate_in_workspace=autopopulate_in_workspace,
                additional_py_modules=config.py_modules,
                py_executable_override=config.py_executable,
                cloud_resource_names=cloud_resource_names,
            )
        [runtime_env] = self.override_and_load_requirements_files(
            [runtime_env],
            requirements_override=config.requirements,
            workspace_requirements_path=workspace_requirements_path,
        )
        [runtime_env] = self.update_env_vars(
            [runtime_env], env_vars_updates=config.env_vars,
        )

        return runtime_env or None

    def _get_compute_config_cloud_resources(
        self, compute_config: Union[ComputeConfigType, str, None], cloud: Optional[str]
    ) -> List[Optional[str]]:
        if isinstance(compute_config, ComputeConfig):
            # single-cloud resource compute config
            return [compute_config.cloud_resource]

        if isinstance(compute_config, MultiResourceComputeConfig):
            return [config.cloud_resource for config in compute_config.configs]

        compute_config_id = self._resolve_compute_config_id(
            compute_config=compute_config, cloud=cloud
        )
        compute_template = self._client.get_compute_config(compute_config_id)
        if compute_template is None or compute_template.config is None:
            raise ValueError(
                f"The compute config '{compute_config_id}' does not exist."
            )

        if compute_template.config.deployment_configs is None:
            return [None]

        return [
            config.cloud_deployment
            for config in compute_template.config.deployment_configs
        ]

    def get_default_name(self) -> str:
        """Get a default name for the job.

        If running inside a workspace, this is generated from the workspace name,
        else it generates a random name.
        """
        # TODO(edoakes): generate two random words instead of UUID here.
        name = f"job-{self.get_current_workspace_name() or str(uuid.uuid4())}"
        self.logger.info(f"No name was specified, using default: '{name}'.")
        return name

    def job_config_to_internal_prod_job_conf(
        self, config: JobConfig, name: str, cloud_id: str, compute_config_id: str,
    ) -> ProductionJobConfig:
        build_id = None
        if config.containerfile is not None:
            build_id = self._image_sdk.build_image_from_containerfile(
                name=f"image-for-job-{name}",
                containerfile=self.get_containerfile_contents(config.containerfile),
                ray_version=config.ray_version,
            )
        elif config.image_uri is not None:
            build_id = self._image_sdk.registery_image(
                image_uri=config.image_uri,
                registry_login_secret=config.registry_login_secret,
                ray_version=config.ray_version,
            )

        if self._image_sdk.enable_image_build_for_tracked_requirements:
            requirements_path_to_be_populated_in_runtime_env = None
            requirements_path = self.client.get_workspace_requirements_path()
            if requirements_path is not None:
                requirements = parse_requirements_file(requirements_path)
                if requirements:
                    build_id = self._image_sdk.build_image_from_requirements(
                        name=f"image-for-job-{name}",
                        base_build_id=self.client.get_default_build_id(),
                        requirements=requirements,
                    )
        else:
            requirements_path_to_be_populated_in_runtime_env = (
                self.client.get_workspace_requirements_path()
            )

        if build_id is None:
            build_id = self.client.get_default_build_id()

        env_vars_from_workspace = self.client.get_workspace_env_vars()
        if env_vars_from_workspace:
            if config.env_vars:
                # the precedence should be cli > workspace
                env_vars_from_workspace.update(config.env_vars)
                config = config.options(env_vars=env_vars_from_workspace)
            else:
                config = config.options(env_vars=env_vars_from_workspace)

        runtime_env = self._populate_runtime_env(
            config,
            cloud_id=cloud_id,
            workspace_requirements_path=requirements_path_to_be_populated_in_runtime_env,
        )

        return ProductionJobConfig(
            entrypoint=config.entrypoint,
            runtime_env=runtime_env,
            build_id=build_id,
            compute_config_id=compute_config_id,
            max_retries=config.max_retries,
            timeout_s=config.timeout_s,
        )

    def create_job_queue_config(
        self, provided_job_queue_config: JobQueueConfig
    ) -> CreateJobQueueConfig:
        job_queue_spec: Optional[JobQueueSpec] = None

        provided_job_queue_spec = provided_job_queue_config.job_queue_spec

        if provided_job_queue_spec:
            compute_config_id = (
                self._resolve_compute_config_id(provided_job_queue_spec.compute_config)
                if provided_job_queue_spec.compute_config
                else None
            )

            job_queue_spec = JobQueueSpec(
                job_queue_name=provided_job_queue_spec.name,
                execution_mode=provided_job_queue_spec.execution_mode,
                compute_config_id=compute_config_id,
                max_concurrency=provided_job_queue_spec.max_concurrency,
                idle_timeout_sec=provided_job_queue_spec.idle_timeout_s,
                auto_termination_threshold_job_count=provided_job_queue_spec.auto_termination_threshold_job_count,
            )

        job_queue_config = CreateJobQueueConfig(
            priority=provided_job_queue_config.priority,
            target_job_queue_name=provided_job_queue_config.target_job_queue_name,
            job_queue_spec=job_queue_spec,
        )
        return job_queue_config

    def submit(self, config: JobConfig) -> str:
        name = config.name or self.get_default_name()
        compute_config_id, cloud_id = self.resolve_compute_config_and_cloud_id(
            compute_config=config.compute_config, cloud=config.cloud
        )

        project_id = self.client.get_project_id(
            parent_cloud_id=cloud_id, name=config.project
        )

        prod_job_config = self.job_config_to_internal_prod_job_conf(
            config=config,
            name=name,
            cloud_id=cloud_id,
            compute_config_id=compute_config_id,
        )

        job_queue_config: Optional[CreateJobQueueConfig] = None

        provided_job_queue_config = config.job_queue_config

        if provided_job_queue_config:
            job_queue_config = self.create_job_queue_config(provided_job_queue_config)

        job: InternalProductionJob = self.client.submit_job(
            CreateInternalProductionJob(
                name=name,
                project_id=project_id,
                workspace_id=self.client.get_current_workspace_id(),
                config=prod_job_config,
                job_queue_config=job_queue_config,
                tags=config.tags,
            )
        )

        self.logger.info(f"Job '{job.name}' submitted, ID: '{job.id}'.")
        self.logger.info(
            f"View the job in the UI: {self.client.get_job_ui_url(job.id)}"
        )
        return job.id

    _BACKEND_JOB_STATUS_TO_JOB_RUN_STATE: ClassVar[
        Dict[BackendJobStatus, JobRunState]
    ] = {
        BackendJobStatus.RUNNING: JobRunState.RUNNING,
        BackendJobStatus.COMPLETED: JobRunState.SUCCEEDED,
        BackendJobStatus.PENDING: JobRunState.STARTING,
        BackendJobStatus.STOPPED: JobRunState.FAILED,
        BackendJobStatus.SUCCEEDED: JobRunState.SUCCEEDED,
        BackendJobStatus.FAILED: JobRunState.FAILED,
        BackendJobStatus.UNKNOWN: JobRunState.UNKNOWN,
    }

    def _job_state_from_job_model(self, model: ProductionJob) -> JobState:
        ha_state = model.state.current_state if model.state else None
        return cast(JobState, HA_JOB_STATE_TO_JOB_STATE.get(ha_state, JobState.UNKNOWN))

    def _job_run_model_to_job_run_status(self, run: Job) -> JobRunStatus:
        state = self._BACKEND_JOB_STATUS_TO_JOB_RUN_STATE.get(
            run.status, JobRunState.UNKNOWN
        )
        return JobRunStatus(name=run.name, state=state)

    def prod_job_config_to_job_config(
        self, prod_job_config: ProductionJobConfig, name: str, project: str,
    ) -> JobConfig:
        runtime_env_config: RayRuntimeEnvConfig = prod_job_config.runtime_env if prod_job_config else None
        compute_config = self.get_user_facing_compute_config(
            prod_job_config.compute_config_id
        )

        # Get image_uri from build_id
        image_uri = None
        if prod_job_config.build_id:
            image_uri_obj = self.client.get_cluster_env_build_image_uri(
                prod_job_config.build_id, use_image_alias=True
            )
            if image_uri_obj:
                image_uri = image_uri_obj.image_uri

        return JobConfig(
            name=name,
            image_uri=image_uri,
            compute_config=compute_config,
            requirements=runtime_env_config.pip if runtime_env_config else None,
            working_dir=runtime_env_config.working_dir if runtime_env_config else None,
            env_vars=runtime_env_config.env_vars if runtime_env_config else None,
            py_executable=runtime_env_config.py_executable
            if runtime_env_config
            else None,
            entrypoint=prod_job_config.entrypoint,
            cloud=compute_config.cloud
            if compute_config and isinstance(compute_config, ComputeConfig)
            else None,
            max_retries=prod_job_config.max_retries
            if prod_job_config.max_retries is not None
            else -1,
            project=project,
        )

    def _job_model_to_status(self, model: ProductionJob, runs: List[Job]) -> JobStatus:
        state = self._job_state_from_job_model(model)
        project_model = self.client.get_project(model.project_id)
        project = (
            project_model.name
            if project_model is not None and project_model.name != "default"
            else None
        )

        prod_job_config: ProductionJobConfig = model.config
        config = self.prod_job_config_to_job_config(
            prod_job_config=prod_job_config, name=model.name, project=project
        )
        runs = [self._job_run_model_to_job_run_status(run) for run in runs]

        return JobStatus(
            name=model.name,
            id=model.id,
            state=state,
            runs=runs,
            config=config,
            creator_id=model.creator_id,
            created_at=model.created_at,
        )

    def _decorated_job_to_status(
        self, decorated_job: DecoratedProductionJob, runs: List[Job]
    ) -> JobStatus:
        """Convert DecoratedProductionJob to JobStatus without extra API call.

        This method works with DecoratedProductionJob directly, avoiding the need
        to make an additional get_job() API call since DecoratedProductionJob
        contains all the necessary fields (including project as MiniProject).
        """
        ha_state = decorated_job.state.current_state if decorated_job.state else None
        state = cast(
            JobState, HA_JOB_STATE_TO_JOB_STATE.get(ha_state, JobState.UNKNOWN)
        )

        # DecoratedProductionJob has project directly as MiniProject
        project = (
            decorated_job.project.name
            if decorated_job.project is not None
            and decorated_job.project.name != "default"
            else None
        )

        prod_job_config: ProductionJobConfig = decorated_job.config
        config = self.prod_job_config_to_job_config(
            prod_job_config=prod_job_config, name=decorated_job.name, project=project
        )
        runs = [self._job_run_model_to_job_run_status(run) for run in runs]

        return JobStatus(
            name=decorated_job.name,
            id=decorated_job.id,
            state=state,
            runs=runs,
            config=config,
            creator_id=decorated_job.creator_id,
            created_at=decorated_job.created_at,
        )

    def _resolve_to_job_model(
        self,
        *,
        name: Optional[str] = None,
        job_id: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> ProductionJob:
        if name is None and job_id is None:
            raise ValueError("One of 'name' or 'job_id' must be provided.")

        if name is not None and job_id is not None:
            raise ValueError("Only one of 'name' or 'job_id' can be provided.")

        if job_id is not None and (cloud is not None or project is not None):
            raise ValueError("'cloud' and 'project' should only be used with 'name'.")

        try:
            model: Optional[ProductionJob] = self.client.get_job(
                name=name,
                job_id=job_id,
                cloud=cloud,
                project=project,
                include_archived=include_archived,
            )
        except Exception as e:
            # Convert API exceptions to RuntimeError for user-friendly error messages
            if name is not None:
                raise RuntimeError(f"Job with name '{name}' was not found.") from e
            else:
                raise RuntimeError(f"Job with ID '{job_id}' was not found.") from e

        if model is None:
            if name is not None:
                raise RuntimeError(f"Job with name '{name}' was not found.")
            else:
                raise RuntimeError(f"Job with ID '{job_id}' was not found.")

        return model

    def status(
        self,
        *,
        name: Optional[str] = None,
        job_id: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> JobStatus:
        job_model = self._resolve_to_job_model(
            name=name,
            job_id=job_id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )
        runs = self.client.get_job_runs(job_model.id)
        return self._job_model_to_status(model=job_model, runs=runs)

    def terminate(
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        job_model = self._resolve_to_job_model(
            name=name,
            job_id=job_id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )
        self.client.terminate_job(job_model.id)
        self.logger.info(f"Marked job '{job_model.name}' for termination")
        return job_model.id

    def archive(
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        job_model = self._resolve_to_job_model(
            name=name,
            job_id=job_id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )

        ha_state = job_model.state.current_state if job_model.state else None
        if ha_state not in TERMINAL_HA_JOB_STATES:
            raise RuntimeError(
                f"Job with ID '{job_model.id}' has not reached a terminal state and cannot be archived."
            )

        self.client.archive_job(job_model.id)
        self.logger.info(f"Job {job_model.id} is successfully archived.")
        return job_model.id

    def delete(
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        """Delete a job and all associated job runs.

        The job must be in a terminal state (SUCCEEDED, TERMINATED, OUT_OF_RETRIES, or BROKEN).
        This action cannot be undone.

        Args:
            job_id: The ID of the job to delete.
            name: The name of the job (alternative to job_id).
            cloud: Cloud filter (only used with name).
            project: Project filter (only used with name).
            include_archived: Include archived jobs when searching by name.
                Ignored when using job_id.

        Returns:
            The ID of the deleted job.

        Raises:
            RuntimeError: If job not found or not in terminal state.
        """
        job_model = self._resolve_to_job_model(
            name=name,
            job_id=job_id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )

        # Validate terminal state (client-side check for better UX)
        ha_state = job_model.state.current_state if job_model.state else None
        if ha_state not in TERMINAL_HA_JOB_STATES:
            raise RuntimeError(
                f"Job '{job_model.name}' (ID: {job_model.id}) has not reached a terminal state "
                f"and cannot be deleted. Current state: {ha_state}"
            )

        self.client.delete_job(job_model.id)
        self.logger.info(
            f"Job '{job_model.name}' (ID: {job_model.id}) has been deleted."
        )
        return job_model.id

    def _stream_logs_for_job_run(
        self, job_run_id: str, next_page_token: Optional[str] = None,
    ) -> Optional[str]:
        """Stream logs for a job run and return updated pagination state.

        Args:
            job_run_id: The ID of the job run to stream logs for
            next_page_token: Token for fetching next page of logs

        Returns:
            next_page_token for the next iteration
        """
        try:
            logs, next_page_token = self.client.stream_logs_for_job_run(
                job_run_id=job_run_id, next_page_token=next_page_token,
            )

            # Print logs line by line
            for line in logs.splitlines():
                if line:  # Skip empty lines
                    print(line)

        except Exception as e:  # noqa: BLE001
            # Don't fail if log streaming fails
            self.logger.warning(f"Error streaming logs: {e}")

        return next_page_token

    def wait(  # noqa: PLR0912, PLR0913
        self,
        *,
        name: Optional[str] = None,
        job_id: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        state: Union[str, JobState] = JobState.SUCCEEDED,
        timeout_s: float = 1800,
        interval_s: float = _POLLING_INTERVAL_SECONDS,
        follow: bool = False,
        include_archived: bool = False,
    ):
        if not isinstance(timeout_s, (int, float)):
            raise TypeError("timeout_s must be a float")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be >= 0")

        if not isinstance(interval_s, (int, float)):
            raise TypeError("interval_s must be a float")
        if interval_s <= 0:
            raise ValueError("interval_s must be >= 0")

        if not isinstance(state, JobState):
            raise TypeError("'state' must be a JobState.")

        job_id_or_name = job_id or name
        job_model = self._resolve_to_job_model(
            name=name,
            job_id=job_id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )
        curr_state = self._job_state_from_job_model(job_model)

        self.logger.info(
            f"Waiting for job '{job_id_or_name}' to reach target state {state}, currently in state: {curr_state}"
        )

        next_page_token = None
        job_run_id = None
        logs_started = False

        for _ in self.timer.poll(timeout_s=timeout_s, interval_s=interval_s):
            job_model = self._resolve_to_job_model(
                name=name,
                job_id=job_id,
                cloud=cloud,
                project=project,
                include_archived=include_archived,
            )
            new_state = self._job_state_from_job_model(job_model)

            if new_state != curr_state:
                self.logger.info(
                    f"Job '{job_id_or_name}' transitioned from {curr_state} to {new_state}"
                )
                curr_state = new_state

            # Stream logs if enabled and job has a job run
            if follow and job_model.last_job_run_id:
                if not logs_started:
                    job_run_id = job_model.last_job_run_id
                    self.logger.info(f"Starting log stream for job run {job_run_id}")
                    logs_started = True

                if job_run_id:
                    next_page_token = self._stream_logs_for_job_run(
                        job_run_id=job_run_id, next_page_token=next_page_token,
                    )

            if curr_state == state:
                self.logger.info(
                    f"Job '{job_id_or_name}' reached target state, exiting"
                )
                break

            if JobState.is_terminal(curr_state):
                raise RuntimeError(
                    f"Job '{job_id_or_name}' reached terminal state '{curr_state}', and will not reach '{state}'."
                )
        else:
            raise TimeoutError(
                f"Job '{job_id_or_name}' did not reach target state {state} within {timeout_s}s. Last seen state: {curr_state}."
            )

    def _resolve_job_run_id(
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        run: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        job_model = self._resolve_to_job_model(
            name=name,
            job_id=job_id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )

        last_job_run_id = job_model.last_job_run_id
        if last_job_run_id is None:
            return ""
        if run is None:
            job_run_id = last_job_run_id
        else:
            runs: List[Job] = self.client.get_job_runs(job_model.id)
            for job_run in runs:
                if job_run.name == run:
                    job_run_id = job_run.id
                    break
            else:
                raise ValueError(
                    f"Job run '{run}' was not found for job '{job_id or name}'."
                )

        return job_run_id

    def get_logs(
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        run: Optional[str] = None,
        mode: Union[str, JobLogMode] = JobLogMode.TAIL,
        max_lines: Optional[int] = None,
        include_archived: bool = False,
    ) -> str:
        if max_lines is not None:
            if not isinstance(max_lines, int):
                raise TypeError("max_lines must be an int")
            if max_lines <= 0:
                raise ValueError("max_lines must be > 0")

        job_run_id = self._resolve_job_run_id(
            job_id=job_id,
            name=name,
            cloud=cloud,
            project=project,
            run=run,
            include_archived=include_archived,
        )

        head = mode == JobLogMode.HEAD
        return self.client.logs_for_job_run(
            job_run_id=job_run_id, head=head, max_lines=max_lines
        )

    def add_tags(
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        tags: Dict[str, str],
        include_archived: bool = False,
    ) -> None:
        if not tags:
            raise ValueError("At least one tag must be provided.")

        if job_id is not None:
            resource_id = job_id
        else:
            if name is None:
                raise ValueError("Either 'job_id' or 'name' must be provided.")
            model = self._resolve_to_job_model(
                job_id=None,
                name=name,
                cloud=cloud,
                project=project,
                include_archived=include_archived,
            )
            resource_id = model.id

        self.client.upsert_resource_tags(ResourceTagResourceType.JOB, resource_id, tags)

    def remove_tags(
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        keys: List[str],
        include_archived: bool = False,
    ) -> None:
        if not keys:
            raise ValueError("At least one tag key must be provided.")

        if job_id is not None:
            resource_id = job_id
        else:
            if name is None:
                raise ValueError("Either 'job_id' or 'name' must be provided.")
            model = self._resolve_to_job_model(
                job_id=None,
                name=name,
                cloud=cloud,
                project=project,
                include_archived=include_archived,
            )
            resource_id = model.id

        self.client.delete_resource_tags(ResourceTagResourceType.JOB, resource_id, keys)

    def list_tags(
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> Dict[str, str]:
        """List tags for a job as a key/value mapping."""
        if job_id is not None:
            resource_id = job_id
        else:
            if name is None:
                raise ValueError("Either 'job_id' or 'name' must be provided.")
            model = self._resolve_to_job_model(
                job_id=None,
                name=name,
                cloud=cloud,
                project=project,
                include_archived=include_archived,
            )
            resource_id = model.id
        records = self.client.list_resource_tags(
            ResourceTagResourceType.JOB, resource_id
        )
        return {r.key: r.value for r in records if r and r.key is not None}

    def list(  # noqa: PLR0913
        self,
        *,
        name: Optional[str] = None,
        job_id: Optional[str] = None,
        project: Optional[str] = None,
        project_id: Optional[str] = None,
        cloud: Optional[str] = None,
        include_all_users: bool = False,
        include_archived: bool = False,
        state_filter: Optional[List[Union[JobState, str]]] = None,
        tags_filter: Optional[Dict[str, List[str]]] = None,
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        sort_field: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> ResultIterator[JobStatus]:
        """List jobs with filtering and pagination.

        Args:
            name: Filter by job name.
            job_id: Fetch a specific job by ID.
            project: Filter by project name.
            project_id: [DEPRECATED] Filter by project ID. Use 'project' instead.
            cloud: Filter by cloud name.
            include_all_users: Include jobs from all users.
            include_archived: Include archived jobs.
            state_filter: Filter by job states (list of JobState or str).
            tags_filter: Filter by tags (dict of key to list of values).
            page_size: Number of items per page. Defaults to server default.
            max_items: Maximum total items to return.
            sort_field: Field to sort by (CREATED_AT, NAME, STATUS, etc.).
            sort_order: Sort order (ASC or DESC).

        Returns:
            ResultIterator that lazily fetches pages of JobStatus objects.
        """
        # Validate page_size
        if page_size is not None and (page_size <= 0 or page_size > 100):
            raise ValueError("page_size must be between 1 and 100, inclusive.")

        # Handle project_id parameter
        if project_id is not None and project is None:
            # Resolve project_id to project name for consistency
            project_model = self.client.get_project(project_id)
            if project_model:
                project = project_model.name

        # If job_id provided, fetch single job
        if job_id is not None:
            try:
                job_model = self._resolve_to_job_model(job_id=job_id)
                runs = self.client.get_job_runs(job_model.id)
                status = self._job_model_to_status(model=job_model, runs=runs)
                results: list = [status]
            except RuntimeError:
                # Job not found - return empty iterator (consistent with --name behavior)
                results = []

            def _fetch_single_job_page(_token: Optional[str]):
                class PageResponse:
                    def __init__(self):
                        self.results = results if _token is None else []
                        self.metadata = ListResponseMetadata(
                            total=len(results), next_paging_token=None
                        )

                return PageResponse()

            return ResultIterator(
                page_token=None,
                max_items=len(results),
                fetch_page=_fetch_single_job_page,
                parse_fn=lambda x: x,
            )

        # Resolve cloud and project IDs
        cloud_id = self.client.get_cloud_id(cloud_name=cloud) if cloud else None
        resolved_project_id = None
        if project:
            resolved_project_id = self.client.get_project_id(
                parent_cloud_id=cloud_id, name=project
            )

        # Get creator_id for filtering
        creator_id = None
        if not include_all_users:
            user = self.client.get_user_info()
            creator_id = user.id if user else None

        # Convert tags dict to API format using utility
        backend_tags_filter = flatten_tag_dict_to_api_list(tags_filter)

        archive_status = (
            ArchiveStatus.ALL if include_archived else ArchiveStatus.NOT_ARCHIVED
        )

        # Convert user-facing JobState values to backend HaJobStates
        ha_job_states_filter = _normalize_state_filter(state_filter)

        def _fetch_page(token: Optional[str]):
            kwargs: Dict[str, Any] = {
                "name": name,
                "project_id": resolved_project_id,
                "creator_id": creator_id,
                "archive_status": archive_status,
                "state_filter": ha_job_states_filter,
                "tags_filter": backend_tags_filter,
                "paging_token": token,
                "sort_field": sort_field,
                "sort_order": sort_order,
            }
            if page_size is not None:
                kwargs["count"] = page_size
            return self.client.list_jobs(**kwargs)

        def _parse_job(decorated_job: DecoratedProductionJob) -> JobStatus:
            runs = self.client.get_job_runs(decorated_job.id)
            # Use DecoratedProductionJob directly instead of making extra API call
            return self._decorated_job_to_status(decorated_job=decorated_job, runs=runs)

        return ResultIterator(
            page_token=None,
            max_items=max_items,
            fetch_page=_fetch_page,
            parse_fn=_parse_job,
        )
