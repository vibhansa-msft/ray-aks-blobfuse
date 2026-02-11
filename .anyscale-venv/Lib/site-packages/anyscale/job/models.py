from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Dict, List, Optional, Union

from anyscale._private.models import ModelBase, ModelEnum
from anyscale._private.workload import WorkloadConfig
from anyscale.commands import command_examples
from anyscale.shared_anyscale_utils.utils import INT_MAX


class JobQueueExecutionMode(ModelEnum):
    """Execution mode for job queues."""

    FIFO = "FIFO"
    LIFO = "LIFO"
    PRIORITY = "PRIORITY"

    __docstrings__: ClassVar[Dict[str, str]] = {
        FIFO: "Executes jobs in chronological order ('first in, first out')",
        LIFO: "Executes jobs in reversed chronological order ('last in, first out')",
        PRIORITY: "Executes jobs in the order induced by ordering their priorities in ascending order, "
        "with 0 being the highest priority",
    }


@dataclass(frozen=True)
class JobQueueSpec(ModelBase):
    """Options defining a job queue.

    When the first job with a given job queue spec is submitted, the job queue will be created.
    Subsequent jobs with the same job queue spec will reuse the queue instead of creating another one.

    Jobs can also target an existing queue using the name parameter.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.job.models import JobQueueSpec, JobQueueExecutionMode

job_queue_spec = JobQueueSpec(
    # Unique name that can be used to target this queue by other jobs.
    name="my-job-queue",
    execution_mode=JobQueueExecutionMode.FIFO,
    # Name of a compute config that will be used to create a cluster to execute jobs in this queue.
    # Must match the compute config of the job if specified.
    compute_config="my-compute-config:1",
    max_concurrency=5,
    idle_timeout_s=3600,
)
"""

    __doc_yaml_example__ = """\
job_queue_spec:
    # Unique name that can be used to target this queue by other jobs.
    name: my-job-queue
    execution_mode: FIFO
    # Name of a compute config that will be used to create a cluster to execute jobs in this queue.
    # Must match the compute config of the job if specified.
    compute_config: my-compute-config:1
    max_concurrency: 5
    idle_timeout_s: 3600
"""

    idle_timeout_s: int = field(
        metadata={
            "docstring": "Timeout that the job queue cluster will be kept running while no jobs are running.",
        }
    )

    def _validate_idle_timeout_s(self, idle_timeout_s: int):
        if not isinstance(idle_timeout_s, int):
            raise TypeError(
                f"'idle_timeout_sec' must be an int (it is {type(idle_timeout_s)})."
            )

        elif idle_timeout_s < 0:
            raise ValueError("'idle_timeout_s' should be >= 0")

    name: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Name of the job queue that can be used to target it when submitting future jobs. "
            "The name of a job queue must be unique within a project.",
        },
    )

    def _validate_name(self, name: Optional[str]):
        if name is not None and not isinstance(name, str):
            raise TypeError(f"'name' must be a string (it is {type(name)}).")

    execution_mode: JobQueueExecutionMode = field(  # type: ignore
        default=JobQueueExecutionMode.FIFO,  # type: ignore
        metadata={
            "docstring": "Execution mode of the jobs submitted into the queue "  # type: ignore
            f"(one of: {','.join([str(m.value) for m in JobQueueExecutionMode])}",  # type: ignore
        },
    )

    def _validate_execution_mode(
        self, execution_mode: JobQueueExecutionMode
    ) -> JobQueueExecutionMode:
        if isinstance(execution_mode, str):
            return JobQueueExecutionMode.validate(execution_mode)
        elif isinstance(execution_mode, JobQueueExecutionMode):
            return execution_mode
        else:
            raise TypeError(
                f"'execution_mode' must be a 'JobQueueExecutionMode' (it is {type(execution_mode)})."
            )

    compute_config: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The name of an existing compute config that will be used to create the job queue "
            "cluster. If not specified, the compute config of the associated job will be used.",
        },
    )

    def _validate_compute_config(self, compute_config: Optional[str]):
        if compute_config is not None and not isinstance(compute_config, str):
            raise TypeError(
                f"'compute_config_id' must be a string (it is {type(compute_config)})."
            )

    max_concurrency: int = field(
        default=1,
        metadata={
            "docstring": "Max number of jobs that can run concurrently."
            "Defaults to 1, meaning only one job can run at a given time.",
        },
    )

    def _validate_max_concurrency(self, max_concurrency: int):
        if not isinstance(max_concurrency, int):
            raise TypeError(
                f"'max_concurrency' must be an int (it is {type(max_concurrency)})."
            )

    auto_termination_threshold_job_count: Optional[int] = field(
        default=None,
        metadata={
            "docstring": "Maximum number of jobs the cluster can run before it "
            "becomes eligible for termination"
        },
    )

    def _validate_auto_termination_threshold_job_count(
        self, auto_termination_threshold_job_count: Optional[int]
    ):
        if auto_termination_threshold_job_count is None:
            return
        if not isinstance(auto_termination_threshold_job_count, int):
            raise TypeError(
                f"'auto_termination_threshold_job_count' must be an int (it is {type(auto_termination_threshold_job_count)})."
            )
        if auto_termination_threshold_job_count <= 0:
            raise ValueError("'auto_termination_threshold_job_count' should be > 0")


@dataclass(frozen=True)
class JobQueueConfig(ModelBase):
    """Configuration options for a job related to using a job queue for scheduling and execution."""

    __doc_py_example__ = """\
import anyscale
from anyscale.job.models import JobQueueConfig

# An example configuration that creates a job queue if one does not exist with the provided options.
job_queue_config = JobQueueConfig(
    # Priority of the job (only relevant if the execution_mode is "PRIORITY").
    priority=100,
    # Specification of the target Job Queue (will be created if does not exist)
    job_queue_spec=JobQueueSpec(
      name="my-job-queue",
      compute_config="my-compute-config:1",
      idle_timeout_s=3600,
    ),
)

# An example config that targets an existing job queue by name.
job_queue_config = JobQueueConfig(
    # Job's priority (only relevant for priority queues)
    priority=100,
    # Name for the job queue this job should be added to (specified in `JobQueueSpec.name`
    # on queue's creation)
    target_job_queue_name="my-new-queue"
)
"""

    __doc_yaml_example__ = """\
# An example configuration that creates a job queue if one does not exist with the provided options.
job_queue_config:
    # Priority of the job (only relevant if the execution_mode is "PRIORITY").
    priority: 100,
    # Specification of the target Job Queue (will be created if does not exist)
    job_queue_spec:
        name: my-job-queue
        compute_config: my-compute-config:1
        idle_timeout_s: 3600

# An example config that targets an existing job queue by name.
job_queue_config:
    # Priority of the job (only relevant if the execution_mode is "PRIORITY").
    priority: 100
    # Name for the job queue this job should be added to (specified in `JobQueueSpec.name`
    # on queue's creation)
    target_job_queue_name: my-new-queue
"""

    priority: Optional[int] = field(
        default=None,
        metadata={
            "docstring": "Job's relative priority (only relevant for Job Queues of type PRIORITY). "
            "Valid values range from 0 (highest) to +inf (lowest). Default value is None",
        },
    )

    def _validate_priority(self, priority: Optional[int]):
        if priority is None:
            return

        if not isinstance(priority, int):
            raise TypeError(f"'priority' must be an int (it is {type(priority)}).")

        elif priority < 0 or priority > INT_MAX:
            raise ValueError(f"'priority' should be >= 0 and <= {INT_MAX}")

    target_job_queue_name: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The name of an existing job queue to schedule this job in. If this is provided, job_queue_spec cannot be."
        },
    )

    def _validate_target_job_queue_name(self, target_job_queue_name: Optional[str]):
        if target_job_queue_name is not None and not isinstance(
            target_job_queue_name, str
        ):
            raise TypeError(
                f"'target_job_queue_name' must be a string (it is {type(target_job_queue_name)})."
            )

    job_queue_spec: Optional[JobQueueSpec] = field(
        default=None,
        metadata={
            "docstring": "Configuration options defining a job queue to be created for the job if needed. If this is provided, target_job_queue_name cannot be."
        },
    )

    def _validate_job_queue_spec(
        self, job_queue_spec: Optional[JobQueueSpec]
    ) -> Optional[JobQueueSpec]:
        # NOTE: Structural validation is performed on the backend
        if isinstance(job_queue_spec, Dict):
            return JobQueueSpec.from_dict(job_queue_spec)
        elif job_queue_spec is None or isinstance(job_queue_spec, JobQueueSpec):
            return job_queue_spec
        else:
            raise TypeError(
                f"'job_queue_spec' must be an instance of 'JobQueueSpec' or a dict (it is {type(job_queue_spec)})."
            )


@dataclass(frozen=True)
class JobConfig(WorkloadConfig):
    """Configuration options for a job."""

    __doc_py_example__ = """\
from anyscale.job.models import JobConfig

config = JobConfig(
    name="my-job",
    entrypoint="python main.py",
    max_retries=1,
    # An inline `ComputeConfig` or `MultiResourceComputeConfig` can also be provided.
    compute_config="my-compute-config:1",
    # A containerfile path can also be provided.
    image_uri="anyscale/image/my-image:1",
)
"""

    __doc_yaml_example__ = """\
name: my-job
entrypoint: python main.py
image_uri: anyscale/image/my-image:1 # (Optional) Exclusive with `containerfile`.
containerfile: /path/to/Dockerfile # (Optional) Exclusive with `image_uri`.
compute_config: my-compute-config:1 # (Optional) An inline dictionary can also be provided.
working_dir: /path/to/working_dir # (Optional) Defaults to `.`.
excludes: # (Optional) List of files to exclude from being packaged up for the job.
    - .git
    - .env
    - .DS_Store
    - __pycache__
requirements: # (Optional) List of requirements files to install. Can also be a path to a requirements.txt.
    - emoji==1.2.0
    - numpy==1.19.5
env_vars: # (Optional) Dictionary of environment variables to set in the job.
    MY_ENV_VAR: my_value
    ANOTHER_ENV_VAR: another_value
py_modules: # (Optional) A list of local directories or remote URIs that will be added to the Python path.
    - /path/to/my_module
    - s3://my_bucket/my_module
cloud: anyscale-prod # (Optional) The name of the Anyscale Cloud.
project: my-project # (Optional) The name of the Anyscale Project.
max_retries: 3 # (Optional) Maximum number of times the job will be retried before being marked failed. Defaults to `1`.
tags:
    team: mlops
    purpose: training

"""

    # Override the `name` field from `WorkloadConfig` so we can document it separately for jobs and services.
    name: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Name of the job. Multiple jobs can be submitted with the same name."
        },
    )

    entrypoint: str = field(
        default="",
        repr=False,
        metadata={
            "docstring": "Command that will be run to execute the job, e.g., `python main.py`."
        },
    )

    def _validate_entrypoint(self, entrypoint: str):
        if not isinstance(entrypoint, str):
            raise TypeError("'entrypoint' must be a string.")

        if not entrypoint:
            raise ValueError("'entrypoint' cannot be empty.")

    max_retries: int = field(
        default=1,
        repr=False,
        metadata={
            "docstring": "Maximum number of times the job will be retried before being marked failed. Defaults to `1`."
        },
    )

    def _validate_max_retries(self, max_retries: int):
        if not isinstance(max_retries, int):
            raise TypeError("'max_retries' must be an int.")

        if max_retries < 0:
            raise ValueError("'max_retries' must be >= 0.")

    job_queue_config: Optional[JobQueueConfig] = field(
        default=None,
        metadata={
            "docstring": "Job's configuration related to scheduling & execution using job queues"
        },
    )

    def _validate_job_queue_config(
        self, job_queue_config: Optional[JobQueueConfig]
    ) -> Optional[JobQueueConfig]:
        # NOTE: Structural validation is performed on the backend
        if isinstance(job_queue_config, Dict):
            return JobQueueConfig.from_dict(job_queue_config)
        elif job_queue_config is None or isinstance(job_queue_config, JobQueueConfig):
            return job_queue_config
        else:
            raise TypeError(
                f"'job_queue_config' must be an instance of 'JobQueueConfig' or a dict (it's {type(job_queue_config)})."
            )

    timeout_s: Optional[int] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "The timeout in seconds for each job run. Set to None for no limit to be set."
        },
    )

    def _validate_timeout_s(self, timeout_s: Optional[int]):
        if timeout_s is not None:
            if not isinstance(timeout_s, int):
                raise TypeError("'timeout_s' must be an int.")

            if timeout_s < 0:
                raise ValueError("'timeout_s' must be >= 0.")

    tags: Optional[Dict[str, str]] = field(
        default=None, metadata={"docstring": "Tags to associate with the job."},
    )

    def _validate_tags(self, tags: Optional[Dict[str, str]]):
        if tags is None:
            return
        if not isinstance(tags, dict):
            raise TypeError("'tags' must be a Dict[str, str].")
        for k, v in tags.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError("'tags' must be a Dict[str, str].")


class JobRunState(ModelEnum):
    """Current state of an individual job run."""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    SUCCEEDED = "SUCCEEDED"
    UNKNOWN = "UNKNOWN"

    __docstrings__: ClassVar[Dict[str, str]] = {
        STARTING: "The job run is being started and is not yet running.",
        RUNNING: "The job run is running.",
        FAILED: "The job run did not finish running or the entrypoint returned an exit code other than 0.",
        SUCCEEDED: "The job run finished running and its entrypoint returned exit code 0.",
        UNKNOWN: "The CLI/SDK received an unexpected state from the API server. In most cases, this means you need to update the CLI.",
    }


@dataclass(frozen=True)
class JobRunStatus(ModelBase):
    """Current status of an individual job run."""

    __doc_py_example__ = """\
import anyscale
from anyscale.job.models import JobRunStatus
run_statuses: List[JobRunStatus] = anyscale.job.status(name="my-job").runs
"""

    __doc_cli_example__ = command_examples.JOB_STATUS_EXAMPLE

    name: str = field(metadata={"docstring": "Name of the job run."})

    def _validate_name(self, name: str):  # noqa: A002
        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")

    state: Union[str, JobRunState] = field(
        metadata={"docstring": "Current state of the job run."}
    )

    def _validate_state(self, state: Union[str, JobRunState]) -> JobRunState:
        return JobRunState.validate(state)


class JobState(ModelEnum):
    """Current state of a job."""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    SUCCEEDED = "SUCCEEDED"
    UNKNOWN = "UNKNOWN"

    _TERMINAL_JOB_STATES: ClassVar[List[str]] = [
        SUCCEEDED,
        FAILED,
    ]

    @classmethod
    def is_terminal(cls, state: "JobState"):
        return state in cls._TERMINAL_JOB_STATES

    __docstrings__: ClassVar[Dict[str, str]] = {
        STARTING: "The job is being started and is not yet running.",
        RUNNING: "The job is running. A job will have state RUNNING if a job run fails and there are remaining retries.",
        FAILED: "The job did not finish running or the entrypoint returned an exit code other than 0 after retrying up to max_retries times.",
        SUCCEEDED: "The job finished running and its entrypoint returned exit code 0.",
        UNKNOWN: "The CLI/SDK received an unexpected state from the API server. In most cases, this means you need to update the CLI.",
    }


class JobSortField(ModelEnum):
    """Fields available for sorting jobs."""

    STATUS = "STATUS"
    CREATED_AT = "CREATED_AT"
    LAST_UPDATED_AT = "LAST_UPDATED_AT"
    ENDED_AT = "ENDED_AT"
    NAME = "NAME"
    ID = "ID"

    __docstrings__: ClassVar[Dict[str, str]] = {
        STATUS: "Sort by job status.",
        CREATED_AT: "Sort by creation timestamp.",
        LAST_UPDATED_AT: "Sort by last update timestamp.",
        ENDED_AT: "Sort by end timestamp.",
        NAME: "Sort by job name.",
        ID: "Sort by job ID.",
    }


class JobSortOrder(ModelEnum):
    """Enum for sort order directions."""

    ASC = "ASC"
    DESC = "DESC"

    __docstrings__: ClassVar[Dict[str, str]] = {
        ASC: "Sort in ascending order.",
        DESC: "Sort in descending order.",
    }


@dataclass(frozen=True)
class JobStatus(ModelBase):
    """Current status of a job."""

    __doc_py_example__ = """\
import anyscale
from anyscale.job.models import JobStatus
status: JobStatus = anyscale.job.status(name="my-job")
"""

    __doc_cli_example__ = """\
$ anyscale job status -n my-job
id: prodjob_3suiybn8r7dhz92yv63jqzm473
name: my-job
state: STARTING
"""

    id: str = field(
        metadata={
            "docstring": "Unique ID of the job (generated when the job is first submitted)."
        }
    )

    def _validate_id(self, id: str):  # noqa: A002
        if not isinstance(id, str):
            raise TypeError("'id' must be a string.")

    name: str = field(
        metadata={
            "docstring": "Name of the job. Multiple jobs can be submitted with the same name."
        },
    )

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")

    state: Union[str, JobState] = field(
        metadata={"docstring": "Current state of the job."}
    )

    def _validate_state(self, state: Union[str, JobState]) -> JobState:
        return JobState.validate(state)

    config: JobConfig = field(
        repr=False, metadata={"docstring": "Configuration of the job."}
    )

    def _validate_config(self, config: JobConfig):
        if not isinstance(config, JobConfig):
            raise TypeError("'config' must be a JobConfig.")

    runs: List[JobRunStatus] = field(metadata={"docstring": "List of job run states."})

    def _validate_runs(self, runs: List[JobRunStatus]):
        for run in runs:
            if not isinstance(run, JobRunStatus):
                raise TypeError("Each run in 'runs' must be a JobRunStatus.")

    creator_id: str = field(
        metadata={"docstring": "ID of the user who created the job.",},
    )

    def _validate_creator_id(self, creator_id: str):
        if creator_id is not None and not isinstance(creator_id, str):
            raise TypeError("'creator_id' must be a string.")

    created_at: Optional[datetime] = field(
        default=None, metadata={"docstring": "Timestamp when the job was created."},
    )

    def _validate_created_at(self, created_at: Optional[datetime]):
        if created_at is not None and not isinstance(created_at, datetime):
            raise TypeError("'created_at' must be a datetime or None.")


class JobLogMode(ModelEnum):
    """Mode to use for getting job logs."""

    HEAD = "HEAD"
    TAIL = "TAIL"

    __docstrings__: ClassVar[Dict[str, str]] = {
        HEAD: "Fetch logs from the start of the job's log.",
        TAIL: "Fetch logs from the end of the job's log.",
    }
