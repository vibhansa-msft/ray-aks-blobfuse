from typing import Optional

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_command
from anyscale.cli_logger import BlockLogger
from anyscale.schedule._private.schedule_sdk import PrivateScheduleSDK
from anyscale.schedule.models import ScheduleConfig, ScheduleState, ScheduleStatus


logger = BlockLogger()

_SCHEDULE_SDK_SINGLETON_KEY = "schedule_sdk"

_APPLY_EXAMPLE = """
import anyscale
from anyscale.job.models import JobConfig
from anyscale.schedule.models import ScheduleConfig

anyscale.schedule.apply(
    ScheduleConfig(
        cron_expression="0 0 * * * *",
        job_config=JobConfig(
            name="my-job",
            entrypoint="python main.py",
            working_dir=".",
        )
    )
)
"""

_APPLY_ARG_DOCSTRINGS = {"config": "The config options defining the schedule."}


@sdk_command(
    _SCHEDULE_SDK_SINGLETON_KEY,
    PrivateScheduleSDK,
    doc_py_example=_APPLY_EXAMPLE,
    arg_docstrings=_APPLY_ARG_DOCSTRINGS,
)
def apply(
    config: ScheduleConfig, *, _private_sdk: Optional[PrivateScheduleSDK] = None
) -> str:
    """Apply or update a schedule.

    Returns the id of the schedule.
    """
    return _private_sdk.apply(config)  # type: ignore


_SET_STATE_EXAMPLE = """
import anyscale
from anyscale.schedule.models import ScheduleState

anyscale.schedule.set_state(
    id="my=schedule-id",
    state=ScheduleState.DISABLED,
)
"""

_SET_STATE_ARG_DOCSTRINGS = {
    "id": "The id of the schedule.",
    "name": "The name of the schedule.",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the job. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
    "state": "The state to set the schedule to.",
}


@sdk_command(
    _SCHEDULE_SDK_SINGLETON_KEY,
    PrivateScheduleSDK,
    doc_py_example=_SET_STATE_EXAMPLE,
    arg_docstrings=_SET_STATE_ARG_DOCSTRINGS,
)
def set_state(
    *,
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    state: ScheduleState,
    _private_sdk: Optional[PrivateScheduleSDK] = None,
) -> str:
    """Set the state of a schedule.

    Returns the id of the schedule.
    """
    return _private_sdk.set_state(  # type: ignore
        id=id, name=name, cloud=cloud, project=project, state=state,
    )


_STATUS_EXAMPLE = """
import anyscale
anyscale.schedule.status(id="cronjob_yt389jvskwht9k2ygx7rj6iz62")
"""

_STATUS_ARG_DOCSTRINGS = {
    "id": "The id of the schedule.",
    "name": "The name of the schedule.",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the job. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _SCHEDULE_SDK_SINGLETON_KEY,
    PrivateScheduleSDK,
    doc_py_example=_STATUS_EXAMPLE,
    arg_docstrings=_STATUS_ARG_DOCSTRINGS,
)
def status(
    *,
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateScheduleSDK] = None,
) -> ScheduleStatus:
    """Return the status of the schedule.
    """
    return _private_sdk.status(id=id, name=name, cloud=cloud, project=project)  # type: ignore


_TRIGGER_EXAMPLE = """
import anyscale
anyscale.schedule.trigger(id="cronjob_yt389jvskwht9k2ygx7rj6iz62")
"""

_TRIGGER_ARG_DOCSTRINGS = {
    "id": "The id of the schedule.",
    "name": "The name of the schedule.",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the job. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _SCHEDULE_SDK_SINGLETON_KEY,
    PrivateScheduleSDK,
    doc_py_example=_TRIGGER_EXAMPLE,
    arg_docstrings=_TRIGGER_ARG_DOCSTRINGS,
)
def trigger(
    *,
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateScheduleSDK] = None,
) -> str:
    """Trigger the execution of the schedule.
    """
    return _private_sdk.trigger(id=id, name=name, cloud=cloud, project=project)  # type: ignore


_LIST_EXAMPLE = """
import anyscale
from anyscale.schedule.models import ScheduleStatus

# List all schedules
for schedule in anyscale.schedule.list(max_items=10):
    print(f"{schedule.name}: {schedule.state}")

# Filter by project
schedules = list(anyscale.schedule.list(project="my-project"))
"""

_LIST_ARG_DOCSTRINGS = {
    "name": "Filter by schedule name.",
    "schedule_id": "Fetch a specific schedule by ID.",
    "project": "Filter by project name.",
    "cloud": "Filter by cloud name.",
    "creator_id": "Filter by creator ID.",
    "include_all_users": "Include schedules from all users.",
    "page_size": "Number of items per page.",
    "max_items": "Maximum total items to return.",
}


@sdk_command(
    _SCHEDULE_SDK_SINGLETON_KEY,
    PrivateScheduleSDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_ARG_DOCSTRINGS,
)
def list(  # noqa: A001, PLR0913
    *,
    name: Optional[str] = None,
    schedule_id: Optional[str] = None,
    project: Optional[str] = None,
    cloud: Optional[str] = None,
    creator_id: Optional[str] = None,
    include_all_users: bool = False,
    page_size: Optional[int] = None,
    max_items: Optional[int] = None,
    _private_sdk: Optional[PrivateScheduleSDK] = None,
) -> ResultIterator[ScheduleStatus]:
    """List schedules with filtering and pagination.

    Returns a ResultIterator that lazily fetches pages of schedules.
    """
    return _private_sdk.list(  # type: ignore
        name=name,
        schedule_id=schedule_id,
        project=project,
        cloud=cloud,
        creator_id=creator_id,
        include_all_users=include_all_users,
        page_size=page_size,
        max_items=max_items,
    )


_URL_EXAMPLE = """
import anyscale

# Get URL by ID
url = anyscale.schedule.url(id="cronjob_xxx")
print(url)

# Get URL by name
url = anyscale.schedule.url(name="my-schedule", cloud="my-cloud", project="my-project")
print(url)
"""

_URL_ARG_DOCSTRINGS = {
    "id": "The schedule ID.",
    "name": "The schedule name (requires cloud and project).",
    "cloud": "Cloud name (required with name).",
    "project": "Project name (required with name).",
}


@sdk_command(
    _SCHEDULE_SDK_SINGLETON_KEY,
    PrivateScheduleSDK,
    doc_py_example=_URL_EXAMPLE,
    arg_docstrings=_URL_ARG_DOCSTRINGS,
)
def url(
    *,
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateScheduleSDK] = None,
) -> str:
    """Get the web UI URL for a schedule."""
    return _private_sdk.url(id=id, name=name, cloud=cloud, project=project)  # type: ignore


_DELETE_EXAMPLE = """
import anyscale

# Delete by ID
anyscale.schedule.delete(id="cronjob_xxx")

# Delete by name
anyscale.schedule.delete(name="my-schedule", cloud="my-cloud", project="my-project")
"""

_DELETE_ARG_DOCSTRINGS = {
    "id": "The schedule ID.",
    "name": "The schedule name (requires cloud and project).",
    "cloud": "Cloud name (required with name).",
    "project": "Project name (required with name).",
}


@sdk_command(
    _SCHEDULE_SDK_SINGLETON_KEY,
    PrivateScheduleSDK,
    doc_py_example=_DELETE_EXAMPLE,
    arg_docstrings=_DELETE_ARG_DOCSTRINGS,
)
def delete(
    *,
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateScheduleSDK] = None,
) -> str:
    """Delete a schedule.

    If the schedule is active, it will be automatically paused before deletion.
    The schedule must have no active triggered jobs.
    Returns the ID of the deleted schedule.
    """
    return _private_sdk.delete(id=id, name=name, cloud=cloud, project=project)  # type: ignore
