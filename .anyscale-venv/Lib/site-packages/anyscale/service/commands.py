from typing import Dict, List, Optional, Union

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_command
from anyscale.service._private.service_sdk import PrivateServiceSDK
from anyscale.service.models import (
    ServiceConfig,
    ServiceLogMode,
    ServiceSortField,
    ServiceSortOrder,
    ServiceState,
    ServiceStatus,
)


_SERVICE_SDK_SINGLETON_KEY = "service_sdk"

_DEPLOY_EXAMPLE = """
import anyscale
from anyscale.service.models import ServiceConfig

anyscale.service.deploy(
    ServiceConfig(
        name="my-service",
        applications=[
            {"import_path": "main:app"},
        ],
        working_dir=".",
    ),
    canary_percent=50,
)
"""

_DEPLOY_ARG_DOCSTRINGS = {
    "configs": "The config options defining the service.",
    "in_place": "Perform an in-place upgrade without starting a new cluster. This can be used for faster iteration during development but is *not* currently recommended for production deploys. This *cannot* be used to change cluster-level options such as image and compute config (they will be ignored).",
    "canary_percent": "The percentage of traffic to send to the canary version of the service (0-100). This can be used to manually shift traffic toward (or away from) the canary version. If not provided, traffic will be shifted incrementally toward the canary version until it reaches 100. Not supported when using --in-place. This is ignored when restarting a service or creating a new service.",
    "max_surge_percent": "Amount of excess capacity allowed to be used while updating the service (0-100). Defaults to 100. Not supported when using --in-place.",
    "versions": "Enable multi-version deployment by providing a JSON array of objects or a JSON object in text format. Defines the version name, traffic and capacity percents per version. Capacity defaults to traffic.",
    "name": "Name of the service. When running in a workspace, this defaults to the workspace name.",
    "cloud": "The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_DEPLOY_EXAMPLE,
    arg_docstrings=_DEPLOY_ARG_DOCSTRINGS,
)
def deploy(
    configs: Union[ServiceConfig, List[ServiceConfig]],
    *,
    in_place: bool = False,
    canary_percent: Optional[int] = None,
    max_surge_percent: Optional[int] = None,
    versions: Optional[str] = None,
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateServiceSDK] = None,
) -> str:
    """Deploy a service.

    If no service with the provided name is running, one will be created, else the existing service will be updated.

    This command is asynchronous, so it always returns immediately.

    Returns the id of the deployed service.
    """
    return _private_sdk.deploy(  # type: ignore
        configs=configs,
        in_place=in_place,
        canary_percent=canary_percent,
        max_surge_percent=max_surge_percent,
        versions=versions,
        name=name,
        cloud=cloud,
        project=project,
    )


_ROLLBACK_EXAMPLE = """
import anyscale

anyscale.service.rollback(name="my-service")
"""

_ROLLBACK_ARG_DOCSTRINGS = {
    "name": "Name of the service. When running in a workspace, this defaults to the workspace name.",
    "cloud": "The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
    "max_surge_percent": "Amount of excess capacity allowed to be used while rolling back to the primary version of the service (0-100). Defaults to 100.",
}


@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_ROLLBACK_EXAMPLE,
    arg_docstrings=_ROLLBACK_ARG_DOCSTRINGS,
)
def rollback(
    name: Optional[str],
    *,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    max_surge_percent: Optional[int] = None,
    _private_sdk: Optional[PrivateServiceSDK] = None,
) -> str:
    """Rollback to the primary version of the service.

    This command can only be used when there is an active rollout in progress. The
    rollout will be cancelled and the service will revert to the primary version.

    This command is asynchronous, so it always returns immediately.

    Returns the id of the rolled back service.
    """
    return _private_sdk.rollback(  # type: ignore
        name=name, cloud=cloud, project=project, max_surge_percent=max_surge_percent
    )


_TERMINATE_EXAMPLE = """
import anyscale

anyscale.service.terminate(name="my-service")
"""

_TERMINATE_ARG_DOCSTRINGS = {
    "id": "ID of the service.",
    "name": "Name of the service. When running in a workspace, this defaults to the workspace name.",
    "cloud": "The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_TERMINATE_EXAMPLE,
    arg_docstrings=_TERMINATE_ARG_DOCSTRINGS,
)
def terminate(
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    *,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateServiceSDK] = None,
) -> str:
    """Terminate a service.

    This command is asynchronous, so it always returns immediately.

    Returns the id of the terminated service.
    """
    return _private_sdk.terminate(id=id, name=name, cloud=cloud, project=project)  # type: ignore


_ARCHIVE_EXAMPLE = """
import anyscale

anyscale.service.archive(name="my-service")
"""

_ARCHIVE_ARG_DOCSTRINGS = {
    "id": "ID of the service.",
    "name": "Name of the service. When running in a workspace, this defaults to the workspace name.",
    "cloud": "The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_ARCHIVE_EXAMPLE,
    arg_docstrings=_ARCHIVE_ARG_DOCSTRINGS,
)
def archive(
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    *,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateServiceSDK] = None,
) -> str:
    """Archive a service.

    This command is asynchronous, so it always returns immediately.

    Returns the ID of the archived service.
    """
    return _private_sdk.archive(id=id, name=name, cloud=cloud, project=project)  # type: ignore


_DELETE_EXAMPLE = """
import anyscale

anyscale.service.delete(name="my-service")
"""

_DELETE_ARG_DOCSTRINGS = {
    "id": "ID of the service.",
    "name": "Name of the service. When running in a workspace, this defaults to the workspace name.",
    "cloud": "The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_DELETE_EXAMPLE,
    arg_docstrings=_DELETE_ARG_DOCSTRINGS,
)
def delete(
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    *,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateServiceSDK] = None,
) -> str:
    """Delete a service.

    This command is asynchronous, so it always returns immediately.

    Returns the ID of the deleted service.
    """
    return _private_sdk.delete(id=id, name=name, cloud=cloud, project=project)  # type: ignore


_STATUS_EXAMPLE = """
import anyscale
from anyscale.service.models import ServiceStatus

status: ServiceStatus = anyscale.service.status(name="my-service")
"""

_STATUS_ARG_DOCSTRINGS = {
    "name": "Name of the service.",
    "cloud": "The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_STATUS_EXAMPLE,
    arg_docstrings=_STATUS_ARG_DOCSTRINGS,
)
def status(
    name: str,
    *,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateServiceSDK] = None,
) -> ServiceStatus:
    """Get the status of a service."""
    return _private_sdk.status(name=name, cloud=cloud, project=project)  # type: ignore


_WAIT_EXAMPLE = """
import anyscale
from anyscale.service.models import ServiceState

anyscale.service.wait(name="my-service", state=ServiceState.RUNNING)
"""

_WAIT_ARG_DOCSTRINGS = {
    "name": "Name of the service.",
    "cloud": "The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
    "state": "The state to wait for the service to reach.",
    "timeout_s": "Timeout to wait for the service to reach the target state.",
}


@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_WAIT_EXAMPLE,
    arg_docstrings=_WAIT_ARG_DOCSTRINGS,
)
def wait(
    name: str,
    *,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    state: Union[str, ServiceState] = ServiceState.RUNNING,
    timeout_s: float = 600,
    _private_sdk: Optional[PrivateServiceSDK] = None,
    _interval_s: float = 5,
):
    """Wait for a service to reach a target state."""
    _private_sdk.wait(  # type: ignore
        name=name,
        cloud=cloud,
        project=project,
        state=ServiceState(state),
        timeout_s=timeout_s,
        interval_s=_interval_s,
    )  # type: ignore


_CONTROLLER_LOGS_EXAMPLE = """
import anyscale

anyscale.service.controller_logs("my-service", canary=True)
"""

_CONTROLLER_LOGS_ARG_DOCSTRINGS = {
    "name": "Name of the service.",
    "cloud": "The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
    "canary": "Whether to show the logs of the canary version of the service. If not provided, the primary version logs will be shown.",
    "mode": "The mode of log fetching to be used. Supported modes can be found in ServiceLogMode. If not provided, ServiceLogMode.TAIL will be used.",
    "max_lines": "The number of log lines to be fetched. If not provided, 1000 lines will be fetched.",
}

# SDK docs for Service tag operations
_TAGS_ADD_EXAMPLE = """
import anyscale

anyscale.service.add_tags(id="svc_123", tags={"team": "mlops", "env": "prod"})
"""

_TAGS_ADD_ARG_DOCSTRINGS = {
    "id": "ID of the service. Provide either id or name.",
    "name": "Name of the service. Provide either id or name.",
    "cloud": "Cloud name (used when resolving by name).",
    "project": "Project name (used when resolving by name).",
    "tags": "Key/value tags to upsert as a map {key: value}.",
}

_TAGS_REMOVE_EXAMPLE = """
import anyscale

anyscale.service.remove_tags(id="svc_123", keys=["team", "env"])
"""

_TAGS_REMOVE_ARG_DOCSTRINGS = {
    "id": "ID of the service. Provide either id or name.",
    "name": "Name of the service. Provide either id or name.",
    "cloud": "Cloud name (used when resolving by name).",
    "project": "Project name (used when resolving by name).",
    "keys": "List of tag keys to remove.",
}


@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_TAGS_ADD_EXAMPLE,
    arg_docstrings=_TAGS_ADD_ARG_DOCSTRINGS,
)
def add_tags(
    *,
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    tags: Dict[str, str],
    _private_sdk: Optional[PrivateServiceSDK] = None,
):
    """Upsert (add/update) tag key/value pairs for a service."""
    return _private_sdk.add_tags(  # type: ignore
        id=id, name=name, cloud=cloud, project=project, tags=tags
    )


@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_TAGS_REMOVE_EXAMPLE,
    arg_docstrings=_TAGS_REMOVE_ARG_DOCSTRINGS,
)
def remove_tags(
    *,
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    keys: List[str],
    _private_sdk: Optional[PrivateServiceSDK] = None,
):
    """Remove tags by key from a service."""
    return _private_sdk.remove_tags(  # type: ignore
        id=id, name=name, cloud=cloud, project=project, keys=keys
    )


_TAGS_LIST_EXAMPLE = """
import anyscale

tags: dict[str, str] = anyscale.service.list_tags(name="my-service")
"""

_TAGS_LIST_ARG_DOCSTRINGS = {
    "id": "ID of the service. Provide either id or name.",
    "name": "Name of the service. Provide either id or name.",
    "cloud": "Cloud name (used when resolving by name).",
    "project": "Project name (used when resolving by name).",
}


@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_TAGS_LIST_EXAMPLE,
    arg_docstrings=_TAGS_LIST_ARG_DOCSTRINGS,
)
def list_tags(
    *,
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateServiceSDK] = None,
) -> Dict[str, str]:
    """List tags for a service as a key/value mapping."""
    return _private_sdk.list_tags(  # type: ignore
        id=id, name=name, cloud=cloud, project=project
    )


# This is a private command that is not exposed to the user.
@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_CONTROLLER_LOGS_EXAMPLE,
    arg_docstrings=_CONTROLLER_LOGS_ARG_DOCSTRINGS,
)
def _controller_logs(
    name: str,
    *,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    canary: bool = False,
    mode: Union[str, ServiceLogMode] = ServiceLogMode.TAIL,
    max_lines: int = 1000,
    _private_sdk: Optional[PrivateServiceSDK] = None,
):
    """Wait for a service to reach a target state."""
    return _private_sdk.controller_logs(  # type: ignore
        name,
        cloud=cloud,
        project=project,
        canary=canary,
        mode=mode,
        max_lines=max_lines,
    )


_LIST_EXAMPLE = """
import anyscale
from anyscale.service.models import ServiceState

# Example: Get the first 50 running services
for svc in anyscale.service.list(max_items=50, state_filter=[ServiceState.RUNNING]):
    print(svc.name)
"""

_LIST_ARG_DOCSTRINGS = {
    "service_id": (
        "If provided, returns just the service with this ID "
        "wrapped in a one-page iterator."
    ),
    "name": "Substring to match against the service name.",
    "state_filter": (
        "List of states to include. "
        "May be `ServiceState` enums or case-insensitive strings."
    ),
    "creator_id": "Filter services by user ID.",
    "cloud": "Name of the Anyscale Cloud to search in.",
    "project": "Name of the Anyscale Project to search in.",
    "include_archived": "Include archived services (default: False).",
    "tags_filter": r"Filter services by tags as a dict: \{key: [values...]\}. Values with the same key are ORed; keys are ANDed.",
    # Paging
    "max_items": "Maximum **total** number of items to yield (default: iterate all).",
    "page_size": "Number of items to fetch per API request (default: API default).",
    # Sorting
    "sort_field": "Field to sort by (`NAME`, `STATUS`, `CREATED_AT`).",
    "sort_order": "Sort direction (`ASC` or `DESC`).",
}


# Public command
@sdk_command(
    _SERVICE_SDK_SINGLETON_KEY,
    PrivateServiceSDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_ARG_DOCSTRINGS,
)
def list(  # noqa: A001, PLR0913
    *,
    # Single-item lookup
    service_id: Optional[str] = None,
    # Filters
    name: Optional[str] = None,
    state_filter: Optional[Union[List[ServiceState], List[str]]] = None,
    creator_id: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    include_archived: bool = False,
    tags_filter: Optional[Dict[str, List[str]]] = None,
    # Paging
    max_items: Optional[int] = None,
    page_size: Optional[int] = None,
    # Sorting
    sort_field: Optional[Union[str, ServiceSortField]] = None,
    sort_order: Optional[Union[str, ServiceSortOrder]] = None,
    # Injected SDK
    _private_sdk: Optional[PrivateServiceSDK] = None,
) -> ResultIterator[ServiceStatus]:
    """List services or fetch a single service by ID."""
    return _private_sdk.list(  # type: ignore
        service_id=service_id,
        name=name,
        state_filter=state_filter,
        creator_id=creator_id,
        cloud=cloud,
        project=project,
        include_archived=include_archived,
        tags_filter=tags_filter,
        max_items=max_items,
        page_size=page_size,
        sort_field=sort_field,
        sort_order=sort_order,
    )
