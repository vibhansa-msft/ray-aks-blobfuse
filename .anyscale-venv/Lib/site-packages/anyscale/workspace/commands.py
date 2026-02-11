from typing import Any, Dict, List, Optional, Tuple, Union

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_command
from anyscale.workspace._private.workspace_sdk import PrivateWorkspaceSDK
from anyscale.workspace.models import (
    UpdateWorkspaceConfig,
    Workspace,
    WorkspaceConfig,
    WorkspaceSortField,
    WorkspaceSortOrder,
    WorkspaceState,
)


_WORKSPACE_SDK_SINGLETON_KEY = "workspace_sdk"

_CREATE_EXAMPLE = """
import anyscale
from anyscale.workspace.models import WorkspaceConfig

anyscale.workspace.create(
    WorkspaceConfig(
        name="my-workspace",
        idle_termination_minutes=120,
    ),
)
"""

_CREATE_ARG_DOCSTRINGS = {"config": "The config for defining the workspace."}
_WAIT_TIMEOUT_SECONDS = 1800.0


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_CREATE_EXAMPLE,
    arg_docstrings=_CREATE_ARG_DOCSTRINGS,
)
def create(
    config: WorkspaceConfig, *, _private_sdk: Optional[PrivateWorkspaceSDK] = None
) -> str:
    """Create a workspace.

    Returns the id of the created workspace.
    """
    return _private_sdk.create(config)  # type: ignore


_START_EXAMPLE = """
import anyscale

anyscale.workspace.start(
    name="my-workspace",
)
"""

_START_ARG_DOCSTRINGS = {
    "name": "Name of the workspace.",
    "id": "Unique ID of the workspace",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the workspace. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_START_EXAMPLE,
    arg_docstrings=_START_ARG_DOCSTRINGS,
)
def start(
    *,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> str:
    """Start a workspace.

    Returns the id of the started workspace.
    """
    return _private_sdk.start(name=name, id=id, cloud=cloud, project=project)  # type: ignore


_TERMINATE_EXAMPLE = """
import anyscale

anyscale.workspace.terminate(
    name="my-workspace",
)
"""

_TERMINATE_ARG_DOCSTRINGS = {
    "name": "Name of the workspace.",
    "id": "Unique ID of the workspace",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the workspace. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_TERMINATE_EXAMPLE,
    arg_docstrings=_TERMINATE_ARG_DOCSTRINGS,
)
def terminate(
    *,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> str:
    """Terminate a workspace.

    Returns the id of the terminated workspace.
    """
    return _private_sdk.terminate(name=name, id=id, cloud=cloud, project=project)  # type: ignore


_STATUS_EXAMPLE = """
import anyscale

status = anyscale.workspace.status(
    name="my-workspace",
)
"""

_STATUS_ARG_DOCSTRINGS = {
    "name": "Name of the workspace.",
    "id": "Unique ID of the workspace",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the workspace. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_STATUS_EXAMPLE,
    arg_docstrings=_STATUS_ARG_DOCSTRINGS,
)
def status(
    *,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> str:
    """Get the status of a workspace.

    Returns the status of the workspace.
    """
    return _private_sdk.status(name=name, id=id, cloud=cloud, project=project)  # type: ignore


_WAIT_EXAMPLE = """
import anyscale

anyscale.workspace.wait(
    name="my-workspace",
)
"""

_WAIT_ARG_DOCSTRINGS = {
    "name": "Name of the workspace.",
    "id": "Unique ID of the workspace",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the workspace. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
    "timeout_s": "The maximum time to wait for the workspace to reach a terminal state.",
    "state": "The desired terminal state to wait for, defaults to RUNNING.",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_WAIT_EXAMPLE,
    arg_docstrings=_WAIT_ARG_DOCSTRINGS,
)
def wait(
    *,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    timeout_s: float = _WAIT_TIMEOUT_SECONDS,
    state: str = WorkspaceState.RUNNING,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> str:
    """Wait for a workspace to reach a terminal state.

    Returns the status of the workspace.
    """
    return _private_sdk.wait(  # type: ignore
        name=name,
        id=id,
        cloud=cloud,
        project=project,
        timeout_s=timeout_s,
        state=state,
    )


_GENERATE_SSH_CONFIG_FILE_EXAMPLE = """
import anyscale
import subprocess

host_name, config_file = anyscale.workspace.generate_ssh_config_file(
    name="my-workspace",
)

# run an ssh command using the generated config file
subprocess.run(["ssh", "-F", config_path, host_name, "ray --version"])
"""

_GENERATE_SSH_CONFIG_FILE_ARG_DOCSTRINGS = {
    "name": "Name of the workspace.",
    "id": "Unique ID of the workspace",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the workspace. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
    "ssh_config_path": "The directory to write the generated config file to.",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_GENERATE_SSH_CONFIG_FILE_EXAMPLE,
    arg_docstrings=_GENERATE_SSH_CONFIG_FILE_ARG_DOCSTRINGS,
)
def generate_ssh_config_file(
    *,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    ssh_config_path: Optional[str] = None,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> Tuple[str, str]:
    """Generate an SSH config file for a workspace.

    Returns the hostname and path to the generated config file.
    """
    return _private_sdk.generate_ssh_config_file(  # type: ignore
        name=name, id=id, cloud=cloud, project=project, ssh_config_path=ssh_config_path,
    )


_RUN_COMMAND_EXAMPLE = """
import anyscale

process = anyscale.workspace.run_command(
    name="my-workspace",
    command="ray_version",
    capture_output=True,
    text=True,
)
print(process.stdout)
"""

_RUN_COMMAND_ARG_DOCSTRINGS = {
    "name": "Name of the workspace.",
    "id": "Unique ID of the workspace",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the workspace. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
    "command": "The command to run.",
    "kwargs": "Additional arguments to pass to subprocess.run.",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_RUN_COMMAND_EXAMPLE,
    arg_docstrings=_RUN_COMMAND_ARG_DOCSTRINGS,
)
def run_command(
    *,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    command: str,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
    **kwargs: Dict[str, Any],
):
    """Run a command in a workspace.

    Returns a subprocess.CompletedProcess object.
    """
    return _private_sdk.run_command(  # type: ignore
        name=name, id=id, cloud=cloud, project=project, command=command, **kwargs,
    )


_PULL_EXAMPLE = """
import anyscale

anyscale.workspace.pull(
    name="my-workspace",
)
"""

_PULL_ARG_DOCSTRINGS = {
    "name": "Name of the workspace.",
    "id": "Unique ID of the workspace",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the workspace. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
    "local_dir": "The local directory to pull the workspace to. If not provided, the current working directory will be used.",
    "pull_git_state": "Whether to pull the git state of the workspace.",
    "rsync_args": "Additional arguments to pass to rsync.",
    "delete": "Whether to delete files in the local directory that are not in the workspace. Excluded files (e.g., .git or custom exclusions) are preserved and not deleted.",
    "direct_ssh": "Whether to use direct SSH connection (port 22) instead of SSH-over-HTTPS tunnel.",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_PULL_EXAMPLE,
    arg_docstrings=_PULL_ARG_DOCSTRINGS,
)
def pull(  # noqa: PLR0913
    *,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    local_dir: Optional[str] = None,
    pull_git_state: bool = False,
    rsync_args: Optional[List[str]] = None,
    delete: bool = False,
    direct_ssh: bool = False,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> None:
    """Pull a workspace to a local directory.

    New files will be created, existing files will be overwritten. With --delete, files
    in the local directory that don't exist in the workspace will be removed. Excluded
    files (like .git) are preserved and not deleted even with --delete.

    Returns the path to the pulled workspace.
    """
    _private_sdk.pull(  # type: ignore
        name=name,
        id=id,
        cloud=cloud,
        project=project,
        local_dir=local_dir,
        pull_git_state=pull_git_state,
        rsync_args=rsync_args,
        delete=delete,
        direct_ssh=direct_ssh,
    )


_PUSH_EXAMPLE = """
import anyscale

anyscale.workspace.push(
    name="my-workspace",
    local_dir="~/workspace",
)
"""

_PUSH_ARG_DOCSTRINGS = {
    "name": "Name of the workspace.",
    "id": "Unique ID of the workspace",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the workspace. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
    "local_dir": "The local directory to push to the workspace. If not provided, the current working directory will be used.",
    "push_git_state": "Whether to push the git state of the workspace.",
    "rsync_args": "Additional arguments to pass to rsync.",
    "delete": "Whether to delete files in the workspace that are not in the local directory. Excluded files (e.g., .git or custom exclusions) are preserved and not deleted.",
    "direct_ssh": "Whether to use direct SSH connection (port 22) instead of SSH-over-HTTPS tunnel.",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_PUSH_EXAMPLE,
    arg_docstrings=_PUSH_ARG_DOCSTRINGS,
)
def push(  # noqa: PLR0913
    *,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    local_dir: Optional[str] = None,
    push_git_state: bool = False,
    rsync_args: Optional[List[str]] = None,
    delete: bool = False,
    direct_ssh: bool = False,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> None:
    """Push a local directory to a workspace.

    New files will be created, existing files will be overwritten. With --delete, files
    in the workspace that don't exist locally will be removed. Excluded files (like .git)
    are preserved and not deleted even with --delete.

    Returns the path to the pushed workspace.
    """
    _private_sdk.push(  # type: ignore
        name=name,
        id=id,
        cloud=cloud,
        project=project,
        local_dir=local_dir,
        push_git_state=push_git_state,
        rsync_args=rsync_args,
        delete=delete,
        direct_ssh=direct_ssh,
    )


_UPDATE_EXAMPLE = """
import anyscale

anyscale.workspace.update(
    id="<workspace-id>",
    config=UpdateWorkspaceConfig(
        name="new-workspace-name",
        idle_termination_minutes=120,
    ),
)
"""

_UPDATE_ARG_DOCSTRINGS = {
    "id": "Unique ID of the workspace",
    "config": "The config for updating the workspace. Unspecified fields will retain their current values, while specified fields will be updated.",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_UPDATE_EXAMPLE,
    arg_docstrings=_UPDATE_ARG_DOCSTRINGS,
)
def update(
    *,
    id: Optional[str] = None,  # noqa: A002
    config: UpdateWorkspaceConfig,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> None:
    """Update a workspace."""
    _private_sdk.update(  # type: ignore
        id=id, config=config,
    )


_GET_EXAMPLE = """
import anyscale
from anyscale.workspace.models import Workspace

workspace: Workspace = anyscale.workspace.get(
    name='my-workspace',
)
"""

_GET_ARG_DOCSTRINGS = {
    "name": "Name of the workspace.",
    "id": "Unique ID of the workspace",
    "cloud": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
    "project": "Named project to use for the workspace. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_GET_EXAMPLE,
    arg_docstrings=_GET_ARG_DOCSTRINGS,
)
def get(
    *,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    include_config: bool = True,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> Workspace:
    """Get a workspace.

    Args:
        include_config: If True (default), fetch full workspace config. Set to False for efficiency.
    """
    return _private_sdk.get(name=name, id=id, cloud=cloud, project=project, include_config=include_config)  # type: ignore


_LIST_EXAMPLE = """
import anyscale
from anyscale.workspace.models import Workspace

for workspace in anyscale.workspace.list(max_items=10):
    print(f"{workspace.name}: {workspace.state}")
"""

_LIST_ARG_DOCSTRINGS = {
    "name": "Filter by workspace name (substring match).",
    "workspace_id": "Fetch a single workspace by ID.",
    "project": "Named project to filter by.",
    "cloud": "Named cloud to filter by.",
    "creator_id": "Filter workspaces by creator user ID.",
    "state_filter": "List of states to include. May be WorkspaceState enums or case-insensitive strings.",
    "tags_filter": "Filter by tags. Dict mapping tag keys to lists of values. Tags with the same key are ORed, different keys are ANDed.",
    "include_config": "If True, fetch full config for each workspace (expensive). Defaults to False for efficiency.",
    "sort_field": "Field to sort by (STATUS, CREATED_AT, LATEST_STARTED_AT). Defaults to status then created_at.",
    "sort_order": "Sort order (ASC or DESC). Defaults to appropriate order for the sort field.",
    "max_items": "Maximum number of items to return. If None, all items are returned.",
    "page_size": "Number of items to fetch per API call (affects performance, not total results).",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_ARG_DOCSTRINGS,
)
def list(  # noqa: A001, PLR0913, PLR0917
    *,
    workspace_id: Optional[str] = None,
    name: Optional[str] = None,
    project: Optional[str] = None,
    cloud: Optional[str] = None,
    creator_id: Optional[str] = None,
    state_filter: Optional[Union[List[WorkspaceState], List[str]]] = None,
    tags_filter: Optional[Dict[str, List[str]]] = None,
    include_config: bool = False,
    sort_field: Optional[Union[str, WorkspaceSortField]] = None,
    sort_order: Optional[Union[str, WorkspaceSortOrder]] = None,
    max_items: Optional[int] = None,
    page_size: Optional[int] = None,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> ResultIterator[Workspace]:
    """List workspaces with optional filters.

    Returns an iterator of Workspace objects. By default, filters to non-terminated states
    and does not fetch the workspace config data.
    """
    return _private_sdk.list(  # type: ignore
        workspace_id=workspace_id,
        name=name,
        project=project,
        cloud=cloud,
        creator_id=creator_id,
        state_filter=state_filter,
        tags_filter=tags_filter,
        include_config=include_config,
        sort_field=sort_field,
        sort_order=sort_order,
        max_items=max_items,
        page_size=page_size,
    )


# Workspace tag operations (SDK)
_TAGS_ADD_EXAMPLE = """
import anyscale

anyscale.workspace.add_tags(id="ws_123", tags={"team": "mlops", "env": "prod"})
"""

_TAGS_ADD_ARG_DOCSTRINGS = {
    "id": "ID of the workspace. Provide either id or name.",
    "name": "Name of the workspace. Provide either id or name.",
    "cloud": "Cloud name (used when resolving by name).",
    "project": "Project name (used when resolving by name).",
    "tags": "Key/value tags to upsert as a map {key: value}.",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
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
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
):
    """Upsert (add/update) tag key/value pairs for a workspace."""
    return _private_sdk.add_tags(  # type: ignore
        id=id, name=name, cloud=cloud, project=project, tags=tags
    )


_TAGS_REMOVE_EXAMPLE = """
import anyscale

anyscale.workspace.remove_tags(id="ws_123", keys=["team", "env"])
"""

_TAGS_REMOVE_ARG_DOCSTRINGS = {
    "id": "ID of the workspace. Provide either id or name.",
    "name": "Name of the workspace. Provide either id or name.",
    "cloud": "Cloud name (used when resolving by name).",
    "project": "Project name (used when resolving by name).",
    "keys": "List of tag keys to remove.",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
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
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
):
    """Remove tags by key from a workspace."""
    return _private_sdk.remove_tags(  # type: ignore
        id=id, name=name, cloud=cloud, project=project, keys=keys
    )


_TAGS_LIST_EXAMPLE = """
import anyscale

tags: dict[str, str] = anyscale.workspace.list_tags(name="my-workspace")
"""

_TAGS_LIST_ARG_DOCSTRINGS = {
    "id": "ID of the workspace. Provide either id or name.",
    "name": "Name of the workspace. Provide either id or name.",
    "cloud": "Cloud name (used when resolving by name).",
    "project": "Project name (used when resolving by name).",
}


@sdk_command(
    _WORKSPACE_SDK_SINGLETON_KEY,
    PrivateWorkspaceSDK,
    doc_py_example=_TAGS_LIST_EXAMPLE,
    arg_docstrings=_TAGS_LIST_ARG_DOCSTRINGS,
)
def list_tags(
    *,
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    project: Optional[str] = None,
    _private_sdk: Optional[PrivateWorkspaceSDK] = None,
) -> Dict[str, str]:
    """List tags for a workspace as a key/value mapping."""
    return _private_sdk.list_tags(  # type: ignore
        id=id, name=name, cloud=cloud, project=project
    )
