from typing import Dict, List, Optional, Tuple, Union

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.workspace._private.workspace_sdk import PrivateWorkspaceSDK
from anyscale.workspace.commands import (
    _CREATE_ARG_DOCSTRINGS,
    _CREATE_EXAMPLE,
    _GENERATE_SSH_CONFIG_FILE_ARG_DOCSTRINGS,
    _GENERATE_SSH_CONFIG_FILE_EXAMPLE,
    _GET_ARG_DOCSTRINGS,
    _GET_EXAMPLE,
    _LIST_ARG_DOCSTRINGS,
    _LIST_EXAMPLE,
    _PULL_ARG_DOCSTRINGS,
    _PULL_EXAMPLE,
    _PUSH_ARG_DOCSTRINGS,
    _PUSH_EXAMPLE,
    _RUN_COMMAND_ARG_DOCSTRINGS,
    _RUN_COMMAND_EXAMPLE,
    _START_ARG_DOCSTRINGS,
    _START_EXAMPLE,
    _STATUS_ARG_DOCSTRINGS,
    _STATUS_EXAMPLE,
    _TAGS_ADD_ARG_DOCSTRINGS,
    _TAGS_ADD_EXAMPLE,
    _TAGS_LIST_ARG_DOCSTRINGS,
    _TAGS_LIST_EXAMPLE,
    _TAGS_REMOVE_ARG_DOCSTRINGS,
    _TAGS_REMOVE_EXAMPLE,
    _TERMINATE_ARG_DOCSTRINGS,
    _TERMINATE_EXAMPLE,
    _UPDATE_ARG_DOCSTRINGS,
    _UPDATE_EXAMPLE,
    _WAIT_ARG_DOCSTRINGS,
    _WAIT_EXAMPLE,
    add_tags as add_tags,
    create as create,
    generate_ssh_config_file as generate_ssh_config_file,
    get as get,
    list as list,  # noqa: A004 - claude_comment("claude-opus-4-5", "SDK public API re-export")
    list_tags as list_tags,
    pull as pull,
    push as push,
    remove_tags as remove_tags,
    run_command as run_command,
    start as start,
    status as status,
    terminate as terminate,
    update as update,
    wait as wait,
)
from anyscale.workspace.models import (
    UpdateWorkspaceConfig,
    Workspace,
    WorkspaceConfig,
    WorkspaceSortField,
    WorkspaceSortOrder,
    WorkspaceState,
)


class WorkspaceSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateWorkspaceSDK(
            client=client, logger=logger, timer=timer
        )

    @sdk_docs(
        doc_py_example=_CREATE_EXAMPLE, arg_docstrings=_CREATE_ARG_DOCSTRINGS,
    )
    def create(self, config: Optional[WorkspaceConfig]) -> str:  # noqa: F811
        """Create a workspace.

        Returns the id of the workspace.
        """
        return self._private_sdk.create(config=config)  # type: ignore

    @sdk_docs(
        doc_py_example=_START_EXAMPLE, arg_docstrings=_START_ARG_DOCSTRINGS,
    )
    def start(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        """Start a workspace.

        Returns the id of the started workspace.
        """
        return self._private_sdk.start(name=name, id=id, cloud=cloud, project=project)

    @sdk_docs(
        doc_py_example=_TERMINATE_EXAMPLE, arg_docstrings=_TERMINATE_ARG_DOCSTRINGS,
    )
    def terminate(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        """Terminate a workspace.

        Returns the id of the terminated workspace.
        """
        return self._private_sdk.terminate(
            name=name, id=id, cloud=cloud, project=project
        )

    @sdk_docs(
        doc_py_example=_STATUS_EXAMPLE, arg_docstrings=_STATUS_ARG_DOCSTRINGS,
    )
    def status(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        """Get the status of a workspace.

        Returns the status of the workspace.
        """
        return self._private_sdk.status(name=name, id=id, cloud=cloud, project=project)

    @sdk_docs(
        doc_py_example=_WAIT_EXAMPLE, arg_docstrings=_WAIT_ARG_DOCSTRINGS,
    )
    def wait(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        timeout_s: float = 1800,
        state: Union[str, WorkspaceState] = WorkspaceState.RUNNING,
    ) -> str:
        """Wait for a workspace to reach a terminal state.

        Returns the id of the workspace.
        """
        return self._private_sdk.wait(
            name=name,
            id=id,
            cloud=cloud,
            project=project,
            timeout_s=timeout_s,
            state=state,
        )

    @sdk_docs(
        doc_py_example=_GENERATE_SSH_CONFIG_FILE_EXAMPLE,
        arg_docstrings=_GENERATE_SSH_CONFIG_FILE_ARG_DOCSTRINGS,
    )
    def generate_ssh_config_file(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        ssh_config_path: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Generate an SSH config file for a workspace.

        Returns a tuple of host name and config file path.
        """
        return self._private_sdk.generate_ssh_config_file(
            name=name,
            id=id,
            cloud=cloud,
            project=project,
            ssh_config_path=ssh_config_path,
        )

    @sdk_docs(
        doc_py_example=_RUN_COMMAND_EXAMPLE, arg_docstrings=_RUN_COMMAND_ARG_DOCSTRINGS,
    )
    def run_command(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        command: str,
        **kwargs,
    ):
        """Run a command on a workspace.

        Returns the output of the command.
        """
        return self._private_sdk.run_command(
            name=name, id=id, cloud=cloud, project=project, command=command, **kwargs
        )

    @sdk_docs(
        doc_py_example=_PULL_EXAMPLE, arg_docstrings=_PULL_ARG_DOCSTRINGS,
    )
    def pull(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        local_dir: Optional[str] = None,
        pull_git_state: bool = False,
        rsync_args: Optional[List[str]] = None,
        delete: bool = False,
    ):
        """Pull a file from a workspace.

        With --delete, files in the local directory that don't exist in the workspace
        will be removed. Excluded files (like .git) are preserved and not deleted.
        """
        self._private_sdk.pull(
            name=name,
            id=id,
            cloud=cloud,
            project=project,
            local_dir=local_dir,
            pull_git_state=pull_git_state,
            rsync_args=rsync_args,
            delete=delete,
        )

    @sdk_docs(
        doc_py_example=_PUSH_EXAMPLE, arg_docstrings=_PUSH_ARG_DOCSTRINGS,
    )
    def push(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        local_dir: Optional[str] = None,
        push_git_state: bool = False,
        rsync_args: Optional[List[str]] = None,
        delete: bool = False,
    ):
        """Push a directory to a workspace.

        With --delete, files in the workspace that don't exist locally will be removed.
        Excluded files (like .git) are preserved and not deleted.
        """
        self._private_sdk.push(
            name=name,
            id=id,
            cloud=cloud,
            project=project,
            local_dir=local_dir,
            push_git_state=push_git_state,
            rsync_args=rsync_args,
            delete=delete,
        )

    @sdk_docs(
        doc_py_example=_UPDATE_EXAMPLE, arg_docstrings=_UPDATE_ARG_DOCSTRINGS,
    )
    def update(  # noqa: F811
        self, *, id: Optional[str] = None, config: UpdateWorkspaceConfig  # noqa: A002
    ):
        """Update a workspace."""
        self._private_sdk.update(
            id=id, config=config,  # type: ignore
        )

    @sdk_docs(
        doc_py_example=_GET_EXAMPLE, arg_docstrings=_GET_ARG_DOCSTRINGS,
    )
    def get(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_config: bool = True,
    ) -> Workspace:
        """Get a workspace.

        Args:
            include_config: If True (default), fetch full workspace config. Set to False for efficiency.
        """
        return self._private_sdk.get(
            name=name,
            id=id,
            cloud=cloud,
            project=project,
            include_config=include_config,
        )

    @sdk_docs(
        doc_py_example=_LIST_EXAMPLE, arg_docstrings=_LIST_ARG_DOCSTRINGS,
    )
    def list(  # noqa: F811, A001, PLR0913, PLR0917
        self,
        *,
        workspace_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        creator_id: Optional[str] = None,
        state_filter: Optional[Union[List["WorkspaceState"], List[str]]] = None,
        tags_filter: Optional[Dict[str, List[str]]] = None,
        include_config: bool = False,
        sort_field: Optional[Union[str, "WorkspaceSortField"]] = None,
        sort_order: Optional[Union[str, "WorkspaceSortOrder"]] = None,
        max_items: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> ResultIterator[Workspace]:
        """List workspaces with optional filters.

        Returns an iterator of Workspace objects. By default, filters to non-terminated states
        and does not fetch expensive config data (set include_config=True if needed).
        """
        return self._private_sdk.list(
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

    @sdk_docs(
        doc_py_example=_TAGS_ADD_EXAMPLE, arg_docstrings=_TAGS_ADD_ARG_DOCSTRINGS,
    )
    def add_tags(  # noqa: F811
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        tags: Dict[str, str],
    ) -> None:
        """Upsert (add/update) tag key/value pairs for a workspace."""
        return self._private_sdk.add_tags(
            id=id, name=name, cloud=cloud, project=project, tags=tags
        )

    @sdk_docs(
        doc_py_example=_TAGS_REMOVE_EXAMPLE, arg_docstrings=_TAGS_REMOVE_ARG_DOCSTRINGS,
    )
    def remove_tags(  # noqa: F811
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        keys: List[str],
    ) -> None:
        """Remove tags by key from a workspace."""
        return self._private_sdk.remove_tags(
            id=id, name=name, cloud=cloud, project=project, keys=keys
        )

    @sdk_docs(
        doc_py_example=_TAGS_LIST_EXAMPLE, arg_docstrings=_TAGS_LIST_ARG_DOCSTRINGS,
    )
    def list_tags(  # noqa: F811
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Dict[str, str]:
        """List tags for a workspace."""
        return self._private_sdk.list_tags(
            id=id, name=name, cloud=cloud, project=project
        )
