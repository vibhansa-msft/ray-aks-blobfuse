from typing import List, Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.project._private.project_sdk import PrivateProjectSDK
from anyscale.project.commands import (
    _ADD_COLLABORATORS_DOCSTRINGS,
    _ADD_COLLABORATORS_EXAMPLE,
    _CREATE_PROJECT_DOCSTRINGS,
    _CREATE_PROJECT_EXAMPLE,
    _DELETE_PROJECT_DOCSTRINGS,
    _DELETE_PROJECT_EXAMPLE,
    _GET_DEFAULT_PROJECT_DOCSTRINGS,
    _GET_DEFAULT_PROJECT_EXAMPLE,
    _GET_PROJECT_DOCSTRINGS,
    _GET_PROJECT_EXAMPLE,
    _LIST_PROJECTS_DOCSTRINGS,
    _LIST_PROJECTS_EXAMPLE,
    add_collaborators as add_collaborators,
    create as create,
    delete as delete,
    get as get,
    get_default as get_default,
    list as list,  # noqa: A004 - claude_comment("claude-opus-4-5", "SDK public API re-export")
)
from anyscale.project.models import (
    CreateProjectCollaborator,
    Project,
    ProjectPermissionLevel as ProjectPermissionLevel,
    ProjectSortField,
    ProjectSortOrder,
)


class ProjectSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateProjectSDK(client=client, logger=logger, timer=timer)

    @sdk_docs(
        doc_py_example=_ADD_COLLABORATORS_EXAMPLE,
        arg_docstrings=_ADD_COLLABORATORS_DOCSTRINGS,
    )
    def add_collaborators(  # noqa: F811
        self, cloud: str, project: str, collaborators: List[CreateProjectCollaborator],
    ) -> None:
        """Batch add collaborators to a project.
        """
        self._private_sdk.add_collaborators(cloud, project, collaborators)

    @sdk_docs(
        doc_py_example=_GET_PROJECT_EXAMPLE, arg_docstrings=_GET_PROJECT_DOCSTRINGS,
    )
    def get(self, project_id: str) -> Project:  # noqa: F811
        """Get details of a project."""
        return self._private_sdk.get(project_id)

    @sdk_docs(
        doc_py_example=_LIST_PROJECTS_EXAMPLE, arg_docstrings=_LIST_PROJECTS_DOCSTRINGS,
    )
    def list(  # noqa: F811
        self,
        *,
        name_contains: Optional[str] = None,
        creator_id: Optional[str] = None,
        parent_cloud_id: Optional[str] = None,
        include_defaults: bool = True,
        max_items: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_field: Optional[ProjectSortField] = None,
        sort_order: Optional[ProjectSortOrder] = None,
    ) -> ResultIterator[Project]:
        """List all projects with optional filters.

        Returns
        -------
        ResultIterator[Project]
            An iterator yielding Project objects. Fetches pages
            lazily as needed. Use a `for` loop to iterate or `list()`
            to consume all at once.
        """
        return self._private_sdk.list(
            name_contains=name_contains,
            creator_id=creator_id,
            parent_cloud_id=parent_cloud_id,
            include_defaults=include_defaults,
            max_items=max_items,
            page_size=page_size,
            sort_field=sort_field,
            sort_order=sort_order,
        )

    @sdk_docs(
        doc_py_example=_CREATE_PROJECT_EXAMPLE,
        arg_docstrings=_CREATE_PROJECT_DOCSTRINGS,
    )
    def create(  # noqa: F811
        self,
        name: str,
        parent_cloud_id: str,
        *,
        description: Optional[str] = None,
        initial_cluster_config: Optional[str] = None,
    ) -> str:
        """Create a project."""
        return self._private_sdk.create(
            name,
            description or "",
            parent_cloud_id,
            initial_cluster_config=initial_cluster_config,
        )

    @sdk_docs(
        doc_py_example=_DELETE_PROJECT_EXAMPLE,
        arg_docstrings=_DELETE_PROJECT_DOCSTRINGS,
    )
    def delete(self, project_id: str):  # noqa: F811
        """Delete a project."""
        self._private_sdk.delete(project_id)

    @sdk_docs(
        doc_py_example=_GET_DEFAULT_PROJECT_EXAMPLE,
        arg_docstrings=_GET_DEFAULT_PROJECT_DOCSTRINGS,
    )
    def get_default(self, parent_cloud_id: str) -> Project:  # noqa: F811
        """Get the default project for a cloud."""
        return self._private_sdk.get_default(parent_cloud_id)
