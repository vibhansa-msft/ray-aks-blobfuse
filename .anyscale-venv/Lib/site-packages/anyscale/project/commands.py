from typing import List, Optional

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_command
from anyscale.project._private.project_sdk import PrivateProjectSDK
from anyscale.project.models import (
    CreateProjectCollaborator,
    Project,
    ProjectSortField,
    ProjectSortOrder,
)


_PROJECT_SDK_SINGLETON_KEY = "project_sdk"

_ADD_COLLABORATORS_EXAMPLE = """
import anyscale
from anyscale.project.models import CreateProjectCollaborator, ProjectPermissionLevel

anyscale.project.add_collaborators(
    cloud="cloud_name",
    project="project_name",
    collaborators=[
        CreateProjectCollaborator(
            email="test1@anyscale.com",
            permission_level=ProjectPermissionLevel.OWNER,
        ),
        CreateProjectCollaborator(
            email="test2@anyscale.com",
            permission_level=ProjectPermissionLevel.WRITE,
        ),
        CreateProjectCollaborator(
            email="test3@anyscale.com",
            permission_level=ProjectPermissionLevel.READONLY,
        ),
    ],
)
"""

_ADD_COLLABORATORS_DOCSTRINGS = {
    "cloud": "The cloud that the project belongs to.",
    "project": "The project to add users to.",
    "collaborators": "The list of collaborators to add to the project.",
}


@sdk_command(
    _PROJECT_SDK_SINGLETON_KEY,
    PrivateProjectSDK,
    doc_py_example=_ADD_COLLABORATORS_EXAMPLE,
    arg_docstrings=_ADD_COLLABORATORS_DOCSTRINGS,
)
def add_collaborators(
    cloud: str,
    project: str,
    collaborators: List[CreateProjectCollaborator],
    *,
    _private_sdk: Optional[PrivateProjectSDK] = None
) -> str:
    """Batch add collaborators to a project.
    """
    return _private_sdk.add_collaborators(cloud, project, collaborators)  # type: ignore


_GET_PROJECT_EXAMPLE = """
import anyscale
from anyscale.project.models import Project

project: Project = anyscale.project.get(project_id="my-project-id")
"""

_GET_PROJECT_DOCSTRINGS = {
    "project_id": "The ID of the project to get details of.",
}


@sdk_command(
    _PROJECT_SDK_SINGLETON_KEY,
    PrivateProjectSDK,
    doc_py_example=_GET_PROJECT_EXAMPLE,
    arg_docstrings=_GET_PROJECT_DOCSTRINGS,
)
def get(
    project_id: str, *, _private_sdk: Optional[PrivateProjectSDK] = None
) -> Project:
    """Get details of a project.
    """
    return _private_sdk.get(project_id)  # type: ignore


_LIST_PROJECTS_EXAMPLE = """
from typing import Iterator

import anyscale
from anyscale.project.models import Project, ProjectSortField, ProjectSortOrder

projects: Iterator[Project] = anyscale.project.list(
    name_contains="my-project",
    creator_id="my-user-id",
    parent_cloud_id="my-cloud-id",
    include_defaults=True,
    max_items=20,
    page_size=10,
    sort_field=ProjectSortField.NAME,
    sort_order=ProjectSortOrder.ASC,
)
for project in projects:
    print(project.name)
"""

_LIST_PROJECTS_DOCSTRINGS = {
    "name_contains": "A string to filter projects by name.",
    "creator_id": "The ID of a creator to filter projects.",
    "parent_cloud_id": "The ID of a parent cloud to filter projects.",
    "include_defaults": "Whether to include default projects.",
    "max_items": "The maximum number of projects to return.",
    "page_size": "The number of projects to return per page.",
    "sort_field": "The field to sort projects by.",
    "sort_order": "The order to sort projects by.",
}


@sdk_command(
    _PROJECT_SDK_SINGLETON_KEY,
    PrivateProjectSDK,
    doc_py_example=_LIST_PROJECTS_EXAMPLE,
    arg_docstrings=_LIST_PROJECTS_DOCSTRINGS,
)
def list(  # noqa: A001
    *,
    name_contains: Optional[str] = None,
    creator_id: Optional[str] = None,
    parent_cloud_id: Optional[str] = None,
    include_defaults: bool = True,
    max_items: Optional[int] = None,
    page_size: Optional[int] = None,
    sort_field: Optional[ProjectSortField] = None,
    sort_order: Optional[ProjectSortOrder] = None,
    _private_sdk: Optional[PrivateProjectSDK] = None
) -> ResultIterator[Project]:
    """List projects.
    """
    return _private_sdk.list(  # type: ignore
        name_contains=name_contains,
        creator_id=creator_id,
        parent_cloud_id=parent_cloud_id,
        include_defaults=include_defaults,
        max_items=max_items,
        page_size=page_size,
        sort_field=sort_field,
        sort_order=sort_order,
    )


_CREATE_PROJECT_EXAMPLE = """
import anyscale

project_id: str = anyscale.project.create(
    name="my-project",
    parent_cloud_id="my-cloud-id",
    description="my-project-description",
)
"""

_CREATE_PROJECT_DOCSTRINGS = {
    "name": "The name of the project.",
    "parent_cloud_id": "The parent cloud that the project belongs to.",
    "description": "The description of the project.",
    "initial_cluster_config": "A YAML string containing the initial cluster config for the project.",
}


@sdk_command(
    _PROJECT_SDK_SINGLETON_KEY,
    PrivateProjectSDK,
    doc_py_example=_CREATE_PROJECT_EXAMPLE,
    arg_docstrings=_CREATE_PROJECT_DOCSTRINGS,
)
def create(
    name: str,
    parent_cloud_id: str,
    *,
    description: Optional[str] = None,
    initial_cluster_config: Optional[str] = None,
    _private_sdk: Optional[PrivateProjectSDK] = None
) -> str:
    """Create a project.
    """
    return _private_sdk.create(  # type: ignore
        name,
        description or "",
        parent_cloud_id,
        initial_cluster_config=initial_cluster_config,
    )


_DELETE_PROJECT_EXAMPLE = """
import anyscale

anyscale.project.delete(project_id="my-project-id")
"""

_DELETE_PROJECT_DOCSTRINGS = {
    "project_id": "The ID of the project to delete.",
}


@sdk_command(
    _PROJECT_SDK_SINGLETON_KEY,
    PrivateProjectSDK,
    doc_py_example=_DELETE_PROJECT_EXAMPLE,
    arg_docstrings=_DELETE_PROJECT_DOCSTRINGS,
)
def delete(project_id: str, *, _private_sdk: Optional[PrivateProjectSDK] = None):
    """Delete a project.
    """
    _private_sdk.delete(project_id)  # type: ignore


_GET_DEFAULT_PROJECT_EXAMPLE = """
import anyscale
from anyscale.project.models import Project

project: Project = anyscale.project.get_default(parent_cloud_id="my-cloud-id")
"""

_GET_DEFAULT_PROJECT_DOCSTRINGS = {
    "parent_cloud_id": "The ID of the parent cloud to get the default project for.",
}


@sdk_command(
    _PROJECT_SDK_SINGLETON_KEY,
    PrivateProjectSDK,
    doc_py_example=_GET_DEFAULT_PROJECT_EXAMPLE,
    arg_docstrings=_GET_DEFAULT_PROJECT_DOCSTRINGS,
)
def get_default(
    parent_cloud_id: str, *, _private_sdk: Optional[PrivateProjectSDK] = None
) -> Project:
    """Get the default project for a cloud.
    """
    return _private_sdk.get_default(parent_cloud_id)  # type: ignore
