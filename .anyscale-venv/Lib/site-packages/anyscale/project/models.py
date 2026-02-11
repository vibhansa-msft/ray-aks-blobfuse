import contextlib
from dataclasses import dataclass, field, fields
from datetime import datetime
import json
from typing import Any, ClassVar, Dict, List, Optional, Union

from anyscale._private.models import ModelBase, ModelEnum


class ProjectPermissionLevel(ModelEnum):
    """Permission levels for project collaborators."""

    OWNER = "OWNER"
    WRITE = "WRITE"
    READONLY = "READONLY"

    __docstrings__: ClassVar[Dict[str, str]] = {
        OWNER: "Owner permission level for the project",
        WRITE: "Write permission level for the project",
        READONLY: "Readonly permission level for the project",
    }


@dataclass(frozen=True)
class CreateProjectCollaborator(ModelBase):
    """User to be added as a collaborator to a project.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.project.models import ProjectPermissionLevel, CreateProjectCollaborator
create_project_collaborator = CreateProjectCollaborator(
   # Email of the user to be added as a collaborator
    email="test@anyscale.com",
    # Permission level for the user to the project (ProjectPermissionLevel.OWNER, ProjectPermissionLevel.WRITE, ProjectPermissionLevel.READONLY)
    permission_level=ProjectPermissionLevel.READONLY,
)
"""

    def _validate_email(self, email: str):
        if not isinstance(email, str):
            raise TypeError("Email must be a string.")

    email: str = field(
        metadata={"docstring": "Email of the user to be added as a collaborator."},
    )

    def _validate_permission_level(
        self, permission_level: ProjectPermissionLevel
    ) -> ProjectPermissionLevel:
        if isinstance(permission_level, str):
            return ProjectPermissionLevel.validate(permission_level)
        elif isinstance(permission_level, ProjectPermissionLevel):
            return permission_level
        else:
            raise TypeError(
                f"'permission_level' must be a 'ProjectPermissionLevel' (it is {type(permission_level)})."
            )

    permission_level: ProjectPermissionLevel = field(  # type: ignore
        default=ProjectPermissionLevel.READONLY,  # type: ignore
        metadata={
            "docstring": "Permission level the added user should have for the project"  # type: ignore
            f"(one of: {','.join([str(m.value) for m in ProjectPermissionLevel])}",  # type: ignore
        },
    )


@dataclass(frozen=True)
class CreateProjectCollaborators(ModelBase):
    """List of users to be added as collaborators to a project.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.project.models import ProjectPermissionLevel, CreateProjectCollaborator, CreateProjectCollaborators
create_project_collaborator = CreateProjectCollaborator(
   # Email of the user to be added as a collaborator
    email="test@anyscale.com",
    # Permission level for the user to the project (ProjectPermissionLevel.OWNER, ProjectPermissionLevel.WRITE, ProjectPermissionLevel.READONLY)
    permission_level=ProjectPermissionLevel.READONLY,
)
create_project_collaborators = CreateProjectCollaborators(
    collaborators=[create_project_collaborator]
)
"""

    collaborators: List[Dict[str, Any]] = field(
        metadata={
            "docstring": "List of users to be added as collaborators to a project."
        },
    )

    def _validate_collaborators(self, collaborators: List[Dict[str, Any]]):
        if not isinstance(collaborators, list):
            raise TypeError("Collaborators must be a list.")


@dataclass(frozen=True)
class ProjectMinimal(ModelBase):
    """Minimal Project object."""

    id: str = field(metadata={"docstring": "ID of the project."},)

    def _validate_id(self, id: str):  # noqa: A002
        if not isinstance(id, str):
            raise TypeError("'id' must be a string.")

    name: str = field(metadata={"docstring": "Name of the project."},)

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")

    description: str = field(metadata={"docstring": "Description of the project."},)

    def _validate_description(self, description: str):
        if not isinstance(description, str):
            raise TypeError("'description' must be a string.")

    created_at: str = field(
        metadata={"docstring": "Datetime of the project creation."},
    )

    def _validate_created_at(self, created_at: str):
        if not isinstance(created_at, str):
            raise TypeError("'created_at' must be a string.")

    creator_id: Optional[str] = field(
        default=None, metadata={"docstring": "ID of the creator of the project."},
    )

    def _validate_creator_id(self, creator_id: Optional[str]):
        if creator_id is not None and not isinstance(creator_id, str):
            raise TypeError("'creator_id' must be a string.")

    parent_cloud_id: Optional[str] = field(
        default=None, metadata={"docstring": "ID of the parent cloud."},
    )

    def _validate_parent_cloud_id(self, parent_cloud_id: Optional[str]):
        if parent_cloud_id is not None and not isinstance(parent_cloud_id, str):
            raise TypeError("'parent_cloud_id' must be a string.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectMinimal":
        # remove any fields that are not in the dataclass
        valid_fields = {field_.name for field_ in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass(frozen=True)
class Project(ProjectMinimal):
    """Project object.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.project.models import Project

project: Project = anyscale.project.get(project_id="my-project-id")
"""

    is_owner: bool = field(
        default=False,
        metadata={"docstring": "Whether the user is the owner of the project."},
    )

    def _validate_is_owner(self, is_owner: bool):
        if not isinstance(is_owner, bool):
            raise TypeError("'is_owner' must be a boolean.")

    is_read_only: bool = field(
        default=False,
        metadata={"docstring": "Whether the user has read-only access to the project."},
    )

    def _validate_is_read_only(self, is_read_only: bool):
        if not isinstance(is_read_only, bool):
            raise TypeError("'is_read_only' must be a boolean.")

    directory_name: str = field(
        default="",
        metadata={
            "docstring": "Directory name of project to be used as working directory of clusters."
        },
    )

    def _validate_directory_name(self, directory_name: str):
        if not isinstance(directory_name, str):
            raise TypeError("'directory_name' must be a string.")

    is_default: bool = field(
        default=False,
        metadata={
            "docstring": "Whether the project is the default project for the organization."
        },
    )

    def _validate_is_default(self, is_default: bool):
        if not isinstance(is_default, bool):
            raise TypeError("'is_default' must be a boolean.")

    initial_cluster_config: Optional[Union[str, Dict[str, Any]]] = field(
        default=None,
        metadata={"docstring": "Initial cluster config associated with the project."},
    )

    def _validate_initial_cluster_config(
        self, initial_cluster_config: Union[str, Dict[str, Any], None]
    ):
        if initial_cluster_config is not None and not isinstance(
            initial_cluster_config, (str, dict)
        ):
            raise TypeError("'initial_cluster_config' must be a string or dictionary.")

    last_used_cloud_id: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "ID of the last cloud used in this project, or by the user if this is a new project."
        },
    )

    def _validate_last_used_cloud_id(self, last_used_cloud_id: Optional[str]):
        if last_used_cloud_id is not None and not isinstance(last_used_cloud_id, str):
            raise TypeError("'last_used_cloud_id' must be a string.")

    owners: List[str] = field(
        default_factory=list,
        metadata={
            "docstring": "List of IDs of users who have owner access to the project."
        },
    )

    def _validate_owners(self, owners: List[str]):
        if not isinstance(owners, list):
            raise TypeError("'owners' must be a list.")
        for owner in owners:
            if not isinstance(owner, str):
                raise TypeError(f"'{owner}' must be a string.")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Project":
        data = d.copy()
        # convert datetime fields to string
        for field_ in fields(cls):
            if field_.name in data and isinstance(data[field_.name], datetime):
                data[field_.name] = data[field_.name].strftime("%Y-%m-%d %H:%M:%S")

        # convert initial_cluster_config JSON string to dict
        if "initial_cluster_config" in data and isinstance(
            data["initial_cluster_config"], str
        ):
            with contextlib.suppress(json.JSONDecodeError):
                data["initial_cluster_config"] = json.loads(
                    data["initial_cluster_config"]
                )

        # convert owners list of dicts to list of string IDs
        if "owners" in data and isinstance(data["owners"], list):
            data["owners"] = [owner.get("id") for owner in data["owners"]]

        # remove any fields that are not in the dataclass
        valid_fields = {field_.name for field_ in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class ProjectSortField(ModelEnum):
    """Field to sort projects by."""

    NAME = "NAME"

    __docstrings__: ClassVar[Dict[str, str]] = {
        NAME: "Sort by project name.",
    }


class ProjectSortOrder(ModelEnum):
    """Direction of sorting."""

    ASC = "ASC"
    DESC = "DESC"

    __docstrings__: ClassVar[Dict[str, str]] = {
        ASC: "Sort in ascending order.",
        DESC: "Sort in descending order.",
    }
