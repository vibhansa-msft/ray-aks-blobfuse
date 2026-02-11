from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Dict, List, Optional, Union

from anyscale._private.models import ImageURI, ModelBase
from anyscale._private.models.model_base import ModelEnum
from anyscale.compute_config.models import (
    compute_config_type_from_dict,
    ComputeConfig,
    ComputeConfigType,
    MultiResourceComputeConfig,
)


class WorkspaceState(ModelEnum):
    """Possible states for a workspace."""

    STARTING = "STARTING"
    UPDATING = "UPDATING"
    RUNNING = "RUNNING"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"
    ERRORED = "ERRORED"
    UNKNOWN = "UNKNOWN"

    __docstrings__: ClassVar[Dict[str, str]] = {
        "STARTING": "The workspace is starting up.",
        "UPDATING": "The workspace is updating.",
        "RUNNING": "The workspace is running.",
        "TERMINATING": "The workspace is terminating.",
        "TERMINATED": "The workspace is terminated.",
        "ERRORED": "The workspace is in an error state.",
        "UNKNOWN": "The workspace state",
    }


class WorkspaceSortField(ModelEnum):
    """Fields available for sorting workspaces."""

    STATUS = "STATUS"
    CREATED_AT = "CREATED_AT"
    LATEST_STARTED_AT = "LATEST_STARTED_AT"

    __docstrings__: ClassVar[Dict[str, str]] = {
        "STATUS": "Sort by workspace status (active first by default).",
        "CREATED_AT": "Sort by creation timestamp.",
        "LATEST_STARTED_AT": "Sort by most recent start time.",
    }


class WorkspaceSortOrder(ModelEnum):
    """Enum for sort order directions."""

    ASC = "ASC"
    DESC = "DESC"

    __docstrings__: ClassVar[Dict[str, str]] = {
        "ASC": "Sort in ascending order.",
        "DESC": "Sort in descending order.",
    }


@dataclass(frozen=True)
class UpdateWorkspaceConfig(ModelBase):
    """Configuration options for updating a workspace."""

    __doc_py_example__ = """\
from anyscale.workspace.models import UpdateWorkspaceConfig

config = UpdateWorkspaceConfig(
    name="new-workpsace-name",
    idle_termination_minutes=220,
    env_vars={"key": "value"},
    requirements="/tmp/requirements.txt",
)
"""

    __doc_yaml_example__ = """\
name: new-workspace-name
idle_termination_minutes: 220
env_vars:
    key: value
requirements: /tmp/requirements.txt
tags:
    team: mlops
    env: prod
"""

    name: Optional[str] = field(
        default=None, metadata={"docstring": "The name of the workspace"}
    )

    def _validate_name(self, name: Optional[str]):
        if name is not None and not isinstance(name, str):
            raise TypeError("'name' must be a string.")

    image_uri: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "URI of an existing image. Exclusive with `containerfile`."
        },
    )

    def _validate_image_uri(self, image_uri: Optional[str]):
        if image_uri is not None and not isinstance(image_uri, str):
            raise TypeError(f"'image_uri' must be an str but it is {type(image_uri)}.")
        if image_uri is not None:
            ImageURI.from_str(image_uri)

    containerfile: Optional[str] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "The file path to a containerfile that will be built into an image before running the workload. Exclusive with `image_uri`."
        },
    )

    def _validate_containerfile(self, containerfile: Optional[str]):
        if containerfile is not None and not isinstance(containerfile, str):
            raise TypeError("'containerfile' must be a string.")

    tags: Optional[Dict[str, str]] = field(
        default=None, metadata={"docstring": "Tags to associate with the workspace."},
    )

    def _validate_tags(self, tags: Optional[Dict[str, str]]):
        if tags is None:
            return
        if not isinstance(tags, dict):
            raise TypeError("'tags' must be a Dict[str, str].")
        for k, v in tags.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError("'tags' must be a Dict[str, str].")

    compute_config: Union[ComputeConfigType, Dict, str, None] = field(
        default=None,
        metadata={
            "docstring": "The name of an existing registered compute config (including version) or an inlined ComputeConfig object."
        },
    )

    def _validate_compute_config(
        self, compute_config: Union[ComputeConfigType, Dict, str, None]
    ) -> Union[None, str, ComputeConfigType]:
        if compute_config is None or isinstance(compute_config, str):
            return compute_config

        if isinstance(compute_config, dict):
            compute_config = compute_config_type_from_dict(compute_config)
        if not isinstance(compute_config, (ComputeConfig, MultiResourceComputeConfig)):
            raise TypeError(
                "'compute_config' must be a string, ComputeConfig, MultiResourceComputeConfig, or corresponding dict"
            )

        return compute_config

    idle_termination_minutes: Optional[int] = field(
        default=None,
        metadata={
            "docstring": "the time in minutes after which the workspace is terminated when idle."
        },
    )

    def _validate_idle_termination_minutes(
        self, idle_termination_minutes: Optional[int]
    ):
        if idle_termination_minutes is not None and not isinstance(
            idle_termination_minutes, int
        ):
            raise ValueError("'idle_termination_minutes' must be an int")

    requirements: Optional[Union[str, List[str]]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "A list of pip requirements or a path to a `requirements.txt` file for the workload."
        },
    )

    def _validate_requirements(self, requirements: Optional[Union[str, List[str]]]):
        if requirements is None or isinstance(requirements, str):
            return

        if not isinstance(requirements, list) or not all(
            isinstance(r, str) for r in requirements
        ):
            raise TypeError(
                "'requirements' must be a string (file path) or list of strings."
            )

    env_vars: Optional[Dict[str, str]] = field(
        default=None,
        repr=True,
        metadata={
            "docstring": "A dictionary of environment variables that will be set for the workload."
        },
    )

    def _validate_env_vars(self, env_vars: Optional[Dict[str, str]]):
        if env_vars is not None and (
            not isinstance(env_vars, dict)
            or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env_vars.items()
            )
        ):
            raise TypeError("'env_vars' must be a Dict[str, str].")

    registry_login_secret: Optional[str] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "A name or identifier of the secret containing credentials to authenticate to the docker registry hosting the image. "
            "This can only be used when 'image_uri' is specified and the image is not hosted on Anyscale."
        },
    )

    def _validate_registry_login_secret(self, registry_login_secret: Optional[str]):
        if registry_login_secret is not None and not isinstance(
            registry_login_secret, str
        ):
            raise TypeError("'registry_login_secret' must be a string.")

    ray_version: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The Ray version (X.Y.Z) specified for this image specified by either an image URI or a containerfile. If you don't specify a Ray version, Anyscale defaults to the latest Ray version available at the time of the Anyscale CLI/SDK release."
        },
    )

    def _validate_ray_version(self, ray_version: Optional[str]):
        if ray_version is not None and not isinstance(ray_version, str):
            raise TypeError("'ray_version' must be a string.")
        if ray_version:
            vs = ray_version.split(".")
            if len(vs) != 3:
                raise ValueError(
                    f"Invalid Ray version format: {ray_version}. Must be in the format 'X.Y.Z'."
                )


@dataclass(frozen=True)
class WorkspaceConfig(UpdateWorkspaceConfig):
    """Configuration options for a workspace."""

    __doc_py_example__ = """\
from anyscale.workspace.models import WorkspaceConfig

config = WorkspaceConfig(
    name="my-workpsace",
    idle_termination_minutes=220,
    env_vars={"key": "value"},
    requirements="/tmp/requirements.txt",
)
"""

    __doc_yaml_example__ = """\
name: my-workspace
idle_termination_minutes: 220
env_vars:
    key: value
requirements: /tmp/requirements.txt
tags:
  team: mlops
  env: prod
"""

    project: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The project for the workload. If not provided, the default project for the cloud will be used."
        },
    )

    def _validate_project(self, project: Optional[str]):
        if project is not None and not isinstance(project, str):
            raise TypeError("'project' must be a string.")

    cloud: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used."
        },
    )

    def _validate_cloud(self, cloud: Optional[str]):
        if cloud is not None and not isinstance(cloud, str):
            raise TypeError("'cloud' must be a string.")


@dataclass(frozen=True)
class Workspace(ModelBase):
    """Workspace information including metadata and state."""

    __doc_py_example__ = """\
import anyscale
from anyscale.workspace.models import Workspace

workspace: Workspace = anyscale.workspace.get(name="workspace-name")
first_workspace: Workspace = next(anyscale.workspace.list(max_items=1), None)
"""

    id: str = field(metadata={"docstring": "Unique identifier of the workspace."})

    def _validate_id(self, id: str):  # noqa: A002
        if not isinstance(id, str):
            raise ValueError("The workspace id must be a string.")

    name: str = field(metadata={"docstring": "The name of the workspace."})

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise ValueError("The workspace name must be a string.")

    state: WorkspaceState = field(
        metadata={"docstring": "The current state of the workspace."}
    )

    def _validate_state(self, state: WorkspaceState):
        WorkspaceState.validate(state)

    project_id: Optional[str] = field(
        default=None,
        metadata={"docstring": "Project identifier that owns this workspace."},
    )

    def _validate_project_id(self, project_id: Optional[str]):
        if project_id is not None and not isinstance(project_id, str):
            raise ValueError("The project id must be a string if provided.")

    cloud_id: Optional[str] = field(
        default=None,
        metadata={"docstring": "Cloud identifier where the workspace is running."},
    )

    def _validate_cloud_id(self, cloud_id: Optional[str]):
        if cloud_id is not None and not isinstance(cloud_id, str):
            raise ValueError("The cloud id must be a string if provided.")

    creator_id: Optional[str] = field(
        default=None,
        metadata={"docstring": "Identifier of the user who created the workspace."},
    )

    def _validate_creator_id(self, creator_id: Optional[str]):
        if creator_id is not None and not isinstance(creator_id, str):
            raise ValueError("The creator id must be a string if provided.")

    creator_email: Optional[str] = field(
        default=None,
        metadata={"docstring": "Email address of the user who created the workspace."},
    )

    def _validate_creator_email(self, creator_email: Optional[str]):
        if creator_email is not None and not isinstance(creator_email, str):
            raise ValueError("The creator email must be a string if provided.")

    created_at: Optional[datetime] = field(
        default=None,
        metadata={"docstring": "Timestamp when the workspace was created."},
    )

    def _validate_created_at(self, created_at: Optional[datetime]):
        if created_at is not None and not isinstance(created_at, datetime):
            raise ValueError("The created_at field must be a datetime if provided.")

    last_started_at: Optional[datetime] = field(
        default=None,
        metadata={"docstring": "Timestamp when the workspace was last started."},
    )

    def _validate_last_started_at(self, last_started_at: Optional[datetime]):
        if last_started_at is not None and not isinstance(last_started_at, datetime):
            raise ValueError(
                "The last_started_at field must be a datetime if provided."
            )

    cluster_id: Optional[str] = field(
        default=None, metadata={"docstring": "Cluster identifier for the workspace."},
    )

    def _validate_cluster_id(self, cluster_id: Optional[str]):
        if cluster_id is not None and not isinstance(cluster_id, str):
            raise ValueError("The cluster id must be a string if provided.")

    config: Optional[Union[WorkspaceConfig, Dict]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Full workspace configuration. Only populated for get() operations, not list()."
        },
    )

    def _validate_config(
        self, config: Optional[Union[WorkspaceConfig, Dict]]
    ) -> Optional[WorkspaceConfig]:
        if config is None:
            return None
        if isinstance(config, dict):
            config = WorkspaceConfig.from_dict(config)
        if not isinstance(config, WorkspaceConfig):
            raise TypeError("'config' must be a WorkspaceConfig or corresponding dict.")
        return config
