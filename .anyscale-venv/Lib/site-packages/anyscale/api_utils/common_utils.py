import os
from typing import Optional, Tuple

from anyscale.client.openapi_client.api.default_api import DefaultApi as InternalApi
from anyscale.client.openapi_client.models import (
    Cloud,
    ExperimentalWorkspace,
)
from anyscale.sdk.anyscale_client.api.default_api import DefaultApi as ExternalApi
from anyscale.sdk.anyscale_client.models import Project


WORKSPACE_ID_ENV_VAR = "ANYSCALE_EXPERIMENTAL_WORKSPACE_ID"

# TODO (sumanthrh): Currently this same functionality of getting workspace id is repeated in many places, we should try to use this util everywhere
def get_current_workspace_id() -> Optional[str]:
    """Returns the ID of the workspace the client is running in (or `None`)."""
    return os.environ.get(WORKSPACE_ID_ENV_VAR, None)


def source_cloud_id_and_project_id(
    *,
    internal_api: InternalApi,
    external_api: ExternalApi,
    cloud_id: Optional[str],
    project_id: Optional[str],
) -> Tuple[str, str]:
    """
        Returns `Tuple[cloud_id, project_id]` based on optionally provided `cloud_id` and `project_id`.

        - If neither `cloud_id` nor `project_id` are provided, the default cloud and its default project are used
        (unless inside a workspace)
        - If both `cloud_id` and `project_id` are provided, we verify that the project is under the cloud.

        Note that the relationship hierarchy is:
        ```
        Cloud -----------<- Project
        1                     Many
        ```
        - Each cloud will have at least one project (the default project).
        - Each project will belong to exactly one cloud.
        """
    if not cloud_id:
        if project_id:
            # 1. No cloud, project
            # (Use the cloud of the project)
            project: Project = external_api.get_project(project_id).result
            cloud_id = project.parent_cloud_id
        else:
            # 2. No cloud, no project
            # (Use the default cloud and its default project)
            # (unless we are in an Anyscale workspace, then use the workspace's cloud and project)
            workspace_id = get_current_workspace_id()
            is_anyscale_workspace = workspace_id is not None
            if is_anyscale_workspace:
                workspace: ExperimentalWorkspace = internal_api.get_workspace_api_v2_experimental_workspaces_workspace_id_get(
                    workspace_id
                ).result
                cloud_id = workspace.cloud_id
                project_id = workspace.project_id
            else:
                cloud: Cloud = external_api.get_default_cloud().result
                cloud_id = cloud.id
                project = external_api.get_default_project(
                    parent_cloud_id=cloud_id
                ).result
                project_id = project.id
    elif not project_id:
        # 3. Cloud, no project
        # (Use the default project of the cloud)
        project = external_api.get_default_project(parent_cloud_id=cloud_id).result
        project_id = project.id
    else:
        # 4. Cloud, project
        # (Verify that the project is in the cloud)
        project = external_api.get_project(project_id).result
        assert (
            project.parent_cloud_id == cloud_id
        ), f"Project {project_id} is not in cloud {cloud_id}."
    return (cloud_id, project_id)  # type: ignore
