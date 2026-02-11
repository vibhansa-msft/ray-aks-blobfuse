from typing import List, Optional

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.client.openapi_client import (
    CreateUserProjectCollaborator,
    CreateUserProjectCollaboratorValue,
    Project as OpenAPIProject,
    ProjectListResponse,
    WriteProject,
)
from anyscale.project.models import (
    CreateProjectCollaborator,
    Project,
    ProjectSortField,
    ProjectSortOrder,
)


MAX_PAGE_SIZE = 50


class PrivateProjectSDK(BaseSDK):
    def add_collaborators(
        self, cloud: str, project: str, collaborators: List[CreateProjectCollaborator]
    ) -> None:
        cloud_id = self.client.get_cloud_id(cloud_name=cloud, compute_config_id=None)
        project_id = self.client.get_project_id(parent_cloud_id=cloud_id, name=project)

        self.client.add_project_collaborators(
            project_id=project_id,
            collaborators=[
                CreateUserProjectCollaborator(
                    value=CreateUserProjectCollaboratorValue(email=collaborator.email),
                    permission_level=collaborator.permission_level.lower(),
                )
                for collaborator in collaborators
            ],
        )

    def get(self, project_id: str) -> Optional[Project]:
        project: OpenAPIProject = self.client.get_project(project_id)
        return Project.from_dict(project.to_dict()) if project else None

    def list(
        self,
        *,
        # filters
        name_contains: Optional[str] = None,
        creator_id: Optional[str] = None,
        parent_cloud_id: Optional[str] = None,
        include_defaults: bool = True,
        # pagination
        max_items: Optional[int] = None,
        page_size: Optional[int] = None,
        # sorting
        sort_field: Optional[ProjectSortField] = None,
        sort_order: Optional[ProjectSortOrder] = None,
    ) -> ResultIterator[Project]:
        if max_items is not None and (max_items <= 0):
            raise ValueError("'max_items' must be greater than 0.")

        if page_size is not None and (page_size <= 0 or page_size > MAX_PAGE_SIZE):
            raise ValueError(
                f"'page_size' must be between 1 and {MAX_PAGE_SIZE}, inclusive."
            )

        def _fetch_page(token: Optional[str],) -> ProjectListResponse:
            return self.client.list_projects(
                name_contains=name_contains,
                creator_id=creator_id,
                parent_cloud_id=parent_cloud_id,
                include_defaults=include_defaults,
                sort_field=str(sort_field) if sort_field else None,
                sort_order=str(sort_order) if sort_order else None,
                paging_token=token,
                count=page_size,
            )

        def _openapi_project_to_sdk_project(openapi_project: OpenAPIProject) -> Project:
            return Project.from_dict(openapi_project.to_dict())

        return ResultIterator(
            page_token=None,
            max_items=max_items,
            fetch_page=_fetch_page,
            parse_fn=_openapi_project_to_sdk_project,
        )

    def create(
        self,
        name: str,
        description: str,
        parent_cloud_id: str,
        *,
        initial_cluster_config: Optional[str] = None,
    ) -> str:
        project = self.client.create_project(
            project=WriteProject(
                name=name,
                description=description,
                parent_cloud_id=parent_cloud_id,
                initial_cluster_config=initial_cluster_config,
            )
        )
        return project.id  # type: ignore

    def delete(
        self, project_id: str,
    ):
        self.client.delete_project(project_id)

    def get_default(self, parent_cloud_id: str,) -> Project:
        project: OpenAPIProject = self.client.get_default_project(parent_cloud_id)
        return Project.from_dict(project.to_dict())
