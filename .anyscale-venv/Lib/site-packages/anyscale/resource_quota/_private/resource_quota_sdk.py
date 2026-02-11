from typing import List, Optional

from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.client.openapi_client.models.create_resource_quota import (
    CreateResourceQuota as CreateResourceQuotaModel,
)
from anyscale.client.openapi_client.models.quota import Quota as QuotaModel
from anyscale.resource_quota.models import CreateResourceQuota, Quota, ResourceQuota


class PrivateResourceQuotaSDK(BaseSDK):
    def create(self, create_resource_quota: CreateResourceQuota) -> ResourceQuota:
        cloud_id = self.client.get_cloud_id(
            cloud_name=create_resource_quota.cloud, compute_config_id=None
        )

        project_id = (
            self.client.get_project_id(
                parent_cloud_id=cloud_id, name=create_resource_quota.project
            )
            if create_resource_quota.project
            else None
        )

        user_id = None
        if create_resource_quota.user_email:
            users = self.client.get_organization_collaborators(
                email=create_resource_quota.user_email
            )

            if len(users) == 0:
                raise ValueError(
                    f"User with email '{create_resource_quota.user_email}' not found."
                )

            if len(users) > 1:
                raise ValueError(
                    f"Multiple users found for email '{create_resource_quota.user_email}'. Please contact Anyscale support."
                )
            user_id = users[0].user_id

        create_resource_quota_model = CreateResourceQuotaModel(
            name=create_resource_quota.name,
            cloud_id=cloud_id,
            project_id=project_id,
            user_id=user_id,
            quota=QuotaModel(
                num_cpus=create_resource_quota.num_cpus,
                num_instances=create_resource_quota.num_instances,
                num_gpus=create_resource_quota.num_gpus,
                num_accelerators=create_resource_quota.num_accelerators,
            ),
        )

        resource_quota = self.client.create_resource_quota(
            create_resource_quota=create_resource_quota_model
        )

        return ResourceQuota(
            id=resource_quota.id,
            name=resource_quota.name,
            cloud_id=resource_quota.cloud_id,
            project_id=resource_quota.project_id,
            user_id=resource_quota.user_id,
            is_enabled=resource_quota.is_enabled,
            created_at=resource_quota.created_at,
            deleted_at=resource_quota.deleted_at,
            quota=Quota(
                num_cpus=resource_quota.quota.num_cpus,
                num_instances=resource_quota.quota.num_instances,
                num_gpus=resource_quota.quota.num_gpus,
                num_accelerators=resource_quota.quota.num_accelerators,
            ),
        )

    def list(
        self,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        creator_id: Optional[str] = None,
        is_enabled: Optional[bool] = None,
        max_items: int = 20,
    ) -> List[ResourceQuota]:
        cloud_id = (
            self.client.get_cloud_id(cloud_name=cloud, compute_config_id=None)
            if cloud
            else None
        )

        resource_quotas = self.client.list_resource_quotas(
            name, cloud_id, creator_id, is_enabled, max_items,
        )

        return [
            ResourceQuota(
                id=resource_quota.id,
                name=resource_quota.name,
                cloud_id=resource_quota.cloud_id,
                project_id=resource_quota.project_id,
                user_id=resource_quota.user_id,
                is_enabled=resource_quota.is_enabled,
                created_at=resource_quota.created_at,
                deleted_at=resource_quota.deleted_at,
                quota=Quota(
                    num_cpus=resource_quota.quota.num_cpus,
                    num_instances=resource_quota.quota.num_instances,
                    num_gpus=resource_quota.quota.num_gpus,
                    num_accelerators=resource_quota.quota.num_accelerators,
                ),
            )
            for resource_quota in resource_quotas
        ]

    def delete(self, resource_quota_id: str) -> None:
        self.client.delete_resource_quota(resource_quota_id=resource_quota_id)

    def set_status(self, resource_quota_id: str, is_enabled: bool) -> None:
        self.client.set_resource_quota_status(
            resource_quota_id=resource_quota_id, is_enabled=is_enabled
        )
