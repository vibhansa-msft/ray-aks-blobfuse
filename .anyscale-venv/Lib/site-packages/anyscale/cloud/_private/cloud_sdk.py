from typing import List, Optional

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models import (
    Cloud as CloudModel,
    CreateCloudCollaborator as CreateCloudCollaboratorModel,
)
from anyscale.cloud.models import (
    Cloud,
    CloudProvider,
    ComputeStack,
    CreateCloudCollaborator,
)
from anyscale.sdk.anyscale_client.models import ClusterState


logger = BlockLogger()


class PrivateCloudSDK(BaseSDK):
    def add_collaborators(
        self, cloud: str, collaborators: List[CreateCloudCollaborator]
    ) -> None:
        cloud_id = self.client.get_cloud_id(cloud_name=cloud, compute_config_id=None)

        self.client.add_cloud_collaborators(
            cloud_id=cloud_id,
            collaborators=[
                CreateCloudCollaboratorModel(
                    email=collaborator.email,
                    permission_level=collaborator.permission_level.lower(),
                )
                for collaborator in collaborators
            ],
        )

    def get(
        self, id: Optional[str], name: Optional[str],  # noqa: A002
    ) -> Optional[Cloud]:
        if (id and name) or (not id and not name):
            raise ValueError("Provide exactly one of 'id' or 'name'.")

        if id:
            openapi_cloud = self.client.get_cloud(cloud_id=id)
        else:
            assert name is not None, "Name must be provided if id is not."
            openapi_cloud = self.client.get_cloud_by_name(name=name)

        return self._to_sdk_cloud(openapi_cloud)

    def get_default(self) -> Optional[Cloud]:
        openapi_cloud = self.client.get_default_cloud()

        return self._to_sdk_cloud(openapi_cloud)

    def list(
        self,
        *,
        cloud_id: Optional[str] = None,
        name: Optional[str] = None,
        max_items: Optional[int] = None,
        page_size: Optional[int] = None,
    ):
        # Single item by ID
        if cloud_id is not None:
            openapi_cloud = self.client.get_cloud(cloud_id=cloud_id)
            cloud = self._to_sdk_cloud(openapi_cloud)
            return [cloud] if cloud is not None else []

        # Single item by name
        if name is not None:
            openapi_cloud = self.client.get_cloud_by_name(name=name)
            cloud = self._to_sdk_cloud(openapi_cloud)
            return [cloud] if cloud is not None else []

        # Iterator over all
        def _fetch(token: Optional[str]):
            return self.client.list_clouds(paging_token=token, count=page_size)

        return ResultIterator[Cloud](
            page_token=None,
            max_items=max_items,
            fetch_page=_fetch,
            parse_fn=self._to_sdk_cloud,
        )

    def _to_sdk_cloud(self, openapi_cloud: Optional["CloudModel"]) -> Optional[Cloud]:
        if openapi_cloud is None:
            return None

        # Validate provider, default to UNKNOWN if validation fails
        if openapi_cloud.provider is not None:
            try:
                provider = CloudProvider.validate(openapi_cloud.provider)
            except ValueError:
                provider = CloudProvider.UNKNOWN
        else:
            provider = CloudProvider.UNKNOWN

        # Validate compute_stack, default to UNKNOWN if validation fails
        if openapi_cloud.compute_stack is not None:
            try:
                compute_stack = ComputeStack.validate(openapi_cloud.compute_stack)
            except ValueError:
                compute_stack = ComputeStack.UNKNOWN
        else:
            compute_stack = ComputeStack.UNKNOWN

        return Cloud(
            id=openapi_cloud.id,
            name=openapi_cloud.name,
            provider=provider,
            region=openapi_cloud.region,
            created_at=openapi_cloud.created_at,
            is_default=openapi_cloud.is_default,
            compute_stack=compute_stack,
        )

    def terminate_system_cluster(self, cloud_id: str, wait: bool) -> str:
        resp = self.client.terminate_system_cluster(cloud_id)
        if wait:
            self._wait_for_system_cluster_status(cloud_id, ClusterState.TERMINATED)
        else:
            logger.info(f"System cluster termination initiated for cloud {cloud_id}.")
        return resp.result.cluster_id

    def _wait_for_system_cluster_status(
        self,
        cloud_id: str,
        goal_status: str,
        timeout_s: int = 500,
        interval_s: int = 10,
    ) -> bool:
        self.logger.info("Waiting for system cluster termination...", end="")
        for _ in self.timer.poll(timeout_s=timeout_s, interval_s=interval_s):
            status = self.client.describe_system_workload_get_status(cloud_id)
            if status == goal_status:
                print(".")
                self.logger.info(f"System cluster for cloud '{cloud_id}' is {status}.")
                return True
            else:
                print(".", end="")
        raise TimeoutError(
            f"Timed out waiting for system cluster termination for cloud '{cloud_id}'. Last seen status: {status}."
        )
