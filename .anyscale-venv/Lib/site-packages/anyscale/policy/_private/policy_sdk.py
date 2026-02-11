from typing import List

from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.client.openapi_client.models import Binding, UpdatePolicyRequest
from anyscale.policy.models import (
    Policy,
    PolicyBinding,
    PolicyConfig,
    PolicySyncStatus,
    ResourcePolicy,
)


class PrivatePolicySDK(BaseSDK):
    """Private SDK for resource policy operations."""

    def set(
        self, resource_type: str, resource_id: str, config: PolicyConfig,
    ):
        """
        Set user group permission policy for a resource.

        Args:
            resource_type: Resource type ('cloud', 'project', or 'organization')
            resource_id: Resource ID (e.g., cld_abc123, prj_xyz789)
            config: Policy configuration with role bindings
        """
        api_bindings = []
        for b in config.bindings:
            role = b.role_name
            if role == "collaborator" and resource_type in ("cloud", "project"):
                role = "write"
            api_bindings.append(Binding(role_name=role, principals=b.principals))
        api_policy = UpdatePolicyRequest(bindings=api_bindings)

        self.client.update_resource_policy(
            resource_type=resource_type, resource_id=resource_id, policy=api_policy,
        )

    def get(self, resource_type: str, resource_id: str,) -> Policy:
        """
        Get user group permission policy for a resource.

        Args:
            resource_type: Resource type ('cloud', 'project', or 'organization')
            resource_id: Resource ID (e.g., cld_abc123, prj_xyz789)

        Returns:
            Policy object with role bindings and sync status.
        """
        response = self.client.get_resource_policy(
            resource_type=resource_type, resource_id=resource_id,
        )

        bindings = []
        for b in response.bindings:
            role = b.role_name
            if role == "write" and resource_type in ("cloud", "project"):
                role = "collaborator"
            bindings.append(PolicyBinding(role_name=role, principals=b.principals))

        return Policy(
            bindings=bindings, sync_status=PolicySyncStatus(response.sync_status),
        )

    def list(self, resource_type: str,) -> List[ResourcePolicy]:
        """
        List permission policies for all resources of a specific type.

        Args:
            resource_type: Resource type to list policies for ('cloud' or 'project')

        Returns:
            List of ResourcePolicy objects with sync status.
        """
        # Convert singular to plural for API (API expects 'clouds' or 'projects')
        api_resource_type = f"{resource_type}s"
        response = self.client.list_resource_policies(resource_type=api_resource_type)

        policies = []
        for item in response.results:
            bindings = []
            for b in item.bindings:
                role = b.role_name
                if role == "write" and resource_type in ("cloud", "project"):
                    role = "collaborator"
                bindings.append(PolicyBinding(role_name=role, principals=b.principals))

            policies.append(
                ResourcePolicy(
                    resource_id=item.resource_id,
                    resource_type=item.resource_type,
                    bindings=bindings,
                    sync_status=PolicySyncStatus(item.sync_status),
                )
            )

        return policies
