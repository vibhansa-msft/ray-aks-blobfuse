from typing import List

from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.user_group.models import UserGroup


class PrivateUserGroupSDK(BaseSDK):
    """Private SDK for user group operations."""

    def list(self, max_items: int = 50) -> List[UserGroup]:
        """
        List user groups in the organization.

        Args:
            max_items: Maximum number of user groups to return.

        Returns:
            List of UserGroup objects.
        """
        response = self.client.list_user_groups(count=max_items)
        if response.results is None:
            return []
        return [UserGroup.from_api_model(ug) for ug in response.results]

    def get(self, group_id: str) -> UserGroup:
        """
        Get a specific user group by ID.

        Args:
            group_id: The ID of the user group.

        Returns:
            UserGroup object.
        """
        api_model = self.client.get_user_group(group_id)
        return UserGroup.from_api_model(api_model)
