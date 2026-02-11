from typing import List, Optional, Union

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.client.openapi_client.models import (
    AdminCreateUser as AdminCreateUserModel,
    CollaboratorType,
    OrganizationCollaborator,
    OrganizationcollaboratorListResponse,
)
from anyscale.user.models import AdminCreatedUser, AdminCreateUser, User


MAX_PAGE_SIZE = 50


class PrivateUserSDK(BaseSDK):
    @staticmethod
    def _normalize_collaborator_type(
        collaborator_type: Optional[Union[str, CollaboratorType]]
    ) -> Optional[str]:
        if collaborator_type is None:
            return None
        if hasattr(collaborator_type, "value"):
            return collaborator_type.value  # type: ignore[attr-defined]
        return str(collaborator_type)

    @staticmethod
    def _collaborator_to_user(collaborator: OrganizationCollaborator) -> User:
        permission_level = collaborator.permission_level
        if hasattr(permission_level, "value"):
            permission_level = permission_level.value  # type: ignore[attr-defined]

        return User(
            name=collaborator.name,
            email=collaborator.email,
            created_at=collaborator.created_at,
            permission_level=str(permission_level),
            user_id=collaborator.user_id,
        )

    def admin_batch_create(
        self, admin_create_users: List[AdminCreateUser]
    ) -> List[AdminCreatedUser]:
        created_users = self.client.admin_batch_create_users(
            [
                AdminCreateUserModel(**admin_create_user.to_dict())
                for admin_create_user in admin_create_users
            ]
        )

        return [
            AdminCreatedUser(
                user_id=created_user.user_id,
                name=created_user.name,
                email=created_user.email,
                created_at=created_user.created_at,
                is_sso_user=created_user.is_sso_user,
                lastname=created_user.lastname,
                title=created_user.title,
            )
            for created_user in created_users
        ]

    def list(
        self,
        *,
        email: Optional[str] = None,
        name: Optional[str] = None,
        collaborator_type: Optional[Union[str, CollaboratorType]] = None,
        is_service_account: Optional[bool] = None,
        max_items: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> ResultIterator[User]:
        if max_items is not None and max_items <= 0:
            raise ValueError("'max_items' must be greater than 0.")

        if page_size is not None and (page_size <= 0 or page_size > MAX_PAGE_SIZE):
            raise ValueError(
                f"'page_size' must be between 1 and {MAX_PAGE_SIZE}, inclusive."
            )

        def _fetch_page(token: Optional[str],) -> OrganizationcollaboratorListResponse:
            return self.client.list_organization_collaborators(
                email=email,
                name=name,
                collaborator_type=self._normalize_collaborator_type(collaborator_type),
                is_service_account=is_service_account,
                count=page_size,
                paging_token=token,
            )

        return ResultIterator(
            page_token=None,
            max_items=max_items,
            fetch_page=_fetch_page,
            parse_fn=self._collaborator_to_user,
        )

    def get(
        self,
        *,
        email: Optional[str] = None,
        name: Optional[str] = None,
        collaborator_type: Optional[Union[str, CollaboratorType]] = None,
        is_service_account: Optional[bool] = None,
    ) -> User:
        if email is None and name is None:
            raise ValueError("Must provide 'email' or 'name'.")

        response = self.client.list_organization_collaborators(
            email=email,
            name=name,
            collaborator_type=self._normalize_collaborator_type(collaborator_type),
            is_service_account=is_service_account,
            count=1,
        )

        if not response.results:
            target = email if email is not None else name
            raise ValueError(f"No user found matching '{target}'.")

        return self._collaborator_to_user(response.results[0])
