from typing import List, Optional

from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.client.openapi_client.models.organization_collaborator import (
    OrganizationCollaborator,
)
from anyscale.service_account.models import ServiceAccount


ONE_HUNDRED_YEARS_IN_SECONDS = 3153600000


class PrivateServiceAccountSDK(BaseSDK):
    def _get_service_account_identifier(
        self, email: Optional[str], name: Optional[str]
    ) -> str:
        # Logic can be simplified but kept verbose for clarity and typing
        if not email and not name:
            raise ValueError("Either email or name must be provided.")
        if email and name:
            raise ValueError("Only one of email or name can be provided.")
        if email:
            return email
        elif name:
            return name

        raise ValueError("Internal server error. Please contact support.")

    def _validate_exactly_one_service_account_per_email_or_name(
        self, service_accounts: List[OrganizationCollaborator], identifier: str,
    ):
        if len(service_accounts) == 0:
            raise ValueError(f"No service account {identifier} found.")

        if len(service_accounts) > 1:
            names = [sa.name for sa in service_accounts]
            emails = [sa.email for sa in service_accounts]
            raise ValueError(
                f"Found {len(service_accounts)} service accounts matching '{identifier}'. "
                f"Names: {names}, Emails: {emails}. This should not happen - please contact support."
            )

    def _get_service_account(
        self, email: Optional[str], name: Optional[str]
    ) -> OrganizationCollaborator:
        identifier = self._get_service_account_identifier(email, name)
        service_accounts = self.client.get_organization_collaborators(
            email=email, name=name, is_service_account=True
        )

        if name is not None:
            service_accounts = [sa for sa in service_accounts if sa.name == name]
        if email is not None:
            service_accounts = [sa for sa in service_accounts if sa.email == email]

        self._validate_exactly_one_service_account_per_email_or_name(
            service_accounts, identifier
        )
        service_account = service_accounts[0]

        return service_account

    def create(self, name: str) -> str:
        service_account = self.client.create_service_account(name=name)

        api_key = self.client.create_api_key(
            ONE_HUNDRED_YEARS_IN_SECONDS, service_account.user_id
        )

        return api_key.server_session_id

    def create_api_key(self, email: Optional[str], name: Optional[str]) -> str:
        service_account = self._get_service_account(email, name)

        api_key = self.client.create_api_key(
            ONE_HUNDRED_YEARS_IN_SECONDS, service_account.user_id
        )

        return api_key.server_session_id

    def list(self, max_items: int = 20,) -> List[ServiceAccount]:
        service_accounts = self.client.get_organization_collaborators(
            is_service_account=True
        )

        return [
            ServiceAccount(
                name=service_account.name,
                created_at=service_account.created_at,
                permission_level=service_account.permission_level,
                email=service_account.email,
            )
            for service_account in service_accounts[:max_items]
        ]

    def delete(self, email: Optional[str], name: Optional[str]) -> None:
        service_account = self._get_service_account(email, name)

        self.client.delete_organization_collaborator(service_account.id)

    def rotate_api_keys(self, email: Optional[str], name: Optional[str]) -> str:
        service_account = self._get_service_account(email, name)

        self.client.rotate_api_key(service_account.user_id)

        api_key = self.client.create_api_key(
            ONE_HUNDRED_YEARS_IN_SECONDS, service_account.user_id
        )

        return api_key.server_session_id
