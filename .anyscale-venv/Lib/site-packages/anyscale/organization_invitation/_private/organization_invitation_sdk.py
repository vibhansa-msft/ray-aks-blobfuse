from typing import List, Tuple

from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.organization_invitation.models import OrganizationInvitation


class PrivateOrganizationInvitationSDK(BaseSDK):
    def create(self, emails: List[str]) -> Tuple[List[str], List[str]]:
        return self.client.create_organization_invitations(emails=emails)

    def list(self) -> List[OrganizationInvitation]:
        invitations = self.client.list_organization_invitations()
        return [
            OrganizationInvitation(
                id=invitation.id,
                email=invitation.email,
                created_at=invitation.created_at,
                expires_at=invitation.expires_at,
            )
            for invitation in invitations
        ]

    def delete(self, email: str) -> str:
        return self.client.delete_organization_invitation(email=email).email
