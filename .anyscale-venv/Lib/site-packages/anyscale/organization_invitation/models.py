from dataclasses import dataclass, field
from datetime import datetime

from anyscale._private.models import ModelBase


@dataclass(frozen=True)
class OrganizationInvitation(ModelBase):
    """Organization invitation model.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.organization_invitation.models import OrganizationInvitation

organization_invitations: List[OrganizationInvitation] = anyscale.organization_invitation.list()
"""

    id: str = field(metadata={"docstring": "ID of the organization invitation."},)

    def _validate_id(self, id: str):  # noqa: A002
        if not isinstance(id, str):
            raise TypeError("id must be a string.")

    email: str = field(metadata={"docstring": "Email of the organization invitation."})

    def _validate_email(self, email: str):
        if not isinstance(email, str):
            raise TypeError("email must be a string.")

    created_at: datetime = field(
        metadata={"docstring": "Creation time of the organization invitation."},
    )

    def _validate_created_at(self, created_at: datetime):
        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime.")

    expires_at: datetime = field(
        metadata={"docstring": "Expiration time of the organization invitation."},
    )

    def _validate_expires_at(self, expires_at: datetime):
        if not isinstance(expires_at, datetime):
            raise TypeError("expires_at must be a datetime.")
