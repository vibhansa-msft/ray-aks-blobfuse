from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Dict

from anyscale._private.models import ModelBase, ModelEnum


# TODO(cynthiakwu): Move this when we have organization collaborator sdk
class OrganizationPermissionLevel(ModelEnum):
    """Permission levels for service accounts in an organization."""

    OWNER = "OWNER"
    COLLABORATOR = "COLLABORATOR"

    __docstrings__: ClassVar[Dict[str, str]] = {
        OWNER: "Owner permission level for the organization",
        COLLABORATOR: "Collaborator permission level for the organization",
    }


@dataclass(frozen=True)
class ServiceAccount(ModelBase):
    """Service account
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.service_account.models import ServiceAccount

service_accounts: List[ServiceAccount] = anyscale.service_account.list()
"""

    name: str = field(metadata={"docstring": "Name of the service account."})

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

    created_at: datetime = field(
        metadata={"docstring": "The timestamp when this service account was created."}
    )

    def _validate_created_at(self, created_at: datetime):
        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime.")

    permission_level: OrganizationPermissionLevel = field(
        metadata={
            "docstring": "The organization permission level of the service account."
        }
    )

    def _validate_permission_level(
        self, permission_level: OrganizationPermissionLevel
    ) -> OrganizationPermissionLevel:
        if isinstance(permission_level, str):
            return OrganizationPermissionLevel.validate(permission_level)
        elif isinstance(permission_level, OrganizationPermissionLevel):
            return permission_level
        else:
            raise TypeError(
                f"'permission_level' must be a 'OrganizationPermissionLevel' (it is {type(permission_level)})."
            )

    email: str = field(metadata={"docstring": "Email of the service account."})

    def _validate_email(self, email: str):
        if not isinstance(email, str):
            raise TypeError("email must be a string.")
