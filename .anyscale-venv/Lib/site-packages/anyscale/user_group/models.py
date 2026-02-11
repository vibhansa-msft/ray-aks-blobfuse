from dataclasses import dataclass, field
from datetime import datetime

from anyscale._private.models import ModelBase


@dataclass(frozen=True)
class UserGroup(ModelBase):
    """A user group in the organization."""

    __doc_py_example__ = """\
import anyscale

user_group = anyscale.user_group.get(id="ug_abc123")
print(f"{user_group.id}: {user_group.name}")
"""

    id: str = field(metadata={"docstring": "The unique identifier of the user group."})

    def _validate_id(self, id: str):  # noqa: A002
        if not isinstance(id, str):
            raise TypeError("id must be a string.")

    name: str = field(metadata={"docstring": "The name of the user group."})

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

    org_id: str = field(metadata={"docstring": "The organization ID."})

    def _validate_org_id(self, org_id: str):
        if not isinstance(org_id, str):
            raise TypeError("org_id must be a string.")

    created_at: datetime = field(
        metadata={"docstring": "When the user group was created."}
    )

    def _validate_created_at(self, created_at: datetime):
        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime.")

    updated_at: datetime = field(
        metadata={"docstring": "When the user group was last updated."}
    )

    def _validate_updated_at(self, updated_at: datetime):
        if not isinstance(updated_at, datetime):
            raise TypeError("updated_at must be a datetime.")

    @classmethod
    def from_api_model(cls, api_model) -> "UserGroup":
        """Create a UserGroup from an API model."""
        return cls(
            id=api_model.id,
            name=api_model.name,
            org_id=api_model.org_id,
            created_at=api_model.created_at,
            updated_at=api_model.updated_at,
        )
