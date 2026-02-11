from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

from anyscale._private.models import ModelBase, ModelEnum


class PolicySyncStatus(ModelEnum):
    """Sync status for resource permission policies."""

    pending = "pending"
    success = "success"
    failed = "failed"

    __docstrings__: ClassVar[Dict[str, str]] = {
        "pending": "Policy is pending synchronization.",
        "success": "Policy has been successfully synchronized.",
        "failed": "Policy synchronization has failed.",
    }


@dataclass(frozen=True)
class PolicyBinding(ModelBase):
    """A binding of a role to a list of principals (user group IDs)."""

    __doc_py_example__ = """\
from anyscale.policy.models import PolicyBinding

binding = PolicyBinding(role_name="collaborator", principals=["ug_abc123"])
"""

    role_name: str = field(
        metadata={
            "docstring": (
                "The role name. For cloud/project policies use 'collaborator' or "
                "'readonly'. For organization "
                "policies use 'owner' or 'collaborator'."
            )
        }
    )

    def _validate_role_name(self, role_name: str) -> str:
        if not isinstance(role_name, str):
            raise TypeError("role_name must be a string.")
        normalized = role_name.strip().lower()
        if normalized == "write":
            normalized = "collaborator"
        allowed = {"owner", "collaborator", "readonly"}
        if normalized not in allowed:
            raise ValueError(
                f"Invalid role_name '{role_name}'. Allowed values: "
                f"{', '.join(sorted(allowed))}."
            )
        return normalized

    principals: List[str] = field(
        metadata={"docstring": "List of user group IDs that have this role."}
    )

    def _validate_principals(self, principals: List[str]):
        if not isinstance(principals, list):
            raise TypeError("principals must be a list.")


@dataclass(frozen=True)
class PolicyConfig(ModelBase):
    """Policy configuration with role bindings."""

    __doc_yaml_example__ = """\
bindings:
  - role_name: collaborator
    principals:
      - ug_abc123
  - role_name: readonly
    principals:
      - ug_def456
      - ug_ghi789
"""

    __doc_py_example__ = """\
from anyscale.policy.models import PolicyBinding, PolicyConfig

config = PolicyConfig(
    bindings=[
        PolicyBinding(role_name="collaborator", principals=["ug_abc123"]),
        PolicyBinding(role_name="readonly", principals=["ug_def456", "ug_ghi789"]),
    ]
)
"""

    bindings: List[PolicyBinding] = field(
        metadata={"docstring": "List of role bindings for the policy."}
    )

    def _validate_bindings(self, bindings: List[PolicyBinding]):
        if not isinstance(bindings, list):
            raise TypeError("bindings must be a list.")


@dataclass(frozen=True)
class ResourcePolicy(ModelBase):
    """Resource policy model representing permissions for a resource."""

    __doc_py_example__ = """\
import anyscale
from anyscale.policy.models import ResourcePolicy

policies = anyscale.policy.list(resource_type="cloud")
for policy in policies:
    print(f"{policy.resource_id}: {policy.bindings} (sync_status: {policy.sync_status})")
"""

    resource_id: str = field(metadata={"docstring": "The ID of the resource."})

    def _validate_resource_id(self, resource_id: str):
        if not isinstance(resource_id, str):
            raise TypeError("resource_id must be a string.")

    resource_type: str = field(
        metadata={"docstring": "The type of the resource (e.g., 'cloud', 'project')."}
    )

    def _validate_resource_type(self, resource_type: str):
        if not isinstance(resource_type, str):
            raise TypeError("resource_type must be a string.")

    bindings: List[PolicyBinding] = field(
        metadata={"docstring": "List of role bindings for the policy."}
    )

    def _validate_bindings(self, bindings: List[PolicyBinding]):
        if not isinstance(bindings, list):
            raise TypeError("bindings must be a list.")

    sync_status: PolicySyncStatus = field(
        metadata={
            "docstring": "Sync status of the policy (pending, success, or failed)."
        }
    )

    def _validate_sync_status(self, sync_status: PolicySyncStatus):
        if not isinstance(sync_status, PolicySyncStatus):
            raise TypeError("sync_status must be a PolicySyncStatus.")


@dataclass(frozen=True)
class Policy(ModelBase):
    """Policy model representing the policy for a single resource."""

    __doc_py_example__ = """\
import anyscale
from anyscale.policy.models import Policy

policy = anyscale.policy.get(resource_type="cloud", resource_id="cld_abc123")
print(f"Sync status: {policy.sync_status}")
for binding in policy.bindings:
    print(f"{binding.role_name}: {binding.principals}")
"""

    bindings: List[PolicyBinding] = field(
        metadata={"docstring": "List of role bindings for the policy."}
    )

    def _validate_bindings(self, bindings: List[PolicyBinding]):
        if not isinstance(bindings, list):
            raise TypeError("bindings must be a list.")

    sync_status: PolicySyncStatus = field(
        metadata={
            "docstring": "Sync status of the policy (pending, success, or failed)."
        }
    )

    def _validate_sync_status(self, sync_status: PolicySyncStatus):
        if not isinstance(sync_status, PolicySyncStatus):
            raise TypeError("sync_status must be a PolicySyncStatus.")
