from typing import List, Optional

from anyscale._private.sdk import sdk_command
from anyscale.policy._private.policy_sdk import PrivatePolicySDK
from anyscale.policy.models import Policy, PolicyConfig, ResourcePolicy


_POLICY_SDK_SINGLETON_KEY = "policy_sdk"

_SET_EXAMPLE = """
import anyscale
from anyscale.policy.models import PolicyConfig, PolicyBinding

# Set policy for a cloud
policy_config = PolicyConfig(
    bindings=[
        PolicyBinding(role_name="collaborator", principals=["ug_abc123"]),
        PolicyBinding(role_name="readonly", principals=["ug_def456", "ug_ghi789"]),
    ]
)
anyscale.policy.set(
    resource_type="cloud",
    resource_id="cld_abc123",
    config=policy_config,
)

# Set policy for your organization (no resource_id needed)
org_policy = PolicyConfig(
    bindings=[
        PolicyBinding(role_name="owner", principals=["ug_admins"]),
        PolicyBinding(role_name="collaborator", principals=["ug_developers"]),
    ]
)
anyscale.policy.set(
    resource_type="organization",
    config=org_policy,
)
"""

_SET_ARG_DOCSTRINGS = {
    "resource_type": "Resource type ('cloud', 'project', or 'organization').",
    "resource_id": "Resource ID (e.g., cld_abc123, prj_xyz789). Required for 'cloud' and 'project' types, not allowed for 'organization'.",
    "config": "Policy configuration with role bindings.",
}

_GET_EXAMPLE = """
import anyscale
from anyscale.policy.models import Policy

# Get policy for a cloud
policy = anyscale.policy.get(resource_type="cloud", resource_id="cld_abc123")
for binding in policy.bindings:
    print(f"{binding.role_name}: {binding.principals}")

# Get policy for your organization (no resource_id needed)
org_policy = anyscale.policy.get(resource_type="organization")
for binding in org_policy.bindings:
    print(f"{binding.role_name}: {binding.principals}")
"""

_GET_ARG_DOCSTRINGS = {
    "resource_type": "Resource type ('cloud', 'project', or 'organization').",
    "resource_id": "Resource ID (e.g., cld_abc123, prj_xyz789). Required for 'cloud' and 'project' types, not allowed for 'organization'.",
}

_LIST_EXAMPLE = """
import anyscale
from anyscale.policy.models import ResourcePolicy

policies = anyscale.policy.list(resource_type="cloud")
for policy in policies:
    print(f"{policy.resource_id}: {policy.bindings}")
"""

_LIST_ARG_DOCSTRINGS = {
    "resource_type": "Resource type to list policies for ('cloud' or 'project').",
}


@sdk_command(
    _POLICY_SDK_SINGLETON_KEY,
    PrivatePolicySDK,
    doc_py_example=_SET_EXAMPLE,
    arg_docstrings=_SET_ARG_DOCSTRINGS,
)
def set(  # noqa: A001
    resource_type: str,
    config: PolicyConfig,
    resource_id: Optional[str] = None,
    *,
    _private_sdk: Optional[PrivatePolicySDK] = None,
):
    """Set user group permission policy for a resource.

    For organization policies, resource_id cannot be specified, the policy will
    be set for your current organization automatically.

    Valid role_name values by resource type:

    **Cloud**:
    - `collaborator`: Read/write access (create, read, update, delete)
    - `readonly`: Read-only access

    **Project**:
    - `collaborator`: Read/write access (create, read, update)
    - `readonly`: Read-only access

    **Organization**:
    - `owner`: Full control (write + collaborator management)
    - `collaborator`: Read/write access to organization resources
    """
    # For organization type, resource_id is not allowed - the backend uses auth_context.organization_id.
    if resource_type.lower() == "organization":
        if resource_id is not None:
            raise ValueError(
                "resource_id cannot be specified for 'organization' resource type. "
                "The policy will be set for your current organization automatically."
            )
        # Use "_" as placeholder since the URL path requires a resource_id parameter
        resource_id = "_"
    elif resource_id is None:
        raise ValueError(
            f"resource_id is required for resource type '{resource_type}'."
        )

    return _private_sdk.set(  # type: ignore
        resource_type=resource_type, resource_id=resource_id, config=config,
    )


@sdk_command(
    _POLICY_SDK_SINGLETON_KEY,
    PrivatePolicySDK,
    doc_py_example=_GET_EXAMPLE,
    arg_docstrings=_GET_ARG_DOCSTRINGS,
)
def get(
    resource_type: str,
    resource_id: Optional[str] = None,
    *,
    _private_sdk: Optional[PrivatePolicySDK] = None,
) -> Policy:
    """Get user group permission policy for a resource.

    For organization policies, resource_id cannot be specified, the policy for
    your current organization will be returned automatically.

    Returns a Policy object with role bindings.
    """
    # For organization type, resource_id is not allowed - the backend uses auth_context.organization_id.
    if resource_type.lower() == "organization":
        if resource_id is not None:
            raise ValueError(
                "resource_id cannot be specified for 'organization' resource type. "
                "The policy will be retrieved for your current organization automatically."
            )
        # Use "_" as placeholder since the URL path requires a resource_id parameter
        resource_id = "_"
    elif resource_id is None:
        raise ValueError(
            f"resource_id is required for resource type '{resource_type}'."
        )

    return _private_sdk.get(  # type: ignore
        resource_type=resource_type, resource_id=resource_id,
    )


@sdk_command(
    _POLICY_SDK_SINGLETON_KEY,
    PrivatePolicySDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_ARG_DOCSTRINGS,
)
def list(  # noqa: A001
    resource_type: str, *, _private_sdk: Optional[PrivatePolicySDK] = None
) -> List[ResourcePolicy]:
    """List permission policies for all resources of a specific type.

    Returns a list of ResourcePolicy objects.
    """
    return _private_sdk.list(resource_type=resource_type)  # type: ignore
