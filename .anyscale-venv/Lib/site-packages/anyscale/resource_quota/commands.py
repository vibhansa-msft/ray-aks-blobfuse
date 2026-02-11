from typing import List, Optional

from anyscale._private.sdk import sdk_command
from anyscale.resource_quota._private.resource_quota_sdk import PrivateResourceQuotaSDK
from anyscale.resource_quota.models import CreateResourceQuota, ResourceQuota


_RESOURCE_QUOTA_SDK_SINGLETON_KEY = "resource_quota_sdk"

_CREATE_EXAMPLE = """
import anyscale
from anyscale.resource_quota.models import CreateResourceQuota

anyscale.resource_quota.create(
    CreateResourceQuota(
        name="my-resource-quota",
        cloud="my-cloud",
        project="my-project",
        num_cpus=2,
    ),
)
"""

_CREATE_DOCSTRINGS = {"create_resource_quota": "The resource quota to be created."}

_LIST_EXAMPLE = """
import anyscale

anyscale.resource_quota.list(
    name="my-resource-quota",
    cloud="my-cloud",
    creator_id="usr_123",
    is_enabled=True,
    max_items=20,
)
"""

_LIST_DOCSTRINGS = {
    "name": "Name of the resource quota.",
    "cloud": "Name of the cloud that this resource quota applies to.",
    "creator_id": "ID of the creator of the resource quota.",
    "is_enabled": "Whether the resource quota is enabled.",
    "max_items": "Maximum number of items to return.",
}

_DELETE_EXAMPLE = """
import anyscale

anyscale.resource_quota.delete(
    resource_quota_id="rq_123",
)
"""

_DELETE_DOCSTRINGS = {
    "resource_quota_id": "ID of the resource quota to delete.",
}

_ENABLE_EXAMPLE = """
import anyscale

anyscale.resource_quota.enable(
    resource_quota_id="rq_123",
)
"""

_ENABLE_DOCSTRINGS = {
    "resource_quota_id": "ID of the resource quota to enable.",
}

_DISABLE_EXAMPLE = """
import anyscale

anyscale.resource_quota.disable(
    resource_quota_id="rq_123",
)
"""

_DISABLE_DOCSTRINGS = {
    "resource_quota_id": "ID of the resource quota to disable.",
}


@sdk_command(
    _RESOURCE_QUOTA_SDK_SINGLETON_KEY,
    PrivateResourceQuotaSDK,
    doc_py_example=_CREATE_EXAMPLE,
    arg_docstrings=_CREATE_DOCSTRINGS,
)
def create(
    create_resource_quota: CreateResourceQuota,
    *,
    _private_sdk: Optional[PrivateResourceQuotaSDK] = None
) -> ResourceQuota:
    """Create a resource quota.
    """
    return _private_sdk.create(create_resource_quota)  # type: ignore


@sdk_command(
    _RESOURCE_QUOTA_SDK_SINGLETON_KEY,
    PrivateResourceQuotaSDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_DOCSTRINGS,
)
def list(  # noqa: A001
    name: Optional[str] = None,
    cloud: Optional[str] = None,
    creator_id: Optional[str] = None,
    is_enabled: Optional[bool] = None,
    max_items: int = 20,
    *,
    _private_sdk: Optional[PrivateResourceQuotaSDK] = None
) -> List[ResourceQuota]:
    """List resource quotas. """
    return _private_sdk.list(name, cloud, creator_id, is_enabled, max_items,)  # type: ignore


@sdk_command(
    _RESOURCE_QUOTA_SDK_SINGLETON_KEY,
    PrivateResourceQuotaSDK,
    doc_py_example=_DELETE_EXAMPLE,
    arg_docstrings=_DELETE_DOCSTRINGS,
)
def delete(
    resource_quota_id: str, *, _private_sdk: Optional[PrivateResourceQuotaSDK] = None
):
    """Delete a resource quota.
    """
    return _private_sdk.delete(resource_quota_id)  # type: ignore


@sdk_command(
    _RESOURCE_QUOTA_SDK_SINGLETON_KEY,
    PrivateResourceQuotaSDK,
    doc_py_example=_ENABLE_EXAMPLE,
    arg_docstrings=_ENABLE_DOCSTRINGS,
)
def enable(
    resource_quota_id: str, *, _private_sdk: Optional[PrivateResourceQuotaSDK] = None
):
    """Enable a resource quota.
    """
    return _private_sdk.set_status(resource_quota_id, True)  # type: ignore


@sdk_command(
    _RESOURCE_QUOTA_SDK_SINGLETON_KEY,
    PrivateResourceQuotaSDK,
    doc_py_example=_DISABLE_EXAMPLE,
    arg_docstrings=_DISABLE_DOCSTRINGS,
)
def disable(
    resource_quota_id: str, *, _private_sdk: Optional[PrivateResourceQuotaSDK] = None
):
    """Disable a resource quota.
    """
    return _private_sdk.set_status(resource_quota_id, False)  # type: ignore
