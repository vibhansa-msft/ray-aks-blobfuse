from typing import List, Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.resource_quota._private.resource_quota_sdk import PrivateResourceQuotaSDK
from anyscale.resource_quota.commands import (
    _CREATE_DOCSTRINGS,
    _CREATE_EXAMPLE,
    _DELETE_DOCSTRINGS,
    _DELETE_EXAMPLE,
    _DISABLE_DOCSTRINGS,
    _DISABLE_EXAMPLE,
    _ENABLE_DOCSTRINGS,
    _ENABLE_EXAMPLE,
    _LIST_DOCSTRINGS,
    _LIST_EXAMPLE,
    create as create,
    delete as delete,
    disable as disable,
    enable as enable,
    list as list,  # noqa: A004 - claude_comment("claude-opus-4-5", "SDK public API re-export")
)
from anyscale.resource_quota.models import CreateResourceQuota, ResourceQuota


class ResourceQuotaSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateResourceQuotaSDK(
            client=client, logger=logger, timer=timer
        )

    @sdk_docs(
        doc_py_example=_CREATE_EXAMPLE, arg_docstrings=_CREATE_DOCSTRINGS,
    )
    def create(  # noqa: F811
        self, create_resource_quota: CreateResourceQuota,
    ) -> ResourceQuota:
        """Create a resource quota.
        """
        return self._private_sdk.create(create_resource_quota)

    @sdk_docs(
        doc_py_example=_LIST_EXAMPLE, arg_docstrings=_LIST_DOCSTRINGS,
    )
    def list(  # noqa: F811
        self,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        creator_id: Optional[str] = None,
        is_enabled: Optional[bool] = None,
        max_items: int = 20,
    ) -> List[ResourceQuota]:
        """List resource quotas.
        """
        return self._private_sdk.list(
            name=name,
            cloud=cloud,
            creator_id=creator_id,
            is_enabled=is_enabled,
            max_items=max_items,
        )

    @sdk_docs(
        doc_py_example=_DELETE_EXAMPLE, arg_docstrings=_DELETE_DOCSTRINGS,
    )
    def delete(  # noqa: F811
        self, resource_quota_id: str,
    ):
        """Delete a resource quota.
        """
        return self._private_sdk.delete(resource_quota_id)

    @sdk_docs(
        doc_py_example=_ENABLE_EXAMPLE, arg_docstrings=_ENABLE_DOCSTRINGS,
    )
    def enable(  # noqa: F811
        self, resource_quota_id: str,
    ):
        """Enable a resource quota.
        """
        return self._private_sdk.set_status(resource_quota_id, True)

    @sdk_docs(
        doc_py_example=_DISABLE_EXAMPLE, arg_docstrings=_DISABLE_DOCSTRINGS,
    )
    def disable(  # noqa: F811
        self, resource_quota_id: str,
    ):
        """Disable a resource quota.
        """
        return self._private_sdk.set_status(resource_quota_id, False)
