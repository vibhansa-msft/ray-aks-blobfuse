from typing import List, Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.service_account._private.service_account_sdk import (
    PrivateServiceAccountSDK,
)
from anyscale.service_account.commands import (
    _CREATE_API_KEY_DOCSTRINGS,
    _CREATE_API_KEY_EXAMPLE,
    _CREATE_DOCSTRINGS,
    _CREATE_EXAMPLE,
    _DELETE_DOCSTRINGS,
    _DELETE_EXAMPLE,
    _LIST_DOCSTRINGS,
    _LIST_EXAMPLE,
    _ROTATE_API_KEYS_DOCSTRINGS,
    _ROTATE_API_KEYS_EXAMPLE,
    create as create,
    create_api_key as create_api_key,
    delete as delete,
    list as list,  # noqa: A004 - claude_comment("claude-opus-4-5", "SDK public API re-export")
    rotate_api_keys as rotate_api_keys,
)
from anyscale.service_account.models import ServiceAccount


class ServiceAccountSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateServiceAccountSDK(
            client=client, logger=logger, timer=timer
        )

    @sdk_docs(
        doc_py_example=_CREATE_EXAMPLE, arg_docstrings=_CREATE_DOCSTRINGS,
    )
    def create(self, name: str) -> str:  # noqa: F811
        """Create a service account and return the API key.
        """
        return self._private_sdk.create(name)

    @sdk_docs(
        doc_py_example=_CREATE_API_KEY_EXAMPLE,
        arg_docstrings=_CREATE_API_KEY_DOCSTRINGS,
    )
    def create_api_key(  # noqa: F811
        self, email: Optional[str] = None, name: Optional[str] = None
    ) -> str:
        """Create an API key for the service account and return the API key.
        """
        return self._private_sdk.create_api_key(email, name)

    @sdk_docs(
        doc_py_example=_LIST_EXAMPLE, arg_docstrings=_LIST_DOCSTRINGS,
    )
    def list(self, max_items: int = 20,) -> List[ServiceAccount]:  # noqa: F811
        """List service accounts.
        """
        return self._private_sdk.list(max_items=max_items,)

    @sdk_docs(
        doc_py_example=_DELETE_EXAMPLE, arg_docstrings=_DELETE_DOCSTRINGS,
    )
    def delete(  # noqa: F811
        self, email: Optional[str] = None, name: Optional[str] = None
    ):
        """Delete a service account.
        """
        return self._private_sdk.delete(email, name)

    @sdk_docs(
        doc_py_example=_ROTATE_API_KEYS_EXAMPLE,
        arg_docstrings=_ROTATE_API_KEYS_DOCSTRINGS,
    )
    def rotate_api_key(
        self, email: Optional[str] = None, name: Optional[str] = None
    ) -> str:  # noqa: F811
        """Rotate all api keys of a service account and return the new API key.
        """
        return self._private_sdk.rotate_api_keys(email, name)
