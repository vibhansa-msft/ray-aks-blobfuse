from typing import List, Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.user._private.user_sdk import PrivateUserSDK
from anyscale.user.commands import (
    _ADMIN_BATCH_CREATE_ARG_DOCSTRINGS,
    _ADMIN_BATCH_CREATE_EXAMPLE,
    _GET_ARG_DOCSTRINGS,
    _GET_EXAMPLE,
    _LIST_ARG_DOCSTRINGS,
    _LIST_EXAMPLE,
    admin_batch_create as admin_batch_create,
    get as get,
    list as list,  # noqa: A004 - claude_comment("claude-opus-4-5", "SDK public API re-export")
)
from anyscale.user.models import AdminCreatedUser, AdminCreateUser, User


class UserSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateUserSDK(client=client, logger=logger, timer=timer)

    @sdk_docs(
        doc_py_example=_ADMIN_BATCH_CREATE_EXAMPLE,
        arg_docstrings=_ADMIN_BATCH_CREATE_ARG_DOCSTRINGS,
    )
    def admin_batch_create(  # noqa: F811
        self, admin_create_users: List[AdminCreateUser],
    ) -> List[AdminCreatedUser]:
        """Batch create, as an admin, users without email verification.
        """
        return self._private_sdk.admin_batch_create(admin_create_users)

    @sdk_docs(
        doc_py_example=_LIST_EXAMPLE, arg_docstrings=_LIST_ARG_DOCSTRINGS,
    )
    def list(  # noqa: F811
        self,
        *,
        email: Optional[str] = None,
        name: Optional[str] = None,
        collaborator_type: Optional[str] = None,
        is_service_account: Optional[bool] = None,
        max_items: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> ResultIterator[User]:
        """List organization collaborators."""
        return self._private_sdk.list(
            email=email,
            name=name,
            collaborator_type=collaborator_type,
            is_service_account=is_service_account,
            max_items=max_items,
            page_size=page_size,
        )

    @sdk_docs(
        doc_py_example=_GET_EXAMPLE, arg_docstrings=_GET_ARG_DOCSTRINGS,
    )
    def get(  # noqa: F811
        self,
        *,
        email: Optional[str] = None,
        name: Optional[str] = None,
        collaborator_type: Optional[str] = None,
        is_service_account: Optional[bool] = None,
    ) -> User:
        """Retrieve a single collaborator by email or name."""
        return self._private_sdk.get(
            email=email,
            name=name,
            collaborator_type=collaborator_type,
            is_service_account=is_service_account,
        )
