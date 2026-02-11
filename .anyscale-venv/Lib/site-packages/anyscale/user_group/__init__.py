from typing import List, Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.user_group._private.user_group_sdk import PrivateUserGroupSDK
from anyscale.user_group.commands import (
    _GET_ARG_DOCSTRINGS,
    _GET_EXAMPLE,
    _LIST_ARG_DOCSTRINGS,
    _LIST_EXAMPLE,
    get as get,
    list as list,  # noqa: A004 - claude_comment("claude-opus-4-5", "SDK public API re-export")
)
from anyscale.user_group.models import UserGroup


class UserGroupSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateUserGroupSDK(
            client=client, logger=logger, timer=timer
        )

    @sdk_docs(
        doc_py_example=_LIST_EXAMPLE, arg_docstrings=_LIST_ARG_DOCSTRINGS,
    )
    def list(self, *, max_items: int = 50,) -> List[UserGroup]:  # noqa: F811
        """List user groups in the organization.

        Returns a list of UserGroup objects.
        """
        return self._private_sdk.list(max_items=max_items)

    @sdk_docs(
        doc_py_example=_GET_EXAMPLE, arg_docstrings=_GET_ARG_DOCSTRINGS,
    )
    def get(self, id: str) -> UserGroup:  # noqa: F811, A002
        """Get a specific user group by ID.

        Returns a UserGroup object.
        """
        return self._private_sdk.get(group_id=id)
