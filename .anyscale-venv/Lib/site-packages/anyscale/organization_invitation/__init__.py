from typing import List, Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.organization_invitation._private.organization_invitation_sdk import (
    PrivateOrganizationInvitationSDK,
)
from anyscale.organization_invitation.commands import (
    _CREATE_ARG_DOCSTRINGS,
    _CREATE_EXAMPLE,
    _DELETE_ARG_DOCSTRINGS,
    _DELETE_EXAMPLE,
    _LIST_ARG_DOCSTRINGS,
    _LIST_EXAMPLE,
    create as create,
    delete as delete,
    list as list,  # noqa: A004 - claude_comment("claude-opus-4-5", "SDK public API re-export")
)


class OrganizationInvitationSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateOrganizationInvitationSDK(
            client=client, logger=logger, timer=timer
        )

    @sdk_docs(
        doc_py_example=_CREATE_EXAMPLE, arg_docstrings=_CREATE_ARG_DOCSTRINGS,
    )
    def create(  # noqa: F811
        self, emails: List[str],
    ):
        """Creates organization invitations for the provided emails
        """
        return self._private_sdk.create(emails=emails)

    @sdk_docs(
        doc_py_example=_LIST_EXAMPLE, arg_docstrings=_LIST_ARG_DOCSTRINGS,
    )
    def list(self):  # noqa: F811
        """Lists organization invitations
        """
        return self._private_sdk.list()

    @sdk_docs(
        doc_py_example=_DELETE_EXAMPLE, arg_docstrings=_DELETE_ARG_DOCSTRINGS,
    )
    def delete(  # noqa: F811
        self, email: str,
    ):
        """Deletes an organization invitation
        """
        return self._private_sdk.delete(email=email)
