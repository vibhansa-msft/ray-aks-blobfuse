from typing import Dict, List, Optional, Tuple

from anyscale._private.sdk import sdk_command
from anyscale.organization_invitation._private.organization_invitation_sdk import (
    PrivateOrganizationInvitationSDK,
)
from anyscale.organization_invitation.models import OrganizationInvitation


_ORGANIZATION_INVITATION_SDK_SINGLETON_KEY = "organization_invitation_sdk"

_CREATE_EXAMPLE = """
import anyscale

anyscale.organization_invitation.create(emails=["test1@anyscale.com","test2@anyscale.com"])
"""

_CREATE_ARG_DOCSTRINGS = {
    "emails": "The emails to send the organization invitations to."
}

_LIST_EXAMPLE = """
import anyscale

anyscale.organization_invitation.list()
"""

_LIST_ARG_DOCSTRINGS: Dict[str, str] = {}

_DELETE_EXAMPLE = """
import anyscale

anyscale.organization_invitation.delete(email="test@anyscale.com")
"""

_DELETE_ARG_DOCSTRINGS = {
    "email": "The email of the organization invitation to delete."
}


@sdk_command(
    _ORGANIZATION_INVITATION_SDK_SINGLETON_KEY,
    PrivateOrganizationInvitationSDK,
    doc_py_example=_CREATE_EXAMPLE,
    arg_docstrings=_CREATE_ARG_DOCSTRINGS,
)
def create(
    emails: List[str],
    *,
    _private_sdk: Optional[PrivateOrganizationInvitationSDK] = None
) -> Tuple[List[str], List[str]]:
    """Creates organization invitations for the provided emails.

    Returns a tuple of successful emails and error messages.
    """
    return _private_sdk.create(emails=emails)  # type: ignore


@sdk_command(
    _ORGANIZATION_INVITATION_SDK_SINGLETON_KEY,
    PrivateOrganizationInvitationSDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_ARG_DOCSTRINGS,
)
def list(  # noqa: A001
    *, _private_sdk: Optional[PrivateOrganizationInvitationSDK] = None
) -> List[OrganizationInvitation]:
    """Lists organization invitations.

    Returns a list of organization invitations.
    """
    return _private_sdk.list()  # type: ignore


@sdk_command(
    _ORGANIZATION_INVITATION_SDK_SINGLETON_KEY,
    PrivateOrganizationInvitationSDK,
    doc_py_example=_DELETE_EXAMPLE,
    arg_docstrings=_DELETE_ARG_DOCSTRINGS,
)
def delete(
    email: str, *, _private_sdk: Optional[PrivateOrganizationInvitationSDK] = None
) -> str:
    """Deletes an organization invitation.

    Returns the email of the deleted organization invitation.
    """
    return _private_sdk.delete(email)  # type: ignore
