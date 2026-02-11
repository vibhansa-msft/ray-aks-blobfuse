from typing import List, Optional

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_command
from anyscale.user._private.user_sdk import PrivateUserSDK
from anyscale.user.models import AdminCreatedUser, AdminCreateUser, User


_USER_SDK_SINGLETON_KEY = "user_sdk"

_ADMIN_BATCH_CREATE_EXAMPLE = """
import anyscale
from anyscale.user.models import AdminCreateUser

anyscale.user.admin_batch_create(
    [AdminCreateUser(
        name="name",
        email="test@anyscale.com",
        password="",
        is_sso_user=False,
        lastname="lastname",
        title="title",
    ),],
)
"""

_ADMIN_BATCH_CREATE_ARG_DOCSTRINGS = {
    "admin_create_users": "Users to be created by an admin.",
}


@sdk_command(
    _USER_SDK_SINGLETON_KEY,
    PrivateUserSDK,
    doc_py_example=_ADMIN_BATCH_CREATE_EXAMPLE,
    arg_docstrings=_ADMIN_BATCH_CREATE_ARG_DOCSTRINGS,
)
def admin_batch_create(
    admin_create_users: List[AdminCreateUser],
    *,
    _private_sdk: Optional[PrivateUserSDK] = None,
) -> List[AdminCreatedUser]:
    """Batch create, as an admin, users without email verification.
    """
    return _private_sdk.admin_batch_create(admin_create_users)  # type: ignore


_LIST_EXAMPLE = """
import anyscale

for user in anyscale.user.list(max_items=10):
    print(user.email)
"""

_LIST_ARG_DOCSTRINGS = {
    "email": "Filter collaborators by exact email address.",
    "name": "Filter collaborators by exact display name.",
    "collaborator_type": (
        "Filter by collaborator type. Accepts values such as 'all_accounts', "
        "'only_service_accounts', or 'only_user_accounts'."
    ),
    "is_service_account": "If provided, filter collaborators by whether they are service accounts.",
    "max_items": "Maximum total number of users to yield (default: iterate all).",
    "page_size": "Number of users fetched per API request (default: API default).",
}


@sdk_command(
    _USER_SDK_SINGLETON_KEY,
    PrivateUserSDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_ARG_DOCSTRINGS,
)
def list(  # noqa: A001
    *,
    email: Optional[str] = None,
    name: Optional[str] = None,
    collaborator_type: Optional[str] = None,
    is_service_account: Optional[bool] = None,
    max_items: Optional[int] = None,
    page_size: Optional[int] = None,
    _private_sdk: Optional[PrivateUserSDK] = None,
) -> ResultIterator[User]:
    """List collaborators within the organization."""
    return _private_sdk.list(  # type: ignore
        email=email,
        name=name,
        collaborator_type=collaborator_type,
        is_service_account=is_service_account,
        max_items=max_items,
        page_size=page_size,
    )


_GET_EXAMPLE = """
import anyscale

user = anyscale.user.get(email="owner@anyscale.com")
print(user.permission_level)
"""

_GET_ARG_DOCSTRINGS = {
    "email": "Email address of the user to retrieve.",
    "name": "Display name of the user to retrieve.",
    "collaborator_type": (
        "Optional collaborator type constraint when fetching the user."
    ),
    "is_service_account": "Filter by whether the user is a service account.",
}


@sdk_command(
    _USER_SDK_SINGLETON_KEY,
    PrivateUserSDK,
    doc_py_example=_GET_EXAMPLE,
    arg_docstrings=_GET_ARG_DOCSTRINGS,
)
def get(
    *,
    email: Optional[str] = None,
    name: Optional[str] = None,
    collaborator_type: Optional[str] = None,
    is_service_account: Optional[bool] = None,
    _private_sdk: Optional[PrivateUserSDK] = None,
) -> User:
    """Retrieve a single collaborator by email or name."""
    return _private_sdk.get(  # type: ignore
        email=email,
        name=name,
        collaborator_type=collaborator_type,
        is_service_account=is_service_account,
    )
