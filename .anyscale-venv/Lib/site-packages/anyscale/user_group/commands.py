from typing import List, Optional

from anyscale._private.sdk import sdk_command
from anyscale.user_group._private.user_group_sdk import PrivateUserGroupSDK
from anyscale.user_group.models import UserGroup


_USER_GROUP_SDK_SINGLETON_KEY = "user_group_sdk"

_LIST_EXAMPLE = """
import anyscale

user_groups = anyscale.user_group.list(max_items=50)
for ug in user_groups:
    print(f"{ug.id}: {ug.name}")
"""

_LIST_ARG_DOCSTRINGS = {"max_items": "Maximum number of user groups to return."}

_GET_EXAMPLE = """
import anyscale

user_group = anyscale.user_group.get(id="ug_abc123")
print(f"{user_group.id}: {user_group.name}")
"""

_GET_ARG_DOCSTRINGS = {"id": "The ID of the user group to retrieve."}


@sdk_command(
    _USER_GROUP_SDK_SINGLETON_KEY,
    PrivateUserGroupSDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_ARG_DOCSTRINGS,
)
def list(  # noqa: A001
    *, max_items: int = 50, _private_sdk: Optional[PrivateUserGroupSDK] = None
) -> List[UserGroup]:
    """List user groups in the organization.

    Returns a list of UserGroup objects.
    """
    return _private_sdk.list(max_items=max_items)  # type: ignore


@sdk_command(
    _USER_GROUP_SDK_SINGLETON_KEY,
    PrivateUserGroupSDK,
    doc_py_example=_GET_EXAMPLE,
    arg_docstrings=_GET_ARG_DOCSTRINGS,
)
def get(
    id: str, *, _private_sdk: Optional[PrivateUserGroupSDK] = None  # noqa: A002
) -> UserGroup:
    """Get a specific user group by ID.

    Returns a UserGroup object.
    """
    return _private_sdk.get(group_id=id)  # type: ignore
