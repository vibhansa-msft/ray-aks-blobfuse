from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from anyscale._private.models import ModelBase


@dataclass(frozen=True)
class AdminCreateUser(ModelBase):
    """User to be created by an admin.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.user.models import AdminCreateUser

admin_create_user = AdminCreateUser(
    # First name of the user to be created.
    name="name",
    # Email of the user to be created.
    email="test@anyscale.com",
    # Password for the user account being created.
    password="",
    # Whether the user is an SSO user. SSO users can log in using SSO.
    is_sso_user=False,
    # Optional last name of the user to be created.
    lastname="lastname",
    # Optional title of the user to be created.
    title="title",
)
"""
    name: str = field(metadata={"docstring": "First name of the user to be created."})

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

    email: str = field(metadata={"docstring": "Email of the user to be created."})

    def _validate_email(self, email: str):
        if not isinstance(email, str):
            raise TypeError("email must be a string.")

    password: Optional[str] = field(
        metadata={"docstring": "Password for the user account being created."}
    )

    def _validate_password(self, password: Optional[str]):
        if password is not None and not isinstance(password, str):
            raise TypeError("password must be a string.")

    is_sso_user: bool = field(
        metadata={
            "docstring": "Whether the user is an SSO user. SSO users can log in using SSO."
        },
    )

    def _validate_is_sso_user(self, is_sso_user: bool):
        if not isinstance(is_sso_user, bool):
            raise TypeError("is_sso_user must be a boolean.")

    lastname: Optional[str] = field(
        default=None,
        metadata={"docstring": "Optional last name of the user to be created."},
    )

    def _validate_lastname(self, lastname: Optional[str]):
        if lastname is not None and not isinstance(lastname, str):
            raise TypeError("lastname must be a string.")

    title: Optional[str] = field(
        default=None,
        metadata={"docstring": "Optional title of the user to be created."},
    )

    def _validate_title(self, title: Optional[str]):
        if title is not None and not isinstance(title, str):
            raise TypeError("title must be a string.")


@dataclass(frozen=True)
class AdminCreateUsers(ModelBase):
    """Users to be created by an admin.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.user.models import AdminCreateUser

admin_create_user = AdminCreateUser(
    # First name of the user to be created.
    name="name",
    # Email of the user to be created.
    email="test@anyscale.com",
    # Password for the user account being created.
    password="",
    # Whether the user is an SSO user. SSO users can log in using SSO.
    is_sso_user=False,
    # Optional last name of the user to be created.
    lastname="lastname",
    # Optional title of the user to be created.
    title="title",
)
admin_create_users = AdminCreateUsers(
    # Users to be created by an admin.
    create_users=[admin_create_user]
)
"""
    create_users: List[Dict[str, Any]] = field(
        metadata={"docstring": "Users to be created by an admin."}
    )

    def _validate_create_users(self, create_users: List[Dict[str, Any]]):
        if not isinstance(create_users, list):
            raise TypeError("create_users must be a list.")


@dataclass(frozen=True)
class AdminCreatedUser(ModelBase):
    """User account created by an admin that has organization collaborator permissions.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.user.models import AdminCreatedUser

admin_create_user = AdminCreateUser(
    # First name of the user to be created.
    name="name",
    # Email of the user to be created.
    email="test@anyscale.com",
    # Password for the user account being created.
    password="",
    # Whether the user is an SSO user. SSO users can log in using SSO.
    is_sso_user=False,
    # Optional last name of the user to be created.
    lastname="lastname",
    # Optional title of the user to be created.
    title="title",
)
admin_created_users: List[AdminCreatedUser] = anyscale.user.admin_batch_create([admin_create_user])
"""
    user_id: str = field(
        metadata={"docstring": "ID of the user that has been created."}
    )

    def _validate_user_id(self, user_id: str):
        if not isinstance(user_id, str):
            raise TypeError("user_id must be a string.")

    name: str = field(
        metadata={"docstring": "First name of the user that has been created."}
    )

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

    email: str = field(
        metadata={"docstring": "Email of the user that has been created."}
    )

    def _validate_email(self, email: str):
        if not isinstance(email, str):
            raise TypeError("email must be a string.")

    created_at: datetime = field(
        metadata={"docstring": "The timestamp of when the user is created."}
    )

    def _validate_created_at(self, created_at: datetime):
        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime.")

    is_sso_user: bool = field(
        metadata={
            "docstring": "Whether the user is an SSO user. SSO users can log in using SSO."
        },
    )

    def _validate_is_sso_user(self, is_sso_user: bool):
        if not isinstance(is_sso_user, bool):
            raise TypeError("is_sso_user must be a boolean.")

    lastname: Optional[str] = field(
        default=None,
        metadata={"docstring": "Optional last name of the user that has been created."},
    )

    def _validate_lastname(self, lastname: Optional[str]):
        if lastname is not None and not isinstance(lastname, str):
            raise TypeError("lastname must be a string.")

    title: Optional[str] = field(
        default=None,
        metadata={"docstring": "Optional title of the user that has been created."},
    )

    def _validate_title(self, title: Optional[str]):
        if title is not None and not isinstance(title, str):
            raise TypeError("title must be a string.")


@dataclass(frozen=True)
class User(ModelBase):
    """Collaborator returned by ``anyscale.user`` APIs."""

    __doc_py_example__ = """\
import anyscale

for user in anyscale.user.list(max_items=5):
    print(f"{user.email} ({user.permission_level})")
"""

    email: str = field(
        metadata={"docstring": "Email address associated with the collaborator."}
    )

    def _validate_email(self, email: str):
        if not isinstance(email, str):
            raise TypeError("email must be a string.")

    name: str = field(metadata={"docstring": "Display name of the collaborator."})

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

    created_at: datetime = field(
        metadata={"docstring": "Timestamp for when the collaborator was created."}
    )

    def _validate_created_at(self, created_at: datetime):
        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime.")

    permission_level: str = field(
        metadata={"docstring": "Organization permission level for the collaborator."}
    )

    def _validate_permission_level(self, permission_level: str):
        if not isinstance(permission_level, str):
            raise TypeError("permission_level must be a string.")

    user_id: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Optional user ID backing the collaborator (may be absent for service accounts)."
        },
    )

    def _validate_user_id(self, user_id: Optional[str]):
        if user_id is not None and not isinstance(user_id, str):
            raise TypeError("user_id must be a string.")
