import json
import os
from typing import Optional

import click

from anyscale.authenticate import (
    CREDENTIALS_DIRS_PERMISSIONS,
    CREDENTIALS_ENVVAR,
    CREDENTIALS_FILE,
    CREDENTIALS_FILE_PERMISSIONS,
)
from anyscale.cli_logger import LogsLogger
from anyscale.client.openapi_client.models.user_info import UserInfo
from anyscale.util import credentials_check_sanity, get_endpoint, get_user_info


class AuthController:
    """The controller for auth CLI command. This does not inherit BaseController
    since it does not involve the actual authentication and interaction with the API.
    """

    def __init__(
        self,
        warn_envvar: bool = True,
        path: str = CREDENTIALS_FILE,
        log_output: bool = True,
    ):
        self.log = LogsLogger(log_output=log_output)
        self.url = get_endpoint("/v2/api-keys")
        self.path = os.path.expanduser(path)
        if warn_envvar:
            self._warn_environment_variable()

    def set(self, token: Optional[str] = None) -> None:
        # if token is provided, skip user interaction
        if token is None:
            # check if the file already exists
            if os.path.exists(self.path) and not click.confirm(
                f"You already have credentials saved in {CREDENTIALS_FILE}. Do you want to overwrite?"
            ):
                self.log.log("Canceled")
                return

            # get token
            self.log.log(
                f"Please copy your credentials from {self.url} and paste them below."
            )
            token = click.prompt("Enter your credentials", hide_input=True).strip()

        # sanity check
        if not credentials_check_sanity(token):
            raise click.ClickException("Invalid credentials")

        # save and done
        self._save_credentials(token)

    def show(self) -> None:
        result = get_user_info()
        block_label = "Identifying the user"
        self.log.open_block(block_label)
        if result is None:
            self.log.info("Not authenticated", block_label=block_label)
        else:
            self.log.info("Successfully authenticated as:", block_label=block_label)
            self.log.info(
                self._auth_info_to_formatted_string(result), block_label=block_label
            )

        self.log.close_block()

    def fix(self) -> None:
        if not os.path.exists(self.path):
            raise click.ClickException(
                f"Failed to fix the permissions of {CREDENTIALS_FILE}. "
                "Please check if you can access the file."
            )
        os.chmod(self.path, CREDENTIALS_FILE_PERMISSIONS)
        self.log.log(f"Successfully fixed the permissions of {CREDENTIALS_FILE}")

    def remove(self, ask=True) -> None:
        # check if the file exists
        if not os.path.exists(self.path):
            raise click.ClickException("Credentials file does not exist.")

        if ask and not click.confirm(
            f"Are you sure you want to remove {CREDENTIALS_FILE}?"
        ):
            self.log.log("Canceled")
            return

        os.remove(self.path)
        self.log.log(f"Removed the credentials from {CREDENTIALS_FILE}.")

    def _auth_info_to_formatted_string(self, userinfo: UserInfo) -> str:
        """
        Try to construct a formatted output str from the retrieved user info.
        If exception happens, just stringfy and return.
        """
        left_pad = " " * 2

        try:
            s = f"{left_pad}{'user name:': <30}{userinfo.name}\n"
            s += f"{left_pad}{'user email:': <30}{userinfo.email}\n"
            s += f"{left_pad}{'user id:': <30}{userinfo.id}\n"

            # NOTE: Currently, we only support 1 organization per user, thus we're flattening the structure.
            # However, this will require change in the future as it can be confusing
            for org in userinfo.organizations:
                s += f"{left_pad}{'organization name:': <30}{org.name}\n"
                s += f"{left_pad}{'organization id:': <30}{org.id}\n"
            s += f"{left_pad}{'organization role:': <30}{userinfo.organization_permission_level}"
            return s
        except AttributeError:
            return userinfo.to_str()

    def _save_credentials(self, token: str) -> None:
        dirname = os.path.dirname(self.path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, CREDENTIALS_DIRS_PERMISSIONS)
        with open(self.path, "w", CREDENTIALS_FILE_PERMISSIONS) as f:
            f.write(self._auth_token_get_json_string(token))
            f.flush()
        os.chmod(self.path, CREDENTIALS_FILE_PERMISSIONS)
        self.log.log(f"Your credentials have been saved to {CREDENTIALS_FILE}")

    def _warn_environment_variable(self) -> None:
        if os.environ.get(CREDENTIALS_ENVVAR) is not None:
            self.log.warning(
                f"You have {CREDENTIALS_ENVVAR} variable set, which will override the credentials file"
            )

    def _auth_token_get_json_string(self, credentials_str: str) -> str:
        return json.dumps({"cli_token": credentials_str})
