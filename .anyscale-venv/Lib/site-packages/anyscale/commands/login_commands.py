import contextlib
import os
import time
import webbrowser

import click

from anyscale.api import ApiClientWrapperInternal
from anyscale.cli_logger import BlockLogger
from anyscale.client import openapi_client
from anyscale.controllers.auth_controller import AuthController
from anyscale.shared_anyscale_utils.conf import ANYSCALE_HOST


log = BlockLogger()


def get_unauthenticated_openapi_client():
    conf = openapi_client.Configuration(host=ANYSCALE_HOST)
    conf.proxy = os.environ.get("https_proxy")
    conf.connection_pool_maxsize = 100
    return openapi_client.DefaultApi(ApiClientWrapperInternal(conf))


@click.command(name="login", help="Log in to Anyscale using a URL.")
@click.option(
    "--no-expire", is_flag=True, default=False, help="Do not expire the token.",
)
@click.option(
    "--expire-in-days",
    type=int,
    default=7,
    help="Expire the token after this many days.",
)
@click.option(
    "--no-browser", is_flag=True, default=False, help="Do not open the browser.",
)
def anyscale_login(no_expire: bool, expire_in_days: int, no_browser: bool) -> None:
    """Log in to Anyscale using a URL
    This is the only unauthenticated API usage in the CLI."""
    if (  # noqa: SIM108 - inline comments explain magic values
        expire_in_days < 0 or no_expire
    ):
        duration_seconds = 0  # no expiration
    else:
        duration_seconds = expire_in_days * 24 * 60 * 60  # seconds

    api_client = get_unauthenticated_openapi_client()

    r = api_client.create_one_time_password_api_v2_users_create_otp_token_get(
        duration_seconds=duration_seconds
    ).result
    token = r.otp

    log.info("Please log in to Anyscale using the following URL:")
    log.info(r.url)

    # Open the URL in the browser. This will work on most platforms.
    # OK to suppress any uncaught exceptions, because the URL will be printed out anyway.
    if not no_browser:
        with contextlib.suppress(Exception):
            try:
                # Only attempt to open browser if it's not a CLI browser (which doesn't support JavaScript)
                if webbrowser.get().basename not in [
                    "www-browser",
                    "elinks",
                    "links",
                    "lynx",
                    "w3m",
                ]:
                    webbrowser.open_new_tab(r.url)
                else:
                    log.info(
                        "Could not open a JavaScript-capable web browser. Open the above URL in your browser manually."
                    )
            except webbrowser.Error:
                log.info(
                    "Could not open a web browser. Open the above URL in your browser manually."
                )

    # give user 3 minutes to log in (3 seconds per attempt)
    for _i in range(60):
        check_result = api_client.check_one_time_password_api_v2_users_request_otp_token_otp_get(
            token
        ).result
        if check_result.valid:
            AuthController().set(check_result.server_session_id)
            log.info("Successfully logged in to Anyscale.")
            if duration_seconds > 0:
                log.info(f"The authentication will expire in {expire_in_days} days.")
            return
        time.sleep(3)

    log.info("Failed to log in to Anyscale. Please try again.")


@click.command(
    name="logout", help="Log out from Anyscale. You CLI token will be revoked."
)
def anyscale_logout() -> None:
    AuthController().remove(ask=False)
    log.info("Successfully logged out from Anyscale.")
