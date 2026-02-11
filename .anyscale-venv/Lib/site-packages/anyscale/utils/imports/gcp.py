import click


# TODO: type the return value
def try_import_gcp_secretmanager():
    try:
        from google.cloud import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP dependency for secrets manager")
            secretmanager,
        )

        return secretmanager
    except ImportError:
        raise click.ClickException(
            "pip package `google-cloud-secret-manager` is not installed locally on this machine but required "
            "for the command. Please install with `pip install 'anyscale[gcp]'`."
        )


def try_import_gcp_utils():
    try:
        from anyscale.utils import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP dependency for CLI commands")
            gcp_utils,
        )

        return gcp_utils
    except ImportError as e:
        raise click.ClickException(
            f"Import error occurred: {e}. Please install GCP-related packages with `pip install 'anyscale[gcp]'`."
        )


def try_import_gcp_verify_lib():
    try:
        import anyscale.gcp_verification as verify_lib  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP dependency for CLI commands")

        return verify_lib
    except ImportError as e:
        raise click.ClickException(
            f"Import error occurred: {e}. Please install GCP-related packages with `pip install 'anyscale[gcp]'`."
        )


def try_import_gcp_managed_setup_utils():
    try:
        from anyscale.utils import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP dependency for CLI commands")
            gcp_managed_setup_utils,
        )

        return gcp_managed_setup_utils
    except ImportError as e:
        raise click.ClickException(
            f"Import error occurred: {e}. Please install GCP-related packages with `pip install 'anyscale[gcp]'`."
        )


def try_import_gcp_exceptions():
    try:
        from google.api_core.exceptions import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP dependency for CLI commands")
            GoogleAPICallError,
        )
        from googleapiclient.errors import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP dependency for CLI commands")
            HttpError,
        )

        return GoogleAPICallError, HttpError
    except ImportError as e:
        raise click.ClickException(
            f"Import error occurred: {e}. Please install GCP-related packages with `pip install 'anyscale[gcp]'`."
        )
