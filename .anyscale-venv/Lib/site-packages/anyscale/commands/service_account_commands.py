from typing import List, Optional

import click
from rich import print as rprint
import tabulate

import anyscale
from anyscale.cli_logger import BlockLogger
from anyscale.service_account.models import ServiceAccount
from anyscale.util import validate_non_negative_arg


DEFAULT_OVERFLOW = "fold"
DEFAULT_COL_WIDTH = 36


log = BlockLogger()  # CLI Logger


@click.group(
    "service-account",
    short_help="Manage service accounts for your anyscale workloads.",
)
def service_account_cli() -> None:
    pass


def _print_new_api_key(api_key: str):
    log.warning(
        "The following API token for the service account will only appear once:",
    )
    log.info(api_key)


def _print_service_account_table(service_accounts: List[ServiceAccount]):
    table_rows = []
    for service_account in service_accounts:
        table_rows.append(
            [
                service_account.name,
                service_account.created_at.strftime("%m/%d/%Y"),
                service_account.permission_level,
                service_account.email,
            ]
        )
    table = tabulate.tabulate(
        table_rows,
        headers=["NAME", "CREATED AT", "ORGANIZATION PERMISSION LEVEL", "EMAIL",],
        tablefmt="plain",
    )

    rprint(f"Service accounts:\n{table}")


@service_account_cli.command(name="create", help="Create a service account.")
@click.option(
    "--name", "-n", help="Name for the service account.", type=str, required=True
)
def create(name: str) -> None:
    try:
        api_key = anyscale.service_account.create(name)

        log.info(f"Service account {name} created successfully.")
        _print_new_api_key(api_key)
    except ValueError as e:
        log.error(f"Error creating service account: {e}")


@service_account_cli.command(
    name="create-api-key", help="Create a new API key for a service account."
)
@click.option(
    "--email", help="Email of the service account to create the new key for.", type=str
)
@click.option(
    "--name", help="Name of the service account to create the new key for.", type=str
)
def create_api_key(email: Optional[str], name: Optional[str]) -> None:
    try:
        api_key = anyscale.service_account.create_api_key(email, name)
        _print_new_api_key(api_key)
    except ValueError as e:
        log.error(f"Error creating API key: {e}")


@service_account_cli.command(name="list", help="List service accounts.")
@click.option(
    "--max-items",
    required=False,
    default=20,
    type=int,
    help="Max items to show in list.",
    callback=validate_non_negative_arg,
)
def list_service_accounts(max_items: int) -> None:
    service_accounts = anyscale.service_account.list(max_items)
    _print_service_account_table(service_accounts)


@service_account_cli.command(name="delete", help="Delete a service account.")
@click.option("--email", help="Email of the service account to delete.", type=str)
@click.option("--name", help="Name of the service account to delete.", type=str)
def delete(email: Optional[str], name: Optional[str]) -> None:
    try:
        anyscale.service_account.delete(email, name)
        log.info(f"Service account {email or name} deleted successfully.")
    except ValueError as e:
        log.error(f"Error deleting service account: {e}")
