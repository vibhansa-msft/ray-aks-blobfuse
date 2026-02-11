import json
from json import dumps as json_dumps
from typing import Dict, Optional

import click
from rich.console import Console
from rich.table import Table

import anyscale
from anyscale._private.anyscale_client import AnyscaleClient
from anyscale.cli_logger import BlockLogger
from anyscale.commands import command_examples
from anyscale.commands.list_util import (
    display_list,
    MAX_PAGE_SIZE,
    NON_INTERACTIVE_DEFAULT_MAX_ITEMS,
    validate_page_size,
)
from anyscale.commands.util import AnyscaleCommand
from anyscale.user.models import AdminCreateUser, AdminCreateUsers, User
from anyscale.util import AnyscaleJSONEncoder, get_endpoint, validate_non_negative_arg


log = BlockLogger()  # CLI Logger


_COLLABORATOR_TYPE_CHOICES = (
    "all_accounts",
    "only_service_accounts",
    "only_user_accounts",
)


def _create_user_list_table(show_header: bool) -> Table:
    table = Table(show_header=show_header, expand=True)
    table.add_column("EMAIL", no_wrap=False, overflow="fold", ratio=3, min_width=20)
    table.add_column("NAME", no_wrap=False, overflow="fold", ratio=2, min_width=12)
    table.add_column("ID", no_wrap=False, overflow="fold", ratio=2, min_width=12)
    table.add_column("ROLE", no_wrap=False, overflow="fold", ratio=1, min_width=10)
    table.add_column(
        "CREATED AT", no_wrap=False, overflow="fold", ratio=1, min_width=14
    )
    return table


def _format_user_output_data(user: User) -> Dict[str, str]:
    return {
        "email": user.email,
        "name": user.name,
        "id": user.user_id or "",
        "role": user.permission_level,
        "created_at": user.created_at.strftime("%Y-%m-%d %H:%M"),
    }


@click.group("user", help="Manage users.")
def user_cli() -> None:
    pass


@user_cli.command(
    name="batch-create",
    cls=AnyscaleCommand,
    example=command_examples.USER_BATCH_CREATE_EXAMPLE,
)
@click.option(
    "--users-file",
    "-f",
    required=True,
    type=str,
    help="Path to a YAML file that contains the information for user accounts to be created.",
)
def admin_batch_create(users_file: str,) -> None:
    """
    Batch create, as an admin, users without email verification.
    """
    log.info("Creating users...")

    create_users = AdminCreateUsers.from_yaml(users_file)

    try:
        created_users = anyscale.user.admin_batch_create(
            admin_create_users=[
                AdminCreateUser(**create_user)
                for create_user in create_users.create_users
            ]
        )
    except ValueError as e:
        log.error(f"Error creating users: {e}")
        return

    log.info(f"{len(created_users)} users created.")


@user_cli.command(
    name="list",
    cls=AnyscaleCommand,
    example=command_examples.USER_LIST_EXAMPLE,
    help="List users within your organization.",
)
@click.option("--email", type=str, help="Filter users by email.")
@click.option("--name", type=str, help="Filter users by display name.")
@click.option(
    "--collaborator-type",
    type=click.Choice(_COLLABORATOR_TYPE_CHOICES, case_sensitive=False),
    help="Filter users by collaborator type.",
)
@click.option(
    "--service-account",
    "service_account",
    flag_value=True,
    help="Only show service accounts.",
)
@click.option(
    "--user-account",
    "service_account",
    flag_value=False,
    help="Only show user accounts (non-service accounts).",
)
@click.option(
    "--max-items",
    type=int,
    default=None,
    callback=validate_non_negative_arg,
    help="Maximum number of users to display when running non-interactively.",
)
@click.option(
    "--page-size",
    type=int,
    default=MAX_PAGE_SIZE,
    show_default=True,
    callback=validate_page_size,
    help=f"Items per page (max {MAX_PAGE_SIZE}).",
)
@click.option(
    "--json/--no-json",
    "json_output",
    default=False,
    help="Render output as JSON instead of a table.",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Enable or disable interactive pagination.",
)
def list_users(  # noqa: A001, PLR0913
    email: Optional[str],
    name: Optional[str],
    collaborator_type: Optional[str],
    service_account: Optional[bool],
    max_items: Optional[int],
    page_size: int,
    json_output: bool,
    interactive: bool,
) -> None:
    """List users within the organization."""
    if max_items is not None and interactive:
        raise click.UsageError("--max-items only allowed with --no-interactive")

    effective_max = max_items
    stderr = Console(stderr=True)
    if not interactive and effective_max is None:
        stderr.print(
            f"Defaulting to {NON_INTERACTIVE_DEFAULT_MAX_ITEMS} items in batch mode; "
            "use --max-items to override."
        )
        effective_max = NON_INTERACTIVE_DEFAULT_MAX_ITEMS

    console = Console()

    stderr.print("[bold]Listing users with:[/]")
    stderr.print(f"• email             = {email or '<any>'}")
    stderr.print(f"• name              = {name or '<any>'}")
    stderr.print(f"• collaborator type = {collaborator_type or '<any>'}")
    stderr.print(
        f"• service account   = "
        f"{service_account if service_account is not None else '<any>'}"
    )
    stderr.print(f"• mode              = {'interactive' if interactive else 'batch'}")
    stderr.print(f"• per-page limit    = {page_size}")
    stderr.print(f"• max-items total   = {effective_max or 'all'}")
    stderr.print(f"\nView your Users in the UI at {get_endpoint('/collaborators')}\n")

    collaborator_type_value = collaborator_type.lower() if collaborator_type else None

    formatter = (
        (lambda user: user.to_dict()) if json_output else _format_user_output_data
    )

    try:
        iterator = anyscale.user.list(
            email=email,
            name=name,
            collaborator_type=collaborator_type_value,
            is_service_account=service_account,
            max_items=None if interactive else effective_max,
            page_size=page_size,
        )

        total = display_list(
            iterator=iter(iterator),
            item_formatter=formatter,
            table_creator=_create_user_list_table,
            json_output=json_output,
            page_size=page_size,
            interactive=interactive,
            max_items=effective_max,
            console=console,
        )

        if not json_output:
            if total > 0:
                stderr.print(f"\nFetched {total} users.")
            else:
                stderr.print("\nNo users found.")
    except ValueError as exc:
        raise click.ClickException(str(exc))
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"Failed to list users: {exc}")


@user_cli.command(
    name="get",
    cls=AnyscaleCommand,
    example=command_examples.USER_GET_EXAMPLE,
    help="Get details for a single user in your organization.",
)
@click.option("--email", type=str, help="Email address of the user.")
@click.option("--name", type=str, help="Display name of the user.")
@click.option(
    "--collaborator-type",
    type=click.Choice(_COLLABORATOR_TYPE_CHOICES, case_sensitive=False),
    help="Optional collaborator type constraint.",
)
@click.option(
    "--service-account",
    "service_account",
    flag_value=True,
    help="Restrict to service accounts.",
)
@click.option(
    "--user-account",
    "service_account",
    flag_value=False,
    help="Restrict to individual user accounts.",
)
@click.option(
    "--json/--no-json",
    "json_output",
    default=False,
    help="Output JSON instead of YAML.",
)
def get_user(  # noqa: A001
    email: Optional[str],
    name: Optional[str],
    collaborator_type: Optional[str],
    service_account: Optional[bool],
    json_output: bool,
) -> None:
    """Retrieve details for a single user by email or ID."""
    if email is None and name is None:
        raise click.UsageError("Provide --email or --name.")

    collaborator_type_value = collaborator_type.lower() if collaborator_type else None

    console = Console()
    stderr = Console(stderr=True)

    stderr.print("[bold]Fetching user with:[/]")
    stderr.print(f"• email             = {email or '<not specified>'}")
    stderr.print(f"• name              = {name or '<not specified>'}")
    stderr.print(f"• collaborator type = {collaborator_type or '<any>'}")
    stderr.print(
        f"• service account   = "
        f"{service_account if service_account is not None else '<any>'}"
    )
    stderr.print(f"\nView your Users in the UI at {get_endpoint('/collaborators')}\n")

    try:
        user = anyscale.user.get(
            email=email,
            name=name,
            collaborator_type=collaborator_type_value,
            is_service_account=service_account,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc))
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"Failed to get user: {exc}")

    if json_output:
        console.print_json(
            json=json_dumps(user.to_dict(), indent=2, cls=AnyscaleJSONEncoder)
        )
        return

    table = _create_user_list_table(show_header=True)
    formatted_data = _format_user_output_data(user)
    table.add_row(
        formatted_data["email"],
        formatted_data["name"],
        formatted_data["id"],
        formatted_data["role"],
        formatted_data["created_at"],
    )
    console.print(table)
    stderr.print("\nFetched 1 user.")


@user_cli.command(
    name="list-permissions",
    cls=AnyscaleCommand,
    example=command_examples.USER_LIST_PERMISSIONS_EXAMPLE,
)
@click.option(
    "--user-id",
    type=str,
    default=None,
    help="Filter to a specific user ID. If not provided, lists permissions for all users.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Write JSON output to a file instead of stdout.",
)
def list_permissions(user_id: Optional[str], output: Optional[str]) -> None:
    """
    List users and their effective cloud/project permissions across the organization.

    Shows each user's access to clouds and projects, combining both individually
    granted permissions and permissions inherited from user groups. Also lists
    the current organization owners.

    Output is JSON. Use --output to save to a file.
    """
    client = AnyscaleClient()
    try:
        log.info("Listing users and their effective permissions...")
        response = client.list_scim_user_permissions(user_id=user_id)
        result = response.get("result", response)

        # Derive per-user org role from org_owners:
        json_output = json.dumps(result, indent=2, sort_keys=True)

        if output:
            with open(output, "w") as f:
                f.write(json_output)
            log.info(f"Results written to {output}")
        else:
            print(json_output)
    except click.ClickException:
        raise
    except Exception as e:  # noqa: BLE001
        log.error(f"Failed to list user permissions: {e}")
        raise click.ClickException(str(e))
