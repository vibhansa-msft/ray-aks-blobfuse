import click
from dateutil import tz
from rich import print as rprint
import tabulate

import anyscale
from anyscale.cli_logger import BlockLogger
from anyscale.commands import command_examples
from anyscale.commands.util import AnyscaleCommand


log = BlockLogger()  # CLI Logger


@click.group("organization-invitation", help="Manage organization invitations.")
def organization_invitation_cli() -> None:
    pass


@organization_invitation_cli.command(
    name="create",
    cls=AnyscaleCommand,
    example=command_examples.ORGANIZATION_INVITATION_CREATE_EXAMPLE,
)
@click.option(
    "--emails",
    required=True,
    type=str,
    help="The emails to send the organization invitations to. Delimited by commas.",
)
def create(emails: str,) -> None:
    """
    Creates organization invitations for the provided emails.
    """
    log.info("Creating organization invitations...")

    success_emails, error_messages = anyscale.organization_invitation.create(
        emails.split(",")
    )

    if success_emails:
        log.info(f"Organization invitations sent to: {', '.join(success_emails)}")

    if error_messages:
        for error_message in error_messages:
            log.error(
                f"Failed to send organization invitations with the following errors: {error_message}"
            )


@organization_invitation_cli.command(
    name="list",
    cls=AnyscaleCommand,
    example=command_examples.ORGANIZATION_INVITATION_LIST_EXAMPLE,
)
def list() -> None:  # noqa: A001
    """
    Lists organization invitations.
    """
    organization_invitations = anyscale.organization_invitation.list()

    table = tabulate.tabulate(
        [
            (
                i.id,
                i.email,
                i.created_at.astimezone(tz=tz.tzlocal()).strftime("%m/%d/%Y %I:%M %p"),
                i.expires_at.astimezone(tz=tz.tzlocal()).strftime("%m/%d/%Y %I:%M %p"),
            )
            for i in organization_invitations
        ],
        headers=["ID", "Email", "Created At", "Expires At"],
    )
    rprint(table)


@organization_invitation_cli.command(
    name="delete",
    cls=AnyscaleCommand,
    example=command_examples.ORGANIZATION_INVITATION_DELETE_EXAMPLE,
)
@click.option(
    "--email",
    required=True,
    type=str,
    help="The email of the organization invitation to delete.",
)
def delete(email: str,) -> None:
    """
    Deletes an organization invitation.
    """
    try:
        organization_invitation_email = anyscale.organization_invitation.delete(email)
    except ValueError as e:
        log.error(f"Failed to delete organization invitation: {e}")
        return

    log.info(f"Organization invitation for {organization_invitation_email} deleted.")
