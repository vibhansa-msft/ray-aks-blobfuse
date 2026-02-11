from typing import List, Optional, Tuple

import click
from rich import print as rprint
import tabulate

import anyscale
from anyscale.cli_logger import BlockLogger
from anyscale.commands import command_examples
from anyscale.commands.util import AnyscaleCommand
from anyscale.resource_quota.models import CreateResourceQuota, ResourceQuota
from anyscale.util import validate_non_negative_arg


log = BlockLogger()  # CLI Logger


@click.group("resource-quota", help="Anyscale resource quota commands.")
def resource_quota_cli() -> None:
    pass


def _format_resource_quotas(resource_quotas: List[ResourceQuota]) -> str:
    table_rows = []
    for resource_quota in resource_quotas:
        table_rows.append(
            [
                resource_quota.id,
                resource_quota.name,
                resource_quota.cloud_id,
                resource_quota.project_id,
                resource_quota.user_id,
                resource_quota.is_enabled,
                resource_quota.created_at.strftime("%m/%d/%Y"),
                resource_quota.deleted_at.strftime("%m/%d/%Y")
                if resource_quota.deleted_at
                else None,
                resource_quota.quota,
            ]
        )
    table = tabulate.tabulate(
        table_rows,
        headers=[
            "ID",
            "NAME",
            "CLOUD ID",
            "PROJECT ID",
            "USER ID",
            "IS ENABLED",
            "CREATED AT",
            "DELETED AT",
            "QUOTA",
        ],
        tablefmt="plain",
    )

    return f"Resource quotas:\n{table}"


@resource_quota_cli.command(
    name="create",
    help="Create a resource quota.",
    cls=AnyscaleCommand,
    is_beta=True,
    example=command_examples.RESOURCE_QUOTAS_CREATE_EXAMPLE,
)
@click.option(
    "-n", "--name", required=True, help="Name of the resource quota to create.",
)
@click.option(
    "--cloud",
    required=True,
    help="Name of the cloud that this resource quota applies to.",
)
@click.option(
    "--project",
    default=None,
    help="Name of the project that this resource quota applies to.",
)
@click.option(
    "--user-email",
    default=None,
    help="Email of the user that this resource quota applies to.",
)
@click.option(
    "--num-cpus",
    required=False,
    help="The quota limit for the number of CPUs.",
    type=int,
)
@click.option(
    "--num-instances",
    required=False,
    help="The quota limit for the number of instances.",
    type=int,
)
@click.option(
    "--num-gpus",
    required=False,
    help="The quota limit for the total number of GPUs.",
    type=int,
)
@click.option(
    "--num-accelerators",
    required=False,
    help="The quota limit for the number of accelerators. Example: --num-accelerators A100-80G 10",
    nargs=2,
    type=(str, int),
    multiple=True,
)
def create(  # noqa: PLR0913
    name: str,
    cloud: str,
    project: Optional[str],
    user_email: Optional[str],
    num_cpus: Optional[int],
    num_instances: Optional[int],
    num_gpus: Optional[int],
    num_accelerators: List[Tuple[str, int]],
) -> None:
    """Creates a resource quota.

    A name and cloud name must be provided.

    `$ anyscale resource-quota create -n my-resource-quota --cloud my-cloud --project my-project --user-email test@myorg.com --num-cpus 10 --num-instances 10 --num-gpus 10 --num-accelerators L4 5 --num-accelerators T4 10`
    """
    create_resource_quota = CreateResourceQuota(
        name=name,
        cloud=cloud,
        project=project,
        user_email=user_email,
        num_cpus=num_cpus,
        num_instances=num_instances,
        num_gpus=num_gpus,
        num_accelerators=dict(num_accelerators),
    )

    try:
        with log.spinner("Creating resource quota..."):
            resource_quota = anyscale.resource_quota.create(create_resource_quota)

        create_resource_quota_message = [f"Name: {name}\nCloud name: {cloud}"]
        if project:
            create_resource_quota_message.append(f"Project name: {project}")
        if user_email:
            create_resource_quota_message.append(f"User email: {user_email}")
        if num_cpus:
            create_resource_quota_message.append(f"Number of CPUs: {num_cpus}")
        if num_instances:
            create_resource_quota_message.append(
                f"Number of instances: {num_instances}"
            )
        if num_gpus:
            create_resource_quota_message.append(f"Number of GPUs: {num_gpus}")
        if num_accelerators:
            create_resource_quota_message.append(
                f"Number of accelerators: {dict(num_accelerators)}"
            )

        log.info("\n".join(create_resource_quota_message))
        log.info(f"Resource quota created successfully ID: {resource_quota.id}")

    except ValueError as e:
        log.error(f"Error creating resource quota: {e}")
        return


@resource_quota_cli.command(
    name="list",
    help="List resource quotas.",
    cls=AnyscaleCommand,
    is_beta=True,
    example=command_examples.RESOURCE_QUOTAS_LIST_EXAMPLE,
)
@click.option(
    "-n", "--name", required=False, help="The name filter for the resource quotas.",
)
@click.option(
    "--cloud", required=False, help="The cloud filter for the resource quotas.",
)
@click.option(
    "--creator-id",
    required=False,
    help="The creator ID filter for the resource quotas.",
)
@click.option(
    "--is-enabled",
    required=False,
    default=None,
    help="The is_enabled filter for the resource quotas.",
    type=bool,
)
@click.option(
    "--max-items",
    required=False,
    default=20,
    type=int,
    help="Max items to show in list.",
    callback=validate_non_negative_arg,
)
def list_resource_quotas(
    name: Optional[str],
    cloud: Optional[str],
    creator_id: Optional[str],
    is_enabled: Optional[bool],
    max_items: int,
) -> None:
    """List resource quotas.

    `$ anyscale resource-quota list -n my-resource-quota --cloud my-cloud`
    """
    resource_quotas = anyscale.resource_quota.list(
        name=name,
        cloud=cloud,
        creator_id=creator_id,
        is_enabled=is_enabled,
        max_items=max_items,
    )

    rprint(_format_resource_quotas(resource_quotas))


@resource_quota_cli.command(
    name="delete",
    help="Delete a resource quota.",
    cls=AnyscaleCommand,
    is_beta=True,
    example=command_examples.RESOURCE_QUOTAS_DELETE_EXAMPLE,
)
@click.option(
    "--id", required=True, help="ID of the resource quota to delete.",
)
def delete(id: str) -> None:  # noqa: A002
    """Deletes a resource quota.

    An ID of resource quota must be provided.

    `$ anyscale resource-quota delete --id rsq_123`
    """
    try:
        with log.spinner("Deleting resource quota..."):
            anyscale.resource_quota.delete(resource_quota_id=id)
    except ValueError as e:
        log.error(f"Error deleting resource quota: {e}")
        return

    log.info(f"Resource quota with ID {id} deleted successfully.")


@resource_quota_cli.command(
    name="enable",
    help="Enable a resource quota.",
    cls=AnyscaleCommand,
    is_beta=True,
    example=command_examples.RESOURCE_QUOTAS_ENABLE_EXAMPLE,
)
@click.option(
    "--id", required=True, help="ID of the resource quota to enable.",
)
def enable(id: str) -> None:  # noqa: A002
    """Enables a resource quota.

    An ID of resource quota must be provided.

    `$ anyscale resource-quota enable --id rsq_123`
    """
    try:
        with log.spinner("Setting resource quota status..."):
            anyscale.resource_quota.enable(resource_quota_id=id)
    except ValueError as e:
        log.error(f"Error enabling resource quota: {e}")
        return

    log.info(f"Enabled resource quota with ID {id} successfully.")


@resource_quota_cli.command(
    name="disable",
    help="Disable a resource quota.",
    cls=AnyscaleCommand,
    is_beta=True,
    example=command_examples.RESOURCE_QUOTAS_DISABLE_EXAMPLE,
)
@click.option(
    "--id", required=True, help="ID of the resource quota to disable.",
)
def disable(id: str) -> None:  # noqa: A002
    """Disables a resource quota.

    An ID of resource quota must be provided.

    `$ anyscale resource-quota disable --id rsq_123`
    """
    try:
        with log.spinner("Setting resource quota status..."):
            anyscale.resource_quota.disable(resource_quota_id=id)
    except ValueError as e:
        log.error(f"Error disabling resource quota: {e}")
        return

    log.info(f"Disabled resource quota with ID {id} successfully.")
