from typing import Optional

import click

from anyscale.cli_logger import BlockLogger
from anyscale.controllers.experimental_integrations_controller import (
    ExperimentalIntegrationsController,
)


log = BlockLogger()  # CLI Logger


@click.group(
    "integration", help="Interact with external integrations on Anyscale.", hidden=True
)
def experimental_integrations_cli() -> None:
    pass


@experimental_integrations_cli.group(
    "wandb", help="Interact with Anyscale W&B integration."
)
def experimental_wandb_cli() -> None:
    pass


@experimental_wandb_cli.command(
    name="enable",
    help="[DEPRECATED] This command is no longer needed to enable the product W&B integration.",
)
@click.option(
    "--cloud-id",
    "-id",
    required=False,
    default=None,
    help="ID of the cloud to enable the W&B integration with.",
)
@click.option(
    "--cloud-name",
    "-n",
    required=False,
    default=None,
    help="Name of the cloud to enable the W&B integration with.",
)
def enable_wandb_integration(
    cloud_id: Optional[str], cloud_name: Optional[str]
) -> None:
    log.warning(
        "`anyscale integration wandb enable` has been deprecated. The product W&B integration "
        "is enabled by default so this command is no longer necessary. We will remove this command "
        "at the end of Q1 2023."
    )

    if not cloud_id and not cloud_name:
        raise click.ClickException(
            "Neither --cloud-id nor --cloud-name were provided. Please specify the "
            "cloud for which to enable the integration through one of these arguments."
        )

    if cloud_id and cloud_name:
        raise click.ClickException(
            "Both --cloud-id and --cloud-name were provided. Please specify the "
            "cloud for which to enable the integration through only one of these arguments."
        )

    integrations_controller = ExperimentalIntegrationsController()
    integrations_controller.enable_wandb_integration(
        cloud_id=cloud_id, cloud_name=cloud_name
    )
