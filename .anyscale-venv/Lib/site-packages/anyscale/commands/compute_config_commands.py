from io import StringIO
import json
from typing import IO, Optional, Tuple

import click
import tabulate
import yaml

import anyscale
from anyscale.cli_logger import BlockLogger
from anyscale.cloud_utils import get_cloud_id_and_name
from anyscale.commands import command_examples
from anyscale.commands.util import AnyscaleCommand
from anyscale.compute_config.models import (
    compute_config_type_from_yaml,
    ComputeConfigVersion,
)
from anyscale.controllers.base_controller import BaseController
from anyscale.controllers.compute_config_controller import ComputeConfigController
from anyscale.util import get_endpoint, validate_non_negative_arg


logger = BlockLogger()


def _validate_name_and_id_args(
    *, positional_name: Optional[str], flag_name: Optional[str], id_flag: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Handles validation and deduplication for the name & ID options.

    The positional name is deprecated and will be removed in the near future.

    Returns (name, ID) -- exactly one of them will be populated.
    """
    if flag_name and positional_name:
        raise click.ClickException(
            "Both -n/--name and [COMPUTE_CONFIG_NAME] were provided. "
            "Use -n/--name only (the positional argument is deprecated)."
        )
    elif positional_name:
        logger.warning(
            "The [COMPUTE_CONFIG_NAME] positional argument is deprecated. "
            "Use the -n/--name flag instead."
        )
        name = positional_name
    elif flag_name:
        name = flag_name
    else:
        name = None

    if name and id_flag:
        raise click.ClickException("Only one of name or ID can be provided.")

    if not name and not id_flag:
        raise click.ClickException(
            "Either -n/--name or --id/--compute-config-id must be provided."
        )

    return name, id_flag


@click.group(
    "compute-config", help="Manage compute configurations.",
)
def compute_config_cli() -> None:
    pass


@compute_config_cli.command(
    name="create",
    help=(
        "Create a new version of a compute config from a YAML file.\n\n"
        "(1) To use the **new schema** defined at "
        "https://docs.anyscale.com/reference/compute-config-api#computeconfig, "
        "use the -f/--config-file flag:\n\n"
        "`anyscale compute-config create -f new_schema_config.yaml`\n\n"
        "(2) To use the **old schema** defined at "
        "https://docs.anyscale.com/reference/compute-config-api/#createclustercomputeconfig-legacy, "
        "use the positional argument:\n\n"
        "`anyscale compute-config create old_schema_config.yaml`\n\n"
    ),
    cls=AnyscaleCommand,
    example=command_examples.COMPUTE_CONFIG_CREATE_EXAMPLE,
)
@click.argument("compute-config-file", type=click.File("rb"), required=False)
@click.option(
    "-n",
    "--name",
    help="Name for the created compute config. This should *not* include a version tag. If a name is not provided, an anonymous compute config is generated. Anonymous compute configs are not accessible in the UI and can only be referenced by their ID.",
    required=False,
    default=None,
    type=str,
)
@click.option(
    "-f",
    "--config-file",
    required=False,
    default=None,
    type=str,
    help="Path to a YAML config file defining the compute config. Schema: https://docs.anyscale.com/reference/compute-config-api#computeconfig.",
)
def create_compute_config(
    compute_config_file: Optional[IO[bytes]],
    config_file: Optional[str],
    name: Optional[str],
):
    if compute_config_file and config_file:
        raise click.ClickException(
            "Only one of the --config-file flag or [COMPUTE_CONFIG_FILE] argument can be provided."
        )

    if compute_config_file is not None:
        ComputeConfigController().create(compute_config_file, name)
    elif config_file is not None:
        config = compute_config_type_from_yaml(config_file)
        anyscale.compute_config.create(config, name=name)
    else:
        raise click.ClickException(
            "Either the --config-file flag or [COMPUTE_CONFIG_FILE] argument must be provided."
        )


@compute_config_cli.command(
    name="archive",
    help=("Archive all versions of a specified compute config.\n\n"),
    cls=AnyscaleCommand,
    example=command_examples.COMPUTE_CONFIG_ARCHIVE_EXAMPLE,
)
@click.argument("compute-config-name", type=str, required=False)
@click.option(
    "-n",
    "--name",
    help="Name of the compute config to archive.",
    required=False,
    type=str,
)
@click.option(
    "--compute-config-id",
    "--id",
    help="ID of the compute config to archive. Alternative to name.",
    required=False,
    type=str,
)
def archive_compute_config(
    compute_config_name: Optional[str],
    name: Optional[str],
    compute_config_id: Optional[str],
) -> None:
    name, cc_id = _validate_name_and_id_args(
        positional_name=compute_config_name, flag_name=name, id_flag=compute_config_id
    )
    anyscale.compute_config.archive(name=name, _id=cc_id)


@compute_config_cli.command(
    name="list",
    help=(
        "List compute configurations with filtering, sorting, and pagination.\n\n"
        "By default, only compute configs created by the current user are returned. "
        "Use --include-shared to include configs shared with you.\n\n"
        "Use --max-items to control page size and --next-token for pagination."
    ),
    cls=AnyscaleCommand,
    example=command_examples.COMPUTE_CONFIG_LIST_EXAMPLE,
)
@click.option(
    "-n",
    "--name",
    required=False,
    default=None,
    help="List information about the compute config with this name.",
)
@click.option(
    "--compute-config-id",
    "--id",
    required=False,
    default=None,
    help="List information about the compute config with this id.",
)
@click.option(
    "--include-shared",
    is_flag=True,
    default=False,
    help="Include all compute configs you have access to.",
)
@click.option(
    "--max-items",
    required=False,
    default=20,
    type=int,
    help="Maximum number of items to return per page (default: 20).",
    callback=validate_non_negative_arg,
)
@click.option(
    "--next-token",
    required=False,
    default=None,
    type=str,
    help="Token for pagination to fetch the next page of results.",
)
@click.option(
    "--cloud-id", required=False, default=None, type=str, help="Filter by cloud ID.",
)
@click.option(
    "--cloud-name",
    required=False,
    default=None,
    type=str,
    help="Filter by cloud name.",
)
@click.option(
    "--sort-by",
    required=False,
    default="last_modified_at",
    type=click.Choice(["name", "created_at", "last_modified_at"], case_sensitive=False),
    help="Field to sort by. Default: last_modified_at",
)
@click.option(
    "--sort-order",
    required=False,
    default="asc",
    type=click.Choice(["asc", "desc"], case_sensitive=False),
    help="Sort order. Default: asc (ascending)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output results in JSON format.",
)
def list_compute_configs(  # noqa: A001, PLR0913
    name: Optional[str],
    compute_config_id: Optional[str],
    include_shared: bool,
    max_items: int,
    next_token: Optional[str],
    cloud_id: Optional[str],
    cloud_name: Optional[str],
    sort_by: str,
    sort_order: str,
    output_json: bool,
):
    """List compute configurations with filtering, sorting, and pagination options."""
    # Validate mutual exclusion: cloud_id and cloud_name cannot be used together
    if cloud_id and cloud_name:
        raise click.ClickException(
            "Error: --cloud-id and --cloud-name are mutually exclusive. "
            "Please provide only one."
        )

    # Use the SDK for listing compute configs
    result = anyscale.compute_config.list(
        name=name,
        _id=compute_config_id,
        cloud_id=cloud_id,
        cloud_name=cloud_name,
        sort_by=sort_by,
        sort_order=sort_order,
        max_items=max_items,
        next_token=next_token,
        include_shared=include_shared,
    )

    # Output in JSON format if requested
    if output_json:
        output_data = {
            "results": [
                {
                    "id": cc.id,
                    "name": cc.name,
                    "cloud_id": cc.config.cloud_id if cc.config else None,
                    "version": cc.version,
                    "created_at": cc.created_at.isoformat() if cc.created_at else None,
                    "last_modified_at": cc.last_modified_at.isoformat()
                    if cc.last_modified_at
                    else None,
                    "url": get_endpoint(f"configurations/cluster-computes/{cc.id}"),
                }
                for cc in result.results
            ],
            "metadata": {"count": result.count, "next_token": result.next_token,},
        }
        print(json.dumps(output_data, indent=2))
        return

    # Build table for display
    api_client = BaseController().anyscale_api_client

    cluster_compute_table = [
        [
            cluster_compute.id,
            cluster_compute.name,
            api_client.get_cloud(cluster_compute.config.cloud_id).result.name
            if cluster_compute.config.cloud_id
            else None,
            cluster_compute.last_modified_at.strftime("%m/%d/%Y, %H:%M:%S"),
            get_endpoint(f"configurations/cluster-computes/{cluster_compute.id}"),
        ]
        for cluster_compute in result.results
    ]

    table = tabulate.tabulate(
        cluster_compute_table,
        headers=["ID", "NAME", "CLOUD", "LAST MODIFIED AT", "URL"],
        tablefmt="plain",
    )
    print(f"Compute configs:\n{table}")

    # Print pagination info if there are more results
    if result.next_token:
        print(
            f"\nMore results available. Use --next-token '{result.next_token}' to fetch the next page."
        )


@compute_config_cli.command(
    name="get",
    help=(
        "Get the details of a compute config.\n\n"
        "The name can contain an optional version, e.g., 'name:version'. "
        "If no version is provided, the latest one will be returned.\n\n"
    ),
    cls=AnyscaleCommand,
    example=command_examples.COMPUTE_CONFIG_GET_EXAMPLE,
)
@click.argument("compute-config-name", required=False)
@click.option(
    "-n", "--name", required=False, default=None, help="Name of the compute config.",
)
@click.option(
    "--compute-config-id",
    "--id",
    required=False,
    default=None,
    help="ID of the compute config. Alternative to name.",
    hidden=True,
)
@click.option(
    "--include-archived", is_flag=True, help="Include archived compute configurations.",
)
@click.option(
    "--cloud-id",
    required=False,
    default=None,
    type=str,
    help="Filter by cloud ID when resolving compute config by name.",
)
@click.option(
    "--cloud-name",
    required=False,
    default=None,
    type=str,
    help="Filter by cloud name when resolving compute config by name.",
)
@click.option(
    "--old-format",
    is_flag=True,
    default=False,
    help="Output the config in the old format: https://docs.anyscale.com/reference/python-sdk/models#createclustercomputeconfig.",
)
def get_compute_config(
    name: Optional[str],
    compute_config_name: Optional[str],
    compute_config_id: Optional[str],
    include_archived: bool,
    cloud_id: Optional[str],
    cloud_name: Optional[str],
    old_format: bool,
):
    """Get details of a specific compute configuration."""
    # Validate mutual exclusion: cloud_id and cloud_name cannot be used together
    if cloud_id and cloud_name:
        raise click.ClickException(
            "Error: --cloud-id and --cloud-name are mutually exclusive. "
            "Please provide only one."
        )

    name, cc_id = _validate_name_and_id_args(
        positional_name=compute_config_name, flag_name=name, id_flag=compute_config_id
    )

    # Resolve cloud filtering parameter
    cloud_filter = None
    if cloud_name:
        cloud_filter = cloud_name
    elif cloud_id:
        # Resolve cloud_id to cloud_name for the SDK
        _, cloud_filter = get_cloud_id_and_name(
            api_client=BaseController().api_client, cloud_id=cloud_id, cloud_name=None
        )

    if old_format:
        ComputeConfigController().get(
            cluster_compute_name=name,
            cluster_compute_id=cc_id,
            include_archived=include_archived,
            cloud_id=cloud_id,
            cloud_name=cloud_name,
        )
    else:
        # New format (YAML) - now supports cloud filtering
        config: ComputeConfigVersion = anyscale.compute_config.get(
            name=name, _id=cc_id, cloud=cloud_filter, include_archived=include_archived,
        )
        stream = StringIO()
        yaml.dump(config.to_dict(), stream, sort_keys=False)
        print(stream.getvalue(), end="")
