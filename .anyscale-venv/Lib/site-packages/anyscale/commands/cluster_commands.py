import os
from typing import Optional

import click

from anyscale.controllers.cluster_controller import ClusterController
from anyscale.util import validate_non_negative_arg


HIDE_BYOD_FLAGS = os.getenv("ANYSCALE_BYOD") != "1"


@click.group("cluster", help="Interact with clusters on Anyscale.")
def cluster_cli() -> None:
    pass


@cluster_cli.command(name="archive", help="Archive a cluster on Anyscale.")
@click.option(
    "--name",
    "-n",
    required=False,
    default=None,
    help="Name of existing cluster to archive.",
)
@click.option(
    "--cluster-id",
    "--id",
    required=False,
    default=None,
    help=(
        "Id of existing cluster to archive. This argument "
        "can be used to archive any cluster you have access to in any project."
    ),
)
@click.option(
    "--project-id",
    required=False,
    default=None,
    help=(
        "Override project id used for this cluster. If not provided, the Anyscale project "
        "context will be used if it exists. Otherwise a default project will be used."
    ),
)
@click.option(
    "--project",
    required=False,
    default=None,
    help=(
        "Override project name used for this cluster. If not provided, the Anyscale project "
        "context will be used if it exists. Otherwise a default project will be used."
    ),
)
@click.option(
    "--cloud-id",
    required=False,
    default=None,
    help=(
        "Use cloud ID to disambiguate only when selecting a cluster to archive with `--name`"
        "that doesn't belong to any project. This requires cloud isolation to be enabled."
    ),
    hidden=True,
)
@click.option(
    "--cloud",
    required=False,
    default=None,
    help=(
        "Use cloud to disambiguate only when selecting a cluster to archive with `--name`"
        "that doesn't belong to any project. This requires cloud isolation to be enabled."
    ),
    hidden=True,
)
def archive(
    name: Optional[str],
    cluster_id: Optional[str],
    project_id: Optional[str],
    project: Optional[str],
    cloud_id: Optional[str],
    cloud: Optional[str],
) -> None:
    cluster_controller = ClusterController()
    cluster_controller.archive(
        cluster_name=name,
        cluster_id=cluster_id,
        project_id=project_id,
        project_name=project,
        cloud_id=cloud_id,
        cloud_name=cloud,
    )


@cluster_cli.command(
    name="list",
    help=(
        "List information about clusters on Anyscale. By default only list "
        "active clusters in current project."
    ),
)
@click.option(
    "--name",
    "-n",
    required=False,
    default=None,
    help="Name of existing cluster to get information about.",
)
@click.option(
    "--cluster-id",
    "--id",
    required=False,
    default=None,
    help=(
        "Id of existing cluster get information about. This argument can be used "
        "to interact with any cluster you have access to in any project."
    ),
)
@click.option(
    "--project-id",
    required=False,
    default=None,
    help=(
        "Override project id used for this cluster. If not provided, the Anyscale project "
        "context will be used if it exists. Otherwise a default project will be used."
    ),
)
@click.option(
    "--project",
    required=False,
    default=None,
    help=(
        "Override project name used for this cluster. If not provided, the Anyscale project "
        "context will be used if it exists. Otherwise a default project will be used."
    ),
)
@click.option(
    "--include-all-projects",
    is_flag=True,
    default=False,
    help="List all active clusters user has access to in any project.",
)
@click.option(
    "--include-inactive",
    is_flag=True,
    default=False,
    help="List clusters of all states.",
)
@click.option(
    "--include-archived",
    is_flag=True,
    default=False,
    help=(
        "List archived clusters as well as unarchived clusters."
        "If not provided, defaults to listing only unarchived clusters."
    ),
)
@click.option(
    "--max-items",
    required=False,
    default=20,
    type=int,
    help="Max items to show in list.",
    callback=validate_non_negative_arg,
)
@click.option(
    "--cloud-id",
    required=False,
    default=None,
    help=(
        "Use cloud ID to disambiguate only when selecting a cluster to list with `--name`"
        "that doesn't belong to any project. This requires cloud isolation to be enabled."
        "Note: This command doesn't support filtering clusters by cloud."
    ),
    hidden=True,
)
@click.option(
    "--cloud",
    required=False,
    default=None,
    help=(
        "Use cloud to disambiguate only when selecting a cluster to list with `--name`"
        "that doesn't belong to any project. This requires cloud isolation to be enabled."
        "Note: This command doesn't support filtering clusters by cloud."
    ),
    hidden=True,
)
def list(  # noqa: A001, PLR0913
    name: Optional[str],
    cluster_id: Optional[str],
    project_id: Optional[str],
    project: Optional[str],
    include_all_projects: bool,
    include_inactive: bool,
    include_archived: bool,
    max_items: int,
    cloud_id: Optional[str],
    cloud: Optional[str],
) -> None:
    cluster_controller = ClusterController()
    cluster_controller.list(
        cluster_name=name,
        cluster_id=cluster_id,
        project_id=project_id,
        project_name=project,
        include_all_projects=include_all_projects,
        include_inactive=include_inactive,
        include_archived=include_archived,
        max_items=max_items,
        cloud_id=cloud_id,
        cloud_name=cloud,
    )


@cluster_cli.command(
    name="network_debug",
    help="Debug local network connectivity to a cluster.",
    hidden=True,
)
@click.argument(
    "cluster_id", required=True, default=None,
)
def network_debug(cluster_id: Optional[str],) -> None:
    """Debug network connectivity to a given Anyscale cluster."""
    cluster_controller = ClusterController()
    cluster_controller.debug_networking(cluster_id=cluster_id)
