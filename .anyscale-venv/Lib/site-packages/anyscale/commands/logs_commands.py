"""
This file holds all of the CLI commands for the "anyscale logs" path. Note that
most of the implementation for this command is in the controller to make the controller
accessible to the SDK in the future.

TODO (shomilj): Bring the controller to feature parity with the CLI.
"""
from datetime import timedelta
import math
from typing import Optional

import click
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models import LogFilter
from anyscale.client.openapi_client.models.node_type import NodeType
from anyscale.controllers.logs_controller import (
    DEFAULT_PAGE_SIZE,
    DEFAULT_PARALLELISM,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_TIMEOUT,
    DEFAULT_TTL,
    LogsController,
)


log = BlockLogger()

# Options to configure core functionality. These are mutually exclusive.
option_download = click.option(
    "-d",
    "--download",
    is_flag=True,
    default=False,
    help="Download logs to the current working directory, or a specified path.",
)
option_tail = click.option(
    "-t", "--tail", type=int, default=-1, help="Read the last N lines of logs."
)

# The "glob" is an optional argument.
# e.g. anyscale logs cluster --cluster-id <cluster-id> [GLOB]
argument_glob = click.argument("glob", type=str, default=None, required=False)

# Options to filter beyond what the subcommand already filters to.
option_node_ip = click.option(
    "-ip", "--node-ip", type=str, default=None, help="Filter logs by a node IP."
)
option_instance_id = click.option(
    "--instance-id", type=str, default=None, help="Filter logs by an instance ID."
)
option_worker_only = click.option(
    "--worker-only", is_flag=True, help="Download logs of only the worker nodes."
)
option_head_only = click.option(
    "--head-only", is_flag=True, help="Download logs of only the head node."
)
option_unpack_combined_logs = click.option(
    "--unpack/--no-unpack",
    default=True,
    help="Whether to unpack the combined-worker.log after downloading.",
    hidden=True,
)

option_ttl = click.option(
    "--ttl",
    type=int,
    default=DEFAULT_TTL,
    hidden=True,
    help="TTL in seconds to pass to the service that generates presigned URL's (default: 4h).",
)
option_parallelism = click.option(
    "--parallelism",
    type=int,
    default=DEFAULT_PARALLELISM,
    hidden=True,
    help="Number of files to download in parallel at a time.",
)

# ADVANCED: Configure the download behavior, only useful if --download enabled.
option_download_dir = click.option(
    "--download-dir", type=str, default=None, help="Directory to download logs into."
)


@click.group(
    "logs", help="Print or download Ray logs for an Anyscale job, service, or cluster.",
)
def log_cli() -> None:
    pass


@log_cli.command(name="cluster", help="Access log files of a cluster.")
@click.option("--id", type=str, required=True, help="Provide a cluster ID.")
@option_download
@option_tail
@argument_glob
@option_node_ip
@option_instance_id
@option_worker_only
@option_head_only
@option_unpack_combined_logs
@option_download_dir
@option_ttl
@option_parallelism
def anyscale_logs_cluster(  # noqa: PLR0913
    id: str,  # noqa: A002
    download: bool,
    tail: int,
    # filters
    glob: Optional[str],
    node_ip: Optional[str],
    instance_id: Optional[str],
    worker_only: bool,
    head_only: bool,
    unpack: bool,
    # list files config
    ttl: int,
    # download files config
    download_dir: Optional[str],
    parallelism: int,
) -> None:
    logs_controller = LogsController()
    execute_anyscale_logs_cluster(
        logs_controller=logs_controller,
        cluster_id=id,
        download=download,
        tail=tail,
        glob=glob,
        node_ip=node_ip,
        instance_id=instance_id,
        worker_only=worker_only,
        head_only=head_only,
        unpack=unpack,
        ttl=ttl,
        download_dir=download_dir,
        parallelism=parallelism,
    )


def execute_anyscale_logs_cluster(  # noqa: PLR0913
    logs_controller: LogsController,
    cluster_id: str,
    download: bool,
    tail: int,
    glob: Optional[str],
    node_ip: Optional[str],
    instance_id: Optional[str],
    worker_only: bool,
    head_only: bool,
    unpack: bool,  # list files config
    ttl: int,
    # download files config
    download_dir: Optional[str],
    parallelism: int,
    resource_id: Optional[str] = None,
):

    node_type: Optional[NodeType] = None
    if worker_only and head_only:
        raise click.ClickException("Cannot specify both --worker-only and --head-only.")
    if worker_only:
        node_type = NodeType.WORKER_NODES
    elif head_only:
        node_type = NodeType.HEAD_NODE

    filter = LogFilter(  # noqa: A001
        cluster_id=cluster_id,
        glob=glob,
        node_ip=node_ip,
        instance_id=instance_id,
        node_type=node_type,
    )

    if download:
        logs_controller.download_logs(
            filter=filter,
            page_size=DEFAULT_PAGE_SIZE,
            timeout=timedelta(seconds=DEFAULT_TIMEOUT),
            read_timeout=timedelta(seconds=DEFAULT_READ_TIMEOUT),
            ttl_seconds=ttl,
            download_dir=download_dir,
            parallelism=parallelism,
            unpack=unpack,
            resource_id=resource_id,
        )

    else:
        # This is for both tailing logs AND for the default behavior (no -t/-d/-f => rendering them in UI).
        console = Console()
        log_group = logs_controller.get_log_group(
            filter=filter,
            page_size=DEFAULT_PAGE_SIZE,
            timeout=timedelta(seconds=DEFAULT_TIMEOUT),
            ttl_seconds=ttl,
        )
        if len(log_group.get_files()) == 0:
            console.print("No results found.")
        elif len(log_group.get_files()) > 1:
            console.print(
                "These are the available log files. To download all files, use --download. To render a specific file, just paste the filename after this command."
            )
            click.echo()
            for session in log_group.get_sessions():
                click.echo()
                renderables = []

                for node in session.get_nodes():
                    table = Table()
                    table.add_column("File Name", justify="left", style="green")
                    table.add_column("Size")
                    for log_file in node.get_files():
                        table.add_row(
                            log_file.file_name, convert_size(log_file.get_size()),
                        )
                    prefix = (
                        "Head Node"
                        if node.node_type == NodeType.HEAD_NODE
                        else "Worker Node"
                    )
                    # TODO (shomilj): When we support GCE, clean this up.
                    instance_id = (
                        f"EC2 Instance ID: {node.instance_id}"
                        if node.instance_id.startswith("i-")
                        else node.instance_id
                    )
                    renderables.append(
                        Panel(
                            Group(table),
                            title=f"{prefix}: {node.node_ip} ({instance_id})",
                            title_align="left",
                        )
                    )
                group = Group(*renderables)
                console.print(
                    Panel(
                        group,
                        title=f"Session: {session.session_id}",
                        title_align="left",
                        padding=(1, 1),
                    )
                )

        else:
            logs_controller.render_logs(
                log_group=log_group,
                parallelism=parallelism,
                read_timeout=timedelta(seconds=DEFAULT_READ_TIMEOUT),
                tail=tail,
            )

        click.echo()


@log_cli.command(
    name="job",
    help="Access log files of a production job. Fetches logs for all job attempts.",
)
@click.option("--id", type=str, required=True, help="Provide a production job ID.")
@option_download
@option_tail
@argument_glob
@option_node_ip
@option_instance_id
@option_worker_only
@option_head_only
@option_unpack_combined_logs
@option_download_dir
@option_ttl
@option_parallelism
def anyscale_logs_job(  # noqa: PLR0913
    id: str,  # noqa: A002
    download: bool,
    tail: int,
    # filters
    glob: Optional[str],
    node_ip: Optional[str],
    instance_id: Optional[str],
    worker_only: bool,
    head_only: bool,
    unpack: bool,
    # list files config
    ttl: int,
    # download files config
    download_dir: Optional[str],
    parallelism: int,
) -> None:
    logs_controller = LogsController()
    cluster_id = logs_controller.get_cluster_id_for_last_prodjob_run(prodjob_id=id)
    execute_anyscale_logs_cluster(
        logs_controller=logs_controller,
        cluster_id=cluster_id,
        download=download,
        tail=tail,
        glob=glob,
        node_ip=node_ip,
        instance_id=instance_id,
        worker_only=worker_only,
        head_only=head_only,
        unpack=unpack,
        ttl=ttl,
        download_dir=download_dir,
        parallelism=parallelism,
        resource_id=id if download else None,
    )


@log_cli.command(name="workspace", help="Access log files of a workspace.")
@click.option("--id", type=str, required=True, help="Provide a workspace ID.")
@option_download
@option_tail
@argument_glob
@option_node_ip
@option_instance_id
@option_worker_only
@option_head_only
@option_unpack_combined_logs
@option_download_dir
@option_ttl
@option_parallelism
def anyscale_logs_workspace(  # noqa: PLR0913
    id: str,  # noqa: A002
    download: bool,
    tail: int,
    # filters
    glob: Optional[str],
    node_ip: Optional[str],
    instance_id: Optional[str],
    worker_only: bool,
    head_only: bool,
    unpack: bool,
    # list files config
    ttl: int,
    # download files config
    download_dir: Optional[str],
    parallelism: int,
) -> None:
    logs_controller = LogsController()
    cluster_id = logs_controller.get_cluster_id_for_workspace(workspace_id=id)
    execute_anyscale_logs_cluster(
        logs_controller=logs_controller,
        cluster_id=cluster_id,
        download=download,
        tail=tail,
        glob=glob,
        node_ip=node_ip,
        instance_id=instance_id,
        worker_only=worker_only,
        head_only=head_only,
        unpack=unpack,
        ttl=ttl,
        download_dir=download_dir,
        parallelism=parallelism,
        resource_id=id if download else None,
    )


@log_cli.command(
    name="service", help="Access log files of a service for a single service version."
)
@click.option("--id", type=str, required=True, help="Provide a service ID.")
@click.option(
    "--version",
    type=str,
    required=False,
    help="Service version name or ID to get logs from. If not specified, uses the latest running version.",
)
@option_download
@option_tail
@argument_glob
@option_node_ip
@option_instance_id
@option_worker_only
@option_head_only
@option_unpack_combined_logs
@option_download_dir
@option_ttl
@option_parallelism
def anyscale_logs_service(  # noqa: PLR0913
    id: str,  # noqa: A002
    version: Optional[str],
    download: bool,
    tail: int,
    # filters
    glob: Optional[str],
    node_ip: Optional[str],
    instance_id: Optional[str],
    worker_only: bool,
    head_only: bool,
    unpack: bool,
    # list files config
    ttl: int,
    # download files config
    download_dir: Optional[str],
    parallelism: int,
) -> None:
    logs_controller = LogsController()
    cluster_id = logs_controller.get_cluster_id_for_service(
        service_id=id, version_name_or_id=version
    )
    execute_anyscale_logs_cluster(
        logs_controller=logs_controller,
        cluster_id=cluster_id,
        download=download,
        tail=tail,
        glob=glob,
        node_ip=node_ip,
        instance_id=instance_id,
        worker_only=worker_only,
        head_only=head_only,
        unpack=unpack,
        ttl=ttl,
        download_dir=download_dir,
        parallelism=parallelism,
        resource_id=id if download else None,
    )


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = math.floor(math.log(size_bytes, 1024))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"
