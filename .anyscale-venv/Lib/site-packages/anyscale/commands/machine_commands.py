"""
This file holds all of the CLI commands for the "anyscale machine" path.
"""
from typing import Optional

import click
import humanize
from rich import print_json
import tabulate

from anyscale.commands.util import LegacyAnyscaleCommand
from anyscale.controllers.machine_controller import MachineController


@click.group(
    "machine", help="Commands to interact with machines in Anyscale.",
)
def machine_cli() -> None:
    pass


@machine_cli.command(
    name="list",
    help="List machines registered to Anyscale.",
    cls=LegacyAnyscaleCommand,
    is_limited_support=True,
    legacy_prefix="anyscale machine",
)
@click.option("--machine-pool-name", type=str, help="Provide a machine pool name.")
@click.option(
    "--format",
    "format_",
    type=click.Choice(["table", "json"]),
    default="table",
    help="The format to return results in.",
)
def list_machines(machine_pool_name: str, format_: str,) -> None:
    machines_controller = MachineController()
    output = machines_controller.list_machines(machine_pool_name)

    if format_ == "json":
        rows = []
        for m in output.machines:
            rows.append(
                {
                    "machine_id": m.machine_id,
                    "hostname": m.hostname,
                    "machine_shape": m.machine_shape,
                    "num_cpus": m.num_cpus,
                    "num_gpus": m.num_gpus,
                    "memory": humanize.naturalsize(m.memory_bytes, binary=True),
                    "connection_state": m.connection_state,
                    "allocation_state": m.allocation_state,
                    "cluster_id": m.cluster_id,
                    "cluster_creator_name": m.cluster_creator_name,
                    "cluster_name": m.cluster_name,
                    "anyscale_version": m.anyscale_version,
                }
            )
        print_json(data=rows)
    elif format_ == "table":
        table = []
        columns = [
            "MACHINE",
            "HOSTNAME",
            "SHAPE",
            "NUM_CPUS",
            "NUM_GPUS",
            "MEMORY",
            "HEALTHY",
            "STATUS",
            "OWNER",
            "CLUSTER ID",
            "CLUSTER NAME",
            "ANYSCALE VERSION",
        ]
        for m in output.machines:
            table.append(
                [
                    m.machine_id,
                    m.hostname,
                    m.machine_shape,
                    m.num_cpus,
                    m.num_gpus,
                    humanize.naturalsize(m.memory_bytes, binary=True),
                    "Yes" if m.connection_state == "connected" else "No",
                    m.allocation_state.title(),
                    m.cluster_creator_name,
                    m.cluster_id,
                    m.cluster_name,
                    m.anyscale_version,
                ]
            )

        print(
            tabulate.tabulate(table, tablefmt="plain", headers=columns, stralign="left")
        )


@machine_cli.command(
    name="delete",
    help="Delete a machine from a machine pool.",
    cls=LegacyAnyscaleCommand,
    is_limited_support=True,
    legacy_prefix="anyscale machine",
)
@click.option("--machine-id", type=str, help="Provide a machine ID.")
@click.option("--machine-pool-name", type=str, help="Provide a machine pool name.")
def delete_machine(machine_id: str, machine_pool_name: Optional[str],) -> None:

    machines_controller = MachineController()
    machines_controller.delete_machine(
        machine_id=machine_id, machine_pool_name=machine_pool_name,
    )

    print(f"Machine {machine_id} has been deleted.")
