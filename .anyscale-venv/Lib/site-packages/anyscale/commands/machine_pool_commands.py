"""
This file holds all of the CLI commands for the "anyscale machine-pool" path.
"""

import json
from typing import Optional

import click
import tabulate
import yaml

from anyscale.client.openapi_client.models import (
    DescribeMachinePoolResponse,
    SchedulerInfo,
)
from anyscale.commands import command_examples
from anyscale.commands.util import AnyscaleCommand
from anyscale.controllers.machine_pool_controller import MachinePoolController


@click.group(
    "machine-pool", help="Commands to interact with machine pools in Anyscale.",
)
def machine_pool_cli() -> None:
    pass


@machine_pool_cli.command(
    name="create",
    help="Create a machine pool in Anyscale.",
    cls=AnyscaleCommand,
    example=command_examples.MACHINE_POOL_CREATE_EXAMPLE,
    is_beta=True,
)
@click.option(
    "--name",
    type=str,
    required=True,
    help="Provide a machine pool name (must be unique within an organization).",
)
def create_machine_pool(name: str) -> None:
    machine_pool_controller = MachinePoolController()
    output = machine_pool_controller.create_machine_pool(machine_pool_name=name,)
    print(
        f"Machine pool {output.machine_pool.machine_pool_name} has been created successfully (ID {output.machine_pool.machine_pool_id})."
    )


@machine_pool_cli.command(
    name="update",
    help="Update a machine pool in Anyscale.",
    cls=AnyscaleCommand,
    example=command_examples.MACHINE_POOL_UPDATE_EXAMPLE,
    is_beta=True,
)
@click.option("--name", type=str, required=True, help="Provide a machine pool name.")
@click.option(
    "--spec-file",
    type=str,
    required=True,
    help="Provide a path to a specification file.",
)
def update_machine_pool(name: str, spec_file: str) -> None:
    machine_pool_controller = MachinePoolController()
    machine_pool_controller.update_machine_pool(
        machine_pool_name=name, spec_file=spec_file
    )
    print(f"Updated machine pool '{name}'.")


@machine_pool_cli.command(
    name="describe",
    help="Describe a machine pool in Anyscale.",
    cls=AnyscaleCommand,
    example=command_examples.MACHINE_POOL_DESCRIBE_EXAMPLE,
    is_beta=True,
)
@click.option("--name", type=str, required=True, help="Provide a machine pool name.")
@click.option(
    "--format",
    "format_",
    type=str,
    default="table",
    required=False,
    help="Output format (table, json).",
)
def describe(name: str, format_: str) -> None:
    machine_pool_controller = MachinePoolController()
    response: DescribeMachinePoolResponse = machine_pool_controller.describe_machine_pool(
        machine_pool_name=name
    )
    scheduler_info: SchedulerInfo = response.scheduler_info  # type: ignore
    if format_ == "json":
        print(json.dumps(scheduler_info.to_dict(), default=str))
    elif format_ == "table":

        def format_time(timestamp):
            return str(timestamp.astimezone().strftime("%m/%d/%Y %I:%M:%S %p %Z"))

        machines_table = []
        columns = [
            "MACHINE ID",
            "TYPE",
            "PARTITION",
            "STATE",
            "WORKLOAD DETAILS",
            "WORKLOAD START TIME",
            "WORKLOAD CREATOR",
            "WORKLOAD SCORE",
            "CLOUD INSTANCE ID",
        ]
        for row in scheduler_info.machines:
            machines_table.append(
                [
                    row.machine_id,
                    row.machine_type,
                    row.partition,
                    row.allocation_state,
                    f"{row.workload_info.workload_type}/{row.workload_info.workload_name}/{row.workload_info.workload_project}/{row.workload_info.workload_cloud}"
                    if row.workload_info.workload_name
                    else "",
                    format_time(row.workload_info.workload_start_time)
                    if row.workload_info.workload_name
                    else "",
                    row.workload_info.workload_creator
                    if row.workload_info.workload_name
                    else "",
                    row.workload_score,
                    row.cloud_instance_id,
                ]
            )

        # Sort by (type, partition, state, workload start time)
        machines_table.sort(key=lambda x: (x[1], x[2], x[3], x[6]))

        print("Machines:")
        print(
            tabulate.tabulate(
                machines_table, tablefmt="outline", headers=columns, stralign="left"
            )
        )

        requests_table = []
        columns = [
            "SIZE",
            "MACHINE TYPE",
            "WORKLOAD DETAILS",
            "WORKLOAD START TIME",
            "WORKLOAD CREATOR",
            "PARTITION SCORES",
        ]
        for row in scheduler_info.requests:
            requests_table.append(
                [
                    row.size,
                    row.machine_type,
                    f"{row.workload_info.workload_type}/{row.workload_info.workload_name}/{row.workload_info.workload_project}/{row.workload_info.workload_cloud}",
                    format_time(row.workload_info.workload_start_time),
                    row.workload_info.workload_creator,
                    row.partition_scores,
                ]
            )

        # Sort by (machine type, workload start time, size)
        print("Requests:")
        print(
            tabulate.tabulate(
                requests_table, tablefmt="outline", headers=columns, stralign="left"
            )
        )

        if (
            scheduler_info.recent_launch_failures
            and len(scheduler_info.recent_launch_failures) > 0
        ):
            print("Recent Launch Failures:")
            for failure in scheduler_info.recent_launch_failures:
                error = failure.error.replace("\n", " ")
                print(f"- [{format_time(failure.timestamp)}] {error}")


@machine_pool_cli.command(
    name="delete",
    help="Delete a machine pool in Anyscale.",
    cls=AnyscaleCommand,
    example=command_examples.MACHINE_POOL_DELETE_EXAMPLE,
    is_beta=True,
)
@click.option("--name", type=str, required=True, help="Provide a machine pool name.")
def delete_machine_pool(name: str) -> None:
    machine_pool_controller = MachinePoolController()
    machine_pool_controller.delete_machine_pool(machine_pool_name=name)
    print(f"Deleted machine pool '{name}'.")


@machine_pool_cli.command(
    name="list",
    help="List machine pools in Anyscale.",
    cls=AnyscaleCommand,
    example=command_examples.MACHINE_POOL_LIST_EXAMPLE,
    is_beta=True,
)
@click.option(
    "--format",
    "format_",
    type=str,
    default="table",
    required=False,
    help="Output format (table, yaml).",
)
def list_machine_pools(format_: str) -> None:
    machine_pool_controller = MachinePoolController()
    result = machine_pool_controller.list_machine_pools()

    if format_ == "table":
        table = []
        columns = [
            "MACHINE POOL",
            "ID",
            "Clouds",
        ]
        for mp in result.machine_pools:
            formatted_cloud_resources = [
                machine_pool_controller.format_cloud_and_cloud_resources(
                    cloud_id, cloud_resource_ids
                )
                for cloud_id, cloud_resource_ids in mp.cloud_resource_ids.items()
            ]

            table.append(
                [
                    mp.machine_pool_name,
                    mp.machine_pool_id,
                    "\n".join(formatted_cloud_resources),
                ]
            )
        print(
            tabulate.tabulate(
                table, tablefmt="simple_grid", headers=columns, stralign="left"
            )
        )
    elif format_ == "yaml":
        rows = []
        for mp in result.machine_pools:
            rows.append(
                {
                    "machine_pool_name": mp.machine_pool_name,
                    "machine_pool_id": mp.machine_pool_id,
                    "cloud_ids": mp.cloud_ids,
                    "spec": mp.spec,
                }
            )
        print(yaml.dump(data=rows, width=float("inf")))  # type: ignore
    else:
        raise click.ClickException(f"Invalid output format '{format}'.")


@machine_pool_cli.command(
    name="attach",
    help="Attach a machine pool to a cloud or cloud resource.",
    cls=AnyscaleCommand,
    example=command_examples.MACHINE_POOL_ATTACH_EXAMPLE,
    is_beta=True,
)
@click.option("--name", type=str, required=True, help="Provide a machine pool name.")
@click.option("--cloud", type=str, required=True, help="Provide a cloud name.")
@click.option(
    "--resource",
    type=str,
    required=False,
    default=None,
    help="For multi-resource clouds, the name of the cloud resource to attach to. If not provided, attaches to the primary cloud resource in the cloud.",
)
def attach_machine_pool_to_cloud(
    name: str, cloud: str, resource: Optional[str] = None
) -> None:
    machine_pool_controller = MachinePoolController()
    machine_pool_controller.attach_machine_pool_to_cloud(
        machine_pool_name=name, cloud_name=cloud, cloud_resource_name=resource
    )
    if resource:
        print(
            f"Attached machine pool '{name}' to resource '{resource}' in cloud '{cloud}'."
        )
    else:
        print(f"Attached machine pool '{name}' to cloud '{cloud}'.")


@machine_pool_cli.command(
    name="detach",
    help="Detach a machine pool from a cloud or cloud resource.",
    cls=AnyscaleCommand,
    example=command_examples.MACHINE_POOL_DETACH_EXAMPLE,
    is_beta=True,
)
@click.option("--name", type=str, required=True, help="Provide a machine pool name.")
@click.option("--cloud", type=str, required=True, help="Provide a cloud name.")
@click.option(
    "--resource",
    type=str,
    required=False,
    default=None,
    help="For multi-resource clouds, the name of the cloud resource to detach from. If not provided, detaches from the primary cloud resource in the cloud.",
)
def detach_machine_pool_from_cloud(
    name: str, cloud: str, resource: Optional[str] = None
) -> None:
    machine_pool_controller = MachinePoolController()
    machine_pool_controller.detach_machine_pool_from_cloud(
        machine_pool_name=name, cloud_name=cloud, cloud_resource_name=resource
    )
    if resource:
        print(
            f"Detached machine pool '{name}' from resource '{resource}' in cloud '{cloud}'."
        )
    else:
        print(f"Detached machine pool '{name}' from cloud '{cloud}'.")
