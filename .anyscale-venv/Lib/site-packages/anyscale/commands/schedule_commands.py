from io import StringIO
from json import dumps as json_dumps
import pathlib
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table
import tabulate
import yaml

import anyscale
from anyscale.cli_logger import BlockLogger
from anyscale.commands import command_examples
from anyscale.commands.list_util import (
    create_table,
    display_list,
    NON_INTERACTIVE_DEFAULT_MAX_ITEMS,
    validate_page_size,
)
from anyscale.commands.util import AnyscaleCommand
from anyscale.controllers.schedule_controller import ScheduleController
from anyscale.schedule.models import (
    JobConfig,
    ScheduleConfig,
    ScheduleState,
    ScheduleStatus,
)
from anyscale.util import get_endpoint, validate_non_negative_arg


log = BlockLogger()  # CLI Logger


@click.group("schedule", help="Create and manage Anyscale Schedules.")
def schedule_cli() -> None:
    pass


def _read_identifiers_from_config_file(path: str):
    """Read the 'name', 'cloud', and 'project' properties from the config file at `path`.

    Return the identifers as a ScheduleIdentifiers object.
    """
    if not pathlib.Path(path).is_file():
        raise click.ClickException(f"Config file not found at path: '{path}'.")

    with open(path) as f:
        config = yaml.safe_load(f)

    if config is None or "job_config" not in config:
        raise click.ClickException(
            f"No 'job_config' property found in config file '{path}'."
        )

    job_config = config.get("job_config")
    name = job_config.get("name", None)
    cloud = job_config.get("cloud", None)
    project = job_config.get("project", None)

    return name, cloud, project


def _validate_schedule_identifiers(
    name: Optional[str], id: Optional[str], config_file: Optional[str]  # noqa: A002
):
    num_passed = sum(val is not None for val in [name, id, config_file])
    if num_passed == 0:
        raise click.ClickException(
            "One of '--name', '--id', or '--config-file' must be provided."
        )

    if num_passed > 1:
        raise click.ClickException(
            "Only one of '--name', '--id', and '--config-file' can be provided."
        )


@schedule_cli.command(
    name="apply", cls=AnyscaleCommand, example=command_examples.SCHEDULE_APPLY_EXAMPLE
)
@click.option(
    "--config-file",
    "-f",
    required=True,
    type=str,
    help="Path to a YAML config file to use for this schedule. Command-line flags will overwrite values read from the file.",
)
@click.option(
    "--name", "-n", required=False, default=None, help="Name of the schedule."
)
def apply(config_file: str, name: Optional[str],) -> None:
    """ Create or Update a Schedule

    The schedule should be specified in a YAML config file.
    """
    if not pathlib.Path(config_file).is_file():
        raise click.ClickException(f"Schedule config file '{config_file}' not found.")

    config = ScheduleConfig.from_yaml(config_file)

    if name is not None:
        assert isinstance(config.job_config, JobConfig)
        config = config.options(job_config=config.job_config.options(name=name),)

    log.info(f"Applying schedule with config {config}.")
    anyscale.schedule.apply(config)


def _create_schedules_table_v2(is_first: bool) -> Table:
    """Create a Rich Table for displaying schedules in v2 mode.

    Args:
        is_first: Whether this is the first page (controls header display).

    Returns:
        Rich Table configured for schedule display.
    """
    columns = [
        ("ID", "cyan", True),
        ("Name", None, False),
        ("State", "green", False),
        ("Cron Expression", None, False),
        ("Timezone", None, False),
        ("Project", None, False),
    ]
    return create_table(columns, is_first)


def _format_schedule_row_v2(schedule: ScheduleStatus) -> Dict[str, Any]:
    """Format a ScheduleStatus for table row or JSON output.

    Args:
        schedule: The ScheduleStatus object to format.

    Returns:
        Dictionary with formatted schedule data.
    """
    return {
        "id": schedule.id or "",
        "name": schedule.name or "",
        "state": str(schedule.state) if schedule.state else "",
        "cron_expression": schedule.config.cron_expression if schedule.config else "",
        "timezone": schedule.config.timezone if schedule.config else "",
        "project": schedule.config.job_config.project
        if schedule.config and schedule.config.job_config
        else "",
    }


def _display_schedules_table(schedules: List[ScheduleStatus]) -> None:
    """Display schedules in a tabulated format."""
    if not schedules:
        print("No schedules found.")
        return

    schedules_table = [
        [
            schedule.name,
            schedule.id,
            str(schedule.state) if schedule.state else None,
            schedule.config.cron_expression if schedule.config else None,
            schedule.config.timezone if schedule.config else None,
        ]
        for schedule in schedules
    ]

    table = tabulate.tabulate(
        schedules_table,
        headers=["NAME", "ID", "STATE", "CRON", "TIMEZONE"],
        tablefmt="plain",
    )
    print(f"SCHEDULES:\n{table}")

    endpoint = get_endpoint("")
    print(f"\nView your schedules at: {endpoint}schedules")


def _print_schedule_list_diagnostics(  # noqa: PLR0913
    stderr: Console,
    name: Optional[str],
    schedule_id: Optional[str],
    project: Optional[str],
    cloud: Optional[str],
    creator_id: Optional[str],
    include_all_users: bool,
    interactive: bool,
    page_size: int,
    effective_max: Optional[int],
) -> None:
    """Prints diagnostic information for the list command."""
    stderr.print("[bold]Listing schedules with:[/]")
    stderr.print(f"• name            = {name or '<any>'}")
    stderr.print(f"• id              = {schedule_id or '<any>'}")
    stderr.print(f"• project         = {project or '<any>'}")
    stderr.print(f"• cloud           = {cloud or '<any>'}")
    stderr.print(f"• creator_id      = {creator_id or '<any>'}")
    stderr.print(f"• include_all     = {include_all_users}")
    stderr.print(f"• mode            = {'interactive' if interactive else 'batch'}")
    stderr.print(f"• per-page limit  = {page_size}")
    stderr.print(f"• max-items total = {effective_max if effective_max else 'all'}")
    stderr.print(f"\nView your Schedules in the UI at {get_endpoint('/schedules')}\n")


# TODO(praneethkaturi): Add --sort option for schedule list.
# Requires backend changes:
#   - backend/server/experimental_cron/cron_jobs_base_model.py (add ScheduleSortField enum)
#   - backend/server/experimental_cron/cron_jobs_router.py (add sort query params)
#   - backend/server/experimental_cron/cron_jobs_dao.py (dynamic sorting)
# And SDK changes:
#   - frontend/cli/anyscale/schedule/_private/schedule_sdk.py
#   - frontend/cli/anyscale/schedule/commands.py
@schedule_cli.command(
    name="list", cls=AnyscaleCommand, example=command_examples.SCHEDULE_LIST_EXAMPLE
)
@click.option(
    "--v2",
    is_flag=True,
    default=False,
    help="[RECOMMENDED] Enable extended filtering options. Needs migration to match return values.",
)
@click.option(
    "--name",
    "-n",
    required=False,
    default=None,
    help="Filter by the name of the schedule.",
)
@click.option("--id", "-i", required=False, default=None, help="Id of the schedule.")
@click.option(
    "--project",
    required=False,
    default=None,
    help="The named Anyscale Project for the schedule. If not provided, the organization default will be used (or, if running in a workspace, the project of the workspace). Only with --v2 flag.",
)
@click.option(
    "--cloud",
    required=False,
    default=None,
    help="The named Anyscale Cloud for the schedule. If not provided, the organization default will be used (or,if running in a workspace, the cloud of the workspace). Only with --v2 flag.",
)
@click.option(
    "--creator-id",
    required=False,
    default=None,
    help="Filter by creator ID. Only with --v2 flag.",
)
@click.option(
    "--max-items",
    type=int,
    callback=validate_non_negative_arg,
    help="Max total items (only with --no-interactive).",
)
@click.option(
    "--page-size",
    required=False,
    default=None,
    type=int,
    help="Number of items per page (1-50). Only with --v2.",
    callback=validate_page_size,
)
@click.option(
    "--json",
    "-j",
    "json_output",
    is_flag=True,
    default=False,
    help="Output results as JSON. Only with --v2.",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    show_default=True,
    help="Enable interactive pagination. Only with --v2.",
)
@click.option(
    "--include-all-users/--only-mine",
    default=False,
    help="Include schedules from all users. Only with --v2.",
)
def list(  # noqa: A001 PLR0913
    v2: bool,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002
    project: Optional[str] = None,
    cloud: Optional[str] = None,
    creator_id: Optional[str] = None,
    max_items: Optional[int] = None,
    page_size: Optional[int] = None,
    json_output: bool = False,
    interactive: bool = True,
    include_all_users: bool = False,
) -> None:
    """List Schedules

    You can optionally filter schedules by name, project, cloud, or creator.
    """
    if v2:
        # Validate max_items only allowed with --no-interactive (v2 only)
        if max_items is not None and interactive:
            raise click.UsageError("--max-items only allowed with --no-interactive")
        # New SDK path with pagination and output options

        # Apply defaults for v2-only options
        effective_page_size = page_size if page_size is not None else 10

        # Compute effective max_items for non-interactive mode
        effective_max = max_items
        if not interactive and effective_max is None:
            effective_max = NON_INTERACTIVE_DEFAULT_MAX_ITEMS

        # Print diagnostics header (not in JSON mode)
        if not json_output:
            stderr = Console(stderr=True)
            _print_schedule_list_diagnostics(
                stderr=stderr,
                name=name,
                schedule_id=id,
                project=project,
                cloud=cloud,
                creator_id=creator_id,
                include_all_users=include_all_users,
                interactive=interactive,
                page_size=effective_page_size,
                effective_max=effective_max if not interactive else None,
            )

        iterator = anyscale.schedule.list(
            name=name,
            schedule_id=id,
            project=project,
            cloud=cloud,
            creator_id=creator_id,
            include_all_users=include_all_users,
            page_size=effective_page_size,
            max_items=effective_max if not interactive else None,
        )

        console = Console()
        total = display_list(
            iterator=iterator,
            item_formatter=_format_schedule_row_v2,
            table_creator=_create_schedules_table_v2,
            json_output=json_output,
            page_size=effective_page_size,
            interactive=interactive,
            max_items=effective_max if not interactive else None,
            console=console,
        )
        if not json_output:
            if total:
                stderr.print(f"\nFetched {total} schedule(s).")
            else:
                stderr.print("\nNo schedules found.")
    else:
        # Legacy path with deprecation warning
        # Check if v2-only options are being used without --v2
        if any(
            [
                project,
                cloud,
                creator_id,
                max_items is not None,
                page_size is not None,
                json_output,
                include_all_users,
                not interactive,
            ]
        ):
            click.echo(
                "ERROR: Options --project, --cloud, --creator-id, --max-items, "
                "--page-size, --json, --include-all-users, and --no-interactive require --v2 flag.\n"
                "Use: anyscale schedule list --v2 [options]",
                err=True,
            )
            raise click.exceptions.Exit(1)

        job_controller = ScheduleController()
        job_controller.list(name=name, id=id)


@schedule_cli.command(
    name="pause", cls=AnyscaleCommand, example=command_examples.SCHEDULE_PAUSE_EXAMPLE
)
@click.option(
    "--config-file",
    "-f",
    required=False,
    type=str,
    help="Path to a YAML config file to use for this schedule.",
)
@click.option(
    "--name", "-n", required=False, default=None, help="Name of the schedule."
)
@click.option("--id", "-i", required=False, default=None, help="Id of the schedule.")
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The named Anyscale Cloud for the schedule. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project to use for the schedule. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
)
def pause(
    config_file: str, name: str, cloud: str, project: str, id: str  # noqa: A002
) -> None:
    """Pause a Schedule.

    You can pause a schedule by config file, name, or id.

    To specify the schedule by name, use the --name flag. You can specify the cloud with --cloud and the project with --project.

    To specify the schedule by id, use the --id flag.

    To specify the schedule by config file, use --config-file. Ensure that name and optionally cloud and project are specified in the
    config file's job config.
    """
    _validate_schedule_identifiers(name=name, id=id, config_file=config_file)

    if id is not None:
        anyscale.schedule.set_state(id=id, state=ScheduleState.DISABLED)
    else:
        if config_file is not None:
            name, cloud, project = _read_identifiers_from_config_file(config_file)

        anyscale.schedule.set_state(
            name=name, cloud=cloud, project=project, state=ScheduleState.DISABLED
        )


@schedule_cli.command(
    name="resume", cls=AnyscaleCommand, example=command_examples.SCHEDULE_RESUME_EXAMPLE
)
@click.option(
    "--config-file",
    "-f",
    required=False,
    type=str,
    help="Path to a YAML config file to use for this schedule.",
)
@click.option(
    "--name", "-n", required=False, default=None, help="Name of the schedule."
)
@click.option("--id", "-i", required=False, default=None, help="Id of the schedule.")
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The named Anyscale Cloud for the schedule. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project to use for the schedule. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
)
def resume(
    config_file: str, name: str, cloud: str, project: str, id: str  # noqa: A002
) -> None:
    """ Resume a Schedule

    You can resume a schedule by config file, name, or id.

    To specify the schedule by name, use the --name flag. You can specify the cloud with --cloud and the project with --project.

    To specify the schedule by id, use the --id flag.

    To specify the schedule by config file, use --config-file. Ensure that name and optionally cloud and project are specified in the
    config file's job config.
    """
    _validate_schedule_identifiers(name=name, id=id, config_file=config_file)

    if id is not None:
        anyscale.schedule.set_state(id=id, state=ScheduleState.ENABLED)
    else:
        if config_file is not None:
            name, cloud, project = _read_identifiers_from_config_file(config_file)

        anyscale.schedule.set_state(
            name=name, cloud=cloud, project=project, state=ScheduleState.ENABLED
        )


@schedule_cli.command(
    name="status", cls=AnyscaleCommand, example=command_examples.SCHEDULE_STATUS_EXAMPLE
)
@click.option(
    "--config-file",
    "-f",
    required=False,
    type=str,
    help="Path to a YAML config file to use for this schedule.",
)
@click.option(
    "--name", "-n", required=False, default=None, help="Name of the schedule."
)
@click.option("--id", "-i", required=False, default=None, help="Id of the schedule.")
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The named Anyscale Cloud for the schedule. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project to use for the schedule. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
)
@click.option(
    "--json",
    "-j",
    is_flag=True,
    default=False,
    help="Output the status in a structured JSON format.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Include verbose details in the status.",
)
def status(
    config_file: str,
    name: str,
    cloud: str,
    project: str,
    id: str,  # noqa: A002
    json: bool,
    verbose: bool,
) -> None:
    """Query the status of a Schedule.

    You can query the status of a schedule by config file, name, or id.

    To specify the schedule by name, use the --name flag. You can specify the cloud with --cloud and the project with --project.

    To specify the schedule by id, use the --id flag.

    To specify the schedule by config file, use --config-file. Ensure that name and optionally cloud and project are specified in the
    config file's job config.
    """
    _validate_schedule_identifiers(name=name, id=id, config_file=config_file)

    if id is not None:
        status = anyscale.schedule.status(id=id)
    else:
        if config_file is not None:
            name, cloud, project = _read_identifiers_from_config_file(config_file)

        status = anyscale.schedule.status(name=name, cloud=cloud, project=project)

    status_dict = status.to_dict()

    if not verbose:
        status_dict.pop("config", None)

    if json:
        print(json_dumps(status_dict, indent=4, sort_keys=False))
    else:
        stream = StringIO()
        yaml.dump(status_dict, stream, sort_keys=False)
        print(stream.getvalue(), end="")


@schedule_cli.command(
    name="run", cls=AnyscaleCommand, example=command_examples.SCHEDULE_RUN_EXAMPLE
)
@click.option(
    "--config-file",
    "-f",
    required=False,
    type=str,
    help="Path to a YAML config file to use for this schedule.",
)
@click.option(
    "--name", "-n", required=False, default=None, help="Name of the schedule."
)
@click.option("--id", "-i", required=False, default=None, help="Id of the schedule.")
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The named Anyscale Cloud for the schedule. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project to use for the schedule. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
)
def trigger(
    config_file: str, name: str, id: str, cloud: str, project: str  # noqa: A002
) -> None:
    """ Manually run a Schedule

    This function takes an existing schedule and runs it now.
    You can specify the schedule by name or id.
    You can also pass in a YAML file as a convinience. This is equivalent to passing in the name specified in the YAML file.
    IMPORTANT: if you pass in a YAML definition that differs from the Schedule defition, the Schedule will NOT be updated.
    Please use the `anyscale schedule apply` command to update the configuration of your schedule
    or use the `anyscale job submit` command to submit a one off job that is not a part of a schedule.
    """

    _validate_schedule_identifiers(name=name, id=id, config_file=config_file)

    if id is not None:
        anyscale.schedule.trigger(id=id)
    else:
        if config_file is not None:
            name, cloud, project = _read_identifiers_from_config_file(config_file)

        anyscale.schedule.trigger(
            name=name, cloud=cloud, project=project,
        )


@schedule_cli.command(
    name="url", cls=AnyscaleCommand, example=command_examples.SCHEDULE_URL_EXAMPLE
)
@click.argument("schedule_config_file", required=False)
@click.option(
    "--name", "-n", required=False, default=None, help="Name of the schedule."
)
@click.option("--id", "-i", required=False, default=None, help="Id of the schedule.")
@click.option("--v2", is_flag=True, help="Use new SDK-based implementation")
@click.option("--cloud", help="Cloud name (required with --name in v2 mode)")
@click.option("--project", help="Project name (required with --name in v2 mode)")
def url(
    schedule_config_file: str,
    id: str,  # noqa: A002
    name: str,
    v2: bool,
    cloud: Optional[str],
    project: Optional[str],
) -> None:
    """Get a Schedule URL

    This function accepts 1 argument, a path to a YAML config file that defines this schedule.
    You can also specify the schedule by name or id.
    """
    if v2:
        result_url = anyscale.schedule.url(
            id=id, name=name, cloud=cloud, project=project,
        )
        click.echo(f"View your schedule at {result_url}")
    else:
        job_controller = ScheduleController()
        resolved_id = job_controller.resolve_file_name_or_id(
            schedule_config_file=schedule_config_file, id=id, name=name
        )
        job_controller.url(resolved_id)


def _validate_delete_identifiers(
    name: Optional[str],
    id: Optional[str],  # noqa: A002
    cloud: Optional[str],
    project: Optional[str],
):
    """Validate identifiers for the delete command.

    Either --id OR --name must be provided.
    When --name is used, --cloud and --project are also required.
    When --id is used, --cloud and --project cannot be used.
    """
    if name is None and id is None:
        raise click.ClickException("One of '--name' or '--id' must be provided.")

    if name is not None and id is not None:
        raise click.ClickException("Only one of '--name' or '--id' can be provided.")

    if id is not None and (cloud is not None or project is not None):
        raise click.ClickException(
            "'--cloud' and '--project' cannot be used with '--id'."
        )

    if name is not None and (cloud is None or project is None):
        raise click.ClickException(
            "'--cloud' and '--project' are required when using '--name'."
        )


@schedule_cli.command(
    name="delete", cls=AnyscaleCommand, example=command_examples.SCHEDULE_DELETE_EXAMPLE
)
@click.option(
    "--name", "-n", required=False, default=None, help="Name of the schedule."
)
@click.option("--id", "-i", required=False, default=None, help="Id of the schedule.")
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The named Anyscale Cloud for the schedule (required with --name).",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project for the schedule (required with --name).",
)
def delete(
    name: Optional[str],
    id: Optional[str],  # noqa: A002
    cloud: Optional[str],
    project: Optional[str],
) -> None:
    """Delete a Schedule.

    If the schedule is active, it will be automatically paused before deletion.
    The schedule must have no active triggered jobs.

    To specify the schedule by id, use the --id flag.

    To specify the schedule by name, use the --name flag along with --cloud and --project.
    """
    _validate_delete_identifiers(name=name, id=id, cloud=cloud, project=project)

    if id is not None:
        anyscale.schedule.delete(id=id)
    else:
        anyscale.schedule.delete(name=name, cloud=cloud, project=project)
