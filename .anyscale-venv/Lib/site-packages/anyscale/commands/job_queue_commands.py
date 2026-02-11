from __future__ import annotations

from datetime import datetime
from enum import Enum
from functools import partial
from json import dumps as json_dumps
import sys
from typing import Dict, get_type_hints, List, Optional, Tuple

import click
from rich.console import Console
from rich.table import Table

from anyscale.client.openapi_client.models.job_queue_sort_directive import (
    JobQueueSortDirective,
)
from anyscale.client.openapi_client.models.job_queue_sort_field import JobQueueSortField
from anyscale.client.openapi_client.models.session_state import SessionState
from anyscale.client.openapi_client.models.sort_order import SortOrder
from anyscale.commands import command_examples
from anyscale.commands.list_util import (
    display_list,
    MAX_PAGE_SIZE,
    NON_INTERACTIVE_DEFAULT_MAX_ITEMS,
    validate_page_size,
)
from anyscale.commands.util import (
    AnyscaleCommand,
    build_kv_table,
    parse_repeatable_tags_to_dict,
    parse_tags_kv_to_str_map,
)
import anyscale.job_queue
from anyscale.job_queue.models import JobQueueStatus, JobQueueStatusKeys
from anyscale.util import get_endpoint, get_user_info, validate_non_negative_arg


@click.group("job-queue", help="Manage Anyscale Job Queues.")
def job_queue_cli() -> None:
    pass


# TODO praneethkaturi: Implement job-queue create command
# - Requires design decision on how to handle compute config
# - Backend API exists: POST /api/v2/job-queues/create
# - SDK method needed: anyscale.job_queue.create()
# - CLI options: --name, --cloud, --project, --max-concurrency,
#   --idle-timeout-s, --execution-mode, --tag


class ViewOption(Enum):
    DEFAULT = "default"
    STATS = "stats"
    ALL = "all"


VIEW_COLUMNS: Dict[ViewOption, List[JobQueueStatusKeys]] = {
    ViewOption.DEFAULT: [
        "name",
        "id",
        "state",
        "creator_email",
        "project_id",
        "created_at",
    ],
    ViewOption.STATS: [
        "id",
        "name",
        "total_jobs",
        "active_jobs",
        "successful_jobs",
        "failed_jobs",
    ],
    ViewOption.ALL: [
        "name",
        "id",
        "state",
        "creator_email",
        "project_id",
        "created_at",
        "max_concurrency",
        "idle_timeout_s",
        "cloud_id",
        "user_provided_id",
        "execution_mode",
        "total_jobs",
        "active_jobs",
        "successful_jobs",
        "failed_jobs",
    ],
}


@job_queue_cli.command(
    name="list",
    help="List job queues.",
    cls=AnyscaleCommand,
    example=command_examples.JOB_QUEUE_LIST,
)
@click.option("--id", "job_queue_id", help="ID of a job queue.")
@click.option("--name", type=str, help="Filter by name.")
@click.option("--cloud", type=str, help="Filter by cloud.")
@click.option("--project", type=str, help="Filter by project.")
@click.option(
    "--include-all-users/--only-mine",
    default=False,
    help="Include job queues not created by current user.",
)
@click.option(
    "--cluster-status",
    type=click.Choice(SessionState.allowable_values, case_sensitive=False),
    help="Filter by cluster status.",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help=(
        "This option can be repeated to filter by multiple tags. "
        "Tags with the same key are ORed, whereas tags with different keys are ANDed. "
        "Example: --tag team:mlops --tag team:infra --tag env:prod. "
        "Filters with team: (mlops OR infra) AND env:prod."
    ),
)
@click.option(
    "--view",
    type=click.Choice([opt.value for opt in ViewOption], case_sensitive=False),
    default=ViewOption.DEFAULT.value,
    help="Columns view.",
    callback=lambda _ctx, _param, value: ViewOption(value),
)
@click.option(
    "--page-size",
    default=10,
    type=int,
    callback=validate_page_size,
    help=f"Items per page (max {MAX_PAGE_SIZE}).",
)
@click.option(
    "--max-items",
    type=int,
    callback=lambda ctx, param, value: validate_non_negative_arg(ctx, param, value)
    if value
    else None,
    help="Non-interactive max items.",
)
@click.option(
    "--sort",
    "sort_dirs",
    multiple=True,
    default=["-created_at"],
    help="Sort by FIELD (prefix with '-' for desc). Repeatable.",
    callback=lambda _ctx, _param, values: _parse_sort_fields("sort", list(values)),
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    show_default=True,
    help="Enable interactive pagination.",
)
@click.option(
    "-j", "--json", "json_output", is_flag=True, default=False, help="JSON output.",
)
@click.option(
    "--include-archived",
    is_flag=True,
    default=False,
    help="Include archived job queues when searching by name. Ignored when using --id.",
)
def list_job_queues(  # noqa: PLR0913
    job_queue_id: Optional[str],
    name: Optional[str],
    cloud: Optional[str],
    project: Optional[str],
    cluster_status: Optional[str],
    tags: List[str],
    include_all_users: bool,
    view: ViewOption,
    page_size: int,
    max_items: Optional[int],
    sort_dirs: List[JobQueueSortDirective],
    interactive: bool,
    json_output: bool,
    include_archived: bool,
) -> None:
    """List and page job queues according to filters and view."""
    if max_items and interactive:
        raise click.UsageError("--max-items only in non-interactive mode")

    # Compute effective max_items for non-interactive mode only
    effective_max = max_items
    if not interactive and effective_max is None:
        effective_max = NON_INTERACTIVE_DEFAULT_MAX_ITEMS

    console = Console()
    stderr = Console(stderr=True)

    if not json_output:
        _print_list_diagnostics(
            stderr=stderr,
            job_queue_id=job_queue_id,
            name=name,
            include_all_users=include_all_users,
            cloud=cloud,
            project=project,
            cluster_status=cluster_status,
            view=view,
            sort_dirs=sort_dirs,
            interactive=interactive,
            page_size=page_size,
            effective_max=effective_max if not interactive else None,
            include_archived=include_archived,
        )

    try:
        user = get_user_info()
        iterator = anyscale.job_queue.list(
            job_queue_id=job_queue_id,
            name=name,
            creator_id=None if include_all_users else (user.id if user else None),
            cloud=cloud,
            project=project,
            tags_filter=parse_repeatable_tags_to_dict(tags) if tags else None,
            page_size=page_size,
            max_items=None if interactive else effective_max,
            sorting_directives=sort_dirs,
            include_archived=include_archived,
        )
        cols = VIEW_COLUMNS[view]
        table_fn = partial(_create_table, view)

        def row_fn(jq: JobQueueStatus) -> Dict[str, str]:
            data = _format_data(jq)
            return {c: data[c] for c in cols}

        total = display_list(
            iterator=iter(iterator),
            item_formatter=row_fn,
            table_creator=table_fn,
            json_output=json_output,
            page_size=page_size,
            interactive=interactive,
            max_items=effective_max,
            console=console,
        )
        if not json_output:
            if total:
                stderr.print(f"\nFetched {total} job-queue(s).")
            else:
                stderr.print("\nNo job-queues found.")

    except Exception as e:  # noqa: BLE001
        stderr.print(f"Error: {e}", style="red")
        sys.exit(1)


@job_queue_cli.command(
    name="update",
    help="Update job queue settings.",
    cls=AnyscaleCommand,
    example=command_examples.JOB_QUEUE_UPDATE,
)
@click.option("--id", "job_queue_id", help="ID of the job queue.")
@click.option("--name", "-n", help="Name of the job queue.")
@click.option("--project", "-p", help="Project name (required when using --name).")
@click.option("--cloud", help="Cloud name (required when using --name).")
@click.option("--max-concurrency", type=int, help="Max number of concurrent jobs.")
@click.option("--idle-timeout-s", type=int, help="Idle timeout in seconds.")
@click.option(
    "--json", "json_output", is_flag=True, default=False, help="JSON output.",
)
def update_job_queue(
    job_queue_id: Optional[str],
    name: Optional[str],
    project: Optional[str],
    cloud: Optional[str],
    max_concurrency: Optional[int],
    idle_timeout_s: Optional[int],
    json_output: bool,
) -> None:
    """Update the max_concurrency or idle_timeout_s of a job queue."""
    if not job_queue_id and not name:
        raise click.ClickException("Provide either --id or --name.")
    if max_concurrency is None and idle_timeout_s is None:
        raise click.ClickException("Specify --max-concurrency or --idle-timeout-s")
    stderr = Console(stderr=True)
    ident = job_queue_id or name or "<unknown>"
    if not json_output:
        stderr.print(f"Updating job queue '{ident}'...")
    try:
        jq = anyscale.job_queue.update(
            job_queue_id=job_queue_id,
            job_queue_name=name,
            project=project,
            cloud=cloud,
            max_concurrency=max_concurrency,
            idle_timeout_s=idle_timeout_s,
        )
        if json_output:
            Console().print_json(json_dumps(_format_data(jq), indent=2))
        else:
            _display_single(jq, stderr, ViewOption.ALL)
    except Exception as e:  # noqa: BLE001
        stderr.print(f"Update failed: {e}", style="red")
        sys.exit(1)


@job_queue_cli.group("tags", help="Manage tags for job queues.")
def job_queue_tags_cli() -> None:
    pass


@job_queue_tags_cli.command(
    name="add",
    help="Add or update tags on a job queue.",
    cls=AnyscaleCommand,
    example=command_examples.JOB_QUEUE_TAGS_ADD_EXAMPLE,
)
@click.option("--id", "job_queue_id", help="ID of a job queue.")
@click.option("--name", "-n", type=str, help="Name of a job queue.")
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Tag in key=value (or key:value) format. Repeat to add multiple.",
)
def add_tags(
    job_queue_id: Optional[str], name: Optional[str], tags: Tuple[str],
) -> None:
    if not job_queue_id and not name:
        raise click.ClickException("Provide either --id or --name.")
    tag_map = parse_tags_kv_to_str_map(tags)
    if not tag_map:
        raise click.ClickException("Provide at least one --tag key=value.")
    anyscale.job_queue.add_tags(job_queue_id=job_queue_id, name=name, tags=tag_map)
    stderr = Console(stderr=True)
    ident = job_queue_id or name or "<unknown>"
    stderr.print(f"Tags updated for job queue '{ident}'.")


@job_queue_tags_cli.command(
    name="remove",
    help="Remove tags by key from a job queue.",
    cls=AnyscaleCommand,
    example=command_examples.JOB_QUEUE_TAGS_REMOVE_EXAMPLE,
)
@click.option("--id", "job_queue_id", help="ID of a job queue.")
@click.option("--name", "-n", type=str, help="Name of a job queue.")
@click.option("--key", "keys", multiple=True, help="Tag key to remove. Repeatable.")
def remove_tags(
    job_queue_id: Optional[str], name: Optional[str], keys: Tuple[str],
) -> None:
    if not job_queue_id and not name:
        raise click.ClickException("Provide either --id or --name.")
    key_list = [k for k in keys if k and k.strip()]
    if not key_list:
        raise click.ClickException("Provide at least one --key to remove.")
    anyscale.job_queue.remove_tags(job_queue_id=job_queue_id, name=name, keys=key_list)
    stderr = Console(stderr=True)
    ident = job_queue_id or name or "<unknown>"
    stderr.print(f"Removed tag keys {key_list} from job queue '{ident}'.")


@job_queue_tags_cli.command(
    name="list",
    help="List tags for a job queue.",
    cls=AnyscaleCommand,
    example=command_examples.JOB_QUEUE_TAGS_LIST_EXAMPLE,
)
@click.option("--id", "job_queue_id", help="ID of a job queue.")
@click.option("--name", "-n", type=str, help="Name of a job queue.")
@click.option("--json", "json_output", is_flag=True, default=False, help="JSON output.")
def list_tags(
    job_queue_id: Optional[str], name: Optional[str], json_output: bool,
) -> None:
    if not job_queue_id and not name:
        raise click.ClickException("Provide either --id or --name.")
    tag_map = anyscale.job_queue.list_tags(job_queue_id=job_queue_id, name=name)
    if json_output:
        Console().print_json(json=json_dumps(tag_map, indent=2))
    else:
        stderr = Console(stderr=True)
        if not tag_map:
            stderr.print("No tags found.")
            return
        pairs = tag_map.items()
        stderr.print(build_kv_table(pairs, title="Tags"))


@job_queue_cli.command(
    name="status",
    help="Show job queue details.",
    cls=AnyscaleCommand,
    example=command_examples.JOB_QUEUE_STATUS,
)
@click.option("--id", "job_queue_id", help="ID of the job queue.")
@click.option("--name", "-n", help="Name of the job queue.")
@click.option("--project", "-p", help="Project name to filter by when using --name.")
@click.option("--cloud", help="Cloud name to filter by when using --name.")
@click.option(
    "--view",
    type=click.Choice([opt.value for opt in ViewOption], case_sensitive=False),
    default=ViewOption.DEFAULT.value,
    help="Columns view.",
    callback=lambda _ctx, _param, value: ViewOption(value),
)
@click.option(
    "--json", "json_output", is_flag=True, default=False, help="JSON output.",
)
@click.option(
    "--include-archived",
    is_flag=True,
    default=False,
    help="Include archived job queues when searching by name. Ignored when using --id.",
)
def status(
    job_queue_id: Optional[str],
    name: Optional[str],
    project: Optional[str],
    cloud: Optional[str],
    view: ViewOption,
    json_output: bool,
    include_archived: bool,
) -> None:
    """Fetch and display a single job queue's details."""
    if not job_queue_id and not name:
        raise click.ClickException("Provide either --id or --name.")
    if job_queue_id and (project or cloud):
        raise click.ClickException(
            "--project and --cloud should only be used with --name, not --id."
        )

    stderr = Console(stderr=True)
    ident = job_queue_id or name or "<unknown>"
    if not json_output:
        stderr.print(f"Fetching job queue '{ident}'...")
    try:
        jq = anyscale.job_queue.status(
            job_queue_id=job_queue_id,
            name=name,
            project=project,
            cloud=cloud,
            include_archived=include_archived,
        )
        if json_output:
            Console().print_json(json_dumps(_format_data(jq), indent=2))
        else:
            # Use ALL view for single item display if not specified
            display_view = ViewOption.ALL if view == ViewOption.DEFAULT else view
            _display_single(jq, stderr, display_view)
    except Exception as e:  # noqa: BLE001
        stderr.print(f"Failed: {e}", style="red")
        sys.exit(1)


@job_queue_cli.command(
    name="archive",
    help="Archive (seal) a job queue. No new jobs can be submitted.",
    cls=AnyscaleCommand,
    example=command_examples.JOB_QUEUE_ARCHIVE_EXAMPLE,
)
@click.option("--id", "job_queue_id", help="ID of the job queue.")
@click.option("--name", "-n", help="Name of the job queue.")
@click.option("--project", help="Project name (required with --name).")
@click.option("--cloud", help="Cloud name (required with --name).")
def archive_job_queue(
    job_queue_id: Optional[str],
    name: Optional[str],
    project: Optional[str],
    cloud: Optional[str],
) -> None:
    """Archive a job queue to prevent new job submissions."""
    if not job_queue_id and not name:
        raise click.ClickException("Provide either --id or --name.")
    if name and not project:
        raise click.ClickException("--project is required when using --name.")
    if name and not cloud:
        raise click.ClickException("--cloud is required when using --name.")
    if job_queue_id and (project or cloud):
        raise click.ClickException(
            "--project and --cloud should only be used with --name, not --id."
        )

    stderr = Console(stderr=True)
    ident = job_queue_id or name or "<unknown>"
    stderr.print(f"Archiving job queue '{ident}'...")

    try:
        jq_id = anyscale.job_queue.archive(
            job_queue_id=job_queue_id, name=name, project=project, cloud=cloud,
        )
        stderr.print(f"Job queue '{ident}' has been archived.")
        stderr.print(f"Query the status with `anyscale job-queue status --id {jq_id}`.")
    except Exception as e:  # noqa: BLE001
        stderr.print(f"Archive failed: {e}", style="red")
        sys.exit(1)


@job_queue_cli.command(
    name="terminate",
    help="Terminate a job queue and all its pending/running jobs.",
    cls=AnyscaleCommand,
    example=command_examples.JOB_QUEUE_TERMINATE_EXAMPLE,
)
@click.option("--id", "job_queue_id", help="ID of the job queue.")
@click.option("--name", "-n", help="Name of the job queue.")
@click.option("--project", help="Project name (required with --name).")
@click.option("--cloud", help="Cloud name (required with --name).")
@click.option(
    "--include-archived",
    is_flag=True,
    default=False,
    help="Include archived job queues when searching by name. Ignored when using --id.",
)
def terminate_job_queue(
    job_queue_id: Optional[str],
    name: Optional[str],
    project: Optional[str],
    cloud: Optional[str],
    include_archived: bool,
) -> None:
    """Terminate a job queue and all its jobs."""
    if not job_queue_id and not name:
        raise click.ClickException("Provide either --id or --name.")
    if name and not project:
        raise click.ClickException("--project is required when using --name.")
    if name and not cloud:
        raise click.ClickException("--cloud is required when using --name.")
    if job_queue_id and (project or cloud):
        raise click.ClickException(
            "--project and --cloud should only be used with --name, not --id."
        )

    stderr = Console(stderr=True)
    ident = job_queue_id or name or "<unknown>"
    stderr.print(f"Terminating job queue '{ident}'...")

    try:
        jq_id = anyscale.job_queue.terminate(
            job_queue_id=job_queue_id,
            name=name,
            project=project,
            cloud=cloud,
            include_archived=include_archived,
        )
        stderr.print(f"Job queue '{ident}' has been marked for termination.")
        stderr.print(f"Query the status with `anyscale job-queue status --id {jq_id}`.")
    except Exception as e:  # noqa: BLE001
        stderr.print(f"Terminate failed: {e}", style="red")
        sys.exit(1)


@job_queue_cli.command(
    name="delete",
    help="Delete a job queue. Jobs previously submitted remain accessible.",
    cls=AnyscaleCommand,
    example=command_examples.JOB_QUEUE_DELETE_EXAMPLE,
)
@click.option("--id", "job_queue_id", help="ID of the job queue.")
@click.option("--name", "-n", help="Name of the job queue.")
@click.option("--project", help="Project name (required with --name).")
@click.option("--cloud", help="Cloud name (required with --name).")
@click.option(
    "--include-archived",
    is_flag=True,
    default=False,
    help="Include archived job queues when searching by name. Ignored when using --id.",
)
def delete_job_queue(
    job_queue_id: Optional[str],
    name: Optional[str],
    project: Optional[str],
    cloud: Optional[str],
    include_archived: bool,
) -> None:
    """Delete a job queue.

    Jobs previously submitted to the queue remain accessible.
    The job queue must have all jobs in terminal state and no running clusters.
    This action cannot be undone.
    """
    if not job_queue_id and not name:
        raise click.ClickException("Provide either --id or --name.")
    if name and not project:
        raise click.ClickException("--project is required when using --name.")
    if name and not cloud:
        raise click.ClickException("--cloud is required when using --name.")
    if job_queue_id and (project or cloud):
        raise click.ClickException(
            "--project and --cloud should only be used with --name, not --id."
        )

    stderr = Console(stderr=True)
    ident = job_queue_id or name or "<unknown>"
    stderr.print(f"Deleting job queue '{ident}'...")

    try:
        anyscale.job_queue.delete(
            job_queue_id=job_queue_id,
            name=name,
            project=project,
            cloud=cloud,
            include_archived=include_archived,
        )
        stderr.print(f"Job queue '{ident}' has been deleted.")
    except Exception as e:  # noqa: BLE001
        stderr.print(f"Delete failed: {e}", style="red")
        sys.exit(1)


def _parse_sort_fields(
    param: str, sort_fields: List[str],
) -> List[JobQueueSortDirective]:
    """Convert a list of string fields into JobQueueSortDirective objects."""
    directives: List[JobQueueSortDirective] = []
    opts = ", ".join(v.lower() for v in JobQueueSortField.allowable_values)
    for field_str in sort_fields:
        desc = field_str.startswith("-")
        raw = field_str.lstrip("-").upper()
        if raw not in JobQueueSortField.allowable_values:
            raise click.UsageError(f"{param} must be one of {opts}")
        directives.append(
            JobQueueSortDirective(
                sort_field=raw, sort_order=SortOrder.DESC if desc else SortOrder.ASC,
            )
        )
    return directives


def _create_table(view: ViewOption, show_header: bool) -> Table:
    """Create a Rich Table with columns based on the selected view."""
    table = Table(show_header=show_header, expand=True)
    for key in VIEW_COLUMNS[view]:
        table.add_column(key.replace("_", " ").upper(), overflow="fold")
    return table


def _format_data(jq: JobQueueStatus) -> Dict[str, str]:
    """Format a JobQueueStatus object into a dictionary of strings for display."""
    data = {}
    for key in get_type_hints(JobQueueStatus):
        value = getattr(jq, key, None)
        if isinstance(value, datetime):
            data[key] = value.strftime("%Y-%m-%d %H:%M:%S") if value else ""
        elif value is None:
            data[key] = ""
        else:
            data[key] = str(value)
    return data


def _display_single(jq: JobQueueStatus, stderr: Console, view: ViewOption,) -> None:
    """Display a single job queue's details in a table using the selected view.

    Args:
        jq: The JobQueueStatus object to display.
        stderr: The Rich Console object to print to.
        view: The ViewOption determining which columns to display.
    """
    table = _create_table(view, show_header=True)
    data = _format_data(jq)
    table.add_row(*(data[col] for col in VIEW_COLUMNS[view]))
    stderr.print(table)


def _print_list_diagnostics(  # noqa: PLR0913
    stderr: Console,
    job_queue_id: Optional[str],
    name: Optional[str],
    include_all_users: bool,
    cloud: Optional[str],
    project: Optional[str],
    cluster_status: Optional[str],
    view: ViewOption,
    sort_dirs: List[JobQueueSortDirective],
    interactive: bool,
    page_size: int,
    effective_max: Optional[int],
    include_archived: bool = False,
) -> None:
    """Prints diagnostic information for the list_job_queues command."""
    formatted_sort_dirs = [
        f"{'-' if d.sort_order == SortOrder.DESC else ''}{(d.sort_field or '').lower()}"
        for d in sort_dirs
    ]

    stderr.print("[bold]Listing job-queues with:[/]")
    stderr.print(f"• id              = {job_queue_id or '<any>'}")
    stderr.print(f"• name            = {name or '<any>'}")
    stderr.print(f"• include_archived= {include_archived}")
    stderr.print(f"• creator         = {'all' if include_all_users else 'mine'}")
    stderr.print(f"• cloud           = {cloud or '<any>'}")
    stderr.print(f"• project         = {project or '<any>'}")
    stderr.print(f"• cluster         = {cluster_status or '<any>'}")
    stderr.print(f"• view            = {view.value}")
    stderr.print(f"• sort            = {formatted_sort_dirs or '<none>'}")
    stderr.print(f"• mode            = {'interactive' if interactive else 'batch'}")
    stderr.print(f"• per-page limit  = {page_size}")
    stderr.print(f"• max-items total = {effective_max if effective_max else 'all'}")
    stderr.print(f"\nView your Job Queues in the UI at {get_endpoint('/job-queues')}\n")
