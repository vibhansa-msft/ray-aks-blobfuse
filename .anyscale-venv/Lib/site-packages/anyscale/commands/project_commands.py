from io import StringIO
from json import dumps as json_dumps
import sys
from typing import Dict, Optional, Tuple

import click
from rich.console import Console
from rich.table import Table
import yaml

import anyscale
from anyscale.cli_logger import BlockLogger
from anyscale.commands import command_examples
from anyscale.commands.list_util import (
    display_list,
    MAX_PAGE_SIZE,
    NON_INTERACTIVE_DEFAULT_MAX_ITEMS,
)
from anyscale.commands.util import AnyscaleCommand
from anyscale.project.models import (
    CreateProjectCollaborator,
    CreateProjectCollaborators,
    Project,
    ProjectMinimal,
    ProjectSortField,
    ProjectSortOrder,
)
from anyscale.util import (
    AnyscaleJSONEncoder,
    get_endpoint,
)


log = BlockLogger()


def _create_project_list_table(show_header: bool) -> Table:
    table = Table(show_header=show_header, expand=True)
    # NAME and ID: larger ratios, can wrap but never truncate
    table.add_column(
        "NAME", no_wrap=False, overflow="fold", ratio=3, min_width=15,
    )
    table.add_column(
        "ID", no_wrap=False, overflow="fold", ratio=2, min_width=12,
    )
    # all other columns will wrap as needed
    for heading in (
        "DESCRIPTION",
        "CREATED AT",
        "CREATOR",
        "PARENT CLOUD ID",
    ):
        table.add_column(
            heading, no_wrap=False, overflow="fold", ratio=1, min_width=8,
        )
    return table


def _format_project_output_data(project: Project) -> Dict[str, str]:
    return {
        "name": project.name,
        "id": project.id,
        "description": project.description,
        "created_at": project.created_at,
        "creator": str(project.creator_id or ""),
        "parent_cloud_id": str(project.parent_cloud_id or ""),
    }


def _parse_sort_option(
    sort: Optional[str],
) -> Tuple[Optional[ProjectSortField], ProjectSortOrder]:
    if not sort:
        return None, ProjectSortOrder.ASC

    # build case-insensitive map of allowed fields
    allowed = {f.value.lower(): f.value for f in ProjectSortField.__members__.values()}

    # detect leading '-' for descending
    if sort.startswith("-"):
        raw = sort[1:]
        order = ProjectSortOrder.DESC
    else:
        raw = sort
        order = ProjectSortOrder.ASC

    key = raw.lower()
    if key not in allowed:
        allowed_names = ", ".join(sorted(allowed.values()))
        raise click.BadParameter(
            f"Invalid sort field '{raw}'. Allowed fields: {allowed_names}"
        )

    return ProjectSortField(allowed[key]), order


@click.group(
    "project",
    short_help="Manage projects on Anyscale.",
    help="Manages projects on Anyscale. A project can be used to organize a collection of jobs.",
)
def project_cli() -> None:
    pass


@project_cli.command(
    name="get",
    help="Get details of a project.",
    cls=AnyscaleCommand,
    example=command_examples.PROJECT_GET_EXAMPLE,
)
@click.option(
    "--id", "-i", type=str, required=True, help="ID of the project.",
)
@click.option(
    "--json",
    "-j",
    is_flag=True,
    default=False,
    help="Output the details in a structured JSON format.",
)
def get(id: str, json: bool = False):  # noqa: A002
    try:
        project: Project = anyscale.project.get(project_id=id)
    except ValueError as e:
        log.error(f"Error getting project details: {e}")
        sys.exit(1)

    console = Console()
    if json:
        json_str = json_dumps(project.to_dict(), indent=2, cls=AnyscaleJSONEncoder)
        console.print_json(json=json_str)
    else:
        stream = StringIO()
        yaml.dump(project.to_dict(), stream, sort_keys=False)
        console.print(stream.getvalue(), end="")


@project_cli.command(
    name="list",
    help="List all projects with optional filters.",
    cls=AnyscaleCommand,
    example=command_examples.PROJECT_LIST_EXAMPLE,
)
@click.option(
    "--name", "-n", type=str, help="A string to filter projects by name.",
)
@click.option(
    "--creator", "-u", type=str, help="The ID of a creator to filter projects.",
)
@click.option(
    "--cloud", "-c", type=str, help="The ID of a parent cloud to filter projects.",
)
@click.option(
    "--include-defaults/--exclude-defaults",
    default=True,
    show_default=True,
    help="Whether to include default projects.",
)
@click.option(
    "--max-items", type=int, help="The maximum number of projects to return.",
)
@click.option(
    "--page-size",
    type=int,
    default=10,
    show_default=True,
    help="The number of projects to return per page.",
)
@click.option(
    "--sort",
    help=(
        "Sort by FIELD (prefix with '-' for desc). "
        f"Allowed: {', '.join(ProjectSortField.__members__.values())}"
    ),
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    show_default=True,
    help="Use interactive paging.",
)
@click.option(
    "--json",
    "-j",
    is_flag=True,
    default=False,
    help="Output the list in a structured JSON format.",
)
def list(  # noqa: A001, PLR0913
    *,
    name: Optional[str] = None,
    creator: Optional[str] = None,
    cloud: Optional[str] = None,
    include_defaults: bool = True,
    max_items: Optional[int] = None,
    page_size: Optional[int] = None,
    sort: Optional[str] = None,
    interactive: bool = True,
    json: bool = False,
):

    if max_items is not None and interactive:
        raise click.UsageError("--max-items only allowed with --no-interactive")

    sort_field, sort_order = _parse_sort_option(sort)

    # normalize max_items
    effective_max = max_items
    if not interactive and effective_max is None:
        stderr = Console(stderr=True)
        stderr.print(
            f"Defaulting to {NON_INTERACTIVE_DEFAULT_MAX_ITEMS} items in batch mode; "
            "use --max-items to override."
        )
        effective_max = NON_INTERACTIVE_DEFAULT_MAX_ITEMS

    console = Console()
    stderr = Console(stderr=True)

    # diagnostics
    stderr.print("[bold]Listing projects with:[/]")
    stderr.print(f"• name_contains      = {name or '<any>'}")
    stderr.print(f"• creator_id         = {creator or '<any>'}")
    stderr.print(f"• parent_cloud_id    = {cloud or '<any>'}")
    stderr.print(f"• include defaults   = {include_defaults}")
    stderr.print(f"• sort-field         = {sort_field or '<none>'}")
    stderr.print(f"• sort-order         = {sort_order or '<none>'}")
    stderr.print(f"• mode               = {'interactive' if interactive else 'batch'}")
    stderr.print(f"• per-page limit     = {page_size}")
    stderr.print(f"• max-items total    = {effective_max or 'all'}")
    stderr.print(f"\nView your Projects in the UI at {get_endpoint('/projects')}\n")

    # choose formatter
    if json:

        def formatter(project):
            return ProjectMinimal.from_dict(project.to_dict()).to_dict()

    else:
        formatter = _format_project_output_data

    total = 0
    try:
        iterator = anyscale.project.list(
            name_contains=name,
            creator_id=creator,
            parent_cloud_id=cloud,
            include_defaults=include_defaults,
            max_items=effective_max,
            page_size=page_size,
            sort_field=sort_field,
            sort_order=sort_order,
        )
        total = display_list(
            iterator=iter(iterator),
            item_formatter=formatter,
            table_creator=_create_project_list_table,
            json_output=json,
            page_size=page_size or MAX_PAGE_SIZE,
            interactive=interactive,
            max_items=effective_max,
            console=console,
        )

        if not json:
            if total > 0:
                stderr.print(f"\nFetched {total} projects.")
            else:
                stderr.print("\nNo projects found.")
    except Exception as e:  # noqa: BLE001
        log.error(f"Failed to list projects: {e}")
        sys.exit(1)


@project_cli.command(
    name="create",
    help="Create a new project.",
    cls=AnyscaleCommand,
    example=command_examples.PROJECT_CREATE_EXAMPLE,
)
@click.option(
    "--name", "-n", type=str, required=True, help="Name of the project.",
)
@click.option(
    "--cloud", "-c", type=str, required=True, help="Parent cloud ID for the project.",
)
@click.option(
    "--description", "-d", type=str, help="Description of the project.",
)
@click.option(
    "--initial-cluster-config",
    "-f",
    type=str,
    help="Initial cluster config for the project.",
)
def create(
    name: str,
    cloud: str,
    *,
    description: Optional[str] = None,
    initial_cluster_config: Optional[str] = None,
) -> None:
    try:
        project_id: str = anyscale.project.create(
            name,
            cloud,
            description=description or "",
            initial_cluster_config=initial_cluster_config,
        )
    except ValueError as e:
        log.error(f"Error creating project: {e}")
        sys.exit(1)

    log.info(f"Created project '{name}' with ID: {project_id}")


@project_cli.command(
    name="delete",
    help="Delete a project.",
    cls=AnyscaleCommand,
    example=command_examples.PROJECT_DELETE_EXAMPLE,
)
@click.option(
    "--id", "-i", type=str, required=True, help="ID of the project to delete.",
)
def delete(id: str):  # noqa: A002
    try:
        anyscale.project.delete(id)
    except ValueError as e:
        log.error(f"Error deleting project: {e}")
        sys.exit(1)

    log.info(f"Deleted project '{id}'")


@project_cli.command(
    name="get-default",
    help="Get the default project for a cloud.",
    cls=AnyscaleCommand,
    example=command_examples.PROJECT_GET_DEFAULT_EXAMPLE,
)
@click.option(
    "--cloud", "-c", type=str, required=True, help="Parent cloud ID for the project.",
)
@click.option(
    "--json",
    "-j",
    is_flag=True,
    default=False,
    help="Output the project in a structured JSON format.",
)
def get_default(cloud: str, json: bool = False):
    try:
        project: Project = anyscale.project.get_default(cloud)
    except ValueError as e:
        log.error(f"Error getting default project for cloud '{cloud}': {e}")
        sys.exit(1)

    console = Console()
    if json:
        json_str = json_dumps(project.to_dict(), indent=2, cls=AnyscaleJSONEncoder)
        console.print_json(json=json_str)
    else:
        stream = StringIO()
        yaml.dump(project.to_dict(), stream, sort_keys=False)
        console.print(stream.getvalue(), end="")


@project_cli.command(
    name="add-collaborators",
    help="Add collaborators to the project.",
    cls=AnyscaleCommand,
    example=command_examples.PROJECT_ADD_COLLABORATORS_EXAMPLE,
)
@click.option(
    "--cloud",
    "-c",
    help="Name of the cloud that the project belongs to.",
    required=True,
)
@click.option(
    "--project",
    "-p",
    help="Name of the project to add collaborators to.",
    required=True,
)
@click.option(
    "--users-file",
    help="Path to a YAML file containing a list of users to add to the project.",
    required=True,
)
def add_collaborators(cloud: str, project: str, users_file: str) -> None:
    collaborators = CreateProjectCollaborators.from_yaml(users_file)

    try:
        anyscale.project.add_collaborators(
            cloud=cloud,
            project=project,
            collaborators=[
                CreateProjectCollaborator(**collaborator)
                for collaborator in collaborators.collaborators
            ],
        )
    except ValueError as e:
        log.error(f"Error adding collaborators to project: {e}")
        return

    log.info(
        f"Successfully added {len(collaborators.collaborators)} collaborators to project {project}."
    )
