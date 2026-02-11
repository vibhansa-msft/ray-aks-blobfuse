from io import StringIO
import json
from typing import Dict, IO, Optional

import click
from rich.console import Console
from rich.table import Table
import yaml

import anyscale
from anyscale.authenticate import get_auth_api_client
from anyscale.commands import command_examples
from anyscale.commands.list_util import (
    display_list,
    MAX_PAGE_SIZE,
    NON_INTERACTIVE_DEFAULT_MAX_ITEMS,
    validate_page_size,
)
from anyscale.commands.util import AnyscaleCommand
from anyscale.image.models import ImageBuild
from anyscale.util import AnyscaleJSONEncoder, get_endpoint, validate_non_negative_arg


@click.group(
    "image", help="Manage images to define dependencies on Anyscale.",
)
def image_cli() -> None:
    pass


def validate_max_items(ctx, param, value):
    if value is None:
        return None
    return validate_non_negative_arg(ctx, param, value)


def _create_image_list_table(show_header: bool) -> Table:
    table = Table(show_header=show_header, expand=True)
    # Allow wrapping for all columns to prevent text cutoff
    table.add_column("NAME", no_wrap=False, overflow="fold", ratio=3, min_width=15)
    table.add_column(
        "LATEST VERSION", no_wrap=False, overflow="fold", ratio=2, min_width=14,
    )
    table.add_column(
        "LATEST URI", no_wrap=False, overflow="fold", ratio=4, min_width=20,
    )
    table.add_column(
        "CREATED BY", no_wrap=False, overflow="fold", ratio=3, min_width=24,
    )
    table.add_column(
        "CREATED AT", no_wrap=False, overflow="fold", ratio=2, min_width=20,
    )
    return table


def _create_image_list_table_verbose(show_header: bool) -> Table:
    table = Table(show_header=show_header, expand=True, box=None)
    table.add_column("NAME", overflow="fold", no_wrap=False)
    table.add_column("ID", overflow="fold", no_wrap=False)
    table.add_column("PROJECT", overflow="fold", no_wrap=False)
    table.add_column("VERSION", overflow="fold", no_wrap=False)
    table.add_column("BUILD ID", overflow="fold", no_wrap=False)
    table.add_column("STATUS", overflow="fold", no_wrap=False)
    table.add_column("ANON", overflow="fold", no_wrap=False)
    table.add_column("URI", overflow="fold", no_wrap=False)
    table.add_column("CREATED BY", overflow="fold", no_wrap=False)
    table.add_column("CREATED AT", overflow="fold", no_wrap=False)
    table.add_column("MODIFIED AT", overflow="fold", no_wrap=False)
    return table


def _format_image_output(image: ImageBuild) -> Dict[str, str]:
    created_at = image.created_at.strftime("%Y-%m-%d %H:%M") if image.created_at else ""
    latest_version = (
        "" if image.latest_build_revision is None else str(image.latest_build_revision)
    )

    return {
        "name": image.name,
        "latest_version": latest_version,
        "latest_uri": image.latest_image_uri or "",
        "created_by": image.creator_email or image.creator_id or "",
        "created_at": created_at,
    }


def _format_image_output_verbose(image: ImageBuild) -> Dict[str, str]:
    created_at = image.created_at.strftime("%Y-%m-%d %H:%M") if image.created_at else ""
    last_modified = (
        image.last_modified_at.strftime("%Y-%m-%d %H:%M")
        if image.last_modified_at
        else ""
    )
    latest_version = (
        "" if image.latest_build_revision is None else str(image.latest_build_revision)
    )
    build_status = str(image.latest_build_status) if image.latest_build_status else ""

    return {
        "name": image.name,
        "id": image.id,
        "project": image.project_id or "",
        "version": latest_version,
        "build_id": image.latest_build_id or "",
        "status": build_status,
        "anon": "Y" if image.is_anonymous else "N",
        "uri": image.latest_image_uri or "",
        "created_by": image.creator_email or image.creator_id or "",
        "created_at": created_at,
        "modified_at": last_modified,
    }


@image_cli.command(
    name="build",
    help=("Build an image from a Containerfile."),
    cls=AnyscaleCommand,
    example=command_examples.IMAGE_BUILD_EXAMPLE,
)
@click.option(
    "--containerfile",
    "-f",
    help="Path to the Containerfile.",
    type=click.File("rb"),
    required=True,
)
@click.option(
    "--name",
    "-n",
    help="Name for the image. If the image with the same name already exists, a new version will be built. Otherwise, a new image will be created.",
    required=True,
    type=str,
)
@click.option(
    "--ray-version",
    "-r",
    help="The Ray version (X.Y.Z) specified for this image specified by either an image URI or a containerfile. If you don't specify a Ray version, Anyscale defaults to the latest Ray version available at the time of the Anyscale CLI/SDK release.",
    type=str,
    default=None,
)
def build(
    containerfile: IO[bytes], name: str, ray_version: Optional[str] = None
) -> None:
    try:
        containerfile_str = containerfile.read().decode("utf-8")
        image_uri = anyscale.image.build(
            containerfile_str, name=name, ray_version=ray_version
        )
        print(f"Image built successfully with URI: {image_uri}")
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"Failed to build image: {e}") from None


@image_cli.command(
    name="get",
    help=("Get details of an image."),
    cls=AnyscaleCommand,
    example=command_examples.IMAGE_GET_EXAMPLE,
)
@click.option(
    "--name",
    "-n",
    help=(
        "Get the details of an image.\n\n"
        "The name can contain an optional version, e.g., 'name:version'. "
        "If no version is provided, the latest one will be used.\n\n"
    ),
    type=str,
    default=None,
    required=True,
)
@click.option(
    "-j", "--json", "json_output", is_flag=True, default=False, help="Output as JSON.",
)
@click.option(
    "--yaml", "yaml_output", is_flag=True, default=False, help="Output as YAML.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show all fields including IDs and metadata.",
)
def get(name: str, json_output: bool, yaml_output: bool, verbose: bool) -> None:
    try:
        image = anyscale.image.get(name=name)

        if json_output:
            print(json.dumps(image.to_dict(), indent=2, cls=AnyscaleJSONEncoder))
        elif yaml_output:
            stream = StringIO()
            yaml.safe_dump(image.to_dict(), stream, sort_keys=False)
            print(stream.getvalue(), end="")
        else:
            console = Console()

            if verbose:
                table = _create_image_list_table_verbose(show_header=True)
                formatted = _format_image_output_verbose(image)
                table.add_row(
                    formatted["name"],
                    formatted["id"],
                    formatted["project"],
                    formatted["version"],
                    formatted["build_id"],
                    formatted["status"],
                    formatted["anon"],
                    formatted["uri"],
                    formatted["created_by"],
                    formatted["created_at"],
                    formatted["modified_at"],
                )
            else:
                table = _create_image_list_table(show_header=True)
                formatted = _format_image_output(image)
                table.add_row(
                    formatted["name"],
                    formatted["latest_version"],
                    formatted["latest_uri"],
                    formatted["created_by"],
                    formatted["created_at"],
                )

            console.print(table)

    except ValueError as e:
        raise click.ClickException(str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"Failed to get image: {e}") from None


@image_cli.command(
    name="list",
    help="List images.",
    cls=AnyscaleCommand,
    example=command_examples.IMAGE_LIST_EXAMPLE,
)
@click.option("--image-id", "--id", help="ID of the image to display.")
@click.option("--name", "-n", help="Substring to match against the image name.")
@click.option(
    "--image-name",
    help="Substring to match against the resolved image URI (BYOD images only).",
)
@click.option("--project", help="Filter images by project name.")
@click.option(
    "--created-by-me",
    is_flag=True,
    default=False,
    help="List images created by me only.",
)
@click.option(
    "--include-archived",
    is_flag=True,
    default=False,
    help="Include archived images in the results.",
)
@click.option(
    "--include-anonymous",
    is_flag=True,
    default=False,
    help="Include anonymous (workspace-scoped) images.",
)
@click.option(
    "--max-items",
    type=int,
    callback=validate_max_items,
    help="Max total items (only with --no-interactive).",
)
@click.option(
    "--page-size",
    type=int,
    default=10,
    show_default=True,
    callback=validate_page_size,
    help=f"Items per page (max {MAX_PAGE_SIZE}).",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    show_default=True,
    help="Use interactive paging.",
)
@click.option(
    "-j",
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Emit structured JSON to stdout.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show all fields including IDs and metadata.",
)
def list(  # noqa: A001, PLR0913
    image_id: Optional[str],
    name: Optional[str],
    image_name: Optional[str],
    project: Optional[str],
    created_by_me: bool,
    include_archived: bool,
    include_anonymous: bool,
    max_items: Optional[int],
    page_size: int,
    interactive: bool,
    json_output: bool,
    verbose: bool,
) -> None:
    if max_items is not None and interactive:
        raise click.UsageError("--max-items only allowed with --no-interactive")

    stderr = Console(stderr=True)
    effective_max = max_items
    if not interactive and effective_max is None:
        stderr.print(
            f"Defaulting to {NON_INTERACTIVE_DEFAULT_MAX_ITEMS} items in batch mode; "
            "use --max-items to override."
        )
        effective_max = NON_INTERACTIVE_DEFAULT_MAX_ITEMS

    console = Console()
    creator_id = None
    if created_by_me:
        auth_block = get_auth_api_client()
        creator_id = auth_block.api_client.get_user_info_api_v2_userinfo_get().result.id

    stderr.print("[bold]Listing images with:[/]")
    stderr.print(f"• name             = {name or '<any>'}")
    stderr.print(f"• image_uri        = {image_name or '<any>'}")
    stderr.print(f"• project          = {project or '<any>'}")
    stderr.print(f"• created_by_me    = {created_by_me}")
    stderr.print(f"• include_archived = {include_archived}")
    stderr.print(f"• include_anonymous= {include_anonymous}")
    stderr.print(f"• mode             = {'interactive' if interactive else 'batch'}")
    stderr.print(f"• per-page limit   = {page_size}")
    stderr.print(f"• max-items total  = {effective_max or 'all'}")
    stderr.print(
        f"\nView your Images in the UI at {get_endpoint('/configurations/app-config-versions')}\n"
    )

    if json_output:

        def formatter(image):
            return image.to_dict()

        table_creator = _create_image_list_table
    elif verbose:
        formatter = _format_image_output_verbose
        table_creator = _create_image_list_table_verbose
    else:
        formatter = _format_image_output
        table_creator = _create_image_list_table

    try:
        iterator = anyscale.image.list(
            image_id=image_id,
            name=name,
            image_name=image_name,
            project=project,
            creator_id=creator_id,
            include_archived=include_archived,
            include_anonymous=include_anonymous,
            max_items=None if interactive else effective_max,
            page_size=page_size,
        )
        total = display_list(
            iterator=iter(iterator),
            item_formatter=formatter,
            table_creator=table_creator,
            json_output=json_output,
            page_size=page_size,
            interactive=interactive,
            max_items=effective_max,
            console=console,
        )

        if not json_output:
            if total > 0:
                stderr.print(f"\nFetched {total} images.")
            else:
                stderr.print("\nNo images found.")
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"Failed to list images: {e}") from None


@image_cli.command(
    name="register",
    help=("Register a custom container image with a container image name."),
    cls=AnyscaleCommand,
    example=command_examples.IMAGE_REGISTER_EXAMPLE,
)
@click.option(
    "--image-uri",
    help="The URI of the custom container image to register.",
    type=str,
    required=True,
)
@click.option(
    "--name",
    "-n",
    help="Name for the container image. If the name already exists, a new version will be built. Otherwise, a new container image will be created.",
    required=True,
    type=str,
)
@click.option(
    "--ray-version",
    "-r",
    help="The Ray version (X.Y.Z) specified for this image specified by either an image URI or a containerfile. If you don't specify a Ray version, Anyscale defaults to the latest Ray version available at the time of the Anyscale CLI/SDK release.",
    type=str,
    default=None,
)
@click.option(
    "--registry-login-secret",
    help="Name or identifier of the secret containing credentials to authenticate to the docker registry hosting the image.",
    type=str,
    default=None,
)
def register(
    image_uri: str,
    name: str,
    ray_version: Optional[str] = None,
    registry_login_secret: Optional[str] = None,
) -> None:
    try:
        built_image_uri = anyscale.image.register(
            image_uri=image_uri,
            registry_login_secret=registry_login_secret,
            ray_version=ray_version,
            name=name,
        )
        print(f"Image registered successfully with URI: {built_image_uri}")
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"Failed to register image: {e}") from None


@image_cli.command(
    name="archive",
    help="Archive an image and all of its versions.",
    cls=AnyscaleCommand,
    example=command_examples.IMAGE_ARCHIVE_EXAMPLE,
)
@click.option(
    "--name",
    "-n",
    help="Name of the image to archive. Can include an optional version tag (e.g., 'name:version').",
    type=str,
    required=True,
)
def archive(name: str) -> None:
    """Archive an image.

    Once archived, the image name will no longer be usable in the organization.
    Archived images can still be viewed using --include-archived in list.
    """
    try:
        anyscale.image.archive(name=name)
        print(f"Image '{name}' archived successfully.")
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"Failed to archive image: {e}") from None
