from io import StringIO
from json import dumps as json_dumps
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.table import Table
import yaml

from anyscale._private.models.image_uri import ImageURI
from anyscale.cli_logger import BlockLogger
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
    convert_kv_strings_to_dict,
    override_env_vars,
    parse_repeatable_tags_to_dict,
    parse_tags_kv_to_str_map,
)
from anyscale.controllers.service_controller import ServiceController
import anyscale.service
from anyscale.service.models import (
    ServiceConfig,
    ServiceLogMode,
    ServiceSortField,
    ServiceSortOrder,
    ServiceState,
    ServiceStatus,
    ServiceVersionStatus,
)
from anyscale.util import (
    AnyscaleJSONEncoder,
    get_endpoint,
    validate_non_negative_arg,
    validate_service_state_filter,
)


log = BlockLogger()  # CLI Logger


@click.group("service")
def service_cli():
    pass


def _read_name_from_config_file(path: str):
    """Read the 'name' property from the config file at `path`.

    This enables reading the name for both the existing (rollout) and new (deploy)
    config file formats.
    """
    if not pathlib.Path(path).is_file():
        raise click.ClickException(f"Config file not found at path: '{path}'.")

    with open(path) as f:
        config = yaml.safe_load(f)

    if config is None or "name" not in config:
        raise click.ClickException(f"No 'name' property found in config file '{path}'.")

    return config["name"]


@service_cli.command(
    name="deploy", help="Deploy or update a service.",
)
@click.argument("import_path", type=str, required=False, default=None)
@click.argument("arguments", nargs=-1, required=False)
@click.option(
    "-f",
    "--config-file",
    required=False,
    default=[],
    type=str,
    multiple=True,
    help="Path to a YAML config file to deploy. When deploying from a file, import path and arguments cannot be provided. Command-line flags will overwrite values read from the file.",
)
@click.option(
    "-n",
    "--name",
    required=False,
    default=None,
    type=str,
    help="Unique name for the service. When running in a workspace, this defaults to the workspace name.",
)
@click.option(
    "--image-uri",
    required=False,
    default=None,
    type=str,
    help="Container image to use for the service. This cannot be changed when using --in-place and is exclusive with --containerfile. When running in a workspace, this defaults to the image of the workspace.",
)
@click.option(
    "--registry-login-secret",
    required=False,
    default=None,
    type=str,
    help="Name or identifier of the secret containing credentials to authenticate to the docker registry hosting the image. "
    "This can only be used when 'image_uri' is specified and the image is not hosted on Anyscale.",
)
@click.option(
    "--containerfile",
    required=False,
    default=None,
    type=str,
    help="Path to a containerfile to build the image to use for the service. This cannot be changed when using --in-place and is exclusive with --image-uri.",
)
@click.option(
    "--ray-version",
    required=False,
    default=None,
    type=str,
    help="The Ray version (X.Y.Z) to the image specified by --image-uri. This is only used when --image-uri is provided. If you don't specify a Ray version, Anyscale defaults to the latest Ray version available at the time of the Anyscale CLI/SDK release.",
)
@click.option(
    "--compute-config",
    required=False,
    default=None,
    type=str,
    help="Named compute configuration to use for the service. This cannot be changed when using --in-place. When running in a workspace, this defaults to the compute configuration of the workspace.",
)
@click.option(
    "-w",
    "--working-dir",
    required=False,
    default=None,
    type=str,
    help="Path to a local directory or a remote URI to a .zip file (S3, GS, HTTP) that will be the working directory for the service. The files in the directory will be automatically uploaded to cloud storage. When running in a workspace, this defaults to the current working directory.",
)
@click.option(
    "-e",
    "--exclude",
    required=False,
    type=str,
    multiple=True,
    help="File pattern to exclude when uploading local directories. This argument can be specified multiple times and the patterns will be appended to the 'excludes' list in the config file (if any).",
)
@click.option(
    "-r",
    "--requirements",
    required=False,
    default=None,
    type=str,
    help="Path to a requirements.txt file containing dependencies for the service. Anyscale installs these dependencies on top of the image. If you deploy a service from a workspace, the default is to use the workspace dependencies, but specifying this option overrides them.",
)
@click.option(
    "-i",
    "--in-place",
    is_flag=True,
    show_default=True,
    default=False,
    help="Perform an in-place upgrade without starting a new cluster. This can be used for faster iteration during development but is *not* currently recommended for production deploys. This *cannot* be used to change cluster-level options such as image and compute config (they will be ignored).",
)
@click.option(
    "--canary-percent",
    required=False,
    default=None,
    type=int,
    help="The percentage of traffic to send to the canary version of the service (0-100). This can be used to manually shift traffic toward (or away from) the canary version. If not provided, traffic will be shifted incrementally toward the canary version until it reaches 100. Not supported when using --in-place. This is ignored when restarting a service or creating a new service.",
)
@click.option(
    "--max-surge-percent",
    required=False,
    default=None,
    type=int,
    help="Amount of excess capacity allowed to be used while updating the service (0-100). Defaults to 100. Not supported when using --in-place.",
)
@click.option(
    "--env",
    required=False,
    multiple=True,
    type=str,
    help="Environment variables to set for the service. The format is 'key=value'. This argument can be specified multiple times. When the same key is also specified in the config file, the value from the command-line flag will overwrite the value from the config file.",
)
@click.option(
    "--py-module",
    required=False,
    default=None,
    multiple=True,
    type=str,
    help="Python modules to be available for import in the Ray workers. Each entry must be a path to a local directory.",
)
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
)
@click.option(
    "--versions",
    required=False,
    default=None,
    type=str,
    help="Defines the traffic and capacity percents per version. Capacity defaults to traffic.",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Tag in key=value (or key:value) format. Repeat to add multiple.",
)
@click.option(
    "--version-name",
    required=False,
    default=None,
    type=str,
    help="Unique name for the service version. This can only be used for single version deployments. For multi-version deployments, specify version names in the config files and --versions.",
)
def deploy(  # noqa: PLR0912, PLR0913 C901
    config_file: List[str],
    import_path: Optional[str],
    arguments: Tuple[str],
    name: Optional[str],
    image_uri: Optional[str],
    registry_login_secret: Optional[str],
    ray_version: Optional[str],
    containerfile: Optional[str],
    compute_config: Optional[str],
    working_dir: Optional[str],
    exclude: Tuple[str],
    requirements: Optional[str],
    in_place: bool,
    canary_percent: Optional[int],
    max_surge_percent: Optional[int],
    env: Optional[Tuple[str]],
    py_module: Tuple[str],
    cloud: Optional[str],
    project: Optional[str],
    versions: Optional[str],
    tags: Optional[Tuple[str]],
    version_name: Optional[str],
):
    """Deploy or update a service.

    If no service with the provided name is running, one will be created, else the existing service will be updated.

    To deploy a single application, the import path and optional arguments can be passed directly as arguments:

    `$ anyscale service deploy main:app arg1=val1`

    To deploy multiple applications, deploy from a config file using `-f`:

    `$ anyscale service deploy -f config.yaml`

    Command-line flags override values in the config file.
    """

    if versions is None:
        if len(config_file) == 1:
            if import_path is not None or len(arguments) > 0:
                raise click.ClickException(
                    "When a config file is provided, import path and application arguments can't be."
                )

            if not pathlib.Path(config_file[0]).is_file():
                raise click.ClickException(f"Config file '{config_file[0]}' not found.")

            config = ServiceConfig.from_yaml(config_file[0])
        elif len(config_file) > 1:
            raise click.ClickException(
                "Multiple config files can be provided only when deploying multiple versions with --versions."
            )
        else:
            # when config_file is not provided.
            if import_path is None:
                raise click.ClickException(
                    "Either config file or import path must be provided."
                )

            if (
                import_path.endswith((".yaml", ".yml"))
                or pathlib.Path(import_path).is_file()
            ):
                log.warning(
                    f"The provided import path '{import_path}' looks like a config file. Did you mean to use '-f config.yaml'?"
                )

            app: Dict[str, Any] = {"import_path": import_path}
            arguments_dict = convert_kv_strings_to_dict(arguments)
            if arguments_dict:
                app["args"] = arguments_dict

            config = ServiceConfig(applications=[app])

        if containerfile and image_uri:
            raise click.ClickException(
                "Only one of '--containerfile' and '--image-uri' can be provided."
            )

        if ray_version and (not image_uri and not containerfile):
            raise click.ClickException(
                "Ray version can only be used with an image or containerfile.",
            )

        if registry_login_secret and (
            not image_uri or ImageURI.from_str(image_uri).is_cluster_env_image()
        ):
            raise click.ClickException(
                "Registry login secret can only be used with an image that is not hosted on Anyscale."
            )

        if name is not None:
            config = config.options(name=name)

        if image_uri is not None:
            config = config.options(image_uri=image_uri)

        if registry_login_secret is not None:
            config = config.options(registry_login_secret=registry_login_secret)

        if ray_version is not None:
            config = config.options(ray_version=ray_version)

        if containerfile is not None:
            config = config.options(containerfile=containerfile)

        if compute_config is not None:
            config = config.options(compute_config=compute_config)

        if working_dir is not None:
            config = config.options(working_dir=working_dir)

        if exclude:
            config = config.options(excludes=[e for e in exclude])

        if requirements is not None:
            config = config.options(requirements=requirements)

        if env:
            config = override_env_vars(config, convert_kv_strings_to_dict(env))

        if py_module:
            for module in py_module:
                if not pathlib.Path(module).is_dir():
                    raise click.ClickException(
                        f"Python module path '{module}' does not exist or is not a directory."
                    )
            config = config.options(py_modules=[*py_module])

        if cloud is not None:
            config = config.options(cloud=cloud)
        if project is not None:
            config = config.options(project=project)

        if version_name is not None:
            config = config.options(version_name=version_name)

        configs = config
    else:
        # When multiple versions are being deployed.
        configs = []
        for config_path in config_file:
            svc_config = ServiceConfig.from_yaml(config_path)
            if name is not None:
                svc_config = svc_config.options(name=name)

            if cloud is not None:
                svc_config = svc_config.options(cloud=cloud)

            if project is not None:
                svc_config = svc_config.options(project=project)

            if image_uri is not None:
                log.warning("--image-uri is ignored.")

            if registry_login_secret is not None:
                log.warning("--registry-login-secret is ignored.")

            if ray_version is not None:
                log.warning("--ray-version is ignored.")

            if containerfile is not None:
                log.warning("--containerfile is ignored.")

            if compute_config is not None:
                log.warning("--compute-config is ignored.")

            if working_dir is not None:
                log.warning("--working-dir is ignored.")

            if exclude:
                log.warning("--exclude is ignored.")

            if requirements is not None:
                log.warning("--requirements is ignored.")

            if env:
                log.warning("--env is ignored.")

            if py_module:
                log.warning("--py-module is ignored.")

            if version_name is not None:
                log.warning("--version-name is ignored.")

            configs.append(svc_config)

    if tags:
        tag_map = parse_tags_kv_to_str_map(tags)
        if tag_map:
            if isinstance(configs, ServiceConfig):
                configs = configs.options(tags=tag_map)
            else:
                configs = [cfg.options(tags=tag_map) for cfg in configs]

    anyscale.service.deploy(
        configs,
        in_place=in_place,
        canary_percent=canary_percent,
        max_surge_percent=max_surge_percent,
        versions=versions,
        name=name,
        cloud=cloud,
        project=project,
    )


@service_cli.command(
    name="status", help="Get the status of a service.",
)
@click.option(
    "-n", "--name", required=False, default=None, type=str, help="Name of the service.",
)
@click.option(
    "-f",
    "--config-file",
    required=False,
    default=None,
    type=str,
    help="Path to a YAML config file to read the name from.",
)
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
)
@click.option(
    "-j",
    "--json",
    is_flag=True,
    default=False,
    help="Output the status in a structured JSON format.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Include verbose details in the status.",
)
def status(
    name: Optional[str],
    config_file: Optional[str],
    cloud: Optional[str],
    project: Optional[str],
    json: bool,
    verbose: bool,
):
    if name is not None and config_file is not None:
        raise click.ClickException(
            "Only one of '--name' and '--config-file' can be provided."
        )

    if config_file is not None:
        name = _read_name_from_config_file(config_file)

    if name is None:
        raise click.ClickException(
            "Service name must be provided using '--name' or in a config file using '-f'."
        )

    status: ServiceStatus = anyscale.service.status(
        name=name, cloud=cloud, project=project
    )
    status_dict = status.to_dict()
    if not verbose:
        # TODO(edoakes): consider adding this as an API on the model itself if it
        # becomes a common pattern.
        status_dict.get("primary_version", {}).pop("config", None)
        status_dict.get("canary_version", {}).pop("config", None)
        # Remove config from all versions in multi-version services
        if status_dict.get("versions"):
            for version in status_dict.get("versions", []):
                version.pop("config", None)

    console = Console()
    if json:
        json_str = json_dumps(status_dict, indent=2, cls=AnyscaleJSONEncoder)
        console.print_json(json=json_str)
    else:
        stream = StringIO()
        yaml.dump(status_dict, stream, sort_keys=False)
        console.print(stream.getvalue(), end="")


@service_cli.command(
    name="wait", help="Wait for a service to enter a target state.",
)
@click.option(
    "-n", "--name", required=False, default=None, type=str, help="Name of the service.",
)
@click.option(
    "-f",
    "--config-file",
    required=False,
    default=None,
    type=str,
    help="Path to a YAML config file to read the name from.",
)
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
)
@click.option(
    "-s",
    "--state",
    default=str(ServiceState.RUNNING),
    help="The ServiceState to wait for the service to reach. Defaults to RUNNING.",
)
@click.option(
    "-t",
    "--timeout-s",
    default=600,
    type=float,
    help="Timeout to wait for the service to reach the target state. Defaults to 600s (10min).",
)
def wait(
    name: Optional[str],
    config_file: Optional[str],
    cloud: Optional[str],
    project: Optional[str],
    state: str,
    timeout_s: int,
):
    try:
        state = ServiceState.validate(state)
    except ValueError as e:
        raise click.ClickException(str(e))

    if name is not None and config_file is not None:
        raise click.ClickException(
            "Only one of '--name' and '--config-file' can be provided."
        )

    if config_file is not None:
        name = _read_name_from_config_file(config_file)

    if name is None:
        raise click.ClickException(
            "Service name must be provided using '--name' or in a config file using '-f'."
        )

    try:
        anyscale.service.wait(
            name=name, cloud=cloud, project=project, state=state, timeout_s=timeout_s
        )
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from None


# This is a private CLI command to be used internally for testing. This is HIDDEN
# from the user and is not documented in the CLI help.
@service_cli.command(
    name="controller-logs", help="View the controller logs of a service.", hidden=True,
)
@click.option(
    "-n", "--name", required=True, type=str, help="Name of the service.",
)
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The Anyscale Cloud of this workload. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace).",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
)
@click.option(
    "--canary",
    required=False,
    default=False,
    type=bool,
    is_flag=True,
    help="Whether to show the logs of the canary version of the service. If not provided, the primary version logs will be shown.",
)
@click.option(
    "--head",
    required=False,
    default=False,
    type=bool,
    is_flag=True,
    help="Used with --max-lines to get `max-lines` lines from the head of the log.",
)
@click.option(
    "--max-lines",
    required=False,
    default=1000,
    type=int,
    help="Used with --head to limit the number of lines output.",
)
def controller_logs(
    name: str,
    cloud: Optional[str],
    project: Optional[str],
    canary: bool,
    head: bool,
    max_lines: int = 1000,
):
    mode = ServiceLogMode.TAIL
    if head:
        mode = ServiceLogMode.HEAD

    try:
        logs = anyscale.service._controller_logs(  # noqa: SLF001
            name=name,
            cloud=cloud,
            project=project,
            canary=canary,
            mode=mode,
            max_lines=max_lines,
        )
        print(logs)
    except ValueError as e:
        raise click.ClickException(str(e))


def validate_max_items(ctx, param, value):
    if value is None:
        return None
    return validate_non_negative_arg(ctx, param, value)


def _parse_sort_option(sort: Optional[str],) -> Tuple[Optional[str], ServiceSortOrder]:
    """
    Given a raw sort string (e.g. "-created_at"), return
    (canonical_field_name, SortOrder).
    """
    if not sort:
        return None, ServiceSortOrder.ASC

    # build case-insensitive map of allowed fields
    allowed = {f.value.lower(): f.value for f in ServiceSortField.__members__.values()}

    # detect leading '-' for descending
    if sort.startswith("-"):
        raw = sort[1:]
        order = ServiceSortOrder.DESC
    else:
        raw = sort
        order = ServiceSortOrder.ASC

    key = raw.lower()
    if key not in allowed:
        allowed_names = ", ".join(sorted(allowed.values()))
        raise click.BadParameter(
            f"Invalid sort field '{raw}'. Allowed fields: {allowed_names}"
        )

    return allowed[key], order


def _create_service_list_table(show_header: bool) -> Table:
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
        "CURRENT STATE",
        "CREATOR",
        "PROJECT",
        "LAST DEPLOYED AT",
    ):
        table.add_column(
            heading, no_wrap=False, overflow="fold", ratio=1, min_width=8,
        )

    return table


def _format_service_output_data(svc: ServiceStatus) -> Dict[str, str]:
    last_deployed_at = ""
    if isinstance(svc.primary_version, ServiceVersionStatus):
        last_deployed_at = svc.primary_version.created_at.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "name": svc.name,
        "id": svc.id,
        "current_state": str(svc.state),
        "creator": str(svc.creator or ""),
        "project": str(svc.project or ""),
        "last_deployed_at": last_deployed_at,
    }


@service_cli.command(
    name="list",
    help="List services.",
    cls=AnyscaleCommand,
    example=command_examples.SERVICE_LIST_EXAMPLE,
)
@click.option("--service-id", "--id", help="ID of the service to display.")
@click.option("--name", "-n", help="Name of the service to display.")
@click.option(
    "--cloud",
    type=str,
    help="The Anyscale Cloud of this workload; defaults to your org/workspace cloud.",
)
@click.option(
    "--project",
    type=str,
    help="Named project to use; defaults to your org/workspace project.",
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
    "--created-by-me",
    is_flag=True,
    default=False,
    help="List services created by me only.",
)
@click.option(
    "--state",
    "-s",
    "state_filter",
    multiple=True,
    callback=validate_service_state_filter,
    help=(
        "Filter by service state (repeatable). "
        f"Allowed: {', '.join(s.value for s in ServiceState)}"
    ),
)
@click.option(
    "--include-archived",
    is_flag=True,
    default=False,
    help="Include archived services.",
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
    "--sort",
    help=(
        "Sort by FIELD (prefix with '-' for desc). "
        f"Allowed: {', '.join(f.value for f in ServiceSortField.__members__.values())}"
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Include full config in JSON output.",
)
@click.option(
    "-j",
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Emit structured JSON to stdout.",
)
def list(  # noqa: PLR0913, A001
    service_id: Optional[str],
    name: Optional[str],
    tags: List[str],
    created_by_me: bool,
    cloud: Optional[str],
    project: Optional[str],
    state_filter: List[str],
    include_archived: bool,
    max_items: Optional[int],
    page_size: int,
    sort: Optional[str],
    json_output: bool,
    interactive: bool,
    verbose: bool,
):
    """List services based on the provided filters."""
    if max_items is not None and interactive:
        raise click.UsageError("--max-items only allowed with --no-interactive")

    # parse sort
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
    stderr.print("[bold]Listing services with:[/]")
    stderr.print(f"• name            = {name or '<any>'}")
    stderr.print(f"• states          = {', '.join(state_filter) or '<all>'}")
    stderr.print(f"• tags            = {', '.join(tags) or '<none>'}")
    stderr.print(f"• created_by_me   = {created_by_me}")
    stderr.print(f"• include_archived= {include_archived}")
    stderr.print(f"• sort            = {sort or '<none>'}")
    stderr.print(f"• mode            = {'interactive' if interactive else 'batch'}")
    stderr.print(f"• per-page limit  = {page_size}")
    stderr.print(f"• max-items total = {effective_max or 'all'}")
    stderr.print(f"\nView your Services in the UI at {get_endpoint('/services')}\n")

    creator_id = (
        ServiceController().get_authenticated_user_id() if created_by_me else None
    )

    # choose formatter
    if json_output:

        def json_formatter(svc: ServiceStatus) -> Dict[str, Any]:
            data = svc.to_dict()
            if not verbose:
                data.get("primary_version", {}).pop("config", None)
                data.get("canary_version", {}).pop("config", None)
            return data

        formatter = json_formatter
    else:
        formatter = _format_service_output_data

    total = 0
    try:
        iterator = anyscale.service.list(
            service_id=service_id,
            name=name,
            state_filter=state_filter,
            tags_filter=parse_repeatable_tags_to_dict(tags) if tags else None,
            creator_id=creator_id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
            max_items=None if interactive else effective_max,
            page_size=page_size,
            sort_field=sort_field,
            sort_order=sort_order,
        )
        total = display_list(
            iterator=iter(iterator),
            item_formatter=formatter,
            table_creator=_create_service_list_table,
            json_output=json_output,
            page_size=page_size,
            interactive=interactive,
            max_items=effective_max,
            console=console,
        )

        if not json_output:
            if total > 0:
                stderr.print(f"\nFetched {total} services.")
            else:
                stderr.print("\nNo services found.")
    except Exception as e:  # noqa: BLE001
        log.error(f"Failed to list services: {e}")
        sys.exit(1)


@service_cli.group("tags", help="Manage tags for services.")
def service_tags_cli() -> None:
    pass


@service_tags_cli.command(
    name="add",
    help="Add or update tags on a service.",
    cls=AnyscaleCommand,
    example=command_examples.SERVICE_TAGS_ADD_EXAMPLE,
)
@click.option("--service-id", "--id", help="ID of the service.")
@click.option("--name", "-n", help="Name of the service.")
@click.option("--cloud", type=str, help="Cloud name (for name resolution).")
@click.option("--project", type=str, help="Project name (for name resolution).")
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Tag in key=value (or key:value) format. Repeat to add multiple.",
)
def add_tags(
    service_id: Optional[str],
    name: Optional[str],
    cloud: Optional[str],
    project: Optional[str],
    tags: Tuple[str],
) -> None:
    if not service_id and not name:
        raise click.ClickException("Provide either --service-id/--id or --name.")
    tag_map = parse_tags_kv_to_str_map(tags)
    if not tag_map:
        raise click.ClickException("Provide at least one --tag key=value.")
    anyscale.service.add_tags(
        id=service_id, name=name, cloud=cloud, project=project, tags=tag_map
    )
    stderr = Console(stderr=True)
    ident = service_id or name or "<unknown>"
    stderr.print(f"Tags updated for service '{ident}'.")


@service_tags_cli.command(
    name="remove",
    help="Remove tags by key from a service.",
    cls=AnyscaleCommand,
    example=command_examples.SERVICE_TAGS_REMOVE_EXAMPLE,
)
@click.option("--service-id", "--id", help="ID of the service.")
@click.option("--name", "-n", help="Name of the service.")
@click.option("--cloud", type=str, help="Cloud name (for name resolution).")
@click.option("--project", type=str, help="Project name (for name resolution).")
@click.option("--key", "keys", multiple=True, help="Tag key to remove. Repeatable.")
def remove_tags(
    service_id: Optional[str],
    name: Optional[str],
    cloud: Optional[str],
    project: Optional[str],
    keys: Tuple[str],
) -> None:
    if not service_id and not name:
        raise click.ClickException("Provide either --service-id/--id or --name.")
    key_list = [k for k in keys if k and k.strip()]
    if not key_list:
        raise click.ClickException("Provide at least one --key to remove.")
    anyscale.service.remove_tags(
        id=service_id, name=name, cloud=cloud, project=project, keys=key_list
    )
    stderr = Console(stderr=True)
    ident = service_id or name or "<unknown>"
    stderr.print(f"Removed tag keys {key_list} from service '{ident}'.")


@service_tags_cli.command(
    name="list",
    help="List tags for a service.",
    cls=AnyscaleCommand,
    example=command_examples.SERVICE_TAGS_LIST_EXAMPLE,
)
@click.option("--service-id", "--id", help="ID of the service.")
@click.option("--name", "-n", help="Name of the service.")
@click.option("--cloud", type=str, help="Cloud name (for name resolution).")
@click.option("--project", type=str, help="Project name (for name resolution).")
@click.option("--json", "json_output", is_flag=True, default=False, help="JSON output.")
def list_tags(
    service_id: Optional[str],
    name: Optional[str],
    cloud: Optional[str],
    project: Optional[str],
    json_output: bool,
) -> None:
    if not service_id and not name:
        raise click.ClickException("Provide either --service-id/--id or --name.")
    tag_map = anyscale.service.list_tags(
        id=service_id, name=name, cloud=cloud, project=project
    )
    if json_output:
        Console().print_json(json=json_dumps(tag_map, indent=2))
    else:
        stderr = Console(stderr=True)
        if not tag_map:
            stderr.print("No tags found.")
            return
        pairs = tag_map.items()
        stderr.print(build_kv_table(pairs, title="Tags"))


# TODO(mowen): Add cloud support for this when we refactor to new SDK method
@service_cli.command(name="rollback", help="Roll back a service.")
@click.option(
    "--service-id", "--id", default=None, help="ID of service.",
)
@click.option("-n", "--name", required=False, default=None, help="Name of service.")
@click.option("--project-id", required=False, help="Filter by project id.")
@click.option(
    "--max-surge-percent",
    required=False,
    default=None,
    type=int,
    help="Max amount of excess capacity allocated during the rollback (0-100).",
)
def rollback(
    service_id: Optional[str],
    name: Optional[str],
    project_id: Optional[str],
    max_surge_percent: Optional[int],
):
    """Perform a rollback for a service that is currently in a rollout."""
    service_controller = ServiceController()
    service_id = service_controller.get_service_id(
        service_id=service_id, service_name=name, project_id=project_id,
    )
    service_controller.rollback(service_id, max_surge_percent)


@service_cli.command(name="terminate", help="Terminate a service.")
@click.option(
    "--service-id", "--id", required=False, help="ID of service.",
)
@click.option("-n", "--name", required=False, help="Name of service.")
@click.option("--project-id", required=False, help="Filter by project id.")
@click.option(
    "-f",
    "--config-file",
    "--service-config-file",
    help="Path to a YAML config file to read the name from. `--service-config-file` is deprecated, use `-f` or `--config-file`.",
)
def terminate(
    service_id: Optional[str],
    name: Optional[str],
    project_id: Optional[str],
    config_file: Optional[str],
):
    """Terminate a service.

    This applies to both v1 and v2 services.
    """
    # TODO: Remove service_controller and use the sdk method. Need to update the sdk method
    # so that it can resolve either service name or config file to the service id.
    service_controller = ServiceController()
    service_id = service_controller.get_service_id(
        service_id=service_id,
        service_name=name,
        service_config_file=config_file,
        project_id=project_id,
    )
    try:
        anyscale.service.terminate(id=service_id)
        log.info(f"Service {service_id} terminate initiated.")
        log.info(
            f"View the service in the UI at {get_endpoint(f'/services/{service_id}')}"
        )
    except Exception as e:  # noqa: BLE001
        log.error(f"Error terminating service: {e}")


@service_cli.command(
    name="archive",
    help="Archive a service.",
    cls=AnyscaleCommand,
    example=command_examples.SERVICE_ARCHIVE_EXAMPLE,
)
@click.option(
    "--service-id", "--id", required=False, help="ID of service.",
)
@click.option(
    "-n", "--name", required=False, default=None, type=str, help="Name of the service.",
)
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The Anyscale Cloud of this workload. If not provided, the organization default will be used.",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
)
def archive(
    service_id: Optional[str],
    name: Optional[str],
    cloud: Optional[str],
    project: Optional[str],
) -> None:
    """Archive a service."""

    if (service_id and name) or (not service_id and not name):
        log.error("Exactly one of '--id/--service-id' or '--name' must be provided.")
        sys.exit(1)

    identifier = name if name else service_id
    try:
        anyscale.service.archive(id=service_id, name=name, cloud=cloud, project=project)
        log.info(f"Successfully archived service: {identifier}")
    except Exception as e:  # noqa: BLE001
        log.error(f"Error archiving service: {e}")


@service_cli.command(
    name="delete",
    help="Delete a service.",
    cls=AnyscaleCommand,
    example=command_examples.SERVICE_DELETE_EXAMPLE,
)
@click.option(
    "--service-id", "--id", required=False, help="ID of service.",
)
@click.option(
    "-n", "--name", required=False, default=None, type=str, help="Name of the service.",
)
@click.option(
    "--cloud",
    required=False,
    default=None,
    type=str,
    help="The Anyscale Cloud of this workload. If not provided, the organization default will be used.",
)
@click.option(
    "--project",
    required=False,
    default=None,
    type=str,
    help="Named project to use for the service. If not provided, the default project for the cloud will be used (or, if running in a workspace, the project of the workspace).",
)
def delete(
    service_id: Optional[str],
    name: Optional[str],
    cloud: Optional[str],
    project: Optional[str],
) -> None:
    """Delete a service"""

    if (service_id and name) or (not service_id and not name):
        log.error("Exactly one of '--id/--service-id' or '--name' must be provided.")
        sys.exit(1)

    identifier = name if name else service_id
    try:
        anyscale.service.delete(id=service_id, name=name, cloud=cloud, project=project)
        log.info(f"Successfully deleted service: {identifier}")
    except Exception as e:  # noqa: BLE001
        log.error(f"Error deleting service: {e}")
