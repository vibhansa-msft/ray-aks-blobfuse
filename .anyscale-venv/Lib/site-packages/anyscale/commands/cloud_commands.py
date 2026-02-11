from io import StringIO
import pathlib
import re
import secrets
from typing import Any, Dict, List, Optional

import boto3
import click
from rich.console import Console
from rich.table import Table
import yaml

import anyscale
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models import (
    AWSConfig,
    AzureConfig,
    CloudDeployment,
    CloudProviders,
    ClusterManagementStackVersions,
    FileStorage,
    GCPConfig,
    KubernetesConfig,
    NetworkingMode,
    NFSMountTarget,
    ObjectStorage,
)
from anyscale.client.openapi_client.models.compute_stack import ComputeStack
from anyscale.cloud.models import CreateCloudCollaborator, CreateCloudCollaborators
from anyscale.cloud_utils import get_cloud_id_and_name, get_organization_id
from anyscale.commands import command_examples
from anyscale.commands.list_util import (
    display_list,
    MAX_PAGE_SIZE,
    NON_INTERACTIVE_DEFAULT_MAX_ITEMS,
    validate_page_size,
)
from anyscale.commands.setup_k8s import (
    setup_kubernetes_cloud,
    setup_kubernetes_cloud_resource,
)
from anyscale.commands.util import AnyscaleCommand, OptionPromptNull
from anyscale.controllers.cloud_controller import CloudController
from anyscale.util import (
    allow_optional_file_storage,
    get_endpoint,
    SharedStorageType,
    validate_non_negative_arg,
)
from anyscale.utils.cloud_utils import validate_aws_credentials
from anyscale.utils.imports.gcp import (
    try_import_gcp_managed_setup_utils,
    try_import_gcp_utils,
)


log = BlockLogger()  # CLI Logger


def setup_vm_cloud_resource(  # noqa: PLR0912, PLR0913
    provider: str,
    region: str,
    cloud_name: Optional[str],
    cloud_id: Optional[str],
    project_id: Optional[str],
    enable_head_node_fault_tolerance: bool,
    shared_storage: SharedStorageType,
    controller: Optional[CloudController] = None,
    boto3_session: Optional[Any] = None,
    anyscale_iam_role_name: Optional[str] = None,
    cluster_node_iam_role_name: Optional[str] = None,
    anyscale_access_service_account: Optional[str] = None,
    pool_name: Optional[str] = None,
) -> None:
    """Set up VM cloud resources for an existing Anyscale cloud."""
    if not cloud_name and not cloud_id:
        raise click.ClickException("Either --cloud or --cloud-id is required.")

    if controller is None:
        controller = CloudController()

    resolved_cloud_id, resolved_cloud_name = get_cloud_id_and_name(
        api_client=controller.api_client, cloud_id=cloud_id, cloud_name=cloud_name,
    )

    controller.log.info(
        f"Adding VM resources to cloud '{resolved_cloud_name}' ({resolved_cloud_id})"
    )

    if provider == "aws":
        if boto3_session is None:
            boto3_session = boto3.Session(region_name=region)
        if not validate_aws_credentials(controller.log, boto3_session):
            raise click.ClickException(
                "Cloud setup requires valid AWS credentials to be set locally. "
                "Learn more: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html"
            )

        resource_id = f"vm-{region}-{secrets.token_hex(4)}"
        if anyscale_iam_role_name is None:
            anyscale_iam_role_name = f"anyscale-iam-role-{secrets.token_hex(4)}"
        if cluster_node_iam_role_name is None:
            cluster_node_iam_role_name = f"{resource_id}-cluster_node_role"

        try:
            anyscale_aws_account = (
                controller.api_client.get_anyscale_aws_account_api_v2_clouds_anyscale_aws_account_get().result.anyscale_aws_account
            )
            cfn_stack = controller.run_cloudformation(
                region=region,
                cloud_id=resolved_cloud_id,
                anyscale_iam_role_name=anyscale_iam_role_name,
                cluster_node_iam_role_name=cluster_node_iam_role_name,
                enable_head_node_fault_tolerance=enable_head_node_fault_tolerance,
                anyscale_aws_account=anyscale_aws_account,
                boto3_session=boto3_session,
                shared_storage=shared_storage,
                resource_id=resource_id,
            )
            controller.update_cloud_with_resources(
                cfn_stack, resolved_cloud_id, region, enable_head_node_fault_tolerance
            )
            controller.wait_for_cloud_to_be_active(
                resolved_cloud_id, CloudProviders.AWS
            )

            controller.log.info(
                f"Successfully added VM resources to cloud '{resolved_cloud_name}'."
            )

        except Exception as e:  # noqa: BLE001
            controller.log.error(str(e))
            raise click.ClickException(f"Failed to add VM resources: {e}")

    elif provider == "gcp":
        if not project_id:
            raise click.ClickException("--project-id is required for GCP clouds.")

        gcp_utils = try_import_gcp_utils()
        setup_utils = try_import_gcp_managed_setup_utils()
        factory = gcp_utils.get_google_cloud_client_factory(controller.log, project_id)

        try:
            organization_id = get_organization_id(controller.api_client)
            anyscale_aws_account = (
                controller.api_client.get_anyscale_aws_account_api_v2_clouds_anyscale_aws_account_get().result.anyscale_aws_account
            )

            setup_utils.enable_project_apis(
                factory, project_id, controller.log, enable_head_node_fault_tolerance
            )

            token = secrets.token_hex(4)
            if anyscale_access_service_account is None:
                anyscale_access_service_account = (
                    f"anyscale-access-{token}@{project_id}.iam.gserviceaccount.com"
                )
            pool_id = f"anyscale-provider-pool-{token}"
            if pool_name is None:
                pool_name = f"projects/{project_id}/locations/global/workloadIdentityPools/{pool_id}"
            deployment_name = f"{resolved_cloud_id}-{token}".replace("_", "-").lower()

            setup_utils.create_workload_identity_pool(
                factory, project_id, pool_id, controller.log
            )
            controller.create_workload_identity_federation_provider(
                factory, project_id, pool_id, anyscale_access_service_account
            )

            controller.run_deployment_manager(
                factory,
                deployment_name,
                resolved_cloud_id,
                project_id,
                region,
                anyscale_access_service_account,
                pool_name,
                anyscale_aws_account,
                organization_id,
                enable_head_node_fault_tolerance,
                shared_storage=shared_storage,
            )
            with controller.log.spinner(
                "Updating Anyscale cloud with cloud resources..."
            ):
                controller.update_cloud_with_resources_gcp(
                    factory,
                    deployment_name,
                    resolved_cloud_id,
                    region,
                    project_id,
                    anyscale_access_service_account,
                    provider_name=f"{pool_name}/providers/anyscale-access",
                )
            controller.wait_for_cloud_to_be_active(
                resolved_cloud_id, CloudProviders.GCP
            )

            controller.log.info(
                f"Successfully added VM resources to cloud '{resolved_cloud_name}'."
            )

        except Exception as e:  # noqa: BLE001
            controller.log.error(str(e))
            raise click.ClickException(f"Failed to add VM resources: {e}")

    else:
        raise click.ClickException(f"Unsupported provider: {provider}")


@click.group(
    "cloud",
    short_help="Configure cloud provider authentication for Anyscale.",
    help="""Configure cloud provider authentication and setup
to allow Anyscale to launch instances in your account.""",
)
def cloud_cli() -> None:
    pass


def _create_cloud_list_table(show_header: bool) -> Table:
    table = Table(show_header=show_header, expand=True)
    table.add_column("NAME", no_wrap=False, overflow="fold", ratio=3, min_width=15)
    table.add_column("ID", no_wrap=False, overflow="fold", ratio=2, min_width=12)
    for heading in ("PROVIDER", "REGION", "DEFAULT", "CREATED AT"):
        table.add_column(heading, no_wrap=False, overflow="fold", ratio=1, min_width=8)
    return table


def _format_cloud_output_data(cloud: Any) -> Dict[str, str]:
    created_at = ""
    if getattr(cloud, "created_at", None):
        created_at = cloud.created_at.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "name": getattr(cloud, "name", ""),
        "id": getattr(cloud, "id", ""),
        "provider": str(getattr(cloud, "provider", "") or ""),
        "region": str(getattr(cloud, "region", "") or ""),
        "default": str(getattr(cloud, "is_default", False)),
        "created_at": created_at,
    }


@cloud_cli.command(name="delete", help="Delete a cloud.")
@click.argument("cloud-name", required=False)
@click.option("--name", "-n", help="Delete cloud by name.", type=str)
@click.option(
    "--cloud-id",
    "--id",
    help="Cloud id to delete. Alternative to cloud name.",
    required=False,
)
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Don't ask for confirmation."
)
def cloud_delete(
    cloud_name: Optional[str], name: Optional[str], cloud_id: Optional[str], yes: bool
) -> None:
    if cloud_name and name and cloud_name != name:
        raise click.ClickException(
            "The positional argument CLOUD_NAME and the keyword argument --name "
            "were both provided. Please only provide one of these two arguments."
        )
    CloudController().delete_cloud(
        cloud_name=cloud_name or name, cloud_id=cloud_id, skip_confirmation=yes
    )


@cloud_cli.command(
    name="set-default",
    help=(
        "Sets default cloud for your organization. This operation can only be performed "
        "by organization admins, and the default cloud must have organization level "
        "permissions."
    ),
)
@click.argument("cloud-name", required=False)
@click.option("--name", "-n", help="Set cloud as default by name.", type=str)
@click.option(
    "--cloud-id",
    "--id",
    help="Cloud id to set as default. Alternative to cloud name.",
    required=False,
)
def cloud_set_default(
    cloud_name: Optional[str], name: Optional[str], cloud_id: Optional[str]
) -> None:
    if cloud_name and name and cloud_name != name:
        raise click.ClickException(
            "The positional argument CLOUD_NAME and the keyword argument --name "
            "were both provided. Please only provide one of these two arguments."
        )
    CloudController().set_default_cloud(
        cloud_name=cloud_name or name, cloud_id=cloud_id
    )


def default_region(provider: str) -> str:
    default_regions = {
        "aws": "us-west-2",
        "gcp": "us-west1",
        "azure": "westus2",
    }
    return default_regions.get(provider, "default")


@cloud_cli.command(name="setup", help="Set up a cloud provider.")
@click.option(
    "--provider",
    help="The cloud provider type.",
    required=False,
    type=click.Choice(["aws", "gcp", "azure"], case_sensitive=False),
)
@click.option(
    "--region",
    cls=OptionPromptNull,
    help="Region to set up the credentials in.",
    required=False,
    default_option="provider",
    default=default_region,
    show_default=True,
)
@click.option("--name", "-n", help="Name of the cloud.", required=True, prompt="Name")
@click.option(
    "--stack",
    help="The compute stack to use (vm or k8s).",
    required=False,
    type=click.Choice(["vm", "k8s"], case_sensitive=False),
    default="vm",
    show_default=True,
)
@click.option(
    "--cluster-name", help="Kubernetes cluster name. (K8s)", required=False, type=str,
)
@click.option(
    "--namespace",
    help="Kubernetes namespace for Anyscale operator. (K8s)",
    required=False,
    type=str,
    default="anyscale-operator",
)
@click.option(
    "--project-id",
    help="Globally Unique project ID for GCP clouds (e.g., my-project-abc123)",
    required=False,
    type=str,
)
@click.option(
    "--resource-group",
    help="Resource group for Azure clouds.",
    required=False,
    type=str,
)
@click.option(
    "--functional-verify",
    help="Verify the cloud is functional. This will check that the cloud can launch workspace/service.",
    required=False,
    is_flag=False,
    flag_value="workspace",
)
@click.option(
    "--anyscale-managed",
    is_flag=True,
    default=False,
    help="Let anyscale create all the resources. (VM)",
)
@click.option(
    "--enable-head-node-fault-tolerance",
    is_flag=True,
    default=False,
    help="Whether to enable head node fault tolerance for services. (VM)",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip asking for confirmation."
)
@click.option(
    "--disable-auto-add-user",
    is_flag=True,
    default=False,
    help=(
        "All users in the organization will be added to clouds created "
        "with `anyscale cloud setup` by default. Specify --disable-auto-add-user to "
        "disable this and instead manually grant users permissions to the cloud."
    ),
)
@click.option(
    "--shared-storage",
    required=False,
    type=click.Choice([e.value for e in SharedStorageType], case_sensitive=False),
    default=SharedStorageType.OBJECT_STORAGE.value,
    show_default=True,
    help="The type of shared storage to use for the cloud. Use 'object-storage' for cloud bucket-based storage (e.g., S3, GCS), or 'nfs' for network file systems. (VM)",
)
@click.option(
    "--values-file",
    help="Path to save the generated Helm values file (for k8s stack, default: auto-generated with timestamp). (K8s)",
    required=False,
    type=str,
)
@click.option(
    "--debug", is_flag=True, default=False, help="Enable debug logging.",
)
@click.option(
    "--operator-chart",
    help="Path to operator chart (skips helm repo add/update). (K8s)",
    required=False,
    type=str,
    hidden=True,
)
@click.option(
    "--skip-resources",
    is_flag=True,
    default=False,
    help=(
        "Create an empty cloud without provisioning resources. "
        "Use this to create a cloud record first and add resources later."
    ),
)
def setup_cloud(  # noqa: PLR0913
    provider: str,
    region: str,
    name: str,
    stack: str,
    cluster_name: Optional[str],
    namespace: str,
    project_id: str,
    resource_group: Optional[str],
    functional_verify: Optional[str],
    anyscale_managed: bool,  # noqa: ARG001
    enable_head_node_fault_tolerance: bool,
    yes: bool,
    disable_auto_add_user: bool,
    shared_storage: str,
    values_file: Optional[str],
    debug: bool,
    operator_chart: Optional[str],
    skip_resources: bool,
) -> None:
    # TODO (congding): remove `anyscale_managed` in the future, now keeping it for compatibility

    # Handle --skip-resources flag: create an empty cloud without provisioning resources
    if skip_resources:
        CloudController().create_empty_cloud(name=name)
        return

    # For normal setup, provider and region are required - prompt if not provided
    if not provider:
        provider = click.prompt(
            "Provider", type=click.Choice(["aws", "gcp", "azure"], case_sensitive=False)
        )
    if not region:
        region = click.prompt("Region", default=default_region(provider))

    # Handle Kubernetes stack
    if stack == "k8s":
        if not cluster_name:
            raise click.ClickException(
                "--cluster-name is required when using --stack=k8s"
            )

        setup_kubernetes_cloud(
            provider=provider,
            region=region,
            name=name,
            cluster_name=cluster_name,
            namespace=namespace,
            project_id=project_id,
            resource_group=resource_group,
            functional_verify=bool(functional_verify),
            yes=yes,
            values_file=values_file,
            debug=debug,
            operator_chart=operator_chart,
        )
        return

    # Handle VM stack
    # Convert string to enum for type safety
    shared_storage_type = SharedStorageType(shared_storage)
    if provider == "aws":
        CloudController().setup_managed_cloud(
            provider=provider,
            region=region,
            name=name,
            functional_verify=functional_verify,
            cluster_management_stack_version=ClusterManagementStackVersions.V2,
            enable_head_node_fault_tolerance=enable_head_node_fault_tolerance,
            yes=yes,
            auto_add_user=(not disable_auto_add_user),
            shared_storage=shared_storage_type,
        )
    elif provider == "gcp":
        if not project_id:
            project_id = click.prompt("GCP Project ID", type=str)
        if project_id[0].isdigit():
            # project ID should start with a letter
            raise click.ClickException(
                "Please provide a valid project ID. Note that project ID is not project number, see https://cloud.google.com/resource-manager/docs/creating-managing-projects#before_you_begin for details."
            )
        CloudController().setup_managed_cloud(
            provider=provider,
            region=region,
            name=name,
            project_id=project_id,
            functional_verify=functional_verify,
            cluster_management_stack_version=ClusterManagementStackVersions.V2,
            enable_head_node_fault_tolerance=enable_head_node_fault_tolerance,
            yes=yes,
            auto_add_user=(not disable_auto_add_user),
            shared_storage=shared_storage_type,
        )


@cloud_cli.command(
    name="list", help=("List information about clouds in your Anyscale organization."),
)
@click.option(
    "--name",
    "-n",
    required=False,
    default=None,
    help="Name of cloud to get information about.",
)
@click.option(
    "--cloud-id",
    "--id",
    required=False,
    default=None,
    help=("Id of cloud to get information about."),
)
@click.option(
    "--max-items",
    required=False,
    default=None,
    type=int,
    help="Maximum number of clouds to return. If not specified, all results are returned.",
    callback=validate_non_negative_arg,
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
def list_cloud(  # noqa: A001
    name: Optional[str],
    cloud_id: Optional[str],
    max_items: Optional[int],
    page_size: int,
    interactive: bool,
    json_output: bool,
) -> None:
    if max_items is not None and interactive:
        raise click.UsageError("--max-items only allowed with --no-interactive")

    effective_max = max_items
    stderr = Console(stderr=True)
    if not interactive and effective_max is None:
        stderr.print(
            f"Defaulting to {NON_INTERACTIVE_DEFAULT_MAX_ITEMS} items in batch mode; "
            "use --max-items to override."
        )
        effective_max = NON_INTERACTIVE_DEFAULT_MAX_ITEMS

    console = Console()

    # diagnostics
    stderr.print("[bold]Listing clouds with:[/]")
    stderr.print(f"• name            = {name or '<any>'}")
    stderr.print(f"• id              = {cloud_id or '<any>'}")
    stderr.print(f"• mode            = {'interactive' if interactive else 'batch'}")
    stderr.print(f"• per-page limit  = {page_size}")
    stderr.print(f"• max-items total = {effective_max or 'all'}")
    stderr.print(f"\nView your Clouds in the UI at {get_endpoint('/clouds')}\n")

    # choose formatter
    if json_output:

        def json_formatter(cloud: Any) -> Dict[str, Any]:
            to_dict = getattr(cloud, "to_dict", None)
            return to_dict() if callable(to_dict) else dict(cloud.__dict__)

        formatter = json_formatter
    else:
        formatter = _format_cloud_output_data

    total = 0
    try:
        iterator = anyscale.cloud.list(
            cloud_id=cloud_id,
            name=name,
            max_items=None if interactive else effective_max,
            page_size=page_size,
        )
        total = display_list(
            iterator=iter(iterator),
            item_formatter=formatter,
            table_creator=_create_cloud_list_table,
            json_output=json_output,
            page_size=page_size,
            interactive=interactive,
            max_items=effective_max,
            console=console,
        )

        # Always print diagnostics to stderr, even in JSON mode.
        if total > 0:
            stderr.print(f"\nFetched {total} clouds.")
        else:
            stderr.print("\nNo clouds found.")
    except Exception as e:  # noqa: BLE001
        log.error(f"Failed to list clouds: {e}")
        raise SystemExit(1)


@cloud_cli.group("resource", help="Manage the configuration for a cloud resource.")
def cloud_resource_group() -> None:
    pass


@cloud_cli.group("config", help="Manage the configuration for a cloud.")
def cloud_config_group() -> None:
    pass


@cloud_resource_group.command(
    name="create",
    help="Create a new cloud resource in an existing cloud.",
    cls=AnyscaleCommand,
    example=command_examples.CLOUD_RESOURCE_CREATE_EXAMPLE,
    is_beta=True,
)
@click.option(
    "--cloud",
    help="The name of the cloud to add the new resource to.",
    type=str,
    required=False,
)
@click.option(
    "--cloud-id",
    help="The ID of the cloud to add the new resource to.",
    type=str,
    required=False,
)
@click.option(
    "--file",
    "-f",
    help="Path to a YAML file defining the cloud resource. Schema: https://docs.anyscale.com/reference/cloud/#cloudresource.",
    required=True,
)
@click.option(
    "--skip-verification",
    is_flag=True,
    default=False,
    help="Skip cloud resource verification.",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip asking for confirmation."
)
def cloud_resource_create(
    cloud: Optional[str],
    cloud_id: Optional[str],
    file: str,
    skip_verification: bool,
    yes: bool,
) -> None:
    try:
        CloudController().create_cloud_resource(
            cloud, cloud_id, file, skip_verification, yes
        )
    except click.ClickException as e:
        print(e)


@cloud_resource_group.command(
    name="setup",
    help="Set up cloud resources for an existing cloud.",
    cls=AnyscaleCommand,
    is_alpha=True,
)
@click.option(
    "--provider",
    help="The cloud provider type.",
    required=True,
    type=click.Choice(["aws", "gcp"], case_sensitive=False),
)
@click.option(
    "--region", help="Region to set up the resources in.", required=True,
)
@click.option(
    "--stack",
    help="The compute stack to use",
    required=False,
    type=click.Choice(["vm", "k8s"], case_sensitive=False),
    default="k8s",
    show_default=True,
)
@click.option(
    "--cloud",
    help="The name of the existing cloud to add resources to. Either this or --cloud-id is required.",
    type=str,
    required=False,
)
@click.option(
    "--cloud-id",
    help="The ID of the existing cloud to add resources to. Either this or --cloud is required.",
    type=str,
    required=False,
)
@click.option(
    "--cluster-name", help="Kubernetes cluster name. (K8s)", required=False, type=str,
)
@click.option(
    "--namespace",
    help="Kubernetes namespace for Anyscale operator. (K8s)",
    required=False,
    type=str,
    default="anyscale-operator",
)
@click.option(
    "--project-id",
    help="Globally Unique project ID for GCP clouds (e.g., my-project-abc123)",
    required=False,
    type=str,
)
@click.option(
    "--functional-verify",
    help="Verify the cloud is functional. This will check that the cloud can launch workspace/service.",
    required=False,
    is_flag=False,
    flag_value="workspace",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip asking for confirmation."
)
@click.option(
    "--values-file",
    help="Path to save the generated Helm values file (K8s: default - auto-generated with timestamp). (K8s)",
    required=False,
    type=str,
)
@click.option(
    "--debug", is_flag=True, default=False, help="Enable debug logging.",
)
@click.option(
    "--operator-chart",
    help="Path to operator chart (skips helm repo add/update). (K8s)",
    required=False,
    type=str,
    hidden=True,
)
@click.option(
    "--resource-name",
    help="Name for the cloud resource (optional, will be auto-generated if not provided)",
    required=False,
    type=str,
    default=None,
)
@click.option(
    "--enable-head-node-fault-tolerance",
    is_flag=True,
    default=False,
    help="Whether to enable head node fault tolerance for services. (VM)",
)
@click.option(
    "--shared-storage",
    required=False,
    type=click.Choice([e.value for e in SharedStorageType], case_sensitive=False),
    default=SharedStorageType.OBJECT_STORAGE.value,
    show_default=True,
    help="The type of shared storage to use. Use 'object-storage' for cloud bucket-based storage (e.g., S3, GCS), or 'nfs' for network file systems. (VM)",
)
def cloud_resource_setup(  # noqa: PLR0913
    provider: str,
    region: str,
    stack: str,
    cloud: Optional[str],
    cloud_id: Optional[str],
    cluster_name: Optional[str],
    namespace: str,
    project_id: Optional[str],
    functional_verify: Optional[str],
    yes: bool,
    values_file: Optional[str],
    debug: bool,
    operator_chart: Optional[str],
    resource_name: Optional[str],
    enable_head_node_fault_tolerance: bool,
    shared_storage: str,
) -> None:
    """
    Set up cloud resources for an existing Anyscale cloud.

    This command sets up infrastructure (S3/GCS buckets, IAM roles, etc.) and creates
    a cloud resource in an existing cloud instead of registering a new cloud.

    For K8s stack: Also installs the Anyscale operator on your Kubernetes cluster.
    For VM stack: Creates CloudFormation (AWS) or Deployment Manager (GCP) resources.
    """
    if stack == "k8s":
        if not cluster_name:
            raise click.ClickException(
                "--cluster-name is required when using --stack=k8s"
            )
        setup_kubernetes_cloud_resource(
            provider=provider,
            region=region,
            cloud_name=cloud,
            cloud_id=cloud_id,
            cluster_name=cluster_name,
            namespace=namespace,
            project_id=project_id,
            functional_verify=bool(functional_verify),
            yes=yes,
            values_file=values_file,
            debug=debug,
            operator_chart=operator_chart,
            resource_name=resource_name,
        )
    elif stack == "vm":
        setup_vm_cloud_resource(
            provider=provider,
            region=region,
            cloud_name=cloud,
            cloud_id=cloud_id,
            project_id=project_id,
            enable_head_node_fault_tolerance=enable_head_node_fault_tolerance,
            shared_storage=SharedStorageType(shared_storage),
        )
    else:
        raise click.ClickException(f"Unsupported stack: {stack}")


@cloud_resource_group.command(
    name="delete",
    help="Remove a cloud resource from an existing cloud.",
    cls=AnyscaleCommand,
    example=command_examples.CLOUD_RESOURCE_DELETE_EXAMPLE,
    is_beta=True,
)
@click.option(
    "--cloud",
    help="The name of the cloud to remove the resource from.",
    type=str,
    required=True,
)
@click.option(
    "--resource",
    help="The name of the cloud resource to remove.",
    type=str,
    required=True,
)
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip asking for confirmation."
)
def cloud_resource_delete(cloud: str, resource: str, yes: bool,) -> None:
    try:
        CloudController().remove_cloud_resource(cloud, resource, yes)
    except click.ClickException as e:
        print(e)


@cloud_cli.command(
    name="update", help=("Update a cloud."),
)
@click.argument("cloud-name", required=False)
@click.option(
    "--cloud-id",
    "--id",
    help="Cloud id to update. Alternative to cloud name.",
    required=False,
)
@click.option("--name", "-n", help="Update configuration of cloud by name.", type=str)
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip asking for confirmation."
)
@click.option(
    "--functional-verify",
    help="Verify the cloud is functional. This will check that the cloud can launch workspace/service.",
    required=False,
    is_flag=False,
    flag_value="workspace",
)
@click.option(
    "--enable-auto-add-user/--disable-auto-add-user",
    default=None,
    help=(
        "If --enable-auto-add-user is specified for a cloud, all users in the organization "
        "will be added to the cloud by default. Note: There may be up to 30 sec delay for all users to be granted "
        "permissions after this feature is enabled. Specifying --disable-auto-add-user will require that users "
        "are manually granted permissions to access the cloud. No existing cloud permissions are altered by specifying this flag."
    ),
)
@click.option(
    "--resources-file",
    "-f",
    help="Path to a YAML file defining a single cloud resource or a list of cloud resources. Only applicable for customer-managed resources. Schema: https://docs.anyscale.com/reference/cloud/#cloudresource.",
    required=False,
)
@click.option(
    "--enable-head-node-fault-tolerance",
    is_flag=True,
    default=False,
    help="Whether to enable head node fault tolerance for services. Only applicable for clouds with Anyscale-managed resources (clouds created via `anyscale cloud setup`).",
)
@click.option(
    "--skip-verification",
    is_flag=True,
    default=False,
    help="Skip cloud resource verification.",
)
def cloud_update(  # noqa: PLR0913
    cloud_name: Optional[str],
    name: Optional[str],
    cloud_id: Optional[str],
    functional_verify: Optional[str],
    yes: bool,
    enable_auto_add_user: Optional[bool],
    resources_file: Optional[str],
    enable_head_node_fault_tolerance: bool,
    skip_verification: bool,
) -> None:
    if cloud_name and name and cloud_name != name:
        raise click.ClickException(
            "The positional argument CLOUD_NAME and the keyword argument --name "
            "were both provided. Please only provide one of these two arguments."
        )
    CloudController().update_cloud(
        cloud_name=cloud_name or name,
        cloud_id=cloud_id,
        enable_auto_add_user=enable_auto_add_user,
        enable_head_node_fault_tolerance=enable_head_node_fault_tolerance,
        resources_file=resources_file,
        functional_verify=functional_verify,
        yes=yes,
        skip_verification=skip_verification,
    )


@cloud_cli.command(
    name="update-storage-cors",
    help="Update CORS configuration on cloud storage to support Anyscale UI features. Works with both managed and customer-managed clouds. When a cloud resource is not specified, updates CORS for all cloud resources under the cloud.",
)
@click.argument("cloud-name", required=False)
@click.option(
    "--cloud-id",
    "--id",
    help="Cloud id to update. Alternative to cloud name.",
    required=False,
)
@click.option("--name", "-n", help="Update storage CORS for cloud by name.", type=str)
@click.option(
    "--resource",
    help="Name of the cloud resource to update. If not provided, updates all cloud resources under the cloud.",
    type=str,
    required=False,
)
@click.option(
    "--resource-id",
    "cloud_resource_id",
    help="Cloud resource ID to update. Alternative to cloud resource name.",
    type=str,
    required=False,
)
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip asking for confirmation."
)
def cloud_update_storage_cors(
    cloud_name: Optional[str],
    name: Optional[str],
    cloud_id: Optional[str],
    resource: Optional[str],
    cloud_resource_id: Optional[str],
    yes: bool,
) -> None:
    if cloud_name and name and cloud_name != name:
        raise click.ClickException(
            "The positional argument CLOUD_NAME and the keyword argument --name "
            "were both provided. Please only provide one of these two arguments."
        )
    if resource and cloud_resource_id:
        raise click.ClickException(
            "Cannot specify both --resource and --resource-id. Please provide only one."
        )
    CloudController().update_cors(
        cloud_name=cloud_name or name,
        cloud_id=cloud_id,
        resource=resource,
        cloud_resource_id=cloud_resource_id,
        yes=yes,
    )


@cloud_config_group.command("get", help="Get the current configuration for a cloud.")
@click.argument("cloud-name", required=False)
@click.option("--name", "-n", help="Update configuration of cloud by name.", type=str)
@click.option(
    "--cloud-id",
    "--id",
    help="Cloud id to get details about. Alternative to cloud name.",
    required=False,
)
@click.option(
    "--resource",
    help="Name of the cloud resource to get details for. If not provided, defaults to the primary resource for the cloud.",
    type=str,
    required=False,
)
@click.option(
    "--resource-id",
    "cloud_resource_id",
    help="Cloud resource ID to get details for. Alternative to cloud resource name.",
    type=str,
    required=False,
)
def cloud_config_get(
    cloud_name: Optional[str],
    name: Optional[str],
    cloud_id: Optional[str],
    resource: Optional[str],
    cloud_resource_id: Optional[str],
) -> None:
    if cloud_name and name and cloud_name != name:
        raise click.ClickException(
            "The positional argument CLOUD_NAME and the keyword argument --name "
            "were both provided. Please only provide one of these two arguments."
        )

    # Validate resource selection options
    if resource and cloud_resource_id:
        raise click.ClickException(
            "Cannot specify both --resource and --resource-id. Please provide only one."
        )

    config = CloudController().get_cloud_config(
        cloud_name=cloud_name or name,
        cloud_id=cloud_id,
        resource=resource,
        cloud_resource_id=cloud_resource_id,
    )
    stream = StringIO()
    yaml.dump(config.spec, stream)
    print(stream.getvalue())


def _validate_cloud_config_update_args(
    cloud_name: Optional[str],
    name: Optional[str],
    resource: Optional[str],
    cloud_resource_id: Optional[str],
    passed_enable_disable_flags: bool,
    spec_file: Optional[str],
) -> None:
    """Validate arguments for cloud config update command."""
    if cloud_name and name and cloud_name != name:
        raise click.ClickException(
            "The positional argument CLOUD_NAME and the keyword argument --name "
            "were both provided. Please only provide one of these two arguments."
        )

    if resource and cloud_resource_id:
        raise click.ClickException(
            "Cannot specify both --resource and --resource-id. Please provide only one."
        )

    if passed_enable_disable_flags and spec_file:
        raise click.ClickException(
            "Invalid combination of arguments: --spec-file should not be provided with any other enable/disable flags."
        )

    if (resource or cloud_resource_id) and not spec_file:
        raise click.ClickException(
            "--resource and --resource-id can only be used with --spec-file."
        )


def _handle_log_ingestion_config(enable_log_ingestion: Optional[bool]) -> None:
    """Handle log ingestion configuration with user prompts."""
    if enable_log_ingestion is True:
        consent_message = click.prompt(
            "--enable-log-ingestion is specified. Please note the logs produced by "
            "your cluster will be ingested into Anyscale's service in region "
            "us-west-2. Your clusters may incur extra data transfer cost from the "
            "cloud provider. If you are sure you want to enable this feature, "
            'please type "consent"',
            type=str,
        )
        if consent_message != "consent":
            raise click.ClickException(
                'You must type "consent" to enable log ingestion.'
            )
    elif enable_log_ingestion is False:
        confirm_response = click.confirm(
            "--disable-log-ingestion is specified. Please note the logs that's "
            "already ingested will not be deleted. Existing clusters will not stop"
            "the log ingestion until you restart them. Logs are automatically "
            "deleted after 30 days from the time of ingestion. Are you sure you "
            "want to disable log ingestion?"
        )
        if not confirm_response:
            raise click.ClickException("You must confirm to disable log ingestion.")


def _handle_system_cluster_config(enable_system_cluster: Optional[bool]) -> None:
    """Handle system cluster configuration with user prompts."""
    confirm_response = True
    if enable_system_cluster is True:
        confirm_response = click.confirm(
            "--enable-system-cluster is specified. Please note that this will enable "
            "system cluster functionality for the cloud and will incur extra cost. "
            "Are you sure you want to enable system cluster?"
        )
    elif enable_system_cluster is False:
        confirm_response = click.confirm(
            "--disable-system-cluster is specified. This will disable system cluster "
            "functionality for the cloud. Please note that this will not terminate "
            "the system cluster if it is currently running. "
            "Are you sure you want to disable system cluster?"
        )

    if enable_system_cluster is not None and not confirm_response:
        raise click.ClickException(
            f"You must confirm to {'enable' if enable_system_cluster else 'disable'} system cluster."
        )


@cloud_config_group.command(
    "update",
    help="Update the current configuration for a cloud.",
    cls=AnyscaleCommand,
    example=command_examples.CLOUD_CONFIG_UPDATE_EXAMPLE,
)
@click.argument("cloud-name", required=False)
@click.option("--name", "-n", help="Update configuration of cloud by name.", type=str)
@click.option(
    "--cloud-id",
    "--id",
    help="Cloud id to update. Alternative to cloud name.",
    required=False,
)
@click.option(
    "--enable-log-ingestion/--disable-log-ingestion",
    default=None,
    help=(
        "If --enable-log-ingestion is specified for a cloud, it will enable the log "
        "viewing and querying UI features for the clusters on this cloud. This will "
        "enable easier debugging. The logs produced by the clusters will "
        "be sent from the data plane to the control plane. Anyscale does not share "
        "this data with any third party or use it for any purpose other than serving "
        "the log UI for the customer. The log will be stored at most 30 days."
        "Please note by disable this feature again, Anyscale doesn't "
        "delete the logs that have already been ingested. Your clusters may incur "
        "extra data transfer cost from the cloud provider by enabling this feature."
    ),
)
@click.option(
    "--enable-system-cluster/--disable-system-cluster",
    default=None,
    help="Enable or disable system cluster functionality.",
    required=False,
)
@click.option(
    "--spec-file",
    type=str,
    required=False,
    help="Provide a path to a specification file.",
)
@click.option(
    "--resource",
    help="Name of the cloud resource to get details for. If not provided, defaults to the primary resource for the cloud.",
    type=str,
    required=False,
)
@click.option(
    "--resource-id",
    "cloud_resource_id",
    help="Cloud resource ID to get details for. Alternative to cloud resource name.",
    type=str,
    required=False,
)
def cloud_config_update(  # noqa: PLR0913
    cloud_name: Optional[str],
    name: Optional[str],
    cloud_id: Optional[str],
    enable_log_ingestion: Optional[bool],
    enable_system_cluster: Optional[bool],
    spec_file: Optional[str],
    resource: Optional[str],
    cloud_resource_id: Optional[str],
) -> None:
    passed_enable_disable_flags = any(
        [enable_log_ingestion is not None, enable_system_cluster is not None]
    )

    _validate_cloud_config_update_args(
        cloud_name,
        name,
        resource,
        cloud_resource_id,
        passed_enable_disable_flags,
        spec_file,
    )

    cloud_name_resolved = cloud_name or name

    if passed_enable_disable_flags:
        _handle_log_ingestion_config(enable_log_ingestion)
        CloudController().update_cloud_config(
            cloud_name=cloud_name_resolved,
            cloud_id=cloud_id,
            enable_log_ingestion=enable_log_ingestion,
        )

        _handle_system_cluster_config(enable_system_cluster)
        CloudController().update_system_cluster_config(
            cloud_name=cloud_name_resolved,
            cloud_id=cloud_id,
            system_cluster_enabled=enable_system_cluster,
        )
    elif spec_file:
        CloudController().update_cloud_config(
            cloud_name=cloud_name_resolved,
            cloud_id=cloud_id,
            spec_file=spec_file,
            resource=resource,
            cloud_resource_id=cloud_resource_id,
        )
    else:
        raise click.ClickException(
            "Please provide at least one of the following arguments: --enable-log-ingestion, --disable-log-ingestion, --enable-system-cluster, --disable-system-cluster, --spec-file."
        )


@cloud_cli.command(
    name="register", help="Register an anyscale cloud with your own resources."
)
@click.option(
    "--provider",
    help="The cloud provider type.",
    required=True,
    type=click.Choice(["aws", "gcp", "azure", "generic"], case_sensitive=False),
)
@click.option(
    "--region",
    cls=OptionPromptNull,
    help="Region to set up the credentials in.",
    required=True,
    default_option="provider",
    default=default_region,
    show_default=True,
)
@click.option(
    "--compute-stack",
    help="The compute stack type (VM or K8S).",
    required=False,
    type=click.Choice([ComputeStack.VM, ComputeStack.K8S], case_sensitive=False),
    default=ComputeStack.VM,
    # TODO (shomilj): Unhide this option when full support for Kubernetes has been rolled out.
    hidden=True,
)
@click.option(
    "--name", "-n", help="Name of the cloud.", required=True,
)
@click.option(
    "--vpc-id", help="The ID of the VPC.", required=False, type=str,
)
@click.option(
    "--subnet-ids",
    help="Comma separated list of subnet ids.",
    required=False,
    type=str,
)
@click.option(
    "--file-storage-id",
    help="File storage ID (e.g. EFS ID for AWS, Filestore instance ID for GCP)",
    required=False,
    type=str,
)
@click.option(
    "--efs-id", help="The EFS ID.", required=False, type=str, hidden=True,
)
@click.option(
    "--anyscale-iam-role-id",
    help="The Anyscale IAM Role ARN.",
    required=False,
    type=str,
)
@click.option(
    "--instance-iam-role-id",
    help="The instance IAM role ARN.",
    required=False,
    type=str,
)
@click.option(
    "--security-group-ids",
    help="IDs of the security groups.",
    required=False,
    type=str,
)
@click.option(
    "--s3-bucket-id", help="S3 bucket ID.", required=False, type=str, hidden=True,
)
@click.option(
    "--external-id",
    help="The trust policy external ID for the cross account IAM role.",
    required=False,
    type=str,
)
@click.option(
    "--memorydb-cluster-id", help="Memorydb cluster ID", required=False, type=str,
)
@click.option(
    "--project-id",
    help="Globally Unique project ID for GCP clouds (e.g., my-project-abc123)",
    required=False,
    type=str,
)
@click.option(
    "--vpc-name", help="VPC name for GCP clouds", required=False, type=str,
)
@click.option(
    "--subnet-names",
    help="Comma separated list of subnet names for GCP clouds",
    required=False,
    type=str,
)
@click.option(
    "--filestore-instance-id",
    help="Filestore instance ID for GCP clouds.",
    required=False,
    type=str,
    hidden=True,
)
@click.option(
    "--filestore-location",
    help="Filestore location for GCP clouds.",
    required=False,
    type=str,
)
@click.option(
    "--anyscale-service-account-email",
    help="Anyscale service account email for GCP clouds.",
    required=False,
    type=str,
)
@click.option(
    "--instance-service-account-email",
    help="Instance service account email for GCP clouds.",
    required=False,
    type=str,
)
@click.option(
    "--provider-name",
    help="Workload Identity Federation provider name for Anyscale access.",
    required=False,
    type=str,
)
@click.option(
    "--firewall-policy-names",
    help="Filewall policy names for GCP clouds",
    required=False,
    type=str,
)
@click.option(
    "--cloud-storage-bucket-name",
    help="A fully qualified storage bucket name for cloud storage, e.g. s3://bucket-name, gs://bucket-name, or abfss://bucket-name@account.dfs.core.windows.net.",
    required=False,
    type=str,
)
@click.option(
    "--cloud-storage-bucket-endpoint",
    help="An endpoint for cloud storage, e.g. used to override the default cloud storage scheme's endpoint (e.g. for S3, this would be passed to the AWS_ENDPOINT_URL environment variable).",
    required=False,
    type=str,
)
@click.option(
    "--cloud-storage-bucket-region",
    help="The region of the cloud storage bucket. If not provided, the region of the cloud will be used to access the cloud storage bucket.",
    required=False,
    type=str,
)
@click.option(
    "--nfs-mount-target",
    help="A comma-separated value representing a (zone, mount target) tuple, e.g. us-west-2a,1.2.3.4 (may be provided multiple times, one for each zone). If only one value is provided (e.g. 1.2.3.4), then that value will be used for all zones.",
    required=False,
    type=str,
    multiple=True,
)
@click.option(
    "--nfs-mount-path",
    help="The path of the NFS server to mount from (e.g. nfs-target-address/nfs-path will be mounted).",
    required=False,
    type=str,
)
@click.option(
    "--persistent-volume-claim",
    help="For Kubernetes deployments only, the name of the persistent volume claim used to mount shared storage into pods. Mutually exclusive with NFS configurations.",
    required=False,
    type=str,
)
@click.option(
    "--csi-ephemeral-volume-driver",
    help="For Kubernetes deployments only, the CSI ephemeral volume driver used to mount shared storage into pods. Mutually exclusive with NFS configurations.",
    required=False,
    type=str,
)
@click.option(
    "--memorystore-instance-name",
    help="Memorystore instance name for GCP clouds",
    required=False,
    type=str,
)
@click.option(
    "--host-project-id",
    help="Host project ID for shared VPC",
    required=False,
    type=str,
)
@click.option(
    "--kubernetes-zones",
    help="On the Kubernetes compute stack, a comma-separated list of zones to launch pods in.",
    required=False,
    type=str,
)
@click.option(
    "--anyscale-operator-iam-identity",
    help="On the Kubernetes compute stack, the cloud provider IAM identity federated with the Anyscale Operator's kubernetes service account, which will be used by Anyscale control plane for validation during Anyscale Operator bootstrap in the dataplane. IN AWS EKS, this is the ARN of the IAM role. For GCP GKE, this is the service account email.",
    required=False,
    type=str,
)
@click.option(
    "--private-network", help="Use private network.", is_flag=True, default=False,
)
@click.option(
    "--functional-verify",
    help="Verify the cloud is functional. This will check that the cloud can launch workspace/service.",
    required=False,
    is_flag=False,
    flag_value="workspace",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip asking for confirmation."
)
@click.option(
    "--skip-verifications",
    help="Skip verifications. This will skip all verifications.",
    required=False,
    is_flag=True,
    type=bool,
    default=False,
)
@click.option(
    "--enable-auto-add-user",
    is_flag=True,
    default=False,
    help=(
        "If --enable-auto-add-user is specified for a cloud, all users in the organization "
        "will be added to the cloud by default. Otherwise users will need to be manually granted "
        "permissions to the cloud. Note: There may be up to 30 sec delay for all users to be granted "
        "permissions after the cloud is created."
    ),
)
@click.option(
    "--resource-file",
    "-f",
    help="Path to a YAML file defining a cloud resource. Schema: https://docs.anyscale.com/reference/cloud/#cloudresource.",
    required=False,
)
@click.option(
    "--azure-tenant-id",
    help="The Azure Tenant ID to use for the cloud.",
    required=False,
    type=str,
)
def register_cloud(  # noqa: PLR0913, PLR0912, C901
    provider: str,
    region: str,
    compute_stack: ComputeStack,
    name: str,
    vpc_id: str,
    subnet_ids: str,
    file_storage_id: str,
    efs_id: str,
    anyscale_iam_role_id: str,
    instance_iam_role_id: str,
    security_group_ids: str,
    s3_bucket_id: str,
    external_id: Optional[str],
    memorydb_cluster_id: str,
    project_id: str,
    vpc_name: str,
    subnet_names: str,
    filestore_instance_id: str,
    filestore_location: str,
    anyscale_service_account_email: str,
    instance_service_account_email: str,
    provider_name: str,
    firewall_policy_names: str,
    cloud_storage_bucket_name: str,
    cloud_storage_bucket_endpoint: Optional[str],
    cloud_storage_bucket_region: Optional[str],
    nfs_mount_target: List[str],
    nfs_mount_path: str,
    persistent_volume_claim: Optional[str],
    csi_ephemeral_volume_driver: Optional[str],
    memorystore_instance_name: str,
    host_project_id: Optional[str],
    kubernetes_zones: Optional[str],
    anyscale_operator_iam_identity: Optional[str],
    azure_tenant_id: Optional[str],
    functional_verify: Optional[str],
    private_network: bool,
    yes: bool,
    skip_verifications: bool,
    enable_auto_add_user: bool,
    resource_file: Optional[str],
) -> None:
    # Load CloudDeployment from the resource file if provided, otherwise build from CLI flags
    if resource_file:
        # Read the spec file.
        path = pathlib.Path(resource_file)
        if not path.exists():
            raise click.ClickException(f"{resource_file} does not exist.")
        if not path.is_file():
            raise click.ClickException(f"{resource_file} is not a file.")

        spec = yaml.safe_load(path.read_text())
        try:
            cloud_resource = CloudDeployment(**spec)

            # Convert nested dict objects to model objects
            if cloud_resource.file_storage:
                cloud_resource.file_storage = FileStorage(**cloud_resource.file_storage)
            if cloud_resource.object_storage:
                cloud_resource.object_storage = ObjectStorage(
                    **cloud_resource.object_storage
                )
            if cloud_resource.aws_config:
                cloud_resource.aws_config = AWSConfig(**cloud_resource.aws_config)
            if cloud_resource.gcp_config:
                cloud_resource.gcp_config = GCPConfig(**cloud_resource.gcp_config)
            if cloud_resource.azure_config:
                cloud_resource.azure_config = AzureConfig(**cloud_resource.azure_config)
            if cloud_resource.kubernetes_config:
                cloud_resource.kubernetes_config = KubernetesConfig(
                    **cloud_resource.kubernetes_config
                )

        except Exception as e:  # noqa: BLE001
            raise click.ClickException(f"Failed to parse cloud resource: {e}")

    else:
        missing_args: List[str] = []

        # Validate K8S-only storage flags
        if (
            persistent_volume_claim or csi_ephemeral_volume_driver
        ) and compute_stack != ComputeStack.K8S:
            raise click.ClickException(
                "--persistent-volume-claim and --csi-ephemeral-volume-driver are only supported with --compute-stack=k8s"
            )

        # Validate mutual exclusivity of storage configurations
        storage_configs = []
        if nfs_mount_target or nfs_mount_path:
            storage_configs.append("NFS")
        if persistent_volume_claim:
            storage_configs.append("persistent volume claim")
        if csi_ephemeral_volume_driver:
            storage_configs.append("CSI ephemeral volume driver")

        if len(storage_configs) > 1:
            raise click.ClickException(
                f"Storage configurations are mutually exclusive. Found: {', '.join(storage_configs)}. "
                "Please specify only one of: --nfs-mount-target/--nfs-mount-path, --persistent-volume-claim, or --csi-ephemeral-volume-driver"
            )

        if provider == "aws":
            if s3_bucket_id and not cloud_storage_bucket_name:
                cloud_storage_bucket_name = s3_bucket_id
            if efs_id and not file_storage_id:
                file_storage_id = efs_id
            # Check for missing required arguments for AWS clouds,
            # based on the compute stack (not all args are required
            # on all compute stacks).
            required_resources = [
                (vpc_id, "--vpc-id", (ComputeStack.VM)),
                (subnet_ids, "--subnet-ids", (ComputeStack.VM)),
                (anyscale_iam_role_id, "--anyscale-iam_role-id", (ComputeStack.VM),),
                (instance_iam_role_id, "--instance-iam-role-id", (ComputeStack.VM)),
                (security_group_ids, "--security-group-ids", (ComputeStack.VM)),
                (
                    cloud_storage_bucket_name,
                    "--cloud-storage-bucket-name",
                    (ComputeStack.VM, ComputeStack.K8S),
                ),
                (kubernetes_zones, "--kubernetes-zones", (ComputeStack.K8S)),
                (
                    anyscale_operator_iam_identity,
                    "--anyscale-operator-iam-identity",
                    (ComputeStack.K8S),
                ),
            ]

            if not allow_optional_file_storage():
                required_resources.append(
                    (file_storage_id, "--file-storage-id", (ComputeStack.VM)),
                )

            for resource in required_resources:
                if compute_stack in resource[2] and resource[0] is None:
                    missing_args.append(resource[1])

            if len(missing_args) > 0:
                raise click.ClickException(f"Please provide a value for {missing_args}")

            cloud_resource = CloudDeployment(
                compute_stack=compute_stack,
                provider=CloudProviders.AWS,
                region=region,
                networking_mode=NetworkingMode.PRIVATE
                if private_network
                else NetworkingMode.PUBLIC,
                object_storage=ObjectStorage(bucket_name=cloud_storage_bucket_name),
                file_storage=FileStorage(
                    file_storage_id=file_storage_id,
                    persistent_volume_claim=persistent_volume_claim,
                    csi_ephemeral_volume_driver=csi_ephemeral_volume_driver,
                )
                if file_storage_id
                or persistent_volume_claim
                or csi_ephemeral_volume_driver
                else None,
                aws_config=AWSConfig(
                    vpc_id=vpc_id,
                    subnet_ids=subnet_ids.split(",") if subnet_ids else [],
                    security_group_ids=security_group_ids.split(",")
                    if security_group_ids
                    else [],
                    anyscale_iam_role_id=anyscale_iam_role_id,
                    external_id=external_id,
                    cluster_iam_role_id=instance_iam_role_id,
                    memorydb_cluster_name=memorydb_cluster_id,
                ),
                kubernetes_config=KubernetesConfig(
                    anyscale_operator_iam_identity=anyscale_operator_iam_identity,
                    zones=kubernetes_zones.split(",") if kubernetes_zones else [],
                )
                if compute_stack == ComputeStack.K8S
                else None,
            )

        elif provider == "gcp":
            if filestore_instance_id and not file_storage_id:
                file_storage_id = filestore_instance_id
            # Keep the parameter naming ({resource}_name or {resource}_id) consistent with GCP to reduce confusion for customers
            # Check if all required parameters are provided
            # memorystore_instance_name and host_project_id are optional for GCP clouds
            required_resources = [
                (project_id, "--project-id", (ComputeStack.VM)),
                (vpc_name, "--vpc-name", (ComputeStack.VM)),
                (subnet_names, "--subnet-names", (ComputeStack.VM)),
                (
                    anyscale_service_account_email,
                    "--anyscale-service-account-email",
                    (ComputeStack.VM),
                ),
                (
                    instance_service_account_email,
                    "--instance-service-account-email",
                    (ComputeStack.VM),
                ),
                (provider_name, "--provider-name", (ComputeStack.VM)),
                (firewall_policy_names, "--firewall-policy-names", (ComputeStack.VM)),
                (
                    cloud_storage_bucket_name,
                    "--cloud-storage-bucket-name",
                    (ComputeStack.VM, ComputeStack.K8S),
                ),
                (kubernetes_zones, "--kubernetes-zones", (ComputeStack.K8S)),
                (
                    anyscale_operator_iam_identity,
                    "--anyscale-operator-iam-identity",
                    (ComputeStack.K8S),
                ),
            ]

            if not allow_optional_file_storage():
                required_resources.extend(
                    [
                        (file_storage_id, "--file-storage-id", (ComputeStack.VM)),
                        (filestore_location, "--filestore-location", (ComputeStack.VM)),
                    ]
                )

            for resource in required_resources:
                if compute_stack in resource[2] and resource[0] is None:
                    missing_args.append(resource[1])

            if len(missing_args) > 0:
                raise click.ClickException(f"Please provide a value for {missing_args}")

            cloud_resource = CloudDeployment(
                compute_stack=compute_stack,
                provider=CloudProviders.GCP,
                region=region,
                networking_mode=NetworkingMode.PRIVATE
                if private_network
                else NetworkingMode.PUBLIC,
                object_storage=ObjectStorage(bucket_name=cloud_storage_bucket_name),
                file_storage=FileStorage(
                    file_storage_id="projects/{}/locations/{}/instances/{}".format(
                        project_id, filestore_location, file_storage_id
                    )
                    if file_storage_id
                    else None,
                    persistent_volume_claim=persistent_volume_claim,
                    csi_ephemeral_volume_driver=csi_ephemeral_volume_driver,
                )
                if file_storage_id
                or persistent_volume_claim
                or csi_ephemeral_volume_driver
                else None,
                gcp_config=GCPConfig(
                    project_id=project_id,
                    host_project_id=host_project_id,
                    provider_name=provider_name,
                    vpc_name=vpc_name,
                    subnet_names=subnet_names.split(",") if subnet_names else [],
                    firewall_policy_names=firewall_policy_names.split(",")
                    if firewall_policy_names
                    else [],
                    anyscale_service_account_email=anyscale_service_account_email,
                    cluster_service_account_email=instance_service_account_email,
                    memorystore_instance_name=memorystore_instance_name,
                ),
                kubernetes_config=KubernetesConfig(
                    anyscale_operator_iam_identity=anyscale_operator_iam_identity,
                    zones=kubernetes_zones.split(",") if kubernetes_zones else [],
                )
                if compute_stack == ComputeStack.K8S
                else None,
            )

        elif provider in ("azure", "generic"):
            # For the 'generic' provider type, for the time being, most fields are optional; only 'name', 'provider', and 'compute-stack' are required.
            if not name:
                raise click.ClickException("Please provide a value for --name.")

            if compute_stack != ComputeStack.K8S:
                raise click.ClickException(
                    "--compute-stack=k8s must be passed to register this Anyscale cloud."
                )

            # Handle parsing / conversion of nfs_mount_targets.
            mount_targets: List[NFSMountTarget] = []
            for target in nfs_mount_target or []:
                parts = [part.strip() for part in target.split(",")]
                if len(parts) == 1:
                    mount_targets.append(NFSMountTarget(address=parts[0]))
                elif len(parts) == 2:
                    mount_targets.append(
                        NFSMountTarget(address=parts[1], zone=parts[0])
                    )
                else:
                    raise click.ClickException(
                        f"Invalid mount target {target}; expected (zone,address) tuple or a singular address."
                    )

            cloud_provider = (
                CloudProviders.AZURE if provider == "azure" else CloudProviders.GENERIC
            )

            cloud_resource = CloudDeployment(
                compute_stack=ComputeStack.K8S,
                provider=cloud_provider,
                region=region or "default",
                object_storage=ObjectStorage(
                    bucket_name=cloud_storage_bucket_name,
                    region=cloud_storage_bucket_region or region,
                    endpoint=cloud_storage_bucket_endpoint,
                )
                if cloud_storage_bucket_name
                else None,
                file_storage=FileStorage(
                    mount_targets=mount_targets,
                    mount_path=nfs_mount_path,
                    persistent_volume_claim=persistent_volume_claim,
                    csi_ephemeral_volume_driver=csi_ephemeral_volume_driver,
                )
                if mount_targets
                or persistent_volume_claim
                or csi_ephemeral_volume_driver
                else None,
                azure_config=AzureConfig(tenant_id=azure_tenant_id)
                if provider == "azure" and azure_tenant_id
                else None,
                kubernetes_config=KubernetesConfig(
                    anyscale_operator_iam_identity=anyscale_operator_iam_identity
                    if provider == "azure"
                    else None,
                    zones=kubernetes_zones.split(",") if kubernetes_zones else [],
                ),
            )

        else:
            raise click.ClickException(
                f"Invalid Cloud provider: {provider}. Available providers are [aws, gcp, azure, generic]."
            )

    if provider == "aws":
        CloudController().register_aws_cloud(
            name=name,
            cloud_resource=cloud_resource,
            functional_verify=functional_verify,
            cluster_management_stack_version=ClusterManagementStackVersions.V2,
            yes=yes,
            skip_verifications=skip_verifications,
            auto_add_user=enable_auto_add_user,
        )
    elif provider == "gcp":
        CloudController().register_gcp_cloud(
            name=name,
            cloud_resource=cloud_resource,
            functional_verify=functional_verify,
            cluster_management_stack_version=ClusterManagementStackVersions.V2,
            yes=yes,
            skip_verifications=skip_verifications,
            auto_add_user=enable_auto_add_user,
        )
    elif provider in ("azure", "generic"):
        CloudController().register_azure_or_generic_cloud(
            name=name,
            provider=provider,
            cloud_resource=cloud_resource,
            auto_add_user=enable_auto_add_user,
        )
    else:
        raise click.ClickException(
            f"Invalid Cloud provider: {provider}. Available providers are [aws, gcp, azure, generic]."
        )


@cloud_cli.command(name="verify", help="Checks the healthiness of a cloud.")
@click.argument("cloud-name", required=False)
@click.option("--name", "-n", help="Verify cloud by name.", type=str)
@click.option(
    "--cloud-id",
    "--id",
    help="Verify cloud by cloud id, alternative to cloud name.",
    required=False,
)
@click.option(
    "--functional-verify",
    help="Verify the cloud is functional. This will check that the cloud can launch workspace/service.",
    required=False,
    is_flag=False,
    flag_value="workspace",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Strict Verify. Treat warnings as failures.",
)
def cloud_verify(
    cloud_name: Optional[str],
    name: Optional[str],
    cloud_id: Optional[str],
    functional_verify: Optional[str],
    strict: bool = False,
) -> bool:
    if cloud_name and name and cloud_name != name:
        raise click.ClickException(
            "The positional argument CLOUD_NAME and the keyword argument --name "
            "were both provided. Please only provide one of these two arguments."
        )

    return CloudController().verify_cloud(
        cloud_name=cloud_name or name,
        cloud_id=cloud_id,
        functional_verify=functional_verify,
        strict=strict,
    )


@cloud_cli.command(
    name="edit",
    help="DEPRECATED. Please use `anyscale cloud update` instead.\n\nEdit registered cloud resource on Anyscale. Only applicable for anyscale registered clouds.",
)
@click.argument("cloud-name", required=False)
@click.option("--name", "-n", help="Edit cloud by name.", type=str)
@click.option(
    "--cloud-id",
    "--id",
    help="Edit cloud by id, alternative to cloud name.",
    required=False,
)
@click.option(
    "--aws-s3-id", help="New S3 bucket ID.", required=False, type=str,
)
@click.option("--aws-efs-id", help="New EFS ID.", required=False, type=str)
@click.option(
    "--aws-efs-mount-target-ip",
    help="New EFS mount target IP.",
    required=False,
    type=str,
)
@click.option(
    "--memorydb-cluster-id",
    help="New AWS Memorydb cluster ID.",
    required=False,
    type=str,
)
@click.option(
    "--gcp-filestore-instance-id",
    help="New GCP filestore instance id.",
    required=False,
    type=str,
)
@click.option(
    "--gcp-filestore-location",
    help="New GCP filestore location.",
    required=False,
    type=str,
)
@click.option(
    "--gcp-cloud-storage-bucket-name",
    help="New GCP Cloud storage bucket name.",
    required=False,
    type=str,
)
@click.option(
    "--memorystore-instance-name",
    help="New Memorystore instance name for GCP clouds",
    required=False,
    type=str,
)
@click.option(
    "--functional-verify",
    help="Verify the cloud is functional. This will check that the cloud can launch workspace/service.",
    required=False,
    is_flag=False,
    flag_value="workspace",
)
@click.option(
    "--enable-auto-add-user/--disable-auto-add-user",
    default=None,
    help=(
        "If --enable-auto-add-user is specified for a cloud, all users in the organization "
        "will be added to the cloud by default. Note: There may be up to 30 sec delay for all users to be granted "
        "permissions after this feature is enabled.\n\n"
        "Specifying --disable-auto-add-user will require that users "
        "are manually granted permissions to access the cloud. No existing cloud permissions are altered by specifying this flag."
    ),
)
def cloud_edit(  # noqa: PLR0913
    cloud_name: Optional[str],
    name: Optional[str],
    cloud_id: Optional[str],
    aws_s3_id: Optional[str],
    aws_efs_id: Optional[str],
    aws_efs_mount_target_ip: Optional[str],
    memorydb_cluster_id: Optional[str],
    gcp_filestore_instance_id: Optional[str],
    gcp_filestore_location: Optional[str],
    gcp_cloud_storage_bucket_name: Optional[str],
    memorystore_instance_name: Optional[str],
    functional_verify: Optional[str],
    enable_auto_add_user: Optional[bool],
) -> None:
    if cloud_name and name and cloud_name != name:
        raise click.ClickException(
            "The positional argument CLOUD_NAME and the keyword argument --name "
            "were both provided. Please only provide one of these two arguments."
        )
    if any(
        [
            aws_s3_id,
            aws_efs_id,
            aws_efs_mount_target_ip,
            memorydb_cluster_id,
            gcp_filestore_instance_id,
            gcp_filestore_location,
            gcp_cloud_storage_bucket_name,
            memorystore_instance_name,
            enable_auto_add_user is not None,
        ]
    ):
        if any([gcp_filestore_instance_id, gcp_filestore_location]) and not all(
            [gcp_filestore_instance_id, gcp_filestore_location]
        ):
            # Make sure both gcp_filestore_instance_id and gcp_filestore_location are provided if you want to edit filestore.
            raise click.ClickException(
                "Please provide both --gcp-filestore-instance-id and --gcp-filestore-location if you want to edit filestore."
            )
        if (
            memorystore_instance_name is not None
            and re.search(
                "projects/.+/locations/.+/instances/.+", memorystore_instance_name
            )
            is None
        ):
            raise click.ClickException(
                "Please provide a valid memorystore instance name. Example: projects/<project number>/locations/<location>/instances/<instance id>"
            )
        CloudController().edit_cloud(
            cloud_name=cloud_name or name,
            cloud_id=cloud_id,
            aws_s3_id=aws_s3_id,
            aws_efs_id=aws_efs_id,
            aws_efs_mount_target_ip=aws_efs_mount_target_ip,
            memorydb_cluster_id=memorydb_cluster_id,
            gcp_filestore_instance_id=gcp_filestore_instance_id,
            gcp_filestore_location=gcp_filestore_location,
            gcp_cloud_storage_bucket_name=gcp_cloud_storage_bucket_name,
            memorystore_instance_name=memorystore_instance_name,
            functional_verify=functional_verify,
            auto_add_user=enable_auto_add_user,
        )
    else:
        raise click.ClickException(
            "Please provide at least one of the following arguments: --aws-s3-id, --aws-efs-id, --aws-efs-mount-target-ip, --memorydb-cluster-id, --gcp-filestore-instance-id, --gcp-filestore-location, --gcp-cloud-storage-bucket-name, --memorystore-instance-name, --enable-auto-add-user, --disable-auto-add-user."
        )


@cloud_cli.command(
    name="add-collaborators",
    help="Add collaborators to the cloud.",
    cls=AnyscaleCommand,
    example=command_examples.CLOUD_ADD_COLLABORATORS_EXAMPLE,
)
@click.option(
    "--cloud", "-c", help="Name of the cloud to add collaborators to.", required=True
)
@click.option(
    "--users-file",
    help="Path to a YAML file containing a list of users to add to the cloud.",
    required=True,
)
def add_collaborators(cloud: str, users_file: str,) -> None:
    collaborators = CreateCloudCollaborators.from_yaml(users_file)

    try:
        anyscale.cloud.add_collaborators(
            cloud=cloud,
            collaborators=[
                CreateCloudCollaborator(**collaborator)
                for collaborator in collaborators.collaborators
            ],
        )
    except ValueError as e:
        log.error(f"Error adding collaborators to cloud: {e}")
        return

    log.info(
        f"Successfully added {len(collaborators.collaborators)} collaborators to cloud {cloud}."
    )


def _get_cloud_info(
    cloud_id: Optional[str],
    name: Optional[str],
    output: Optional[str],
    include_status: bool,
) -> None:
    """
    Internal helper to retrieve cloud information.

    :param cloud_id: The ID of the cloud to retrieve.
    :param name: The name of the cloud to retrieve.
    :param output: Optional file path to write output to.
    :param include_status: If True, include status fields (created_at, is_default,
        operator_status, operator_status_details). If False, these fields are hidden.
    """
    # Validate that exactly one of --name or --cloud-id is provided
    if (cloud_id and name) or (not cloud_id and not name):
        log.error("Please provide exactly one of --name or --cloud-id.")
        return

    try:
        cloud = anyscale.cloud.get(id=cloud_id, name=name)

        if not cloud:
            log.error("Cloud not found.")
            return

        # Include all cloud resources for the cloud.
        cloud_resources = CloudController().get_formatted_cloud_resources(
            cloud_id=cloud.id
        )

        if include_status:
            result = {
                "name": cloud.name,
                "id": cloud.id,
                "created_at": cloud.created_at,
                "is_default": cloud.is_default,
                "resources": cloud_resources,
            }
        else:
            # Remove status fields from cloud resources for cleaner output.
            # Use `anyscale cloud status` to see full status information.
            status_fields_to_hide = {
                "created_at",
                "is_default",
                "operator_status",
                "operator_status_details",
            }
            for resource in cloud_resources:
                for field in status_fields_to_hide:
                    resource.pop(field, None)

            result = {
                "name": cloud.name,
                "id": cloud.id,
                "resources": cloud_resources,
            }

        if output:
            with open(output, "w") as f:
                yaml.dump(result, f, sort_keys=False)
        else:
            print(yaml.dump(result, sort_keys=False))

    except ValueError as e:
        log.error(f"Error retrieving cloud: {e}")


@cloud_cli.command(
    name="get",
    help="Get information about a specific cloud.",
    cls=AnyscaleCommand,
    example=command_examples.CLOUD_GET_CLOUD_EXAMPLE,
)
@click.option(
    "--name",
    "-n",
    help="Name of the cloud to get information about.",
    type=str,
    required=False,
)
@click.option(
    "--cloud-id",
    "--id",
    help="ID of the cloud to get information about.",
    type=str,
    required=False,
)
@click.option(
    "--output",
    "-o",
    help="File to write the output YAML to.",
    type=str,
    required=False,
)
def get_cloud(
    cloud_id: Optional[str], name: Optional[str], output: Optional[str]
) -> None:
    """
    Retrieve a cloud by its name or ID and display its details.

    This command outputs a simplified format suitable for use with `anyscale cloud update`.
    For full cloud status information including operator status, use `anyscale cloud status`.

    :param cloud_id: The ID of the cloud to retrieve.
    :param name: The name of the cloud to retrieve.
    """
    _get_cloud_info(cloud_id, name, output, include_status=False)


@cloud_cli.command(
    name="status",
    help="Get full status information about a specific cloud including operator status.",
    cls=AnyscaleCommand,
    example=command_examples.CLOUD_STATUS_EXAMPLE,
)
@click.option(
    "--name",
    "-n",
    help="Name of the cloud to get status for.",
    type=str,
    required=False,
)
@click.option(
    "--cloud-id",
    "--id",
    help="ID of the cloud to get status for.",
    type=str,
    required=False,
)
@click.option(
    "--output",
    "-o",
    help="File to write the output YAML to.",
    type=str,
    required=False,
)
def cloud_status(
    cloud_id: Optional[str], name: Optional[str], output: Optional[str]
) -> None:
    """
    Retrieve full status information for a cloud including operator status details.

    This command outputs all fields including created_at, is_default, operator_status,
    and operator_status_details. For a simplified format suitable for `anyscale cloud update`,
    use `anyscale cloud get`.

    :param cloud_id: The ID of the cloud to retrieve.
    :param name: The name of the cloud to retrieve.
    """
    _get_cloud_info(cloud_id, name, output, include_status=True)


@cloud_cli.command(
    name="get-default",
    help="Get the default cloud for your organization.",
    cls=AnyscaleCommand,
    example=command_examples.CLOUD_GET_DEFAULT_CLOUD_EXAMPLE,
)
def get_default_cloud() -> None:
    """
    Retrieve and display the default cloud configured for your organization.
    """
    try:
        default_cloud = anyscale.cloud.get_default()

        if not default_cloud:
            log.error("No default cloud found.")
            return

        cloud_dict = (
            default_cloud.to_dict()
            if hasattr(default_cloud, "to_dict")
            else default_cloud.__dict__
        )

        print(yaml.dump(cloud_dict, sort_keys=False))

    except ValueError as e:
        log.error(f"Error retrieving default cloud: {e}")


@cloud_cli.command(
    name="jobs-report",
    help=(
        "Generate a report of the jobs created in the last 7 days in HTML format. "
        "Shows unused CPU-hours, unused GPU-hours, and other data."
    ),
    cls=AnyscaleCommand,
    hidden=True,
)
@click.option(
    "--cloud-id",
    help="ID of the cloud to generate a report on.",
    type=str,
    required=True,
)
@click.option(
    "--csv",
    help="Outputs the report in CSV format.",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
)
@click.option(
    "--out",
    help="Output file name for the report. (Default jobs_report.html)",
    type=str,
    required=False,
    default=None,
)
@click.option(
    "--sort-by",
    help=(
        "Column to sort by. (Default created_at). "
        "created_at: Job creation time. "
        "gpu: Unused GPU hours. "
        "cpu: Unused CPU hours. "
        "instances: Number of instances."
    ),
    type=click.Choice(["created_at", "gpu", "cpu", "instances"], case_sensitive=False),
    required=False,
    default="created_at",
)
@click.option(
    "--sort-order",
    help="Sort order. (Default desc)",
    type=click.Choice(["asc", "desc"], case_sensitive=False),
    required=False,
    default="desc",
)
def generate_jobs_report(
    cloud_id: str, csv: bool, out: Optional[str], sort_by: str, sort_order: str
) -> None:
    """
    Generate a report of the jobs created in the last 7 days in HTML format.
    Shows unused CPU-hours, unused GPU-hours, and other data.
    :param cloud_id: The ID of the cloud to generate a report on.
    :param csv: Outputs the report in CSV format.
    :param out: Output file name for the report.
    """
    if out is None:
        out = "jobs_report.html" if not csv else "jobs_report.csv"

    try:
        CloudController().generate_jobs_report(
            cloud_id, csv, out, sort_by, sort_order == "asc"
        )
    except ValueError as e:
        log.error(f"Error generating jobs report: {e}")


@cloud_cli.command(
    name="terminate-system-cluster",
    help="Terminate the system cluster for a specific given cloud.",
    cls=AnyscaleCommand,
    example=command_examples.CLOUD_TERMINATE_SYSTEM_CLUSTER_EXAMPLE,
)
@click.option(
    "--cloud-id",
    "--id",
    help="ID of the cloud to terminate the system cluster for.",
    type=str,
    required=True,
)
@click.option(
    "-w",
    "--wait",
    required=False,
    default=False,
    type=bool,
    is_flag=True,
    help="Block this CLI command and print logs until the job finishes.",
)
def terminate_system_cluster(cloud_id: str, wait: Optional[bool]) -> None:
    """
    Terminate the system cluster for a specific cloud.

    :param cloud_id: The ID of the cloud to terminate the system cluster for.
    :param wait: If True, wait for the system cluster to be terminated before returning. Defaults to False.
    """
    try:
        anyscale.cloud.terminate_system_cluster(cloud_id, wait)
    except ValueError as e:
        log.error(f"Error terminating system cluster: {e}")
