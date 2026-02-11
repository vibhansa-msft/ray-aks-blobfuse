import ipaddress
import re
from typing import Iterable, List, Optional, Tuple

import click
from click import ClickException
from google.api_core.exceptions import NotFound, PermissionDenied
from google.cloud import compute_v1
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.constants import (
    PUBLIC_ACCESS_PREVENTION_ENFORCED,
    PUBLIC_ACCESS_PREVENTION_INHERITED,
)
from google.iam.v1.policy_pb2 import Binding
from googleapiclient.errors import HttpError

from anyscale.cli_logger import CloudSetupLogger
from anyscale.client.openapi_client.models import CloudAnalyticsEventCloudResource
from anyscale.shared_anyscale_utils.conf import ANYSCALE_CORS_ORIGIN
from anyscale.utils.cloud_utils import CloudSetupError
from anyscale.utils.gcp_managed_setup_utils import enable_project_apis
from anyscale.utils.gcp_utils import (
    binding_from_dictionary,
    check_policy_bindings,
    check_required_policy_bindings,
    GCP_REQUIRED_APIS,
    GCPLogger,
    GoogleCloudClientFactory,
)
from anyscale.utils.network_verification import (
    check_inbound_firewall_permissions,
    FirewallRule,
    GCP_SUBNET_CAPACITY,
    Protocol,
)


# Refer to https://cloud.google.com/python/docs/reference/file/latest/google.cloud.filestore_v1.
# services.cloud_filestore_manager.CloudFilestoreManagerClient, filestore name is in the format
# projects/{project_id}/locations/{location}/instances/{instance_id}.
_FILESTORE_NAME_REGEX_PATTERN = r"projects/(?P<project_id>[^/]+)/locations/(?P<location>[^/]+)/instances/(?P<instance_id>[^/]+)"

# A subnet with this purpose must be in the VPC for private load balancers to work.
# See: https://cloud.google.com/load-balancing/docs/proxy-only-subnets
PROXY_ONLY_SUBNET_PURPOSE = "REGIONAL_MANAGED_PROXY"


def check_gcp_cors_rules(cors_rules: List[dict]) -> Tuple[bool, str]:
    """
    Check if GCP GCS CORS rules are correctly configured for Anyscale.

    This is a shared helper used by both cloud verification and CORS update utilities.

    Args:
        cors_rules: List of CORS config dictionaries from GCS bucket

    Returns:
        Tuple of (is_correct, reason)
    """
    if not cors_rules:
        return False, "No CORS configuration exists"

    for cors_config in cors_rules:
        if (
            ANYSCALE_CORS_ORIGIN in cors_config.get("origin", [])
            and "*" in cors_config.get("responseHeader", [])
            and "GET" in cors_config.get("method", [])
        ):
            return True, "CORS already configured correctly"

    return False, "CORS missing required responseHeader for file viewer"


def verify_gcp_networking(  # noqa: PLR0911, PLR0913
    factory: GoogleCloudClientFactory,
    vpc_name: Optional[str],
    subnet_ids: Optional[List[str]],
    project_id: str,
    cloud_region: str,
    logger: GCPLogger,
    strict: bool = False,
    is_private_service_cloud: bool = False,
    ignore_capacity_errors: bool = False,
) -> bool:
    """Verify the existence and connectedness of the VPC & Subnet."""
    # TODO Verify Internet Gateway
    if not vpc_name:
        logger.internal.warning("No VPC provided. Please provide a VPC.")
        return False

    try:
        vpc = factory.compute_v1.NetworksClient().get(
            project=project_id, network=vpc_name
        )
    except NotFound:
        logger.log_resource_not_found_error("VPC", vpc_name, project_id)
        return False

    if not subnet_ids:
        logger.internal.warning(
            "No subnets provided. Please provide at least one subnet."
        )
        return False

    subnet_name = subnet_ids[0]  # TODO (congding): multiple subnets provided
    if len(subnet_ids) > 1:
        logger.internal.warning(
            "Multiple subnets provided. Only taking the first subnet and ignoring the rest."
        )
        if strict:
            return False
    try:
        subnet = factory.compute_v1.SubnetworksClient().get(
            project=project_id, subnetwork=subnet_name, region=cloud_region,
        )
    except NotFound:
        logger.log_resource_not_found_error("Subnet", subnet_name, project_id)
        return False

    if subnet.network != vpc.self_link:
        logger.internal.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_SUBNET,
            CloudSetupError.SUBNET_NOT_IN_VPC,
        )
        logger.internal.error(f"Subnet {subnet_name} is not part of {vpc_name}!")
        return False

    if is_private_service_cloud and not _get_proxy_only_subnet_in_vpc(
        factory, project_id, cloud_region, vpc_name
    ):
        logger.internal.warning(
            f"Could not find a subnet with purpose `{PROXY_ONLY_SUBNET_PURPOSE}` in the VPC {vpc_name}. A proxy-only subnet is required to run private Anyscale Services in your account. See https://cloud.google.com/load-balancing/docs/proxy-only-subnets for more details."
        )
        if strict:
            return False

    return _gcp_subnet_has_enough_capacity(
        subnet, logger.internal, ignore_capacity_errors
    )


def _gcp_subnet_has_enough_capacity(
    subnet: compute_v1.types.compute.Subnetwork,
    logger: CloudSetupLogger,
    ignore_capacity_errors: bool = False,
) -> bool:
    """Verify if the subnet provided has a large enough IP address block."""
    return ignore_capacity_errors or bool(
        GCP_SUBNET_CAPACITY.verify_network_capacity(
            cidr_block_str=subnet.ip_cidr_range,
            resource_name=subnet.name,
            logger=logger,
        )
    )


def _get_proxy_only_subnet_in_vpc(
    factory: GoogleCloudClientFactory,
    project_id: str,
    cloud_region: str,
    vpc_name: str,
) -> Optional[compute_v1.types.compute.Subnetwork]:
    # Check if there exists a subnet in the VPC and region with purpose `REGIONAL_MANAGED_PROXY`
    subnets = factory.compute_v1.SubnetworksClient().list(
        project=project_id, region=cloud_region
    )
    for subnet in subnets:
        subnet_vpc = subnet.network.split("/")[-1]
        if vpc_name == subnet_vpc and subnet.purpose == PROXY_ONLY_SUBNET_PURPOSE:
            return subnet
    return None


def verify_gcp_project(  # noqa: PLR0911
    factory: GoogleCloudClientFactory,
    project_id: str,
    enable_memorystore_api: bool,
    logger: GCPLogger,
    strict: bool = False,
) -> bool:
    """Verify if the project exists and is active.

    NOTE: This also checks that Compute Engine Service Agent is configured in the default way.
    """
    project_client = factory.resourcemanager_v3.ProjectsClient()
    try:
        project = project_client.get_project(name=f"projects/{project_id}")
    except (NotFound, PermissionDenied):
        logger.log_resource_not_found_error("Project", project_id)
        return False
    if project.state != project.State.ACTIVE:
        logger.internal.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_PROJECT,
            CloudSetupError.PROJECT_NOT_ACTIVE,
        )
        logger.internal.error(
            f"Project {project_id} is in state: {project.state}, not active"
        )
        return False

    iam_policies = project_client.get_iam_policy(resource=f"projects/{project_id}")
    project_number = project.name.split("/")[1]

    if not check_policy_bindings(
        iam_policies.bindings,
        f"serviceAccount:service-{project_number}@compute-system.iam.gserviceaccount.com",
        {"roles/compute.serviceAgent"},
    ):
        logger.internal.warning(
            "The Compute Engine Service Agent does not have the standard IAM Role of 'roles/compute.serviceAgent' in your project.\n"
            "This is not recommended by Google and may result in an inability for Compute Engine to function properly:\n"
            "https://cloud.google.com/compute/docs/access/service-accounts#compute_engine_service_account"
        )
        if strict:
            return False

    # Verify that APIs are Enabled
    service_usage_client = factory.build("serviceusage", "v1")
    if enable_memorystore_api:
        apis = GCP_REQUIRED_APIS + ["redis.googleapis.com"]  # Memorystore for Redis
    else:
        apis = GCP_REQUIRED_APIS
    response = (
        service_usage_client.services()
        .batchGet(
            parent=f"projects/{project_id}",
            names=[f"projects/{project_id}/services/{api}" for api in apis],
        )
        .execute()
    )

    unenabled_apis = [
        service["config"]["name"]
        for service in response["services"]
        if service["state"] != "ENABLED"
    ]
    if len(unenabled_apis) == 0:
        return True
    logger.spinner.stop()
    if not logger.yes and not click.confirm(
        f"The following APIs are not enabled in your project: {unenabled_apis}\n"
        "Anyscale won't function properly if you don't enable these APIs.\n"
        "Would you like to enable these APIs and proceed?"
    ):
        logger.spinner.start()
        logger.internal.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_PROJECT,
            CloudSetupError.API_NOT_ENABLED,
        )
        return False

    spinner_text = logger.spinner.text
    logger.spinner.text = "Enabling APIs..."
    logger.spinner.start()

    # enable all required apis, the operation is idompotent
    try:
        enable_project_apis(
            factory, project_id, logger.internal, enable_memorystore_api
        )
    except ClickException as e:
        logger.internal.error(e.message)
        logger.spinner.text = spinner_text
        return False

    # APIs are enabled successfully
    logger.spinner.text = spinner_text
    return True


def verify_gcp_access_service_account(
    factory: GoogleCloudClientFactory,
    anyscale_access_service_account: Optional[str],
    project_id: str,
    logger: GCPLogger,
) -> bool:
    """Verify if the Service Account meant to grant Anyscale access to this cloud has access to the specified project.

    NOTE: We verify that this service account can call signBlob on itself because this is necessary for downloading logs.
    """
    service_account_client = factory.build("iam", "v1").projects().serviceAccounts()
    try:
        service_account_iam_policy = service_account_client.getIamPolicy(
            resource=f"projects/-/serviceAccounts/{anyscale_access_service_account}"
        ).execute()
    except HttpError as e:
        if e.status_code == 404:
            logger.log_resource_not_found_error(
                "Anyscale Access Service Account", anyscale_access_service_account
            )
        else:
            logger.internal.log_resource_exception(
                CloudAnalyticsEventCloudResource.GCP_SERVICE_ACCOUNT, e
            )
            logger.internal.error(str(e))
        return False

    if service_account_iam_policy.get("bindings") is None or not check_policy_bindings(
        binding_from_dictionary(service_account_iam_policy["bindings"]),
        f"serviceAccount:{anyscale_access_service_account}",
        {"roles/iam.serviceAccountTokenCreator"},
    ):
        logger.confirm_missing_permission(
            f"Service Account {anyscale_access_service_account} must have the `iam.serviceAccounts.signBlob` permission to perform log download from the Anyscale platform.\n"
            "Please grant the `roles/iam.serviceAccountTokenCreator` role to this Service Account to pass this check."
        )

    project_client = factory.resourcemanager_v3.ProjectsClient()
    iam_policies = project_client.get_iam_policy(resource=f"projects/{project_id}")
    if not check_policy_bindings(
        iam_policies.bindings,
        f"serviceAccount:{anyscale_access_service_account}",
        {"roles/editor", "roles/owner"},
    ):
        logger.confirm_missing_permission(
            f"Service Account {anyscale_access_service_account} must have either `roles/editor` or `roles/owner` roles on the Project {project_id}.\n"
            "A more fine-grained set of permissions may be possible, but may break Anyscale's functionality."
        )

    return True


def verify_gcp_dataplane_service_account(
    factory: GoogleCloudClientFactory,
    service_account: Optional[str],
    project_id: str,
    logger: GCPLogger,
    strict: bool = False,
) -> bool:
    """Verify that the Service Account for compute instances exists *in* the specified project.

    Compute Engine's ability to use this role
    This relies on the fact that Compute Engine Service Agent has the roles/compute.serviceAgent Role
    """
    service_account_client = factory.build("iam", "v1").projects().serviceAccounts()
    try:
        resp = service_account_client.get(
            name=f"projects/-/serviceAccounts/{service_account}"
        ).execute()
    except HttpError as e:
        if e.status_code == 404:
            logger.log_resource_not_found_error(
                "Dataplane Service Account", service_account
            )
        else:
            logger.internal.log_resource_exception(
                CloudAnalyticsEventCloudResource.GCP_SERVICE_ACCOUNT, e
            )
            logger.internal.error(str(e))
        return False

    if resp["projectId"] != project_id:
        logger.internal.warning(
            "Service Account {} is in project {}, not {}. Please ensure that `constraints/iam.disableCrossProjectServiceAccountUsage` is not enforced.\n"
            "See: https://cloud.google.com/iam/docs/attach-service-accounts#enabling-cross-project for more information.".format(
                service_account, resp["projectId"], project_id
            )
        )
        if strict:
            return False

    project_client = factory.resourcemanager_v3.ProjectsClient()
    project_iam_bindings = project_client.get_iam_policy(
        resource=f"projects/{project_id}"
    )
    if not check_policy_bindings(
        project_iam_bindings.bindings,
        f"serviceAccount:{service_account}",
        {"roles/artifactregistry.reader", "roles/viewer"},
    ):
        logger.internal.warning(
            f"The dataplane Service Account {service_account} does not have `roles/artifactregistry.reader` or `roles/viewer` on {project_id}.\n"
            "This is safe to ignore if you are not using your own container images or are configuring access a different way."
        )
        if strict:
            return False

    return True


def verify_firewall_policy(  # noqa: PLR0911, PLR0912, C901, PLR0913
    factory: GoogleCloudClientFactory,
    firewall_policy_ids: Optional[List[str]],
    vpc_name: Optional[str],
    subnet_ids: Optional[List[str]],
    project_id: str,
    cloud_region: str,
    use_shared_vpc: bool,
    is_private_service_cloud: bool,
    logger: GCPLogger,
    strict: bool = False,
) -> bool:
    """Checks if the given firewall exists at either the Global or Regional level."""
    if not firewall_policy_ids:
        logger.internal.warning(
            "No firewall policies provided. Please provide at least one firewall policy."
        )
        return False

    firewall_policy = firewall_policy_ids[
        0
    ]  # TODO (congding): multiple firewall ids provided
    if len(firewall_policy_ids) > 1:
        logger.internal.warning(
            "Multiple firewall policies provided. Only taking the first firewall policy and ignoring the rest."
        )
        if strict:
            return False

    firewall = compute_v1.types.compute.FirewallPolicy()
    try:
        firewall = factory.compute_v1.NetworkFirewallPoliciesClient().get(
            project=project_id, firewall_policy=firewall_policy
        )
    except NotFound:
        logger.internal.info(
            f"Global firweall policy {firewall_policy} not found, trying in region: {cloud_region}."
        )
    if not firewall:
        try:
            firewall = factory.compute_v1.RegionNetworkFirewallPoliciesClient().get(
                project=project_id, firewall_policy=firewall_policy, region=cloud_region
            )
        except NotFound:
            logger.log_resource_not_found_error(
                "Firewall Policy", firewall_policy, project_id
            )
            return False

    if not any(
        association.attachment_target.split("/")[-1] == vpc_name
        for association in firewall.associations
    ):
        logger.internal.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_FIREWALL_POLICY,
            CloudSetupError.FIREWALL_NOT_ASSOCIATED_WITH_VPC,
        )
        logger.internal.error(
            "Firewall policy {} is not associated with the VPC {}, but is associated with the following VPCs: {}".format(
                firewall_policy, vpc_name, firewall.associations
            )
        )
        return False

    if not subnet_ids:
        logger.internal.warning(
            "No subnets provided. Please provide at least one subnet."
        )
        return False

    subnet_obj = factory.compute_v1.SubnetworksClient().get(
        project=project_id, subnetwork=subnet_ids[0], region=cloud_region
    )  # TODO (congding): multiple subnets provided
    subnet = ipaddress.ip_network(subnet_obj.ip_cidr_range)

    rules = _firewall_rules_from_proto_resp(firewall.rules)
    if not check_inbound_firewall_permissions(
        rules,
        Protocol.tcp,
        {22},
        ipaddress.ip_network("0.0.0.0/0"),
        check_ingress_source_range=False,
    ):
        logger.internal.warning(
            "Firewall policy does not allow inbound access on port 22. Certain features may be unavailable."
        )
        if strict:
            return False
    if not check_inbound_firewall_permissions(
        rules,
        Protocol.tcp,
        {443},
        ipaddress.ip_network("0.0.0.0/0"),
        check_ingress_source_range=False,
    ):
        logger.internal.warning(
            "Firewall policy does not allow inbound access on port 443. Accessing features like Jupyter & Ray Dashboard may not work."
        )
        if strict:
            return False

    # If shared VPC is used, we need to check the firewall rule for Anyscale services is added
    # Since our service account cannot update the firewall policy in host project, so we need to make sure the rule is added during cloud setup
    if use_shared_vpc and not (
        check_inbound_firewall_permissions(
            rules, Protocol.tcp, {8000}, ipaddress.ip_network("35.191.0.0/16")
        )
        and check_inbound_firewall_permissions(
            rules, Protocol.tcp, {8000}, ipaddress.ip_network("130.211.0.0/22")
        )
    ):
        logger.internal.warning(
            "Firewall policy does not allow inbound access on port 8000 from '35.191.0.0/16' and '130.211.0.0/22'. "
            "This is required for Anyscale services to work correctly. "
            "See https://cloud.google.com/load-balancing/docs/health-checks#fw-rule for more details."
        )
        if strict:
            return False

    if use_shared_vpc and is_private_service_cloud:
        if not vpc_name:
            logger.internal.warning("No VPC provided. Please provide a VPC.")
            return False

        proxy_only_subnet = _get_proxy_only_subnet_in_vpc(
            factory, project_id, cloud_region, vpc_name
        )
        if not proxy_only_subnet:
            logger.internal.warning(
                f"Could not find a subnet with purpose `{PROXY_ONLY_SUBNET_PURPOSE}` in the VPC {vpc_name}. A proxy-only subnet is required to run private Anyscale Services in your account. See https://cloud.google.com/load-balancing/docs/proxy-only-subnets for more details."
            )
            if strict:
                return False
        elif not check_inbound_firewall_permissions(
            rules,
            Protocol.tcp,
            {8000},
            ipaddress.ip_network(proxy_only_subnet.ip_cidr_range),
        ):
            logger.internal.warning(
                "Firewall policy does not allow inbound access from the proxy-only subnet on port 8000. "
                "This is required for private Anyscale services to work correctly. "
            )
            if strict:
                return False

    if not check_inbound_firewall_permissions(rules, Protocol.tcp, None, subnet):
        logger.internal.error(
            "Firewall policy does not allow for internal communication, see https://cloud.google.com/vpc/docs/using-firewalls#common-use-cases-allow-internal for how to configure such a rule."
        )
        logger.internal.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_FIREWALL_POLICY,
            CloudSetupError.INTERNAL_COMMUNICATION_NOT_ALLOWED,
        )
        return False
    return True


def _firewall_rules_from_proto_resp(
    rules: Iterable[compute_v1.types.compute.FirewallPolicyRule],
) -> List[FirewallRule]:
    """Convert Firewall Rules into a Python class suitable for comparison.

    NOTE: We only check "ALLOW" rules & do not check priority.
    """
    return [
        FirewallRule(
            direction=rule.direction,
            protocol=Protocol.from_val(l4config.ip_protocol),
            ports=l4config.ports,
            network=ipaddress.ip_network(addr),
        )
        for rule in rules
        if rule.action == "allow"
        for addr in rule.match.src_ip_ranges
        for l4config in (
            rule.match.layer4_configs
            or [
                compute_v1.types.compute.FirewallPolicyRuleMatcherLayer4Config(
                    ip_protocol="all"
                )
            ]
        )
    ]


def verify_filestore(
    factory: GoogleCloudClientFactory,
    file_store_instance_name: str,
    vpc_name: Optional[str],
    cloud_region: str,
    logger: GCPLogger,
    strict: bool = False,
) -> bool:
    """Verify Filestore exists & that it is connected to the correct VPC.

    TODO: Warn about Filestore size if it is 'too small'."""
    client = factory.filestore_v1.CloudFilestoreManagerClient()
    try:
        file_store = client.get_instance(name=file_store_instance_name)
    except NotFound:
        logger.log_resource_not_found_error("Filestore", file_store_instance_name)
        return False

    file_store_match = re.match(_FILESTORE_NAME_REGEX_PATTERN, file_store.name)
    if file_store_match:
        filestore_location = file_store_match.group("location")
        # Filestore location can be region or zone.
        if cloud_region not in filestore_location:
            logger.internal.warning(
                f"Filestore is in {filestore_location}, but this cloud is being set up in {cloud_region}. This can result in cross-region network egress charges, refer to https://cloud.google.com/vpc/network-pricing#egress-within-gcp."
            )
            if strict:
                return False
    else:
        logger.internal.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_FILESTORE,
            CloudSetupError.FILESTORE_NAME_MALFORMED,
        )
        logger.internal.error(
            f"Failed to get filestore location, since filestore name doesn't match format {_FILESTORE_NAME_REGEX_PATTERN}!"
        )
        return False

    file_store_networks = [v.network for v in file_store.networks]
    if not any(vpc_name == network.split("/")[-1] for network in file_store_networks):
        logger.internal.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_FILESTORE,
            CloudSetupError.FILESTORE_NOT_CONNECTED_TO_VPC,
        )
        logger.internal.error(
            f"Filestore is not connected to {vpc_name}, but to {file_store_networks}. "
            f"This cannot be edited on an existing Filestore instance. Please recreate the filestore and connect it to {vpc_name}."
        )
        return False

    return True


def verify_cloud_storage(  # noqa: PLR0911, PLR0912, PLR0913
    factory: GoogleCloudClientFactory,
    bucket_name: Optional[str],
    controlplane_service_account: Optional[str],
    dataplane_service_account: Optional[str],
    project_id: str,
    cloud_region: str,
    logger: GCPLogger,
    strict: bool = False,
):
    """Verify that the Google Cloud Storage Bucket exists & raises warnings about improper configurations."""
    bucket_client = factory.storage.Client(project_id)
    try:
        bucket = bucket_client.get_bucket(bucket_name)
    except NotFound:
        logger.log_resource_not_found_error("Google Cloud Storage Bucket", bucket_name)
        return False

    if not bucket.iam_configuration.uniform_bucket_level_access_enabled:
        logger.internal.warning(
            f"Bucket {bucket_name}, does not have Uniform Bucket Access enabled, "
            "this impedes Anyscale's ability to verify bucket access."
        )

    public_access_protection = bucket.iam_configuration.public_access_prevention
    if public_access_protection not in (
        PUBLIC_ACCESS_PREVENTION_INHERITED,
        PUBLIC_ACCESS_PREVENTION_ENFORCED,
    ):
        logger.internal.warning(
            f"Bucket {bucket_name} has public access prevention set to {public_access_protection}, not enforced or inherited."
        )
        if strict:
            return False

    correct_region, bucket_region = _check_bucket_region(bucket, cloud_region)
    if not correct_region:
        logger.internal.warning(
            f"Bucket {bucket_name} is in region {bucket_region}, but this cloud is being set up in {cloud_region}."
            "This can result in degraded cluster launch & logging performance."
        )
        if strict:
            return False

    has_correct_cors, _cors_reason = check_gcp_cors_rules(bucket.cors or [])
    if not has_correct_cors:
        logger.internal.warning(
            f"Bucket {bucket_name} does not have the correct CORS rule for Anyscale. This is safe to ignore if you are not using Anyscale UI.\n"
            "If you are using the UI, please create the correct CORS rule for Anyscale according to https://docs.anyscale.com/cloud-deployment/gcp/deploy-cloud?cloud-deployment=custom#4-create-an-anyscale-cloud"
        )
        if strict:
            return False

    iam_policy = bucket.get_iam_policy()
    iam_bindings = binding_from_dictionary(iam_policy.bindings)

    permission_warning = (
        "The {location} Service Account {email} requires the following permissions on Bucket {bucket} to operate correctly:\n"
        "* storage.buckets.get\n* storage.objects.[ get | list | create ]\n* storage.multipartUploads.[ abort | create | listParts ]\n"
        "We suggest granting the predefined roles of `roles/storage.legacyBucketReader` and `roles/storage.objectAdmin`."
    )

    if not _verify_service_account_on_bucket(
        f"serviceAccount:{controlplane_service_account}", iam_bindings
    ):
        logger.confirm_missing_permission(
            permission_warning.format(
                location="Anyscale access",
                email=controlplane_service_account,
                bucket=bucket_name,
            )
        )
        if strict:
            return False

    if not _verify_service_account_on_bucket(
        f"serviceAccount:{dataplane_service_account}", iam_bindings
    ):
        logger.confirm_missing_permission(
            permission_warning.format(
                location="Dataplane",
                email=dataplane_service_account,
                bucket=bucket_name,
            )
        )
        if strict:
            return False

    return True


def _verify_service_account_on_bucket(
    service_account: str, iam_bindings: List[Binding]
) -> bool:
    """Verifies that the given service account has roles that ensure the following list of permissions:
    * storage.buckets.get: Get bucket info
    * storage.objects.[ get | list | create ]
    * storage.multipartUploads.[ abort | create | listParts ]
    """
    if check_policy_bindings(iam_bindings, service_account, {"roles/storage.admin"},):
        return True
    return any(
        check_required_policy_bindings(iam_bindings, service_account, combination)
        for combination in [
            {
                # Best Permissions (legacy for `buckets.get`)
                "roles/storage.legacyBucketReader",
                "roles/storage.objectViewer",
                "roles/storage.objectCreator",
            },
            {
                # Most Brief permissions (legacy for `buckets.get`)
                "roles/storage.objectAdmin",
                "roles/storage.legacyBucketReader",
            },
            {
                # Optimal Legacy Roles
                "roles/storage.legacyBucketWriter",
                "roles/storage.legacyObjectReader",
            },
            {
                # Legacy Roles (extra object permissions)
                "roles/storage.legacyBucketWriter",
                "roles/storage.legacyObjectOwner",
            },
            {
                # Legacy Roles (extra bucket permissions)
                "roles/storage.legacyBucketOwner",
                "roles/storage.legacyObjectReader",
            },
            {
                # Legacy Roles (extra bucket/object permissions)
                "roles/storage.legacyBucketOwner",
                "roles/storage.legacyObjectOwner",
            },
        ]
    )


def _check_bucket_region(bucket: Bucket, region: str) -> Tuple[bool, str]:
    if bucket.location_type == "dual-region":
        return region.upper() in bucket.data_locations, ",".join(bucket.data_locations)
    elif bucket.location_type == "region":
        return region.upper() == bucket.location, bucket.location

    # Bucket is `multi-region`, so check if the location (`EU`, `ASIA`, `US`)
    # is the 'prefix' of the region.
    return region.upper().startswith(bucket.location), bucket.location


def verify_memorystore(  # noqa: PLR0911
    factory: GoogleCloudClientFactory,
    redis_instance_name: str,
    cloud_vpc_name: Optional[str],
    logger: GCPLogger,
    strict: bool = False,
):
    """Verify that the Google Memorystore exists & raises warnings about improper configurations."""
    client = factory.redis_v1.CloudRedisClient()
    try:
        redis_instance = client.get_instance(name=redis_instance_name)
    except (NotFound, PermissionDenied):
        logger.log_resource_not_found_error("Memorystore", redis_instance_name)
        return False

    # Verify memorystore is in the same VPC as the cloud
    redis_vpc_name = redis_instance.authorized_network.split("/")[-1]
    if cloud_vpc_name != redis_vpc_name:
        logger.internal.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_MEMORYSTORE,
            CloudSetupError.MEMORYSTORE_NOT_CONNECTED_TO_VPC,
        )
        logger.internal.error(
            f"Memorystore is not connected to {cloud_vpc_name}, but to {redis_vpc_name}. "
            f"This cannot be edited on an existing Memorystore instance. Please recreate the Memorystore and connect it to {cloud_vpc_name}."
        )
        return False

    # Verify memorystore is the standard tier
    if redis_instance.tier.name != "STANDARD_HA":
        logger.internal.warning(
            "Memorystore is not in the standard tier. This can result in degraded performance and availability. Note that this cannot be changed on an existing Memorystore instance."
        )
        if strict:
            return False

    # Verify that memorystore instance has the maxmemory-policy set to allkeys-lru
    if redis_instance.redis_configs.get("maxmemory-policy") != "allkeys-lru":
        logger.internal.warning(
            "Memorystore does not have the maxmemory-policy set to allkeys-lru. This can result in degraded performance and availability."
        )
        if strict:
            return False

    # Verify that the memorystore instance has the read replica mode turned on
    if redis_instance.read_replicas_mode.name != "READ_REPLICAS_ENABLED":
        logger.internal.error(
            "Memorystore does not have the read replica mode turned on. This is required for ray head node HA."
        )
        logger.internal.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_MEMORYSTORE,
            CloudSetupError.MEMORYSTORE_READ_REPLICAS_DISABLED,
        )
        return False

    # Verify that the memorystore has tls disabled
    if redis_instance.transit_encryption_mode.name != "DISABLED":
        logger.internal.error(
            "Memorystore has TLS enabled. Please create a new memorystore instance with TLS disabled."
        )
        return False

    return True
