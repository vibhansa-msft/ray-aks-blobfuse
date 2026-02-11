import difflib
import json
import pprint
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError
from click import ClickException

from anyscale.aws_iam_policies import (
    AMAZON_ECR_READONLY_ACCESS_POLICY_NAME,
    ANYSCALE_IAM_PERMISSIONS_EC2_STEADY_STATE,
    ANYSCALE_IAM_PERMISSIONS_SERVICE_STEADY_STATE,
    get_anyscale_aws_iam_assume_role_policy,
    get_anyscale_iam_permissions_ec2_restricted,
)
from anyscale.cli_logger import CloudSetupLogger
from anyscale.client.openapi_client.models.aws_memory_db_cluster_config import (
    AWSMemoryDBClusterConfig,
)
from anyscale.client.openapi_client.models.cloud_analytics_event_cloud_resource import (
    CloudAnalyticsEventCloudResource,
)
from anyscale.client.openapi_client.models.subnet_id_with_availability_zone_aws import (
    SubnetIdWithAvailabilityZoneAWS,
)
from anyscale.shared_anyscale_utils.aws import AwsRoleArn
from anyscale.shared_anyscale_utils.conf import (
    ANYSCALE_CORS_EXPOSE_HEADERS,
    ANYSCALE_CORS_ORIGIN,
)
from anyscale.util import (  # pylint:disable=private-import
    _get_subnet,
    contains_control_plane_role,
    filter_actions_associated_with_role,
    filter_actions_from_policy_document,
    verify_data_plane_role_assume_role_policy,
)
from anyscale.utils.cloud_utils import CloudSetupError
from anyscale.utils.network_verification import (
    AWS_SUBNET_CAPACITY,
    AWS_VPC_CAPACITY,
)
from anyscale.utils.s3 import verify_s3_access


# This needs to be kept in sync with the Ray autoscaler in
# https://github.com/ray-project/ray/blob/eb9c5d8fa70b1c360b821f82c7697e39ef94b25e/python/ray/autoscaler/_private/aws/config.py
# It should go away with the SSM refactor.
DEFAULT_RAY_IAM_ROLE = "ray-autoscaler-v1"

S3_ARN_PREFIX = "arn:aws:s3:::"
S3_STORAGE_PREFIX = "s3://"
GCS_STORAGE_PREFIX = "gs://"

HTTPS_INGRESS_PORT = 443


def check_aws_cors_rules(cors_rules: List[Any]) -> Tuple[bool, str]:
    """
    Check if AWS S3 CORS rules are correctly configured for Anyscale.

    This is a shared helper used by both cloud verification and CORS update utilities.

    Args:
        cors_rules: List of CORS rule dictionaries from S3 bucket (can be dict or CORSRuleTypeDef)

    Returns:
        Tuple of (is_correct, reason)
    """
    if not cors_rules:
        return False, "No CORS configuration exists"

    for rule in cors_rules:
        if not isinstance(rule, dict):
            return False, f"Malformed CORS rule: {rule}"
        expose_headers = rule.get("ExposeHeaders", [])
        if (
            ANYSCALE_CORS_ORIGIN in rule.get("AllowedOrigins", [])
            and "*" in rule.get("AllowedHeaders", [])
            and "GET" in rule.get("AllowedMethods", [])
            and all(h in expose_headers for h in ANYSCALE_CORS_EXPOSE_HEADERS)
        ):
            return True, "CORS already configured correctly"

    return False, "CORS missing required ExposeHeaders for file viewer"


def compare_dicts_diff(d1: Dict[Any, Any], d2: Dict[Any, Any]) -> str:
    """Returns a string representation of the difference of the two dictionaries.
    Example:

    Input:
    print(compare_dicts_diff({"a": {"c": 1}, "b": 2}, {"a": {"c": 2}, "d": 3}))

    Output:
    - {'a': {'c': 1}, 'b': 2}
    ?             ^    ^   ^

    + {'a': {'c': 2}, 'd': 3}
    ?             ^    ^   ^
    """

    return "\n" + "\n".join(
        difflib.ndiff(pprint.pformat(d1).splitlines(), pprint.pformat(d2).splitlines())
    )


AWS_RESOURCE_DICT: Dict[str, str] = {
    "VPC": CloudAnalyticsEventCloudResource.AWS_VPC,
    "Subnet": CloudAnalyticsEventCloudResource.AWS_SUBNET,
    "Security group": CloudAnalyticsEventCloudResource.AWS_SECURITY_GROUP,
    "S3 bucket": CloudAnalyticsEventCloudResource.AWS_S3_BUCKET,
    "EFS": CloudAnalyticsEventCloudResource.AWS_EFS,
    "CloudFormation stack": CloudAnalyticsEventCloudResource.AWS_CLOUDFORMATION,
    "MemoryDB cluster": CloudAnalyticsEventCloudResource.AWS_MEMORYDB,
}


def log_resource_not_found_error(
    resource_name: str, resource_id: str, logger: CloudSetupLogger
) -> None:
    resource = AWS_RESOURCE_DICT.get(resource_name)
    if resource:
        logger.log_resource_error(resource, CloudSetupError.RESOURCE_NOT_FOUND)
    logger.error(
        f"Could not find {resource_name} with id {resource_id}. Please validate that you're using the correct AWS account/credentials and that the resource values are correct"
    )


def verify_aws_vpc(
    aws_vpc_id: Optional[str],
    boto3_session: boto3.Session,
    logger: CloudSetupLogger,
    ignore_capacity_errors: bool = False,  # TODO: Probably don't do this forever. Its kinda hacky
    strict: bool = False,  # strict is currently unused # noqa: ARG001
) -> bool:
    logger.info("Verifying VPC ...")
    if not aws_vpc_id:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_VPC,
            CloudSetupError.MISSING_CLOUD_RESOURCE_ID,
        )
        logger.error("Missing VPC id.")
        return False

    ec2 = boto3_session.resource("ec2")
    vpc = ec2.Vpc(aws_vpc_id)

    # Verify the VPC exists
    try:
        vpc.load()
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidVpcID.NotFound":
            log_resource_not_found_error("VPC", aws_vpc_id, logger)
            return False
        else:
            logger.log_resource_exception(CloudAnalyticsEventCloudResource.AWS_VPC, e)
        raise e

    # Verify that the VPC has "enough" capacity.
    if ignore_capacity_errors or AWS_VPC_CAPACITY.verify_network_capacity(
        cidr_block_str=vpc.cidr_block, resource_name=vpc.id, logger=logger
    ):
        logger.info(f"VPC {vpc.id} verification succeeded.")
        return True
    return False


def _get_subnets_from_subnet_ids(
    subnet_ids: List[str], region: str, logger: CloudSetupLogger
) -> List[Any]:
    return [
        _get_subnet(subnet_arn=subnet_id, region=region, logger=logger)
        for subnet_id in subnet_ids
    ]


def verify_aws_subnets(  # noqa: PLR0911, PLR0912
    aws_vpc_id: Optional[str],
    aws_subnet_ids: List[str],
    region: str,
    is_private_network: bool,
    logger: CloudSetupLogger,
    ignore_capacity_errors: bool = False,  # TODO: Probably don't do this forever. Its kinda hacky
    strict: bool = False,
) -> bool:
    """Verify the subnets cloud resource of a cloud."""

    logger.info("Verifying subnets ...")

    if not aws_vpc_id:
        logger.error("Missing VPC ID.")
        return False

    if not aws_subnet_ids:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_SUBNET,
            CloudSetupError.MISSING_CLOUD_RESOURCE_ID,
        )
        logger.error("Missing subnet IDs.")
        return False

    # We must have at least 2 subnets since services requires 2 different subnets to setup ALB.
    if len(aws_subnet_ids) < 2:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_SUBNET, CloudSetupError.ONLY_ONE_SUBNET
        )
        logger.error(
            "Need at least 2 subnets for a cloud. This is required for Anyscale services to function properly."
        )
        return False

    subnets = _get_subnets_from_subnet_ids(
        subnet_ids=aws_subnet_ids, region=region, logger=logger
    )
    subnet_azs = set()

    for subnet, subnet_id in zip(subnets, aws_subnet_ids):
        # Verify subnet exists
        if not subnet:
            log_resource_not_found_error("Subnet", subnet_id, logger)
            return False

        # Verify the Subnet has "enough" capacity.
        if (
            not AWS_SUBNET_CAPACITY.verify_network_capacity(
                cidr_block_str=subnet.cidr_block, resource_name=subnet.id, logger=logger
            )
            and not ignore_capacity_errors
        ):
            return False

        # Verify that the subnet is in the provided VPC all of these are in the same VPC.
        if subnet.vpc_id != aws_vpc_id:
            logger.log_resource_error(
                CloudAnalyticsEventCloudResource.AWS_SUBNET,
                CloudSetupError.SUBNET_NOT_IN_VPC,
            )
            logger.error(
                f"The subnet {subnet_id} is not in a vpc of this cloud. The vpc of this subnet is {subnet.vpc_id} and the vpc of this cloud is {aws_vpc_id}."
            )
            return False

        # Verify that the subnet is auto-assigning public IP addresses if it's not private.
        if not is_private_network and not subnet.map_public_ip_on_launch:
            logger.warning(
                f"The subnet {subnet_id} does not have the 'Auto-assign Public IP' option enabled. This is not currently supported."
            )
            if strict:
                return False
        if is_private_network and subnet.map_public_ip_on_launch:
            logger.warning(
                f"The private subnet {subnet_id} shouldn't have 'Auto-assign Public IP' option enabled."
                "Please remove the `--private-network` flag if you don't want to deploy your Ray clusters in private subnets."
            )
            if strict:
                return False

        # Success!
        logger.info(f"Subnet {subnet.id}'s verification succeeded.")
        subnet_azs.add(subnet.availability_zone)

    if len(subnet_azs) < 2:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_SUBNET, CloudSetupError.ONLY_ONE_AZ,
        )
        logger.error(
            "Subnets should be in at least 2 Availability Zones. This is required for Anyscale services to function properly."
        )
        return False

    logger.info(f"Subnets {aws_subnet_ids} verification succeeded.")
    return True


def associate_aws_subnets_with_azs(
    aws_subnet_ids: List[str], region: str, logger: CloudSetupLogger
) -> List[SubnetIdWithAvailabilityZoneAWS]:
    """This function combines the subnets with its availability zone.
    """

    subnets = _get_subnets_from_subnet_ids(
        subnet_ids=aws_subnet_ids, region=region, logger=logger
    )

    # combine subnet and its availability zone
    subnet_ids_with_availability_zones = [
        SubnetIdWithAvailabilityZoneAWS(
            subnet_id=subnet.id, availability_zone=subnet.availability_zone,
        )
        for subnet in subnets
    ]

    return subnet_ids_with_availability_zones


def _get_role_from_arn(
    role_arn: str, boto3_session: boto3.Session, logger: CloudSetupLogger,
) -> Any:
    iam = boto3_session.resource("iam")

    # Validate the role exists.
    # `.load()` will throw an exception if the Role does not exist.
    try:
        role = iam.Role(AwsRoleArn.from_string(role_arn).to_role_name())
        role.load()
        return role
    except ClientError as e:
        logger.log_resource_exception(CloudAnalyticsEventCloudResource.AWS_IAM_ROLE, e)
        if e.response["Error"]["Code"] == "NoSuchEntity":
            logger.error(f"Could not find role: {role.name if role else 'unknown'}")
            return None
        raise e


def verify_aws_iam_roles(  # noqa: PLR0911, PLR0912
    control_plane_role: Optional[str],
    data_plane_role: Optional[str],
    boto3_session: boto3.Session,
    anyscale_aws_account: str,
    logger: CloudSetupLogger,
    cloud_id: str,
    strict: bool = False,
    _use_strict_iam_permissions: bool = False,
) -> bool:

    logger.info("Verifying IAM roles ...")
    if not control_plane_role:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_IAM_ROLE,
            CloudSetupError.MISSING_CLOUD_RESOURCE_ID,
        )
        logger.error("Missing Anyscale IAM role.")
        return False
    if not data_plane_role:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_IAM_ROLE,
            CloudSetupError.MISSING_CLOUD_RESOURCE_ID,
        )
        logger.error("Missing Cluster Node IAM role.")
        return False
    accounts = {
        AwsRoleArn.from_string(role).account_id
        for role in [control_plane_role, data_plane_role]
    }
    if len(accounts) != 1:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_IAM_ROLE,
            CloudSetupError.IAM_ROLE_ACCOUNT_MISMATCH,
        )
        logger.error(
            f"All IAM roles must be in the same AWS account: {control_plane_role}, {data_plane_role}"
        )
        return False

    # verifying control plane role: anyscale iam role
    anyscale_iam_role = _get_role_from_arn(control_plane_role, boto3_session, logger)
    if not anyscale_iam_role:
        return False
    assume_role_policy_document = anyscale_iam_role.assume_role_policy_document
    if not contains_control_plane_role(
        assume_role_policy_document=assume_role_policy_document,
        anyscale_aws_account=anyscale_aws_account,
    ):
        logger.warning(
            f"Anyscale IAM role {anyscale_iam_role.arn} does not contain expected assume role policy. It must allow assume role from arn:aws:iam::{anyscale_aws_account}:root."
        )
        expected_assume_role_policy_document = get_anyscale_aws_iam_assume_role_policy(
            anyscale_aws_account=anyscale_aws_account
        )
        logger.warning(
            compare_dicts_diff(
                assume_role_policy_document, expected_assume_role_policy_document
            )
        )
        if strict:
            return False

    # Verify EC2 steady state permissions
    # If permissions are missing, log warning message
    anyscale_iam_permissions_ec2 = ANYSCALE_IAM_PERMISSIONS_EC2_STEADY_STATE
    if _use_strict_iam_permissions:
        anyscale_iam_permissions_ec2 = get_anyscale_iam_permissions_ec2_restricted(
            cloud_id
        )

    allow_actions_expected = filter_actions_from_policy_document(
        anyscale_iam_permissions_ec2
    )
    allow_actions_on_role = filter_actions_associated_with_role(
        boto3_session, anyscale_iam_role
    )
    allow_actions_missing = allow_actions_expected - allow_actions_on_role

    if allow_actions_missing:
        logger.warning(
            f"IAM role {anyscale_iam_role.arn} does not have sufficient permissions for cluster management. We suggest adding these actions to ensure that cluster management works properly: {allow_actions_missing}. "
        )

    # Verify Service Steady State permissions
    # If service permissions are missing, display confirmation message to user if they would like to continue
    allow_actions_expected = filter_actions_from_policy_document(
        ANYSCALE_IAM_PERMISSIONS_SERVICE_STEADY_STATE
    )
    allow_actions_missing = allow_actions_expected - allow_actions_on_role
    if allow_actions_missing:
        logger.print_red_error_message(
            "[SERVICES V2] Permissions are missing to enable services v2 "
        )
        logger.confirm_missing_permission(
            f"For IAM role {anyscale_iam_role.arn}, we suggest adding the following actions:\n{pprint.pformat(allow_actions_missing)}.\n"
        )
        if strict:
            return False

    # verifying data plane role: ray autoscaler role
    cluster_node_role = _get_role_from_arn(data_plane_role, boto3_session, logger)
    if not cluster_node_role:
        return False
    assume_role_policy_document = cluster_node_role.assume_role_policy_document
    if not verify_data_plane_role_assume_role_policy(
        assume_role_policy_document=assume_role_policy_document,
    ):
        logger.error(
            f"Cluster Node IAM role {cluster_node_role.arn} does not contain expected assume role policy. It must give trust to AWS service EC2."
        )
        return False

    policy_names = [
        policy.policy_name for policy in cluster_node_role.attached_policies.all()
    ]
    if AMAZON_ECR_READONLY_ACCESS_POLICY_NAME not in policy_names:
        logger.warning(
            f"Dataplane role {cluster_node_role.arn} does not contain policy {AMAZON_ECR_READONLY_ACCESS_POLICY_NAME}. This is safe to ignore if you are not pulling custom Docker Images from an ECR repository."
        )
        if strict:
            return False

    if (
        len(
            [
                profile
                for profile in cluster_node_role.instance_profiles.all()
                if profile.name == cluster_node_role.name
            ]
        )
        == 0
    ):
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_IAM_ROLE,
            CloudSetupError.INSTANCE_PROFILE_NOT_FOUND,
        )
        logger.error(
            f"Dataplane role {cluster_node_role.arn} is required to have an instance profile with the name {cluster_node_role.name}."
            "\nPlease create this instance profile and associate it to the role."
        )
        return False

    logger.info(
        f"IAM roles {control_plane_role}, {data_plane_role} verification succeeded."
    )
    return True


def is_internal_communication_allowed(
    ip_permissions: List, aws_security_group_ids: List[str]
) -> bool:
    """
    This is a helper function to check if the security group allows internal communication.
    It can be used for both inbound and outbound permissions.
    """
    sg_rule_with_self = []  # type: ignore
    for sg_rule in ip_permissions:
        if sg_rule.get("IpProtocol") == "-1":
            sg_rule_with_self.extend(sg_rule.get("UserIdGroupPairs"))  # type: ignore
    return any(
        sg_rule.get("GroupId") in aws_security_group_ids
        for sg_rule in sg_rule_with_self
    )


def verify_aws_security_groups(  # noqa: PLR0912, PLR0911
    aws_security_group_ids: Optional[List[str]],
    boto3_session: boto3.Session,
    logger: CloudSetupLogger,
    strict: bool = False,
) -> bool:
    logger.info("Verifying security groups ...")
    if not aws_security_group_ids:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_SECURITY_GROUP,
            CloudSetupError.MISSING_CLOUD_RESOURCE_ID,
        )
        logger.error("Missing security group IDs.")
        return False

    ec2 = boto3_session.resource("ec2")

    anyscale_security_groups = []
    for anyscale_security_group_id in aws_security_group_ids:
        anyscale_security_group = ec2.SecurityGroup(anyscale_security_group_id)
        try:
            anyscale_security_group.load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "InvalidGroup.NotFound":
                log_resource_not_found_error(
                    "Security group", anyscale_security_group_id, logger
                )
                return False
            else:
                logger.log_resource_exception(
                    CloudAnalyticsEventCloudResource.AWS_SECURITY_GROUP, e
                )
            raise e
        anyscale_security_groups.append(anyscale_security_group)

    inbound_ip_permissions = [
        ip_permission
        for anyscale_security_group in anyscale_security_groups
        for ip_permission in anyscale_security_group.ip_permissions
    ]
    outbound_ip_permissions = [
        ip_permission
        for anyscale_security_group in anyscale_security_groups
        for ip_permission in anyscale_security_group.ip_permissions_egress
    ]
    inbound_ip_permissions_with_specific_port = {
        ip_permission["FromPort"]
        for ip_permission in inbound_ip_permissions
        if "FromPort" in ip_permission
    }

    # Check inbound permissions
    if not any(
        inbound_ip_permission_port == HTTPS_INGRESS_PORT
        for inbound_ip_permission_port in inbound_ip_permissions_with_specific_port
    ):
        logger.warning(
            f"Security groups {aws_security_group_ids} do not contain inbound permission for port {HTTPS_INGRESS_PORT}. This port is used for interaction with the clusters from Anyscale UI."
        )
        if strict:
            return False

    expected_open_ports = [
        HTTPS_INGRESS_PORT,
        22,
    ]  # 22 was previously used for SSH but is no longer required.
    if len(inbound_ip_permissions_with_specific_port) > len(expected_open_ports):
        logger.warning(
            f"Security groups {aws_security_group_ids} allows access to more than {expected_open_ports}. This may not be safe by default."
        )
        if strict:
            return False

    # Check internal communication is allowed
    if not is_internal_communication_allowed(
        inbound_ip_permissions, aws_security_group_ids
    ):
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_SECURITY_GROUP,
            CloudSetupError.INTERNAL_COMMUNICATION_NOT_ALLOWED,
        )
        logger.error(
            f"Security groups {aws_security_group_ids} do not contain inbound permission for all ports for traffic from the same security group."
        )
        return False

    # Check outbound permissions
    if not is_internal_communication_allowed(
        outbound_ip_permissions, aws_security_group_ids
    ):
        logger.warning(
            f"Security groups {aws_security_group_ids} do not contain outbound permission for all protocols for traffic from the same security group. "
            f"This is required for certain network device such as EFA."
        )
        if strict:
            return False

    logger.info(f"Security group {aws_security_group_ids} verification succeeded.")
    return True


def verify_aws_s3(  # noqa: PLR0911, PLR0912
    aws_s3_id: Optional[str],
    control_plane_role: Optional[str],
    data_plane_role: Optional[str],
    boto3_session: boto3.Session,
    region: str,
    logger: CloudSetupLogger,
    strict: bool = False,
) -> bool:
    logger.info("Verifying S3 ...")
    if not aws_s3_id:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_S3_BUCKET,
            CloudSetupError.MISSING_CLOUD_RESOURCE_ID,
        )
        logger.error("Missing S3 ID.")
        return False

    s3 = boto3_session.resource("s3")
    bucket_name = aws_s3_id.split(":")[-1]
    s3_bucket = s3.Bucket(bucket_name)

    # Check for the existence of `creation_date` because this incurs a `list_bucket` call.
    # Calling `.load()` WILL NOT ERROR in cases where the caller does not have access to the bucket.
    if s3_bucket.creation_date is None:
        log_resource_not_found_error("S3 bucket", aws_s3_id, logger)
        return False

    has_correct_cors_rule = False
    """
    Verify CORS rules. The correct CORS rule should look like:
    [{
        "AllowedHeaders": [
            "*"
        ],
        "AllowedMethods": [
            "GET",
            "PUT",
            "POST",
            "HEAD",
            "DELETE"
        ],
        "AllowedOrigins": [
            "https://console.anyscale-staging.com"
        ],
        "ExposeHeaders": [
            "Accept-Ranges",
            "Content-Range",
            "Content-Length"
        ]
    }]
    """

    try:
        cors_rules = s3_bucket.Cors().cors_rules
    except ClientError as e:
        logger.log_resource_exception(CloudAnalyticsEventCloudResource.AWS_S3_BUCKET, e)
        if e.response["Error"]["Code"] == "NoSuchCORSConfiguration":
            logger.warning(
                f"S3 bucket {bucket_name} does not have CORS rules. This is safe to ignore if you are not using Anyscale UI. Otherwise please create the correct CORS rule for Anyscale according to https://docs.anyscale.com/cloud-deployment/aws/manage-clouds#s3"
            )
            cors_rules = []
            if strict:
                return False
        else:
            raise e

    has_correct_cors_rule, cors_reason = check_aws_cors_rules(cors_rules)
    if "Malformed" in cors_reason:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_S3_BUCKET,
            CloudSetupError.MALFORMED_CORS_RULE,
        )
        logger.error(cors_reason)
        return False

    if not has_correct_cors_rule:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_S3_BUCKET,
            CloudSetupError.INCORRECT_CORS_RULE,
        )
        logger.warning(
            f"S3 bucket {bucket_name} does not have the correct CORS rule for Anyscale. This is safe to ignore if you are not using Anyscale UI. Otherwise please create the correct CORS rule for Anyscale according to https://docs.anyscale.com/cloud-deployment/aws/manage-clouds#appendix-detailed-resource-requirements"
        )
        if strict:
            return False

    returned_bucket_location = boto3_session.client("s3").get_bucket_location(
        Bucket=bucket_name
    )["LocationConstraint"]

    # LocationConstraint is `None` if the bucket is located in us-east-1
    bucket_region = returned_bucket_location or "us-east-1"
    if bucket_region != region:
        logger.warning(
            f"S3 bucket {bucket_name} is in region {bucket_region}, but this cloud is being set up in {region}."
            "This can result in degraded cluster launch & logging performance as well as additional cross-region costs."
        )
        if strict:
            return False

    if not control_plane_role or not data_plane_role:
        return False

    role = _get_role_from_arn(control_plane_role, boto3_session, logger)
    if not verify_s3_access(boto3_session, s3_bucket, role, logger):
        logger.warning(
            f"S3 Bucket {bucket_name} does not appear to have correct permissions for the Anyscale Control Plane role {role.name}"
        )
        if strict:
            return False

    role = _get_role_from_arn(data_plane_role, boto3_session, logger)
    if not verify_s3_access(boto3_session, s3_bucket, role, logger):
        logger.warning(
            f"S3 Bucket {bucket_name} does not appear to have correct permissions for the Data Plane role {role.name}"
        )
        if strict:
            return False
    logger.info(f"S3 {aws_s3_id} verification succeeded.")
    return True


def _get_network_interfaces_from_mount_targets(
    mount_targets_response: dict, boto3_session: Any, logger: CloudSetupLogger
) -> List[Any]:
    ec2 = boto3_session.resource("ec2")
    network_interfaces = []
    for network_interface_id in [
        mount_target["NetworkInterfaceId"]
        for mount_target in mount_targets_response["MountTargets"]
    ]:
        network_interface = ec2.NetworkInterface(network_interface_id)
        try:
            network_interface.load()
        except ClientError as e:
            logger.warning(f"Network interface loading error: {e}")
            continue
        network_interfaces.append(network_interface)
    return network_interfaces


def verify_aws_efs(  # noqa: PLR0911, PLR0912, C901
    aws_efs_id: Optional[str],
    aws_efs_mount_target_ips: Optional[List[str]],
    aws_subnet_ids: List[str],
    aws_security_groups: Optional[List[str]],
    boto3_session: boto3.Session,
    logger: CloudSetupLogger,
    strict: bool = False,
) -> bool:
    logger.info("Verifying EFS ...")
    if not aws_efs_id:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_EFS,
            CloudSetupError.MISSING_CLOUD_RESOURCE_ID,
        )
        logger.error("Missing EFS ID.")
        return False
    if not aws_subnet_ids:
        logger.error("Missing subnet IDs.")
        return False
    if not aws_security_groups:
        logger.error("Missing security group IDs.")
        return False

    client = boto3_session.client("efs")
    try:
        file_systems_response = client.describe_file_systems(FileSystemId=aws_efs_id)
    except ClientError as e:
        if e.response["Error"]["Code"] == "FileSystemNotFound":
            log_resource_not_found_error("EFS", aws_efs_id, logger)
            return False
        else:
            logger.log_resource_exception(CloudAnalyticsEventCloudResource.AWS_EFS, e)
        raise e

    if len(file_systems_response.get("FileSystems", [])) == 0:
        log_resource_not_found_error("EFS", aws_efs_id, logger)
        return False

    # verify that there is a mount target for each subnet and security group
    mount_targets_response = client.describe_mount_targets(FileSystemId=aws_efs_id)
    mount_targets = mount_targets_response.get("MountTargets")
    if not mount_targets:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_EFS,
            CloudSetupError.MOUNT_TARGET_NOT_FOUND,
        )
        logger.error(f"EFS with id {aws_efs_id} does not contain mount targets.")
        return False

    # verify the mount target ID stored in our database is still valid
    mount_target_ips = [mount_target["IpAddress"] for mount_target in mount_targets]
    if aws_efs_mount_target_ips and any(
        ip not in mount_target_ips for ip in aws_efs_mount_target_ips
    ):
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_EFS,
            CloudSetupError.INVALID_MOUNT_TARGET,
        )
        logger.error(
            f"Mount target registered with the cloud no longer exists. EFS ID: {aws_efs_id} IP address: {aws_efs_mount_target_ips}. Please make sure you have the correct AWS credentials set. If the EFS mount target has been deleted, please recreate the cloud or contact Anyscale for support."
        )
        logger.error(
            f"Valid mount target IPs for EFS ID {aws_efs_id} are {mount_target_ips}. "
            "If this happens during cloud edit, ensure that: "
            "1) If only editing aws_efs_mount_target_ip, it belongs to the existing EFS ID. "
            "2) If editing both efs_id and efs_mount_target_ip, the new IP is a valid target for the new efs_id."
        )
        return False

    network_interfaces = _get_network_interfaces_from_mount_targets(
        mount_targets_response, boto3_session, logger
    )

    expected_security_group_id = aws_security_groups[0]

    # Condition 1: No matching network interface in EFS mount targets for a subnet.
    # - 1.1: EFS has mount targets in other subnets.
    #        Subnet communicates with EFS through cross AZ, incurring cross AZ costs. (warning)
    # - 1.2: EFS doesn't have mount target, the subnet cannot communicate with EFS. (error)
    #   (note) A previous check ensures EFS has mount targets if this point is reached.
    # --------------------------------------------------------------------------------------------------------------
    # Condition 2: Subnet has a matching network interface in EFS mount targets.
    # - 2.1: Network interface has a registered security group. (happy path)
    # - 2.2: Network interface lacks a registered security group but has another security group.
    #   If configured correctly, subnet can still communicate with EFS. (warning)
    # --------------------------------------------------------------------------------------------------------------
    for subnet_id in aws_subnet_ids:
        contains_subnet_id = False
        contains_registered_security_group = False
        for network_interface in network_interfaces:
            network_interface_security_group_ids = [
                group["GroupId"]
                for group in network_interface.groups
                if group.get("GroupId")
            ]
            if network_interface.subnet_id == subnet_id:
                contains_subnet_id = True
                if expected_security_group_id in network_interface_security_group_ids:
                    contains_registered_security_group = True
                    break
        if not contains_subnet_id:
            # condition 1.1.
            logger.warning(
                f"EFS with id {aws_efs_id} does not contain a mount target with the subnet {subnet_id}, which might introduce cross AZ networking cost."
            )
            if strict:
                return False
        elif not contains_registered_security_group:
            # condition 2.2.
            logger.warning(
                f"EFS with id {aws_efs_id} does not contain a mount target with the subnet {subnet_id} and security group id {aws_security_groups[0]}. This misconfiguration might pose security risks and incur connection issues, preventing the EFS from working as expected."
            )
            if strict:
                return False
    try:
        backup_policy_response = client.describe_backup_policy(FileSystemId=aws_efs_id)
        backup_policy_status = backup_policy_response.get("BackupPolicy", {}).get(
            "Status", ""
        )
        if backup_policy_status != "ENABLED":
            logger.warning(f"EFS {aws_efs_id} backup policy is not enabled.")
            if strict:
                return False
    except ClientError as e:
        if e.response["Error"]["Code"] == "PolicyNotFound":
            logger.warning(f"EFS {aws_efs_id} backup policy not found.")
            if strict:
                return False
        else:
            raise e

    # Verify efs policy
    if not _verify_aws_efs_policy(boto3_session, aws_efs_id, logger, strict):
        return False

    logger.info(f"EFS {aws_efs_id} verification succeeded.")
    return True


def _verify_aws_efs_policy(
    boto3_session: boto3.Session,
    efs_id: str,
    logger: CloudSetupLogger,
    strict: bool = False,
) -> bool:
    """
    Verify that the EFS policy has sufficient permissions.
    """
    client = boto3_session.client("efs")

    efs_policy = None
    try:
        efs_policy_response = client.describe_file_system_policy(FileSystemId=efs_id)
        efs_policy = json.loads(efs_policy_response["Policy"])
    except ClientError as e:
        if e.response["Error"]["Code"] != "PolicyNotFound":
            # If the policy is not found, we'll skip the check
            logger.error(f"Failed to describe file system policy: {e}")
            return False

    if efs_policy:
        expected_actions = [
            "elasticfilesystem:ClientRootAccess",
            "elasticfilesystem:ClientWrite",
            "elasticfilesystem:ClientMount",
        ]
        actions = filter_actions_from_policy_document(efs_policy)
        missing_actions = set(expected_actions) - set(actions)
        if missing_actions:
            logger.warning(
                f"EFS {efs_id} does not have sufficient permissions. "
                f"We suggest adding these actions to ensure that efs works properly: {missing_actions}. "
            )
            if strict:
                return False

    return True


def verify_aws_cloudformation_stack(
    aws_cloudformation_stack_id: Optional[str],
    boto3_session: boto3.Session,
    logger: CloudSetupLogger,
    strict: bool = False,  # strict is currently unused # noqa: ARG001
) -> bool:
    logger.info("Verifying CloudFormation stack ...")
    if not aws_cloudformation_stack_id:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_CLOUDFORMATION,
            CloudSetupError.MISSING_CLOUD_RESOURCE_ID,
        )
        logger.error("Missing CloudFormation stack id.")
        return False

    cloudformation = boto3_session.resource("cloudformation")
    stack = cloudformation.Stack(aws_cloudformation_stack_id)
    try:
        stack.load()
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationError":
            log_resource_not_found_error(
                "CloudFormation stack", aws_cloudformation_stack_id, logger,
            )
            return False
        else:
            logger.log_resource_exception(
                CloudAnalyticsEventCloudResource.AWS_CLOUDFORMATION, e
            )
        raise e

    logger.info(
        f"CloudFormation stack {aws_cloudformation_stack_id} verification succeeded."
    )
    return True


def verify_aws_memorydb_cluster(  # noqa: PLR0911, PLR0912
    memorydb_cluster_config: Optional[AWSMemoryDBClusterConfig],
    aws_security_groups: List[str],
    aws_vpc_id: str,
    aws_subnet_ids: List[str],
    boto3_session: boto3.Session,
    logger: CloudSetupLogger,
    strict: bool = False,  # strict is currently unused # noqa: ARG001
) -> bool:
    """Verify that the MemoryDB cluster exists and is in the available state."""
    logger.info("Verifying MemoryDB ...")
    if not memorydb_cluster_config:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.AWS_MEMORYDB,
            CloudSetupError.MISSING_CLOUD_RESOURCE_ID,
        )
        logger.error("Missing MemoryDB cluster id.")
        return False

    client = boto3_session.client("memorydb")
    try:
        response = client.describe_clusters(
            ClusterName=memorydb_cluster_config.id.split("/")[-1],
            ShowShardDetails=True,
        )
        if not response.get("Clusters"):
            log_resource_not_found_error(
                "MemoryDB cluster", memorydb_cluster_config.id, logger
            )
            return False

        # verify that the subnet group has the same security group as the cloud
        security_group_id = response["Clusters"][0].get(
            "SecurityGroups", [{"SecurityGroupId": "NOT_SPECIFIED"}]
        )[0]["SecurityGroupId"]
        if security_group_id != aws_security_groups[0]:
            logger.warning(
                f"MemoryDB cluster {memorydb_cluster_config.id} has security group {security_group_id} that is not the same as the cloud's security group {aws_security_groups[0]}."
            )
            if strict:
                return False

        # verify that the cluster is in the cloud's VPC
        subnet_group_response = client.describe_subnet_groups(
            SubnetGroupName=response["Clusters"][0]["SubnetGroupName"]
        )
        if subnet_group_response["SubnetGroups"][0]["VpcId"] != aws_vpc_id:
            logger.warning(
                f"MemoryDB cluster {memorydb_cluster_config.id} is not in the same VPC as the cloud."
            )
            if strict:
                return False

        # verify that the subnet group has the subset of subnets that the cloud has
        for subnet in subnet_group_response["SubnetGroups"][0]["Subnets"]:
            if subnet["Identifier"] not in aws_subnet_ids:
                logger.warning(
                    f"MemoryDB cluster {memorydb_cluster_config.id} has subnet {subnet['Identifier']} that is not one of the subnets in the cloud."
                )
                if strict:
                    return False

        # verify that the cluster has parameter group with the maxmemory-policy set to allkeys-lru
        parameter_response = client.describe_parameters(
            ParameterGroupName=response["Clusters"][0]["ParameterGroupName"]
        )
        for param in parameter_response["Parameters"]:
            if param["Name"] == "maxmemory-policy" and param["Value"] != "allkeys-lru":
                logger.warning(
                    f"MemoryDB cluster {memorydb_cluster_config.id} should have parameter group with maxmemory-policy set to allkeys-lru instead of {param['Value']}."
                )
                if strict:
                    return False

        # verify TLS is enabled
        if not response["Clusters"][0]["TLSEnabled"]:
            logger.error(
                f"MemoryDB cluster {memorydb_cluster_config.id} has TLS disabled. Please create a memorydb cluster with TLS enabled."
            )
            return False

        # verify that each shard in the cluster has at least 2 nodes for high availability
        for shard in response["Clusters"][0]["Shards"]:
            if len(shard["Nodes"]) < 2:
                logger.error(
                    f"MemoryDB cluster {memorydb_cluster_config.id} has shard {shard['Name']} with less than 2 nodes. This is not enough for high availability. Please make sure each shard has at least 2 nodes."
                )
                return False

    except ClientError as e:
        logger.log_resource_exception(CloudAnalyticsEventCloudResource.AWS_MEMORYDB, e)
        raise ClickException(
            f"Failed to verify MemoryDB cluster {memorydb_cluster_config.id}.\nError: {e}"
        )

    return True
