"""
Fetches data required and formats output for `anyscale cloud` commands.
"""

import copy
from datetime import datetime, timedelta
import difflib
import json
from os import getenv
import pathlib
import re
import secrets
import time
from typing import Any, Dict, List, MutableSequence, Optional, Tuple
import uuid

import boto3
from boto3.resources.base import ServiceResource as Boto3Resource
from botocore.exceptions import ClientError, NoCredentialsError
import click
from click import Abort, ClickException
import colorama
from rich.progress import Progress, track
import yaml

from anyscale import __version__ as anyscale_version
from anyscale.aws_iam_policies import get_anyscale_iam_permissions_ec2_restricted
from anyscale.cli_logger import CloudSetupLogger
from anyscale.client.openapi_client.models import (
    AWSConfig,
    AWSMemoryDBClusterConfig,
    Cloud,
    CloudAnalyticsEventCloudResource,
    CloudAnalyticsEventCommandName,
    CloudAnalyticsEventName,
    CloudDeployment,
    CloudDeploymentConfig,
    CloudProviders,
    CloudState,
    ClusterManagementStackVersions,
    ComputeStack,
    CreateCloudResource,
    CreateCloudResourceGCP,
    DecoratedCloudResource,
    EditableCloudResource,
    EditableCloudResourceGCP,
    FileStorage,
    GCPConfig,
    GCPFileStoreConfig,
    NetworkingMode,
    NFSMountTarget,
    ObjectStorage,
    WriteCloud,
)
from anyscale.cloud_resource import (
    associate_aws_subnets_with_azs,
    GCS_STORAGE_PREFIX,
    S3_ARN_PREFIX,
    S3_STORAGE_PREFIX,
    verify_aws_cloudformation_stack,
    verify_aws_efs,
    verify_aws_iam_roles,
    verify_aws_memorydb_cluster,
    verify_aws_s3,
    verify_aws_security_groups,
    verify_aws_subnets,
    verify_aws_vpc,
)
from anyscale.cloud_utils import (
    get_cloud_id_and_name,
    get_cloud_resource_by_cloud_id,
    get_organization_id,
)
from anyscale.conf import ANYSCALE_IAM_ROLE_NAME
from anyscale.controllers.base_controller import BaseController
from anyscale.controllers.cloud_functional_verification_controller import (
    CloudFunctionalVerificationController,
    CloudFunctionalVerificationType,
)
from anyscale.controllers.kubernetes_verifier import KubernetesCloudDeploymentVerifier
from anyscale.job._private.job_sdk import (
    HA_JOB_STATE_TO_JOB_STATE,
    TERMINAL_HA_JOB_STATES,
)
from anyscale.shared_anyscale_utils.aws import AwsRoleArn
from anyscale.shared_anyscale_utils.conf import ANYSCALE_ENV, ANYSCALE_HOST
from anyscale.util import (  # pylint:disable=private-import
    _client,
    _get_aws_efs_mount_target_ip,
    _get_memorydb_cluster_config,
    _get_role,
    _update_external_ids_for_policy,
    confirm,
    GCP_DEPLOYMENT_MANAGER_TIMEOUT_SECONDS_LONG,
    get_anyscale_cross_account_iam_policies,
    get_available_regions,
    get_user_env_aws_account,
    prepare_cloudformation_template,
    REDIS_TLS_ADDRESS_PREFIX,
    SharedStorageType,
)
from anyscale.utils.cloud_update_utils import (
    CLOUDFORMATION_TIMEOUT_SECONDS_LONG,
    get_or_create_memorydb,
    MEMORYDB_REDIS_PORT,
    try_delete_customer_drifts_policy,
    update_iam_role,
)
from anyscale.utils.cloud_utils import (
    _unroll_resources_for_aws_list_call,
    CloudEventProducer,
    CloudSetupError,
    get_errored_resources_and_reasons,
    modify_memorydb_parameter_group,
    validate_aws_credentials,
    verify_anyscale_access,
    wait_for_lb_resource_termination,
)
from anyscale.utils.cors_update_utils import (
    check_aws_cors_needs_update,
    check_azure_cors_needs_update,
    check_gcp_cors_needs_update,
    update_aws_cors,
    update_azure_cors,
    update_gcp_cors,
)
from anyscale.utils.imports.gcp import (
    try_import_gcp_managed_setup_utils,
    try_import_gcp_utils,
    try_import_gcp_verify_lib,
)


ROLE_CREATION_RETRIES = 30
ROLE_CREATION_INTERVAL_SECONDS = 1

# AWS Partner Network (APN) identification for partner attribution
TAG_AWS_APN_ID = "aws-apn-id"
TAG_VALUE_AWS_APN_ID = "pc:8q2qyyettux52j80lzixegorb"

try:
    CLOUDFORMATION_TIMEOUT_SECONDS = int(
        getenv("CLOUDFORMATION_TIMEOUT_SECONDS", "300")
    )
except ValueError:
    raise Exception(
        f"CLOUDFORMATION_TIMEOUT_SECONDS is set to {getenv('CLOUDFORMATION_TIMEOUT_SECONDS')}, which is not a valid integer."
    )


try:
    GCP_DEPLOYMENT_MANAGER_TIMEOUT_SECONDS = int(
        getenv("GCP_DEPLOYMENT_MANAGER_TIMEOUT_SECONDS", "600")
    )
except ValueError:
    raise Exception(
        f"GCP_DEPLOYMENT_MANAGER_TIMEOUT_SECONDS is set to {getenv('GCP_DEPLOYMENT_MANAGER_TIMEOUT_SECONDS')}, which is not a valid integer."
    )

IGNORE_CAPACITY_ERRORS = getenv("IGNORE_CAPACITY_ERRORS") is not None

# Constants forked from ray.autoscaler._private.aws.config
RAY = "ray-autoscaler"
DEFAULT_RAY_IAM_ROLE = RAY + "-v1"

# Only used in cloud edit.
BASE_ROLLBACK_COMMAND = "anyscale cloud edit --cloud-id={cloud_id}"


class CloudController(BaseController):
    def __init__(
        self,
        log: Optional[CloudSetupLogger] = None,
        initialize_auth_api_client: bool = True,
        cli_token: Optional[str] = None,
    ):
        """
        :param log: The logger to use for this controller. If not provided, a new logger will be created.
        :param initialize_auth_api_client: Whether to initialize the auth api client.
        :param cli_token: The CLI token to use for this controller. If provided, the CLI token will be used instead
        of the one in the config file or the one in the environment variable. This is for setting up AIOA clouds only.
        """
        if log is None:
            log = CloudSetupLogger()

        super().__init__(
            initialize_auth_api_client=initialize_auth_api_client, cli_token=cli_token,
        )

        self.log = log
        self.log.open_block("Output")
        if self.initialize_auth_api_client:
            self.cloud_event_producer = CloudEventProducer(
                cli_version=anyscale_version, api_client=self.api_client
            )

    def create_empty_cloud(self, name: str,) -> str:
        """
        Create an empty cloud shell in PENDING_RESOURCES state.

        The cloud can be configured with credentials later via the setup flow.
        Provider and region will be determined when resources are added.

        This is useful for BYOC onboarding where users want to create the cloud
        record first and then provide credentials/resources later.

        Returns the cloud ID.
        """
        # Make a direct API call to the /clouds/empty endpoint
        body_params = {
            "name": name,
        }

        response = self.api_client.api_client.call_api(
            "/api/v2/clouds/empty",
            "POST",
            path_params={},
            query_params=[],
            header_params={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body=body_params,
            post_params=[],
            files={},
            response_type="CloudResponse",
            auth_settings=[],
            _return_http_data_only=True,
            _preload_content=True,
        )

        self.log.info(f"Created empty cloud '{name}' with ID: {response.result.id}")
        self.log.info("Add cloud resources to complete setup.")
        return response.result.id

    def _get_anyscale_cross_account_iam_policies(
        self, cloud_id: str, _use_strict_iam_permissions: bool,
    ) -> List[Dict[str, str]]:
        iam_policy_parameters = get_anyscale_cross_account_iam_policies()
        if _use_strict_iam_permissions:
            anyscale_iam_permissions_ec2 = get_anyscale_iam_permissions_ec2_restricted(
                cloud_id
            )
            for parameter in iam_policy_parameters:
                if (
                    parameter["ParameterKey"]
                    == "AnyscaleCrossAccountIAMPolicySteadyState"
                ):
                    parameter["ParameterValue"] = json.dumps(
                        anyscale_iam_permissions_ec2
                    )
                    break
        return iam_policy_parameters

    def create_cloudformation_stack(  # noqa: PLR0913
        self,
        region: str,
        cloud_id: str,
        anyscale_iam_role_name: str,
        cluster_node_iam_role_name: str,
        enable_head_node_fault_tolerance: bool,
        anyscale_aws_account: str,
        _use_strict_iam_permissions: bool = False,  # This should only be used in testing.
        boto3_session: Optional[boto3.Session] = None,
        is_anyscale_hosted: bool = False,
        anyscale_hosted_network_info: Optional[Dict[str, Any]] = None,
        shared_storage: SharedStorageType = SharedStorageType.OBJECT_STORAGE,
    ):
        if boto3_session is None:
            boto3_session = boto3.Session(region_name=region)

        cfn_client = boto3_session.client("cloudformation", region_name=region)
        cfn_stack_name = cloud_id.replace("_", "-").lower()

        cfn_template_body = prepare_cloudformation_template(
            region,
            cfn_stack_name,
            cloud_id,
            enable_head_node_fault_tolerance,
            boto3_session,
            is_anyscale_hosted=is_anyscale_hosted,
            shared_storage=shared_storage,
        )

        parameters = [
            {"ParameterKey": "EnvironmentName", "ParameterValue": ANYSCALE_ENV},
            {"ParameterKey": "AnyscaleCLIVersion", "ParameterValue": anyscale_version},
            {"ParameterKey": "CloudID", "ParameterValue": cloud_id},
            {
                "ParameterKey": "AnyscaleAWSAccountID",
                "ParameterValue": anyscale_aws_account,
            },
            {
                "ParameterKey": "ClusterNodeIAMRoleName",
                "ParameterValue": cluster_node_iam_role_name,
            },
            {
                "ParameterKey": "MemoryDBRedisPort",
                "ParameterValue": MEMORYDB_REDIS_PORT,
            },
        ]
        if not is_anyscale_hosted:
            parameters.append(
                {
                    "ParameterKey": "EnableEFS",
                    "ParameterValue": "true"
                    if shared_storage == SharedStorageType.NFS
                    else "false",
                }
            )
            parameters.append(
                {
                    "ParameterKey": "AnyscaleCrossAccountIAMRoleName",
                    "ParameterValue": anyscale_iam_role_name,
                }
            )
            cross_account_iam_policies = self._get_anyscale_cross_account_iam_policies(
                cloud_id, _use_strict_iam_permissions
            )
            for parameter in cross_account_iam_policies:
                parameters.append(parameter)

        tags: MutableSequence[Any] = []
        # Add AWS APN ID tag for partner attribution
        tags.append({"Key": TAG_AWS_APN_ID, "Value": TAG_VALUE_AWS_APN_ID},)
        if is_anyscale_hosted:
            # Add is-anyscale-hosted tag to the cloudformation stack which
            # the reconcile process will use to determine if the use the
            # shared resources.
            tags.append({"Key": "anyscale:cloud-id", "Value": cloud_id},)
            tags.append({"Key": "anyscale:is-anyscale-hosted", "Value": "true"},)

            # VPC ID are needed for creating security groups and subnets
            # are needed for mount targets in shared VPC clouds.
            if anyscale_hosted_network_info is not None:
                parameters.append(
                    {
                        "ParameterKey": "VpcId",
                        "ParameterValue": anyscale_hosted_network_info["vpc_id"],
                    },
                )

        cfn_client.create_stack(
            StackName=cfn_stack_name,
            TemplateBody=cfn_template_body,
            Parameters=parameters,  # type: ignore
            Capabilities=["CAPABILITY_NAMED_IAM", "CAPABILITY_AUTO_EXPAND"],
            Tags=tags,
        )

    def run_cloudformation(  # noqa: PLR0913
        self,
        region: str,
        cloud_id: str,
        anyscale_iam_role_name: str,
        cluster_node_iam_role_name: str,
        enable_head_node_fault_tolerance: bool,
        anyscale_aws_account: str,
        _use_strict_iam_permissions: bool = False,  # This should only be used in testing.
        boto3_session: Optional[boto3.Session] = None,
        shared_storage: SharedStorageType = SharedStorageType.OBJECT_STORAGE,
        resource_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run cloudformation to create the AWS resources for a cloud.

        When enable_head_node_fault_tolerance is set to True, a memorydb cluster will be created.

        Args:
            resource_id: Optional unique identifier for the resource. If provided, used as the
                CFN stack name prefix. If not provided, defaults to cloud_id (backward compatible).
        """
        if boto3_session is None:
            boto3_session = boto3.Session(region_name=region)

        cfn_client = boto3_session.client("cloudformation", region_name=region)
        # Use resource_id if provided, otherwise fall back to cloud_id for backward compatibility
        cfn_stack_name = (resource_id or cloud_id).replace("_", "-").lower()

        cfn_template_body = prepare_cloudformation_template(
            region,
            cfn_stack_name,
            cloud_id,
            enable_head_node_fault_tolerance,
            boto3_session,
            shared_storage=shared_storage,
        )

        cross_account_iam_policies = self._get_anyscale_cross_account_iam_policies(
            cloud_id, _use_strict_iam_permissions
        )

        parameters = [
            {"ParameterKey": "EnvironmentName", "ParameterValue": ANYSCALE_ENV},
            {"ParameterKey": "AnyscaleCLIVersion", "ParameterValue": anyscale_version},
            {"ParameterKey": "CloudID", "ParameterValue": cloud_id},
            {
                "ParameterKey": "AnyscaleAWSAccountID",
                "ParameterValue": anyscale_aws_account,
            },
            {
                "ParameterKey": "AnyscaleCrossAccountIAMRoleName",
                "ParameterValue": anyscale_iam_role_name,
            },
            {
                "ParameterKey": "ClusterNodeIAMRoleName",
                "ParameterValue": cluster_node_iam_role_name,
            },
            {
                "ParameterKey": "MemoryDBRedisPort",
                "ParameterValue": MEMORYDB_REDIS_PORT,
            },
            {
                "ParameterKey": "EnableEFS",
                "ParameterValue": "true"
                if shared_storage == SharedStorageType.NFS
                else "false",
            },
        ]
        for parameter in cross_account_iam_policies:
            parameters.append(parameter)

        self.log.debug("cloudformation body:")
        self.log.debug(cfn_template_body)

        cfn_client.create_stack(
            StackName=cfn_stack_name,
            TemplateBody=cfn_template_body,
            Parameters=parameters,  # type: ignore
            Capabilities=["CAPABILITY_NAMED_IAM", "CAPABILITY_AUTO_EXPAND"],
        )

        stacks = cfn_client.describe_stacks(StackName=cfn_stack_name)
        cfn_stack = stacks["Stacks"][0]
        cfn_stack_url = f"https://{region}.console.aws.amazon.com/cloudformation/home?region={region}#/stacks/stackinfo?stackId={cfn_stack['StackId']}"
        self.log.info(f"\nTrack progress of cloudformation at {cfn_stack_url}")
        with self.log.spinner("Creating cloud resources through cloudformation..."):
            start_time = time.time()
            end_time = (
                start_time + CLOUDFORMATION_TIMEOUT_SECONDS_LONG
                if enable_head_node_fault_tolerance
                else start_time + CLOUDFORMATION_TIMEOUT_SECONDS
            )
            while time.time() < end_time:
                stacks = cfn_client.describe_stacks(StackName=cfn_stack_name)
                cfn_stack = stacks["Stacks"][0]
                if cfn_stack["StackStatus"] in (
                    "CREATE_FAILED",
                    "ROLLBACK_COMPLETE",
                    "ROLLBACK_IN_PROGRESS",
                ):
                    bucket_name = f"anyscale-{ANYSCALE_ENV}-data-{cfn_stack_name}"
                    try:
                        boto3_session.client("s3", region_name=region).delete_bucket(
                            Bucket=bucket_name
                        )
                        self.log.info(f"Successfully deleted {bucket_name}")
                    except Exception as e:  # noqa: BLE001
                        if not (
                            isinstance(e, ClientError)
                            and e.response["Error"]["Code"] == "NoSuchBucket"
                        ):
                            self.log.error(
                                f"Unable to delete the S3 bucket created by the cloud formation stack, please manually delete {bucket_name}"
                            )

                    # Describe events to get error reason
                    error_details = get_errored_resources_and_reasons(
                        cfn_client, cfn_stack_name
                    )
                    self.log.log_resource_error(
                        CloudAnalyticsEventCloudResource.AWS_CLOUDFORMATION,
                        f"Cloudformation stack failed to deploy. Detailed errors: {error_details}",
                    )
                    # Provide link to cloudformation
                    raise ClickException(
                        f"Failed to set up cloud resources. Please check your cloudformation stack for errors and to ensure that all resources created in this cloudformation stack were deleted: {cfn_stack_url}"
                    )
                if cfn_stack["StackStatus"] == "CREATE_COMPLETE":
                    if enable_head_node_fault_tolerance:
                        modify_memorydb_parameter_group(
                            cfn_stack_name, region, boto3_session
                        )
                    self.log.info(
                        f"Cloudformation stack {cfn_stack['StackId']} Completed"
                    )
                    break

                time.sleep(1)

            if time.time() > end_time:
                self.log.log_resource_error(
                    CloudAnalyticsEventCloudResource.AWS_CLOUDFORMATION,
                    "Cloudformation timed out.",
                )
                raise ClickException(
                    f"Timed out creating AWS resources. Please check your cloudformation stack for errors. {cfn_stack['StackId']}"
                )
        return cfn_stack  # type: ignore

    def run_deployment_manager(  # noqa: PLR0913
        self,
        factory: Any,
        deployment_name: str,
        cloud_id: str,
        project_id: str,
        region: str,
        anyscale_access_service_account: str,
        workload_identity_pool_name: str,
        anyscale_aws_account: str,
        organization_id: str,
        enable_head_node_fault_tolerance: bool,
        shared_storage: SharedStorageType = SharedStorageType.OBJECT_STORAGE,
    ):
        setup_utils = try_import_gcp_managed_setup_utils()

        anyscale_access_service_account_name = anyscale_access_service_account.split(
            "@"
        )[0]
        deployment_config = setup_utils.generate_deployment_manager_config(
            region,
            project_id,
            cloud_id,
            anyscale_access_service_account_name,
            workload_identity_pool_name,
            anyscale_aws_account,
            organization_id,
            enable_head_node_fault_tolerance,
            shared_storage=shared_storage,
        )

        self.log.debug("GCP Deployment Manager resource config:")
        self.log.debug(deployment_config)

        deployment = {
            "name": deployment_name,
            "target": {"config": {"content": deployment_config,},},
            "labels": [{"key": "anyscale-cloud-id", "value": cloud_id},],
        }

        deployment_client = factory.build("deploymentmanager", "v2")
        response = (
            deployment_client.deployments()
            .insert(project=project_id, body=deployment)
            .execute()
        )
        deployment_url = f"https://console.cloud.google.com/dm/deployments/details/{deployment_name}?project={project_id}"

        timeout_seconds = (
            GCP_DEPLOYMENT_MANAGER_TIMEOUT_SECONDS_LONG
            if enable_head_node_fault_tolerance
            else GCP_DEPLOYMENT_MANAGER_TIMEOUT_SECONDS
        )
        self.log.info(
            f"Note that it may take up to {int(timeout_seconds / 60)} minutes to create resources on GCP."
        )
        self.log.info(f"Track progress of Deployment Manager at {deployment_url}")
        with self.log.spinner("Creating cloud resources through Deployment Manager..."):
            start_time = time.time()
            end_time = start_time + timeout_seconds
            while time.time() < end_time:
                current_operation = (
                    deployment_client.operations()
                    .get(operation=response["name"], project=project_id)
                    .execute()
                )
                if current_operation.get("status", None) == "DONE":
                    if "error" in current_operation:
                        self.log.error(f"Error: {current_operation['error']}")
                        self.log.log_resource_error(
                            CloudAnalyticsEventCloudResource.GCP_DEPLOYMENT,
                            current_operation["error"],
                        )
                        raise ClickException(
                            f"Failed to set up cloud resources. Please check your Deployment Manager for errors and delete all resources created in this deployment: {deployment_url}"
                        )
                    # happy path
                    self.log.info("Deployment succeeded.")
                    return True
                time.sleep(1)
            # timeout
            self.log.log_resource_error(
                CloudAnalyticsEventCloudResource.GCP_DEPLOYMENT, "Deployment timed out."
            )
            raise ClickException(
                f"Timed out creating GCP resources. Please check your deployment for errors and delete all resources created in this deployment: {deployment_url}"
            )

    def get_cloud_resource_from_cfn_stack(
        self,
        cfn_stack: Dict[str, Any],
        cloud_id: str,
        region: str,
        enable_head_node_fault_tolerance: bool,
        anyscale_hosted_network_info: Optional[Dict[str, Any]] = None,
        anyscale_control_plane_role_arn: str = "",
    ) -> CloudDeployment:
        if "Outputs" not in cfn_stack:
            raise ClickException(
                f"Timed out setting up cloud resources. Please check your cloudformation stack for errors. {cfn_stack['StackId']}"
            )

        cfn_resources = {}
        for resource in cfn_stack["Outputs"]:
            resource_type = resource["OutputKey"]
            resource_value = resource["OutputValue"]
            assert (
                resource_value is not None
            ), f"{resource_type} is not created properly. Please delete the cloud and try creating agian."
            cfn_resources[resource_type] = resource_value

        aws_subnets_with_availability_zones = (
            json.loads(cfn_resources["SubnetsWithAvailabilityZones"])
            if not anyscale_hosted_network_info
            else anyscale_hosted_network_info["subnet_ids"]
        )
        aws_vpc_id = (
            cfn_resources["VPC"]
            if not anyscale_hosted_network_info
            else anyscale_hosted_network_info["vpc_id"]
        )
        aws_security_groups = [cfn_resources["AnyscaleSecurityGroup"]]

        aws_efs_id = cfn_resources.get("EFS")
        aws_efs_mount_target_ip = cfn_resources.get("EFSMountTargetIP")
        anyscale_iam_role_arn = cfn_resources.get(
            "AnyscaleIAMRole", anyscale_control_plane_role_arn
        )
        cluster_node_iam_role_arn = cfn_resources["NodeIamRole"]

        if enable_head_node_fault_tolerance:
            memorydb = json.loads(cfn_resources["MemoryDB"])
            memorydb_cluster_config = AWSMemoryDBClusterConfig(
                id=memorydb["arn"],
                endpoint=f'{REDIS_TLS_ADDRESS_PREFIX}{memorydb["ClusterEndpointAddress"]}:{MEMORYDB_REDIS_PORT}',
            )
        else:
            memorydb_cluster_config = None

        bucket_name = cfn_resources.get("S3Bucket", "")
        if bucket_name.startswith(S3_ARN_PREFIX):
            bucket_name = bucket_name[len(S3_ARN_PREFIX) :]

        return CloudDeployment(
            compute_stack=ComputeStack.VM,
            provider=CloudProviders.AWS,
            region=region,
            networking_mode=NetworkingMode.PUBLIC,
            object_storage=ObjectStorage(bucket_name=S3_STORAGE_PREFIX + bucket_name),
            file_storage=FileStorage(
                file_storage_id=aws_efs_id,
                mount_targets=[NFSMountTarget(address=aws_efs_mount_target_ip)],
            )
            if aws_efs_id
            else None,
            aws_config=AWSConfig(
                vpc_id=aws_vpc_id,
                subnet_ids=[
                    subnet_with_az["subnet_id"]
                    for subnet_with_az in aws_subnets_with_availability_zones
                ],
                zones=[
                    subnet_with_az["availability_zone"]
                    for subnet_with_az in aws_subnets_with_availability_zones
                ],
                security_group_ids=aws_security_groups,
                anyscale_iam_role_id=anyscale_iam_role_arn,
                external_id=cloud_id,
                cluster_iam_role_id=cluster_node_iam_role_arn,
                memorydb_cluster_arn=memorydb_cluster_config.id
                if memorydb_cluster_config
                else None,
                memorydb_cluster_endpoint=memorydb_cluster_config.endpoint
                if memorydb_cluster_config
                else None,
                cloudformation_id=cfn_stack["StackId"],
            ),
        )

    def update_cloud_with_resources(
        self,
        cfn_stack: Dict[str, Any],
        cloud_id: str,
        region: str,
        enable_head_node_fault_tolerance: bool,
    ):
        cloud_resource = self.get_cloud_resource_from_cfn_stack(
            cfn_stack, cloud_id, region, enable_head_node_fault_tolerance
        )
        with self.log.spinner("Updating Anyscale cloud with cloud resources..."):
            self.api_client.add_cloud_resource_api_v2_clouds_cloud_id_add_resource_put(
                cloud_id=cloud_id, cloud_deployment=cloud_resource,
            )

    def update_cloud_with_resources_gcp(  # noqa: PLR0913
        self,
        factory: Any,
        deployment_name: str,
        cloud_id: str,
        region: str,
        project_id: str,
        anyscale_access_service_account: str,
        provider_name: str,
    ):
        setup_utils = try_import_gcp_managed_setup_utils()
        gcp_utils = try_import_gcp_utils()
        anyscale_access_service_account_name = anyscale_access_service_account.split(
            "@"
        )[0]

        cloud_resources = setup_utils.get_deployment_resources(
            factory, deployment_name, project_id, anyscale_access_service_account_name
        )
        gcp_vpc_id = cloud_resources["compute.v1.network"]
        gcp_subnet_ids = [cloud_resources["compute.v1.subnetwork"]]
        gcp_cluster_node_service_account_email = f'{cloud_resources["iam.v1.serviceAccount"]}@{anyscale_access_service_account.split("@")[-1]}'
        gcp_anyscale_iam_service_account_email = anyscale_access_service_account
        gcp_firewall_policy = cloud_resources[
            "gcp-types/compute-v1:networkFirewallPolicies"
        ]
        gcp_firewall_policy_ids = [gcp_firewall_policy]
        gcp_cloud_storage_bucket_id = cloud_resources["storage.v1.bucket"]
        gcp_deployment_manager_id = deployment_name

        gcp_filestore_config = None
        if cloud_resources.get("filestore_location") and cloud_resources.get(
            "filestore_instance"
        ):
            gcp_filestore_config = gcp_utils.get_gcp_filestore_config(
                factory,
                project_id,
                gcp_vpc_id,
                cloud_resources["filestore_location"],
                cloud_resources["filestore_instance"],
                self.log,
            )
        memorystore_instance_config = gcp_utils.get_gcp_memorystore_config(
            factory, cloud_resources.get("memorystore_name"),
        )
        try:
            setup_utils.configure_firewall_policy(
                factory, gcp_vpc_id, project_id, gcp_firewall_policy
            )

            cloud_resource = CloudDeployment(
                compute_stack=ComputeStack.VM,
                provider=CloudProviders.GCP,
                region=region,
                networking_mode=NetworkingMode.PUBLIC,
                object_storage=ObjectStorage(
                    bucket_name=GCS_STORAGE_PREFIX + gcp_cloud_storage_bucket_id
                ),
                file_storage=FileStorage(
                    file_storage_id=gcp_filestore_config.instance_name,
                    mount_targets=[
                        NFSMountTarget(address=gcp_filestore_config.mount_target_ip)
                    ],
                    mount_path=gcp_filestore_config.root_dir,
                )
                if gcp_filestore_config
                else None,
                gcp_config=GCPConfig(
                    project_id=project_id,
                    provider_name=provider_name,
                    vpc_name=gcp_vpc_id,
                    subnet_names=gcp_subnet_ids,
                    firewall_policy_names=gcp_firewall_policy_ids,
                    anyscale_service_account_email=gcp_anyscale_iam_service_account_email,
                    cluster_service_account_email=gcp_cluster_node_service_account_email,
                    memorystore_instance_name=memorystore_instance_config.name
                    if memorystore_instance_config
                    else None,
                    memorystore_endpoint=memorystore_instance_config.endpoint
                    if memorystore_instance_config
                    else None,
                    deployment_manager_id=gcp_deployment_manager_id,
                ),
            )

            self.api_client.add_cloud_resource_api_v2_clouds_cloud_id_add_resource_put(
                cloud_id=cloud_id, cloud_deployment=cloud_resource,
            )
        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            for firewall_policy_name in gcp_firewall_policy_ids:
                setup_utils.remove_firewall_policy_associations(
                    factory, project_id, firewall_policy_name
                )
            raise ClickException(
                f"Error occurred when updating resources for the cloud {cloud_id}."
            )

    def prepare_for_managed_cloud_setup(
        self,
        region: str,
        cloud_name: str,
        cluster_management_stack_version: ClusterManagementStackVersions,
        auto_add_user: bool,
        is_aioa: bool = False,
        boto3_session: Optional[boto3.Session] = None,
    ) -> Tuple[str, str]:
        regions_available = get_available_regions(boto3_session)
        if region not in regions_available:
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.PREPROCESS_COMPLETE,
                succeeded=False,
                internal_error=f"Region '{region}' is not available.",
            )
            raise ClickException(
                f"Region '{region}' is not available. Regions available are: "
                f"{', '.join(map(repr, regions_available))}"
            )

        for _ in range(5):
            anyscale_iam_role_name = "{}-{}".format(
                ANYSCALE_IAM_ROLE_NAME, secrets.token_hex(4)
            )
            try:
                role = _get_role(anyscale_iam_role_name, region, boto3_session)
            except Exception as e:  # noqa: BLE001
                if isinstance(e, NoCredentialsError):
                    # Rewrite the error message to be more user friendly
                    self.cloud_event_producer.produce(
                        CloudAnalyticsEventName.PREPROCESS_COMPLETE,
                        succeeded=False,
                        internal_error="Unable to locate AWS credentials.",
                    )
                    raise ClickException(
                        "Unable to locate AWS credentials. Please make sure you have AWS credentials configured."
                    )
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.PREPROCESS_COMPLETE,
                    succeeded=False,
                    internal_error=str(e),
                )
                raise ClickException(f"Failed to get IAM role: {e}")
            if role is None:
                break
        else:
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.PREPROCESS_COMPLETE,
                succeeded=False,
                internal_error="Unable to find an available AWS IAM Role name",
            )
            raise ClickException(
                "We weren't able to connect your account with the Anyscale because we weren't able to find an available IAM Role name in your account. Please reach out to support or your SA for assistance."
            )

        user_aws_account_id = get_user_env_aws_account(region)
        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.PREPROCESS_COMPLETE, succeeded=True,
        )
        try:
            created_cloud = self.api_client.create_cloud_api_v2_clouds_post(
                write_cloud=WriteCloud(
                    provider="AWS",
                    region=region,
                    credentials=AwsRoleArn.from_role_name(
                        user_aws_account_id, anyscale_iam_role_name
                    ).to_string(),
                    name=cloud_name,
                    is_bring_your_own_resource=False,
                    cluster_management_stack_version=cluster_management_stack_version,
                    auto_add_user=auto_add_user,
                    is_aioa=is_aioa,
                )
            ).result
            self.cloud_event_producer.set_cloud_id(created_cloud.id)
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.CLOUD_RECORD_INSERTED, succeeded=True,
            )
        except ClickException as e:
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.CLOUD_RECORD_INSERTED,
                succeeded=False,
                internal_error=str(e),
            )
            raise

        return anyscale_iam_role_name, created_cloud.id

    def prepare_for_managed_cloud_setup_gcp(  # noqa: PLR0913
        self,
        project_id: str,
        region: str,
        cloud_name: str,
        factory: Any,
        cluster_management_stack_version: ClusterManagementStackVersions,
        auto_add_user: bool,
        is_aioa: bool = False,
    ) -> Tuple[str, str, str]:
        setup_utils = try_import_gcp_managed_setup_utils()

        # choose an service account name and create a provider pool
        for _ in range(5):
            token = secrets.token_hex(4)
            anyscale_access_service_account = (
                f"anyscale-access-{token}@{project_id}.iam.gserviceaccount.com"
            )
            service_account = setup_utils.get_anyscale_gcp_access_service_acount(
                factory, anyscale_access_service_account
            )
            if service_account is not None:
                continue
            pool_id = f"anyscale-provider-pool-{token}"
            wordload_identity_pool = setup_utils.get_workload_identity_pool(
                factory, project_id, pool_id
            )
            if wordload_identity_pool is None:
                break
        else:
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.PREPROCESS_COMPLETE,
                succeeded=False,
                internal_error="Unable to find an available GCP service account name and create a provider pool.",
            )
            raise ClickException(
                "We weren't able to connect your account with the Anyscale because we weren't able to find an available service account name and create a provider pool in your GCP project. Please reach out to support or your SA for assistance."
            )

        # build credentials
        project_number = setup_utils.get_project_number(factory, project_id)
        provider_id = "anyscale-access"
        pool_name = f"{project_number}/locations/global/workloadIdentityPools/{pool_id}"
        provider_name = f"{pool_name}/providers/{provider_id}"
        credentials = json.dumps(
            {
                "project_id": project_id,
                "provider_id": provider_name,
                "service_account_email": anyscale_access_service_account,
            }
        )
        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.PREPROCESS_COMPLETE, succeeded=True,
        )

        # create a cloud
        try:
            created_cloud = self.api_client.create_cloud_api_v2_clouds_post(
                write_cloud=WriteCloud(
                    provider="GCP",
                    region=region,
                    credentials=credentials,
                    name=cloud_name,
                    is_bring_your_own_resource=False,
                    cluster_management_stack_version=cluster_management_stack_version,
                    auto_add_user=auto_add_user,
                    is_aioa=is_aioa,
                )
            ).result
            self.cloud_event_producer.set_cloud_id(created_cloud.id)
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.CLOUD_RECORD_INSERTED, succeeded=True,
            )
        except ClickException as e:
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.CLOUD_RECORD_INSERTED,
                succeeded=False,
                internal_error=str(e),
            )
            raise
        return anyscale_access_service_account, pool_name, created_cloud.id

    def create_workload_identity_federation_provider(
        self,
        factory: Any,
        project_id: str,
        pool_id: str,
        anyscale_access_service_account: str,
    ):
        setup_utils = try_import_gcp_managed_setup_utils()
        # create provider pool
        pool_display_name = "Anyscale provider pool"
        pool_description = f"Workload Identity Provider Pool for Anyscale access service account {anyscale_access_service_account}"

        wordload_identity_pool = setup_utils.create_workload_identity_pool(
            factory, project_id, pool_id, self.log, pool_display_name, pool_description,
        )
        try:
            # create provider
            provider_display_name = "Anyscale Access"
            provider_id = "anyscale-access"
            anyscale_aws_account = (
                self.api_client.get_anyscale_aws_account_api_v2_clouds_anyscale_aws_account_get().result.anyscale_aws_account
            )
            organization_id = get_organization_id(self.api_client)
            setup_utils.create_anyscale_aws_provider(
                factory,
                organization_id,
                wordload_identity_pool,
                provider_id,
                anyscale_aws_account,
                provider_display_name,
                self.log,
            )
        except ClickException as e:
            # delete provider pool if there's an exception
            setup_utils.delete_workload_identity_pool(
                factory, wordload_identity_pool, self.log
            )
            raise ClickException(
                f"Error occurred when trying to set up workload identity federation: {e}"
            )

    def wait_for_cloud_to_be_active(
        self, cloud_id: str, cloud_provider: CloudProviders
    ) -> None:
        """
        Waits for the cloud to be active
        """
        with self.log.spinner("Setting up resources on Anyscale for your new cloud..."):
            try:
                # The ingress for cloud admin zone may not be ready yet, so we need to wait for it to be active.
                # We use the cloud functional verification controller to create a cluster compute config to wait
                # for the cloud admin zone to be active.
                if self.initialize_auth_api_client:
                    CloudFunctionalVerificationController(
                        self.cloud_event_producer, self.log
                    ).get_or_create_cluster_compute(cloud_id, cloud_provider)
            except Exception:  # noqa: BLE001
                self.log.error(
                    "Timed out waiting for cloud admin zone to be active. Your cloud may not be set up properly. Please reach out to Anyscale support for assistance."
                )

    def setup_managed_cloud(  # noqa: PLR0912, PLR0913
        self,
        *,
        provider: str,
        region: str,
        name: str,
        functional_verify: Optional[str],
        cluster_management_stack_version: ClusterManagementStackVersions,
        enable_head_node_fault_tolerance: bool,
        project_id: Optional[str] = None,
        is_aioa: bool = False,
        yes: bool = False,
        boto3_session: Optional[
            boto3.Session
        ] = None,  # This is used by AIOA cloud setup
        _use_strict_iam_permissions: bool = False,  # This should only be used in testing.
        auto_add_user: bool = True,
        shared_storage: SharedStorageType = SharedStorageType.OBJECT_STORAGE,
    ) -> None:
        """
        Sets up a cloud provider
        """
        # TODO (congding): split this function into smaller functions per cloud provider
        functions_to_verify = self._validate_functional_verification_args(
            functional_verify
        )
        if provider == "aws":
            if boto3_session is None:
                # If boto3_session is not provided, we will create a new session with the given region.
                boto3_session = boto3.Session(region_name=region)
            if not validate_aws_credentials(self.log, boto3_session):
                raise ClickException(
                    "Cloud setup requires valid AWS credentials to be set locally. Learn more: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html"
                )
            self.cloud_event_producer.init_trace_context(
                CloudAnalyticsEventCommandName.SETUP, CloudProviders.AWS
            )
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.COMMAND_START, succeeded=True,
            )
            with self.log.spinner("Preparing environment for cloud setup..."):
                (
                    anyscale_iam_role_name,
                    cloud_id,
                ) = self.prepare_for_managed_cloud_setup(
                    region,
                    name,
                    cluster_management_stack_version,
                    auto_add_user,
                    is_aioa,
                    boto3_session,
                )

            try:
                from anyscale.commands.cloud_commands import (  # noqa: PLC0415
                    setup_vm_cloud_resource,
                )

                setup_vm_cloud_resource(
                    provider=provider,
                    region=region,
                    cloud_name=None,
                    cloud_id=cloud_id,
                    project_id=None,
                    enable_head_node_fault_tolerance=enable_head_node_fault_tolerance,
                    shared_storage=shared_storage,
                    controller=self,
                    boto3_session=boto3_session,
                    anyscale_iam_role_name=anyscale_iam_role_name,
                    cluster_node_iam_role_name=f"{cloud_id}-cluster_node_role",
                )
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.RESOURCES_CREATED, succeeded=True,
                )
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.INFRA_SETUP_COMPLETE, succeeded=True,
                )
            except Exception as e:  # noqa: BLE001
                self.log.error(str(e))
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.RESOURCES_CREATED,
                    succeeded=False,
                    logger=self.log,
                    internal_error=str(e),
                )
                self.api_client.delete_cloud_api_v2_clouds_cloud_id_delete(
                    cloud_id=cloud_id
                )
                raise ClickException("Cloud setup failed!")

            self.verify_aws_cloud_quotas(region=region, boto3_session=boto3_session)

            self.log.info(f"Successfully created cloud {name}, and it's ready to use.")
        elif provider == "gcp":
            if project_id is None:
                raise click.ClickException("Please provide a value for --project-id")
            gcp_utils = try_import_gcp_utils()
            setup_utils = try_import_gcp_managed_setup_utils()
            factory = gcp_utils.get_google_cloud_client_factory(self.log, project_id)
            self.cloud_event_producer.init_trace_context(
                CloudAnalyticsEventCommandName.SETUP, CloudProviders.GCP
            )
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.COMMAND_START, succeeded=True,
            )
            with self.log.spinner("Preparing environment for cloud setup..."):
                try:
                    organization_id = get_organization_id(self.api_client)
                    anyscale_aws_account = (
                        self.api_client.get_anyscale_aws_account_api_v2_clouds_anyscale_aws_account_get().result.anyscale_aws_account
                    )
                    # Enable APIs in the given GCP project
                    setup_utils.enable_project_apis(
                        factory, project_id, self.log, enable_head_node_fault_tolerance
                    )
                    # We need the Google APIs Service Agent to have security admin permissions on the project
                    # so that we can set IAM policy on Anyscale access service account
                    project_number = setup_utils.get_project_number(
                        factory, project_id
                    ).split("/")[-1]
                    google_api_service_agent = f"serviceAccount:{project_number}@cloudservices.gserviceaccount.com"
                    setup_utils.append_project_iam_policy(
                        factory,
                        project_id,
                        "roles/iam.securityAdmin",
                        google_api_service_agent,
                    )

                except ClickException as e:
                    self.cloud_event_producer.produce(
                        CloudAnalyticsEventName.PREPROCESS_COMPLETE,
                        succeeded=False,
                        logger=self.log,
                        internal_error=str(e),
                    )
                    self.log.error(str(e))
                    raise ClickException("Cloud setup failed!")

                (
                    anyscale_access_service_account,
                    pool_name,
                    cloud_id,
                ) = self.prepare_for_managed_cloud_setup_gcp(
                    project_id,
                    region,
                    name,
                    factory,
                    cluster_management_stack_version,
                    auto_add_user,
                    is_aioa,
                )

            pool_id = pool_name.split("/")[-1]
            deployment_name = cloud_id.replace("_", "-").lower()
            deployment_succeed = False
            try:
                with self.log.spinner(
                    "Creating workload identity federation provider for Anyscale access..."
                ):
                    self.create_workload_identity_federation_provider(
                        factory, project_id, pool_id, anyscale_access_service_account
                    )
                deployment_succeed = self.run_deployment_manager(
                    factory,
                    deployment_name,
                    cloud_id,
                    project_id,
                    region,
                    anyscale_access_service_account,
                    pool_name,
                    anyscale_aws_account,
                    organization_id,
                    enable_head_node_fault_tolerance,
                    shared_storage=shared_storage,
                )
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.RESOURCES_CREATED, succeeded=True,
                )
                with self.log.spinner(
                    "Updating Anyscale cloud with cloud resources..."
                ):
                    self.update_cloud_with_resources_gcp(
                        factory,
                        deployment_name,
                        cloud_id,
                        region,
                        project_id,
                        anyscale_access_service_account,
                        provider_name=f"{pool_name}/providers/anyscale-access",
                    )
                self.wait_for_cloud_to_be_active(cloud_id, CloudProviders.GCP)
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.INFRA_SETUP_COMPLETE, succeeded=True,
                )
            except Exception as e:  # noqa: BLE001
                self.log.error(str(e))
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.RESOURCES_CREATED,
                    succeeded=False,
                    internal_error=str(e),
                    logger=self.log,
                )
                self.api_client.delete_cloud_api_v2_clouds_cloud_id_delete(
                    cloud_id=cloud_id
                )
                if deployment_succeed:
                    # only clean up deployment if it's created successfully
                    # otherwise keep the deployment for customers to check the errors
                    setup_utils.delete_gcp_deployment(
                        factory, project_id, deployment_name
                    )
                setup_utils.delete_workload_identity_pool(factory, pool_name, self.log)
                raise ClickException("Cloud setup failed!")

            self.log.info(f"Successfully created cloud {name}, and it's ready to use.")
        else:
            raise ClickException(
                f"Invalid Cloud provider: {provider}. Available providers are [aws, gcp]."
            )

        if len(functions_to_verify) > 0:
            CloudFunctionalVerificationController(
                self.cloud_event_producer, self.log
            ).start_verification(
                cloud_id,
                self._get_cloud_provider_from_str(provider),
                functions_to_verify,
                yes,
            )

    def _add_redis_cluster_aws(
        self, cloud: Any, cloud_resource: CloudDeployment, yes: bool = False
    ):
        # Check if memorydb cluster already exists
        aws_config = cloud_resource.aws_config
        if isinstance(aws_config, dict):
            aws_config = AWSConfig(**aws_config)
            cloud_resource.aws_config = aws_config

        if aws_config and aws_config.memorydb_cluster_name is not None:
            self.log.info(
                f"AWS memorydb {aws_config.memorydb_cluster_name} already exists. "
            )
            return

        if not aws_config or not aws_config.cloudformation_id:
            raise ClickException(
                f"This cloud {cloud.name}({cloud.id}) does not have an associated cloudformation stack. Please contact Anyscale support."
            )

        # Get or create memorydb cluster
        try:
            memorydb_cluster_id = get_or_create_memorydb(
                cloud.region, aws_config.cloudformation_id, self.log, yes,
            )
            memorydb_cluster_config = _get_memorydb_cluster_config(
                memorydb_cluster_id, cloud.region, self.log
            )
        except Exception as e:  # noqa: BLE001
            raise ClickException(f"Failed to create memorydb cluster: {e}")

        # Update cloud resource record
        try:
            # Update the aws_config with memorydb cluster info
            cloud_resource.aws_config.memorydb_cluster_name = memorydb_cluster_id
            cloud_resource.aws_config.memorydb_cluster_arn = memorydb_cluster_config.id
            cloud_resource.aws_config.memorydb_cluster_endpoint = (
                memorydb_cluster_config.endpoint
            )
            self.api_client.update_cloud_resources_api_v2_clouds_cloud_id_resources_put(
                cloud_id=cloud.id, cloud_deployment=[cloud_resource],
            )
        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            raise ClickException(
                "Failed to update resources on Anyscale. Please reach out to Anyscale support or your SA for assistance."
            )

    def _add_redis_cluster_gcp(
        self, cloud: Any, cloud_resource: CloudDeployment, yes: bool = False
    ):
        gcp_config = cloud_resource.gcp_config
        if isinstance(gcp_config, dict):
            gcp_config = GCPConfig(**gcp_config)
            cloud_resource.gcp_config = gcp_config

        if gcp_config and gcp_config.memorystore_instance_name is not None:
            self.log.info(
                f"GCP memorystore {gcp_config.memorystore_instance_name} already exists. "
            )
            return

        if not gcp_config or not gcp_config.deployment_manager_id:
            raise ClickException(
                f"This cloud {cloud.name}({cloud.id}) does not have an associated GCP deployment manager. Please contact Anyscale support."
            )

        gcp_utils = try_import_gcp_utils()
        managed_setup_utils = try_import_gcp_managed_setup_utils()
        project_id = self._get_project_id(cloud, cloud.name, cloud.id)
        factory = gcp_utils.get_google_cloud_client_factory(self.log, project_id)
        try:
            memorystore_instance_name = managed_setup_utils.get_or_create_memorystore_gcp(
                factory,
                cloud.id,
                gcp_config.deployment_manager_id,
                project_id,
                cloud.region,
                self.log,
                yes,
            )
            memorystore_instance_config = gcp_utils.get_gcp_memorystore_config(
                factory, memorystore_instance_name
            )
        except Exception as e:  # noqa: BLE001
            raise ClickException(f"Failed to create GCP memorystore. Error: {e}")

        # Update cloud resource record
        try:
            # Update the gcp_config with memorystore instance info
            cloud_resource.gcp_config.memorystore_instance_name = (
                memorystore_instance_name
            )
            cloud_resource.gcp_config.memorystore_endpoint = (
                memorystore_instance_config.endpoint
            )
            self.api_client.update_cloud_resources_api_v2_clouds_cloud_id_resources_put(
                cloud_id=cloud.id, cloud_deployment=[cloud_resource],
            )
        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            raise ClickException(
                "Cloud update failed! Please reach out to Anyscale support or your SA for assistance."
            )

    def _update_auto_add_user_field(self, auto_add_user: bool, cloud) -> None:
        if cloud.auto_add_user == auto_add_user:
            self.log.info(
                f"No update required to the auto add user value. Cloud {cloud.name}({cloud.id}) "
                f"has auto add user {'enabled' if auto_add_user else 'disabled'}."
            )
        else:
            self.api_client.update_cloud_auto_add_user_api_v2_clouds_cloud_id_auto_add_user_put(
                cloud.id, auto_add_user=auto_add_user
            )
            if auto_add_user:
                self.log.info(
                    f"Auto add user for cloud {cloud.name}({cloud.id}) has been successfully enabled. Note: There may be up to 30 "
                    "sec delay for all users to be granted permissions after this feature is enabled."
                )
            else:
                self.log.info(
                    f"Auto add user for cloud {cloud.name}({cloud.id}) has been successfully disabled. No existing "
                    "cloud permissions were altered by this flag. Users added to the organization in the future will not "
                    "automatically be added to this cloud."
                )

    def _update_customer_aggregated_logs_config(self, cloud_id: str, is_enabled: bool):
        self.api_client.update_customer_aggregated_logs_config_api_v2_clouds_cloud_id_update_customer_aggregated_logs_config_put(
            cloud_id=cloud_id, is_enabled=is_enabled,
        )

    def update_cloud(  # noqa: PLR0912, C901
        self,
        cloud_name: Optional[str],
        cloud_id: Optional[str],
        functional_verify: Optional[str],
        enable_auto_add_user: Optional[bool] = None,
        enable_head_node_fault_tolerance: bool = False,
        resources_file: Optional[str] = None,
        yes: bool = False,
        skip_verification: bool = False,
    ) -> None:
        functions_to_verify = self._validate_functional_verification_args(
            functional_verify
        )

        cloud_id, cloud_name = get_cloud_id_and_name(
            self.api_client, cloud_id, cloud_name
        )
        cloud = self.api_client.get_cloud_api_v2_clouds_cloud_id_get(cloud_id).result

        if enable_auto_add_user is not None:
            self._update_auto_add_user_field(enable_auto_add_user, cloud)

        if cloud.is_bring_your_own_resource or cloud.is_bring_your_own_resource is None:
            # Customer-managed resources (cloud register), use resources file.
            if resources_file:
                self.update_cloud_resources(
                    cloud_name=cloud_name,
                    cloud_id=cloud_id,
                    resources_file=resources_file,
                    skip_verification=skip_verification,
                    yes=yes,
                )
            else:
                self.log.info(
                    "No cloud resources file provided, skipping cloud resource update."
                )
        else:
            # Anyscale-managed resources (cloud setup), can only update head node fault tolerance & inline IAM policies.
            self.update_managed_cloud(
                cloud=cloud,
                enable_head_node_fault_tolerance=enable_head_node_fault_tolerance,
                yes=yes,
            )

        self.log.info("Cloud update completed.")

        if len(functions_to_verify) > 0:
            CloudFunctionalVerificationController(
                self.cloud_event_producer, self.log
            ).start_verification(
                cloud_id,
                self._get_cloud_provider_from_str(cloud.provider),
                functions_to_verify,
                yes,
            )

    def update_managed_cloud(  # noqa: PLR0912, C901
        self, cloud: Cloud, enable_head_node_fault_tolerance: bool, yes: bool = False,
    ) -> None:
        """
        Updates managed cloud.

        If `enable_head_node_fault_tolerance` is set to True, we will add redis clusters to the cloud.
        Otherwise it only updates the inline IAM policies associated with the cross account IAM role for AWS clouds.
        """

        cloud_resources = self.api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
            cloud_id=cloud.id,
        ).results

        if not cloud_resources:
            raise ClickException(
                f"No cloud resources found for cloud {cloud.name} ({cloud.id})"
            )

        if len(cloud_resources) != 1:
            raise ClickException(
                f"Multiple cloud resources found for Anyscale-managed cloud {cloud.name} ({cloud.id})"
            )

        cloud_resource: CloudDeployment = (
            self._convert_decorated_cloud_resource_to_cloud_deployment(
                cloud_resources[0]
            )
        )

        if anyscale_version == "0.0.0-dev":
            confirm(
                "You are using a development version of Anyscale CLI. Still want to update the cloud?",
                yes,
            )

        self.cloud_event_producer.init_trace_context(
            CloudAnalyticsEventCommandName.UPDATE, cloud.provider, cloud.id
        )
        if enable_head_node_fault_tolerance:
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.COMMAND_START, succeeded=True,
            )
            try:
                if cloud.provider == CloudProviders.AWS:
                    self._add_redis_cluster_aws(cloud, cloud_resource, yes)
                elif cloud.provider == CloudProviders.GCP:
                    self._add_redis_cluster_gcp(cloud, cloud_resource, yes)
                else:
                    raise ClickException(
                        f"Unsupported cloud provider {cloud.provider}. Only AWS and GCP are supported for fault tolerance."
                    )
            except Exception as e:  # noqa: BLE001
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.REDIS_CLUSTER_ADDED,
                    succeeded=False,
                    internal_error=str(e),
                )
                raise ClickException(f"Cloud update failed! {e}")
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.REDIS_CLUSTER_ADDED, succeeded=True
            )
        else:
            # Update IAM permissions
            if cloud.provider != CloudProviders.AWS:
                raise ClickException(
                    f"Unsupported cloud provider {cloud.provider}. Only AWS is supported for updating inline IAM policies."
                )
            aws_config = cloud_resource.aws_config
            if isinstance(aws_config, dict):
                aws_config = AWSConfig(**aws_config)

            if not aws_config or not aws_config.cloudformation_id:
                raise ClickException(
                    f"This cloud {cloud.name}({cloud.id}) does not have an associated cloudformation stack. Please contact Anyscale support."
                )

            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.COMMAND_START, succeeded=True,
            )
            try:
                update_iam_role(
                    cloud.region, aws_config.cloudformation_id, self.log, yes
                )
            except Exception as e:  # noqa: BLE001
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.IAM_ROLE_UPDATED,
                    succeeded=False,
                    internal_error=str(e),
                )
                raise ClickException(f"Cloud update failed! {e}")
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.IAM_ROLE_UPDATED, succeeded=True,
            )

    def update_cors(
        self,
        cloud_name: Optional[str],
        cloud_id: Optional[str],
        resource: Optional[str] = None,
        cloud_resource_id: Optional[str] = None,
        yes: bool = False,
    ) -> None:
        """Update CORS configuration for cloud storage.

        This method updates CORS configuration on cloud storage buckets (S3, GCS, Azure Blob)
        to support Anyscale UI features like the file viewer. Works with both managed and
        customer-managed clouds.

        Args:
            cloud_name: Name of the cloud to update CORS for.
            cloud_id: ID of the cloud to update CORS for.
            resource: Name of the cloud resource to update. If not provided along with
                cloud_resource_id, updates all cloud resources.
            cloud_resource_id: ID of the cloud resource to update. If not provided along
                with resource, updates all cloud resources.
            yes: Skip asking for confirmation.
        """
        cloud_id, cloud_name = get_cloud_id_and_name(
            self.api_client, cloud_id, cloud_name
        )
        assert cloud_id is not None and cloud_name is not None
        cloud = self.api_client.get_cloud_api_v2_clouds_cloud_id_get(cloud_id).result

        cloud_resources = self.api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
            cloud_id=cloud_id,
        ).results

        if not cloud_resources:
            raise ClickException(
                f"No cloud resources found for cloud {cloud_name} ({cloud_id})"
            )

        # Convert all cloud resources
        all_cloud_resources = [
            self._convert_decorated_cloud_resource_to_cloud_deployment(r)
            for r in cloud_resources
        ]

        # Determine which resources to update
        if resource or cloud_resource_id:
            # Update a specific cloud resource
            resolved_resource_id = self._resolve_cloud_resource_id(
                cloud_id, resource, cloud_resource_id
            )
            resources_to_update = [
                r
                for r in all_cloud_resources
                if r.cloud_resource_id == resolved_resource_id
            ]
            if not resources_to_update:
                raise ClickException(
                    f"Cloud resource {cloud_resource_id or resource} not found in cloud {cloud_name} ({cloud_id})"
                )
        else:
            # Update all cloud resources
            resources_to_update = all_cloud_resources
            self.log.info(
                f"Updating CORS for all {len(resources_to_update)} cloud resource(s) in cloud {cloud_name}"
            )

        failed_resources: List[Tuple[str, str]] = []
        for cloud_resource in resources_to_update:
            resource_name = cloud_resource.name or cloud_resource.cloud_resource_id
            object_storage = cloud_resource.object_storage
            if isinstance(object_storage, dict):
                object_storage = ObjectStorage(**object_storage)
            bucket_name = object_storage.bucket_name if object_storage else None
            self.log.info(
                f"Updating CORS for cloud resource: {resource_name} (bucket: {bucket_name})"
            )
            try:
                self._update_cloud_cors(cloud, cloud_resource, yes)
            except ClickException as e:
                self.log.error(
                    f"Failed to update CORS for {resource_name}: {e.message}"
                )
                failed_resources.append((resource_name, e.message))

        if failed_resources:
            failed_names = ", ".join(name for name, _ in failed_resources)
            self.log.warning(
                f"CORS update completed with {len(failed_resources)} failure(s): {failed_names}"
            )
        else:
            self.log.info("CORS update completed.")

    def _update_cloud_cors(
        self, cloud: Cloud, cloud_resource: CloudDeployment, yes: bool = False,
    ) -> None:
        """Update CORS configuration for cloud storage."""
        self.cloud_event_producer.init_trace_context(
            CloudAnalyticsEventCommandName.UPDATE, cloud.provider, cloud.id
        )
        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.COMMAND_START, succeeded=True,
        )

        try:
            provider = cloud_resource.provider or cloud.provider
            if provider == CloudProviders.AWS:
                self._update_cors_aws(cloud, cloud_resource, yes)
            elif provider == CloudProviders.GCP:
                self._update_cors_gcp(cloud, cloud_resource, yes)
            elif provider == CloudProviders.AZURE:
                self._update_cors_azure(cloud, cloud_resource, yes)
            else:
                raise ClickException(
                    f"CORS update not supported for provider {provider}"
                )
        except ClickException:
            # Re-raise ClickExceptions as-is (they're user-facing errors)
            raise
        except Exception as e:  # noqa: BLE001
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.RESOURCES_EDITED,
                succeeded=False,
                internal_error=str(e),
            )
            raise ClickException(f"CORS update failed: {e}")

        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.RESOURCES_EDITED, succeeded=True,
        )

    def _update_cors_aws(
        self, cloud: Cloud, cloud_resource: CloudDeployment, yes: bool
    ) -> None:
        """Update CORS for AWS S3 bucket."""
        object_storage = cloud_resource.object_storage
        if isinstance(object_storage, dict):
            object_storage = ObjectStorage(**object_storage)

        if not object_storage or not object_storage.bucket_name:
            raise ClickException("No S3 bucket configured for this cloud")

        bucket_name = object_storage.bucket_name
        if bucket_name.startswith(S3_STORAGE_PREFIX):
            bucket_name = bucket_name[len(S3_STORAGE_PREFIX) :]
        if bucket_name.startswith(S3_ARN_PREFIX):
            bucket_name = bucket_name[len(S3_ARN_PREFIX) :]

        needs_update, reason = check_aws_cors_needs_update(bucket_name, cloud.region)

        if not needs_update:
            self.log.info(f"S3 bucket {bucket_name}: {reason}")
            return

        self.log.info(f"S3 bucket {bucket_name} needs CORS update: {reason}")
        confirm(f"Update CORS configuration for S3 bucket {bucket_name}?", yes)

        update_aws_cors(bucket_name, cloud.region)
        self.log.info(f"Successfully updated CORS for S3 bucket {bucket_name}")

    def _update_cors_gcp(
        self, cloud: Cloud, cloud_resource: CloudDeployment, yes: bool  # noqa: ARG002
    ) -> None:
        """Update CORS for GCP GCS bucket."""
        object_storage = cloud_resource.object_storage
        if isinstance(object_storage, dict):
            object_storage = ObjectStorage(**object_storage)

        gcp_config = cloud_resource.gcp_config
        if isinstance(gcp_config, dict):
            gcp_config = GCPConfig(**gcp_config)

        if not object_storage or not object_storage.bucket_name:
            raise ClickException("No GCS bucket configured for this cloud")

        bucket_name = object_storage.bucket_name
        if bucket_name.startswith(GCS_STORAGE_PREFIX):
            bucket_name = bucket_name[len(GCS_STORAGE_PREFIX) :]

        project_id = gcp_config.project_id if gcp_config else None

        needs_update, reason = check_gcp_cors_needs_update(bucket_name, project_id)

        if not needs_update:
            self.log.info(f"GCS bucket {bucket_name}: {reason}")
            return

        self.log.info(f"GCS bucket {bucket_name} needs CORS update: {reason}")
        confirm(f"Update CORS configuration for GCS bucket {bucket_name}?", yes)

        update_gcp_cors(bucket_name, project_id)
        self.log.info(f"Successfully updated CORS for GCS bucket {bucket_name}")

    def _update_cors_azure(
        self, cloud: Cloud, cloud_resource: CloudDeployment, yes: bool  # noqa: ARG002
    ) -> None:
        """Update CORS for Azure Blob storage."""
        object_storage = cloud_resource.object_storage
        if isinstance(object_storage, dict):
            object_storage = ObjectStorage(**object_storage)

        if not object_storage or not object_storage.bucket_name:
            raise ClickException("No Azure storage configured for this cloud")

        # Parse: abfss://container@account.dfs.core.windows.net
        match = re.match(
            r"abfss://[^@]+@([^.]+)\.dfs\.core\.windows\.net",
            object_storage.bucket_name,
        )
        if not match:
            raise ClickException(
                f"Invalid Azure storage URL: {object_storage.bucket_name}"
            )

        account_name = match.group(1)

        needs_update, reason = check_azure_cors_needs_update(account_name)

        if not needs_update:
            self.log.info(f"Azure storage {account_name}: {reason}")
            return

        self.log.info(f"Azure storage {account_name} needs CORS update: {reason}")
        confirm(f"Update CORS configuration for Azure storage {account_name}?", yes)

        update_azure_cors(account_name)
        self.log.info(f"Successfully updated CORS for Azure storage {account_name}")

    # Avoid displaying fields with empty values (since the values for optional fields default to None).
    def _remove_empty_values(self, d):
        if isinstance(d, dict):
            return {
                k: self._remove_empty_values(v)
                for k, v in d.items()
                if self._remove_empty_values(v)
            }
        if isinstance(d, list):
            return [self._remove_empty_values(v) for v in d]
        return d

    def get_decorated_cloud_resources(
        self, cloud_id: str
    ) -> List[DecoratedCloudResource]:
        cloud = self.api_client.get_cloud_api_v2_clouds_cloud_id_get(
            cloud_id=cloud_id,
        ).result

        if cloud.is_aioa:
            raise ValueError(
                "Listing cloud resources is only supported for customer-hosted clouds."
            )

        try:
            return self.api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
                cloud_id=cloud_id,
            ).results
        except Exception as e:  # noqa: BLE001
            raise ClickException(
                f"Failed to get cloud resources for cloud {cloud.name} ({cloud_id}). Error: {e}"
            )

    def get_formatted_cloud_resources(self, cloud_id: str) -> List[Any]:
        cloud_resources = self.get_decorated_cloud_resources(cloud_id)
        formatted_cloud_resources = [
            self._remove_empty_values(cloud_resource.to_dict())
            for cloud_resource in cloud_resources
        ]
        # Remove the deprecated cloud_deployment_id field.
        for d in formatted_cloud_resources:
            d.pop("cloud_deployment_id", None)
        return formatted_cloud_resources

    def _convert_decorated_cloud_resource_to_cloud_deployment(
        self, decorated_cloud_resource: DecoratedCloudResource
    ) -> CloudDeployment:
        # DecoratedCloudResource has extra fields that are not in CloudDeployment.
        allowed_keys = set(CloudDeployment.attribute_map.keys())
        allowed_keys.remove(
            "cloud_deployment_id"
        )  # Remove deprecated cloud_deployment_id field.
        return CloudDeployment(
            **{
                k: v
                for k, v in decorated_cloud_resource.to_dict().items()
                if k in allowed_keys
            }
        )

    def get_cloud_resources(self, cloud_id: str) -> List[CloudDeployment]:
        decorated_cloud_resources = self.get_decorated_cloud_resources(cloud_id)
        return [
            self._convert_decorated_cloud_resource_to_cloud_deployment(resource)
            for resource in decorated_cloud_resources
        ]

    def update_aws_anyscale_iam_role(
        self,
        cloud_id: str,
        region: str,
        anyscale_iam_role_id: Optional[str],
        external_id: Optional[str],
    ) -> Tuple[Optional[Boto3Resource], Optional[str]]:
        """
        Updates the Anyscale IAM role's assume policy to include the cloud ID as the external ID.
        Returns the role and the original policy document.
        """
        if not anyscale_iam_role_id:
            # anyscale_iam_role_id is optional for k8s
            return None, None

        organization_id = get_organization_id(self.api_client)
        if external_id and not external_id.startswith(organization_id):
            raise ClickException(
                f"Invalid external ID: external ID must start with the organization ID: {organization_id}"
            )

        # Update anyscale IAM role's assume policy to include the cloud id as the external ID
        role = _get_role(
            AwsRoleArn.from_string(anyscale_iam_role_id).to_role_name(), region
        )
        if role is None:
            self.log.log_resource_error(
                CloudAnalyticsEventCloudResource.AWS_IAM_ROLE,
                CloudSetupError.RESOURCE_NOT_FOUND,
            )
            raise ClickException(f"Failed to access IAM role {anyscale_iam_role_id}.")

        iam_role_original_policy = role.assume_role_policy_document  # type: ignore
        if external_id is None:
            try:
                new_policy = _update_external_ids_for_policy(
                    iam_role_original_policy, cloud_id
                )
                role.AssumeRolePolicy().update(PolicyDocument=json.dumps(new_policy))  # type: ignore
            except ClientError as e:
                self.log.log_resource_exception(
                    CloudAnalyticsEventCloudResource.AWS_IAM_ROLE, e
                )
                raise e
        else:
            fetched_external_ids = [
                statement.setdefault("Condition", {})
                .setdefault("StringEquals", {})
                .setdefault("sts:ExternalId", [])
                for statement in iam_role_original_policy.get("Statement", [])  # type: ignore
            ]
            external_id_in_policy = all(
                external_id == fetched_external_id
                if isinstance(fetched_external_id, str)
                else external_id in fetched_external_id
                for fetched_external_id in fetched_external_ids
            )
            if not external_id_in_policy:
                raise ClickException(
                    f"External ID {external_id} is not in the assume role policy of {anyscale_iam_role_id}."
                )

        return role, iam_role_original_policy

    def _generate_diff(self, existing: List[Any], new: List[Any]) -> str:
        """
        Generates a diff between the existing and new dicts.
        """

        diff = difflib.unified_diff(
            yaml.dump(existing).splitlines(keepends=True),
            yaml.dump(new).splitlines(keepends=True),
            lineterm="",
        )

        formatted_diff = ""
        for d in diff:
            if d.startswith("+") and not d.startswith("+++"):
                formatted_diff += "{}{}{}".format(
                    colorama.Fore.GREEN, d, colorama.Style.RESET_ALL
                )
            elif d.startswith("-") and not d.startswith("---"):
                formatted_diff += "{}{}{}".format(
                    colorama.Fore.RED, d, colorama.Style.RESET_ALL
                )
            else:
                formatted_diff += d

        return formatted_diff.strip()

    # Returns the role and original IAM policy, so that we can revert it if creating the cloud resource fails.
    def _preprocess_aws(  # noqa: PLR0912
        self, cloud_id: str, deployment: CloudDeployment
    ) -> Tuple[Optional[Boto3Resource], Optional[str]]:
        if not deployment.aws_config and not deployment.file_storage:
            return None, None

        if not validate_aws_credentials(self.log):
            raise ClickException(
                "Updating cloud resources requires valid AWS credentials to be set locally. Learn more: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html"
            )

        role, iam_role_original_policy = None, None

        # Get EFS mount target IP.
        file_storage = None
        if deployment.file_storage:
            if isinstance(deployment.file_storage, dict):
                file_storage = FileStorage(**deployment.file_storage)
            else:
                file_storage = deployment.file_storage

            if file_storage.file_storage_id:
                try:
                    boto3_session = boto3.Session(region_name=deployment.region)
                    efs_mount_target_ip = _get_aws_efs_mount_target_ip(
                        boto3_session, file_storage.file_storage_id,
                    )
                    if not efs_mount_target_ip:
                        raise ClickException(
                            f"EFS mount target IP not found for {file_storage.file_storage_id}."
                        )
                    file_storage.mount_targets = [
                        NFSMountTarget(address=efs_mount_target_ip)
                    ]
                except ClientError as e:
                    self.log.log_resource_exception(
                        CloudAnalyticsEventCloudResource.AWS_EFS, e
                    )
                    raise e

            deployment.file_storage = file_storage

        if deployment.aws_config:
            if isinstance(deployment.aws_config, dict):
                aws_config = AWSConfig(**deployment.aws_config)
            else:
                aws_config = deployment.aws_config

            assert deployment.region

            # Update Anyscale IAM role's assume policy to include the cloud ID as the external ID.
            role, iam_role_original_policy = self.update_aws_anyscale_iam_role(
                cloud_id,
                deployment.region,
                aws_config.anyscale_iam_role_id,
                aws_config.external_id,
            )
            if aws_config.external_id is None:
                aws_config.external_id = cloud_id

            # Get zones corresponding to subnet IDs.
            if aws_config.subnet_ids:
                subnets_with_azs = associate_aws_subnets_with_azs(
                    aws_config.subnet_ids, deployment.region, self.log
                )
                aws_config.zones = [s.availability_zone for s in subnets_with_azs]

            # Get memorydb config.
            if aws_config.memorydb_cluster_name:
                memorydb_cluster_config = _get_memorydb_cluster_config(
                    aws_config.memorydb_cluster_name, deployment.region, self.log,
                )
                assert memorydb_cluster_config
                aws_config.memorydb_cluster_arn = memorydb_cluster_config.id
                aws_config.memorydb_cluster_endpoint = memorydb_cluster_config.endpoint

            deployment.aws_config = aws_config

        return role, iam_role_original_policy

    def _preprocess_gcp(
        self, deployment: CloudDeployment,
    ):
        if not deployment.gcp_config:
            return

        if isinstance(deployment.gcp_config, dict):
            gcp_config = GCPConfig(**deployment.gcp_config)
        else:
            gcp_config = deployment.gcp_config

        deployment.gcp_config = gcp_config
        if not deployment.file_storage and not gcp_config.memorystore_instance_name:
            return

        if not gcp_config.project_id:
            raise ClickException(
                '"project_id" is required to configure filestore or memorystore'
            )

        gcp_utils = try_import_gcp_utils()
        factory = gcp_utils.get_google_cloud_client_factory(
            self.log, gcp_config.project_id
        )

        # Get Filestore mount target IP and root dir.
        if deployment.file_storage:
            if isinstance(deployment.file_storage, dict):
                fs = FileStorage(**deployment.file_storage)
            else:
                fs = deployment.file_storage

            if fs.file_storage_id:
                if not gcp_config.vpc_name:
                    raise ClickException(
                        '"vpc_name" is required to configure filestore'
                    )
                filestore_config = gcp_utils.get_gcp_filestore_config_from_full_name(
                    factory, gcp_config.vpc_name, fs.file_storage_id, self.log,
                )
                if not filestore_config:
                    raise ClickException(
                        f"Filestore config not found for {fs.file_storage_id}."
                    )
                fs.mount_path = filestore_config.root_dir
                fs.mount_targets = [
                    NFSMountTarget(address=filestore_config.mount_target_ip)
                ]

            deployment.file_storage = fs

        # Get Memorystore config.
        if gcp_config.memorystore_instance_name:
            memorystore_config = gcp_utils.get_gcp_memorystore_config(
                factory, gcp_config.memorystore_instance_name
            )
            assert memorystore_config
            gcp_config.memorystore_endpoint = memorystore_config.endpoint

            deployment.gcp_config = gcp_config

    def create_cloud_resource(
        self,
        cloud: Optional[str],
        cloud_id: Optional[str],
        spec_file: str,
        skip_verification: bool = False,
        yes: bool = False,
    ) -> str:
        cloud_id, _ = get_cloud_id_and_name(
            self.api_client, cloud_id=cloud_id, cloud_name=cloud
        )
        assert cloud_id

        # Read the spec file.
        path = pathlib.Path(spec_file)
        if not path.exists():
            raise ClickException(f"{spec_file} does not exist.")
        if not path.is_file():
            raise ClickException(f"{spec_file} is not a file.")

        spec = yaml.safe_load(path.read_text())
        try:
            new_deployment = CloudDeployment(**spec)
        except Exception as e:  # noqa: BLE001
            raise ClickException(f"Failed to parse cloud resource: {e}")

        if new_deployment.provider == CloudProviders.AWS:
            self._preprocess_aws(cloud_id=cloud_id, deployment=new_deployment)
        elif new_deployment.provider == CloudProviders.GCP:
            self._preprocess_gcp(deployment=new_deployment)

        if not skip_verification and not self.verify_cloud_deployment(
            cloud_id=cloud_id, cloud_deployment=new_deployment
        ):
            raise ClickException("Cloud resource verification failed.")

        # Log an additional warning if a new deployment is being added but a deployment with the same AWS/GCP region already exists.
        existing_resources = {
            resource.cloud_resource_id: resource
            for resource in self.get_cloud_resources(cloud_id)
        }
        existing_stack_provider_regions = {
            (d.compute_stack, d.provider, d.region)
            for d in existing_resources.values()
            if d.provider in (CloudProviders.AWS, CloudProviders.GCP)
        }
        if (
            new_deployment.compute_stack,
            new_deployment.provider,
            new_deployment.region,
        ) in existing_stack_provider_regions:
            self.log.warning(
                f"A {new_deployment.provider} {new_deployment.compute_stack} resource in region {new_deployment.region} already exists."
            )
            confirm("Would you like to proceed with adding this cloud resource?", yes)

        # Add the resource.
        try:
            response = self.api_client.add_cloud_resource_api_v2_clouds_cloud_id_add_resource_put(
                cloud_id=cloud_id, cloud_deployment=new_deployment,
            )
        except Exception as e:  # noqa: BLE001
            raise ClickException(f"Failed to add cloud resource: {e}")

        # Extract cloud_resource_id from the response
        cloud_resource_id = response.result.cloud_resource_id

        self.log.info(
            f"Successfully created cloud resource{' ' + (new_deployment.name or '')} in cloud {cloud or cloud_id}!"
        )

        return cloud_resource_id

    def update_cloud_resources(  # noqa: PLR0912, C901
        self,
        cloud_name: Optional[str],
        cloud_id: Optional[str],
        resources_file: str,
        skip_verification: bool = False,
        yes: bool = False,
    ):
        if not cloud_id:
            cloud_id, _ = get_cloud_id_and_name(self.api_client, cloud_name=cloud_name)
        assert cloud_id

        # Read the spec file.
        path = pathlib.Path(resources_file)
        if not path.exists():
            raise ClickException(f"{resources_file} does not exist.")
        if not path.is_file():
            raise ClickException(f"{resources_file} is not a file.")

        spec = yaml.safe_load(path.read_text())

        # Normalize spec to a list of cloud resources.
        # Accepts either:
        # 1. A list of CloudResource objects: [{"cloud_resource_id": ..., ...}, ...]
        # 2. A Cloud object with a "resources" key: {"name": ..., "resources": [...]}
        #    (This is the output format of `anyscale cloud get`)
        if isinstance(spec, dict):
            # Check if this is a Cloud object (output of `anyscale cloud get`)
            # by looking for the "resources" key containing a list.
            if "resources" in spec and isinstance(spec["resources"], list):
                spec = spec["resources"]
            else:
                raise ClickException(
                    "Invalid cloud resources file format. Must contain either a list of CloudResources, "
                    "or a Cloud object with a 'resources' key (output of `anyscale cloud get`)."
                )
        elif not isinstance(spec, list):
            raise ClickException(
                "Invalid cloud resources file format. Must contain either a list of CloudResources, "
                "or a Cloud object with a 'resources' key (output of `anyscale cloud get`)."
            )

        # Get the existing spec.
        existing_resources = self.get_cloud_resources(cloud_id=cloud_id)

        if len(existing_resources) > len(spec):
            raise ClickException(
                "Please use `anyscale cloud resource delete` to remove cloud resources."
            )
        if len(existing_resources) < len(spec):
            raise ClickException(
                "Please use `anyscale cloud resource create` to add cloud resources."
            )

        existing_resources_dict = {
            resource.cloud_resource_id: resource for resource in existing_resources
        }

        all_deployments: List[CloudDeployment] = []
        updated_deployments: List[CloudDeployment] = []
        for d in spec:
            try:
                deployment = CloudDeployment(**d)
            except Exception as e:  # noqa: BLE001
                try:
                    # Try to parse the cloud deployment as a DecoratedCloudResource as well,
                    # which has extra fields that are not in CloudDeployment.
                    deployment = self._convert_decorated_cloud_resource_to_cloud_deployment(
                        DecoratedCloudResource(**d)
                    )
                except:  # noqa: E722
                    # Raise original error from parsing as CloudDeployment.
                    raise ClickException(f"Failed to parse cloud resource: {e}")

            if not deployment.cloud_resource_id:
                raise ClickException(
                    "All cloud resources must include a cloud_resource_id."
                )
            if deployment.cloud_resource_id not in existing_resources_dict:
                raise ClickException(
                    f"Cloud resource {deployment.cloud_resource_id} not found."
                )
            if deployment.provider == CloudProviders.PCP:
                raise ClickException(
                    "Please use the `anyscale machine-pool` CLI to update machine pools."
                )

            all_deployments.append(deployment)

            # Compare using same logic as diff generation to avoid inconsistencies
            existing = existing_resources_dict[deployment.cloud_resource_id]
            deployment_clean = self._remove_empty_values(deployment.to_dict())
            existing_clean = self._remove_empty_values(existing.to_dict())
            if deployment_clean != existing_clean:
                updated_deployments.append(deployment)

        # Diff the existing and new specs and confirm.
        diff = self._generate_diff(
            [self._remove_empty_values(r.to_dict()) for r in existing_resources],
            [self._remove_empty_values(r.to_dict()) for r in all_deployments],
        )
        if not diff:
            self.log.info("No changes detected.")
            return

        self.log.info(f"Detected the following changes:\n{diff}")

        confirm("Would you like to proceed with updating this cloud?", yes)

        # Preprocess the deployments if necessary.
        for deployment in updated_deployments:
            if deployment.provider == CloudProviders.AWS:
                self._preprocess_aws(cloud_id=cloud_id, deployment=deployment)
            elif deployment.provider == CloudProviders.GCP:
                self._preprocess_gcp(deployment=deployment)

            # Skip verification for Kubernetes stacks or if explicitly requested
            if deployment.compute_stack == ComputeStack.K8S:
                self.log.info("Skipping verification for Kubernetes compute stack.")
            elif not skip_verification and not self.verify_cloud_deployment(
                cloud_id=cloud_id, cloud_deployment=deployment
            ):
                raise ClickException(
                    f"Verification failed for cloud resource {deployment.name or deployment.cloud_resource_id}."
                )

        # Update the cloud resources.
        try:
            self.api_client.update_cloud_resources_api_v2_clouds_cloud_id_resources_put(
                cloud_id=cloud_id, cloud_deployment=updated_deployments,
            )
        except Exception as e:  # noqa: BLE001
            raise ClickException(f"Failed to update cloud resources: {e}")

        self.log.info(f"Successfully updated cloud {cloud_name or cloud_id}.")

    def remove_cloud_resource(
        self, cloud_name: str, resource_name: str, yes: bool,
    ):
        confirm(
            f"Please confirm that you would like to remove resource {resource_name} from cloud {cloud_name}.",
            yes,
        )

        cloud_id, _ = get_cloud_id_and_name(self.api_client, cloud_name=cloud_name)
        try:
            with self.log.spinner("Removing cloud resource..."):
                self.api_client.remove_cloud_resource_api_v2_clouds_cloud_id_remove_resource_delete(
                    cloud_id=cloud_id, cloud_resource_name=resource_name,
                )
        except Exception as e:  # noqa: BLE001
            raise ClickException(f"Failed to remove cloud resource: {e}")

        self.log.warning(
            "The trust policy or service account that provides access to Anyscale's control plane needs to be deleted manually if you no longer wish for Anyscale to have access."
        )

        self.log.info(
            f"Successfully removed resource {resource_name} from cloud {cloud_name}!"
        )

    def _resolve_cloud_resource_id(
        self,
        cloud_id: str,
        resource: Optional[str] = None,
        cloud_resource_id: Optional[str] = None,
    ) -> str:
        """
        Resolve cloud resource ID based on resolution order: id > name > primary.

        Returns the resolved cloud_resource_id to use for API calls.
        """
        # If cloud_resource_id is provided, use it directly
        if cloud_resource_id:
            self.log.info(f"Using provided cloud resource ID: {cloud_resource_id}")
            return cloud_resource_id

        # If resource name is provided, resolve by name
        if resource:
            # Get all cloud resources to resolve by name
            cloud_resources = self.api_client.get_cloud_deployments_api_v2_clouds_cloud_id_deployments_get(
                cloud_id=cloud_id
            ).results

            if not cloud_resources:
                raise RuntimeError(f"No cloud resources found for cloud {cloud_id}")

            # Find resources with matching name
            matching_resources = [r for r in cloud_resources if r.name == resource]

            if not matching_resources:
                raise RuntimeError(
                    f"No cloud resource found with name '{resource}' in cloud {cloud_id}"
                )

            if len(matching_resources) > 1:
                raise RuntimeError(
                    f"Multiple cloud resources found with name '{resource}'. "
                    f"Please use --cloud-resource-id to specify which resource to use. "
                    f"Available resource IDs: {[r.cloud_deployment_id for r in matching_resources]}"
                )

            resolved_id = matching_resources[0].cloud_deployment_id
            self.log.info(f"Resolved resource name '{resource}' to ID: {resolved_id}")
            return resolved_id

        # Default to primary resource (marked with is_default=True)
        cloud_resources = self.api_client.get_cloud_deployments_api_v2_clouds_cloud_id_deployments_get(
            cloud_id=cloud_id
        ).results

        if not cloud_resources:
            raise RuntimeError(f"No cloud resources found for cloud {cloud_id}")

        # Find primary resource (is_default=True)
        primary_resources = [r for r in cloud_resources if r.is_default]

        if not primary_resources:
            raise RuntimeError(f"No primary cloud resource found for cloud {cloud_id}")

        if len(primary_resources) > 1:
            raise RuntimeError(
                f"Multiple primary cloud resources found for cloud {cloud_id}"
            )

        resolved_id = primary_resources[0].cloud_deployment_id
        self.log.info(f"Using primary cloud resource ID: {resolved_id}")
        return resolved_id

    def get_cloud_config(
        self,
        cloud_name: Optional[str] = None,
        cloud_id: Optional[str] = None,
        resource: Optional[str] = None,
        cloud_resource_id: Optional[str] = None,
    ) -> CloudDeploymentConfig:
        """Get a cloud's current JSON configuration."""

        resolved_cloud_id, cloud_name = get_cloud_id_and_name(
            self.api_client, cloud_id, cloud_name
        )

        # Resolve the cloud resource ID to use
        resolved_cloud_resource_id = self._resolve_cloud_resource_id(
            resolved_cloud_id, resource, cloud_resource_id
        )

        config: CloudDeploymentConfig = self.api_client.get_cloud_deployment_config_api_v2_clouds_cloud_id_deployment_cloud_deployment_id_config_get(
            cloud_id=resolved_cloud_id, cloud_deployment_id=resolved_cloud_resource_id
        ).result

        return config

    def update_cloud_config(
        self,
        cloud_name: Optional[str] = None,
        cloud_id: Optional[str] = None,
        enable_log_ingestion: Optional[bool] = None,
        spec_file: Optional[str] = None,
        resource: Optional[str] = None,
        cloud_resource_id: Optional[str] = None,
    ):
        """Update a cloud's configuration."""
        if enable_log_ingestion is None and spec_file is None:
            return

        resolved_cloud_id, cloud_name = get_cloud_id_and_name(
            self.api_client, cloud_id, cloud_name
        )
        if enable_log_ingestion is not None:
            self._update_customer_aggregated_logs_config(
                cloud_id=resolved_cloud_id,
                is_enabled=enable_log_ingestion,  # type: ignore
            )
            self.log.info(
                f"Successfully updated log ingestion configuration for cloud, "
                f"{resolved_cloud_id} to {enable_log_ingestion}"
            )
        elif spec_file is not None:
            path = pathlib.Path(spec_file)
            if not path.exists():
                raise FileNotFoundError(f"File {spec_file} does not exist.")

            if not path.is_file():
                raise ValueError(f"File {spec_file} is not a file.")

            # Resolve the cloud resource ID to use
            resolved_cloud_resource_id = self._resolve_cloud_resource_id(
                resolved_cloud_id, resource, cloud_resource_id
            )

            spec = yaml.safe_load(path.read_text())
            config = CloudDeploymentConfig(spec=spec)
            self.api_client.update_cloud_deployment_config_api_v2_clouds_cloud_id_deployment_cloud_deployment_id_config_put(
                cloud_id=resolved_cloud_id,
                cloud_deployment_id=resolved_cloud_resource_id,
                cloud_deployment_config=config,
            )
            self.log.info(
                f"Successfully updated cloud configuration for cloud {cloud_name} "
                f"(resource: {resolved_cloud_resource_id})"
            )

    def set_default_cloud(
        self, cloud_name: Optional[str], cloud_id: Optional[str],
    ) -> None:
        """
        Sets default cloud for caller's organization. This operation can only be performed
        by organization admins, and the default cloud must have organization level
        permissions.
        """

        cloud_id, cloud_name = get_cloud_id_and_name(
            self.api_client, cloud_id, cloud_name
        )

        self.api_client.update_default_cloud_api_v2_organizations_update_default_cloud_post(
            cloud_id=cloud_id
        )

        self.log.info(f"Updated default cloud to {cloud_name}")

    def update_system_cluster_config(
        self,
        cloud_name: Optional[str],
        cloud_id: Optional[str],
        system_cluster_enabled: Optional[bool],
    ) -> None:
        """Update system cluster configuration for a cloud."""
        if system_cluster_enabled is None:
            return

        cloud_id, cloud_name = get_cloud_id_and_name(
            self.api_client, cloud_id, cloud_name
        )

        self.api_client.update_system_cluster_config_api_v2_clouds_cloud_id_update_system_cluster_config_put(
            cloud_id=cloud_id, is_enabled=system_cluster_enabled,
        )
        if system_cluster_enabled:
            self.log.info(f"Successfully enabled system cluster for cloud {cloud_id}")
        else:
            self.log.info(
                f"Successfully disabled system cluster for cloud {cloud_id}\n"
                "Note: if the system cluster is currently running, it will continue to run until it is terminated.\n"
                "To terminate the system cluster, use the CLI command: "
                f"anyscale cloud terminate-system-cluster --cloud-id {cloud_id} --wait"
            )

    def _passed_or_failed_str_from_bool(self, is_passing: bool) -> str:
        return "PASSED" if is_passing else "FAILED"

    @staticmethod
    def _get_cloud_provider_from_str(provider: str) -> CloudProviders:
        if provider.lower() == "aws":
            return CloudProviders.AWS
        elif provider.lower() == "gcp":
            return CloudProviders.GCP
        else:
            raise ClickException(
                f"Unsupported provider {provider}. Supported providers are [aws, gcp]."
            )

    def _validate_functional_verification_args(
        self, functional_verify: Optional[str]
    ) -> List[CloudFunctionalVerificationType]:
        if functional_verify is None:
            return []
        # validate functional_verify
        functions_to_verify = set()
        for function in functional_verify.split(","):
            fn = getattr(CloudFunctionalVerificationType, function.upper(), None)
            if fn is None:
                raise ClickException(
                    f"Unsupported function {function} for --functional-verify. "
                    f"Supported functions: {[function.lower() for function in CloudFunctionalVerificationType]}"
                )
            functions_to_verify.add(fn)
        return list(functions_to_verify)

    def verify_cloud(  # noqa: PLR0911
        self,
        *,
        cloud_name: Optional[str],
        cloud_id: Optional[str],
        functional_verify: Optional[str],
        boto3_session: Optional[Any] = None,
        strict: bool = False,
        _use_strict_iam_permissions: bool = False,  # This should only be used in testing.
        yes: bool = False,
    ) -> bool:
        """
        Verifies a cloud by name or id, including all cloud resources.

        Note: If your changes involve operations that may require additional permissions
        (for example, `boto3_session.client("efs").describe_backup_policy`), it's important
        to run the end-to-end test `bk_e2e/test_cloud.py` locally before pushing the changes.
        This way, you can ensure that your changes will not break the tests.
        """
        functions_to_verify = self._validate_functional_verification_args(
            functional_verify
        )

        cloud_id, cloud_name = get_cloud_id_and_name(
            self.api_client, cloud_id, cloud_name
        )

        assert cloud_id is not None

        cloud = self.api_client.get_cloud_api_v2_clouds_cloud_id_get(cloud_id).result

        if cloud.state in (CloudState.DELETING, CloudState.DELETED):
            self.log.info(
                f"This cloud {cloud_name}({cloud_id}) is either during deletion or deleted. Skipping verification."
            )
            return False

        try:
            cloud_resources = self.api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
                cloud_id=cloud_id,
            ).results
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to retrieve cloud resources: {e}")
            return False

        if not cloud_resources:
            self.log.error("No cloud resources found for this cloud")
            return False

        self.cloud_event_producer.init_trace_context(
            CloudAnalyticsEventCommandName.VERIFY,
            cloud_provider=cloud.provider,
            cloud_id=cloud_id,
        )
        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.COMMAND_START, succeeded=True
        )

        cloud_resource_results = []
        for cloud_resource in cloud_resources:
            try:
                cloud_resource_name = (
                    cloud_resource.name or cloud_resource.cloud_resource_id
                )

                self.log.info(f"Verifying cloud resource: {cloud_resource_name}")
                result = self.verify_cloud_deployment(
                    cloud_id,
                    cloud_resource,
                    strict=strict,
                    _use_strict_iam_permissions=_use_strict_iam_permissions,
                    boto3_session=boto3_session,
                )
                cloud_resource_results.append((cloud_resource_name, result))

            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                cloud_resource_name = getattr(cloud_resource, "name", None) or getattr(
                    cloud_resource, "cloud_resource_id", "unknown"
                )
                self.log.error(
                    f"Failed to verify cloud resource {cloud_resource_name}: {e}"
                )
                cloud_resource_results.append((cloud_resource_name, False))

        self._print_cloud_resource_verification_results(cloud_resource_results)

        overall_success = all(result for _, result in cloud_resource_results)

        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.RESOURCES_VERIFIED, succeeded=overall_success,
        )

        if not overall_success:
            return False

        if len(functions_to_verify) > 0:
            return CloudFunctionalVerificationController(
                self.cloud_event_producer, self.log
            ).start_verification(cloud_id, cloud.provider, functions_to_verify, yes=yes)

        return True

    def verify_cloud_deployment(
        self,
        cloud_id: str,
        cloud_deployment: CloudDeployment,
        strict: bool = False,
        _use_strict_iam_permissions: bool = False,  # This should only be used in testing.
        boto3_session: Optional[boto3.Session] = None,
    ) -> bool:
        if cloud_deployment.compute_stack == ComputeStack.VM:
            if cloud_deployment.provider == CloudProviders.AWS:
                return self.verify_aws_cloud_resources_for_cloud_deployment(
                    cloud_id=cloud_id,
                    cloud_deployment=cloud_deployment,
                    strict=strict,
                    _use_strict_iam_permissions=_use_strict_iam_permissions,
                    boto3_session=boto3_session,
                )
            elif cloud_deployment.provider == CloudProviders.GCP:
                return self.verify_gcp_cloud_resources_from_cloud_deployment(
                    cloud_id=cloud_id, cloud_deployment=cloud_deployment, strict=strict,
                )
            else:
                raise ValueError(
                    f"Unsupported cloud provider: {cloud_deployment.provider}"
                )
        elif cloud_deployment.compute_stack == ComputeStack.K8S:
            return KubernetesCloudDeploymentVerifier(self.log, self.api_client).verify(
                cloud_deployment
            )
        else:
            raise ValueError(
                f"Unsupported compute stack: {cloud_deployment.compute_stack}"
            )

    def verify_aws_cloud_resources_for_cloud_deployment(
        self,
        cloud_id: str,
        cloud_deployment: CloudDeployment,
        strict: bool = False,
        _use_strict_iam_permissions: bool = False,  # This should only be used in testing.
        boto3_session: Optional[boto3.Session] = None,
        logger: CloudSetupLogger = None,
    ) -> bool:
        assert cloud_deployment.region
        assert cloud_deployment.aws_config
        aws_config = cloud_deployment.aws_config
        file_storage = cloud_deployment.file_storage
        object_storage = cloud_deployment.object_storage

        # Convert dict to ObjectStorage object if needed
        if object_storage is not None and isinstance(object_storage, dict):
            object_storage = ObjectStorage(**object_storage)

        if boto3_session is None:
            boto3_session = boto3.Session(region_name=cloud_deployment.region)

        return self.verify_aws_cloud_resources(
            aws_vpc_id=aws_config.vpc_id,
            aws_subnet_ids=aws_config.subnet_ids or [],
            aws_control_plane_role=aws_config.anyscale_iam_role_id,
            aws_data_plane_role=aws_config.cluster_iam_role_id,
            aws_security_groups=aws_config.security_group_ids,
            aws_s3_id=object_storage.bucket_name[len(S3_STORAGE_PREFIX) :]
            if object_storage and object_storage.bucket_name
            else None,
            aws_efs_id=file_storage.file_storage_id if file_storage else None,
            aws_efs_mount_target_ip=file_storage.mount_targets[0].address
            if file_storage and file_storage.mount_targets
            else None,
            aws_cloudformation_stack_id=None,
            memorydb_cluster_config=self._get_memorydb_config_for_verification(
                aws_config, cloud_deployment.region
            ),
            boto3_session=boto3_session,
            region=cloud_deployment.region,
            cloud_id=cloud_id,
            is_bring_your_own_resource=True,
            is_private_network=cloud_deployment.networking_mode
            == NetworkingMode.PRIVATE,
            strict=strict,
            _use_strict_iam_permissions=_use_strict_iam_permissions,
            logger=logger,
        )

    def _get_memorydb_config_for_verification(
        self, aws_config, region: str
    ) -> Optional[AWSMemoryDBClusterConfig]:
        """Get MemoryDB cluster config for verification, fetching endpoint from AWS if needed."""
        if not aws_config.memorydb_cluster_name:
            return None

        # If we already have the endpoint, use it
        if aws_config.memorydb_cluster_endpoint:
            return AWSMemoryDBClusterConfig(
                id=aws_config.memorydb_cluster_name,
                endpoint=aws_config.memorydb_cluster_endpoint,
            )

        # Otherwise, fetch it from AWS
        try:
            return _get_memorydb_cluster_config(
                aws_config.memorydb_cluster_name, region, self.log,
            )
        except Exception as e:  # noqa: BLE001
            self.log.warning(
                f"Could not fetch MemoryDB cluster config for {aws_config.memorydb_cluster_name}: {e}"
            )
            return None

    def verify_aws_cloud_resources_for_create_cloud_resource(  # noqa: PLR0913
        self,
        cloud_resource: CreateCloudResource,
        boto3_session: boto3.Session,
        region: str,
        is_private_network: bool,
        cloud_id: str,
        is_bring_your_own_resource: bool = False,
        ignore_capacity_errors: bool = IGNORE_CAPACITY_ERRORS,
        logger: CloudSetupLogger = None,
        strict: bool = False,
        _use_strict_iam_permissions: bool = False,  # This should only be used in testing.
    ) -> bool:
        subnet_ids = (
            [
                subnet_id_with_az.subnet_id
                for subnet_id_with_az in cloud_resource.aws_subnet_ids_with_availability_zones
            ]
            if cloud_resource.aws_subnet_ids_with_availability_zones
            else []
        )
        aws_control_plane_role = (
            cloud_resource.aws_iam_role_arns[0]
            if cloud_resource.aws_iam_role_arns
            else None
        )
        aws_data_plane_role = (
            cloud_resource.aws_iam_role_arns[1]
            if cloud_resource.aws_iam_role_arns
            and len(cloud_resource.aws_iam_role_arns) > 1
            else None
        )
        return self.verify_aws_cloud_resources(
            aws_vpc_id=cloud_resource.aws_vpc_id,
            aws_subnet_ids=subnet_ids,
            aws_control_plane_role=aws_control_plane_role,
            aws_data_plane_role=aws_data_plane_role,
            aws_security_groups=cloud_resource.aws_security_groups,
            aws_s3_id=cloud_resource.aws_s3_id,
            aws_efs_id=cloud_resource.aws_efs_id,
            aws_efs_mount_target_ip=cloud_resource.aws_efs_mount_target_ip,
            aws_cloudformation_stack_id=cloud_resource.aws_cloudformation_stack_id,
            memorydb_cluster_config=cloud_resource.memorydb_cluster_config,
            boto3_session=boto3_session,
            region=region,
            cloud_id=cloud_id,
            is_bring_your_own_resource=is_bring_your_own_resource,
            ignore_capacity_errors=ignore_capacity_errors,
            logger=logger,
            is_private_network=is_private_network,
            strict=strict,
            _use_strict_iam_permissions=_use_strict_iam_permissions,
        )

    def verify_aws_cloud_resources(  # noqa: PLR0913
        self,
        *,
        aws_vpc_id: Optional[str],
        aws_subnet_ids: List[str],
        aws_control_plane_role: Optional[str],
        aws_data_plane_role: Optional[str],
        aws_security_groups: Optional[List[str]],
        aws_s3_id: Optional[str],
        aws_efs_id: Optional[str],
        aws_efs_mount_target_ip: Optional[str],
        aws_cloudformation_stack_id: Optional[str],
        memorydb_cluster_config: Optional[AWSMemoryDBClusterConfig],
        boto3_session: boto3.Session,
        region: str,
        is_private_network: bool,
        cloud_id: str,
        is_bring_your_own_resource: bool = False,
        ignore_capacity_errors: bool = IGNORE_CAPACITY_ERRORS,
        logger: CloudSetupLogger = None,
        strict: bool = False,
        _use_strict_iam_permissions: bool = False,  # This should only be used in testing.
    ) -> bool:
        if not logger:
            logger = self.log

        verify_aws_vpc_result = verify_aws_vpc(
            aws_vpc_id=aws_vpc_id,
            boto3_session=boto3_session,
            logger=logger,
            ignore_capacity_errors=ignore_capacity_errors,
            strict=strict,
        )

        verify_aws_subnets_result = verify_aws_subnets(
            aws_vpc_id=aws_vpc_id,
            aws_subnet_ids=aws_subnet_ids,
            region=region,
            logger=logger,
            ignore_capacity_errors=ignore_capacity_errors,
            is_private_network=is_private_network,
            strict=strict,
        )

        anyscale_aws_account = (
            self.api_client.get_anyscale_aws_account_api_v2_clouds_anyscale_aws_account_get().result.anyscale_aws_account
        )

        verify_aws_iam_roles_result = verify_aws_iam_roles(
            control_plane_role=aws_control_plane_role,
            data_plane_role=aws_data_plane_role,
            boto3_session=boto3_session,
            anyscale_aws_account=anyscale_aws_account,
            logger=logger,
            strict=strict,
            cloud_id=cloud_id,
            _use_strict_iam_permissions=_use_strict_iam_permissions,
        )
        verify_aws_security_groups_result = verify_aws_security_groups(
            aws_security_group_ids=aws_security_groups,
            boto3_session=boto3_session,
            logger=logger,
            strict=strict,
        )
        verify_aws_s3_result = verify_aws_s3(
            aws_s3_id=aws_s3_id,
            control_plane_role=aws_control_plane_role,
            data_plane_role=aws_data_plane_role,
            boto3_session=boto3_session,
            region=region,
            logger=logger,
            strict=strict,
        )
        verify_aws_efs_result = True
        if aws_efs_id:
            verify_aws_efs_result = verify_aws_efs(
                aws_efs_id=aws_efs_id,
                aws_efs_mount_target_ips=[aws_efs_mount_target_ip]
                if aws_efs_mount_target_ip
                else [],
                aws_subnet_ids=aws_subnet_ids,
                aws_security_groups=aws_security_groups,
                boto3_session=boto3_session,
                logger=logger,
                strict=strict,
            )
        # Cloudformation is only used in managed cloud setup. Set to True in BYOR case because it's not used.
        verify_aws_cloudformation_stack_result = True
        if not is_bring_your_own_resource:
            verify_aws_cloudformation_stack_result = verify_aws_cloudformation_stack(
                aws_cloudformation_stack_id=aws_cloudformation_stack_id,
                boto3_session=boto3_session,
                logger=logger,
                strict=strict,
            )
        if memorydb_cluster_config is not None:
            assert aws_security_groups
            assert aws_vpc_id
            verify_aws_memorydb_cluster_result = verify_aws_memorydb_cluster(
                memorydb_cluster_config=memorydb_cluster_config,
                aws_security_groups=aws_security_groups,
                aws_vpc_id=aws_vpc_id,
                aws_subnet_ids=aws_subnet_ids,
                boto3_session=boto3_session,
                logger=logger,
                strict=strict,
            )

        verify_anyscale_access_result = verify_anyscale_access(
            self.api_client, cloud_id, CloudProviders.AWS, logger
        )

        verification_result_summary = [
            "Verification result:",
            f"anyscale access: {self._passed_or_failed_str_from_bool(verify_anyscale_access_result)}",
            f"vpc: {self._passed_or_failed_str_from_bool(verify_aws_vpc_result)}",
            f"subnets: {self._passed_or_failed_str_from_bool(verify_aws_subnets_result)}",
            f"iam roles: {self._passed_or_failed_str_from_bool(verify_aws_iam_roles_result)}",
            f"security groups: {self._passed_or_failed_str_from_bool(verify_aws_security_groups_result)}",
            f"s3: {self._passed_or_failed_str_from_bool(verify_aws_s3_result)}",
            f"cloudformation stack: {self._passed_or_failed_str_from_bool(verify_aws_cloudformation_stack_result) if not is_bring_your_own_resource else 'N/A'}",
        ]
        if aws_efs_id:
            verification_result_summary.append(
                f"efs: {self._passed_or_failed_str_from_bool(verify_aws_efs_result)}"
            )
        if memorydb_cluster_config is not None:
            verification_result_summary.append(
                f"memorydb cluster: {self._passed_or_failed_str_from_bool(verify_aws_memorydb_cluster_result)}"
            )

        logger.info("\n".join(verification_result_summary))

        self.verify_aws_cloud_quotas(region=region, boto3_session=boto3_session)

        return all(
            [
                verify_anyscale_access_result,
                verify_aws_vpc_result,
                verify_aws_subnets_result,
                verify_aws_iam_roles_result,
                verify_aws_security_groups_result,
                verify_aws_s3_result,
                verify_aws_efs_result,
                verify_aws_cloudformation_stack_result
                if not is_bring_your_own_resource
                else True,
            ]
        )

    def verify_aws_cloud_quotas(
        self, *, region: str, boto3_session: Optional[Any] = None
    ):
        """
        Checks the AWS EC2 instance quotas and warns users if they are not good enough
        to support LLM workloads
        """
        if boto3_session is None:
            boto3_session = boto3.Session(region_name=region)

        QUOTAS_CONFIG = {
            "L-3819A6DF": {
                "description": "All G and VT Spot Instance Requests",
                "min": 512,
            },
            "L-34B43A08": {
                "description": "All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests",
                "min": 512,
            },
            "L-7212CCBC": {
                "description": "All P4, P3 and P2 Spot Instance Requests",
                "min": 224,
            },
            "L-DB2E81BA": {
                "description": "Running On-Demand G and VT instances",
                "min": 512,
            },
            "L-1216C47A": {
                "description": "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances",
                "min": 544,
            },
            "L-417A185B": {"description": "Running On-Demand P instances", "min": 224,},
        }

        quota_client = boto3_session.client("service-quotas", region_name=region)

        self.log.info("Checking quota values...")
        # List of tuples of quota code, current quota value
        invalid_quotas = []
        for quota_code, config in QUOTAS_CONFIG.items():
            quota = quota_client.get_service_quota(
                ServiceCode="ec2", QuotaCode=quota_code
            )
            if quota["Quota"]["Value"] < config["min"]:
                invalid_quotas.append((quota_code, quota["Quota"]["Value"]))

        if invalid_quotas:
            quota_errors = [
                f"- \"{QUOTAS_CONFIG[quota_code]['description']}\" should be at least {QUOTAS_CONFIG[quota_code]['min']} (curr: {value})"
                for quota_code, value in invalid_quotas
            ]
            quota_error_str = "\n".join(quota_errors)
            self.log.warning(
                "Your AWS account does not have enough quota to support LLM workloads. "
                "Please request quota increases for the following quotas:\n"
                f"{quota_error_str}\n\nFor instructions on how to increase quotas, visit this link: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html#request-increase"
            )

    def _print_cloud_resource_verification_results(
        self, cloud_resource_results: List[Tuple[str, bool]]
    ) -> None:
        """Print verification results for multiple cloud resources"""
        self.log.info("=" * 60)
        self.log.info("CLOUD RESOURCE VERIFICATION RESULTS:")
        self.log.info("=" * 60)

        for cloud_resource_name, success in cloud_resource_results:
            status = "PASSED" if success else "FAILED"
            self.log.info(f"{cloud_resource_name}: {status}")

        self.log.info("=" * 60)

        passed_count = sum(1 for _, success in cloud_resource_results if success)
        total_count = len(cloud_resource_results)

        if passed_count == total_count:
            self.log.info(
                f"Overall Result: ALL {total_count} cloud resources verified successfully"
            )

    def register_azure_or_generic_cloud(
        self,
        name: str,
        provider: str,
        cloud_resource: CloudDeployment,
        auto_add_user: bool = False,
    ) -> None:
        cloud_provider = (
            CloudProviders.AZURE if provider == "azure" else CloudProviders.GENERIC
        )
        self.cloud_event_producer.init_trace_context(
            CloudAnalyticsEventCommandName.REGISTER, cloud_provider,
        )
        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.COMMAND_START, succeeded=True
        )

        # Attempt to create the cloud.
        try:
            created_cloud = self.api_client.create_cloud_api_v2_clouds_post(
                write_cloud=WriteCloud(
                    name=name,
                    region=cloud_resource.region,
                    provider=cloud_provider,
                    is_bring_your_own_resource=True,
                    cluster_management_stack_version=ClusterManagementStackVersions.V2,
                    auto_add_user=auto_add_user,
                    credentials="",
                )
            )
            cloud_id = created_cloud.result.id
            self.cloud_event_producer.set_cloud_id(cloud_id)
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.CLOUD_RECORD_INSERTED, succeeded=True
            )
        except ClickException as e:
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.CLOUD_RECORD_INSERTED,
                succeeded=False,
                internal_error=str(e),
            )
            raise

        # Attempt to create the cloud resource.
        try:
            with self.log.spinner("Registering Anyscale cloud resources..."):
                self.api_client.add_cloud_resource_api_v2_clouds_cloud_id_add_resource_put(
                    cloud_id=cloud_id, cloud_deployment=cloud_resource,
                )

            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.INFRA_SETUP_COMPLETE, succeeded=True
            )

        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.INFRA_SETUP_COMPLETE,
                succeeded=False,
                internal_error=str(e),
            )

            # Delete the cloud if registering the cloud fails
            self.api_client.delete_cloud_api_v2_clouds_cloud_id_delete(
                cloud_id=cloud_id
            )
            raise ClickException(f"Cloud registration failed! {e}")

        # TODO (shomilj): Fetch & optionally run the Helm installation here.

        # Get the cloud resource ID to pass to the helm command.
        cloud_resources = self.api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
            cloud_id=cloud_id,
        ).results
        assert (
            len(cloud_resources) == 1
        ), f"Expected 1 cloud resource, got {len(cloud_resources)}"
        cloud_resource_id = cloud_resources[0].cloud_resource_id

        # Use CLI token to helm command
        helm_command = self._generate_helm_upgrade_command(
            provider=provider,
            cloud_deployment_id=cloud_resource_id,
            region=cloud_resource.region
            if cloud_provider == CloudProviders.AZURE
            else None,
            operator_iam_identity=cloud_resource.kubernetes_config.anyscale_operator_iam_identity
            if cloud_provider == CloudProviders.AZURE
            and cloud_resource.kubernetes_config
            else None,
            anyscale_cli_token=None,  # TODO: use $ANYSCALE_CLI_TOKEN placeholder
        )

        self.log.info(
            f"Cloud registration complete! To install the Anyscale operator, run:\n\n{helm_command}"
        )

    def register_aws_cloud(  # noqa: C901, PLR0912
        self,
        *,
        name: str,
        cloud_resource: CloudDeployment,
        functional_verify: Optional[str] = None,
        cluster_management_stack_version: ClusterManagementStackVersions = ClusterManagementStackVersions.V2,
        yes: bool = False,
        skip_verifications: bool = False,
        auto_add_user: bool = False,
    ):
        functions_to_verify = self._validate_functional_verification_args(
            functional_verify
        )

        if not validate_aws_credentials(self.log):
            raise ClickException(
                "Cloud registration requires valid AWS credentials to be set locally. Learn more: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html"
            )

        assert cloud_resource.aws_config

        if not (
            cloud_resource.object_storage and cloud_resource.object_storage.bucket_name
        ):
            raise click.ClickException(
                "Cloud object storage is required for AWS cloud registration."
            )
        if cloud_resource.object_storage.bucket_name.startswith(S3_ARN_PREFIX):
            cloud_resource.object_storage.bucket_name = cloud_resource.object_storage.bucket_name[
                len(S3_ARN_PREFIX) :
            ]
        if not cloud_resource.object_storage.bucket_name.startswith(S3_STORAGE_PREFIX):
            cloud_resource.object_storage.bucket_name = (
                S3_STORAGE_PREFIX + cloud_resource.object_storage.bucket_name
            )

        self.cloud_event_producer.init_trace_context(
            CloudAnalyticsEventCommandName.REGISTER, CloudProviders.AWS
        )
        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.COMMAND_START, succeeded=True
        )

        if cloud_resource.compute_stack == ComputeStack.K8S:
            # On K8S, we don't need to collect credentials;
            # instead, write a random value into this field
            # to maintain the property that each cloud's
            # credentials are unique.
            credentials = uuid.uuid4().hex
        else:
            credentials = cloud_resource.aws_config.anyscale_iam_role_id

        # Create a cloud without cloud resources first
        try:
            created_cloud = self.api_client.create_cloud_api_v2_clouds_post(
                write_cloud=WriteCloud(
                    provider="AWS",
                    region=cloud_resource.region,
                    credentials=credentials,
                    name=name,
                    is_bring_your_own_resource=True,
                    is_private_cloud=cloud_resource.networking_mode
                    == NetworkingMode.PRIVATE,
                    cluster_management_stack_version=cluster_management_stack_version,
                    auto_add_user=auto_add_user,
                    external_id=cloud_resource.aws_config.external_id,
                )
            )
            cloud_id = created_cloud.result.id
            self.cloud_event_producer.set_cloud_id(cloud_id)
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.CLOUD_RECORD_INSERTED, succeeded=True
            )
        except ClickException as e:
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.CLOUD_RECORD_INSERTED,
                succeeded=False,
                internal_error=str(e),
            )
            raise

        try:
            role, iam_role_original_policy = self._preprocess_aws(
                cloud_id=cloud_id, deployment=cloud_resource
            )
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.PREPROCESS_COMPLETE, succeeded=True
            )
        except Exception as e:  # noqa: BLE001
            error = str(e)
            self.log.error(error)
            error_msg_for_event = str(e)
            if isinstance(e, NoCredentialsError):
                # If it is a credentials error, rewrite the error to be more clear
                error = "Unable to locate AWS credentials. Cloud registration requires valid AWS credentials to be set locally. Learn more: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html"
                error_msg_for_event = "Unable to locate AWS credentials."
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.PREPROCESS_COMPLETE,
                succeeded=False,
                logger=self.log,
                internal_error=error_msg_for_event,
            )
            # Delete the cloud if registering the cloud fails
            self.api_client.delete_cloud_api_v2_clouds_cloud_id_delete(
                cloud_id=cloud_id
            )
            try:
                if (
                    iam_role_original_policy is not None
                    and cloud_resource.aws_config.external_id is None
                ):
                    # Revert the assume policy back to the original policy
                    role.AssumeRolePolicy().update(  # type: ignore
                        PolicyDocument=json.dumps(iam_role_original_policy)
                    )
            except Exception as revert_error:  # noqa: BLE001
                raise ClickException(
                    f"Cloud registration failed! {error}. Failed to revert the trust policy back to the original policy. {revert_error}"
                )
            raise ClickException(f"Cloud registration failed! {error}")

        try:
            # Verify cloud resources meet our requirement
            # Verification is only performed for VM compute stack.
            # TODO (shomilj): Add verification to the K8S compute stack as well.
            if cloud_resource.compute_stack != ComputeStack.K8S:
                with self.log.spinner("Verifying cloud resources...") as spinner:
                    if (
                        not skip_verifications
                        and not self.verify_aws_cloud_resources_for_cloud_deployment(
                            cloud_id=cloud_id,
                            cloud_deployment=cloud_resource,
                            logger=CloudSetupLogger(spinner_manager=spinner),
                        )
                    ):
                        raise ClickException(
                            "Please make sure all the resources provided meet the requirements and try again."
                        )

                confirm(
                    "Please review the output from verification for any warnings. Would you like to proceed with cloud creation?",
                    yes,
                )
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.RESOURCES_VERIFIED, succeeded=True
                )
        # Since the verify runs for a while, it's possible the user aborts the process, which throws a KeyboardInterrupt.
        except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001
            self.log.error(str(e))
            internal_error = str(e)
            if isinstance(e, Abort):
                internal_error = "User aborted."
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.RESOURCES_VERIFIED,
                succeeded=False,
                logger=self.log,
                internal_error=internal_error,
            )
            # Delete the cloud if registering the cloud fails
            self.api_client.delete_cloud_api_v2_clouds_cloud_id_delete(
                cloud_id=cloud_id
            )
            try:
                if (
                    iam_role_original_policy is not None
                    and cloud_resource.aws_config.external_id is None
                ):
                    # Revert the assume policy back to the original policy
                    role.AssumeRolePolicy().update(  # type: ignore
                        PolicyDocument=json.dumps(iam_role_original_policy)
                    )
            except Exception as revert_error:  # noqa: BLE001
                raise ClickException(
                    f"Cloud registration failed! {e}. Failed to revert the trust policy back to the original policy. {revert_error}"
                )

            raise ClickException(f"Cloud registration failed! {e}")

        try:
            with self.log.spinner(
                "Updating Anyscale cloud with cloud resource..."
            ) as spinner:
                # Update cloud with verified cloud resources.
                self.api_client.add_cloud_resource_api_v2_clouds_cloud_id_add_resource_put(
                    cloud_id=cloud_id, cloud_deployment=cloud_resource,
                )
            # For now, only wait for the cloud to be active if the compute stack is VM.
            # TODO (shomilj): support this fully for Kubernetes after provider metadata
            # checks are removed.
            if cloud_resource.compute_stack == ComputeStack.K8S:
                # Get the cloud resource ID to pass to the helm command.
                cloud_resources = self.api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
                    cloud_id=cloud_id,
                ).results
                assert (
                    len(cloud_resources) == 1
                ), f"Expected 1 cloud resource, got {len(cloud_resources)}"
                cloud_resource_id = cloud_resources[0].cloud_resource_id

                helm_command = self._generate_helm_upgrade_command(
                    provider="aws",
                    cloud_deployment_id=cloud_resource_id,
                    region=cloud_resource.region,
                )
                self.log.info(
                    f"Cloud registration complete! To install the Anyscale operator, run:\n\n{helm_command}"
                )
            else:
                self.wait_for_cloud_to_be_active(cloud_id, CloudProviders.AWS)
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.INFRA_SETUP_COMPLETE, succeeded=True
            )
        # Since the update runs for a while, it's possible the user aborts the process, which throws a KeyboardInterrupt.
        except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001
            # Delete the cloud if registering the cloud fails
            self.log.error(str(e))
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.INFRA_SETUP_COMPLETE,
                succeeded=False,
                internal_error=str(e),
            )
            self.api_client.delete_cloud_api_v2_clouds_cloud_id_delete(
                cloud_id=cloud_id
            )
            try:
                if (
                    iam_role_original_policy is not None
                    and cloud_resource.aws_config.external_id is None
                ):
                    # Revert the assume policy back to the original policy
                    role.AssumeRolePolicy().update(  # type: ignore
                        PolicyDocument=json.dumps(iam_role_original_policy)
                    )
            except Exception as revert_error:  # noqa: BLE001
                raise ClickException(
                    f"Cloud registration failed! {e}. Failed to revert the trust policy back to the original policy. {revert_error}"
                )

            raise ClickException(f"Cloud registration failed! {e}")

        self.log.info(f"Successfully created cloud {name}, and it's ready to use.")

        if len(functions_to_verify) > 0:
            CloudFunctionalVerificationController(
                self.cloud_event_producer, self.log
            ).start_verification(cloud_id, CloudProviders.AWS, functions_to_verify, yes)

    def verify_gcp_cloud_resources_from_cloud_deployment(
        self,
        cloud_id: str,
        cloud_deployment: CloudDeployment,
        strict: bool = False,
        yes: bool = False,
        is_private_service_cloud: bool = False,
        ignore_capacity_errors: bool = IGNORE_CAPACITY_ERRORS,
    ) -> bool:
        assert cloud_deployment.region
        assert cloud_deployment.gcp_config
        gcp_config = cloud_deployment.gcp_config
        file_storage = cloud_deployment.file_storage
        object_storage = cloud_deployment.object_storage

        if object_storage is not None and isinstance(object_storage, dict):
            object_storage = ObjectStorage(**object_storage)
        return self.verify_gcp_cloud_resources(
            project_id=gcp_config.project_id,
            vpc_id=gcp_config.vpc_name,
            subnet_ids=gcp_config.subnet_names,
            firewall_policy_ids=gcp_config.firewall_policy_names,
            control_plane_service_account=gcp_config.anyscale_service_account_email,
            data_plane_service_account=gcp_config.cluster_service_account_email,
            cloud_storage_bucket_name=(
                object_storage.bucket_name[len(GCS_STORAGE_PREFIX) :]
                if object_storage and object_storage.bucket_name
                else None
            ),
            filestore_instance_name=file_storage.file_storage_id
            if file_storage
            else None,
            memorystore_instance_name=gcp_config.memorystore_instance_name,
            region=cloud_deployment.region,
            cloud_id=cloud_id,
            host_project_id=gcp_config.host_project_id,
            strict=strict,
            yes=yes,
            is_private_service_cloud=is_private_service_cloud,
            ignore_capacity_errors=ignore_capacity_errors,
        )

    def verify_gcp_cloud_resources_from_create_cloud_resource(  # noqa: PLR0913
        self,
        *,
        cloud_resource: CreateCloudResourceGCP,
        project_id: str,
        region: str,
        cloud_id: str,
        yes: bool,
        host_project_id: Optional[str] = None,
        factory: Any = None,
        strict: bool = False,
        is_private_service_cloud: bool = False,
        ignore_capacity_errors: bool = IGNORE_CAPACITY_ERRORS,
    ) -> bool:
        return self.verify_gcp_cloud_resources(
            project_id=project_id,
            vpc_id=cloud_resource.gcp_vpc_id,
            subnet_ids=cloud_resource.gcp_subnet_ids,
            firewall_policy_ids=cloud_resource.gcp_firewall_policy_ids,
            control_plane_service_account=cloud_resource.gcp_anyscale_iam_service_account_email,
            data_plane_service_account=cloud_resource.gcp_cluster_node_service_account_email,
            cloud_storage_bucket_name=cloud_resource.gcp_cloud_storage_bucket_id,
            filestore_instance_name=cloud_resource.gcp_filestore_config.instance_name
            if cloud_resource.gcp_filestore_config
            else None,
            memorystore_instance_name=cloud_resource.memorystore_instance_config.name
            if cloud_resource.memorystore_instance_config
            else None,
            region=region,
            cloud_id=cloud_id,
            yes=yes,
            host_project_id=host_project_id,
            factory=factory,
            strict=strict,
            is_private_service_cloud=is_private_service_cloud,
            ignore_capacity_errors=ignore_capacity_errors,
        )

    def verify_gcp_cloud_resources(  # noqa: PLR0913
        self,
        *,
        project_id: str,
        vpc_id: Optional[str],
        subnet_ids: Optional[List[str]],
        firewall_policy_ids: Optional[List[str]],
        control_plane_service_account: Optional[str],
        data_plane_service_account: Optional[str],
        cloud_storage_bucket_name: Optional[str],
        filestore_instance_name: Optional[str],
        memorystore_instance_name: Optional[str],
        region: str,
        cloud_id: str,
        yes: bool = False,
        host_project_id: Optional[str] = None,
        factory: Any = None,
        strict: bool = False,
        is_private_service_cloud: bool = False,
        ignore_capacity_errors: bool = IGNORE_CAPACITY_ERRORS,
    ) -> bool:
        gcp_utils = try_import_gcp_utils()
        if not factory:
            factory = gcp_utils.get_google_cloud_client_factory(self.log, project_id)
        GCPLogger = gcp_utils.GCPLogger
        verify_lib = try_import_gcp_verify_lib()

        network_project_id = host_project_id if host_project_id else project_id
        use_shared_vpc = bool(host_project_id)

        with self.log.spinner("Verifying cloud resources...") as spinner:
            gcp_logger = GCPLogger(self.log, project_id, spinner, yes)
            verify_gcp_project_result = verify_lib.verify_gcp_project(
                factory=factory,
                project_id=project_id,
                enable_memorystore_api=memorystore_instance_name is not None,
                logger=gcp_logger,
                strict=strict,
            )
            verify_gcp_access_service_account_result = verify_lib.verify_gcp_access_service_account(
                factory=factory,
                anyscale_access_service_account=control_plane_service_account,
                project_id=project_id,
                logger=gcp_logger,
            )
            verify_gcp_dataplane_service_account_result = verify_lib.verify_gcp_dataplane_service_account(
                factory=factory,
                service_account=data_plane_service_account,
                project_id=project_id,
                logger=gcp_logger,
                strict=strict,
            )
            verify_gcp_networking_result = verify_lib.verify_gcp_networking(
                factory=factory,
                vpc_name=vpc_id,
                subnet_ids=subnet_ids,
                project_id=network_project_id,
                cloud_region=region,
                logger=gcp_logger,
                strict=strict,
                is_private_service_cloud=is_private_service_cloud,
                ignore_capacity_errors=ignore_capacity_errors,
            )
            verify_firewall_policy_result = verify_lib.verify_firewall_policy(
                factory=factory,
                firewall_policy_ids=firewall_policy_ids,
                vpc_name=vpc_id,
                subnet_ids=subnet_ids,
                project_id=network_project_id,
                cloud_region=region,
                use_shared_vpc=use_shared_vpc,
                is_private_service_cloud=is_private_service_cloud,
                logger=gcp_logger,
                strict=strict,
            )
            verify_cloud_storage_result = verify_lib.verify_cloud_storage(
                factory=factory,
                bucket_name=cloud_storage_bucket_name,
                controlplane_service_account=control_plane_service_account,
                dataplane_service_account=data_plane_service_account,
                project_id=project_id,
                cloud_region=region,
                logger=gcp_logger,
                strict=strict,
            )
            verify_anyscale_access_result = verify_anyscale_access(
                self.api_client, cloud_id, CloudProviders.GCP, self.log
            )
            verify_filestore_result = True
            if filestore_instance_name:
                verify_filestore_result = verify_lib.verify_filestore(
                    factory=factory,
                    file_store_instance_name=filestore_instance_name,
                    vpc_name=vpc_id,
                    cloud_region=region,
                    logger=gcp_logger,
                    strict=strict,
                )
            verify_memorystore_result = True
            if memorystore_instance_name:
                verify_memorystore_result = verify_lib.verify_memorystore(
                    factory=factory,
                    redis_instance_name=memorystore_instance_name,
                    cloud_vpc_name=vpc_id,
                    logger=gcp_logger,
                    strict=strict,
                )

        verification_results = [
            "Verification result:",
            f"anyscale access: {self._passed_or_failed_str_from_bool(verify_anyscale_access_result)}",
            f"project: {self._passed_or_failed_str_from_bool(verify_gcp_project_result)}",
            f"vpc and subnet: {self._passed_or_failed_str_from_bool(verify_gcp_networking_result)}",
            f"anyscale iam service account: {self._passed_or_failed_str_from_bool(verify_gcp_access_service_account_result)}",
            f"cluster node service account: {self._passed_or_failed_str_from_bool(verify_gcp_dataplane_service_account_result)}",
            f"firewall policy: {self._passed_or_failed_str_from_bool(verify_firewall_policy_result)}",
            f"cloud storage: {self._passed_or_failed_str_from_bool(verify_cloud_storage_result)}",
        ]

        if filestore_instance_name:
            verification_results.append(
                f"filestore: {self._passed_or_failed_str_from_bool(verify_filestore_result)}"
            )
        if memorystore_instance_name:
            verification_results.append(
                f"memorystore: {self._passed_or_failed_str_from_bool(verify_memorystore_result)}"
            )

        self.log.info("\n".join(verification_results))

        return all(
            [
                verify_anyscale_access_result,
                verify_gcp_project_result,
                verify_gcp_access_service_account_result,
                verify_gcp_dataplane_service_account_result,
                verify_gcp_networking_result,
                verify_firewall_policy_result,
                verify_filestore_result,
                verify_cloud_storage_result,
                verify_memorystore_result,
            ]
        )

    def _validate_gcp_config(self, compute_stack: ComputeStack, gcp_config: GCPConfig):
        if gcp_config.project_id and gcp_config.project_id[0].isdigit():
            # project ID should start with a letter
            raise click.ClickException(
                "Please provide a valid project ID. Note that project ID is not project number, see https://cloud.google.com/resource-manager/docs/creating-managing-projects#before_you_begin for details."
            )

        if (
            compute_stack != ComputeStack.K8S
            and re.search(
                "projects\\/[0-9]*\\/locations\\/global\\/workloadIdentityPools\\/.+\\/providers\\/[a-z0-9-]*$",
                gcp_config.provider_name,
            )
            is None
        ):
            raise click.ClickException(
                "Please provide a valid, fully qualified provider name. Only lowercase letters, numbers, and dashes are allowed. Example: projects/<project number>/locations/global/workloadIdentityPools/<pool name>/providers/<provider id>"
            )

        if (
            gcp_config.memorystore_instance_name is not None
            and re.search(
                "projects/.+/locations/.+/instances/.+",
                gcp_config.memorystore_instance_name,
            )
            is None
        ):
            raise click.ClickException(
                "Please provide a valid memorystore instance name. Example: projects/<project number>/locations/<location>/instances/<instance id>"
            )

        if (
            gcp_config.host_project_id is not None
            and gcp_config.host_project_id[0].isdigit()
        ):
            # project ID should start with a letter
            raise click.ClickException(
                "Please provide a valid host project ID. Note that project ID is not project number, see https://cloud.google.com/resource-manager/docs/creating-managing-projects#before_you_begin for details."
            )

    def register_gcp_cloud(  # noqa: C901, PLR0912
        self,
        *,
        name: str,
        cloud_resource: CloudDeployment,
        functional_verify: Optional[str] = None,
        cluster_management_stack_version: ClusterManagementStackVersions = ClusterManagementStackVersions.V2,
        yes: bool = False,
        skip_verifications: bool = False,
        auto_add_user: bool = False,
    ):
        functions_to_verify = self._validate_functional_verification_args(
            functional_verify
        )

        assert cloud_resource.compute_stack
        assert cloud_resource.gcp_config
        self._validate_gcp_config(
            cloud_resource.compute_stack, cloud_resource.gcp_config
        )

        if not (
            cloud_resource.object_storage and cloud_resource.object_storage.bucket_name
        ):
            raise click.ClickException(
                "Cloud object storage is required for GCP cloud registration."
            )
        if not cloud_resource.object_storage.bucket_name.startswith(GCS_STORAGE_PREFIX):
            cloud_resource.object_storage.bucket_name = (
                GCS_STORAGE_PREFIX + cloud_resource.object_storage.bucket_name
            )

        self.cloud_event_producer.init_trace_context(
            CloudAnalyticsEventCommandName.REGISTER, CloudProviders.GCP
        )
        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.COMMAND_START, succeeded=True
        )

        try:
            if cloud_resource.compute_stack == ComputeStack.K8S:
                # On K8S, we don't need to collect credentials;
                # instead, write a random trace_id into this field
                # to maintain the property that each cloud's
                # credentials are unique.
                random_id = uuid.uuid4().hex
                credentials = json.dumps(
                    {
                        "provider_id": "",
                        "project_id": random_id,
                        "service_account_email": random_id,
                    }
                )
            else:
                credentials_dict = {
                    "project_id": cloud_resource.gcp_config.project_id or "",
                    "provider_id": cloud_resource.gcp_config.provider_name or "",
                    "service_account_email": cloud_resource.gcp_config.anyscale_service_account_email
                    or "",
                }
                if cloud_resource.gcp_config.host_project_id:
                    credentials_dict[
                        "host_project_id"
                    ] = cloud_resource.gcp_config.host_project_id
                credentials = json.dumps(credentials_dict)

            # NOTE: For now we set the is_private_service_cloud to be the same as is_private_cloud
            # We don't expose this to the user yet since it's not recommended.
            is_private_network = (
                cloud_resource.networking_mode == NetworkingMode.PRIVATE
            )
            is_private_service_cloud = is_private_network

            created_cloud = self.api_client.create_cloud_api_v2_clouds_post(
                write_cloud=WriteCloud(
                    provider="GCP",
                    region=cloud_resource.region,
                    credentials=credentials,
                    name=name,
                    is_bring_your_own_resource=True,
                    is_private_cloud=is_private_network,
                    cluster_management_stack_version=cluster_management_stack_version,
                    is_private_service_cloud=is_private_service_cloud,
                    auto_add_user=auto_add_user,
                )
            )
            cloud_id = created_cloud.result.id
            self.cloud_event_producer.set_cloud_id(cloud_id)
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.CLOUD_RECORD_INSERTED, succeeded=True
            )
        except ClickException as e:
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.CLOUD_RECORD_INSERTED,
                succeeded=False,
                internal_error=str(e),
            )
            raise

        try:
            file_storage = cloud_resource.file_storage
            # Check if filestore is enabled
            enable_filestore = file_storage and file_storage.file_storage_id

            # Normally, for Kubernetes clouds, we don't need a VPC name, since networking is managed by Kubernetes.
            # For Kubernetes clouds on GCP where Filestore is enabled, we require the VPC name, since it is needed
            # to look up the relevant Mount Target IP for Filestore in the VPC.
            if cloud_resource.compute_stack == ComputeStack.K8S:
                if enable_filestore and not cloud_resource.gcp_config.vpc_name:
                    raise ClickException(
                        "Please provide the name of the VPC that your Kubernetes cluster is running inside of."
                    )
                memorystore_instance_name = (
                    cloud_resource.gcp_config.memorystore_instance_name
                )
                if (
                    enable_filestore or memorystore_instance_name
                ) and not cloud_resource.gcp_config.project_id:
                    raise ClickException("Please provide a project ID.")

            self._preprocess_gcp(cloud_resource)

            # Verification is only performed for VM compute stack.
            # TODO (shomilj): Add verification to the K8S compute stack as well.
            if cloud_resource.compute_stack != ComputeStack.K8S:
                if (
                    not skip_verifications
                    and not self.verify_gcp_cloud_resources_from_cloud_deployment(
                        cloud_id=cloud_id,
                        cloud_deployment=cloud_resource,
                        yes=yes,
                        is_private_service_cloud=is_private_service_cloud,
                    )
                ):
                    raise ClickException(
                        "Please make sure all the resources provided meet the requirements and try again."
                    )

                confirm(
                    "Please review the output from verification for any warnings. Would you like to proceed with cloud creation?",
                    yes,
                )
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.RESOURCES_VERIFIED, succeeded=True
                )
        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            internal_error = str(e)
            if isinstance(e, Abort):
                internal_error = "User aborted."
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.RESOURCES_VERIFIED,
                succeeded=False,
                logger=self.log,
                internal_error=internal_error,
            )

            # Delete the cloud if registering the cloud fails
            self.api_client.delete_cloud_api_v2_clouds_cloud_id_delete(
                cloud_id=cloud_id
            )
            raise ClickException(f"Cloud registration failed! {e}")

        try:
            with self.log.spinner("Updating Anyscale cloud with cloud resources..."):
                # Update cloud with verified cloud resources.
                self.api_client.add_cloud_resource_api_v2_clouds_cloud_id_add_resource_put(
                    cloud_id=cloud_id, cloud_deployment=cloud_resource,
                )
            # For now, only wait for the cloud to be active if the compute stack is VM.
            # TODO (shomilj): support this fully for Kubernetes after provider metadata
            # checks are removed.
            if cloud_resource.compute_stack == ComputeStack.K8S:
                # Get the cloud resource ID to pass to the helm command.
                cloud_resources = self.api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
                    cloud_id=cloud_id,
                ).results
                assert (
                    len(cloud_resources) == 1
                ), f"Expected 1 cloud resource, got {len(cloud_resources)}"
                cloud_resource_id = cloud_resources[0].cloud_resource_id

                helm_command = self._generate_helm_upgrade_command(
                    provider="gcp",
                    cloud_deployment_id=cloud_resource_id,
                    region=cloud_resource.region,
                    operator_iam_identity=cloud_resource.gcp_config.anyscale_service_account_email,
                )
                gcloud_command = self._generate_gcp_workload_identity_command(
                    anyscale_service_account_email=cloud_resource.gcp_config.anyscale_service_account_email,
                    project_id=cloud_resource.gcp_config.project_id,
                    namespace="<namespace>",
                )
                self.log.info(
                    f"Cloud registration complete! To install the Anyscale operator, run:\n\n{helm_command}\n\nThen configure workload identity by running:\n\n{gcloud_command}"
                )
            else:
                self.wait_for_cloud_to_be_active(cloud_id, CloudProviders.GCP)

            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.INFRA_SETUP_COMPLETE, succeeded=True
            )

        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            self.cloud_event_producer.produce(
                CloudAnalyticsEventName.INFRA_SETUP_COMPLETE,
                succeeded=False,
                internal_error=str(e),
            )
            # Delete the cloud if registering the cloud fails
            self.api_client.delete_cloud_api_v2_clouds_cloud_id_delete(
                cloud_id=cloud_id
            )
            raise ClickException(f"Cloud registration failed! {e}")

        self.log.info(f"Successfully created cloud {name}, and it's ready to use.")

        if len(functions_to_verify) > 0:
            CloudFunctionalVerificationController(
                self.cloud_event_producer, self.log
            ).start_verification(cloud_id, CloudProviders.GCP, functions_to_verify, yes)

    def delete_cloud(  # noqa: PLR0912, C901
        self,
        cloud_name: Optional[str],
        cloud_id: Optional[str],
        skip_confirmation: bool,
    ) -> bool:
        """
        Deletes a cloud by name or id.
        TODO Delete all GCE resources on cloud delete
        Including: Anyscale maanged resources, ALB resources, and TLS certs
        """
        cloud_id, cloud_name = get_cloud_id_and_name(
            self.api_client, cloud_id, cloud_name
        )

        # get cloud
        cloud: Cloud = self.api_client.get_cloud_api_v2_clouds_cloud_id_get(
            cloud_id=cloud_id
        ).result

        cloud_provider = cloud.provider
        assert cloud_provider in (
            CloudProviders.AWS,
            CloudProviders.GCP,
            CloudProviders.AZURE,
            CloudProviders.GENERIC,
        ), f"Cloud provider {cloud_provider} not supported yet"

        # Get cloud resources using the new API
        cloud_resources = self.api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
            cloud_id=cloud_id,
        ).results

        if not cloud_resources:
            # No cloud resources found, directly delete the cloud
            try:
                self.api_client.delete_cloud_api_v2_clouds_cloud_id_delete(
                    cloud_id=cloud_id
                )
            except ClickException as e:
                self.log.error(e)
                raise ClickException(f"Failed to delete cloud with name {cloud_name}.")

            self.log.info(f"Deleted cloud with name {cloud_name}.")
            return True

        confirmation_msg = f"\nIf the cloud {cloud_id} is deleted, you will not be able to access existing clusters of this cloud.\n"
        if cloud.is_bring_your_own_resource:
            confirmation_msg += "Note that Anyscale does not delete any of the cloud provider resources created by you.\n"

        confirmation_msg += "For more information, refer to the documentation "
        if cloud_provider == CloudProviders.AWS:
            confirmation_msg += "https://docs.anyscale.com/administration/cloud-deployment/manage-aws-cloud#delete-an-anyscale-cloud\n"
        elif cloud_provider == CloudProviders.GCP:
            confirmation_msg += "https://docs.anyscale.com/administration/cloud-deployment/manage-gcp-cloud#delete-an-anyscale-cloud\n"

        confirmation_msg += "Continue?"
        confirm(confirmation_msg, skip_confirmation)

        # set cloud state to DELETING
        try:
            with self.log.spinner("Preparing to delete Anyscale cloud..."):
                self.api_client.update_cloud_state_api_v2_clouds_cloud_id_state_put(
                    cloud_id=cloud_id, state=CloudState.DELETING,
                )

            # Get the default cloud resource for cleanup
            primary_cloud_resource = cloud_resources[0]

            if cloud_provider == CloudProviders.AWS:
                if not (cloud.is_aioa or cloud.compute_stack == ComputeStack.K8S):
                    # Delete services resources
                    self.delete_aws_lb_cfn_stack(cloud=cloud)
                    with self.log.spinner("Deleting load balancing resources..."):
                        wait_for_lb_resource_termination(
                            api_client=self.api_client, cloud_id=cloud_id
                        )
                    self.delete_aws_tls_certificates(cloud=cloud)

                self.delete_all_aws_resources(cloud, primary_cloud_resource)
            elif cloud_provider == CloudProviders.GCP:
                with self.log.spinner("Deleting load balancing resources..."):
                    wait_for_lb_resource_termination(
                        api_client=self.api_client, cloud_id=cloud_id
                    )
                self.delete_all_gcp_resources(cloud, primary_cloud_resource)
        except Exception as e:  # noqa: BLE001
            confirm(
                f"Error while trying to clean up {cloud_provider} resources:\n{e}\n"
                f"Please check your {cloud_provider} account for relevant errors.\n"
                "Do you want to force delete this cloud? You will need to clean up any associated resources on your own.\n"
                "Continue with force deletion?",
                skip_confirmation,
            )

        # Tear down admin zone and mark cloud as deleted
        with self.log.spinner("Deleting Anyscale cloud (this may take 2-5 minutes)..."):
            try:
                self.api_client.delete_cloud_api_v2_clouds_cloud_id_delete(
                    cloud_id=cloud_id
                )
            except ClickException as e:
                self.log.error(e)
                raise ClickException(f"Failed to delete cloud with name {cloud_name}.")

        self.log.info(f"Deleted cloud with name {cloud_name}.")
        return True

    def delete_all_gcp_resources(
        self, cloud: Cloud, cloud_resource: Optional[DecoratedCloudResource]
    ):
        if cloud.is_aioa or cloud.compute_stack == ComputeStack.K8S:
            # No resources to delete for hosted and k8s clouds
            return True

        if not cloud_resource or not cloud_resource.gcp_config:
            raise ClickException(
                f"This cloud {cloud.id} does not have GCP cloud resource configuration."
            )

        setup_utils = try_import_gcp_managed_setup_utils()
        gcp_utils = try_import_gcp_utils()
        gcp_config = cloud_resource.gcp_config
        provider = gcp_config.provider_name
        service_account = gcp_config.anyscale_service_account_email
        project_id = gcp_config.project_id
        factory = gcp_utils.get_google_cloud_client_factory(self.log, project_id)

        # Delete services resources
        setup_utils.delete_gcp_tls_certificates(factory, project_id, cloud.id)

        # Clean up cloud resources
        if cloud.is_bring_your_own_resource is False:  # managed cloud
            self.delete_gcp_managed_cloud(cloud=cloud, cloud_resource=cloud_resource)
        else:
            self.log.warning(
                f"The workload identity federation provider pool {provider} and service account {service_account} that allows Anyscale to access your GCP account is still in place. Please delete it manually if you no longer wish anyscale to have access."
            )

        return True

    def delete_all_aws_resources(
        self, cloud: Cloud, cloud_resource: Optional[DecoratedCloudResource]
    ) -> bool:
        if cloud.is_aioa or cloud.compute_stack == ComputeStack.K8S:
            # No resources to delete for hosted and k8s clouds
            return True

        # Clean up cloud resources
        if cloud.is_bring_your_own_resource is False:  # managed cloud
            # Delete AWS cloud resources by deleting the cfn stack
            self.delete_aws_managed_cloud(cloud=cloud, cloud_resource=cloud_resource)
        else:  # registered cloud
            self.log.warning(
                f"The trust policy that allows Anyscale to assume {cloud.credentials} is still in place. Please delete it manually if you no longer wish anyscale to have access."
            )

        return True

    def delete_aws_lb_cfn_stack(self, cloud: Cloud) -> bool:
        tag_name = "anyscale-cloud-id"
        key_name = cloud.id

        cfn_client = _client("cloudformation", cloud.region)
        # Get a list of all the CloudFormation stacks with the specified tag
        stacks = cfn_client.list_stacks()

        stacks = _unroll_resources_for_aws_list_call(
            cfn_client.list_stacks, "StackSummaries"
        )

        resources_to_cleanup = []

        for stack in stacks:
            if "StackName" in stack and "StackId" in stack:
                cfn_stack_arn = stack["StackId"]
                stack_description = cfn_client.describe_stacks(StackName=cfn_stack_arn)

                # Extract the tags from the response
                if (
                    "Stacks" in stack_description
                    and stack_description["Stacks"]
                    and "Tags" in stack_description["Stacks"][0]
                ):
                    tags = stack_description["Stacks"][0]["Tags"]
                    for tag in tags:
                        if (
                            "Key" in tag
                            and tag["Key"] == tag_name
                            and "Value" in tag
                            and tag["Value"] == key_name
                        ):
                            resources_to_cleanup.append(cfn_stack_arn)

        resource_delete_status = []
        for cfn_stack_arn in resources_to_cleanup:
            resource_delete_status.append(
                self.delete_aws_cloudformation_stack(
                    cfn_stack_arn=cfn_stack_arn, cloud=cloud
                )
            )

        return all(resource_delete_status)

    def delete_aws_tls_certificates(self, cloud: Cloud) -> bool:
        tag_name = "anyscale-cloud-id"
        key_name = cloud.id

        acm = boto3.client("acm", cloud.region)

        certificates = _unroll_resources_for_aws_list_call(
            acm.list_certificates, "CertificateSummaryList"
        )

        matching_certs = []

        for certificate in certificates:
            if "CertificateArn" in certificate:
                certificate_arn = certificate["CertificateArn"]
                response = acm.list_tags_for_certificate(CertificateArn=certificate_arn)

                if "Tags" in response:
                    tags = response["Tags"]
                    for tag in tags:
                        if (
                            "Key" in tag
                            and tag["Key"] == tag_name
                            and "Value" in tag
                            and tag["Value"] == key_name
                        ):
                            matching_certs.append(certificate_arn)

        resource_delete_status = []
        for certificate_arn in matching_certs:
            resource_delete_status.append(self.delete_tls_cert(certificate_arn, cloud))
        return all(resource_delete_status)

    def delete_aws_managed_cloud(
        self, cloud: Cloud, cloud_resource: Optional[DecoratedCloudResource]
    ) -> bool:
        if (
            not cloud_resource
            or not cloud_resource.aws_config
            or not cloud_resource.aws_config.cloudformation_id
        ):
            raise ClickException(
                f"This cloud {cloud.id} does not have a cloudformation stack."
            )

        cfn_stack_arn = cloud_resource.aws_config.cloudformation_id

        # If the cloud is updated, the cross account IAM role might have an inline policy for customer drifts
        # We delete the inline policy first otherwise cfn stack deletion would fail
        try_delete_customer_drifts_policy(cloud=cloud)

        bucket_name = (
            cloud_resource.object_storage.bucket_name
            if cloud_resource.object_storage
            else None
        )
        if bucket_name:
            self.log.info(
                f"\nThe S3 bucket ({bucket_name}) associated with this cloud still exists."
                "\nIf you no longer need the data associated with this bucket, please delete it."
            )

        return self.delete_aws_cloudformation_stack(
            cfn_stack_arn=cfn_stack_arn, cloud=cloud
        )

    def delete_aws_cloudformation_stack(self, cfn_stack_arn: str, cloud: Cloud) -> bool:
        cfn_client = _client("cloudformation", cloud.region)

        cfn_stack_url = f"https://{cloud.region}.console.aws.amazon.com/cloudformation/home?region={cloud.region}#/stacks/stackinfo?stackId={cfn_stack_arn}"

        try:
            cfn_client.delete_stack(StackName=cfn_stack_arn)
        except ClientError:
            raise ClickException(
                f"Failed to delete cloudformation stack {cfn_stack_arn}.\nPlease view it at {cfn_stack_url}"
            ) from None

        self.log.info(f"\nTrack progress of cloudformation at {cfn_stack_url}")
        with self.log.spinner(
            f"Deleting cloud resource {cfn_stack_arn} through cloudformation..."
        ):
            end_time = time.time() + CLOUDFORMATION_TIMEOUT_SECONDS_LONG
            while time.time() < end_time:
                try:
                    cfn_stack = cfn_client.describe_stacks(StackName=cfn_stack_arn)[
                        "Stacks"
                    ][0]
                except ClientError as e:
                    raise ClickException(
                        f"Failed to fetch the cloudformation stack {cfn_stack_arn}. Please check you have the right AWS credentials and the cloudformation stack still exists. Error details: {e}"
                    ) from None

                if cfn_stack["StackStatus"] == "DELETE_COMPLETE":
                    self.log.info(
                        f"Cloudformation stack {cfn_stack['StackId']} is deleted."
                    )
                    break

                if cfn_stack["StackStatus"] in ("DELETE_FAILED"):
                    # Provide link to cloudformation
                    raise ClickException(
                        f"Failed to delete cloud resources. Please check your cloudformation stack for errors. {cfn_stack_url}"
                    )
                time.sleep(1)

            if time.time() > end_time:
                raise ClickException(
                    f"Timed out deleting AWS resources. Please check your cloudformation stack for errors. {cfn_stack['StackId']}"
                )

        return True

    def delete_tls_cert(self, certificate_arn: str, cloud: Cloud) -> bool:
        acm = boto3.client("acm", cloud.region)

        try:
            acm.delete_certificate(CertificateArn=certificate_arn)
        except ClientError as e:
            raise ClickException(
                f"Failed to delete TLS certificate {certificate_arn}: {e}"
            ) from None

        return True

    def delete_gcp_managed_cloud(
        self, cloud: Cloud, cloud_resource: Optional[DecoratedCloudResource]
    ) -> bool:
        if (
            not cloud_resource
            or not cloud_resource.gcp_config
            or not cloud_resource.gcp_config.deployment_manager_id
        ):
            raise ClickException(
                f"This cloud {cloud.id} does not have a deployment in GCP deployment manager."
            )
        setup_utils = try_import_gcp_managed_setup_utils()
        gcp_utils = try_import_gcp_utils()

        project_id = cloud_resource.gcp_config.project_id
        factory = gcp_utils.get_google_cloud_client_factory(self.log, project_id)
        deployment_name = cloud_resource.gcp_config.deployment_manager_id
        deployment_url = f"https://console.cloud.google.com/dm/deployments/details/{deployment_name}?project={project_id}"

        self.log.info(f"\nTrack progress of Deployment Manager at {deployment_url}")

        with self.log.spinner("Deleting cloud resources through Deployment Manager..."):
            # Remove firewall policies
            if cloud_resource.gcp_config.firewall_policy_names:
                for firewall_policy in cloud_resource.gcp_config.firewall_policy_names:
                    # try delete the associations
                    setup_utils.remove_firewall_policy_associations(
                        factory, project_id, firewall_policy
                    )

            # Delete the deployment
            setup_utils.update_deployment_with_bucket_only(
                factory, project_id, deployment_name
            )

        bucket_name = (
            cloud_resource.object_storage.bucket_name
            if cloud_resource.object_storage
            else None
        )
        if bucket_name:
            self.log.info(
                f"\nThe cloud bucket ({bucket_name}) associated with this cloud still exists."
                "\nIf you no longer need the data associated with this bucket, please delete it."
            )
        return True

    ### Edit cloud ###
    def _get_cloud_resource_value(self, cloud_resource: Any, resource_type: str) -> Any:
        # Special case -- memorydb_cluster_id
        if resource_type == "memorydb_cluster_id":
            memorydb_cluster_config = getattr(
                cloud_resource, "memorydb_cluster_config", None
            )
            if memorydb_cluster_config is None:
                return None
            else:
                # memorydb_cluster_config.id has format "arn:aws:memorydb:us-east-2:815664363732:cluster/cloud-edit-test".
                value = memorydb_cluster_config.id.split("/")[-1]
                return value

        # Special case -- memorystore_instance_name
        if resource_type == "memorystore_instance_name":
            memorystore_instance_config = getattr(
                cloud_resource, "memorystore_instance_config", None
            )
            if memorystore_instance_config is None:
                return None
            else:
                value = memorystore_instance_config.name
                return value

        # Normal cases.
        value = getattr(cloud_resource, resource_type, None)
        if value is None:
            self.log.warning(f"Old value of {resource_type} is None.")
            return None
        return value

    def _generate_edit_details_logs(
        self, cloud_resource: Any, edit_details: Dict[str, Optional[str]]
    ) -> List[str]:
        details_logs = []
        for resource_type, value in edit_details.items():
            if value:
                old_value = self._get_cloud_resource_value(
                    cloud_resource, resource_type
                )
                if old_value == value:
                    raise ClickException(
                        f"Specified resource is the same as existed resource -- {resource_type}: {value}"
                    )
                details_logs.append(f"{resource_type}: from {old_value} -> {value}")
        return details_logs

    def _generate_rollback_command(
        self, cloud_id: str, cloud_resource: Any, edit_details: Dict[str, Optional[str]]
    ) -> str:
        rollback_command = BASE_ROLLBACK_COMMAND.format(cloud_id=cloud_id)
        for resource_type, value in edit_details.items():
            if value:
                old_value = self._get_cloud_resource_value(
                    cloud_resource, resource_type
                )
                if old_value is not None:
                    # The resource type names are in CreateCloudResource (backend/server/api/product/models/clouds.py).
                    # The cli command names are in cloud_edit (frontend/cli/anyscale/commands:cloud_commands).
                    # The relationship between their names are cli_command_name = resource_type_name.replace("_", "-").
                    # e.g. cli_command_name: aws-s3-id & resource_type_name: aws_s3_id.
                    formatted_resource_type = resource_type.replace("_", "-")
                    rollback_command += f" --{formatted_resource_type}={old_value}"
        # If the only edit field is redis and it was None originally, rollback_command didn't appened any args, reset the rollback command to be empty.
        if rollback_command == BASE_ROLLBACK_COMMAND.format(cloud_id=cloud_id):
            rollback_command = ""
        return rollback_command

    def _edit_aws_cloud(  # noqa: PLR0912, PLR0913
        self,
        *,
        cloud_id: str,
        cloud_name: str,
        cloud: Any,
        cloud_resource: Any,
        aws_s3_id: Optional[str],
        aws_efs_id: Optional[str],
        aws_efs_mount_target_ip: Optional[str],
        memorydb_cluster_id: Optional[str],
        yes: bool = False,
    ):
        # Log edit details.
        self.log.open_block("EditDetail", "\nEdit details...")
        edit_details = {
            "aws_s3_id": aws_s3_id,
            "aws_efs_id": aws_efs_id,
            "aws_efs_mount_target_ip": aws_efs_mount_target_ip,
            "memorydb_cluster_id": memorydb_cluster_id,
        }
        details_logs = self._generate_edit_details_logs(cloud_resource, edit_details)
        self.log.info(
            self.log.highlight(
                "Cloud {} ({}) edit details: \n{}".format(
                    cloud_name, cloud_id, "; \n".join(details_logs)
                )
            )
        )
        self.log.close_block("EditDetail")

        try:
            boto3_session = boto3.Session(region_name=cloud.region)
            if aws_efs_id and not aws_efs_mount_target_ip:
                # Get the mount target IP for new aws_efs_ip (consistent with cloud register).
                aws_efs_mount_target_ip = _get_aws_efs_mount_target_ip(
                    boto3_session, aws_efs_id
                )
                if not aws_efs_mount_target_ip:
                    raise ClickException(
                        f"Failed to get the mount target IP for new aws_efs_ip {aws_efs_id}, please make sure the aws_efs_ip exists and it has mount targets."
                    )
        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            raise ClickException(
                f"Failed to get the mount target IP for new aws_efs_ip {aws_efs_id}, please make sure it has mount targets."
            )
        try:
            memorydb_cluster_config = None
            if memorydb_cluster_id:
                memorydb_cluster_config = _get_memorydb_cluster_config(
                    memorydb_cluster_id, cloud.region, self.log
                )
        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            raise ClickException(
                f"Failed to get the memorydb cluster config for new memorydb_cluster_id {memorydb_cluster_id}, please make sure it's created."
            )

        # Static Verify.
        self.log.open_block(
            "Verify", "Start cloud static verification on the specified resources..."
        )
        new_cloud_resource = copy.deepcopy(cloud_resource)
        if aws_s3_id:
            new_cloud_resource.aws_s3_id = aws_s3_id
        if aws_efs_id:
            new_cloud_resource.aws_efs_id = aws_efs_id
        if aws_efs_mount_target_ip:
            new_cloud_resource.aws_efs_mount_target_ip = aws_efs_mount_target_ip
        if memorydb_cluster_id:
            new_cloud_resource.memorydb_cluster_config = memorydb_cluster_config

        if not self.verify_aws_cloud_resources_for_create_cloud_resource(
            cloud_resource=new_cloud_resource,
            boto3_session=boto3_session,
            region=cloud.region,
            cloud_id=cloud_id,
            is_bring_your_own_resource=cloud.is_bring_your_own_resource,
            is_private_network=cloud.is_private_cloud
            if cloud.is_private_cloud
            else False,
        ):
            raise ClickException(
                "Cloud edit failed because resource failed verification. Please check the verification results above, fix them, and try again."
            )

        self.log.info(
            self.log.highlight(
                "Please make sure you checked the warnings from above verification results."
            )
        )
        self.log.close_block("Verify")

        self.log.open_block(
            "Reminder", "Please read the following reminder carefully..."
        )
        self.log.info(
            self.log.highlight(
                "If there are running workloads utilizing the old resources, you may want to retain them. Please note that this edit will not automatically remove any old resources. If you wish to delete them, you'll need to handle it."
            )
        )
        self.log.info(
            self.log.highlight(
                "The cloud resources we are going to edit {} ({}): \n{}".format(
                    cloud_name, cloud_id, "; \n".join(details_logs)
                )
            )
        )
        self.log.close_block("Reminder")

        confirm(
            "Are you sure you want to edit these cloud resource? ", yes,
        )

        # Execute edit cloud.
        self.log.open_block("Edit", "Start editing cloud...")
        try:
            self.api_client.edit_cloud_resource_api_v2_clouds_with_cloud_resource_router_cloud_id_patch(
                cloud_id=cloud_id,
                editable_cloud_resource=EditableCloudResource(
                    aws_s3_id=aws_s3_id,
                    aws_efs_id=aws_efs_id,
                    aws_efs_mount_target_ip=aws_efs_mount_target_ip,
                    memorydb_cluster_config=memorydb_cluster_config,
                ),
            )
        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            raise ClickException(
                "Edit cloud resource failed! The backend server might be under maintainance right now, please reach out to support or your SA for assistance."
            )

        # Hint customer rollback command.
        rollback_command = self._generate_rollback_command(
            cloud_id, cloud_resource, edit_details
        )
        self.log.info(
            self.log.highlight(
                f"Cloud {cloud_name}({cloud_id}) is successfully edited."
            )
        )
        if rollback_command:
            self.log.info(
                self.log.highlight(
                    f"If you want to revert the edit, you can edit it back to the original cloud with: `{rollback_command}`"
                )
            )
        self.log.close_block("Edit")

    def _get_project_id(self, cloud: Any, cloud_name: str, cloud_id: str) -> str:
        try:
            return json.loads(cloud.credentials)["project_id"]
        except Exception:  # noqa: BLE001
            raise ClickException(
                f"Failed to get project id for cloud {cloud_name}({cloud_id}). Please ensure the provided cloud_name/cloud_id exists."
            )

    def _get_host_project_id(
        self, cloud: Any, cloud_name: str, cloud_id: str
    ) -> Optional[str]:
        try:
            credentials = json.loads(cloud.credentials)
            return credentials.get("host_project_id")
        except Exception:  # noqa: BLE001
            raise ClickException(
                f"Failed to get host project id for cloud {cloud_name}({cloud_id}). Please ensure the provided cloud_name/cloud_id exists."
            )

    def _get_gcp_filestore_config(
        self,
        gcp_filestore_instance_id: str,
        gcp_filestore_location: str,
        project_id: str,
        cloud_resource: Any,
        gcp_utils,
    ) -> GCPFileStoreConfig:
        try:
            factory = gcp_utils.get_google_cloud_client_factory(self.log, project_id)
            return gcp_utils.get_gcp_filestore_config(
                factory,
                project_id,
                cloud_resource.gcp_vpc_id,
                gcp_filestore_location,
                gcp_filestore_instance_id,
                self.log,
            )
        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            raise ClickException(
                f"Failed to construct the gcp_filestore_config from project {project_id}, please double check the provided filestore_location {gcp_filestore_location} and filestore_instance_id {gcp_filestore_instance_id}."
            )

    def _get_gcp_edit_details(
        self,
        *,
        cloud_resource: Any,
        edit_details: Dict[str, Optional[str]],
        gcp_filestore_config: Optional[GCPFileStoreConfig],
        gcp_filestore_instance_id: Optional[str],
        gcp_filestore_location: Optional[str],
        gcp_utils,
    ) -> List[str]:
        details_logs = self._generate_edit_details_logs(cloud_resource, edit_details)
        if gcp_filestore_config:
            (
                old_filestore_location,
                old_filestore_instance_id,
            ) = gcp_utils.get_filestore_location_and_instance_id(
                cloud_resource.gcp_filestore_config
            )
            if (
                old_filestore_instance_id == gcp_filestore_instance_id
                and old_filestore_location == gcp_filestore_location
            ):
                raise ClickException(
                    f"Specified resource is the same as existed resource -- gcp_filestore_instance_id: {gcp_filestore_instance_id}; gcp_filestore_location: {gcp_filestore_location}"
                )
            details_logs.append(
                f"filestore_instance_id: from {old_filestore_instance_id} -> {gcp_filestore_instance_id}"
            )
            details_logs.append(
                f"filestore_location: from {old_filestore_location} -> {gcp_filestore_location}"
            )
        return details_logs

    def _generate_rollback_command_for_gcp(
        self,
        cloud_id: str,
        cloud_resource: Any,
        edit_details: Dict[str, Optional[str]],
        gcp_filestore_config: Optional[GCPFileStoreConfig],
        gcp_utils,
    ):
        rollback_cmd = self._generate_rollback_command(
            cloud_id, cloud_resource, edit_details
        )
        if gcp_filestore_config:
            (
                old_filestore_location,
                old_filestore_instance_id,
            ) = gcp_utils.get_filestore_location_and_instance_id(
                cloud_resource.gcp_filestore_config
            )
            rollback_cmd += f" --gcp-filestore-instance-id={old_filestore_instance_id}"
            rollback_cmd += f" --gcp-filestore-location={old_filestore_location}"
        rollback_cmd = rollback_cmd.replace(
            "gcp-cloud-storage-bucket-id", "gcp-cloud-storage-bucket-name"
        )
        return rollback_cmd

    def _get_azure_audience(self) -> str:
        """
        Get Azure audience from the control plane API.

        Returns:
            Azure audience string with /.default scope

        Raises:
            ClickException: If the Azure audience is not configured on the server
        """
        try:
            response = (
                self.api_client.get_azure_operator_audience_api_v2_clouds_anyscale_azure_operator_audience_get()
            )
            audience = response.result.azure_operator_audience
            if audience:
                return f"{audience}/.default"
            raise ClickException("Azure audience not configured for this environment.")
        except Exception as e:  # noqa: BLE001
            raise ClickException(f"Failed to get Azure audience from API: {e}")

    def _generate_helm_upgrade_command(
        self,
        provider: str,
        cloud_deployment_id: str,
        region: Optional[str] = None,
        operator_iam_identity: Optional[str] = None,
        anyscale_cli_token: Optional[str] = None,
    ) -> str:
        """
        Generate the helm upgrade command for installing the Anyscale operator.

        Args:
            provider: Cloud provider ('aws', 'gcp', 'azure', 'generic')
            cloud_deployment_id: The cloud deployment ID from registration
            region: Cloud region (optional for generic provider)
            operator_iam_identity: IAM identity for the operator (GCP service account email, Azure client ID)
            anyscale_cli_token: CLI token (required for Azure and generic providers)

        Returns:
            Formatted helm upgrade command string
        """
        command_parts = [
            "helm upgrade <release-name> anyscale/anyscale-operator",
            f"  --set-string global.cloudDeploymentId={cloud_deployment_id}",
            f"  --set-string global.cloudProvider={provider}",
        ]

        # Add region only for AWS (using global.aws.region)
        # Region field is deprecated for other providers
        if region and provider == "aws":
            command_parts.append(f"  --set-string global.aws.region={region}")

        # Add provider-specific parameters
        if provider == "gcp" and operator_iam_identity:
            command_parts.append(
                f"  --set-string global.auth.iamIdentity={operator_iam_identity}"
            )
        elif provider == "azure":
            if operator_iam_identity:
                command_parts.append(
                    "  --set-string global.auth.iamIdentity=<Azure Client ID>"
                )

            if anyscale_cli_token:
                command_parts.append(
                    f"  --set-string global.auth.anyscaleCliToken={anyscale_cli_token}"
                )
            else:
                # Require Azure audience for non-token authentication
                azure_audience = self._get_azure_audience()
                if azure_audience:
                    command_parts.append(
                        f"  --set-string global.auth.audience={azure_audience}"
                    )

        elif provider == "generic":
            if anyscale_cli_token:
                command_parts.append(
                    f"  --set-string global.auth.anyscaleCliToken={anyscale_cli_token}"
                )
            else:
                command_parts.append(
                    "  --set-string global.auth.anyscaleCliToken=$ANYSCALE_CLI_TOKEN"
                )

        # Add common parameters
        command_parts.extend(
            [
                "  --set-string workloads.serviceAccount.name=anyscale-operator",
                "  --namespace <namespace>",
                "  --create-namespace",
                "  --wait",
                "  -i",
            ]
        )

        return " \\\n".join(command_parts)

    def _generate_gcp_workload_identity_command(
        self,
        anyscale_service_account_email: str,
        project_id: str,
        namespace: str = "<namespace>",
    ) -> str:
        """
        Generate the gcloud command for setting up workload identity for GCP.

        Args:
            anyscale_service_account_email: The GCP service account email
            project_id: The GCP project ID
            namespace: The Kubernetes namespace (defaults to <namespace> placeholder)

        Returns:
            Formatted gcloud iam service-accounts add-iam-policy-binding command
        """
        return f"""gcloud iam service-accounts add-iam-policy-binding {anyscale_service_account_email} \\
    --role roles/iam.workloadIdentityUser \\
    --member "serviceAccount:{project_id}.svc.id.goog[{namespace}/anyscale-operator]" """

    def _edit_gcp_cloud(  # noqa: PLR0912, PLR0913
        self,
        *,
        cloud_id: str,
        cloud_name: str,
        cloud: Any,
        cloud_resource: Any,
        gcp_filestore_instance_id: Optional[str],
        gcp_filestore_location: Optional[str],
        gcp_cloud_storage_bucket_name: Optional[str],
        memorystore_instance_name: Optional[str],
        yes: bool = False,
    ):
        project_id = self._get_project_id(cloud, cloud_name, cloud_id)
        host_project_id = self._get_host_project_id(cloud, cloud_name, cloud_id)
        gcp_utils = try_import_gcp_utils()

        gcp_filestore_config = None
        if gcp_filestore_instance_id and gcp_filestore_location:
            gcp_filestore_config = self._get_gcp_filestore_config(
                gcp_filestore_instance_id,
                gcp_filestore_location,
                project_id,
                cloud_resource,
                gcp_utils,
            )

        memorystore_instance_config = None
        if memorystore_instance_name:
            factory = gcp_utils.get_google_cloud_client_factory(self.log, project_id)
            memorystore_instance_config = gcp_utils.get_gcp_memorystore_config(
                factory, memorystore_instance_name
            )

        # Log edit details.
        self.log.open_block("EditDetail", "\nEdit details...")
        edit_details = {
            "gcp_cloud_storage_bucket_id": gcp_cloud_storage_bucket_name,
            "memorystore_instance_name": memorystore_instance_name,
        }
        details_logs = self._get_gcp_edit_details(
            cloud_resource=cloud_resource,
            edit_details=edit_details,
            gcp_filestore_config=gcp_filestore_config,
            gcp_filestore_instance_id=gcp_filestore_instance_id,
            gcp_filestore_location=gcp_filestore_location,
            gcp_utils=gcp_utils,
        )
        self.log.info(
            self.log.highlight(
                "Cloud edit details {} ({}): \n{}".format(
                    cloud_name, cloud_id, "; \n".join(details_logs)
                )
            )
        )
        self.log.close_block("EditDetail")

        # Static Verify.
        self.log.open_block(
            "Verify", "Start cloud static verification on the specified resources..."
        )
        new_cloud_resource = copy.deepcopy(cloud_resource)
        if gcp_filestore_config:
            new_cloud_resource.gcp_filestore_config = gcp_filestore_config
        if gcp_cloud_storage_bucket_name:
            new_cloud_resource.gcp_cloud_storage_bucket_id = (
                gcp_cloud_storage_bucket_name
            )
        if memorystore_instance_config:
            new_cloud_resource.memorystore_instance_config = memorystore_instance_config
        if not self.verify_gcp_cloud_resources_from_create_cloud_resource(
            cloud_resource=new_cloud_resource,
            project_id=project_id,
            host_project_id=host_project_id,
            region=cloud.region,
            cloud_id=cloud_id,
            yes=False,
        ):
            raise ClickException(
                "Cloud edit failed because resource failed verification. Please check the verification results above, fix them, and try again."
            )

        self.log.info(
            self.log.highlight(
                "Please make sure you checked the warnings from above verification results."
            )
        )
        self.log.close_block("Verify")

        self.log.open_block(
            "Reminder", "Please read the following reminder carefully..."
        )
        self.log.info(
            self.log.highlight(
                "If there are running workloads utilizing the old resources, you may want to retain them. Please note that this edit will not automatically remove any old resources. If you wish to delete them, you'll need to handle it."
            )
        )
        self.log.info(
            self.log.highlight(
                "The cloud resources we are going to edit {} ({}): \n{}".format(
                    cloud_name, cloud_id, "; \n".join(details_logs)
                )
            )
        )
        self.log.close_block("Reminder")

        confirm(
            "Are you sure you want to edit these cloud resource? ", yes,
        )

        # Execute edit cloud.
        self.log.open_block("Edit", "Start editing cloud...")
        try:
            self.api_client.edit_cloud_resource_api_v2_clouds_with_cloud_resource_gcp_router_cloud_id_patch(
                cloud_id=cloud_id,
                editable_cloud_resource_gcp=EditableCloudResourceGCP(
                    gcp_filestore_config=gcp_filestore_config,
                    gcp_cloud_storage_bucket_id=gcp_cloud_storage_bucket_name,
                    memorystore_instance_config=memorystore_instance_config,
                ),
            )
        except Exception as e:  # noqa: BLE001
            self.log.error(str(e))
            raise ClickException(
                "Edit cloud resource failed! The backend server might be under maintainance right now, please reach out to support or your SA for assistance."
            )

        # Hint customer rollback command.
        rollback_command = self._generate_rollback_command_for_gcp(
            cloud_id, cloud_resource, edit_details, gcp_filestore_config, gcp_utils,
        )
        self.log.info(
            self.log.highlight(
                f"Cloud {cloud_name}({cloud_id}) is successfully edited."
            )
        )
        if rollback_command:
            self.log.info(
                self.log.highlight(
                    f"If you want to revert the edit, you can edit it back to the original cloud with: `{rollback_command}`"
                )
            )
        self.log.close_block("Edit")

    def edit_cloud(  # noqa: PLR0912,PLR0913
        self,
        *,
        cloud_name: Optional[str],
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
        yes: bool = False,
        auto_add_user: Optional[bool] = None,
    ):
        """Edit aws cloud.

        The editable fields are in EditableCloudResource (AWS) /EditableCloudResourceGCP (GCP).
        Steps:
        1. Log the edits (from old to new values).
        2. Static verify cloud resources with updated values.
        3. Prompt the customer for confirmation based on verification results.
        4. Update the cloud resource (calls backend API to modify the database).
        5. Conduct a functional verification, if specified.
        """
        functions_to_verify = self._validate_functional_verification_args(
            functional_verify
        )
        cloud_id, cloud_name = get_cloud_id_and_name(
            self.api_client, cloud_id, cloud_name
        )
        cloud = self.api_client.get_cloud_api_v2_clouds_cloud_id_get(cloud_id).result

        if not cloud.is_bring_your_own_resource:
            raise ClickException(
                f"Cloud {cloud_name}({cloud_id}) is not a cloud with customer defined resources, currently we don't support editing cloud resource values of managed clouds."
            )

        cloud_resource = get_cloud_resource_by_cloud_id(
            cloud_id, cloud.provider, self.api_client
        )
        if cloud_resource is None:
            raise ClickException(
                f"Cloud {cloud_name}({cloud_id}) does not contain resource records."
            )

        if auto_add_user is not None:
            self._update_auto_add_user_field(auto_add_user, cloud)

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
            ]
        ):
            if cloud.provider == "AWS":
                self.cloud_event_producer.init_trace_context(
                    CloudAnalyticsEventCommandName.EDIT,
                    CloudProviders.AWS,
                    cloud_id=cloud_id,
                )
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.COMMAND_START, succeeded=True
                )
                if (
                    not any(
                        [
                            aws_s3_id,
                            aws_efs_id,
                            aws_efs_mount_target_ip,
                            memorydb_cluster_id,
                        ]
                    )
                ) or any(
                    [
                        gcp_filestore_instance_id,
                        gcp_filestore_location,
                        gcp_cloud_storage_bucket_name,
                        memorystore_instance_name,
                    ]
                ):
                    self.cloud_event_producer.produce(
                        CloudAnalyticsEventName.RESOURCES_EDITED,
                        succeeded=False,
                        internal_error="specified resource and provider mismatch",
                    )
                    raise ClickException(
                        "Specified resource and provider mismatch -- the cloud's provider is AWS, please make sure all the resource you want to edit is for AWS as well."
                    )
                try:
                    self._edit_aws_cloud(
                        cloud_id=cloud_id,  # type: ignore
                        cloud_name=cloud_name,  # type: ignore
                        cloud=cloud,
                        cloud_resource=cloud_resource,
                        aws_s3_id=aws_s3_id,
                        aws_efs_id=aws_efs_id,
                        aws_efs_mount_target_ip=aws_efs_mount_target_ip,
                        memorydb_cluster_id=memorydb_cluster_id,
                        yes=yes,
                    )
                except Exception as e:  # noqa: BLE001
                    self.cloud_event_producer.produce(
                        CloudAnalyticsEventName.RESOURCES_EDITED,
                        succeeded=False,
                        logger=self.log,
                        internal_error=str(e),
                    )
                    raise e
            elif cloud.provider == "GCP":
                self.cloud_event_producer.init_trace_context(
                    CloudAnalyticsEventCommandName.EDIT,
                    CloudProviders.GCP,
                    cloud_id=cloud_id,
                )
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.COMMAND_START, succeeded=True
                )
                if (
                    not any(
                        [
                            gcp_filestore_instance_id,
                            gcp_filestore_location,
                            gcp_cloud_storage_bucket_name,
                            memorystore_instance_name,
                        ]
                    )
                ) or any(
                    [
                        aws_s3_id,
                        aws_efs_id,
                        aws_efs_mount_target_ip,
                        memorydb_cluster_id,
                    ]
                ):
                    self.cloud_event_producer.produce(
                        CloudAnalyticsEventName.RESOURCES_EDITED,
                        succeeded=False,
                        internal_error="specified resource and provider mismatch",
                    )
                    raise ClickException(
                        "Specified resource and provider mismatch -- the cloud's provider is GCP, please make sure all the resource you want to edit is for GCP as well."
                    )
                try:
                    self._edit_gcp_cloud(
                        cloud_id=cloud_id,  # type: ignore
                        cloud_name=cloud_name,  # type: ignore
                        cloud=cloud,
                        cloud_resource=cloud_resource,
                        gcp_filestore_instance_id=gcp_filestore_instance_id,
                        gcp_filestore_location=gcp_filestore_location,
                        gcp_cloud_storage_bucket_name=gcp_cloud_storage_bucket_name,
                        memorystore_instance_name=memorystore_instance_name,
                        yes=yes,
                    )
                except Exception as e:  # noqa: BLE001
                    self.cloud_event_producer.produce(
                        CloudAnalyticsEventName.RESOURCES_EDITED,
                        succeeded=False,
                        logger=self.log,
                        internal_error=str(e),
                    )
                    raise e
            else:
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.RESOURCES_EDITED,
                    succeeded=False,
                    internal_error="invalid cloud provider",
                )
                raise ClickException(
                    f"Unsupported cloud provider {cloud.provider} for cloud edit!"
                )

        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.RESOURCES_EDITED, succeeded=True
        )
        # Functional verify.
        if len(functions_to_verify) > 0:
            functional_verify_succeed = CloudFunctionalVerificationController(
                self.cloud_event_producer, self.log
            ).start_verification(
                cloud_id,
                self._get_cloud_provider_from_str(cloud.provider),
                functions_to_verify,
                yes,
            )
            if not functional_verify_succeed:
                raise ClickException(
                    "Cloud functional verification failed. Please consider the following options:\n"
                    "1. Create a new cloud (we recommend)\n"
                    "2. Double-check the resources specified in the edit details, and the verification results, modify the resource if necessary, run `anyscale cloud verify (optional with functional-verify)` to verify again.\n"
                    "3. Edit the resource back to original if you still want to use this cloud.\n"
                )

    ### End of edit cloud ###

    def _transform_job_for_report(self, job) -> dict:
        """Extract common job fields for report output."""
        finished_at = str(job.finished_at) if job.finished_at else ""
        duration = str(job.finished_at - job.created_at) if job.finished_at else ""
        return {
            "job_id": job.job_id,
            "job_name": job.job_name,
            "job_state": HA_JOB_STATE_TO_JOB_STATE[job.job_state],
            "created_at": str(job.created_at),
            "finished_at": finished_at,
            "duration": duration,
            "unused_cpu_hours": job.job_report.unused_cpu_hours or "",
            "unused_gpu_hours": job.job_report.unused_gpu_hours or "",
            "max_instances_launched": job.job_report.max_instances_launched or "",
        }

    def _write_jobs_report_csv(self, out_file, jobs) -> None:
        """Write jobs report in CSV format."""
        out_file.write(
            "Job ID,Job name,Job state,Created at,Finished at,Duration,"
            "Unused CPU hours,Unused GPU hours,Max concurrent instances\n"
        )
        for job in track(jobs, description="Generating report..."):
            d = self._transform_job_for_report(job)
            out_file.write(
                f'"{d["job_id"]}","{d["job_name"]}","{d["job_state"]}",'
                f'"{d["created_at"]}","{d["finished_at"]}","{d["duration"]}",'
                f'"{d["unused_cpu_hours"]}","{d["unused_gpu_hours"]}",'
                f'"{d["max_instances_launched"]}"\n'
            )

    def _write_jobs_report_html(self, out_file, jobs, end_time, total_jobs) -> None:
        """Write jobs report in HTML format."""
        out_file.write(
            f"""
<html>
<head>
<title>Jobs Report - {end_time!s}</title>
<style>
    table {{
        border: 1px solid black;
        border-collapse: collapse;
    }}
    th, td {{
        border: 1px solid black;
        border-collapse: collapse;
        padding: 8px;
    }}
</style>
</head>
<body>
<h1>Job Report - {end_time!s}</h1>
<p>Total jobs reported (finished jobs): {len(jobs)}</p>
<p>Total jobs in the last 7 days: {total_jobs}</p>
<table>
<thead>
<tr>
<th>Job ID</th>
<th>Job name</th>
<th>Job state</th>
<th>Created at</th>
<th>Finished at</th>
<th>Duration</th>
<th>Unused CPU hours</th>
<th>Unused GPU hours</th>
<th>Max concurrent instances</th>
</tr>
</thead>
<tbody>
"""
        )

        for job in track(jobs, description="Generating report..."):
            d = self._transform_job_for_report(job)
            out_file.write(
                f"""
<tr>
<td><a target="_blank" rel="noreferrer" href="{ANYSCALE_HOST}/jobs/{d["job_id"]}">{d["job_id"]}</a></td>
<td>{d["job_name"]}</td>
<td>{d["job_state"]}</td>
<td>{d["created_at"]}</td>
<td>{d["finished_at"]}</td>
<td>{d["duration"]}</td>
<td>{d["unused_cpu_hours"]}</td>
<td>{d["unused_gpu_hours"]}</td>
<td>{d["max_instances_launched"]}</td>
</tr>
"""
            )

        out_file.write(
            """
</tbody>
</table>
"""
        )

    def generate_jobs_report(
        self, cloud_id: str, csv: bool, out_path: str, sort: str, sort_order_asc: bool
    ) -> None:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)

        full_results = []
        paging_token: Optional[str] = None
        count_per_page = 20
        curr_page_results = None
        total_jobs: Optional[int] = None

        with Progress() as progress:
            download_task = progress.add_task("Downloading jobs...", total=None)

            while curr_page_results is None or len(curr_page_results) == count_per_page:
                response = self.api_client.list_job_reports_api_v2_job_reports_get(
                    cloud_id,
                    start_time=start_time,
                    end_time=end_time,
                    paging_token=paging_token,
                    count=count_per_page,
                )
                curr_page_results = response.results
                full_results.extend(curr_page_results)
                paging_token = response.metadata.next_paging_token
                total_jobs = response.metadata.total
                progress.update(
                    download_task, total=total_jobs, advance=len(curr_page_results)
                )

            progress.update(download_task, completed=total_jobs)

        if not full_results:
            self.log.info("No jobs found in the last 7 days.")
            return

        filtered_results = [
            job
            for job in full_results
            if job.job_state in TERMINAL_HA_JOB_STATES and job.job_report is not None
        ]

        sort_keys = {
            "created_at": lambda x: x.created_at,
            "gpu": lambda x: x.job_report.unused_gpu_hours or 0,
            "cpu": lambda x: x.job_report.unused_cpu_hours or 0,
            "instances": lambda x: x.job_report.max_instances_launched or 0,
        }
        if sort in sort_keys:
            filtered_results.sort(key=sort_keys[sort], reverse=not sort_order_asc)

        with open(out_path, "w") as out_file:
            if csv:
                self._write_jobs_report_csv(out_file, filtered_results)
            else:
                self._write_jobs_report_html(
                    out_file, filtered_results, end_time, total_jobs
                )
