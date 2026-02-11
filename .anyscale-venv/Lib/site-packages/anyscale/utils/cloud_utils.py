import contextlib
from enum import Enum
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
import uuid

import boto3
from botocore.exceptions import ClientError
from click.exceptions import ClickException

from anyscale.cli_logger import CloudSetupLogger
from anyscale.client.openapi_client.api.default_api import DefaultApi
from anyscale.client.openapi_client.models import (
    CloudAnalyticsEvent,
    CloudAnalyticsEventCloudResource,
    CloudAnalyticsEventCommandName,
    CloudAnalyticsEventError,
    CloudAnalyticsEventName,
    CloudProviders,
    CreateAnalyticsEvent,
)
from anyscale.cloud_utils import get_cloud_id_and_name
from anyscale.shared_anyscale_utils.utils.collections import flatten


V = TypeVar("V")


def get_organization_default_cloud(api_client: DefaultApi) -> Optional[str]:
    """Return default cloud name for organization if it exists and
        if user has correct permissions for it.

        Returns:
            Name of default cloud name for organization if it exists and
            if user has correct permissions for it.
        """
    user = api_client.get_user_info_api_v2_userinfo_get().result
    organization = user.organizations[0]  # Each user only has one org
    if organization.default_cloud_id:
        try:
            # Check permissions
            _, cloud_name = get_cloud_id_and_name(
                api_client, cloud_id=organization.default_cloud_id
            )
            return str(cloud_name)
        except Exception:  # noqa: BLE001
            return None
    return None


def get_default_cloud(
    api_client: DefaultApi, cloud_name: Optional[str]
) -> Tuple[str, str]:
    """Returns the cloud id from cloud name.
    If cloud name is not provided, returns the default cloud name if exists in organization.
    If default cloud name does not exist returns last used cloud.
    """

    if cloud_name is None:
        default_cloud_name = get_organization_default_cloud(api_client)
        if default_cloud_name:
            cloud_name = default_cloud_name
    return get_cloud_id_and_name(api_client, cloud_name=cloud_name)


def verify_anyscale_access(
    api_client: DefaultApi,
    cloud_id: str,
    cloud_provider: CloudProviders,
    logger: CloudSetupLogger,
) -> bool:
    try:
        api_client.verify_access_api_v2_cloudsverify_access_cloud_id_get(cloud_id)
        return True
    except ClickException as e:
        if cloud_provider == CloudProviders.AWS:
            logger.log_resource_error(
                CloudAnalyticsEventCloudResource.AWS_IAM_ROLE,
                CloudSetupError.ANYSCALE_ACCESS_DENIED,
            )
        elif cloud_provider == CloudProviders.GCP:
            logger.log_resource_error(
                CloudAnalyticsEventCloudResource.GCP_SERVICE_ACCOUNT,
                CloudSetupError.ANYSCALE_ACCESS_DENIED,
            )
        logger.error(
            f"Anyscale's control plane is unable to access resources on your cloud provider.\n{e}"
        )
        return False


def modify_memorydb_parameter_group(
    parameter_group_name: str,
    region: str,
    boto3_session: Optional[boto3.Session] = None,
) -> None:
    """
    Modify the memorydb parameter group to set the maxmemory-policy to allkeys-lru.

    This is not done in the cloudformation template because we have to create the paramter group first and then modify it.
    """
    try:
        if boto3_session is None:
            boto3_session = boto3.Session()
        memorydb_client = boto3_session.client("memorydb", region_name=region)
        memorydb_client.update_parameter_group(
            ParameterGroupName=parameter_group_name,
            ParameterNameValues=[
                {"ParameterName": "maxmemory-policy", "ParameterValue": "allkeys-lru",}
            ],
        )
    except ClientError as e:
        # TODO (allenyin): add memorydb to the cloud provider error list.
        raise ClickException(
            f"Failed to modify memorydb parameter group {parameter_group_name}. Please make sure you have permission to perform UpdateParameterGroup on memorydb clusters and try again. \n{e}"
        )


def wait_for_lb_resource_termination(
    api_client: DefaultApi,
    cloud_id: str,
    timeout_s: int = 900,  # 15 minute timeout
    poll_interval_s: int = 10,  # Poll every 10 seconds
):
    start = time.time()
    while time.time() - start < timeout_s:
        response = api_client.get_lb_resource_api_v2_clouds_cloud_id_get_lb_resource_post(
            cloud_id=cloud_id
        ).result
        if response.is_terminated:
            break
        time.sleep(poll_interval_s)
    else:
        raise ClickException(
            f"LB resources and namespace termination timed out after {timeout_s} seconds."
        )


class CloudSetupError(str, Enum):
    ANYSCALE_ACCESS_DENIED = "ANYSCALE_ACCESS_DENIED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    CIDR_BLOCK_TOO_SMALL = "CIDR_BLOCK_TOO_SMALL"
    MISSING_CLOUD_RESOURCE_ID = "MISSING_CLOUD_RESOURCE_ID"
    ONLY_ONE_SUBNET = "ONLY_ONE_SUBNET"
    SUBNET_NOT_IN_VPC = "SUBNET_NOT_IN_VPC"
    ONLY_ONE_AZ = "ONLY_ONE_AZ"
    IAM_ROLE_ACCOUNT_MISMATCH = "IAM_ROLE_ACCOUNT_MISMATCH"
    INSTANCE_PROFILE_NOT_FOUND = "INSTANCE_PROFILE_NOT_FOUND"
    INTERNAL_COMMUNICATION_NOT_ALLOWED = "INTERNAL_COMMUNICATION_NOT_ALLOWED"
    MALFORMED_CORS_RULE = "MALFORMED_CORS_RULE"
    INCORRECT_CORS_RULE = "INCORRECT_CORS_RULE"
    MOUNT_TARGET_NOT_FOUND = "MOUNT_TARGET_NOT_FOUND"
    INVALID_MOUNT_TARGET = "INVALID_MOUNT_TARGET"
    PROJECT_NOT_ACTIVE = "PROJECT_NOT_ACTIVE"
    API_NOT_ENABLED = "API_NOT_ENABLED"
    FIREWALL_NOT_ASSOCIATED_WITH_VPC = "FIREWALL_NOT_ASSOCIATED_WITH_VPC"
    FILESTORE_NAME_MALFORMED = "FILESTORE_NAME_MALFORMED"
    FILESTORE_NOT_CONNECTED_TO_VPC = "FILESTORE_NOT_CONNECTED_TO_VPC"
    MEMORYSTORE_NOT_CONNECTED_TO_VPC = "MEMORYSTORE_NOT_CONNECTED_TO_VPC"
    MEMORYSTORE_READ_REPLICAS_DISABLED = "MEMORYSTORE_READ_REPLICAS_DISABLED"
    MEMORYDB_CLUSTER_UNAVAILABLE = "MEMORYDB_CLUSTER_UNAVAILABLE"


class CloudEventProducer:
    """
    Produce events during cloud setup/register/verify
    """

    def __init__(self, cli_version: str, api_client: DefaultApi):
        self.api_client = api_client
        self.cloud_id: Optional[str] = None
        self.cloud_provider: Optional[CloudProviders] = None
        self.cli_version = cli_version

    def init_trace_context(
        self,
        command_name: CloudAnalyticsEventCommandName,
        cloud_provider: CloudProviders,
        cloud_id: Optional[str] = None,
    ):
        self.trace_id = str(uuid.uuid4().hex)
        self.command_name = command_name
        self.raw_command_input = str(" ".join(sys.argv[1:]))
        self.cloud_id = cloud_id
        self.cloud_provider = cloud_provider

    def set_cloud_id(self, cloud_id: str):
        self.cloud_id = cloud_id

    def produce(
        self,
        event_name: CloudAnalyticsEventName,
        succeeded: bool,
        logger: Optional[CloudSetupLogger] = None,
        internal_error: Optional[str] = None,
    ):
        with contextlib.suppress(Exception):
            # shouldn't block cloud setup even if cloud event generation fails
            error = None
            if not succeeded:
                cloud_provider_error = None
                if logger:
                    cloud_provider_error_list = logger.get_cloud_provider_errors()
                    logger.clear_cloud_provider_errors()
                    if len(cloud_provider_error_list) > 0:
                        cloud_provider_error = cloud_provider_error_list
                error = CloudAnalyticsEventError(
                    internal_error=internal_error,
                    cloud_provider_error=cloud_provider_error,
                )

            self.api_client.produce_analytics_event_api_v2_analytics_post(
                CreateAnalyticsEvent(
                    cloud_analytics_event=CloudAnalyticsEvent(
                        cli_version=self.cli_version,
                        trace_id=self.trace_id,
                        cloud_id=self.cloud_id,
                        succeeded=succeeded,
                        command_name=self.command_name,
                        raw_command_input=self.raw_command_input,
                        cloud_provider=self.cloud_provider,
                        event_name=event_name,
                        error=error,
                    )
                )
            )


def validate_aws_credentials(
    logger: CloudSetupLogger, boto3_session: Optional[Any] = None
) -> bool:
    try:
        if boto3_session is None:
            boto3_session = boto3.Session()
        boto3.client("sts").get_caller_identity()
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to validate AWS credentials: {e}")
        return False


def get_errored_resources_and_reasons(
    cfn_client: Any, stack_name: str
) -> Dict[str, str]:
    """
    Describes the CloudFormation stack events and extracts the failure reasons.
    """
    error_details: Dict[str, str] = {}
    response = cfn_client.describe_stack_events(StackName=stack_name)
    error_details.update(
        extract_cloudformation_failure_reasons(response["StackEvents"])
    )
    while response.get("NextToken") is not None:
        response = cfn_client.describe_stack_events(
            StackName=stack_name, NextToken=response["NextToken"]
        )
        error_details.update(
            extract_cloudformation_failure_reasons(response["StackEvents"])
        )
    return error_details


def extract_cloudformation_failure_reasons(
    events: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    Extracts the failure reasons from the CloudFormation events.
    """
    failure_reasons: Dict[str, str] = {}
    for event in events:
        if event.get("ResourceStatus") == "CREATE_FAILED":
            error_resource = event.get("LogicalResourceId", "UnknownResource")
            error_reason = event.get("ResourceStatusReason", "Unknown reason")
            if "Resource creation cancelled" in error_reason:
                # Resource creation cancelled due to other resource failures.
                # Skip this error.
                continue
            failure_reasons[error_resource] = error_reason
    return failure_reasons


def _unroll_resources_for_aws_list_call(aws_list_function: Callable, list_key: str):
    def token_extractor(response) -> Optional[str]:
        return response.get("NextToken")

    # NOTE: `NextToken` is expected to have `str` type and therefore we have
    #       we have to workaround its optionality by completely omitting it
    #       from the call
    responses = unroll_pagination(
        lambda next_token: aws_list_function(
            **({"NextToken": next_token} if next_token else {}),  # type: ignore
        ),
        token_extractor,
    )

    return flatten(*[r[list_key] for r in responses if list_key in r])


def unroll_pagination(
    paginated_method: Callable, next_token_extractor: Callable,
):
    """Handles paginated method's invocation by
    - Repeatedly invoking the method (injecting the token from its previous response)
    - Collecting all of the responses in a list
    """

    next_token: Optional[str] = None
    results = []
    while True:
        r = paginated_method(next_token)
        results.append(r)
        next_token = next_token_extractor(r)
        if not next_token:
            break

    return results
