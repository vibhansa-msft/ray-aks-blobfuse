from contextlib import contextmanager
from copy import deepcopy
import datetime
from enum import Enum
import ipaddress
import json
import logging
import os
import random
import shutil
import string
import sys
import time
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from urllib.parse import urljoin

import boto3
from boto3.resources.base import ServiceResource as Boto3Resource
from botocore.config import Config
from botocore.exceptions import ClientError, NoRegionError
import click
from packaging import version
import requests
from requests import Response

from anyscale.authenticate import get_auth_api_client
from anyscale.aws_iam_policies import ANYSCALE_IAM_POLICIES, AnyscaleIAMPolicy
from anyscale.cli_logger import BlockLogger, CloudSetupLogger
from anyscale.client.openapi_client.api.default_api import DefaultApi as ProductApi
from anyscale.client.openapi_client.models import (
    AWSMemoryDBClusterConfig,
    ServiceEventCurrentState,
)
from anyscale.client.openapi_client.models.cloud_analytics_event_cloud_resource import (
    CloudAnalyticsEventCloudResource,
)
from anyscale.client.openapi_client.models.decorated_compute_template import (
    DecoratedComputeTemplate,
)
from anyscale.client.openapi_client.models.ha_job_states import HaJobStates
from anyscale.client.openapi_client.models.user_info import UserInfo
from anyscale.cluster_compute import get_cluster_compute_from_name
import anyscale.conf
from anyscale.conf import MINIMUM_RAY_VERSION
from anyscale.feature_flags import FLAG_DEFAULT_WORKING_DIR_FOR_PROJ
from anyscale.sdk.anyscale_client.api.default_api import DefaultApi as BaseApi
from anyscale.sdk.anyscale_client.models.cluster import Cluster
from anyscale.sdk.anyscale_client.models.compute_template import ComputeTemplate
import anyscale.shared_anyscale_utils.conf as shared_anyscale_conf
from anyscale.utils.cloud_utils import CloudSetupError


logger = logging.getLogger(__file__)

BOTO_MAX_RETRIES = 5
PROJECT_NAME_ENV_VAR = "ANYSCALE_PROJECT_NAME"

VALID_BYOD_PYTHON_VERSIONS = ["py36", "py37", "py38", "py39"]

REDIS_TLS_ADDRESS_PREFIX = "rediss://"

log = BlockLogger()  # Anyscale CLI Logger

VPC_CIDR_RANGE = "10.0.0.0/16"

DEFAULT_RAY_VERSION = "1.7.0"

# The V2 stack has some delay before changing the state of a new cluster from
# Terminated to StartingUp. Allow the terminated state for the duration of this
# grace period before erroring.
TERMINATED_STATE_GRACE_PERIOD_SECONDS = 5 * 60  # 5 minutes

MEMORY_DB_OUTPUT = """  MemoryDB:
    Description: MemoryDB cluster
    Value:
      Fn::ToJsonString:
        arn: !GetAtt MemoryDB.ARN
        ClusterEndpointAddress: !GetAtt MemoryDB.ClusterEndpoint.Address"""

MEMORY_DB_RESOURCE = """  MemoryDBSubnetGroup:
    Type: AWS::MemoryDB::SubnetGroup
    Properties:
      Description: Anyscale managed MemoryDB subnet group
      SubnetGroupName: !Ref AWS::StackName
      SubnetIds:
{}
      Tags:
        - Key: anyscale-cloud-id
          Value: !Ref CloudID

  MemoryDBParameterGroup:
    Type: AWS::MemoryDB::ParameterGroup
    Properties:
      Description: Parameter group for anyscale managed MemoryDB
      Family: memorydb_redis7
      ParameterGroupName:  !Ref AWS::StackName
      Tags:
        - Key: anyscale-cloud-id
          Value: !Ref CloudID

  MemoryDB:
    Type: AWS::MemoryDB::Cluster
    Properties:
      ACLName: open-access
      Description: Anyscale managed MemoryDB
      ClusterName: !Ref AWS::StackName
      NodeType: db.t4g.small
      Port: !Ref MemoryDBRedisPort
      SubnetGroupName: !Ref MemoryDBSubnetGroup
      SecurityGroupIds:
        - !Ref AnyscaleSecurityGroup
      EngineVersion: "7.0"
      ParameterGroupName: !Ref MemoryDBParameterGroup
      TLSEnabled: true
      Tags:
        - Key: anyscale-cloud-id
          Value: !Ref CloudID"""

# Some resources (e.g. memorystore) take a long time to create, so we need to increase the timeout.
GCP_DEPLOYMENT_MANAGER_TIMEOUT_SECONDS_LONG = 600  # 10 minutes


class AnyscaleJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Anyscale models, handling datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def confirm(msg: str, yes: bool) -> Optional[bool]:
    return None if yes else click.confirm(msg, abort=True)


class SharedStorageType(str, Enum):
    """Enum for shared storage types."""

    OBJECT_STORAGE = "object-storage"
    NFS = "nfs"


class AnyscaleEndpointFormatter:
    def __init__(self, host: Optional[str] = None):
        self.host = host or shared_anyscale_conf.ANYSCALE_HOST

    def get_endpoint(self, endpoint: str) -> str:
        return str(urljoin(self.host, endpoint))

    def get_job_endpoint(self, job_id: str) -> str:
        return self.get_endpoint(f"/jobs/{job_id}")

    def get_schedule_endpoint(self, schedule_id: str) -> str:
        return self.get_endpoint(f"/scheduled-jobs/{schedule_id}")


def get_endpoint(endpoint: str, host: Optional[str] = None) -> str:
    return str(urljoin(host or shared_anyscale_conf.ANYSCALE_HOST, endpoint))


def send_json_request_raw(
    endpoint: str,
    json_args: Dict[str, Any],
    method: str = "GET",
    cli_token: Optional[str] = None,
    host: Optional[str] = None,
) -> Response:
    get_auth_api_client(cli_token=cli_token, host=host)

    url = get_endpoint(endpoint, host=host)
    cookies = {"cli_token": cli_token or anyscale.conf.CLI_TOKEN or ""}
    try:
        if method == "GET":
            resp = requests.get(url, params=json_args, cookies=cookies)
        elif method == "POST":
            resp = requests.post(url, json=json_args, cookies=cookies)
        elif method == "DELETE":
            resp = requests.delete(url, json=json_args, cookies=cookies)
        elif method == "PATCH":
            resp = requests.patch(url, data=json_args, cookies=cookies)
        elif method == "PUT":
            resp = requests.put(url, json=json_args, cookies=cookies)
        else:
            raise AssertionError(f"unknown method {method}")
    except requests.exceptions.ConnectionError:
        raise click.ClickException(f"Failed to connect to anyscale server at {url}")

    return resp


def send_json_request(
    endpoint: str,
    json_args: Dict[str, Any],
    method: str = "GET",
    cli_token: Optional[str] = None,
    host: Optional[str] = None,
) -> Dict[str, Any]:
    resp = send_json_request_raw(
        endpoint, json_args, method=method, cli_token=cli_token, host=host,
    )

    if not resp.ok:
        if resp.status_code == 500:
            raise click.ClickException(
                "There was an internal error in this command. "
                "Please report this to the Anyscale team at support@anyscale.com "
                "with the token '{}'.".format(resp.headers["x-trace-id"])
            )

        raise click.ClickException(f"{resp.status_code}: {resp.text}.")

    if resp.status_code == 204:
        return {}

    json_resp: Dict[str, Any] = resp.json()
    if "error" in json_resp:
        raise click.ClickException("{}".format(json_resp["error"]))

    return json_resp


def deserialize_datetime(s: str) -> datetime.datetime:
    if sys.version_info < (3, 7) and s[-3:-2] == ":":
        s = s[:-3] + s[-2:]

    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%z")


def humanize_timestamp(timestamp: datetime.datetime) -> str:
    delta = datetime.datetime.now(datetime.timezone.utc) - timestamp
    offset = float(delta.seconds + (delta.days * 60 * 60 * 24))
    delta_s = int(offset % 60)
    offset /= 60
    delta_m = int(offset % 60)
    offset /= 60
    delta_h = int(offset % 24)
    offset /= 24
    delta_d = int(offset)

    if delta_d >= 1:
        return "{} day{} ago".format(delta_d, "s" if delta_d > 1 else "")
    if delta_h > 0:
        return "{} hour{} ago".format(delta_h, "s" if delta_h > 1 else "")
    if delta_m > 0:
        return "{} minute{} ago".format(delta_m, "s" if delta_m > 1 else "")
    else:
        return "{} second{} ago".format(delta_s, "s" if delta_s > 1 else "")


def get_requirements(requirements_path: str) -> str:
    with open(requirements_path) as f:
        return f.read()


def _resource(name: str, region: str) -> Boto3Resource:
    boto_config = Config(retries={"max_attempts": BOTO_MAX_RETRIES})
    return boto3.resource(name, region, config=boto_config)  # type: ignore


def _client(name: str, region: str) -> Any:
    return _resource(name, region).meta.client


def _get_role(
    role_name: str, region: str, boto3_session: Optional[boto3.Session] = None
) -> Optional[Boto3Resource]:
    iam = (
        _resource("iam", region)
        if boto3_session is None
        else boto3_session.resource("iam")
    )
    role = iam.Role(role_name)  # type: ignore
    try:
        role.load()
        return role  # type: ignore
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "NoSuchEntity":
            return None
        else:
            raise exc


def _get_aws_efs_mount_target_ip(boto3_session: Any, efs_id: str) -> Optional[str]:
    client = boto3_session.client("efs")
    try:
        mount_targets_response = client.describe_mount_targets(FileSystemId=efs_id)
        if not mount_targets_response.get("MountTargets"):
            logger.warning(f"EFS with id {efs_id} does not contain mount targets.")
            return None
        return mount_targets_response.get("MountTargets")[0].get("IpAddress")
    except ClientError as e:
        if e.response["Error"]["Code"] == "FileSystemNotFound":
            return None
        raise e


def _get_subnet(
    subnet_arn: str, region: str, logger: CloudSetupLogger
) -> Optional[Boto3Resource]:
    ec2 = _resource("ec2", region)  # TODO: take a resource as an argument
    subnet = ec2.Subnet(subnet_arn)  # type: ignore
    try:
        subnet.load()
        return subnet  # type: ignore
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidSubnetID.NotFound":
            logger.log_resource_error(
                CloudAnalyticsEventCloudResource.AWS_SUBNET,
                CloudSetupError.RESOURCE_NOT_FOUND,
            )
            raise click.ClickException(
                f"{subnet_arn} does not exist. Please make sure the subnet arn is correct and the subnet is in the same region as the cloud."
            )
        else:
            logger.log_resource_exception(
                CloudAnalyticsEventCloudResource.AWS_SUBNET, e
            )
        raise e


def _get_memorydb_cluster_config(
    memorydb_cluster_id: str, region: str, logger: CloudSetupLogger
) -> Optional[AWSMemoryDBClusterConfig]:
    try:
        memorydb_client = boto3.client("memorydb", region_name=region)
        response = memorydb_client.describe_clusters(ClusterName=memorydb_cluster_id)

        if not response.get("Clusters") or not response.get("Clusters")[0]:
            logger.log_resource_error(
                CloudAnalyticsEventCloudResource.AWS_MEMORYDB,
                CloudSetupError.RESOURCE_NOT_FOUND,
            )
            raise click.ClickException(
                f"MemoryDB cluster with id {memorydb_cluster_id} does not exist."
            )
        cluster = response["Clusters"][0]
        if cluster["Status"] != "available":
            logger.log_resource_error(
                CloudAnalyticsEventCloudResource.AWS_MEMORYDB,
                CloudSetupError.MEMORYDB_CLUSTER_UNAVAILABLE,
            )
            raise click.ClickException(
                f"MemoryDB cluster with id {memorydb_cluster_id} is not currently available. Please make sure the cluster is available and try again."
            )
        endpoint = cluster["ClusterEndpoint"]

        return AWSMemoryDBClusterConfig(
            id=cluster["ARN"],
            endpoint=f"{REDIS_TLS_ADDRESS_PREFIX}{endpoint['Address']}:{endpoint['Port']}",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ClusterNotFoundFault":
            logger.log_resource_error(
                CloudAnalyticsEventCloudResource.AWS_MEMORYDB,
                CloudSetupError.RESOURCE_NOT_FOUND,
            )
        else:
            logger.log_resource_exception(
                CloudAnalyticsEventCloudResource.AWS_MEMORYDB, e
            )
        raise e


def get_available_regions(boto3_session: Optional[boto3.Session] = None) -> List[str]:
    if boto3_session is None:
        boto3_session = boto3.Session()
    try:
        client = boto3_session.client("ec2")
    except NoRegionError:
        # If there is no region, default to `us-west-2`
        client = boto3_session.client("ec2", region_name="us-west-2")
    return [region["RegionName"] for region in client.describe_regions()["Regions"]]


def get_availability_zones(
    region: str, boto3_session: Optional[boto3.Session] = None
) -> Dict[str, str]:
    """
    Returns a mapping of availability zone ids to names for the given region.
    """
    if boto3_session is None:
        boto3_session = boto3.Session()
    client = boto3_session.client("ec2", region_name=region)
    return {
        zone["ZoneId"]: zone["ZoneName"]
        for zone in client.describe_availability_zones()["AvailabilityZones"]
    }


def get_memorydb_supported_zones(
    region: str, zone_ids_to_names: Dict[str, str]
) -> List[str]:
    """
    Returns a list of supported zone names for MemoryDB in the given region.
    """
    with open(f"{anyscale.conf.ROOT_DIR_PATH}/memorydb_supported_zones.json") as f:
        # There's no API to get the supported zones for MemoryDB, so we need to read it from a file.
        memorydb_supported_zones = json.load(f)

    supported_zone_ids_for_cloud_region = memorydb_supported_zones.get(region, [])
    if len(supported_zone_ids_for_cloud_region) == 0:
        raise click.ClickException(f"MemoryDB is not supported in the region {region}.")
    return [
        zone_ids_to_names[zone_id] for zone_id in supported_zone_ids_for_cloud_region
    ]


def get_project_directory_name(project_id: str, api_client: ProductApi = None) -> str:
    if api_client is None:
        api_client = get_auth_api_client().api_client

    # TODO (yiran): return error early if project doesn't exist.
    resp = api_client.get_project_api_v2_projects_project_id_get(project_id)
    if api_client.check_is_feature_flag_on_api_v2_userinfo_check_is_feature_flag_on_get(
        FLAG_DEFAULT_WORKING_DIR_FOR_PROJ
    ).result.is_on:
        directory_name = resp.result.directory_name
    else:
        directory_name = resp.result.name
    assert len(directory_name) > 0, "Empty directory name found."
    return cast(str, directory_name)


def get_working_dir(project_id: str, api_client: ProductApi = None) -> str:
    project_directory_name = get_project_directory_name(project_id, api_client)
    return f"/home/ray/{project_directory_name}"


# Python 3.7 and before have different wheel name conventions.
# Though product today only support Python 3.9 or above.
_LEGACY_PY_VERSIONS = ["3" + str(x) for x in range(8)]


def get_wheel_url(
    ray_commit: str,
    ray_version: str,
    py_version: Optional[str] = None,
    sys_platform: Optional[str] = None,
) -> str:
    """Return S3 URL for the given release spec or 'latest'."""
    if py_version is None:
        py_version = "".join(str(x) for x in sys.version_info[0:2])
    if sys_platform is None:
        sys_platform = sys.platform

    if sys_platform == "darwin":
        if py_version not in _LEGACY_PY_VERSIONS:
            platform = "macosx_10_15_x86_64"
        else:
            platform = "macosx_10_15_intel"
    elif sys_platform == "win32":
        platform = "win_amd64"
    else:
        platform = "manylinux2014_x86_64"

    py_version_malloc = (
        f"{py_version}m" if py_version in _LEGACY_PY_VERSIONS else py_version
    )

    if "dev" in ray_version:
        ray_release = f"master/{ray_commit}"
    else:
        ray_release = f"releases/{ray_version}/{ray_commit}"
    return (
        "https://s3-us-west-2.amazonaws.com/ray-wheels/"
        "{}/ray-{}-cp{}-cp{}-{}.whl".format(
            ray_release, ray_version, py_version, py_version_malloc, platform
        )
    )


@contextmanager
def updating_printer() -> Generator[Callable[[str], None], None, None]:
    cols, _ = shutil.get_terminal_size()

    def print_status(status: str) -> None:
        lines = status.splitlines()
        first_line = lines[0]
        truncated_first_line = (
            first_line[0:cols]
            if len(first_line) <= cols and len(lines) == 1
            else (first_line[0 : cols - 3] + "...")
        )
        # Clear the line first
        print("\r" + " " * cols, end="\r")
        print(truncated_first_line, end="", flush=True)

    try:
        yield print_status
    finally:
        # Clear out the status and return to the beginning to reprint
        print("\r" + " " * cols, end="\r", flush=True)


def wait_for_session_start(  # noqa: PLR0912
    project_id: str,
    session_name: str,
    api_client: Optional[ProductApi] = None,
    log: BlockLogger = log,
    block_label: Optional[str] = None,
) -> str:
    if block_label:
        log.info(
            f"Waiting for cluster {BlockLogger.highlight(session_name)} to start. This may take a few minutes",
            block_label=block_label,
        )
    else:
        log.info(
            f"Waiting for cluster {session_name} to start. This may take a few minutes"
        )

    if api_client is None:
        api_client = get_auth_api_client().api_client

    start_time = time.time()
    with updating_printer() as print_status:
        while True:
            sessions = api_client.list_sessions_api_v2_sessions_get(
                project_id=project_id, name=session_name, active_only=False
            ).results

            if len(sessions) > 0:
                session = sessions[0]

                # TODO: Remove "session.host_name" check once https://github.com/anyscale/product/issues/15502 is fixed
                # A cluster may have "running" state while its dns is not set up yet. When DNS is not ready,
                # many cluster operations, like ray job submission will fail. So we wait until DNS info
                # is ready.
                if (
                    session.state == "Running"
                    and session.pending_state is None
                    and session.host_name
                ):
                    return cast(str, session.id)

                # Check for start up errors
                if (
                    session.state_data
                    and session.state_data.startup
                    and session.state_data.startup.startup_error
                ):
                    raise click.ClickException(
                        f"Error while starting cluster {session_name}: {session.state_data.startup.startup_error}"
                    )
                elif (
                    session.state
                    and "Errored" in session.state
                    and session.pending_state is None
                ):
                    raise click.ClickException(
                        f"Error while starting cluster {session_name}: Cluster startup failed due to an error ({session.state})."
                    )
                elif (
                    session.state
                    and session.state in {"Terminated", "Stopped"}
                    and session.pending_state is None
                ):
                    if time.time() - start_time < TERMINATED_STATE_GRACE_PERIOD_SECONDS:
                        # The V2 stack has some delay before transitioning from Terminated to StartingUp.
                        # Don't error until the grace period has passed
                        print_status("Waiting for start up...")
                    else:
                        # Cluster is created in Terminated state; Check pending state to see if it is pending transition.
                        raise click.ClickException(
                            f"Error while starting cluster {session_name}: Cluster is still in stopped/terminated state."
                        )
                elif (
                    session.state_data
                    and session.state_data.startup
                    and session.state_data.startup.startup_progress
                ):
                    # Print the latest status
                    print_status(
                        "Starting up " + session.state_data.startup.startup_progress
                    )
                elif (
                    session.state != "StartingUp"
                    and session.pending_state == "StartingUp"
                ):
                    print_status("Waiting for start up...")
            else:
                raise click.ClickException(
                    f"Error while starting cluster {session_name}: Cluster doesn't exist."
                )

            time.sleep(2)


def populate_session_args(cluster_config_str: str, config_file_name: str) -> str:
    import jinja2  # noqa: PLC0415 - codex_reason("gpt5.2", "optional templating dependency for config rendering")

    env = jinja2.Environment()
    t = env.parse(cluster_config_str)
    for elem in t.body[0].nodes:  # type: ignore
        if isinstance(elem, jinja2.nodes.Getattr) and elem.attr not in os.environ:
            prefixed_command = " ".join(
                [f"{elem.attr}=<value>", "anyscale"] + sys.argv[1:]
            )
            raise click.ClickException(
                f"\tThe environment variable {elem.attr} was not set, yet it is required "
                f"for configuration file {config_file_name}.\n\tPlease specify {elem.attr} "
                f"by prefixing the command.\n\t\t{prefixed_command}"
            )

    template = jinja2.Template(cluster_config_str)
    cluster_config_filled = template.render(env=os.environ)
    return cluster_config_filled


def get_user_info() -> Optional[UserInfo]:
    try:
        api_client = get_auth_api_client().api_client
    except click.exceptions.ClickException:
        return None
    return api_client.get_user_info_api_v2_userinfo_get().result


def generate_slug(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def validate_non_negative_arg(ctx, param, value):  # noqa: ARG001
    """
    Checks that an integer option to click command is non-negative.
    """
    if value is not None and value < 0:
        raise click.ClickException(
            f"Please specify a non-negative value for {param.opts[0]}"
        )
    return value


def validate_service_state_filter(
    ctx, param, value: Tuple[str, ...]  # noqa: ARG001
) -> List[str]:
    """Validate ServiceEventCurrentState values."""
    if not value:
        return []

    allowable_values_upper = {
        s.upper() for s in ServiceEventCurrentState.allowable_values
    }
    allowed_values_str = ", ".join(ServiceEventCurrentState.allowable_values)

    for state_str in value:
        state_upper = state_str.upper()
        if state_upper not in allowable_values_upper:
            raise click.ClickException(
                f"'{state_str}' is not a valid value for {param.opts[0]}. Allowed values: {allowed_values_str}"
            )

    return [s.upper() for s in value]


def validate_workspace_state_filter(
    ctx, param, value: Tuple[str, ...]  # noqa: ARG001
) -> List[str]:
    """Validate WorkspaceState values (case-insensitive) and return uppercase."""
    if not value:
        return []

    # Import here to avoid circular dependency
    from anyscale.workspace.models import (  # noqa: PLC0415 - codex_reason("gpt5.2", "avoid circular dependency during CLI import")
        WorkspaceState,
    )

    allowable_values_upper = {s.value.upper() for s in WorkspaceState}
    allowed_values_str = ", ".join(s.value for s in WorkspaceState)

    for state_str in value:
        state_upper = state_str.upper()
        if state_upper not in allowable_values_upper:
            raise click.ClickException(
                f"'{state_str}' is not a valid value for {param.opts[0]}. Allowed values: {allowed_values_str}"
            )

    # Return uppercase WorkspaceState values (e.g., "RUNNING")
    return [s.upper() for s in value]


def _update_external_ids_for_policy(
    original_policy: Dict[str, Any], new_external_id: str
):
    """Gets All External IDs From policy Dict."""
    policy = deepcopy(original_policy)
    external_ids = [
        statement.setdefault("Condition", {})
        .setdefault("StringEquals", {})
        .setdefault("sts:ExternalId", [])
        for statement in policy.get("Statement", [])
    ]

    external_ids = [
        [i, new_external_id] if isinstance(i, str) else i + [new_external_id]
        for i in external_ids
    ]

    # remove duplicate external IDs
    external_ids = [sorted(set(ids)) for ids in external_ids]

    _ = [
        policy["Statement"][i]["Condition"]["StringEquals"].update(
            {"sts:ExternalId": external_ids[i]}
        )
        for i in range(len(policy.get("Statement", [])))
    ]
    return policy


def extract_versions_from_image_name(image_name: str) -> Tuple[str, str]:
    """Returns the python version and ray extracted from an image tag.
    This should be used when creating BYOD images.

    Args:
        image_name: e.g. anyscale/ray-ml:1.11.1-py38-gpu

    Returns:
        The (python version, ray version), e.g. ("py38", "1.11.1")
    """
    # e.g. 1.11.1-py32-gpu
    image_version = image_name.split(":")[-1]
    parts = image_version.split("-")
    # e.g. 1.11.1

    if len(parts) < 2:
        raise ValueError(
            f"Expected the docker image name have an image version tag (something like ray-ml:1.11.1-py38-gpu), got {image_version}."
        )

    ray_version = parts[0]
    # Verify ray_version is valid.
    _ray_version_major_minor(ray_version)

    python_version = parts[1]
    _check_python_version(python_version)
    return (python_version, ray_version)


def _ray_version_major_minor(ray_version: str) -> Tuple[int, int]:
    """Takes in a Ray version such as "1.9.0rc1", "1.10.2", "2.0.0dev0".

    Returns the major minor pair e.g. (1,9) (1,10) (2,0).

    To avoid introducing undesirable dependencies, partly duplicates logic from the Anyscale
    backend.
    """
    invalid_ray_version_msg = (
        f"The Ray version `{ray_version}` has an unexpected format."
    )
    version_components = ray_version.split(".")
    assert len(version_components) >= 2, invalid_ray_version_msg
    major_str, minor_str = version_components[:2]
    assert major_str.isnumeric() and minor_str.isnumeric(), invalid_ray_version_msg
    major_int, minor_int = int(major_str), int(minor_str)
    return (major_int, minor_int)


def _check_python_version(python_version: str) -> None:
    assert (
        python_version in VALID_BYOD_PYTHON_VERSIONS
    ), f"Expected python_version to be one of {VALID_BYOD_PYTHON_VERSIONS}, got {python_version}."


def sleep_till(wake_time: float) -> None:
    """Sleep till the designated time"""
    now = time.time()
    if now >= wake_time:
        return
    time.sleep(wake_time - now)


def poll(
    interval_secs: float = 1,
    timeout_secs: Optional[float] = None,
    max_iter: Optional[int] = None,
) -> Generator[int, None, None]:
    """Poll every interval_secs, until timeout_secs, or max iterations has been reached.
    Yield the iteration number, starting at 1.
    """
    now = time.time()
    end_time = now + timeout_secs if timeout_secs else None
    count = 0
    should_continue_iter = max_iter is None or count < max_iter
    should_continue_time = end_time is None or now < end_time
    while should_continue_iter and should_continue_time:
        count += 1
        wake_time = now + interval_secs
        yield count
        sleep_till(wake_time)
        now = time.time()
        should_continue_iter = max_iter is None or count < max_iter
        should_continue_time = end_time is None or now < end_time


def is_anyscale_workspace() -> bool:
    return "ANYSCALE_EXPERIMENTAL_WORKSPACE_ID" in os.environ


def is_anyscale_cluster() -> bool:
    return "ANYSCALE_SESSION_ID" in os.environ


def credentials_check_sanity(credentials_str: str) -> bool:
    """
    The main goal of this function is to perform a minimal sanity check
    to make sure that the CLI token string (entered by user or read from file)
    is not totally broken.
    refer to https://www.notion.so/anyscale-hq/Authentication-Infrastructure
    """
    # Old token style
    if credentials_str.startswith("sss_"):
        return True
    # Future token style
    return bool(credentials_str.startswith("a") and credentials_str.count("_") > 0)


def get_current_cluster_id() -> Optional[str]:
    """If we are running on an Anyscale Cluster, return the id from the environment."""
    return os.getenv("ANYSCALE_SESSION_ID")


def str_data_size(s: str) -> int:
    """Returns the size of the string when encoded to raw bytes."""
    return len(s.encode("utf-8"))


def get_user_env_aws_account(region: str) -> str:
    """Get the AWS account used in the user environment"""
    return boto3.client("sts", region_name=region).get_caller_identity()["Account"]


def generate_inline_policy_parameter(policy: AnyscaleIAMPolicy) -> str:
    """Generate the inline policy paramter for the cross account role"""
    return f"""  {policy.parameter_key}:
    Description: {policy.parameter_description}
    Type: String"""


def generate_inline_policy_resource(policy: AnyscaleIAMPolicy) -> str:
    """Generate the inline policy resource for the cross account role"""
    return f"""  {policy.resource_logical_id}:
    Type: AWS::IAM::Policy
    Properties:
      PolicyDocument: !Ref {policy.parameter_key}
      PolicyName: {policy.policy_name}
      Roles:
        - !Ref customerRole"""


def get_anyscale_cross_account_iam_policies() -> List[Dict[str, str]]:
    return [
        {
            "ParameterKey": policy.parameter_key,
            "ParameterValue": policy.policy_document,
        }
        for policy in ANYSCALE_IAM_POLICIES
    ]


def prepare_cloudformation_template(
    region: str,
    cfn_stack_name: str,
    cloud_id: str,
    enable_head_node_fault_tolerance: bool,
    boto3_session: Optional[boto3.Session] = None,
    is_anyscale_hosted: bool = False,
    shared_storage: SharedStorageType = SharedStorageType.OBJECT_STORAGE,
) -> str:
    if is_anyscale_hosted:
        with open(f"{anyscale.conf.ROOT_DIR_PATH}/anyscale-cloud-setup-oa.yaml") as f:
            body = f.read()
    else:
        with open(f"{anyscale.conf.ROOT_DIR_PATH}/anyscale-cloud-setup.yaml") as f:
            body = f.read()
        body = body.replace(
            "$ALLOWED_ORIGIN", shared_anyscale_conf.ANYSCALE_CORS_ORIGIN
        )

    zone_ids_to_names = get_availability_zones(region, boto3_session)
    azs = sorted(zone_ids_to_names.values())
    subnet_templates: List[str] = []
    subnets_route_table_association: List[str] = []
    subnets_with_availability_zones: List[str] = []
    efs_mount_targets: List[str] = []

    vpc_cidr = ipaddress.ip_network(VPC_CIDR_RANGE)
    if len(azs) > 4:
        subnet_cidrs = list(vpc_cidr.subnets(prefixlen_diff=3))
    else:
        subnet_cidrs = list(vpc_cidr.subnets(prefixlen_diff=2))

    for i, az in enumerate(azs):
        subnet_templates.append(
            f"""
  Subnet{i}:
    Type: AWS::EC2::Subnet
    Properties:
        VpcId: !Ref VPC
        AvailabilityZone: {az}
        CidrBlock: {subnet_cidrs[i]}
        MapPublicIpOnLaunch: true
        Tags:
        - Key: Name
          Value: {cfn_stack_name}-subnet-{az}
        - Key: anyscale-cloud-id
          Value: {cloud_id}"""
        )

        subnets_route_table_association.append(
            f"""
  Subnet{i}RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
        RouteTableId: !Ref PublicRouteTable
        SubnetId: !Ref Subnet{i}"""
        )

        subnets_with_availability_zones.append(
            f'{{"subnet_id": !Ref Subnet{i}, "availability_zone": !GetAtt Subnet{i}.AvailabilityZone}}'
        )

        if not is_anyscale_hosted and shared_storage == SharedStorageType.NFS:
            efs_mount_targets.append(
                f"""
  EFSMountTarget{i}:
    Type: AWS::EFS::MountTarget
    Properties:
        FileSystemId: !Ref EFS
        SecurityGroups:
          - !Ref AnyscaleSecurityGroup
        SubnetId: !Ref Subnet{i}"""
            )

    body = body.replace("$SUBNETS_TEMPLATES", "\n".join(subnet_templates))
    body = body.replace(
        "$SUBNETS_ROUTE_TABLE_ASSOCIATION", "\n".join(subnets_route_table_association),
    )
    body = body.replace(
        "$SUBNETS_WITH_AVAILABILITY_ZONES",
        f'[{",".join(subnets_with_availability_zones)}]',
    )
    if not is_anyscale_hosted:
        body = body.replace("$EFSMountTargets", "\n".join(efs_mount_targets))

        # Only include EFS mount target output if mount targets are actually created
        if efs_mount_targets and shared_storage == SharedStorageType.NFS:
            efs_mount_target_output = """
  EFSMountTargetIP:
    Description: Anyscale managed EFS mount target
    Value: !GetAtt
      - EFSMountTarget0
      - IpAddress"""
            body = body.replace("$EFS_MOUNT_TARGET_OUTPUT", efs_mount_target_output)
        else:
            body = body.replace("$EFS_MOUNT_TARGET_OUTPUT", "")
    else:
        # For anyscale hosted, remove the EFS mount target output placeholder
        body = body.replace("$EFS_MOUNT_TARGET_OUTPUT", "")

    iam_policy_parameters = [
        generate_inline_policy_parameter(policy) for policy in ANYSCALE_IAM_POLICIES
    ]
    body = body.replace(
        "$ANYSCALE_CROSS_ACCOUNT_IAM_POLICY_PARAMETERS",
        "\n\n".join(iam_policy_parameters),
    )

    iam_policy_resources = [
        generate_inline_policy_resource(policy) for policy in ANYSCALE_IAM_POLICIES
    ]
    body = body.replace(
        "$ANYSCALE_CROSS_ACCOUNT_IAM_POLICY_RESOURCES",
        "\n\n".join(iam_policy_resources),
    )

    if enable_head_node_fault_tolerance:
        supported_zone_names = get_memorydb_supported_zones(region, zone_ids_to_names)
        allowed_az_indices = sorted(azs.index(az) for az in supported_zone_names)
        body = body.replace("$MEMORY_DB_OUTPUT", MEMORY_DB_OUTPUT,)
        body = body.replace(
            "$MEMORY_DB_RESOURCE",
            MEMORY_DB_RESOURCE.format(
                "\n".join([f"        - !Ref Subnet{i}" for i in allowed_az_indices])
            ),
        )
    else:
        body = body.replace("$MEMORY_DB_OUTPUT", "")
        body = body.replace("$MEMORY_DB_RESOURCE", "")

    return body


def get_latest_ray_version():
    """
    Gets latest Ray version from PYPI. This method should not
    assume Ray is already installed.
    """
    try:
        response = requests.get("https://pypi.org/pypi/ray/json")
        latest_version = response.json()["info"]["version"]
    except Exception as e:  # noqa: BLE001
        log.debug(
            f"Unable to get latest Ray version from https://pypi.org/pypi/ray/json {e!s}"
        )
        latest_version = DEFAULT_RAY_VERSION
    return latest_version


def get_ray_and_py_version_for_default_cluster_env() -> Tuple[str, str]:
    py_version = "".join(str(x) for x in sys.version_info[0:2])
    try:
        import ray  # noqa: PLC0415 - codex_reason("gpt5.2", "optional Ray dependency for local version check")

        ray_version = ray.__version__
        if version.parse(ray_version) < version.parse(MINIMUM_RAY_VERSION):
            raise ValueError(
                f"No default cluster env for Ray version {ray_version}. Please upgrade "
                f'to a version >= {MINIMUM_RAY_VERSION} with `pip install "ray>={MINIMUM_RAY_VERSION}"`.'
            )
        if "dev0" in ray_version:
            raise ValueError(
                f"Your locally installed Ray version is {ray_version}. "
                "There is no default cluster environments for nightly versions of Ray."
            )
    except ImportError:
        # Use latest Ray version if Ray not locally installed, because Anyscale cluster envs for
        # new Ray versions are available before the open source package is released on PYPI.
        ray_version = get_latest_ray_version()
        log.debug(
            f"Ray is not installed locally. Using latest Ray version {ray_version} for "
            "the cluster env."
        )

    return ray_version, py_version


def validate_job_config_dict(
    config_dict: Dict[str, Any], api_client: ProductApi
) -> None:
    """
    Throws an exception if there are invalid values in the config dict.
    """
    compute_config: Optional[Union[ComputeTemplate, DecoratedComputeTemplate]] = None
    if "compute_config" in config_dict and isinstance(
        config_dict["compute_config"], str
    ):
        compute_config = get_cluster_compute_from_name(
            config_dict["compute_config"],
            api_client,
            cloud_name=config_dict.get("cloud"),
        )
    elif "compute_config_id" in config_dict:
        cluster_compute_id = config_dict["compute_config_id"]
        compute_config = api_client.get_compute_template_api_v2_compute_templates_template_id_get(
            cluster_compute_id
        ).result

    if compute_config and compute_config.archived_at:
        raise click.ClickException(
            "This job is using an archived compute config. To submit this job, use another compute config."
        )


# User-facing job states for --v2 mode (maps to multiple backend HaJobStates)
ALLOWED_USER_JOB_STATES = ["STARTING", "RUNNING", "SUCCEEDED", "FAILED"]


def validate_list_jobs_state_filter(_, param, value) -> List[str]:  # noqa: ARG001
    """
    Validate the job state filter for list jobs CLI method.

    Accepts both user-facing states (STARTING, RUNNING, SUCCEEDED, FAILED) for --v2
    and backend HaJobStates (SUCCESS, PENDING, etc.) for legacy compatibility.
    """
    if not value:
        return []
    for each_value in value:
        upper = each_value.upper()
        # Accept both user-facing states (for v2) and backend states (for legacy)
        if (
            upper not in ALLOWED_USER_JOB_STATES
            and upper not in HaJobStates.allowable_values
        ):
            raise click.ClickException(
                f"{each_value} is not valid for {param.opts[0]}. "
                f"For --v2 use: {', '.join(ALLOWED_USER_JOB_STATES)}"
            )
    return [each_value.upper() for each_value in value]


def get_cluster_model_for_current_workspace(
    anyscale_api_client: BaseApi,
) -> Optional[Cluster]:
    """If run within a workspace, returns the `Cluster` info for that workspace.

    Else, returns `None`.
    """
    session_id = os.environ.get("ANYSCALE_SESSION_ID")
    if not session_id or not is_anyscale_workspace():
        return None

    return anyscale_api_client.get_cluster(session_id).result


def populate_unspecified_cluster_configs(
    config: Dict[str, Any], workspace_cluster: Cluster, *, populate_name: bool = False
):
    """Populates unspecified fields in the config from a workspace cluster model.

    This is used to fill in smart defaults when deploying from a workspace.

    Fields that are defaulted:
        - `project_id` if it is not specified.
        - `build_id` if none of {`build_id, `cluster_env`} are specified.
        - `compute_config_id` if none of {`cloud`, `compute_config`, `compute_config_id`} are specified.
        - `name` if `populate_name` is passed and it is not specified.

    The defaulted name will be the name of the wor
    """
    if "project_id" not in config and "project" not in config:
        config["project_id"] = workspace_cluster.project_id

    if "build_id" not in config and "cluster_env" not in config:
        config["build_id"] = workspace_cluster.cluster_environment_build_id

    if (
        "compute_config" not in config
        and "compute_config_id" not in config
        and "cloud" not in config
    ):
        config["compute_config_id"] = workspace_cluster.cluster_compute_id

    if populate_name and "name" not in config:
        name = workspace_cluster.name
        # All workspace cluster names should start with this prefix.
        # Defensively default to the name as-is if they don't.
        if name.startswith("workspace-cluster-"):
            name = name[len("workspace-cluster-") :]

        config["name"] = name

    return config


def populate_unspecified_cluster_configs_from_current_workspace(
    config: Dict[str, Any], anyscale_api_client: BaseApi, *, populate_name: bool = False
) -> Dict[str, Any]:
    workspace_cluster = get_cluster_model_for_current_workspace(anyscale_api_client)
    if workspace_cluster is not None:
        config = populate_unspecified_cluster_configs(
            config, workspace_cluster, populate_name=populate_name
        )

    return config


def filter_actions_from_policy_document(
    policy_document: Dict[Any, Any],
    action_filter: Optional[Callable[[Dict], bool]] = None,
) -> set:
    if action_filter is None:
        action_filter = lambda statement: statement["Effect"] == "Allow"  # noqa: E731
    return {
        action
        for statement in _coerce_to_list(policy_document.get("Statement", {}))
        for action in _coerce_to_list(statement.get("Action"))
        if action_filter(statement)
    }


def filter_actions_associated_with_role(
    boto3_session: boto3.Session,
    role: Boto3Resource,
    action_filter: Optional[Callable[[Dict], bool]] = None,
) -> Set[str]:
    iam = boto3_session.resource("iam")
    attached_policy_documents = [
        iam.PolicyVersion(policy.arn, policy.default_version_id).document
        for policy in role.attached_policies.all()  # type: ignore
    ]

    role_policy_documents = [policy.policy_document for policy in role.policies.all()]  # type: ignore

    list_of_allow_actions_sets = [
        filter_actions_from_policy_document(policy_document, action_filter)
        for policy_document in role_policy_documents + attached_policy_documents
    ]
    return (
        set.union(*list_of_allow_actions_sets) if list_of_allow_actions_sets else set()
    )


def contains_control_plane_role(
    assume_role_policy_document: Dict[str, Any], anyscale_aws_account: str
) -> bool:
    def action_filter(statement: dict):
        if not statement:
            return False
        # Ensure it is `Allow` & `sts:AssumeRole`
        if (
            statement["Effect"] != "Allow"
            or statement.get("Action") != "sts:AssumeRole"
        ):
            return False

        expected_accounts = {
            f"arn:aws:iam::{anyscale_aws_account}:root",
            anyscale_aws_account,
        }
        return any(
            aws_principal in expected_accounts
            for aws_principal in _coerce_to_list(statement["Principal"].get("AWS"))
        )

    return (
        len(
            filter_actions_from_policy_document(
                assume_role_policy_document, action_filter
            )
        )
        > 0
    )


def verify_data_plane_role_assume_role_policy(
    assume_role_policy_document: Dict[str, Any]
) -> bool:
    def action_filter(statement: dict):
        # Ensure it is `Allow` & `sts:AssumeRole`
        if not statement:
            return False
        if (
            statement["Effect"] != "Allow"
            or statement.get("Action") != "sts:AssumeRole"
        ):
            return False

        return any(
            service == "ec2.amazonaws.com"
            for service in _coerce_to_list(statement["Principal"].get("Service"))
        )

    return (
        len(
            filter_actions_from_policy_document(
                assume_role_policy_document, action_filter
            )
        )
        > 0
    )


T = TypeVar("T")


def _coerce_to_list(maybe_list: Union[T, List[T]]) -> List[T]:
    return maybe_list if isinstance(maybe_list, list) else [maybe_list]


def allow_optional_file_storage(api_client: Optional[ProductApi] = None) -> bool:
    if api_client is None:
        api_client = get_auth_api_client().api_client
    return api_client.check_is_feature_flag_on_api_v2_userinfo_check_is_feature_flag_on_get(
        "allow-optional-file-storage"
    ).result.is_on
