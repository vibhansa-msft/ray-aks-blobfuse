from functools import partial
import importlib
import re
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from click import ClickException
from google.api_core.exceptions import Forbidden, NotFound, PermissionDenied
import google.auth
from google.auth.credentials import Credentials, CredentialsWithQuotaProject
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.auth.transport.requests import Request
import google.cloud
from google.iam.v1.policy_pb2 import Binding
from google.oauth2 import service_account
from googleapiclient.discovery import build as api_client_build
from googleapiclient.errors import HttpError

from anyscale.cli_logger import CloudSetupLogger
from anyscale.client.openapi_client.models.cloud_analytics_event_cloud_resource import (
    CloudAnalyticsEventCloudResource,
)
from anyscale.client.openapi_client.models.gcp_file_store_config import (
    GCPFileStoreConfig,
)
from anyscale.client.openapi_client.models.gcp_memorystore_instance_config import (
    GCPMemorystoreInstanceConfig,
)
from anyscale.util import confirm
from anyscale.utils.cloud_utils import CloudSetupError


GCP_REQUIRED_APIS = [
    "compute.googleapis.com",  # Compute Engine
    "file.googleapis.com",  # Filestore
    "storage-component.googleapis.com",  # Cloud Storage
    "storage.googleapis.com",  # Cloud Storage
    "deploymentmanager.googleapis.com",  # Deployment Manager
    "cloudresourcemanager.googleapis.com",  # Resource Manager
    "certificatemanager.googleapis.com",  # Certificate Manager for Services
]

GCP_RESOURCE_DICT: Dict[str, CloudAnalyticsEventCloudResource] = {
    "VPC": CloudAnalyticsEventCloudResource(CloudAnalyticsEventCloudResource.GCP_VPC),
    "Subnet": CloudAnalyticsEventCloudResource(
        CloudAnalyticsEventCloudResource.GCP_SUBNET
    ),
    "Project": CloudAnalyticsEventCloudResource(
        CloudAnalyticsEventCloudResource.GCP_PROJECT
    ),
    "Anyscale Access Service Account": CloudAnalyticsEventCloudResource(
        CloudAnalyticsEventCloudResource.GCP_SERVICE_ACCOUNT
    ),
    "Dataplane Service Account": CloudAnalyticsEventCloudResource(
        CloudAnalyticsEventCloudResource.GCP_SERVICE_ACCOUNT
    ),
    "Firewall Policy": CloudAnalyticsEventCloudResource(
        CloudAnalyticsEventCloudResource.GCP_FIREWALL_POLICY
    ),
    "Filestore": CloudAnalyticsEventCloudResource(
        CloudAnalyticsEventCloudResource.GCP_FILESTORE
    ),
    "Google Cloud Storage Bucket": CloudAnalyticsEventCloudResource(
        CloudAnalyticsEventCloudResource.GCP_STORAGE_BUCKET
    ),
    "Memorystore": CloudAnalyticsEventCloudResource(
        CloudAnalyticsEventCloudResource.GCP_MEMORYSTORE
    ),
}


class GCPLogger:
    def __init__(
        self, logger: CloudSetupLogger, project_id: str, spinner: Any, yes: bool = False
    ):
        self.internal = logger
        self.project_id = project_id
        self.spinner = spinner
        self.yes = yes

    def log_resource_not_found_error(
        self, resource_name: str, resource_id: str, project_id: Optional[str] = None,
    ):
        if resource_name == "Project":
            self.internal.error(
                f"Could not find {resource_name} with id {resource_id}. Please validate that you're using the correct GCP project."
            )
        else:
            project_id = project_id or self.project_id
            self.internal.error(
                f"Could not find {resource_name} with id {resource_id} in project {project_id}. Please validate that you're using the correct GCP project and that the resource values are correct."
            )
        resource = GCP_RESOURCE_DICT.get(resource_name)
        if resource:
            self.internal.log_resource_error(
                resource, CloudSetupError.RESOURCE_NOT_FOUND
            )

    def confirm_missing_permission(self, error_str: str):
        self.internal.error(error_str)
        self.spinner.stop()
        confirm(
            "If the service account has these required permissions granted via other roles\n"
            "or via a group, press 'y' to continue with verification or 'N' to abort.",
            yes=self.yes,
        )
        self.spinner.start()


class GoogleCloudClientFactory:
    """Factory to generate both Google Cloud Client libraries & Google API Client libraries.

    Google Cloud Client libraries are instantiated by:
    ```
        factory = GoogleCloudClientFactory(credentials=AnonymousCredentials())
        client = factory.compute_v1.ExampleClient()
    ```

    Google API Client libraries are instantiated by:
    ```
        factory = GoogleCloudClientFactory(credentials=AnonymousCredentials())
        client = factory.build("iam", "v1")
    ```
    """

    def __init__(self, credentials: Credentials, force_rest=False, **kwargs):
        kwargs["credentials"] = credentials
        self.kwargs = kwargs
        self.force_rest = force_rest

    def __getattr__(self, client_library: str):
        """Get a wrapped Google Cloud Client library that injects default values from the factory."""
        if not hasattr(google.cloud, client_library):
            importlib.import_module(f"google.cloud.{client_library}")
        module = getattr(google.cloud, client_library)
        kwargs = self.kwargs.copy()

        # NOTE the `storage` library only supports HTTP, but doesn't
        # have the `transport` argument in its signature.
        if self.force_rest and client_library != "storage":
            kwargs["transport"] = "rest"

        class WrappedClient:
            def __getattr__(self, client_type: str):
                return partial(getattr(module, client_type), **kwargs)

        return WrappedClient()

    def build(self, service_name: str, version: str):
        """Return a Google API Client with default values from the factor"""
        return api_client_build(
            service_name, version, cache_discovery=False, **self.kwargs
        )


def get_application_default_credentials(
    logger: CloudSetupLogger,
) -> Tuple[Credentials, Optional[str]]:
    """Get application default credentials, or run `gcloud` to try to log in."""
    try:
        credentials, project = google.auth.default(
            default_scopes="https://www.googleapis.com/auth/cloud-platform"
        )
        # Don't try to refresh credentials if they are from a Service Account Key File
        if isinstance(credentials, service_account.Credentials):
            return credentials, project
        try:
            credentials.refresh(Request())
            return credentials, project
        except RefreshError:
            logger.warning(
                "Reauthentication is needed, running `gcloud auth application-default login`"
            )
    except DefaultCredentialsError:
        logger.warning(
            "Could not automatically determine Google Application Default Credentials, trying to authenticate via GCloud"
        )
    auth_login = subprocess.run(
        ["gcloud", "auth", "application-default", "login"], check=False
    )
    if auth_login.returncode != 0:
        raise RuntimeError("Failed to authenticate via gcloud")

    return google.auth.default()


def get_google_cloud_client_factory(logger: CloudSetupLogger, project_id: str):
    credentials, credentials_project = get_application_default_credentials(logger)

    # Add a trivial call to the API to verify that the credentials are valid
    try:
        factory = GoogleCloudClientFactory(credentials=credentials)
        factory.build("serviceusage", "v1").services().get(
            name=f"projects/{project_id}/services/serviceusage.googleapis.com"
        ).execute()
    except HttpError as e:
        logger.error(e)
        raise ClickException(
            f"Failed to get project {project_id}. \n"
            "Please make sure:\n"
            "1) the project exists\n"
            "2) the account you're using has the `owner` role on the project\n"
            "3) the credentials are valid (run `gcloud auth application-default login` to reauthenticate)"
        )

    # swtich quota project to given project
    if credentials_project != project_id:
        logger.info(
            f"Default credentials are for {credentials_project}, but this cloud is being configured for {project_id}.\n"
            f"Switching quota project to {project_id}"
        )
        if isinstance(credentials, CredentialsWithQuotaProject):
            credentials = credentials.with_quota_project(project_id)
            factory = GoogleCloudClientFactory(credentials=credentials)

    return factory


def binding_from_dictionary(
    inp: List[Dict[str, Union[List[str], str]]]
) -> List[Binding]:
    return [Binding(role=b["role"], members=b["members"]) for b in inp]


def check_policy_bindings(
    iam_policy: List[Binding], member: str, possible_roles: Set[str]
) -> bool:
    """Checks if `member` has any role in `possible_roles` given the specified iam_policy."""
    return any(
        policy.role in possible_roles and member in policy.members
        for policy in iam_policy
    )


def check_required_policy_bindings(
    iam_policy: List[Binding], member: str, required_roles: Set[str]
) -> bool:
    """Checks if `member` has all roles in `required_roles` given the specified iam_policy."""
    granted_roles = {policy.role for policy in iam_policy if member in policy.members}
    return required_roles.issubset(granted_roles)


def get_gcp_filestore_config(
    factory: GoogleCloudClientFactory,
    project_id: str,
    vpc_name: str,
    filestore_location: str,
    filestore_instance_id: str,
    logger: CloudSetupLogger,
):
    instance_name = "projects/{}/locations/{}/instances/{}".format(
        project_id, filestore_location, filestore_instance_id
    )
    return get_gcp_filestore_config_from_full_name(
        factory=factory, vpc_name=vpc_name, instance_name=instance_name, logger=logger,
    )


def get_gcp_filestore_config_from_full_name(
    factory: GoogleCloudClientFactory,
    vpc_name: str,
    instance_name: str,
    logger: CloudSetupLogger,
):
    if not re.search("projects/.+/locations/.+/instances/.+", instance_name):
        raise ValueError(
            "Please provide the full filestore instance name. Example: projects/<project number>/locations/<location>/instances/<instance id>"
        )

    client = factory.filestore_v1.CloudFilestoreManagerClient()
    try:
        file_store = client.get_instance(name=instance_name)
    except NotFound as e:
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_FILESTORE,
            CloudSetupError.RESOURCE_NOT_FOUND,
        )
        raise ClickException(
            f"Could not find Filestore with id {instance_name}. Please validate that you're using the correct GCP project and that the resource values are correct. Error details: {e}"
        )
    root_dir = file_store.file_shares[0].name
    for v in file_store.networks:
        # v.network can be <vpc_name> or projects/<project_number>/global/networks/<vpc_name>
        network_name = v.network.split("/")[-1]
        if vpc_name == network_name:
            mount_target_ip = v.ip_addresses[0]
            break
    else:
        logger.error(
            f"Filestore {instance_name} is not connected to {vpc_name}, but to {[v.network for v in file_store.networks]}. "
            f"This cannot be edited on an existing Filestore instance. Please recreate the filestore and connect it to {vpc_name}."
        )
        logger.log_resource_error(
            CloudAnalyticsEventCloudResource.GCP_FILESTORE,
            CloudSetupError.FILESTORE_NOT_CONNECTED_TO_VPC,
        )
        raise ClickException(
            f"Filestore {instance_name} is not connected to {vpc_name}."
        )
    return GCPFileStoreConfig(
        instance_name=instance_name, root_dir=root_dir, mount_target_ip=mount_target_ip,
    )


def get_filestore_location_and_instance_id(
    gcp_filestore_config: GCPFileStoreConfig,
) -> Tuple[str, str]:
    instance_name = gcp_filestore_config.instance_name
    # instance name follows format of "projects/{}/locations/{}/instances/{}"
    pattern = (
        r"projects/[^/]+/locations/(?P<location>[^/]+)/instances/(?P<instance_id>[^/]+)"
    )
    match = re.match(pattern, instance_name)

    if match:
        filestore_location = match.group("location")
        filestore_instance_id = match.group("instance_id")
        return filestore_location, filestore_instance_id
    else:
        raise ClickException(
            f"Could not parse Filestore instance name {instance_name}."
        )


def get_gcp_memorystore_config(
    factory: GoogleCloudClientFactory, instance_name: Optional[str]
) -> Optional[GCPMemorystoreInstanceConfig]:
    """Get the Memorystore instance config from GCP
    Returns None if instance_name is None
    """
    if not instance_name:
        return None

    client = factory.redis_v1.CloudRedisClient()

    try:
        instance = client.get_instance(name=instance_name)
    except (NotFound, PermissionDenied, Forbidden) as e:
        raise ClickException(
            f"Error occurred when trying to access the memorystore instance {instance_name}: {e}.\nPlease validate that you're using the correct GCP project and that the resource values are correct."
        )
    return GCPMemorystoreInstanceConfig(
        name=instance_name, endpoint=instance.host + ":" + str(instance.port),
    )
