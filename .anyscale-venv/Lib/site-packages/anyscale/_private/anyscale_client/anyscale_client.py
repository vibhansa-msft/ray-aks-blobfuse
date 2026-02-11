from abc import ABC, abstractmethod
import contextlib
from datetime import datetime
from functools import wraps
import io
import json
import logging
import os
import pathlib
import re
import time
from typing import Any, Callable, Dict, Generator, IO, List, Optional, Tuple
from urllib.parse import urlparse

from openapi_client.exceptions import ApiException
import requests
from rich.style import Style
import smart_open

from anyscale._private.anyscale_client.common import (
    AnyscaleClientInterface,
    DEFAULT_PYTHON_VERSION,
    DEFAULT_RAY_VERSION,
    RUNTIME_ENV_PACKAGE_FORMAT,
)
from anyscale._private.models.image_uri import ImageURI
from anyscale._private.utils.progress_util import FileDownloadProgress
from anyscale.api_utils.logs_util import _download_log_from_s3_url_sync
from anyscale.authenticate import AuthenticationBlock, get_auth_api_client
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.api.default_api import DefaultApi as InternalApi
from anyscale.client.openapi_client.models import (
    AdminCreatedUser,
    AdminCreateUser,
    AnyscaleServiceAccount,
    ApiKeyParameters,
    ApplyProductionServiceMultiVersionV2Model,
    ArchiveStatus,
    Cloud,
    CloudDataBucketAccessMode,
    CloudDataBucketFileType,
    CloudDataBucketPresignedUrlRequest,
    CloudDataBucketPresignedUrlResponse,
    CloudDataBucketPresignedUrlScheme,
    CloudNameOptions,
    ClusteroperationResponse,
    CollaboratorType,
    ComputeTemplate,
    ComputeTemplateConfig,
    ComputeTemplateQuery,
    CreateCloudCollaborator,
    CreateComputeTemplate,
    CreateExperimentalWorkspace,
    CreateInternalProductionJob,
    CreateOrganizationInvitation,
    CreateResourceQuota,
    CreateUserProjectCollaborator,
    DecoratedCloudResource,
    DecoratedComputeTemplate,
    DecoratedjobqueueListResponse,
    DecoratedlistserviceapimodelListResponse,
    DecoratedproductionjobListResponse,
    DecoratedProductionServiceV2APIModel,
    DecoratedProductionServiceV2VersionAPIModel,
    DecoratedSession,
    DeleteResourceTagsRequest,
    ExperimentalWorkspace,
    GetOrCreateBuildFromImageUriRequest,
    InternalProductionJob,
    JobQueueSortDirective,
    JobQueuesQuery,
    ListResourceQuotasQuery,
    OrganizationCollaborator,
    OrganizationcollaboratorListResponse,
    OrganizationInvitation,
    PolicyResponse,
    Project,
    ProjectBase,
    ProjectListResponse,
    ResourcepolicyitemListResponse,
    ResourceQuota,
    ResourceQuotaStatus,
    ResourceTagResourceType,
    ServerSessionToken,
    SessionSshKey,
    SessionState,
    StartSessionOptions,
    StopSessionOptions,
    SystemWorkloadName,
    UpdatePolicyRequest,
    UpsertResourceTagsRequest,
    UserGroup,
    UsergroupListResponse,
    UserInfo,
    WorkspaceDataplaneProxiedArtifacts,
    WriteProject,
)
from anyscale.client.openapi_client.models.create_schedule import CreateSchedule
from anyscale.client.openapi_client.models.databricks_connection_info import (
    DatabricksConnectionInfo,
)
from anyscale.client.openapi_client.models.decorated_application_template import (
    DecoratedApplicationTemplate,
)
from anyscale.client.openapi_client.models.decorated_job_queue import DecoratedJobQueue
from anyscale.client.openapi_client.models.decorated_schedule import DecoratedSchedule
from anyscale.client.openapi_client.models.decoratedapplicationtemplate_list_response import (
    DecoratedapplicationtemplateListResponse,
)
from anyscale.client.openapi_client.models.decoratedschedule_list_response import (
    DecoratedscheduleListResponse,
)
from anyscale.client.openapi_client.models.production_job import ProductionJob
from anyscale.client.openapi_client.models.resource_tag_record import ResourceTagRecord
from anyscale.client.openapi_client.models.update_job_queue_request import (
    UpdateJobQueueRequest,
)
from anyscale.client.openapi_client.rest import ApiException as InternalApiException
from anyscale.cluster_compute import parse_cluster_compute_name_version
from anyscale.feature_flags import FLAG_DEFAULT_WORKING_DIR_FOR_PROJ
from anyscale.sdk.anyscale_client.api.default_api import DefaultApi as ExternalApi
from anyscale.sdk.anyscale_client.models import (
    ApplyProductionServiceV2Model,
    Cluster,
    ClusterCompute,
    ClusterComputeConfig,
    ClusterComputesQuery,
    ClusterEnvironment,
    ClusterEnvironmentBuild,
    ClusterenvironmentbuildListResponse,
    ClusterEnvironmentBuildStatus,
    ClusterEnvironmentsQuery,
    CreateBYODClusterEnvironment,
    CreateBYODClusterEnvironmentConfigurationSchema,
    CreateClusterEnvironment,
    CreateClusterEnvironmentBuild,
    Job as APIJobRun,
    ProductionServiceV2VersionModel,
    Project as ProjectExternal,
    RollbackServiceModel,
    TextQuery,
)
from anyscale.sdk.anyscale_client.models.jobs_query import JobsQuery
from anyscale.sdk.anyscale_client.models.jobs_sort_field import JobsSortField
from anyscale.sdk.anyscale_client.models.page_query import PageQuery
from anyscale.sdk.anyscale_client.models.sort_by_clause_jobs_sort_field import (
    SortByClauseJobsSortField,
)
from anyscale.sdk.anyscale_client.models.sort_order import SortOrder
from anyscale.sdk.anyscale_client.rest import ApiException as ExternalApiException
from anyscale.shared_anyscale_utils.bytes_util import Bytes
from anyscale.shared_anyscale_utils.conf import ANYSCALE_HOST
from anyscale.shared_anyscale_utils.latest_ray_version import LATEST_RAY_VERSION
from anyscale.util import (
    get_cluster_model_for_current_workspace,
    get_endpoint,
    is_anyscale_workspace,
)
from anyscale.utils.connect_helpers import search_entities
from anyscale.utils.runtime_env import (
    is_workspace_dependency_tracking_disabled,
    parse_dot_env_file,
    WORKSPACE_REQUIREMENTS_FILE_PATH,
    zip_local_dir,
)
from anyscale.utils.workspace_notification import (
    WORKSPACE_NOTIFICATION_ADDRESS,
    WorkspaceNotification,
)


WORKSPACE_ID_ENV_VAR = "ANYSCALE_EXPERIMENTAL_WORKSPACE_ID"
OVERWRITE_EXISTING_CLOUD_STORAGE_FILES = (
    os.environ.get("ANYSCALE_OVERWRITE_EXISTING_CLOUD_STORAGE_FILES", "0") == "1"
)

# internal_logger is used for logging internal errors or debug messages that we do not expect users to see.
internal_logger = logging.getLogger(__name__)

# A decorator to handle ApiException and raise ValueError with the error message.
def handle_api_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ApiException, InternalApiException, ExternalApiException) as e:
            if e.status >= 400 and e.status < 500:
                try:
                    body_dict = json.loads(e.body)
                    msg = body_dict["error"]["detail"]
                    raise ValueError(msg) from None
                except (KeyError, TypeError):
                    # ApiException doesn't conform to expected format, raise original error
                    raise e from None
            raise e from None

    return wrapper


@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.

    >>> with set_env(PLUGINS_DIR='test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type environ: dict[str, unicode]
    :param environ: Environment variables to set
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


class AWSS3ClientInterface(ABC):
    @abstractmethod
    def download_fileobj(self, Bucket: str, Key: str, Fileobj: IO[Any],) -> None:
        """Download a file from an S3 bucket to a file-like object."""
        raise NotImplementedError


class GCSBlobInterface(ABC):
    @abstractmethod
    def download_to_file(self, fileobj: IO[Any]) -> None:
        """Download the blob to a file-like object."""
        raise NotImplementedError

    @abstractmethod
    def exists(self) -> bool:
        """Check if the blob exists."""
        raise NotImplementedError


class GCSBucketInterface(ABC):
    @abstractmethod
    def blob(self, object_name: str) -> GCSBlobInterface:
        """Get a blob object for the given object name."""
        raise NotImplementedError


class GCPGCSClientInterface(ABC):
    @abstractmethod
    def bucket(self, bucket: str) -> GCSBucketInterface:
        """Get a bucket object for the given bucket name."""
        raise NotImplementedError


class AnyscaleClient(AnyscaleClientInterface):
    # Number of entries to fetch per request for list endpoints.
    LIST_ENDPOINT_COUNT = 50

    def __init__(
        self,
        *,
        api_clients: Optional[Tuple[ExternalApi, InternalApi]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        workspace_requirements_file_path: str = WORKSPACE_REQUIREMENTS_FILE_PATH,
        logger: Optional[BlockLogger] = None,
        host: Optional[str] = None,
        s3_client: Optional[AWSS3ClientInterface] = None,
        gcs_client: Optional[GCPGCSClientInterface] = None,
    ):
        if api_clients is None:
            auth_block: AuthenticationBlock = get_auth_api_client(
                raise_structured_exception=True
            )
            api_clients = (auth_block.anyscale_api_client, auth_block.api_client)

        self._external_api_client, self._internal_api_client = api_clients
        self._workspace_requirements_file_path = workspace_requirements_file_path
        self._sleep = sleep or time.sleep
        self._s3_client = s3_client
        self._gcs_client = gcs_client

        # Cached IDs and models to avoid duplicate lookups.
        self._default_project_id_from_cloud_id: Dict[Optional[str], str] = {}
        self._cloud_id_cache: Dict[Optional[str], str] = {}
        self._cluster_env_build_cache: Dict[str, ClusterEnvironmentBuild] = {}
        self._current_workspace_cluster: Optional[Cluster] = None
        self._logger = logger or BlockLogger()
        self._host = host or ANYSCALE_HOST

    @property
    def s3_client(self) -> AWSS3ClientInterface:
        if self._s3_client is None:
            # initialize the s3 client lazily so that we import the boto3 library only when needed.
            try:
                import boto3  # noqa: PLC0415 - codex_reason("gpt5.2", "optional AWS SDK dependency")
                import botocore.config  # noqa: PLC0415 - codex_reason("gpt5.2", "optional AWS SDK dependency")
            except ImportError:
                raise RuntimeError(
                    "Could not import the Amazon S3 Python API via `import boto3`.  Please check your installation or try running `pip install boto3`."
                )
            self._s3_client = boto3.client(  # type: ignore
                "s3", config=botocore.config.Config(signature_version="s3v4")
            )
        return self._s3_client  # type: ignore

    @property
    def gcs_client(self) -> GCPGCSClientInterface:
        if self._gcs_client is None:
            # initialize the gcs client lazily so that we import the google cloud storage library only when needed.
            try:
                from google.cloud import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP storage dependency")
                    storage,
                )
            except ImportError:
                raise RuntimeError(
                    "Could not import the Google Storage Python API via `from google.cloud import storage`.  Please check your installation or try running `pip install --upgrade google-cloud-storage`."
                )
            self._gcs_client = storage.Client()
        return self._gcs_client  # type: ignore

    @property
    def host(self) -> str:
        return self._host

    @property
    def logger(self) -> BlockLogger:
        return self._logger

    def get_job_ui_url(self, job_id: str) -> str:
        return get_endpoint(f"/jobs/{job_id}", host=self.host)

    def get_service_ui_url(self, service_id: str) -> str:
        return get_endpoint(f"/services/{service_id}", host=self.host)

    def get_compute_config_ui_url(
        self, compute_config_id: str, *, cloud_id: str
    ) -> str:
        return get_endpoint(
            f"/v2/{cloud_id}/compute-configs/{compute_config_id}", host=self.host
        )

    def get_build_ui_url(self, cluster_env_id, build_id: str) -> str:
        return get_endpoint(
            f"v2/container-images/{cluster_env_id}/versions/{build_id}", host=self.host
        )

    def get_current_workspace_id(self) -> Optional[str]:
        return os.environ.get(WORKSPACE_ID_ENV_VAR, None)

    def inside_workspace(self) -> bool:
        return self.get_current_workspace_id() is not None

    def get_workspace_requirements_path(self) -> Optional[str]:
        if (
            not self.inside_workspace()
            or is_workspace_dependency_tracking_disabled()
            or not pathlib.Path(self._workspace_requirements_file_path).is_file()
        ):
            return None

        return self._workspace_requirements_file_path

    def _download_file_from_google_cloud_storage(
        self, bucket: str, object_name: str
    ) -> Optional[bytes]:
        try:
            bucket_obj = self.gcs_client.bucket(bucket)
            blob = bucket_obj.blob(object_name)
            fileobj = io.BytesIO()
            if blob.exists():
                blob.download_to_file(fileobj)
                return fileobj.getvalue()
            return None
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to download the working directory from Google Cloud Storage. Error {e!r}"
                "Please validate you have exported cloud credentials with the correct read permissions and the intended bucket exists in your Cloud Storage account. "
            ) from e

    def _download_file_from_s3(self, bucket: str, object_key: str) -> Optional[bytes]:
        try:
            from botocore.exceptions import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional AWS SDK dependency")
                ClientError,
            )
        except Exception:  # noqa: BLE001
            raise RuntimeError(
                "Could not download file from S3: Could not import the Amazon S3 Python API via `import boto3`.  Please check your installation or try running `pip install boto3`."
            )
        try:
            fileobj = io.BytesIO()
            self.s3_client.download_fileobj(bucket, object_key, fileobj)
            return fileobj.getvalue()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None
            raise
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to download the working directory from S3. Error {e!r}"
                "Please validate you have exported cloud credentials with the correct read permissions and the intended bucket exists in your S3 account. "
            ) from e

    def _download_file_from_remote_storage(self, remote_uri: str) -> Optional[bytes]:
        parsed_uri = urlparse(remote_uri)
        service = parsed_uri.scheme
        bucket = parsed_uri.netloc
        object_name = parsed_uri.path.lstrip("/")
        if service == "s3":
            return self._download_file_from_s3(bucket, object_name)
        if service == "gs":
            return self._download_file_from_google_cloud_storage(bucket, object_name)
        return None

    def get_workspace_env_vars(self) -> Optional[Dict[str, str]]:
        system_storage_path = os.environ.get("ANYSCALE_INTERNAL_SYSTEM_STORAGE", "")
        workspace_id = os.environ.get(WORKSPACE_ID_ENV_VAR, "")
        workspace_artifact_path = (
            os.path.join(
                system_storage_path, "workspace_tracking_dependencies", workspace_id,
            )
            if workspace_id and system_storage_path
            else ""
        )
        workspace_dot_env_path = (
            os.path.join(workspace_artifact_path, ".env")
            if workspace_artifact_path
            else ""
        )

        if not self.inside_workspace() or not workspace_dot_env_path:
            return None

        dot_env = self._download_file_from_remote_storage(workspace_dot_env_path)
        if dot_env:
            parsed_dot_env = parse_dot_env_file(dot_env)
            if parsed_dot_env:
                self.logger.info(
                    f"Using workspace runtime dependencies env vars: {parsed_dot_env}."
                )
            return parsed_dot_env
        return None

    @handle_api_exceptions
    def get_current_workspace_cluster(self) -> Optional[Cluster]:
        # Checks for the existence of the ANYSCALE_EXPERIMENTAL_WORKSPACE_ID env var.
        if not is_anyscale_workspace():
            return None

        if self._current_workspace_cluster is None:
            # Picks up the cluster ID from the ANYSCALE_SESSION_ID env var.
            self._current_workspace_cluster = get_cluster_model_for_current_workspace(
                self._external_api_client
            )

        return self._current_workspace_cluster

    def _get_project_id_by_name(
        self, *, parent_cloud_id: Optional[str] = None, name: Optional[str] = None
    ) -> str:
        """Resolves `name`, a project name, to a project ID.

        Args:
            parent_cloud_id: If specified, return the project that has this `parent_cloud_id`. \
                Else (`None`), return the first project with the given `name`. \
                Defaults to `None`.
            name: The name of the project.
        """
        # First try to find project by name and parent cloud id
        if parent_cloud_id is not None and name is not None:
            response = self._external_api_client.search_projects(
                {
                    "name": {"equals": name,},
                    "parent_cloud_id": {"equals": parent_cloud_id,},
                }
            )
            if response.results:
                # Project name is unique within a cloud, so we can just return the first result
                return response.results[0].id

        # Then find if project with name already exists
        matching_projects = self._internal_api_client.find_project_by_project_name_api_v2_projects_find_by_name_get(
            name
        ).results
        if len(matching_projects) == 0:
            raise ValueError(f"Project '{name}' was not found.")
        else:
            for project in matching_projects:
                if (
                    parent_cloud_id is None
                    or project.parent_cloud_id == parent_cloud_id
                ):
                    return project.id
            raise ValueError(
                f"{len(matching_projects)} project(s) found with name '{name}' and none matched cloud_id '{parent_cloud_id}'"
            )

    def _get_project_id_by_cloud_id(
        self, *, parent_cloud_id: Optional[str] = None,
    ) -> str:
        workspace_cluster = self.get_current_workspace_cluster()
        if workspace_cluster is not None:
            if (
                workspace_cluster.cluster_compute_config is not None
                and workspace_cluster.cluster_compute_config.cloud_id == parent_cloud_id
            ):
                return workspace_cluster.project_id
            elif workspace_cluster.cluster_compute_id is not None:
                workspace_cluster_compute = self.get_compute_config(
                    workspace_cluster.cluster_compute_id
                )
                if (
                    workspace_cluster_compute is not None
                    and workspace_cluster_compute.config is not None
                    and workspace_cluster_compute.config.cloud_id == parent_cloud_id
                ):
                    return workspace_cluster.project_id

        if self._default_project_id_from_cloud_id.get(parent_cloud_id) is None:
            # Cloud isolation organizations follow the permissions model in https://docs.anyscale.com/organization-and-user-account/access-controls
            default_project: ProjectExternal = self._external_api_client.get_default_project(
                parent_cloud_id=parent_cloud_id
            ).result
            self._default_project_id_from_cloud_id[parent_cloud_id] = default_project.id

        return self._default_project_id_from_cloud_id[parent_cloud_id]

    @handle_api_exceptions
    def get_project_id(
        self, *, parent_cloud_id: Optional[str] = None, name: Optional[str] = None
    ) -> str:
        if name is not None:
            return self._get_project_id_by_name(
                parent_cloud_id=parent_cloud_id, name=name
            )
        else:
            return self._get_project_id_by_cloud_id(parent_cloud_id=parent_cloud_id)

    def _get_cloud_id_for_compute_config_id(self, compute_config_id: str) -> str:
        cluster_compute: ClusterCompute = self._external_api_client.get_cluster_compute(
            compute_config_id
        ).result
        cluster_compute_config: ClusterComputeConfig = cluster_compute.config
        return cluster_compute_config.cloud_id

    def _get_cloud_id_by_name(self, cloud_name: str) -> Optional[str]:
        try:
            return self._internal_api_client.find_cloud_by_name_api_v2_clouds_find_by_name_post(
                CloudNameOptions(name=cloud_name),
            ).result.id
        except InternalApiException as e:
            if e.status == 404:
                return None

            raise e from None

    @handle_api_exceptions
    def get_cloud_id(
        self, cloud_name: Optional[str] = None, compute_config_id: Optional[str] = None
    ) -> str:
        if cloud_name is not None and compute_config_id is not None:
            raise ValueError(
                "Only one of cloud_name or compute_config_id should be provided."
            )

        if compute_config_id is not None:
            return self._get_cloud_id_for_compute_config_id(compute_config_id)

        if cloud_name in self._cloud_id_cache:
            return self._cloud_id_cache[cloud_name]

        if cloud_name is not None:
            cloud_id = self._get_cloud_id_by_name(cloud_name)
            if cloud_id is None:
                raise RuntimeError(f"Cloud '{cloud_name}' not found.")
        elif self.inside_workspace():
            workspace_cluster = self.get_current_workspace_cluster()
            assert workspace_cluster is not None
            # NOTE(edoakes): the Cluster model has a compute_config_config model that includes
            # its cloud ID, but it's not always populated.
            # TODO(edoakes): add cloud_id to the Cluster model to avoid a second RTT.
            cloud_id = self._get_cloud_id_for_compute_config_id(
                workspace_cluster.cluster_compute_id
            )
        else:
            cloud_id = self._external_api_client.get_default_cloud().result.id

        assert cloud_id is not None
        self._cloud_id_cache[cloud_name] = cloud_id
        return cloud_id

    @handle_api_exceptions
    def get_cloud(self, *, cloud_id: str) -> Optional[Cloud]:
        try:
            cloud: Cloud = self._external_api_client.get_cloud(cloud_id=cloud_id).result
            return cloud
        except InternalApiException as e:
            if e.status == 404:
                return None

            raise e from None

    @handle_api_exceptions
    def get_cloud_by_name(self, *, name) -> Optional[Cloud]:
        cloud_id = self.get_cloud_id(cloud_name=name)
        return self.get_cloud(cloud_id=cloud_id)

    @handle_api_exceptions
    def get_cloud_resource_by_name(
        self, cloud_id: str, cloud_resource_name: str
    ) -> Optional[DecoratedCloudResource]:
        return self._internal_api_client.find_cloud_resource_by_name_api_v2_clouds_cloud_id_find_cloud_resource_by_name_post(
            cloud_id=cloud_id, cloud_resource_name=cloud_resource_name,
        ).result

    @handle_api_exceptions
    def get_default_cloud(self) -> Optional[Cloud]:
        try:
            return self._external_api_client.get_default_cloud().result
        except InternalApiException as e:
            if e.status == 404:
                return None

            raise e from None

    @handle_api_exceptions
    def list_clouds(
        self, *, paging_token: Optional[str] = None, count: Optional[int] = None
    ):
        return self._internal_api_client.list_clouds_api_v2_clouds_get(
            paging_token=paging_token, count=count or self.LIST_ENDPOINT_COUNT
        )

    @handle_api_exceptions
    def add_cloud_collaborators(
        self, cloud_id: str, collaborators: List[CreateCloudCollaborator]
    ) -> None:
        self._internal_api_client.batch_create_cloud_collaborators_api_v2_clouds_cloud_id_collaborators_users_batch_create_post(
            cloud_id, collaborators
        )

    @handle_api_exceptions
    def terminate_system_cluster(self, cloud_id: str) -> ClusteroperationResponse:
        return self._internal_api_client.terminate_system_cluster_api_v2_system_workload_cloud_id_terminate_post(
            cloud_id
        )

    @handle_api_exceptions
    def describe_system_workload_get_status(self, cloud_id: str) -> str:
        res = self._internal_api_client.describe_system_workload_api_v2_system_workload_cloud_id_describe_post(
            cloud_id, SystemWorkloadName.RAY_OBS_EVENTS_API_SERVICE, start_cluster=False
        ).result
        return res.status

    @handle_api_exceptions
    def create_compute_config(
        self, config: ComputeTemplateConfig, *, name: Optional[str] = None
    ) -> Tuple[str, str]:
        result: ComputeTemplate = self._internal_api_client.create_compute_template_api_v2_compute_templates_post(
            create_compute_template=CreateComputeTemplate(
                config=config, name=name, anonymous=name is None, new_version=True
            )
        ).result
        return f"{result.name}:{result.version}", result.id

    @handle_api_exceptions
    def get_compute_config(
        self, compute_config_id: str
    ) -> Optional[DecoratedComputeTemplate]:
        try:
            return self._internal_api_client.get_compute_template_api_v2_compute_templates_template_id_get(
                compute_config_id
            ).result
        except InternalApiException as e:
            if e.status == 404:
                return None

            raise e from None

    @handle_api_exceptions
    def get_compute_config_id(
        self,
        compute_config_name: Optional[str] = None,
        cloud: Optional[str] = None,
        *,
        include_archived: bool = False,
    ) -> Optional[str]:
        if compute_config_name is not None:
            name, version = parse_cluster_compute_name_version(compute_config_name)
            if version is None:
                # Setting `version=-1` will return only the latest version if there are multiple.
                version = -1

            cloud_id = (
                self.get_cloud_id(cloud_name=cloud) if cloud is not None else None
            )

            cluster_computes = self._internal_api_client.search_compute_templates_api_v2_compute_templates_search_post(
                ComputeTemplateQuery(
                    orgwide=True,
                    name={"equals": name},
                    include_anonymous=True,
                    archive_status=ArchiveStatus.ALL
                    if include_archived
                    else ArchiveStatus.NOT_ARCHIVED,
                    version=version,
                    cloud_id=cloud_id,
                )
            ).results

            if len(cluster_computes) == 0:
                return None

            compute_template: DecoratedComputeTemplate = cluster_computes[0]
            return compute_template.id

        # If the compute config name is not provided, we pick an appropriate default.
        #
        #   - If running in a workspace:
        #       * If auto_select_worker_config enabled: we switch over to a standardized
        #         default compute config (copying over any cluster-level attributes, e.g.
        #         max-gpus).
        #       * Otherwise, we use the workspace's compute config.
        #
        #   - Otherwise, we use the default compute config provided by the API.

        workspace_cluster = self.get_current_workspace_cluster()
        if workspace_cluster is not None:
            workspace_compute_config: DecoratedComputeTemplate = self.get_compute_config(
                workspace_cluster.cluster_compute_id
            )
            workspace_config: ClusterComputeConfig = workspace_compute_config.config
            if workspace_config.auto_select_worker_config:
                standard_template = self._build_standard_compute_template_from_existing_auto_config(
                    workspace_config
                )
                _, compute_config_id = self.create_compute_config(standard_template)
                return compute_config_id
            else:
                return workspace_cluster.cluster_compute_id

        return self.get_default_compute_config(cloud_id=self.get_cloud_id()).id

    @handle_api_exceptions
    def archive_compute_config(self, *, compute_config_id):
        self._internal_api_client.archive_compute_template_api_v2_compute_templates_compute_template_id_archive_post(
            compute_config_id
        )

    @handle_api_exceptions
    def search_cluster_computes(self, query: Dict[str, Any]):
        """Search for cluster computes matching the provided query.

        Uses the external API client (SDK) to search for cluster computes.
        """

        # Convert dict to ClusterComputesQuery model
        cluster_computes_query = ClusterComputesQuery(**query)
        return self._external_api_client.search_cluster_computes(cluster_computes_query)

    @handle_api_exceptions
    def get_default_compute_config(
        self, *, cloud_id: str, cloud_resource_id: Optional[str] = None
    ) -> ClusterCompute:
        return self._external_api_client.get_default_cluster_compute(
            cloud_id=cloud_id, cloud_resource_id=cloud_resource_id
        ).result

    def _build_standard_compute_template_from_existing_auto_config(
        self, compute_config: ClusterComputeConfig
    ) -> ComputeTemplateConfig:
        """
        Build a standard compute template config from an existing compute config.

        1. Pull the default compute template config.
        2. Disable scheduling on the head node.
        3. Enable auto_select_worker_config.
        4. Copy any cluster-level flags from the provided compute config to the template.
        """
        # Retrieve the default cluster compute config for the cloud.
        default_compute_template: DecoratedComputeTemplate = self._external_api_client.get_default_cluster_compute(
            cloud_id=compute_config.cloud_id,
        ).result.config

        # Disable head node scheduling.
        if default_compute_template.head_node_type.resources is None:
            default_compute_template.head_node_type.resources = {}
        default_compute_template.head_node_type.resources["CPU"] = 0

        # Ensure auto_select_worker_config is enabled.
        default_compute_template.auto_select_worker_config = True

        # Copy flags set at a cluster level over to the workspace.
        #
        # NOTE (shomilj): If there are more attributes we want to
        # persist from the provided compute config --> the compute
        # config used for deploying the service, we should copy them
        # over here.
        default_compute_template.flags = compute_config.flags

        return default_compute_template

    @handle_api_exceptions
    def get_cluster_env_build(self, build_id: str) -> Optional[ClusterEnvironmentBuild]:
        # Check cache first
        if build_id in self._cluster_env_build_cache:
            return self._cluster_env_build_cache[build_id]

        # Fetch from API if not in cache
        try:
            res = self._external_api_client.get_cluster_environment_build(build_id)
            build = res.result
        except ExternalApiException as e:
            if e.status == 404:
                return None

            raise e from None

        # Store in cache ONLY if the build exists and is in a terminal state
        if build:
            terminal_states = {
                ClusterEnvironmentBuildStatus.SUCCEEDED,
                ClusterEnvironmentBuildStatus.FAILED,
                ClusterEnvironmentBuildStatus.CANCELED,
            }
            if build.status in terminal_states:
                self._cluster_env_build_cache[build_id] = build

        return build

    @handle_api_exceptions
    def get_cluster_env_build_image_uri(
        self, cluster_env_build_id: str, use_image_alias: bool = False
    ) -> Optional[ImageURI]:
        try:
            build = self.get_cluster_env_build(cluster_env_build_id)
            if build is None:
                return None

            cluster_env = self._external_api_client.get_cluster_environment(
                build.cluster_environment_id
            ).result
            return ImageURI.from_cluster_env_build(cluster_env, build, use_image_alias)
        except ExternalApiException as e:
            if e.status == 404:
                return None

            raise e from None

    @handle_api_exceptions
    def archive_image(self, *, image_id: str) -> None:
        """Archive an image (cluster environment) by ID."""
        self._internal_api_client.archive_cluster_environment_api_v2_application_templates_application_template_id_archive_post(
            image_id
        )

    @handle_api_exceptions
    def get_default_build_id(self) -> str:
        workspace_cluster = self.get_current_workspace_cluster()
        if workspace_cluster is not None:
            return workspace_cluster.cluster_environment_build_id
        result: ClusterEnvironmentBuild = self._external_api_client.get_default_cluster_environment_build(
            DEFAULT_PYTHON_VERSION, DEFAULT_RAY_VERSION,
        ).result
        return result.id

    @handle_api_exceptions
    def get_cluster_env_by_name(self, name: str) -> Optional[ClusterEnvironment]:
        resp = self._external_api_client.search_cluster_environments(
            ClusterEnvironmentsQuery(
                name=TextQuery(equals=name),  # pyright: ignore reportGeneralTypeIssues
                paging=PageQuery(count=1),
                include_anonymous=True,
            )
        )
        if resp.results:
            return resp.results[0]
        return None

    @handle_api_exceptions
    def get_application_template(
        self, application_template_id: str
    ) -> Optional[DecoratedApplicationTemplate]:
        try:
            return self._internal_api_client.get_application_template_api_v2_application_templates_application_template_id_get(
                application_template_id
            ).result
        except InternalApiException as e:
            if e.status == 404:
                return None
            raise e from None

    @handle_api_exceptions
    def list_application_templates(  # noqa: PLR0913
        self,
        *,
        name: Optional[str] = None,
        image_name: Optional[str] = None,
        creator_id: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
        defaults_first: bool = False,
        count: Optional[int] = None,
        paging_token: Optional[str] = None,
    ) -> DecoratedapplicationtemplateListResponse:
        project_id = None
        if project:
            project_id = self.get_project_id(name=project)

        return self._internal_api_client.list_application_templates_api_v2_application_templates_get(
            project_id=project_id,
            creator_id=creator_id,
            name_contains=name,
            image_name_contains=image_name,
            include_archived=include_archived,
            defaults_first=defaults_first,
            paging_token=paging_token,
            count=count if count is not None else self.LIST_ENDPOINT_COUNT,
        )

    @handle_api_exceptions
    def list_cluster_env_builds(
        self, cluster_env_id: str,
    ) -> Generator[ClusterEnvironmentBuild, None, None]:
        paging_token = None
        while True:
            resp: ClusterenvironmentbuildListResponse = self._external_api_client.list_cluster_environment_builds(
                cluster_environment_id=cluster_env_id,
                count=self.LIST_ENDPOINT_COUNT,
                paging_token=paging_token,
                desc=True,
            )
            for build in resp.results:
                yield build

            if resp.metadata.next_paging_token is None:
                break
            paging_token = resp.metadata.next_paging_token

    def _wait_for_build_to_succeed(
        self, build_id: str, poll_interval_seconds=3, timeout_secs=3600,
    ):
        """Periodically check the status of the build operation until it completes.
        Raise a RuntimeError if the build fails or cancelled.
        Raise a TimeoutError if the build does not complete within the timeout.
        """
        elapsed_secs = 0
        while elapsed_secs < timeout_secs:
            build = self.get_cluster_env_build(build_id)
            if build.status == ClusterEnvironmentBuildStatus.SUCCEEDED:
                self.logger.info("")
                return
            elif build.status == ClusterEnvironmentBuildStatus.FAILED:
                raise RuntimeError(f"Image build {build_id} failed.")
            elif build.status == ClusterEnvironmentBuildStatus.CANCELED:
                raise RuntimeError(f"Image build {build_id} unexpectedly cancelled.")

            elapsed_secs += poll_interval_seconds
            self.logger.info(
                f"Waiting for image build to complete. Elapsed time: {elapsed_secs} seconds.",
                end="\r",
            )
            self._sleep(poll_interval_seconds)
        raise TimeoutError(
            f"Timed out waiting for image build {build_id} to complete after {timeout_secs}s."
        )

    def _find_or_create_cluster_env(
        self,
        cluster_env_name: str,
        anonymous: bool,
        *,
        image_uri: Optional[str] = None,
        registry_login_secret: Optional[str] = None,
        ray_version: Optional[str] = None,
    ) -> ClusterEnvironment:
        """
        Find or create a cluster environment with the given name.

        There're two possible race conditions:
        1) A tries to create a cluster env with the same name as B, but B has already created it.
        -> A will get a 409 conflict error and should retry to get the existing cluster env.
        2) A and B creates two identical builds under the same cluster env. This would cause job queue to
        reject the job submission.
        -> Cluster env and BYOD build are created within the same transaction, so the latter one will fail
        with a 409 conflict error. The former one will succeed and the latter one should retry to get the
        existing cluster env.
        """
        existing_cluster_env = self.get_cluster_env_by_name(cluster_env_name)
        if existing_cluster_env is not None:
            return existing_cluster_env

        try:
            if image_uri:
                # For BYOD builds, we should create a build along with the cluster env.
                cluster_environment = self._external_api_client.create_byod_cluster_environment(
                    CreateBYODClusterEnvironment(
                        name=cluster_env_name,
                        config_json=CreateBYODClusterEnvironmentConfigurationSchema(
                            docker_image=image_uri,
                            ray_version=ray_version
                            if ray_version
                            else LATEST_RAY_VERSION,
                            registry_login_secret=registry_login_secret,
                        ),
                        anonymous=anonymous,
                    )
                ).result
            else:
                cluster_environment = self._external_api_client.create_cluster_environment(
                    CreateClusterEnvironment(name=cluster_env_name, anonymous=anonymous)
                ).result
            return cluster_environment
        except ExternalApiException as e:
            if e.status != 409:
                raise e from None
            # Retry to get the existing cluster env because it might be created by another process.
            existing_cluster_env = self.get_cluster_env_by_name(cluster_env_name)
            if existing_cluster_env is None:
                raise e from None
            return existing_cluster_env

    @handle_api_exceptions
    def get_cluster_env_build_id_from_containerfile(
        self,
        cluster_env_name: str,
        containerfile: str,
        anonymous: bool = True,
        ray_version: Optional[str] = None,
    ) -> str:
        cluster_env = self._find_or_create_cluster_env(
            cluster_env_name, anonymous=anonymous
        )
        for build in self.list_cluster_env_builds(cluster_env.id):
            if (
                build.status == ClusterEnvironmentBuildStatus.SUCCEEDED
                and build.containerfile == containerfile
                # we don't need to check the ray_version because checking the containerfile is enough.
            ):
                return build.id

        try:
            build_op = self._external_api_client.create_cluster_environment_build(
                CreateClusterEnvironmentBuild(
                    cluster_environment_id=cluster_env.id,
                    containerfile=containerfile,
                    ray_version=ray_version,  # we don't use the latest version here if ray_version is Noneb/c the backend will try to parse the base image to decide the ray version.
                )
            ).result
        except ExternalApiException as e:
            if e.status == 400:
                raise RuntimeError(
                    "Invalid containerfile. Please check the syntax and try again.",
                    e.body,
                ) from None

        build_url = self.get_build_ui_url(
            cluster_env.id, build_op.cluster_environment_build_id
        )
        self.logger.info(f"Building image. View it in the UI: {build_url}")
        self._wait_for_build_to_succeed(build_op.cluster_environment_build_id)
        self.logger.info("Image build succeeded.")

        return build_op.cluster_environment_build_id

    @handle_api_exceptions
    def get_cluster_env_build_id_from_image_uri(
        self,
        image_uri: ImageURI,
        registry_login_secret: Optional[str] = None,
        ray_version: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        build = self._internal_api_client.get_or_create_build_from_image_uri_api_v2_builds_get_or_create_build_from_image_uri_post(
            GetOrCreateBuildFromImageUriRequest(
                image_uri=str(image_uri),
                registry_login_secret=registry_login_secret,
                ray_version=ray_version,
                cluster_env_name=name,
            )
        ).result
        return build.id

    @handle_api_exceptions
    def send_workspace_notification(
        self, notification: WorkspaceNotification,
    ):
        if not self.inside_workspace():
            return

        try:
            r = requests.post(WORKSPACE_NOTIFICATION_ADDRESS, json=notification.dict())
            r.raise_for_status()
        except Exception:
            internal_logger.exception(
                "Failed to send workspace notification. "
                "This should not happen, so please contact Anyscale support."
            )

    @handle_api_exceptions
    def get_service(
        self,
        name: str,
        *,
        cloud: Optional[str],
        project: Optional[str],
        include_archived: bool = False,
    ) -> Optional[DecoratedProductionServiceV2APIModel]:
        # we don't have an api to get a service by name, so we need to list services and filter by name
        resp = self.list_services(
            name=name, cloud=cloud, project=project, include_archived=include_archived
        )
        for result in resp.results:
            if result.name == name:
                return result
        return None

    @handle_api_exceptions
    def get_service_by_id(
        self, service_id: str
    ) -> Optional[DecoratedProductionServiceV2APIModel]:
        return self._internal_api_client.get_service_api_v2_services_v2_service_id_get(
            service_id
        ).result

    @handle_api_exceptions
    def list_services(  # noqa: PLR0913
        self,
        *,
        name: Optional[str] = None,
        state_filter: Optional[List[str]] = None,
        tag_filter: Optional[List[str]] = None,
        creator_id: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
        count: Optional[int] = None,
        paging_token: Optional[str] = None,
        sort_field: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> DecoratedlistserviceapimodelListResponse:
        cloud_id = self.get_cloud_id(cloud_name=cloud) if cloud else None
        project_id = (
            self.get_project_id(parent_cloud_id=cloud_id, name=project)
            if project
            else None
        )

        return self._internal_api_client.list_services_api_v2_services_v2_get(
            project_id=project_id,
            cloud_id=cloud_id,
            name=name,
            state_filter=state_filter,
            creator_id=creator_id,
            archive_status=ArchiveStatus.ALL
            if include_archived
            else ArchiveStatus.NOT_ARCHIVED,
            tag_filter=tag_filter,
            count=count if count else self.LIST_ENDPOINT_COUNT,
            paging_token=paging_token,
            sort_field=sort_field,
            sort_order=sort_order,
        )

    def get_service_versions(
        self, service_id: str, read_all_versions: bool = False,
    ) -> List[DecoratedProductionServiceV2VersionAPIModel]:
        resp = self._internal_api_client.get_service_versions_api_v2_services_v2_service_id_versions_get(
            service_id=service_id,
        )

        if not read_all_versions:
            return resp.results

        all_versions: List[DecoratedProductionServiceV2VersionAPIModel] = []
        all_versions.extend(resp.results)
        paging_token = resp.metadata.next_paging_token

        while paging_token is not None:
            resp = self._internal_api_client.get_service_versions_api_v2_services_v2_service_id_versions_get(
                service_id=service_id,
                count=self.LIST_ENDPOINT_COUNT,
                paging_token=paging_token,
            )
            all_versions.extend(resp.results)
            paging_token = resp.metadata.next_paging_token

        return all_versions

    @handle_api_exceptions
    def get_project(self, project_id: str) -> Project:
        return self._internal_api_client.get_project_api_v2_projects_project_id_get(
            project_id
        ).result

    @handle_api_exceptions
    def list_projects(
        self,
        *,
        name_contains: Optional[str] = None,
        creator_id: Optional[str] = None,
        parent_cloud_id: Optional[str] = None,
        include_defaults: bool = True,
        sort_field: Optional[str] = None,
        sort_order: Optional[str] = None,
        paging_token: Optional[str] = None,
        count: Optional[int] = None,
    ) -> ProjectListResponse:
        return self._internal_api_client.list_projects_api_v2_projects_get(
            name_contains=name_contains,
            creator_id=creator_id,
            parent_cloud_id=parent_cloud_id,
            include_defaults=include_defaults,
            sort_field=sort_field,
            sort_order=sort_order,
            paging_token=paging_token,
            count=count or self.LIST_ENDPOINT_COUNT,
        )

    @handle_api_exceptions
    def create_project(self, project: WriteProject) -> ProjectBase:
        return self._internal_api_client.create_project_api_v2_projects_post(
            project
        ).result

    @handle_api_exceptions
    def delete_project(self, project_id: str) -> None:
        self._internal_api_client.delete_project_api_v2_projects_project_id_delete(
            project_id
        )

    @handle_api_exceptions
    def get_default_project(self, parent_cloud_id: str) -> Project:
        return self._internal_api_client.get_default_project_api_v2_projects_default_project_get(
            parent_cloud_id=parent_cloud_id,
        ).result

    @handle_api_exceptions
    def add_project_collaborators(
        self, project_id: str, collaborators: List[CreateUserProjectCollaborator]
    ) -> None:
        self._internal_api_client.batch_create_project_collaborators_api_v2_projects_project_id_collaborators_users_batch_create_post(
            project_id, collaborators
        )

    @handle_api_exceptions
    def get_job(
        self,
        *,
        name: Optional[str],
        job_id: Optional[str],
        cloud: Optional[str],
        project: Optional[str],
        include_archived: bool = False,
    ) -> Optional[ProductionJob]:
        if job_id is not None:
            # God mode: job_id lookup always finds job regardless of archive status
            try:
                return self._external_api_client.get_production_job(job_id).result
            except ExternalApiException as e:
                if e.status == 404:
                    return None
                raise e from None
        else:
            # Name-based lookup: respect archive filtering using internal API
            paging_token = None
            cloud_id = self.get_cloud_id(cloud_name=cloud)
            project_id = self.get_project_id(parent_cloud_id=cloud_id, name=project)
            archive_status = (
                ArchiveStatus.ALL if include_archived else ArchiveStatus.NOT_ARCHIVED
            )
            result: Optional[ProductionJob] = None
            while True:
                resp = self.list_jobs(
                    name=name,
                    project_id=project_id,
                    archive_status=archive_status,
                    count=self.LIST_ENDPOINT_COUNT,
                    paging_token=paging_token,
                )
                for job in resp.results:
                    if (
                        job is not None
                        and job.name == name
                        and (result is None or job.created_at > result.created_at)
                    ):
                        result = job

                paging_token = resp.metadata.next_paging_token
                if paging_token is None:
                    break

            return result

    @handle_api_exceptions
    def get_job_runs(self, job_id: str) -> List[APIJobRun]:
        job_runs: List[APIJobRun] = search_entities(
            self._external_api_client.search_jobs,
            JobsQuery(
                ha_job_id=job_id,
                show_ray_client_runs_only=False,
                sort_by_clauses=[
                    SortByClauseJobsSortField(
                        sort_field=JobsSortField.CREATED_AT, sort_order=SortOrder.ASC,
                    )
                ],
                paging=PageQuery(),
            ),
        )
        return job_runs

    @handle_api_exceptions
    def get_job_queue(self, job_queue_id: str) -> Optional[DecoratedJobQueue]:
        try:
            return self._internal_api_client.get_job_queue_api_v2_job_queues_job_queue_id_get(
                job_queue_id
            ).result
        except InternalApiException as e:
            if e.status == 404:
                return None

            raise e from None

    @handle_api_exceptions
    def update_job_queue(
        self,
        job_queue_id: str,
        max_concurrency: Optional[int] = None,
        idle_timeout_s: Optional[int] = None,
    ) -> DecoratedJobQueue:
        if max_concurrency is None and idle_timeout_s is None:
            raise ValueError("No fields to update")

        return self._internal_api_client.update_job_queue_api_v2_job_queues_job_queue_id_put(
            job_queue_id,
            update_job_queue_request=UpdateJobQueueRequest(
                max_concurrency=max_concurrency, idle_timeout_sec=idle_timeout_s,
            ),
        ).result

    @handle_api_exceptions
    def archive_job_queue(self, job_queue_id: str) -> None:
        """Archive (seal) a job queue. No new jobs can be submitted."""
        self._internal_api_client.archive_job_queue_api_v2_job_queues_job_queue_id_archive_post(
            job_queue_id=job_queue_id
        )

    @handle_api_exceptions
    def terminate_job_queue(self, job_queue_id: str):
        """Terminate a job queue and all its jobs."""
        return self._internal_api_client.terminate_job_queue_api_v2_job_queues_job_queue_id_terminate_post(
            job_queue_id=job_queue_id
        )

    @handle_api_exceptions
    def delete_job_queue(self, job_queue_id: str) -> None:
        """Delete a job queue. Jobs previously submitted remain accessible."""
        self._internal_api_client.delete_job_queue_api_v2_job_queues_job_queue_id_delete(
            job_queue_id=job_queue_id
        )

    @handle_api_exceptions
    def list_job_queues(  # noqa: PLR0913
        self,
        *,
        name: Optional[str] = None,
        creator_id: Optional[str] = None,
        cluster_status: Optional[SessionState] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        tags_filter: Optional[Dict[str, List[str]]] = None,
        count: Optional[int] = None,
        paging_token: Optional[str] = None,
        sorting_directives: Optional[List[JobQueueSortDirective]] = None,
        include_archived: bool = False,
    ) -> DecoratedjobqueueListResponse:
        cloud_id = self.get_cloud_id(cloud_name=cloud) if cloud else None
        project_id = (
            self.get_project_id(parent_cloud_id=cloud_id, name=project)
            if project
            else None
        )
        archive_status = (
            ArchiveStatus.ALL if include_archived else ArchiveStatus.NOT_ARCHIVED
        )

        return self._internal_api_client.list_job_queues_api_v2_job_queues_post(
            job_queues_query=JobQueuesQuery(
                name=TextQuery(equals=name) if name else None,
                creator_id=creator_id,
                cluster_status=cluster_status,
                project_id=project_id,
                cloud_id=cloud_id,
                archive_status=archive_status,
                tags_filter=tags_filter,
                paging=PageQuery(count=count, paging_token=paging_token),
                sorting_directives=sorting_directives,
            ),
        )

    @handle_api_exceptions
    def list_jobs(  # noqa: PLR0913
        self,
        *,
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        creator_id: Optional[str] = None,
        state_filter: Optional[List[str]] = None,
        archive_status: Optional[str] = None,
        tags_filter: Optional[List[str]] = None,
        count: Optional[int] = None,
        paging_token: Optional[str] = None,
        sort_field: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> DecoratedproductionjobListResponse:
        # Build kwargs dynamically to avoid passing None for count,
        # which causes TypeError in OpenAPI client validation
        kwargs: Dict[str, Any] = {
            "project_id": project_id,
            "name": name,
            "creator_id": creator_id,
            "type_filter": "BATCH_JOB",
            "archive_status": archive_status,
            "state_filter": state_filter,
            "tag_filter": tags_filter,
            "paging_token": paging_token,
            "sort_field": sort_field,
            "sort_order": sort_order,
        }
        if count is not None:
            kwargs["count"] = count
        return self._internal_api_client.list_decorated_jobs_api_v2_decorated_ha_jobs_get(
            **kwargs
        )

    @handle_api_exceptions
    def rollout_service(
        self, model: ApplyProductionServiceV2Model
    ) -> DecoratedProductionServiceV2APIModel:
        result = self._internal_api_client.apply_service_api_v2_services_v2_apply_put(
            model
        ).result
        return result

    @handle_api_exceptions
    def rollout_service_multi_version(
        self, model: ApplyProductionServiceMultiVersionV2Model
    ) -> DecoratedProductionServiceV2APIModel:
        result = self._internal_api_client.apply_service_multi_version_api_v2_services_v2_apply_multi_version_put(
            model
        ).result
        return result

    @handle_api_exceptions
    def rollback_service(
        self, service_id: str, *, max_surge_percent: Optional[int] = None
    ) -> DecoratedProductionServiceV2APIModel:
        result = self._internal_api_client.rollback_service_api_v2_services_v2_service_id_rollback_post(
            service_id,
            rollback_service_model=RollbackServiceModel(
                max_surge_percent=max_surge_percent
            ),
        )
        return result

    @handle_api_exceptions
    def terminate_service(
        self, service_id: str
    ) -> DecoratedProductionServiceV2APIModel:
        result = self._internal_api_client.terminate_service_api_v2_services_v2_service_id_terminate_post(
            service_id
        )
        return result

    @handle_api_exceptions
    def archive_service(self, service_id: str) -> DecoratedProductionServiceV2APIModel:
        result = self._internal_api_client.archive_service_api_v2_services_v2_service_id_archive_post(
            service_id
        )
        return result

    @handle_api_exceptions
    def delete_service(self, service_id: str) -> None:
        self._internal_api_client.delete_service_api_v2_services_v2_service_id_delete(
            service_id
        )

    @handle_api_exceptions
    def submit_job(self, model: CreateInternalProductionJob) -> InternalProductionJob:
        job: InternalProductionJob = self._internal_api_client.create_job_api_v2_decorated_ha_jobs_create_post(
            model,
        ).result
        return job

    @handle_api_exceptions
    def terminate_job(self, job_id: str):
        self._external_api_client.terminate_job(job_id)

    @handle_api_exceptions
    def archive_job(self, job_id: str):
        self._internal_api_client.archive_job_api_v2_decorated_ha_jobs_production_job_id_archive_post(
            job_id
        )

    @handle_api_exceptions
    def delete_job(self, job_id: str) -> None:
        self._internal_api_client.delete_job_api_v2_decorated_ha_jobs_production_job_id_delete(
            job_id
        )

    def _upload_local_runtime_env(
        self,
        cloud_id: str,
        cloud_resource_id: Optional[str],
        zip_file_bytes: bytes,
        content_hash: str,
        overwrite_existing_file: bool,
    ) -> CloudDataBucketPresignedUrlResponse:
        file_name = RUNTIME_ENV_PACKAGE_FORMAT.format(content_hash=content_hash)
        request = CloudDataBucketPresignedUrlRequest(
            file_type=CloudDataBucketFileType.RUNTIME_ENV_PACKAGES,
            file_name=file_name,
            access_mode=CloudDataBucketAccessMode.WRITE,
            cloud_resource_id=cloud_resource_id,
        )
        info: CloudDataBucketPresignedUrlResponse = self._internal_api_client.generate_cloud_data_bucket_presigned_url_api_v2_clouds_cloud_id_generate_cloud_data_bucket_presigned_url_post(
            cloud_id, request
        ).result

        # Skip the upload entirely if the file already exists.
        if info.file_exists and not overwrite_existing_file:
            internal_logger.debug(
                f"Skipping file upload for '{file_name}' because it already exists in cloud storage."
            )
            return info

        if info.url_scheme == CloudDataBucketPresignedUrlScheme.SMART_OPEN:
            # If the presigned URL scheme is SMART_OPEN, upload to cloud storage using the provided bucket name, path, & environment, and the smart_open library.
            bucket_name = info.bucket_name
            bucket_path = info.bucket_path

            env_vars: Dict[str, str] = {
                "AWS_ENDPOINT_URL": info.url,
            }
            with set_env(**env_vars), smart_open.open(
                f"{bucket_name}/{bucket_path}", "wb",
            ) as fout:
                fout.write(zip_file_bytes)

        else:
            # Default to HTTP PUT.
            internal_logger.debug(f"Uploading file '{file_name}' to cloud storage.")
            headers = None
            if info.file_uri.startswith("azure") or info.file_uri.startswith("abfss"):
                headers = {
                    "x-ms-blob-type": "BlockBlob",
                    "x-ms-version": "2025-07-05",
                    "x-ms-date": datetime.utcnow().strftime(
                        "%a, %d %b %Y %H:%M:%S GMT"
                    ),
                    "x-ms-blob-content-type": "application/zip",
                }
            requests.put(
                info.url, data=zip_file_bytes, headers=headers
            ).raise_for_status()

        return info

    @handle_api_exceptions
    def upload_local_dir_to_cloud_storage(
        self,
        local_dir: str,
        *,
        cloud_id: str,
        excludes: Optional[List[str]] = None,
        overwrite_existing_file: bool = OVERWRITE_EXISTING_CLOUD_STORAGE_FILES,
        cloud_resource_name: Optional[str] = None,
    ) -> str:
        if not pathlib.Path(local_dir).is_dir():
            raise RuntimeError(f"Path '{local_dir}' is not a valid directory.")

        cloud_resource_id = None
        if cloud_resource_name is not None:
            cloud_resources = self._internal_api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
                cloud_id=cloud_id,
            ).results
            cloud_resource_id = next(
                (
                    cloud_resource.cloud_resource_id
                    for cloud_resource in cloud_resources
                    if cloud_resource.name == cloud_resource_name
                ),
                None,
            )
            if cloud_resource_id is None:
                raise ValueError(
                    f"Cloud resource '{cloud_resource_name}' not found in cloud '{cloud_id}'"
                )

        with zip_local_dir(local_dir, excludes=excludes) as (
            _,
            zip_file_bytes,
            content_hash,
        ):
            info = self._upload_local_runtime_env(
                cloud_id=cloud_id,
                cloud_resource_id=cloud_resource_id,
                zip_file_bytes=zip_file_bytes,
                content_hash=content_hash,
                overwrite_existing_file=overwrite_existing_file,
            )
            return info.file_uri

    def upload_local_dir_to_cloud_storage_multi_cloud_resource(
        self,
        local_dir: str,
        *,
        cloud_id: str,
        cloud_resource_names: List[Optional[str]],
        excludes: Optional[List[str]] = None,
        overwrite_existing_file: bool = False,
    ) -> str:
        if not pathlib.Path(local_dir).is_dir():
            raise RuntimeError(f"Path '{local_dir}' is not a valid directory.")

        all_cloud_resources = self._internal_api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
            cloud_id=cloud_id,
        ).results
        cloud_resource_names_to_ids = {
            cloud_resource.name: cloud_resource.cloud_resource_id
            for cloud_resource in all_cloud_resources
        }

        bucket_paths = set()

        with zip_local_dir(local_dir, excludes=excludes) as (
            _,
            zip_file_bytes,
            content_hash,
        ):
            for cloud_resource_name in cloud_resource_names:
                if cloud_resource_name is not None:
                    if cloud_resource_name not in cloud_resource_names_to_ids:
                        raise ValueError(
                            f"Cloud resource '{cloud_resource_name}' not found in cloud '{cloud_id}'"
                        )
                    cloud_resource_id = cloud_resource_names_to_ids[cloud_resource_name]
                else:
                    cloud_resource_id = None

                info = self._upload_local_runtime_env(
                    cloud_id=cloud_id,
                    cloud_resource_id=cloud_resource_id,
                    zip_file_bytes=zip_file_bytes,
                    content_hash=content_hash,
                    overwrite_existing_file=overwrite_existing_file,
                )
                bucket_paths.add(info.bucket_path)

            assert len(bucket_paths) == 1
            return bucket_paths.pop()

    def _fetch_log_chunks(self, job_run_id: str) -> Tuple[List[str], Any]:
        all_log_chunk_urls = []
        MAX_PAGE_SIZE = 1000
        next_page_token = None
        bearer_token = None
        while True:
            log_download_result = self._internal_api_client.get_job_logs_download_v2_api_v2_logs_job_logs_download_v2_job_id_get(
                job_id=job_run_id,
                next_page_token=next_page_token,
                page_size=MAX_PAGE_SIZE,
            ).result

            if bearer_token is None:
                bearer_token = log_download_result.bearer_token

            all_log_chunk_urls.extend(
                [chunk.chunk_url for chunk in log_download_result.log_chunks]
            )

            next_page_token = log_download_result.next_page_token
            if next_page_token is None:
                break

        return all_log_chunk_urls, bearer_token

    def _fetch_log_chunks_for_controller_logs(
        self, cluster_id: str
    ) -> Tuple[List[str], Any]:
        all_log_chunk_urls = []
        MAX_PAGE_SIZE = 1000
        next_page_token = None
        bearer_token = None
        while True:
            log_download_result = self._internal_api_client.get_serve_logs_download_api_v2_logs_serve_logs_download_cluster_id_get(
                cluster_id=cluster_id,
                next_page_token=next_page_token,
                page_size=MAX_PAGE_SIZE,
            ).result

            if bearer_token is None:
                bearer_token = log_download_result.bearer_token

            all_log_chunk_urls.extend(
                [chunk.chunk_url for chunk in log_download_result.log_chunks]
            )

            next_page_token = log_download_result.next_page_token
            if next_page_token is None:
                break

        return all_log_chunk_urls, bearer_token

    def _read_log_lines(  # noqa: PLR0912
        self,
        log_chunk_urls: List[str],
        head: bool,
        bearer_token: Any,
        max_lines: Optional[int],
        parse_json: Optional[bool] = None,
    ) -> str:
        # TODO(mowen): Would be nice to return some placeholder here to allow all new
        # logs to be read from a particular point onwards.
        # TODO(aguo): Change this to be a generator to avoid loading all logs into memory at once
        # and to gradually load logs as needed.

        def parse_json_line(line: str) -> str:
            json_line = json.loads(line)
            # This is the default schema for ray core logger but users
            # could technically use any schema they want for structured logs.
            # Fall back to spitting out the json in the worst-case scenario.
            if "asctime" in json_line and "message" in json_line:
                return f"{json_line['asctime']} {json_line['message']}"
            elif "message" in json_line:
                return json_line["message"]
            # This is the worst-case scenario but very unlikely. Users would
            # not likely be changing "message" to something-else.
            return line

        result_lines: List[str] = []
        step = 1 if head else -1
        line_count = 0
        for chunk_url in log_chunk_urls[::step]:
            log_lines = _download_log_from_s3_url_sync(
                chunk_url, bearer_token=bearer_token
            ).splitlines()

            if max_lines is not None:
                num_lines_to_add = min(len(log_lines), max_lines - line_count)
            else:
                num_lines_to_add = len(log_lines)
            line_count += num_lines_to_add

            if head:
                lines_to_add = log_lines[:num_lines_to_add]
            else:
                lines_to_add = log_lines[-1 * num_lines_to_add :]

            if parse_json is not False:
                try:
                    lines_to_add = [parse_json_line(line) for line in lines_to_add]
                except json.JSONDecodeError:
                    if parse_json is True:
                        raise ValueError(
                            "Failed to parse logs as JSON. Logs are not all in JSON format."
                        )
                    # If we fail to parse_json, we should always just use plain text going forward.
                    parse_json = False
                    # lines_to_add should already be plain text, so continue on...

            result_lines = (
                result_lines + lines_to_add if head else lines_to_add + result_lines
            )

            if line_count == max_lines:
                break

        if not result_lines:
            return ""
        return "\n".join(result_lines) + "\n"

    @handle_api_exceptions
    def logs_for_job_run(
        self,
        job_run_id: str,
        head: bool = False,
        max_lines: Optional[int] = None,
        parse_json: Optional[bool] = None,
    ) -> str:
        """
        Retrieves logs from the streaming job logs folder in S3/GCS
        Args:
        - parse_json: If true, we will always attempt to parse the logs as JSON.
            If false, we will always attempt to parse the logs as text. If None, we
            will attempt to parse the logs as JSON and fall back to text if parsing
            fails.
        """

        all_log_chunk_urls, bearer_token = self._fetch_log_chunks(job_run_id)

        logs = self._read_log_lines(
            all_log_chunk_urls, head, bearer_token, max_lines, parse_json=parse_json
        )
        return logs

    @handle_api_exceptions
    def stream_logs_for_job_run(
        self, job_run_id: str, next_page_token: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """Stream logs incrementally for a job run with pagination support.

        Args:
            job_run_id: The ID of the job run to fetch logs for
            next_page_token: Token for fetching the next page of logs (for incremental streaming)

        Returns:
            Tuple of (logs, next_page_token) where next_page_token can be used for the next call
        """
        # Fetch only the new log chunks since the last call
        if next_page_token:
            # Incremental fetch - get only new chunks
            log_download_result = self._internal_api_client.get_job_logs_download_v2_api_v2_logs_job_logs_download_v2_job_id_get(
                job_id=job_run_id, next_page_token=next_page_token,
            ).result
        else:
            # First fetch - get all available chunks
            log_download_result = self._internal_api_client.get_job_logs_download_v2_api_v2_logs_job_logs_download_v2_job_id_get(
                job_id=job_run_id,
            ).result

        # Download and concatenate log chunks
        log_chunk_urls = [chunk.chunk_url for chunk in log_download_result.log_chunks]
        bearer_token = log_download_result.bearer_token

        logs = self._read_log_lines(
            log_chunk_urls,
            head=False,
            bearer_token=bearer_token,
            max_lines=None,
            parse_json=False,
        )

        # Return logs and the token for the next page
        new_next_page_token = (
            log_download_result.next_page_token
            if len(log_download_result.log_chunks) > 0
            else next_page_token
        )

        return logs, new_next_page_token

    @handle_api_exceptions
    def controller_logs_for_service_version(
        self,
        service_version: ProductionServiceV2VersionModel,
        head: bool = False,
        max_lines: Optional[int] = None,
        parse_json: Optional[bool] = None,
    ) -> str:
        """
        Returns the controller logs associated with a particular service version.

        Args:
        - parse_json: If true, we will always attempt to parse the logs as JSON.
            If false, we will always attempt to parse the logs as text. If None, we
            will attempt to parse the logs as JSON and fall back to text if parsing
            fails.
        """
        service_version_id = service_version.id[-8:]
        if not len(service_version.production_job_ids):
            raise ValueError(
                f"Service version '{service_version_id}' canary version is not ready. Please try again later..."
            )

        job = self._internal_api_client.get_job_api_v2_decorated_ha_jobs_production_job_id_get(
            service_version.production_job_ids[0]
        ).result

        if not job:
            raise ValueError(
                f"Service version '{service_version_id}' canary version is not ready. Please try again later..."
            )

        if not job.state.cluster:
            raise ValueError(
                f"Service version '{service_version_id}' canary version is not ready. Please try again later..."
            )

        cluster_id = job.state.cluster.id

        all_log_chunk_urls, bearer_token = self._fetch_log_chunks_for_controller_logs(
            cluster_id
        )

        logs = self._read_log_lines(
            all_log_chunk_urls, head, bearer_token, max_lines, parse_json=parse_json
        )
        return logs

    @handle_api_exceptions
    def apply_schedule(self, model: CreateSchedule) -> DecoratedSchedule:
        return self._internal_api_client.create_or_update_job_api_v2_experimental_cron_jobs_put(
            model
        ).result

    @handle_api_exceptions
    def get_schedule(
        self,
        *,
        name: Optional[str],
        id: Optional[str],  # noqa: A002
        cloud: Optional[str],
        project: Optional[str],
    ) -> Optional[DecoratedSchedule]:
        if id is not None:
            try:
                return self._internal_api_client.get_cron_job_api_v2_experimental_cron_jobs_cron_job_id_get(
                    id
                ).result
            except ExternalApiException as e:
                if e.status == 404:
                    return None
                raise e from None
        else:
            paging_token = None
            cloud_id = self.get_cloud_id(cloud_name=cloud)
            project_id = self.get_project_id(parent_cloud_id=cloud_id, name=project)
            result: Optional[DecoratedSchedule] = None
            while True:
                resp = self._internal_api_client.list_cron_jobs_api_v2_experimental_cron_jobs_get(
                    project_id=project_id,
                    name=name,
                    count=self.LIST_ENDPOINT_COUNT,
                    paging_token=paging_token,
                )
                for schedule in resp.results:
                    if schedule is not None and schedule.name == name:
                        result = schedule
                        break

                paging_token = resp.metadata.next_paging_token
                if paging_token is None:
                    break

            return result

    @handle_api_exceptions
    def set_schedule_state(self, id: str, is_paused: bool):  # noqa: A002
        _ = self._internal_api_client.pause_cron_job_api_v2_experimental_cron_jobs_cron_job_id_pause_post(
            id, {"is_paused": is_paused}
        ).result

    @handle_api_exceptions
    def trigger_schedule(self, id: str):  # noqa: A002
        self._internal_api_client.trigger_cron_job_api_v2_experimental_cron_jobs_cron_job_id_trigger_post(
            id
        )

    @handle_api_exceptions
    def list_schedules(
        self,
        *,
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        cloud_id: Optional[str] = None,
        creator_id: Optional[str] = None,
        count: Optional[int] = None,
        paging_token: Optional[str] = None,
    ) -> DecoratedscheduleListResponse:
        """List schedules with filtering and pagination."""
        # Build kwargs dynamically to avoid passing None for count,
        # which causes TypeError in OpenAPI client validation
        kwargs: Dict[str, Any] = {
            "project_id": project_id,
            "cloud_id": cloud_id,
            "name": name,
            "creator_id": creator_id,
            "paging_token": paging_token,
        }
        if count is not None:
            kwargs["count"] = count
        return self._internal_api_client.list_cron_jobs_api_v2_experimental_cron_jobs_get(
            **kwargs
        )

    @handle_api_exceptions
    def delete_schedule(self, schedule_id: str) -> None:
        """Delete a schedule.

        If the schedule is active, it will be automatically paused before deletion.
        The schedule must have no active triggered jobs.

        Args:
            schedule_id: ID of the schedule to delete
        """
        self._internal_api_client.delete_cron_job_api_v2_experimental_cron_jobs_cron_job_id_delete(
            cron_job_id=schedule_id
        )

    @handle_api_exceptions
    def create_workspace(self, model: CreateExperimentalWorkspace) -> str:
        return self._internal_api_client.create_workspace_api_v2_experimental_workspaces_post(
            create_experimental_workspace=model,
        ).result.id

    @handle_api_exceptions
    def get_workspace(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Optional[ExperimentalWorkspace]:
        """Get a workspace by either name or id. Filter by cloud and project.

        Returns None if not found.
        """
        if id is not None:
            try:
                return self._internal_api_client.get_workspace_api_v2_experimental_workspaces_workspace_id_get(
                    id
                ).result
            except ExternalApiException as e:
                if e.status == 404:
                    return None
                raise e from None
        else:
            paging_token = None
            cloud_id = self.get_cloud_id(cloud_name=cloud)
            project_id = self.get_project_id(parent_cloud_id=cloud_id, name=project)
            resp = self._internal_api_client.list_workspaces_api_v2_experimental_workspaces_get(
                project_id=project_id,
                name=name,
                count=self.LIST_ENDPOINT_COUNT,
                paging_token=paging_token,
            )

            if len(resp.results) == 0:
                return None

            workspace = resp.results[0]
            return workspace

    @handle_api_exceptions
    def update_workspace(
        self,
        *,
        workspace_id: Optional[str],
        name: Optional[str] = None,
        compute_config_id: Optional[str] = None,
        cluster_environment_build_id: Optional[str] = None,
        idle_timeout_minutes: Optional[int] = None,
    ):
        workspace = self.get_workspace(id=workspace_id)
        if not workspace:
            raise ValueError(f"Workspace '{workspace_id}' not found.")

        if name:
            # Update the workspace name with workspaces patch API
            self._internal_api_client.patch_workspace_api_v2_experimental_workspaces_workspace_id_patch(
                workspace_id=workspace_id,
                json_patch_operation=[
                    {"op": "replace", "path": "/name", "value": name,},
                ],
            )

        if compute_config_id or cluster_environment_build_id or idle_timeout_minutes:
            # Use the internal session API to update cluster config and idle timeout
            # This matches what the UI does and is more efficient than the external API
            self._internal_api_client.put_session_cluster_config_with_session_idle_timeout_api_v2_sessions_session_id_cluster_config_with_session_idle_timeout_put(
                session_id=workspace.cluster_id,
                compute_template_id=compute_config_id,
                build_id=cluster_environment_build_id,
                idle_timeout=idle_timeout_minutes,
            )

    @handle_api_exceptions
    def update_workspace_dependencies_offline_only(
        self, workspace_id: str, requirements: List[str]
    ):
        return self._internal_api_client.put_workspace_proxied_dataplane_artifacts_api_v2_experimental_workspaces_workspace_id_proxied_dataplane_artifacts_put(
            workspace_id=workspace_id,
            workspace_dataplane_proxied_artifacts={
                "requirements": "\n".join(requirements),
            },
        )

    @handle_api_exceptions
    def update_workspace_env_vars_offline_only(
        self, workspace_id: str, env_vars: Dict[str, str]
    ):
        return self._internal_api_client.put_workspace_proxied_dataplane_artifacts_api_v2_experimental_workspaces_workspace_id_proxied_dataplane_artifacts_put(
            workspace_id=workspace_id,
            workspace_dataplane_proxied_artifacts={
                "environment_variables": [
                    f"{key}={value}" for key, value in env_vars.items()
                ],
            },
        )

    @handle_api_exceptions
    def get_workspace_proxied_dataplane_artifacts(
        self, workspace_id: str
    ) -> WorkspaceDataplaneProxiedArtifacts:
        return self._internal_api_client.get_workspace_proxied_dataplane_artifacts_api_v2_experimental_workspaces_workspace_id_proxied_dataplane_artifacts_get(
            workspace_id
        ).result

    @handle_api_exceptions
    def start_workspace(self, workspace_id: str):
        """Start a workspace."""
        workspace_model = self.get_workspace(id=workspace_id)
        if workspace_model is None:
            raise ValueError(f"Workspace '{workspace_id}' not found.")

        return self._internal_api_client.start_session_api_v2_sessions_session_id_start_post(
            session_id=workspace_model.cluster_id,
            start_session_options=StartSessionOptions(),
        )

    @handle_api_exceptions
    def terminate_workspace(self, workspace_id: str):
        """Terminate a workspace."""
        workspace_model = self.get_workspace(id=workspace_id)
        if workspace_model is None:
            raise ValueError(f"Workspace '{workspace_id}' not found.")

        options = StopSessionOptions(
            terminate=True,
            workers_only=False,
            keep_min_workers=False,
            take_snapshot=False,
        )
        return self._internal_api_client.stop_session_api_v2_sessions_session_id_stop_post(
            session_id=workspace_model.cluster_id, stop_session_options=options,
        )

    @handle_api_exceptions
    def get_workspace_cluster(
        self, workspace_id: Optional[str]
    ) -> Optional[DecoratedSession]:
        workspace = self.get_workspace(id=workspace_id)
        if not workspace:
            raise ValueError(f"Workspace '{workspace_id}' not found.")

        result = self._internal_api_client.get_decorated_cluster_api_v2_decorated_sessions_cluster_id_get(
            workspace.cluster_id
        )
        return result.result

    @handle_api_exceptions
    def get_cluster_head_node_ip(self, cluster_id: str) -> str:
        head_ip = self._internal_api_client.get_session_head_ip_api_v2_sessions_session_id_head_ip_get(
            cluster_id
        ).result.head_ip
        return head_ip

    @handle_api_exceptions
    def get_cluster_ssh_key(self, cluster_id: str) -> SessionSshKey:
        return self._internal_api_client.get_session_ssh_key_api_v2_sessions_session_id_ssh_key_get(
            cluster_id
        ).result

    @handle_api_exceptions
    def get_workspace_default_dir_name(self, workspace_id) -> str:
        workspace = self.get_workspace(id=workspace_id)
        assert workspace, f"Workspace '{workspace_id}' not found."
        project = self._internal_api_client.get_project_api_v2_projects_project_id_get(
            workspace.project_id
        ).result
        if self._internal_api_client.check_is_feature_flag_on_api_v2_userinfo_check_is_feature_flag_on_get(
            FLAG_DEFAULT_WORKING_DIR_FOR_PROJ
        ).result.is_on:
            return project.directory_name
        else:
            return project.name

    @handle_api_exceptions
    def download_aggregated_instance_usage_csv(
        self,
        start_date,
        end_date,
        cloud_id=None,
        project_id=None,
        directory=None,
        hide_progress_bar=False,
    ) -> str:
        with FileDownloadProgress() as progress:
            task_id = progress.add_task(
                description="Preparing aggregated instance usage CSV",
                visible=not hide_progress_bar,
            )

            resp = self._internal_api_client.download_aggregated_instance_usage_csv_api_v2_aggregated_instance_usage_download_csv_get(
                start_date=start_date,
                end_date=end_date,
                cloud_id=cloud_id,
                project_id=project_id,
                _preload_content=False,
            )

            progress.update(
                task_id, description="Downloading aggregated instance usage CSV"
            )

            # Set the total size of the download for the progress bar
            total_size = int(resp.headers.get("content-length", 0))
            progress.update(task_id, total=total_size)

            # Construct the filepath to save the downloaded file
            content_disposition = resp.headers.get("content-disposition", "")
            filename_regex = re.search(
                r'filename="?(?P<filename>[^"]+)"?', content_disposition
            )

            if not filename_regex:
                filename = f"aggregated_instance_usage_{start_date}_{end_date}.zip"
            else:
                filename = filename_regex.group("filename")

            filepath = os.path.join(directory, filename) if directory else filename

            # Download the file
            try:
                with open(filepath, "wb") as f:
                    for chunk in resp.stream(Bytes.MB):
                        if chunk:
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"Failed to download to '{filepath}': {e}") from None

            progress.update(task_id, completed=total_size)
            progress.console.print(
                f"Download complete! File saved to '{filepath}'",
                style=Style(bold=True, color="green"),
            )

        return filepath

    @handle_api_exceptions
    def create_api_key(
        self, duration: float, user_id: Optional[str]
    ) -> ServerSessionToken:
        return self._internal_api_client.create_api_key_api_v2_users_create_api_key_post(
            ApiKeyParameters(user_id=user_id, duration=duration)
        ).result

    @handle_api_exceptions
    def rotate_api_key(self, user_id: str) -> None:
        self._internal_api_client.rotate_api_key_for_user_api_v2_organization_collaborators_rotate_api_key_for_user_user_id_post(
            user_id
        )

    @handle_api_exceptions
    def admin_batch_create_users(
        self, admin_create_users: List[AdminCreateUser]
    ) -> List[AdminCreatedUser]:
        return self._internal_api_client.admin_batch_create_users_api_v2_users_admin_batch_create_post(
            admin_create_users
        ).results

    @handle_api_exceptions
    def create_organization_invitations(
        self, emails: List[str]
    ) -> Tuple[List[str], List[str]]:
        results = self._internal_api_client.batch_create_invitations_api_v2_organization_invitations_batch_create_post(
            [CreateOrganizationInvitation(email=email) for email in emails]
        ).results

        success_emails = []
        error_messages = []

        for idx, result in enumerate(results):
            if result.data:
                success_emails.append(emails[idx])
            else:
                error_messages.append(result.error.detail)

        return success_emails, error_messages

    @handle_api_exceptions
    def list_organization_invitations(self) -> List[OrganizationInvitation]:
        results = (
            self._internal_api_client.list_invitations_api_v2_organization_invitations_get().results
        )

        return results

    @handle_api_exceptions
    def delete_organization_invitation(self, email: str) -> OrganizationInvitation:
        invitation = self._internal_api_client.list_invitations_api_v2_organization_invitations_get(
            email=email
        ).results

        if len(invitation) == 0:
            raise ValueError(f"Invitation for email '{email}' not found.")
        elif len(invitation) > 1:
            raise ValueError(
                f"Multiple invitations found for email '{email}'. Please contact Anyscale support."
            )

        invitation_id = invitation[0].id

        return self._internal_api_client.invalidate_invitation_api_v2_organization_invitations_invitation_id_invalidate_post(
            invitation_id
        ).result

    @handle_api_exceptions
    def list_organization_collaborators(
        self,
        *,
        email: Optional[str] = None,
        name: Optional[str] = None,
        collaborator_type: Optional[CollaboratorType] = None,
        is_service_account: Optional[bool] = None,
        count: Optional[int] = None,
        paging_token: Optional[str] = None,
    ) -> OrganizationcollaboratorListResponse:
        collaborator_type_value = None
        if collaborator_type is not None:
            collaborator_type_value = (
                collaborator_type.value
                if hasattr(collaborator_type, "value")
                else str(collaborator_type)
            )

        return self._internal_api_client.list_organization_collaborators_api_v2_organization_collaborators_get(
            email=email,
            name=name,
            collaborator_type=collaborator_type_value,
            is_service_account=is_service_account,
            count=count if count is not None else self.LIST_ENDPOINT_COUNT,
            paging_token=paging_token,
        )

    @handle_api_exceptions
    def get_organization_collaborators(
        self,
        email: Optional[str] = None,
        name: Optional[str] = None,
        collaborator_type: Optional[CollaboratorType] = None,
        is_service_account: Optional[bool] = None,  # noqa: ARG002
    ) -> List[OrganizationCollaborator]:
        response = self.list_organization_collaborators(
            email=email,
            name=name,
            collaborator_type=collaborator_type,
            is_service_account=is_service_account,
        )

        return response.results

    @handle_api_exceptions
    def delete_organization_collaborator(self, identity_id: str) -> None:
        self._internal_api_client.remove_organization_collaborator_api_v2_organization_collaborators_identity_id_delete(
            identity_id
        )

    @handle_api_exceptions
    def create_service_account(self, name: str) -> AnyscaleServiceAccount:
        return self._internal_api_client.create_service_account_api_v2_users_service_accounts_post(
            name
        ).result

    @handle_api_exceptions
    def create_resource_quota(
        self, create_resource_quota: CreateResourceQuota
    ) -> ResourceQuota:
        return self._internal_api_client.create_resource_quota_api_v2_resource_quotas_post(
            create_resource_quota
        ).result

    @handle_api_exceptions
    def list_resource_quotas(
        self,
        name: Optional[str] = None,
        cloud_id: Optional[str] = None,
        creator_id: Optional[str] = None,
        is_enabled: Optional[bool] = None,
        max_items: int = 20,
    ) -> List[ResourceQuota]:

        query = ListResourceQuotasQuery(
            name=TextQuery(equals=name) if name else None,
            cloud_id=cloud_id,
            creator_id=creator_id,
            is_enabled=is_enabled,
            paging=PageQuery(count=max_items),
        )

        resource_quotas = self._internal_api_client.search_resource_quotas_api_v2_resource_quotas_search_post(
            query
        ).results

        return resource_quotas

    @handle_api_exceptions
    def delete_resource_quota(self, resource_quota_id: str) -> None:
        self._internal_api_client.delete_resource_quota_api_v2_resource_quotas_resource_quota_id_delete(
            resource_quota_id
        )

    @handle_api_exceptions
    def set_resource_quota_status(
        self, resource_quota_id: str, is_enabled: bool
    ) -> None:
        _ = self._internal_api_client.set_resource_quota_status_api_v2_resource_quotas_resource_quota_id_status_patch(
            resource_quota_id, ResourceQuotaStatus(is_enabled=is_enabled)
        ).result

    @handle_api_exceptions
    def upsert_resource_tags(
        self,
        resource_type: ResourceTagResourceType,
        resource_id: str,
        tags: Dict[str, str],
    ) -> None:
        req = UpsertResourceTagsRequest(
            resource_type=resource_type, resource_id=resource_id, tags=tags
        )
        self._internal_api_client.upsert_resource_tags_api_v2_tags_resource_put(req)

    @handle_api_exceptions
    def delete_resource_tags(
        self, resource_type: ResourceTagResourceType, resource_id: str, keys: List[str],
    ) -> None:
        req = DeleteResourceTagsRequest(
            resource_type=resource_type, resource_id=resource_id, keys=keys
        )
        self._internal_api_client.delete_resource_tags_api_v2_tags_resource_delete(req)

    @handle_api_exceptions
    def list_resource_tags(
        self, resource_type: ResourceTagResourceType, resource_id: str
    ) -> List[ResourceTagRecord]:
        resp = self._internal_api_client.get_tags_for_resource_api_v2_tags_resource_get(
            resource_type, resource_id
        )
        result = resp.result
        if result is None or result.tags is None:
            return []
        return list(result.tags)

    @handle_api_exceptions
    def list_user_groups(
        self, *, count: int = 50, paging_token: Optional[str] = None,
    ) -> UsergroupListResponse:
        return self._internal_api_client.list_user_groups_api_v2_user_groups_get(
            count=count, paging_token=paging_token,
        )

    @handle_api_exceptions
    def get_user_group(self, group_id: str) -> UserGroup:
        response = self._internal_api_client.get_user_group_api_v2_user_groups_group_id_get(
            group_id
        )
        return response.result

    @handle_api_exceptions
    def list_user_group_memberships(self) -> Dict:
        """List all user groups with their members."""
        api = self._internal_api_client
        response = api.api_client.call_api(
            "/api/v2/user_groups/memberships/list",
            "GET",
            query_params=[],
            response_type="object",
            auth_settings=["HTTPBearer"],
            _return_http_data_only=True,
        )
        return response

    @handle_api_exceptions
    def update_resource_policy(
        self, resource_type: str, resource_id: str, policy: UpdatePolicyRequest,
    ) -> None:
        self._internal_api_client.update_resource_policy_api_v2_policy_resource_type_resource_id_put(
            resource_type=resource_type,
            resource_id=resource_id,
            update_policy_request=policy,
        )

    @handle_api_exceptions
    def get_resource_policy(
        self, resource_type: str, resource_id: str,
    ) -> PolicyResponse:
        response = self._internal_api_client.get_resource_policy_api_v2_policy_resource_type_resource_id_get(
            resource_type=resource_type, resource_id=resource_id,
        )
        return response.result

    @handle_api_exceptions
    def list_resource_policies(
        self, resource_type: str,
    ) -> ResourcepolicyitemListResponse:
        return self._internal_api_client.list_resource_policies_api_v2_policy_resource_type_get(
            resource_type=resource_type,
        )

    @handle_api_exceptions
    def migrate_scim_permissions(self, *, dry_run: bool = True) -> Dict:
        """Migrate organization permissions to SCIM-based user group permissions."""
        api = self._internal_api_client
        if hasattr(
            api, "migrate_scim_permissions_api_v2_scim_migrate_permissions_post"
        ):
            resp = api.migrate_scim_permissions_api_v2_scim_migrate_permissions_post(
                dry_run=dry_run
            )
            if hasattr(resp, "to_dict"):
                return resp.to_dict()  # type: ignore[no-any-return]
            return {"result": getattr(resp, "result", resp)}

        response = api.api_client.call_api(
            "/api/v2/scim/migrate-permissions",
            "POST",
            query_params=[("dry_run", dry_run)],
            response_type="object",
            auth_settings=["HTTPBearer"],
            _return_http_data_only=True,
        )
        return response

    @handle_api_exceptions
    def list_scim_user_permissions(self, user_id: Optional[str] = None) -> Dict:
        """List users and their effective cloud/project permissions, plus org owners."""
        api = self._internal_api_client
        query_params = []
        if user_id:
            query_params.append(("user_id", user_id))
        response = api.api_client.call_api(
            "/api/v2/scim/list-user-permissions",
            "GET",
            query_params=query_params,
            response_type="object",
            auth_settings=["HTTPBearer"],
            _return_http_data_only=True,
        )
        return response

    @handle_api_exceptions
    def scim_migration_preview(self) -> Dict:
        """Preview permission changes from users' perspective before SCIM enforcement."""
        api = self._internal_api_client
        response = api.api_client.call_api(
            "/api/v2/scim/scim-migration-preview",
            "GET",
            response_type="object",
            auth_settings=["HTTPBearer"],
            _return_http_data_only=True,
        )
        return response

    @handle_api_exceptions
    def list_databricks_connections(
        self, *, name: Optional[str] = None
    ) -> List[DatabricksConnectionInfo]:
        """List Databricks connections."""
        response = self._internal_api_client.list_databricks_connections_api_v2_integrations_connections_databricks_get(
            name=name
        )
        return response.connections

    @handle_api_exceptions
    def get_databricks_connection(self, connection_id: str) -> DatabricksConnectionInfo:
        """Get a Databricks connection by ID."""
        response = self._internal_api_client.get_databricks_connection_api_v2_integrations_connections_databricks_connection_id_get(
            connection_id=connection_id
        )
        return response.result

    @handle_api_exceptions
    def get_user_info(self) -> UserInfo:
        """Get information about the current user."""
        return self._internal_api_client.get_user_info_api_v2_userinfo_get().result
