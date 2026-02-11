from collections import defaultdict
from datetime import date, datetime
import logging
import os
from typing import Any, DefaultDict, Dict, Generator, List, Optional, Tuple, Union
from unittest.mock import Mock
import uuid

from anyscale._private.anyscale_client.common import (
    AnyscaleClientInterface,
    WORKSPACE_CLUSTER_NAME_PREFIX,
)
from anyscale._private.models.image_uri import ImageURI
from anyscale._private.models.integrations import ConnectionType
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models import (
    AdminCreatedUser,
    AdminCreateUser,
    AnyscaleServiceAccount,
    ApplyProductionServiceMultiVersionV2Model,
    Binding,
    Cloud,
    CloudListResponse,
    CloudProviders,
    ClusterOperation,
    ClusteroperationResponse,
    ClusterState,
    CollaboratorType,
    ComputeTemplateConfig,
    CreateCloudCollaborator,
    CreateExperimentalWorkspace,
    CreateInternalProductionJob,
    CreateResourceQuota,
    CreateUserProjectCollaborator,
    DecoratedCloudResource,
    DecoratedComputeTemplate,
    DecoratedlistserviceapimodelListResponse,
    DecoratedProductionServiceV2APIModel,
    DecoratedProductionServiceV2VersionAPIModel,
    DeletedPlatformFineTunedModel,
    ExperimentalWorkspace,
    FineTunedModel,
    FineTuneType,
    HaJobGoalStates,
    HaJobStates,
    InternalProductionJob,
    ListResponseMetadata,
    MiniCloud,
    MiniUser,
    OrganizationBaseRole,
    OrganizationCollaborator,
    OrganizationcollaboratorListResponse,
    OrganizationInvitation,
    OrganizationPermissionLevel,
    PolicyResponse,
    ProductionJob,
    ProductionJobStateTransition,
    Project,
    ProjectBase,
    ProjectListResponse,
    ResourcePolicyItem,
    ResourcepolicyitemListResponse,
    ResourceQuota,
    ResourceTagRecord,
    ResourceTagResourceType,
    ServerSessionToken,
    UpdatePolicyRequest,
    UserGroup,
    UsergroupListResponse,
    WorkspaceDataplaneProxiedArtifacts,
    WriteProject,
)
from anyscale.client.openapi_client.models.create_schedule import CreateSchedule
from anyscale.client.openapi_client.models.decorated_application_template import (
    DecoratedApplicationTemplate,
)
from anyscale.client.openapi_client.models.decorated_job_queue import DecoratedJobQueue
from anyscale.client.openapi_client.models.decorated_schedule import DecoratedSchedule
from anyscale.client.openapi_client.models.decorated_session import DecoratedSession
from anyscale.client.openapi_client.models.decoratedapplicationtemplate_list_response import (
    DecoratedapplicationtemplateListResponse,
)
from anyscale.client.openapi_client.models.decoratedjobqueue_list_response import (
    DecoratedjobqueueListResponse,
)
from anyscale.client.openapi_client.models.decoratedproductionjob_list_response import (
    DecoratedproductionjobListResponse,
)
from anyscale.client.openapi_client.models.decoratedschedule_list_response import (
    DecoratedscheduleListResponse,
)
from anyscale.client.openapi_client.models.job_queue_sort_directive import (
    JobQueueSortDirective,
)
from anyscale.client.openapi_client.models.mini_build import MiniBuild
from anyscale.client.openapi_client.models.session_ssh_key import SessionSshKey
from anyscale.cluster_compute import parse_cluster_compute_name_version
from anyscale.sdk.anyscale_client.configuration import Configuration
from anyscale.sdk.anyscale_client.models import (
    ApplyProductionServiceV2Model,
    Cluster,
    ClusterCompute,
    ClusterComputeConfig,
    ClusterEnvironmentBuild,
    ClusterEnvironmentBuildStatus,
    ComputeNodeType,
    Job as APIJobRun,
    ProductionServiceV2VersionModel,
    Resources as ComputeConfigResources,
    ServiceEventCurrentState,
    ServiceVersionState,
    SessionState,
)
from anyscale.sdk.anyscale_client.models.cluster_environment import ClusterEnvironment
from anyscale.sdk.anyscale_client.models.list_response_metadata import (
    ListResponseMetadata as SDKListResponseMetadata,
)
from anyscale.shared_anyscale_utils.latest_ray_version import LATEST_RAY_VERSION
from anyscale.utils.workspace_notification import WorkspaceNotification


block_logger = BlockLogger()
logger = logging.getLogger(__name__)

OPENAPI_NO_VALIDATION = Configuration()
OPENAPI_NO_VALIDATION.client_side_validation = False


def _matches_tag_filter(workspace, tag_filter: List[str]) -> bool:
    """Check if workspace matches tag filter (backend format: List["key:value"]).

    Tags with the same key are ORed, different keys are ANDed.
    """
    if not tag_filter:
        return True

    # Parse tag filter into dict
    filter_dict: Dict[str, List[str]] = {}
    for tag_str in tag_filter:
        if ":" not in tag_str:
            continue
        key, value = tag_str.split(":", 1)
        filter_dict.setdefault(key, []).append(value)

    # Get workspace tags
    workspace_tags = getattr(workspace, "tags", {}) or {}

    # Check each key (ANDed)
    for key, required_values in filter_dict.items():
        workspace_values = workspace_tags.get(key, [])
        if isinstance(workspace_values, str):
            workspace_values = [workspace_values]

        # At least one value must match (ORed)
        if not any(val in workspace_values for val in required_values):
            return False

    return True


class FakeAnyscaleClient(AnyscaleClientInterface):
    OPENAPI_NO_VALIDATION = OPENAPI_NO_VALIDATION
    BASE_UI_URL = "http://fake.com"
    CLOUD_BUCKET = "s3://fake-bucket/{cloud_id}"
    DEFAULT_CLOUD_ID = "fake-default-cloud-id"
    DEFAULT_CLOUD_NAME = "fake-default-cloud"
    DEFAULT_PROJECT_NAME = "fake-default-project"
    DEFAULT_PROJECT_ID = "fake-default-project-id"
    DEFAULT_CLUSTER_COMPUTE_NAME = "fake-default-cluster-compute"
    DEFAULT_CLUSTER_COMPUTE_ID = "fake-default-cluster-compute-id"
    DEFAULT_CLUSTER_ENV_BUILD_ID = "fake-default-cluster-env-build-id"
    DEFAULT_USER_ID = "fake-user-id"
    DEFAULT_USER_EMAIL = "user@email.com"
    DEFAULT_ORGANIZATION_ID = "fake-org-id"

    WORKSPACE_ID = "fake-workspace-id"
    WORKSPACE_CLOUD_ID = "fake-workspace-cloud-id"
    WORKSPACE_CLUSTER_ID = "fake-workspace-cluster-id"
    WORKSPACE_PROJECT_ID = "fake-workspace-project-id"
    WORKSPACE_CLUSTER_COMPUTE_ID = "fake-workspace-cluster-compute-id"
    WORKSPACE_CLUSTER_COMPUTE_NAME = "fake-workspace-cluster-compute"
    WORKSPACE_CLUSTER_ENV_BUILD_ID = "fake-workspace-cluster-env-build-id"

    SCHEDULE_NEXT_TRIGGER_AT_TIME = datetime.utcnow()

    def __init__(self) -> None:
        self._builds: Dict[str, ClusterEnvironmentBuild] = {
            self.DEFAULT_CLUSTER_ENV_BUILD_ID: ClusterEnvironmentBuild(
                id=self.DEFAULT_CLUSTER_ENV_BUILD_ID,
                cluster_environment_id="default-cluster-env-id",
                docker_image_name="docker.io/my/base-image:latest",
                status=ClusterEnvironmentBuildStatus.SUCCEEDED,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            self.WORKSPACE_CLUSTER_ENV_BUILD_ID: ClusterEnvironmentBuild(
                id=self.WORKSPACE_CLUSTER_ENV_BUILD_ID,
                cluster_environment_id="workspace-cluster-env-id",
                docker_image_name="docker.io/my/base-ws-image:latest",
                status=ClusterEnvironmentBuildStatus.SUCCEEDED,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
        }
        self._images: Dict[str, ClusterEnvironment] = {
            "default-cluster-env-id": ClusterEnvironment(
                id="default-cluster-env-id",
                name="default-cluster-env",
                anonymous=True,
                is_default=True,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            "workspace-cluster-env-id": ClusterEnvironment(
                id="workspace-cluster-env-id",
                name="default-workspace-cluster-env",
                anonymous=True,
                is_default=True,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
        }
        self._archived_images: Dict[str, ClusterEnvironment] = {}
        self._application_template_metadata: Dict[str, Dict[str, Any]] = {}
        self._ensure_application_template_metadata(
            "default-cluster-env-id",
            name="default-cluster-env",
            creator_id=self.DEFAULT_USER_ID,
            creator_email=self.DEFAULT_USER_EMAIL,
            anonymous=True,
            is_default=True,
        )
        self._ensure_application_template_metadata(
            "workspace-cluster-env-id",
            name="default-workspace-cluster-env",
            creator_id=self.DEFAULT_USER_ID,
            creator_email=self.DEFAULT_USER_EMAIL,
            anonymous=True,
            is_default=True,
        )
        self._compute_config_name_to_ids: DefaultDict[str, List[str]] = defaultdict(
            list
        )
        self._compute_config_id_to_cloud_id: Dict[str, str] = {}
        self._compute_configs: Dict[str, ClusterCompute] = {}
        self._archived_compute_configs: Dict[str, ClusterCompute] = {}
        self._workspace_cluster: Optional[Cluster] = None
        self._workspace_dependency_tracking_enabled: bool = False
        self._services: Dict[str, DecoratedProductionServiceV2APIModel] = {}
        self._versions: Dict[
            str, Dict[str, ProductionServiceV2VersionModel]
        ] = defaultdict(dict)
        self._archived_services: Dict[str, DecoratedProductionServiceV2APIModel] = {}
        self._deleted_services: Dict[str, DecoratedProductionServiceV2APIModel] = {}
        self._jobs: Dict[str, ProductionJob] = {}
        self._job_runs: Dict[str, List[APIJobRun]] = defaultdict(list)
        self._job_queues: Dict[str, DecoratedJobQueue] = {}
        self._project_to_id: Dict[Optional[str] : Dict[Optional[str], str]] = {}  # type: ignore
        self._projects: Dict[str, Project] = {}
        self._project_collaborators: Dict[str, List[CreateUserProjectCollaborator]] = {}
        self._rolled_out_model: Optional[ApplyProductionServiceV2Model] = None
        self._sent_workspace_notifications: List[WorkspaceNotification] = []
        self._rolled_back_service: Optional[Tuple[str, Optional[int]]] = None
        self._terminated_service: Optional[str] = None
        self._archived_jobs: Dict[str, ProductionJob] = {}
        self._deleted_jobs: Dict[str, ProductionJob] = {}
        self._requirements_path: Optional[str] = None
        self._upload_uri_mapping: Dict[str, str] = {}
        self._upload_bucket_path_mapping: Dict[
            str, Tuple[List[Optional[str]], str]
        ] = {}
        self._submitted_job: Optional[CreateInternalProductionJob] = None
        self._env_vars: Optional[Dict[str, str]] = None
        self._job_run_logs: Dict[str, str] = {}
        self._controller_logs: Dict[str, str] = {}
        self._schedules: Dict[str, DecoratedSchedule] = {}
        self._schedule_trigger_counts: Dict[str, int] = defaultdict(int)
        self._users: Dict[str, AdminCreatedUser] = {}
        self._workspaces: Dict[str, ExperimentalWorkspace] = {}
        self._workspaces_dependencies: Dict[str, List[str]] = {}
        self._workspaces_env_vars: Dict[str, Dict[str, str]] = {}
        self._clusters_headnode_ip: Dict[str, str] = {}
        self._clusters_ssh_key: Dict[str, SessionSshKey] = {}
        self._api_keys: Dict[str, List[str]] = defaultdict(list)
        self._organization_collaborators: List[OrganizationCollaborator] = []
        self._organization_invitations: Dict[str, OrganizationInvitation] = {}
        self._resource_quotas: Dict[str, ResourceQuota] = {}
        self._system_cluster_status: Dict[str, str] = {}
        self.upsert_resource_tags_calls: List[Tuple[str, str, Dict[str, str]]] = []
        self.delete_resource_tags_calls: List[Tuple[str, str, List[str]]] = []
        # Store resource tags in-memory for list operations
        self._resource_tags_store: Dict[Tuple[str, str], Dict[str, str]] = {}

        # Store connections for testing - keyed by connection ID
        self._connections: Dict[str, Any] = {}

        # Cloud resource ID -> DecoratedCloudResource
        self._cloud_resources: Dict[str, DecoratedCloudResource] = {}

        # Cloud ID -> Cloud.
        self._clouds: Dict[str, Cloud] = {
            self.DEFAULT_CLOUD_ID: Cloud(
                id=self.DEFAULT_CLOUD_ID,
                name=self.DEFAULT_CLOUD_NAME,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
        }

        # Cloud ID -> collaborators
        self._cloud_collaborators: Dict[str, List[CreateCloudCollaborator]] = {}

        # Cloud ID -> default ClusterCompute.
        compute_config = ClusterCompute(
            id=self.DEFAULT_CLUSTER_COMPUTE_ID,
            name=self.DEFAULT_CLUSTER_COMPUTE_NAME,
            config=ClusterComputeConfig(
                cloud_id=self.DEFAULT_CLOUD_ID,
                head_node_type=ComputeNodeType(
                    name="default-head-node",
                    instance_type="m5.2xlarge",
                    resources=ComputeConfigResources(
                        cpu=8, gpu=1, local_vars_configuration=OPENAPI_NO_VALIDATION
                    ),
                ),
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

        workspace_compute_config = ClusterCompute(
            id=self.WORKSPACE_CLUSTER_COMPUTE_ID,
            name=self.WORKSPACE_CLUSTER_COMPUTE_NAME,
            config=ClusterComputeConfig(
                cloud_id=self.WORKSPACE_CLOUD_ID,
                head_node_type=ComputeNodeType(
                    name="default-head-node",
                    instance_type="m5.2xlarge",
                    resources=ComputeConfigResources(
                        cpu=8, gpu=1, local_vars_configuration=OPENAPI_NO_VALIDATION
                    ),
                ),
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

        # Key is (cloud_id, cloud_resource_id) where cloud_resource_id can be None
        self._default_compute_configs: Dict[
            Tuple[str, Optional[str]], ClusterCompute
        ] = {
            (self.DEFAULT_CLOUD_ID, None): compute_config,
            (self.WORKSPACE_CLOUD_ID, None): workspace_compute_config,
        }
        self.add_compute_config(compute_config)
        self.add_compute_config(workspace_compute_config)

        # Add mock internal API client for list_workspaces
        self._internal_api_client = Mock()
        self._internal_api_client.list_workspaces_api_v2_experimental_workspaces_get = (
            self._fake_list_workspaces
        )

    def _fake_list_workspaces(  # noqa: PLR0913, PLR0917
        self,
        project_id=None,
        cloud_id=None,
        name=None,
        creator_id=None,
        state_filter=None,
        tag_filter=None,
        count=50,  # noqa: ARG002
        paging_token=None,  # noqa: ARG002
        sort_field=None,  # noqa: ARG002
        sort_order=None,  # noqa: ARG002
    ):
        """Mock implementation of list_workspaces API for testing."""
        results = list(self._workspaces.values())

        # Filter by name (substring match)
        if name:
            results = [w for w in results if name in w.name]

        # Filter by project
        if project_id:
            results = [w for w in results if w.project_id == project_id]

        # Filter by cloud
        if cloud_id:
            results = [w for w in results if w.cloud_id == cloud_id]

        # Filter by creator
        if creator_id:
            results = [w for w in results if w.creator_id == creator_id]

        # Filter by state
        if state_filter is not None:
            results = [w for w in results if w.state and w.state in state_filter]

        # Filter by tags (tag_filter is List[str] in "key:value" format)
        if tag_filter:
            results = [w for w in results if _matches_tag_filter(w, tag_filter)]

        # Return a mock response with results and metadata
        return Mock(results=results, metadata=Mock(next_paging_token=None))

    def get_user_info(self) -> Any:
        """Return mock user information for testing."""
        # Return a Mock with the minimal attributes needed by SDK code
        return Mock(id=self.DEFAULT_USER_ID, email=self.DEFAULT_USER_EMAIL)

    def list_databricks_connections(self, *, name: Optional[str] = None) -> List[Any]:
        """Return mock Databricks connections for testing."""
        connections = list(self._connections.values())
        if name is not None:
            connections = [c for c in connections if c.name == name]
        return connections

    def get_databricks_connection(self, connection_id: str) -> Any:
        """Get a Databricks connection by ID."""
        if connection_id not in self._connections:
            raise ValueError(f"Connection {connection_id} not found")
        return self._connections[connection_id]

    def add_third_party_connection(
        self,
        connection_id: str,
        name: str,
        connection_type: str = ConnectionType.DATABRICKS_U2M,
    ) -> None:
        """Add a mock third-party connection for testing."""
        mock_conn = Mock()
        mock_conn.id = connection_id
        mock_conn.name = name
        mock_conn.connection_type = connection_type
        self._connections[connection_id] = mock_conn

    def get_job_ui_url(self, job_id: str) -> str:
        return f"{self.BASE_UI_URL}/jobs/{job_id}"

    def get_service_ui_url(self, service_id: str) -> str:
        return f"{self.BASE_UI_URL}/services/{service_id}"

    def get_compute_config_ui_url(
        self, compute_config_id: str, *, cloud_id: str
    ) -> str:
        return f"{self.BASE_UI_URL}/v2/{cloud_id}/compute-configs/{compute_config_id}"

    def set_inside_workspace(
        self,
        inside_workspace: bool,
        *,
        requirements_path: Optional[str] = None,
        cluster_name: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ):
        self._requirements_path = requirements_path
        self._env_vars = env_vars
        if inside_workspace:
            self._workspace_cluster = Cluster(
                id=self.WORKSPACE_CLUSTER_ID,
                name=cluster_name
                if cluster_name is not None
                else WORKSPACE_CLUSTER_NAME_PREFIX + "test",
                project_id=self.WORKSPACE_PROJECT_ID,
                cluster_compute_id=self.WORKSPACE_CLUSTER_COMPUTE_ID,
                cluster_environment_build_id=self.WORKSPACE_CLUSTER_ENV_BUILD_ID,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            )
        else:
            self._workspace_cluster = None

    def get_current_workspace_id(self) -> Optional[str]:
        return (
            self.WORKSPACE_ID
            if self.get_current_workspace_cluster() is not None
            else None
        )

    def inside_workspace(self) -> bool:
        return self.get_current_workspace_cluster() is not None

    def get_workspace_env_vars(self) -> Optional[Dict[str, str]]:
        return self._env_vars

    def get_workspace_requirements_path(self) -> Optional[str]:
        if self.inside_workspace():
            return self._requirements_path
        return None

    def get_current_workspace_cluster(self) -> Optional[Cluster]:
        return self._workspace_cluster

    @property
    def sent_workspace_notifications(self) -> List[WorkspaceNotification]:
        return self._sent_workspace_notifications

    def send_workspace_notification(self, notification: WorkspaceNotification):
        if self.inside_workspace():
            self._sent_workspace_notifications.append(notification)

    def _find_project_cloud_id_tuples_by_name(self, name):
        """Returns list of (cloud_id, project_id) tuples with name == name."""
        project_id_cloud_id_pairs = []
        for cloud_id, cloud_project_dict in self._project_to_id.items():
            for p_name, p_id in cloud_project_dict.items():
                if name == p_name:
                    project_id_cloud_id_pairs.append((cloud_id, p_id))
        return project_id_cloud_id_pairs

    def _get_project_id_by_name(
        self, *, parent_cloud_id: Optional[str] = None, name: Optional[str] = None
    ) -> str:
        # items of existing_projects are (cloud_id, project_id)
        existing_projects = self._find_project_cloud_id_tuples_by_name(name)
        if len(existing_projects) == 0:
            raise ValueError(f"Project '{name}' was not found.")
        else:
            for cloud_id, project_id in existing_projects:
                if cloud_id == parent_cloud_id:
                    return project_id
            raise ValueError(
                f"{len(existing_projects)} project(s) found with name '{name}' and none matched cloud_id '{parent_cloud_id}'"
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

        return self.DEFAULT_PROJECT_ID

    def get_project_id(
        self,
        *,
        parent_cloud_id: Optional[str] = None,  # noqa: ARG002
        name: Optional[str] = None,  # noqa: ARG002
    ) -> str:
        if name is not None:
            return self._get_project_id_by_name(
                parent_cloud_id=parent_cloud_id, name=name
            )
        else:
            return self._get_project_id_by_cloud_id(parent_cloud_id=parent_cloud_id)

    def get_cloud_id(
        self,
        *,
        cloud_name: Optional[str] = None,
        compute_config_id: Optional[str] = None,
    ) -> str:
        assert not (cloud_name and compute_config_id)
        workspace_cluster = self.get_current_workspace_cluster()
        if workspace_cluster is not None:
            return self.WORKSPACE_CLOUD_ID

        if compute_config_id is not None:
            return self._compute_configs[compute_config_id].config.cloud_id

        if cloud_name is None:
            return self.DEFAULT_CLOUD_ID

        for cloud in self._clouds.values():
            if cloud.name == cloud_name:
                return cloud.id

        raise RuntimeError(f"Cloud with name '{cloud_name}' not found.")

    def get_image_uri_from_build_id(self, build_id: str) -> Optional[ImageURI]:
        cluster_env_build_id = self._builds[build_id].cluster_environment_id
        return self.get_cluster_env_build_image_uri(cluster_env_build_id)

    def add_build(self, build: ClusterEnvironmentBuild):
        self._builds[build.id] = build
        self._touch_application_template(build.cluster_environment_id)

    def add_image(self, image: ClusterEnvironment):
        self._images[image.id] = image
        self._ensure_application_template_metadata(
            image.id,
            name=image.name,
            creator_id=getattr(image, "creator_id", None),
            anonymous=getattr(image, "anonymous", False),
            is_default=getattr(image, "is_default", False),
        )

    def _ensure_application_template_metadata(  # noqa: PLR0913
        self,
        env_id: str,
        *,
        name: Optional[str] = None,
        creator_id: Optional[str] = None,
        creator_email: Optional[str] = None,
        creator_name: Optional[str] = None,
        anonymous: Optional[bool] = None,
        is_default: Optional[bool] = None,
        project_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        last_modified_at: Optional[datetime] = None,
    ) -> None:
        meta = self._application_template_metadata.setdefault(
            env_id,
            {
                "name": name or env_id,
                "creator_id": creator_id or self.DEFAULT_USER_ID,
                "creator_email": creator_email or self.DEFAULT_USER_EMAIL,
                "creator_name": creator_name or (creator_id or self.DEFAULT_USER_ID),
                "anonymous": anonymous if anonymous is not None else False,
                "is_default": is_default if is_default is not None else False,
                "project_id": project_id,
                "created_at": created_at or datetime.utcnow(),
                "last_modified_at": last_modified_at or datetime.utcnow(),
                "deleted_at": None,
            },
        )

        if name is not None:
            meta["name"] = name
        if creator_id is not None:
            meta["creator_id"] = creator_id
        if creator_email is not None:
            meta["creator_email"] = creator_email
        if creator_name is not None:
            meta["creator_name"] = creator_name
        if anonymous is not None:
            meta["anonymous"] = anonymous
        if is_default is not None:
            meta["is_default"] = is_default
        if project_id is not None:
            meta["project_id"] = project_id
        if created_at is not None:
            meta["created_at"] = created_at
        if last_modified_at is not None:
            meta["last_modified_at"] = last_modified_at

    def _touch_application_template(self, env_id: str) -> None:
        meta = self._application_template_metadata.get(env_id)
        if meta is not None:
            meta["last_modified_at"] = datetime.utcnow()

    def _get_latest_build_for_env(
        self, env_id: str
    ) -> Optional[ClusterEnvironmentBuild]:
        latest: Optional[ClusterEnvironmentBuild] = None
        for build in self._builds.values():
            if build.cluster_environment_id == env_id:
                build_revision = build.revision or 0
                if latest is None or build_revision > (latest.revision or 0):
                    latest = build
        return latest

    def _build_application_template_model(
        self, env_id: str
    ) -> Optional[DecoratedApplicationTemplate]:
        env = self._images.get(env_id)
        meta = self._application_template_metadata.get(env_id)
        if env is None or meta is None:
            return None

        latest_build = self._get_latest_build_for_env(env_id)
        mini_build = None
        docker_image_name = None
        if latest_build is not None:
            docker_image_name = latest_build.docker_image_name
            if docker_image_name is None:
                image_uri = self.get_cluster_env_build_image_uri(
                    latest_build.id, use_image_alias=True
                )
                docker_image_name = image_uri.image_uri if image_uri else None

            status_value = None
            if latest_build.status is not None:
                status_value = (
                    latest_build.status.value.lower()
                    if hasattr(latest_build.status, "value")
                    else str(latest_build.status).lower()
                )

            mini_build = MiniBuild(
                id=latest_build.id,
                revision=latest_build.revision,
                status=status_value,
                application_template_name=meta.get("name"),
                application_template_id=env_id,
                docker_image_name=docker_image_name,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            )

        creator = MiniUser(
            id=meta.get("creator_id"),
            name=meta.get("creator_name"),
            username=meta.get("creator_name"),
            email=meta.get("creator_email"),
            lastname=None,
            deleted_at=None,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

        return DecoratedApplicationTemplate(
            id=env_id,
            name=meta.get("name"),
            project_id=meta.get("project_id"),
            organization_id=self.DEFAULT_ORGANIZATION_ID,
            creator_id=meta.get("creator_id"),
            created_at=meta.get("created_at"),
            last_modified_at=meta.get("last_modified_at"),
            deleted_at=meta.get("deleted_at"),
            archiver_id=None,
            archived_at=None,
            is_default=meta.get("is_default", False),
            anonymous=meta.get("anonymous", False),
            creator=creator,
            latest_build=mini_build,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

    def add_cloud(self, cloud: Cloud):
        self._clouds[cloud.id] = cloud
        self._system_cluster_status[cloud.id] = ClusterState.RUNNING

        # Create a default compute config for this cloud
        if cloud.id not in self._default_compute_configs:
            default_compute_config = ClusterCompute(
                id=f"default-compute-config-{cloud.id}",
                name=f"default-compute-config-{cloud.name}",
                config=ClusterComputeConfig(
                    cloud_id=cloud.id,
                    head_node_type=ComputeNodeType(
                        name="default-head-node",
                        instance_type="m5.2xlarge",
                        resources=ComputeConfigResources(
                            cpu=8, gpu=1, local_vars_configuration=OPENAPI_NO_VALIDATION
                        ),
                    ),
                    local_vars_configuration=OPENAPI_NO_VALIDATION,
                ),
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            )
            self._default_compute_configs[cloud.id] = default_compute_config
            self.add_compute_config(default_compute_config)

    def get_cloud(self, *, cloud_id: str) -> Optional[Cloud]:
        return self._clouds.get(cloud_id, None)

    def get_cloud_by_name(self, *, name: str) -> Optional[Cloud]:
        for c in self._clouds.values():
            if c.name == name:
                return c
        return None

    def get_cloud_resource_by_name(
        self, cloud_id: str, cloud_resource_name: str
    ) -> Optional[DecoratedCloudResource]:
        for cloud_resource in self._cloud_resources.values():
            if (
                cloud_resource.cloud_deployment_id == cloud_id
                and cloud_resource.name == cloud_resource_name
            ):
                return cloud_resource
        return None

    def add_cloud_resource(self, cloud_resource: DecoratedCloudResource) -> None:
        self._cloud_resources[cloud_resource.cloud_resource_id] = cloud_resource

    def get_default_cloud(self) -> Optional[Cloud]:
        return self._clouds.get(self.DEFAULT_CLOUD_ID, None)

    def list_clouds(
        self, *, paging_token: Optional[str] = None, count: Optional[int] = None
    ) -> CloudListResponse:
        # Simple fake: ignore paging_token and just return up to `count` items
        clouds = list(self._clouds.values())
        if count is not None:
            clouds = clouds[:count]
        return CloudListResponse(
            results=clouds,
            metadata=ListResponseMetadata(
                next_paging_token=paging_token,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

    def add_cloud_collaborators(
        self, cloud_id: str, collaborators: List[CreateCloudCollaborator]
    ) -> None:
        existing_collaborators = [
            collaborator.email
            for collaborator in self._cloud_collaborators.get(cloud_id, [])
        ]

        for collaborator in collaborators:
            if collaborator.email in existing_collaborators:
                raise ValueError(
                    f"Collaborator with email '{collaborator.email}' already exists in cloud '{cloud_id}'."
                )

        if cloud_id not in self._cloud_collaborators:
            self._cloud_collaborators[cloud_id] = collaborators
        else:
            self._cloud_collaborators[cloud_id].extend(collaborators)

    def terminate_system_cluster(
        self, cloud_id: str
    ) -> Optional[ClusteroperationResponse]:
        self._system_cluster_status[cloud_id] = ClusterState.TERMINATING
        return ClusteroperationResponse(
            result=ClusterOperation(
                id="sop_123",
                completed=False,
                progress=None,
                result=None,
                cluster_id="fake-system-cluster-id",
                cluster_operation_type="terminate",
            )
        )

    def describe_system_workload_get_status(self, cloud_id: str) -> str:
        self._system_cluster_status[cloud_id] = ClusterState.TERMINATED
        return self._system_cluster_status[cloud_id]

    def add_compute_config(self, compute_config: DecoratedComputeTemplate) -> int:
        compute_config.version = (
            len(self._compute_config_name_to_ids[compute_config.name]) + 1
        )
        self._compute_configs[compute_config.id] = compute_config
        self._compute_config_name_to_ids[compute_config.name].append(compute_config.id)

        return compute_config.version

    def create_compute_config(
        self, config: ComputeTemplateConfig, *, name: Optional[str] = None
    ) -> Tuple[str, str]:
        unique_id = str(uuid.uuid4())
        compute_config_id = f"compute-config-id-{unique_id}"
        if name is None:
            anonymous = True
            name = f"anonymous-compute-config-{unique_id}"
        else:
            anonymous = False

        version = self.add_compute_config(
            DecoratedComputeTemplate(
                id=compute_config_id,
                name=name,
                config=config,
                anonymous=anonymous,
                creator_id=self.DEFAULT_USER_ID,
                created_at=datetime.utcnow(),
                last_modified_at=datetime.utcnow(),
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
        )
        return f"{name}:{version}", compute_config_id

    def get_compute_config(
        self, compute_config_id: str
    ) -> Optional[DecoratedComputeTemplate]:
        if compute_config_id in self._compute_configs:
            return self._compute_configs[compute_config_id]

        if compute_config_id in self._archived_compute_configs:
            return self._archived_compute_configs[compute_config_id]

        return None

    def get_compute_config_id(
        self,
        compute_config_name: Optional[str] = None,
        cloud: Optional[str] = None,
        *,
        include_archived=False,
    ) -> Optional[str]:
        if compute_config_name is not None:
            name, version = parse_cluster_compute_name_version(compute_config_name)
            if name not in self._compute_config_name_to_ids:
                return None

            if version is None:
                version = len(self._compute_config_name_to_ids[name])

            compute_config_id = self._compute_config_name_to_ids[name][version - 1]
            if (
                not include_archived
                and compute_config_id in self._archived_compute_configs
            ):
                return None

            return compute_config_id

        workspace_cluster = self.get_current_workspace_cluster()
        if workspace_cluster is not None:
            return workspace_cluster.cluster_compute_id

        cloud_id = self.get_cloud_id(cloud_name=cloud)
        return self.get_default_compute_config(cloud_id=cloud_id).id

    def archive_compute_config(self, *, compute_config_id: str):
        archived_config = self._compute_configs.pop(compute_config_id)
        archived_config.archived_at = datetime.utcnow()
        self._archived_compute_configs[compute_config_id] = archived_config

    def search_cluster_computes(self, query: Dict[str, Any]):  # noqa: PLR0912
        """Search for cluster computes matching the provided query.

        Implements filtering, sorting, and pagination for testing purposes.
        """
        # Create a mock response object that matches the expected structure
        class MockSearchResponse:
            def __init__(self, results_list, next_token=None):
                self.results = results_list
                self.metadata = Mock()
                self.metadata.next_paging_token = next_token

        # Start with all compute configs
        results = list(self._compute_configs.values())

        # Apply filtering
        if "name" in query and "equals" in query["name"]:
            target_name = query["name"]["equals"]
            results = [cc for cc in results if cc.name == target_name]

        if "creator_id" in query:
            creator_id = query["creator_id"]
            results = [cc for cc in results if cc.creator_id == creator_id]

        if "cloud_id" in query:
            cloud_id = query["cloud_id"]
            results = [
                cc for cc in results if cc.config and cc.config.cloud_id == cloud_id
            ]

        # Apply sorting
        if "sort_by_clauses" in query:
            for sort_clause in reversed(query["sort_by_clauses"]):
                sort_field = (
                    sort_clause.get("sort_field")
                    if isinstance(sort_clause, dict)
                    else getattr(sort_clause, "sort_field", None)
                )
                sort_order = (
                    sort_clause.get("sort_order")
                    if isinstance(sort_clause, dict)
                    else getattr(sort_clause, "sort_order", None)
                )

                # Convert enum to string if needed
                if sort_field is not None and hasattr(sort_field, "value"):
                    sort_field = sort_field.value  # type: ignore
                if sort_order is not None and hasattr(sort_order, "value"):
                    sort_order = sort_order.value  # type: ignore

                reverse = sort_order == "DESC"

                if sort_field == "NAME":
                    results.sort(key=lambda x: x.name or "", reverse=reverse)
                elif sort_field == "CREATED_AT":
                    results.sort(
                        key=lambda x: x.created_at or datetime.min, reverse=reverse
                    )
                elif sort_field == "LAST_MODIFIED_AT":
                    results.sort(
                        key=lambda x: x.last_modified_at or datetime.min,
                        reverse=reverse,
                    )

        # Apply pagination
        paging = query.get("paging", {})
        max_items = paging.get("count", len(results))
        paging_token = paging.get("paging_token")

        start_index = 0
        if paging_token:
            try:
                start_index = int(paging_token)
            except (ValueError, TypeError):
                start_index = 0

        paginated_results = results[start_index : start_index + max_items]

        # Calculate next token
        next_token = None
        if start_index + max_items < len(results):
            next_token = str(start_index + max_items)

        return MockSearchResponse(paginated_results, next_token)

    def is_archived_compute_config(self, compute_config_id: str) -> bool:
        return compute_config_id in self._archived_compute_configs

    def set_default_compute_config(
        self,
        compute_config: ClusterCompute,
        *,
        cloud_id: str,
        cloud_resource_id: Optional[str] = None,
    ):
        self._default_compute_configs[(cloud_id, cloud_resource_id)] = compute_config

    def get_default_compute_config(
        self, *, cloud_id: str, cloud_resource_id: Optional[str] = None
    ) -> ClusterCompute:
        key = (cloud_id, cloud_resource_id)
        if key in self._default_compute_configs:
            return self._default_compute_configs[key]
        # Fallback to default cloud's config if specific cloud doesn't have one
        return self._default_compute_configs[(self.DEFAULT_CLOUD_ID, None)]

    def list_cluster_env_builds(
        self, cluster_env_id: str,
    ) -> Generator[ClusterEnvironmentBuild, None, None]:
        for v in self._builds.values():
            if v.cluster_environment_id == cluster_env_id:
                yield v

    def get_non_default_cluster_env_builds(self) -> List[ClusterEnvironmentBuild]:
        return [
            v
            for v in self._builds.values()
            if v.id
            not in [
                self.DEFAULT_CLUSTER_ENV_BUILD_ID,
                self.WORKSPACE_CLUSTER_ENV_BUILD_ID,
            ]
        ]

    def get_default_build_id(self) -> str:
        workspace_cluster = self.get_current_workspace_cluster()
        if workspace_cluster is not None:
            return workspace_cluster.cluster_environment_build_id
        return self.DEFAULT_CLUSTER_ENV_BUILD_ID

    def get_cluster_env_build(self, build_id: str) -> Optional[ClusterEnvironmentBuild]:
        return self._builds.get(build_id, None)

    def get_cluster_env_by_name(self, name) -> Optional[ClusterEnvironment]:
        for v in self._images.values():
            if v.name == name:
                return v
        return None

    def get_cluster_env_build_id_from_containerfile(
        self,
        cluster_env_name: str,
        containerfile: str,
        anonymous: bool,
        ray_version: Optional[str] = None,  # noqa: ARG002
    ) -> str:
        for build in self._builds.values():
            if build.containerfile == containerfile:
                cluster_env = self._images.get(build.cluster_environment_id, None)  # type: ignore
                if cluster_env is not None and cluster_env.name == cluster_env_name:
                    return build.id  # type: ignore
        # create a new one if not found
        cluster_env_id = f"cluster-env-id-{uuid.uuid4()!s}"
        self._images[cluster_env_id] = ClusterEnvironment(
            id=cluster_env_id,
            name=cluster_env_name,
            anonymous=anonymous,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )
        self._ensure_application_template_metadata(
            cluster_env_id, name=cluster_env_name, anonymous=anonymous,
        )
        latest_build = None
        for build in self._builds.values():
            if build.cluster_environment_id == cluster_env_id and (latest_build is None or build.revision > latest_build.revision):  # type: ignore
                latest_build = build
        build_id = f"cluster-env-build-id-{uuid.uuid4()!s}"
        build = ClusterEnvironmentBuild(
            id=build_id,
            cluster_environment_id=cluster_env_id,
            containerfile=containerfile,
            status=ClusterEnvironmentBuildStatus.SUCCEEDED,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
            ray_version=ray_version,
            revision=latest_build.revision + 1 if latest_build is not None else 1,  # type: ignore
        )
        self.add_build(build)
        return build_id

    def get_cluster_env_build_id_from_image_uri(
        self,
        image_uri: ImageURI,
        registry_login_secret: Optional[str] = None,
        ray_version: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        for build in self._builds.values():
            build_image_uri = self.get_cluster_env_build_image_uri(build.id)
            if (
                build_image_uri.image_uri == image_uri.image_uri
                if build_image_uri
                else False
            ):
                return build.id  # type: ignore
        cluster_env_id = f"cluster-env-id-{uuid.uuid4()!s}"
        cluster_env_name = name if name else image_uri.to_cluster_env_name()
        self._images[cluster_env_id] = ClusterEnvironment(
            id=cluster_env_id,
            name=cluster_env_name,
            anonymous=False,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )
        self._ensure_application_template_metadata(
            cluster_env_id, name=cluster_env_name, anonymous=False,
        )
        build_id = f"cluster-env-build-id-{uuid.uuid4()!s}"
        build = ClusterEnvironmentBuild(
            id=build_id,
            cluster_environment_id=cluster_env_id,
            docker_image_name=image_uri.image_uri,
            status=ClusterEnvironmentBuildStatus.SUCCEEDED,
            registry_login_secret=registry_login_secret,
            ray_version=ray_version if ray_version else LATEST_RAY_VERSION,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )
        self.add_build(build)
        return build_id

    def get_cluster_env_build_image_uri(
        self, cluster_env_build_id: str, use_image_alias: bool = False
    ) -> Optional[ImageURI]:
        build = self._builds.get(cluster_env_build_id, None)
        if build is None:
            return None
        if build.docker_image_name is not None and not use_image_alias:
            return ImageURI.from_str(build.docker_image_name)
        cluster_env = self._images[build.cluster_environment_id]
        return ImageURI.from_cluster_env_build(cluster_env, build)

    def archive_image(self, *, image_id: str) -> None:
        """Archive an image (cluster environment) by ID."""
        if image_id not in self._images:
            raise ValueError(f"Image '{image_id}' not found.")
        image = self._images[image_id]
        image.archived_at = datetime.utcnow()
        self._archived_images[image_id] = self._images.pop(image_id)

    def get_application_template(
        self, application_template_id: str
    ) -> Optional[DecoratedApplicationTemplate]:
        return self._build_application_template_model(application_template_id)

    def list_application_templates(  # noqa: PLR0912
        self,
        *,
        name: Optional[str],
        image_name: Optional[str],
        creator_id: Optional[str],
        project: Optional[str],
        include_archived: bool,
        defaults_first: bool,
        count: Optional[int],
        paging_token: Optional[str],
    ) -> DecoratedapplicationtemplateListResponse:
        templates: List[DecoratedApplicationTemplate] = []
        for env_id in list(self._application_template_metadata.keys()):
            template = self._build_application_template_model(env_id)
            if template is None:
                continue

            meta = self._application_template_metadata.get(env_id, {})
            if not include_archived and meta.get("deleted_at") is not None:
                continue
            if name and (template.name or "").find(name) == -1:
                continue
            if creator_id and template.creator_id != creator_id:
                continue

            if project:
                try:
                    project_id_filter = self.get_project_id(name=project)
                except Exception:  # noqa: BLE001
                    project_id_filter = project
                if project_id_filter and template.project_id != project_id_filter:
                    continue

            if image_name:
                latest_build = template.latest_build
                docker_name = (
                    latest_build.docker_image_name if latest_build is not None else None
                )
                if not docker_name or image_name not in docker_name:
                    continue

            templates.append(template)

        if defaults_first:
            templates.sort(
                key=lambda tpl: (
                    0 if tpl.is_default else 1,
                    -(
                        tpl.created_at.timestamp()
                        if tpl.created_at is not None
                        else 0.0
                    ),
                )
            )
        else:
            templates.sort(
                key=lambda tpl: tpl.created_at or datetime.fromtimestamp(0),
                reverse=True,
            )

        total = len(templates)
        start_index = 0
        if paging_token:
            try:
                start_index = int(paging_token)
            except ValueError:
                start_index = 0

        page_count = count if count is not None else total
        page = templates[start_index : start_index + page_count]
        next_token = (
            str(start_index + page_count)
            if (start_index + page_count) < total
            else None
        )

        metadata = SDKListResponseMetadata(
            total=total,
            next_paging_token=next_token,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )
        return DecoratedapplicationtemplateListResponse(
            results=page,
            metadata=metadata,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

    def update_service(self, model: DecoratedProductionServiceV2APIModel):
        self._services[model.id] = model
        if model.versions is not None:
            for version in model.versions:
                self._versions[model.id][version.id] = version

    def get_service(
        self,
        name: str,
        *,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> Optional[DecoratedProductionServiceV2APIModel]:
        cloud_id = self.get_cloud_id(cloud_name=cloud)
        cloud_project_dict = self._project_to_id.get(cloud_id, None)
        project_id = (
            cloud_project_dict.get(project, None) if cloud_project_dict else None
        )
        for service in self._services.values():
            if service.name == name and (
                project_id is None or service.project_id == project_id
            ):
                return service

        if include_archived:
            for service in self._archived_services.values():
                if service.name == name and (
                    project_id is None or service.project_id == project_id
                ):
                    return service

        return None

    def build_project_with_args(self, **kwargs) -> Project:
        # set values for required fields if not provided
        if "id" not in kwargs:
            kwargs["id"] = f"project-id-{uuid.uuid4()!s}"
        if "name" not in kwargs:
            kwargs["name"] = f"project-{kwargs['id']}"
        if "description" not in kwargs:
            kwargs["description"] = f"project-description-{kwargs['id']}"
        if "parent_cloud_id" not in kwargs:
            kwargs["parent_cloud_id"] = self.DEFAULT_CLOUD_ID
        if "created_at" not in kwargs:
            kwargs["created_at"] = datetime.utcnow()
        if "is_owner" not in kwargs:
            kwargs["is_owner"] = True
        if "is_read_only" not in kwargs:
            kwargs["is_read_only"] = False
        if "directory_name" not in kwargs:
            kwargs["directory_name"] = "default"
        if "is_default" not in kwargs:
            kwargs["is_default"] = False
        return Project(**kwargs, local_vars_configuration=OPENAPI_NO_VALIDATION)

    def create_project_with_args(self, **kwargs) -> str:
        project = self.build_project_with_args(**kwargs)
        project_id: str = project.id  # type: ignore
        self._projects[project_id] = project
        return project_id

    def get_project_by_id_or_name(
        self, *, project_id: Optional[str] = None, project_name: Optional[str] = None,
    ) -> Optional[Project]:
        if project_id:
            return self._projects.get(project_id, None)
        if project_name:
            for project in self._projects.values():
                if project.name == project_name:
                    return project
        return None

    def get_project(self, project_id: str) -> Optional[Project]:
        for cloud_project_dict in self._project_to_id.values():
            for p_name, p_id in cloud_project_dict.items():
                if p_id == project_id:
                    # return stub project
                    return self.build_project_with_args(id=p_id, name=p_name,)
        return None

    def list_projects(
        self,
        *,
        name_contains: Optional[str] = None,
        creator_id: Optional[str] = None,
        parent_cloud_id: Optional[str] = None,
        include_defaults: bool = True,
        sort_field: Optional[str] = None,  # noqa: ARG002
        sort_order: Optional[str] = None,  # noqa: ARG002
        paging_token: Optional[str] = None,  # noqa: ARG002
        count: Optional[int] = None,
    ) -> ProjectListResponse:
        projects = list(self._projects.values())
        if name_contains:
            projects = [p for p in projects if p.name and name_contains in p.name]
        if creator_id:
            projects = [p for p in projects if p.creator_id == creator_id]
        if parent_cloud_id:
            projects = [p for p in projects if p.parent_cloud_id == parent_cloud_id]
        if not include_defaults:
            projects = [p for p in projects if not p.is_default]
        if sort_field and sort_order and sort_field == "NAME":
            projects.sort(
                key=lambda x: x.name if x.name else "", reverse=sort_order == "DESC"
            )
        if count:
            projects = projects[:count]
        return ProjectListResponse(
            results=projects,
            metadata=ListResponseMetadata(
                next_paging_token=paging_token,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

    def create_project(self, project: WriteProject) -> ProjectBase:
        project_id = f"project-id-{uuid.uuid4()!s}"
        self._projects[project_id] = Project(
            id=project_id,
            name=project.name,
            description=project.description,
            cloud_id=project.cloud_id,
            initial_cluster_config=project.initial_cluster_config,
            parent_cloud_id=project.parent_cloud_id,
            created_at=datetime.utcnow(),
            creator_id=self.DEFAULT_USER_ID,
            is_default=False,
            is_owner=True,
            is_read_only=False,
            directory_name="default",
            owners=[
                MiniUser(
                    id=self.DEFAULT_USER_ID,
                    email=self.DEFAULT_USER_EMAIL,
                    local_vars_configuration=OPENAPI_NO_VALIDATION,
                )
            ],
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )
        return ProjectBase(
            id=project_id, local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

    def delete_project(self, project_id: str) -> None:
        if project_id not in self._projects:
            raise ValueError(f"Project {project_id} not found")
        self._projects.pop(project_id)

    def get_default_project(self, parent_cloud_id: str) -> Project:
        for project in self._projects.values():
            if project.parent_cloud_id == parent_cloud_id and project.is_default:
                return project
        raise ValueError(f"No default project found for cloud {parent_cloud_id}")

    def add_project_collaborators(
        self, project_id: str, collaborators: List[CreateUserProjectCollaborator]
    ):
        existing_collaborators = [
            collaborator.value.email if collaborator.value else None
            for collaborator in self._project_collaborators.get(project_id, [])
        ]

        for collaborator in collaborators:
            if (
                collaborator.value
                and collaborator.value.email in existing_collaborators
            ):
                raise ValueError(
                    f"Collaborator with email '{collaborator.value.email}' already exists in project '{project_id}'."
                )

        if project_id not in self._project_collaborators:
            self._project_collaborators[project_id] = collaborators
        else:
            self._project_collaborators[project_id].extend(collaborators)

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
            if job_id in self._jobs:
                return self._jobs[job_id]
            if job_id in self._archived_jobs:
                return self._archived_jobs[job_id]
            return None
        else:
            # Name-based lookup: respect archive filtering
            cloud_id = self.get_cloud_id(cloud_name=cloud)
            cloud_project_dict = self._project_to_id.get(cloud_id, None)
            project_id = (
                cloud_project_dict.get(project, None) if cloud_project_dict else None
            )

            # Build the list of jobs to search
            target_jobs = list(self._jobs.values())
            if include_archived:
                target_jobs.extend(list(self._archived_jobs.values()))

            result: ProductionJob = None
            for job in target_jobs:
                if (
                    job is not None
                    and job.name == name
                    and (project_id is None or job.project_id == project_id)
                    and (result is None or job.created_at > result.created_at)
                ):
                    result = job

        return result

    def get_job_runs(self, job_id: str) -> List[APIJobRun]:
        return self._job_runs.get(job_id, [])

    def update_job(self, model: ProductionJob):
        self._jobs[model.id] = model

    def update_job_run(self, prod_job_id: str, model: APIJobRun):
        self._job_runs[prod_job_id].append(model)

    def list_job_queues(  # noqa: PLR0913, PLR0917
        self,
        *,
        name: Optional[str] = None,
        creator_id: Optional[str] = None,  # noqa: ARG002
        cluster_status: Optional[SessionState] = None,  # noqa: ARG002
        project: Optional[str] = None,  # noqa: ARG002
        cloud: Optional[str] = None,  # noqa: ARG002
        tags_filter: Optional[Dict[str, List[str]]] = None,  # noqa: ARG002
        count: Optional[int] = None,
        paging_token: Optional[str] = None,  # noqa: ARG002
        _sorting_directives: Optional[
            List[JobQueueSortDirective]
        ] = None,  # noqa: ARG002
        include_archived: bool = False,  # noqa: ARG002
    ) -> DecoratedjobqueueListResponse:
        """Mock implementation of list_job_queues API for testing.

        Returns a DecoratedjobqueueListResponse object to match the real API.
        """
        results = list(self._job_queues.values())
        if name:
            results = [jq for jq in results if jq.name == name]
        if count is not None:
            results = results[:count]

        return DecoratedjobqueueListResponse(
            results=results,
            metadata=ListResponseMetadata(total=len(results), next_paging_token=None),
        )

    def list_jobs(  # noqa: PLR0913, PLR0917
        self,
        *,
        name: Optional[str] = None,
        project_id: Optional[str] = None,  # noqa: ARG002
        creator_id: Optional[str] = None,  # noqa: ARG002
        state_filter: Optional[List[str]] = None,  # noqa: ARG002
        archive_status: Optional[str] = None,  # noqa: ARG002
        tags_filter: Optional[List[str]] = None,  # noqa: ARG002
        count: Optional[int] = None,
        paging_token: Optional[str] = None,  # noqa: ARG002
        sort_field: Optional[str] = None,  # noqa: ARG002
        sort_order: Optional[str] = None,  # noqa: ARG002
    ) -> DecoratedproductionjobListResponse:
        """Mock implementation of list_jobs API for testing.

        Returns a DecoratedproductionjobListResponse object to match the real API.
        """
        results = list(self._jobs.values())
        if name:
            results = [job for job in results if job.name == name]
        if count is not None:
            results = results[:count]

        return DecoratedproductionjobListResponse(
            results=results,
            metadata=ListResponseMetadata(total=len(results), next_paging_token=None),
        )

    def get_job_queue(self, job_queue_id: str) -> Optional[DecoratedJobQueue]:
        return self._job_queues.get(job_queue_id, None)

    def update_job_queue(self, model: DecoratedJobQueue):
        self._job_queues[model.id] = model

    def delete_job_queue(self, job_queue_id: str) -> None:
        """Mock implementation of delete_job_queue API for testing."""
        if job_queue_id in self._job_queues:
            del self._job_queues[job_queue_id]

    def register_project_by_name(
        self,
        name: str,
        cloud: str = DEFAULT_CLOUD_NAME,
        project_id: Optional[str] = None,
    ) -> str:
        """Helper method to create project name to project id mapping."""
        cloud_id = self.get_cloud_id(cloud_name=cloud)
        if cloud_id not in self._project_to_id:
            self._project_to_id[cloud_id] = {}
        cloud_project_dict = self._project_to_id[cloud_id]
        if name in cloud_project_dict:
            return cloud_project_dict[name]
        if project_id is None:
            project_id = f"project-id-{uuid.uuid4()!s}"
        cloud_project_dict[name] = project_id
        return project_id

    @property
    def rolled_out_model(
        self,
    ) -> Optional[
        Union[ApplyProductionServiceV2Model, ApplyProductionServiceMultiVersionV2Model]
    ]:
        return self._rolled_out_model

    def rollout_service(
        self, model: ApplyProductionServiceV2Model
    ) -> DecoratedProductionServiceV2APIModel:
        self._rolled_out_model = model
        # TODO(mowen): This feels convoluted, is there a better way to pull cloud name and project name from the model?
        project_model = self.get_project(model.project_id)
        project = project_model.name if project_model else None
        compute_config = self.get_compute_config(model.compute_config_id)
        cloud_id = compute_config.config.cloud_id if compute_config else None
        cloud_model = self.get_cloud(cloud_id=cloud_id)
        cloud = cloud_model.name if cloud_model else None
        existing_service = self.get_service(model.name, project=project, cloud=cloud)
        if existing_service is not None:
            service_id = existing_service.id
        else:
            service_id = f"service-id-{uuid.uuid4()!s}"

        primary_version = ProductionServiceV2VersionModel(
            id=str(uuid.uuid4()),
            created_at=datetime.now(),
            version=model.version,
            current_state=ServiceVersionState.RUNNING,
            weight=100,
            build_id=model.build_id,
            compute_config_id=model.compute_config_id,
            ray_serve_config=model.ray_serve_config,
            ray_gcs_external_storage_config=model.ray_gcs_external_storage_config,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )
        service = DecoratedProductionServiceV2APIModel(
            id=service_id,
            name=model.name,
            project_id=model.project_id,
            cloud_id=self.get_cloud_id(compute_config_id=model.compute_config_id),
            current_state=ServiceEventCurrentState.STARTING,
            base_url=f"http://{model.name}.fake.url",
            auth_token="fake-auth-token",
            primary_version=primary_version,
            versions=[primary_version],
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )
        self.update_service(service)
        return service

    def rollout_service_multi_version(
        self, model: ApplyProductionServiceMultiVersionV2Model
    ) -> DecoratedProductionServiceV2APIModel:
        self._rolled_out_model = model
        version = model.service_versions[0]
        project_model = self.get_project(version.project_id)
        project = project_model.name if project_model else None
        compute_config = self.get_compute_config(version.compute_config_id)
        cloud_id = compute_config.config.cloud_id if compute_config else None
        cloud_model = self.get_cloud(cloud_id=cloud_id)
        cloud = cloud_model.name if cloud_model else None
        existing_service = self.get_service(version.name, project=project, cloud=cloud)
        if existing_service is not None:
            service_id = existing_service.id
        else:
            service_id = f"service-id-{uuid.uuid4()!s}"

        service_versions = []
        for version in model.service_versions:
            service_version = ProductionServiceV2VersionModel(
                id=str(uuid.uuid4()),
                created_at=datetime.now(),
                version=version.version,
                current_state=ServiceVersionState.RUNNING,
                weight=version.traffic_percent,
                build_id=version.build_id,
                compute_config_id=version.compute_config_id,
                ray_serve_config=version.ray_serve_config,
                ray_gcs_external_storage_config=version.ray_gcs_external_storage_config,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            )
            service_versions.append(service_version)

        service = DecoratedProductionServiceV2APIModel(
            id=service_id,
            name=version.name,
            project_id=version.project_id,
            cloud_id=self.get_cloud_id(compute_config_id=version.compute_config_id),
            current_state=ServiceEventCurrentState.STARTING,
            base_url=f"http://{version.name}.fake.url",
            auth_token="fake-auth-token",
            primary_version="",
            versions=service_versions,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )
        self.update_service(service)
        return service

    @property
    def submitted_job(self) -> Optional[CreateInternalProductionJob]:
        return self._submitted_job

    @property
    def rolled_back_service(self) -> Optional[Tuple[str, Optional[int]]]:
        return self._rolled_back_service

    def rollback_service(
        self, service_id: str, *, max_surge_percent: Optional[int] = None
    ):
        self._rolled_back_service = (service_id, max_surge_percent)

    @property
    def terminated_service(self) -> Optional[str]:
        return self._terminated_service

    def terminate_service(self, service_id: str):
        self._terminated_service = service_id
        self._services[service_id].current_state = ServiceEventCurrentState.TERMINATED
        self._services[service_id].canary_version = None
        if self._services[service_id].primary_version is not None:
            # The backend leaves the primary_version populated upon termination.
            self._services[service_id].primary_version.weight = 100
            self._services[
                service_id
            ].primary_version.current_state = ServiceVersionState.TERMINATED

    @property
    def archived_services(self) -> Dict[str, DecoratedProductionServiceV2APIModel]:
        return self._archived_services

    def archive_service(self, service_id: str):
        self._archived_services[service_id] = self._services.pop(service_id)

    @property
    def deleted_services(self) -> Dict[str, DecoratedProductionServiceV2APIModel]:
        return self._deleted_services

    def delete_service(self, service_id: str):
        self._deleted_services[service_id] = self._services.pop(service_id)

    def submit_job(self, model: CreateInternalProductionJob) -> InternalProductionJob:
        self._submitted_job = model

        job = InternalProductionJob(
            id=f"job-{uuid.uuid4()!s}",
            name=model.name,
            config=model.config,
            state=ProductionJobStateTransition(
                current_state=HaJobStates.PENDING,
                goal_state=HaJobGoalStates.SUCCESS,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            project_id=model.project_id,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

        self.update_job(job)
        return job

    def terminate_job(self, job_id: str):
        if job_id in self._jobs:
            self._jobs[job_id].state = HaJobStates.TERMINATED
        elif job_id in self._archived_jobs:
            self._archived_jobs[job_id].state = HaJobStates.TERMINATED

    def archive_job(self, job_id: str):
        if job_id in self._archived_jobs:
            # Already archived, idempotent
            return
        self._archived_jobs[job_id] = self._jobs.pop(job_id)

    def is_archived_job(self, job_id: str) -> bool:
        return job_id in self._archived_jobs

    def delete_job(self, job_id: str) -> None:
        if job_id in self._jobs:
            self._deleted_jobs[job_id] = self._jobs.pop(job_id)
        elif job_id in self._archived_jobs:
            self._deleted_jobs[job_id] = self._archived_jobs.pop(job_id)

    def is_deleted_job(self, job_id: str) -> bool:
        return job_id in self._deleted_jobs

    def upload_local_dir_to_cloud_storage(
        self,
        local_dir: str,
        *,
        cloud_id: str,
        excludes: Optional[List[str]] = None,  # noqa: ARG002
        overwrite_existing_file: bool = False,  # noqa: ARG002
        cloud_resource_name: Optional[str] = None,
    ) -> str:
        # Ensure that URIs are consistent for the same passed directory.
        bucket = self.CLOUD_BUCKET.format(cloud_id=cloud_id)
        if cloud_resource_name is not None:
            bucket += f"_{cloud_resource_name}"
        if local_dir not in self._upload_uri_mapping:
            self._upload_uri_mapping[
                local_dir
            ] = f"{bucket}/fake_pkg_{uuid.uuid4()!s}.zip"

        return self._upload_uri_mapping[local_dir]

    def upload_local_dir_to_cloud_storage_multi_cloud_resource(
        self,
        local_dir: str,
        *,
        cloud_id: str,
        cloud_resource_names: List[Optional[str]],
        excludes: Optional[List[str]] = None,  # noqa: ARG002
        overwrite_existing_file: bool = False,  # noqa: ARG002
    ) -> str:
        bucket = self.CLOUD_BUCKET.format(cloud_id=cloud_id)
        if local_dir not in self._upload_bucket_path_mapping:
            self._upload_bucket_path_mapping[local_dir] = (
                cloud_resource_names,
                f"{bucket}/fake_pkg_{uuid.uuid4()!s}.zip",
            )
        return self._upload_bucket_path_mapping[local_dir][1]

    def add_job_run_logs(self, job_run_id: str, logs: str):
        self._job_run_logs[job_run_id] = logs

    def logs_for_job_run(
        self,
        job_run_id: str,
        head: bool = False,  # noqa: ARG002
        tail: bool = True,  # noqa: ARG002
        max_lines: Optional[int] = None,  # noqa: ARG002
        parse_json: Optional[bool] = None,  # noqa: ARG002
    ) -> str:
        log_lines = self._job_run_logs.get(job_run_id, "").splitlines()
        if max_lines is None:
            max_lines = len(log_lines)
        if head:
            return "\n".join(log_lines[:max_lines])
        else:
            return "\n".join(log_lines[-1 * max_lines :])

    def add_controller_logs(self, service_version_id: str, logs: str):
        self._controller_logs[service_version_id] = logs

    def controller_logs_for_service_version(
        self,
        service_version: ProductionServiceV2VersionModel,
        head: bool = False,
        max_lines: Optional[int] = None,
        parse_json: Optional[bool] = None,  # noqa: ARG002
    ) -> str:
        log_lines = self._controller_logs.get(service_version.id, "").splitlines()
        if max_lines is None:
            max_lines = len(log_lines)
        if head:
            return "\n".join(log_lines[:max_lines])
        else:
            return "\n".join(log_lines[-1 * max_lines :])

    def get_schedule(
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str],  # noqa: A002
        cloud: Optional[str],
        project: Optional[str],
    ) -> Optional[DecoratedSchedule]:
        if id is not None:
            return self._schedules.get(id, None)
        else:
            cloud_id = self.get_cloud_id(cloud_name=cloud)
            cloud_project_dict = self._project_to_id.get(cloud_id, None)
            project_id = (
                cloud_project_dict.get(project, None) if cloud_project_dict else None
            )
            result: DecoratedSchedule = None
            for schedule in self._schedules.values():
                if (
                    schedule is not None
                    and schedule.name == name
                    and (project_id is None or schedule.project_id == project_id)
                ):
                    result = schedule
                    break

        return result

    def update_schedule(self, model: DecoratedSchedule):
        self._schedules[model.id] = model

    def apply_schedule(self, model: CreateSchedule) -> DecoratedSchedule:
        schedule = DecoratedSchedule(
            id=f"sched-{uuid.uuid4()!s}",
            name=model.name,
            project_id=model.project_id,
            config=model.config,
            schedule=model.schedule,
            # Fill in dummy time to represent schedule is enabled.
            next_trigger_at=self.SCHEDULE_NEXT_TRIGGER_AT_TIME,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

        self.update_schedule(schedule)
        return schedule

    def set_schedule_state(self, id: str, is_paused: bool):  # noqa: A002
        if is_paused:
            self._schedules[id].next_trigger_at = None
        else:
            self._schedules[id].next_trigger_at = self.SCHEDULE_NEXT_TRIGGER_AT_TIME

    def schedule_is_enabled(self, id: str) -> bool:  # noqa: A002
        return self._schedules[id].next_trigger_at is not None

    def trigger_schedule(self, id: str):  # noqa: A002
        self._schedule_trigger_counts[id] += 1

    def trigger_counts(self, id: str):  # noqa: A002
        return self._schedule_trigger_counts[id]

    def delete_schedule(self, schedule_id: str) -> None:
        """Delete a schedule from the fake client."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
        if schedule_id in self._schedule_trigger_counts:
            del self._schedule_trigger_counts[schedule_id]

    def list_schedules(
        self,
        *,
        name: Optional[str] = None,
        project_id: Optional[str] = None,  # noqa: ARG002
        cloud_id: Optional[str] = None,  # noqa: ARG002
        creator_id: Optional[str] = None,  # noqa: ARG002
        count: Optional[int] = None,
        paging_token: Optional[str] = None,  # noqa: ARG002
    ) -> DecoratedscheduleListResponse:
        """Mock implementation of list_schedules API for testing.

        Returns a DecoratedscheduleListResponse object to match the real API.
        """
        results = list(self._schedules.values())
        if name:
            results = [schedule for schedule in results if schedule.name == name]
        total = len(results)
        if count is not None:
            results = results[:count]
        return DecoratedscheduleListResponse(
            results=results,
            metadata=ListResponseMetadata(total=total, next_paging_token=None),
        )

    def create_workspace(self, model: CreateExperimentalWorkspace) -> str:
        workspace_id = uuid.uuid4()

        # this usually happens on the backend
        compute_config = self.get_compute_config(model.compute_config_id)
        assert compute_config is not None
        compute_config.idle_timeout_minutes = model.idle_timeout_minutes

        workspace = ExperimentalWorkspace(
            id=f"workspace-id-{workspace_id!s}",
            name=model.name,
            project_id=model.project_id or self.get_project_id(),
            compute_config_id=model.compute_config_id,
            environment_id=model.cluster_environment_build_id,
            cloud_id=model.cloud_id or self.get_cloud_id(),
            created_at=datetime.utcnow(),
            creator_id=self.DEFAULT_USER_ID,
            creator_email=self.DEFAULT_USER_EMAIL,
            organization_id=self.DEFAULT_ORGANIZATION_ID,
            cluster_id=self.DEFAULT_CLOUD_ID,
            state=SessionState.RUNNING
            if not model.skip_start
            else SessionState.TERMINATED,
        )
        self._workspaces[workspace.id] = workspace
        return workspace.id

    def get_workspace(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Optional[ExperimentalWorkspace]:
        if id is not None:
            return self._workspaces.get(id, None)
        else:
            cloud_id = self.get_cloud_id(cloud_name=cloud)
            cloud_project_dict = self._project_to_id.get(cloud_id, None)
            project_id = (
                cloud_project_dict.get(project, None) if cloud_project_dict else None
            )
            result: ExperimentalWorkspace = None
            for workspace in self._workspaces.values():
                if (
                    workspace is not None
                    and workspace.name == name
                    and (project_id is None or workspace.project_id == project_id)
                ):
                    result = workspace
                    break

        return result

    @property
    def workspaces(self) -> Dict[str, ExperimentalWorkspace]:
        return self._workspaces

    def start_workspace(self, workspace_id: str):
        workspace = self._workspaces.get(workspace_id, None)
        if workspace is None:
            raise ValueError(f"Workspace '{workspace_id}' not found.")
        workspace.state = SessionState.RUNNING

    def terminate_workspace(self, workspace_id: str):
        workspace = self._workspaces.get(workspace_id, None)
        if workspace is None:
            raise ValueError(f"Workspace '{workspace_id}' not found.")
        workspace.state = SessionState.TERMINATED

    def update_workspace_dependencies_offline_only(
        self, workspace_id: Optional[str], requirements: List[str]
    ):
        assert workspace_id is not None
        self._workspaces_dependencies[workspace_id] = requirements

    def update_workspace_env_vars_offline_only(
        self, workspace_id: Optional[str], env_vars: Dict[str, str]
    ):
        assert workspace_id is not None
        self._workspaces_env_vars[workspace_id] = env_vars

    def get_workspace_cluster(
        self, workspace_id: Optional[str]
    ) -> Optional[DecoratedSession]:
        workspace_model = (
            self._workspaces.get(workspace_id, None) if workspace_id else None
        )
        compute_config = self.get_compute_config(workspace_model.compute_config_id)
        assert compute_config is not None
        return Mock(
            name=f"workspace-cluster-{workspace_model.name}",
            build_id=workspace_model.environment_id,
            state=workspace_model.state,
            project_id=workspace_model.project_id,
            cloud_id=workspace_model.cloud_id,
            idle_timeout=compute_config.idle_timeout_minutes,
        )

    def get_workspace_proxied_dataplane_artifacts(
        self, workspace_id: str
    ) -> WorkspaceDataplaneProxiedArtifacts:
        env_vars_dict = self._workspaces_env_vars.get(workspace_id, None)
        return WorkspaceDataplaneProxiedArtifacts(
            requirements=self._workspaces_dependencies.get(workspace_id, None),
            environment_variables=[
                f"{key}={value}" for key, value in env_vars_dict.items()
            ]
            if env_vars_dict
            else None,
        )

    def get_cluster_head_node_ip(self, cluster_id: str) -> str:
        return self._clusters_headnode_ip.get(cluster_id, "")

    def get_cluster_ssh_key(self, cluster_id: str) -> SessionSshKey:
        return self._clusters_ssh_key.get(cluster_id, None)  # type: ignore

    def get_workspace_default_dir_name(self, workspace_id) -> str:
        workspace = self._workspaces.get(workspace_id, None)
        if workspace is None:
            raise ValueError(f"Workspace '{workspace_id}' not found.")
        return "default"

    def delete_finetuned_model(self, model_id: str) -> DeletedPlatformFineTunedModel:
        return DeletedPlatformFineTunedModel(id=model_id, deleted_at=datetime.utcnow())

    def list_finetuned_models(
        self,
        cloud_id: Optional[str],  # noqa: ARG002
        project_id: Optional[str],  # noqa: ARG002
        max_items: int,
    ) -> List[FineTunedModel]:
        return [
            FineTunedModel(
                id="test-model-id",
                model_id="test-model-id",
                base_model_id="my_base_model_id",
                ft_type=FineTuneType.LORA,
                creator_id="",
                creator_email="",
                created_at=datetime.utcnow(),
                storage_uri="s3://fake_bucket/fake_folder/",
            )
            for _ in range(max_items)
        ]

    def update_workspace(
        self,
        *,
        workspace_id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        compute_config_id: Optional[str] = None,
        cluster_environment_build_id: Optional[str] = None,
        idle_timeout_minutes: Optional[int] = None,
    ):
        if workspace_id not in self._workspaces:
            raise ValueError(f"Workspace '{workspace_id}' not found.")

        workspace = self._workspaces[workspace_id]

        if name:
            workspace.name = name

        if compute_config_id:
            workspace.compute_config_id = compute_config_id

        if cluster_environment_build_id:
            workspace.environment_id = cluster_environment_build_id

        if idle_timeout_minutes:
            workspace.idle_timeout_minutes = idle_timeout_minutes

            compute_config = self.get_compute_config(workspace.compute_config_id)
            assert compute_config is not None
            compute_config.idle_timeout_minutes = idle_timeout_minutes

    def download_aggregated_instance_usage_csv(
        self,
        start_date: date,
        end_date: date,
        cloud_id: Optional[str] = None,  # noqa: ARG002
        project_id: Optional[str] = None,  # noqa: ARG002
        directory: Optional[str] = None,
        hide_progress_bar: bool = False,  # noqa: ARG002
    ) -> str:
        filename = f"aggregated_instance_usage_{start_date}_{end_date}.zip"
        filepath = os.path.join(directory, filename) if directory else filename

        return filepath

    def create_api_key(
        self, duration: float, user_id: Optional[str]  # noqa: ARG002
    ) -> ServerSessionToken:
        api_key = f"{uuid.uuid4()!s}"
        api_keys = self._api_keys.get(user_id or "usr_1", [])
        self._api_keys[user_id or "usr_1"] = api_keys + [api_key]

        return ServerSessionToken(server_session_id=api_key)

    def rotate_api_key(self, user_id: str) -> None:
        if user_id not in self._api_keys:
            raise ValueError(f"User '{user_id}' not found.")
        self._api_keys[user_id] = [f"{uuid.uuid4()!s}"]

    def admin_batch_create_users(
        self, admin_create_users: List[AdminCreateUser]
    ) -> List[AdminCreatedUser]:
        created_users = []
        for admin_create_user in admin_create_users:
            if admin_create_user.email in self._users:
                raise ValueError(
                    f"User with email '{admin_create_user.email}' already exists."
                )

            user_id = f"user-id-{uuid.uuid4()!s}"
            created_user = AdminCreatedUser(
                user_id=user_id,
                email=admin_create_user.email,
                created_at=datetime.utcnow(),
                name=admin_create_user.name,
                lastname=admin_create_user.lastname,
                title=admin_create_user.title,
                is_sso_user=admin_create_user.is_sso_user,
            )
            created_users.append(created_user)
            self._users[admin_create_user.email] = created_user

        return created_users

    def create_organization_invitations(
        self, emails: List[str]
    ) -> Tuple[List[str], List[str]]:
        success_emails, error_messages = [], []
        for email in emails:
            if email in self._users:
                error_messages.append(
                    f"User with email {email} is already a member of your organization."
                )
            else:
                orginv_id = f"orginv_{uuid.uuid4()!s}"
                self._organization_invitations[email] = OrganizationInvitation(
                    email=email,
                    id=orginv_id,
                    organization_id="org_1",
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow(),
                )
                success_emails.append(email)

        return success_emails, error_messages

    def list_organization_invitations(self) -> List[OrganizationInvitation]:
        return list(self._organization_invitations.values())

    def delete_organization_invitation(self, email: str) -> OrganizationInvitation:
        if email not in self._organization_invitations:
            raise ValueError(f"Invitation for email '{email}' not found.")

        return self._organization_invitations.pop(email)

    def list_organization_collaborators(
        self,
        *,
        email: Optional[str] = None,
        name: Optional[str] = None,
        collaborator_type: Optional[CollaboratorType] = None,  # noqa: ARG002
        is_service_account: Optional[bool] = None,  # noqa: ARG002
        count: Optional[int] = None,
        paging_token: Optional[str] = None,  # noqa: ARG002
    ) -> OrganizationcollaboratorListResponse:
        filtered_results: List[OrganizationCollaborator] = []
        for organization_collaborator in self._organization_collaborators:
            if email and organization_collaborator.email != email:
                continue
            if name and organization_collaborator.name != name:
                continue

            filtered_results.append(organization_collaborator)

        effective_count = count if count is not None else len(filtered_results)
        results = filtered_results[:effective_count]

        return OrganizationcollaboratorListResponse(
            results=results,
            metadata=ListResponseMetadata(
                total=len(filtered_results),
                next_paging_token=None,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

    def get_organization_collaborators(
        self,
        email: Optional[str] = None,
        name: Optional[str] = None,
        collaborator_type: Optional[CollaboratorType] = None,  # noqa: ARG002
        is_service_account: Optional[bool] = None,  # noqa: ARG002
    ) -> List[OrganizationCollaborator]:
        response = self.list_organization_collaborators(
            email=email,
            name=name,
            collaborator_type=collaborator_type,
            is_service_account=is_service_account,
        )

        return list(response.results)

    def delete_organization_collaborator(self, identity_id: str) -> None:
        for organization_collaborator in self._organization_collaborators:
            if organization_collaborator.id == identity_id:
                self._organization_collaborators.remove(organization_collaborator)
                return

        raise ValueError(
            f"Organization collaborator with id '{identity_id}' not found."
        )

    def create_service_account(self, name) -> AnyscaleServiceAccount:
        for organization_collaborator in self._organization_collaborators:
            if organization_collaborator.name == name:
                raise ValueError(f"Service account with name '{name}' already exists.")

        identity_id = f"id_{uuid.uuid4()!s}"
        user_id = f"usr_{uuid.uuid4()!s}"
        organization_collaborator = OrganizationCollaborator(
            permission_level=OrganizationPermissionLevel.COLLABORATOR,
            base_role=OrganizationBaseRole.COLLABORATOR,
            additional_roles=[],
            id=identity_id,
            name=name,
            email=f"{name}@service-account.com",
            user_id=user_id,
            created_at=datetime.utcnow(),
        )

        self._organization_collaborators.append((organization_collaborator))

        return AnyscaleServiceAccount(
            user_id=organization_collaborator.user_id,
            name=organization_collaborator.name,
            organization_id="org_1",
            email=organization_collaborator.email,
            permission_level=organization_collaborator.permission_level,
            created_at=organization_collaborator.created_at,
        )

    def create_resource_quota(
        self, create_resource_quota: CreateResourceQuota
    ) -> ResourceQuota:
        resource_quota_id = f"rq_{uuid.uuid4()!s}"
        resource_quota = ResourceQuota(
            id=resource_quota_id,
            name=create_resource_quota.name,
            cloud_id=create_resource_quota.cloud_id,
            project_id=create_resource_quota.project_id,
            quota=create_resource_quota.quota,
            is_enabled=True,
            created_at=datetime.utcnow(),
            creator=MiniUser(
                id="user_1",
                email="test@anyscale.com",
                name="Test User",
                username="testuser",
            ),
            cloud=MiniCloud(
                id=create_resource_quota.cloud_id,
                name="Test Cloud",
                provider=CloudProviders.AWS,
            ),
        )

        self._resource_quotas[resource_quota_id] = resource_quota

        return self._resource_quotas[resource_quota_id]

    def list_resource_quotas(
        self,
        name: Optional[str] = None,
        cloud_id: Optional[str] = None,
        creator_id: Optional[str] = None,
        is_enabled: Optional[bool] = None,
        max_items: int = 20,
    ) -> List[ResourceQuota]:
        results = []
        for resource_quota in self._resource_quotas.values():
            if name and resource_quota.name != name:
                continue
            if cloud_id and resource_quota.cloud_id != cloud_id:
                continue
            if creator_id and resource_quota.creator.id != creator_id:
                continue
            if is_enabled is not None and resource_quota.is_enabled != is_enabled:
                continue
            results.append(resource_quota)

        return results[:max_items]

    def delete_resource_quota(self, resource_quota_id: str) -> None:
        if resource_quota_id not in self._resource_quotas:
            raise ValueError(f"Resource Quota with id '{resource_quota_id}' not found.")

        self._resource_quotas.pop(resource_quota_id)

    def set_resource_quota_status(
        self, resource_quota_id: str, is_enabled: bool
    ) -> None:
        resource_quota = self._resource_quotas.get(resource_quota_id)
        if resource_quota is None:
            raise ValueError(f"Resource Quota with id '{resource_quota_id}' not found.")

        resource_quota.is_enabled = is_enabled

    def get_service_by_id(
        self, service_id: str
    ) -> Optional[DecoratedProductionServiceV2APIModel]:
        if service_id in self._services:
            return self._services[service_id]
        if service_id in self._archived_services:
            return self._archived_services[service_id]
        if service_id in self._deleted_services:
            return self._deleted_services[service_id]
        return None

    def list_services(  # noqa: PLR0913
        self,
        *,
        name: Optional[str] = None,
        state_filter: Optional[List[str]] = None,
        tag_filter: Optional[List[str]] = None,  # noqa: ARG002
        creator_id: Optional[str] = None,  # noqa: ARG002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
        count: Optional[int] = None,
        paging_token: Optional[str] = None,  # noqa: ARG002
        sort_field: Optional[str] = None,  # noqa: ARG002
        sort_order: Optional[str] = None,  # noqa: ARG002
    ) -> DecoratedlistserviceapimodelListResponse:
        target_services = list(self._services.values())
        if include_archived:
            target_services.extend(list(self._archived_services.values()))

        target_cloud_id = self.get_cloud_id(cloud_name=cloud) if cloud else None
        target_project_id = (
            self.get_project_id(parent_cloud_id=target_cloud_id, name=project)
            if project
            else None
        )

        filtered_results = []
        for svc in target_services:
            if name is not None and svc.name != name:
                continue
            # Ensure state comparison works whether svc.current_state is Enum or str
            current_state_str = (
                svc.current_state.value
                if hasattr(svc.current_state, "value")
                else svc.current_state
            )
            if state_filter is not None and current_state_str not in state_filter:
                continue
            if target_project_id is not None and svc.project_id != target_project_id:
                continue
            if target_cloud_id is not None and svc.cloud_id != target_cloud_id:
                continue

            filtered_results.append(svc)

        if count is None:
            count = len(filtered_results)
        final_results = filtered_results[:count]

        response = DecoratedlistserviceapimodelListResponse(
            results=final_results,
            metadata=ListResponseMetadata(
                total=len(final_results),
                next_paging_token=None,  # Fake doesn't support paging
            ),
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )
        return response

    def get_service_versions(
        self, service_id: str, read_all_versions: bool = False  # noqa: ARG002
    ) -> List[DecoratedProductionServiceV2VersionAPIModel]:
        return list(self._versions[service_id].values())

    def upsert_resource_tags(
        self,
        resource_type: ResourceTagResourceType,
        resource_id: str,
        tags: Dict[str, str],
    ) -> None:
        self.upsert_resource_tags_calls.append((resource_type, resource_id, tags))
        key = (str(resource_type), resource_id)
        current = self._resource_tags_store.get(key, {})
        current.update(tags)
        self._resource_tags_store[key] = current

    def delete_resource_tags(
        self, resource_type: ResourceTagResourceType, resource_id: str, keys: List[str],
    ) -> None:
        self.delete_resource_tags_calls.append((resource_type, resource_id, keys))
        key = (str(resource_type), resource_id)
        current = self._resource_tags_store.get(key, {})
        for k in keys:
            current.pop(k, None)
        self._resource_tags_store[key] = current

    def list_resource_tags(
        self, resource_type: ResourceTagResourceType, resource_id: str
    ) -> List[ResourceTagRecord]:
        key = (str(resource_type), resource_id)
        tags = self._resource_tags_store.get(key, {})
        now = datetime.utcnow()
        records: List[ResourceTagRecord] = []
        for k, v in tags.items():
            records.append(
                ResourceTagRecord(
                    id=f"tag_{uuid.uuid4()}",
                    organization_id=self.DEFAULT_ORGANIZATION_ID,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    key=k,
                    value=v,
                    created_at=now,
                    updated_at=now,
                    local_vars_configuration=OPENAPI_NO_VALIDATION,
                )
            )
        return records

    def list_user_groups(
        self, *, count: int = 50, paging_token: Optional[str] = None,  # noqa: ARG002
    ) -> UsergroupListResponse:
        now = datetime.utcnow()
        fake_groups = [
            UserGroup(
                id="ug_fake_001",
                name="Engineering",
                org_id=self.DEFAULT_ORGANIZATION_ID,
                created_at=now,
                updated_at=now,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            UserGroup(
                id="ug_fake_002",
                name="Data Science",
                org_id=self.DEFAULT_ORGANIZATION_ID,
                created_at=now,
                updated_at=now,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
        ]
        return UsergroupListResponse(
            results=fake_groups[:count],
            metadata=ListResponseMetadata(
                total=len(fake_groups),
                next_paging_token=None,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

    def get_user_group(self, group_id: str) -> UserGroup:
        now = datetime.utcnow()
        # Return consistent data with list_user_groups
        fake_groups = {
            "ug_fake_001": "Engineering",
            "ug_fake_002": "Data Science",
        }
        group_name = fake_groups.get(group_id, "Fake User Group")

        return UserGroup(
            id=group_id,
            name=group_name,
            org_id=self.DEFAULT_ORGANIZATION_ID,
            created_at=now,
            updated_at=now,
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

    def list_user_group_memberships(self) -> Dict:
        """List all user groups with their members."""
        return {
            "result": {
                "groups": [
                    {
                        "group_id": "ug_fake_001",
                        "group_name": "Engineering",
                        "members": [
                            {"user_id": "usr_001", "user_email": "alice@example.com",},
                            {
                                "user_id": "usr_002",
                                "user_email": "charlie@example.com",
                            },
                        ],
                    },
                    {
                        "group_id": "ug_fake_002",
                        "group_name": "Data Science",
                        "members": [
                            {"user_id": "usr_003", "user_email": "bob@example.com",},
                        ],
                    },
                ]
            }
        }

    def update_resource_policy(
        self,
        resource_type: str,  # noqa: ARG002
        resource_id: str,  # noqa: ARG002
        policy: UpdatePolicyRequest,  # noqa: ARG002
    ) -> None:
        pass

    def get_resource_policy(
        self, resource_type: str, resource_id: str,  # noqa: ARG002
    ) -> PolicyResponse:
        return PolicyResponse(
            bindings=[
                Binding(
                    role_name="write",
                    principals=["ug_fake_001"],
                    local_vars_configuration=OPENAPI_NO_VALIDATION,
                ),
                Binding(
                    role_name="readonly",
                    principals=["ug_fake_002"],
                    local_vars_configuration=OPENAPI_NO_VALIDATION,
                ),
            ],
            sync_status="success",
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

    def list_resource_policies(
        self, resource_type: str,
    ) -> ResourcepolicyitemListResponse:
        return ResourcepolicyitemListResponse(
            results=[
                ResourcePolicyItem(
                    resource_id="cld_fake_001",
                    resource_type=resource_type,
                    bindings=[
                        Binding(
                            role_name="write",
                            principals=["ug_fake_001"],
                            local_vars_configuration=OPENAPI_NO_VALIDATION,
                        ),
                    ],
                    sync_status="success",
                    local_vars_configuration=OPENAPI_NO_VALIDATION,
                ),
            ],
            metadata=ListResponseMetadata(
                total=1,
                next_paging_token=None,
                local_vars_configuration=OPENAPI_NO_VALIDATION,
            ),
            local_vars_configuration=OPENAPI_NO_VALIDATION,
        )

    def migrate_scim_permissions(self, *, dry_run: bool = True) -> Dict:
        """Fake implementation of SCIM permission migration."""
        return {
            "result": {
                "organization_id": "org_fake_001",
                "total_users_processed": 2,
                "users_migrated": [
                    {
                        "user_id": "usr_fake_001",
                        "user_email": "user1@example.com",
                        "identity_id": "id_fake_001",
                        "org_owner_demoted": True,
                        "cloud_permissions_to_remove": [
                            {
                                "cloud_id": "cld_001",
                                "action": "read_group",
                                "is_admin": False,
                            },
                        ],
                        "project_permissions_to_remove": [],
                        "readonly_overrides_to_remove": [],
                        "org_permission_changes": [],
                    },
                    {
                        "user_id": "usr_fake_002",
                        "user_email": "user2@example.com",
                        "identity_id": "id_fake_002",
                        "org_owner_demoted": False,
                        "cloud_permissions_to_remove": [],
                        "project_permissions_to_remove": [],
                        "readonly_overrides_to_remove": [],
                        "org_permission_changes": [],
                    },
                ],
                "errors": [],
                "dry_run": dry_run,
            }
        }

    def list_scim_user_permissions(self, user_id: Optional[str] = None) -> Dict:
        """Fake implementation of listing effective user permissions."""
        all_users = [
            {
                "user_id": "usr_fake_001",
                "user_email": "user1@example.com",
                "clouds": [
                    {
                        "cloud_id": "cld_fake_001",
                        "cloud_name": "prod-cloud",
                        "role": "collaborator",
                        "projects": [
                            {
                                "project_id": "prj_fake_001",
                                "project_name": "prod-project",
                                "role": "readonly",
                            }
                        ],
                    }
                ],
            }
        ]
        users = all_users
        if user_id:
            users = [u for u in all_users if u["user_id"] == user_id]
        return {
            "result": {
                "organization_id": "org_fake_001",
                "org_owners": [
                    {"user_id": "usr_owner_001", "user_email": "owner1@example.com"},
                ],
                "users": users,
            }
        }

    def scim_migration_preview(self) -> Dict:
        """Fake implementation of SCIM migration preview."""
        return {
            "result": {
                "organization_id": "org_fake_001",
                "users_with_changes": [
                    {
                        "user_id": "usr_fake_001",
                        "user_email": "user1@example.com",
                        "cloud_changes": [
                            {
                                "resource_id": "cld_fake_001",
                                "resource_name": "prod-cloud",
                                "old_role": "collaborator",
                                "new_role": "readonly",
                            }
                        ],
                        "project_changes": [],
                        "org_role_change": {
                            "resource_id": "org_fake_001",
                            "resource_name": "organization",
                            "old_role": "owner",
                            "new_role": "collaborator",
                        },
                    }
                ],
                "users_to_remove": [
                    {"user_id": "usr_fake_002", "user_email": "user2@example.com"}
                ],
            }
        }
