import asyncio
import contextlib
import os
import logging
import copy
from time import sleep, time
from typing import Any, Callable, Dict, List, Optional, Union

from anyscale.api import configure_tcp_keepalive
from anyscale.sdk.anyscale_client.models.cluster_compute_config import (
    ClusterComputeConfig,
)
from anyscale.sdk.anyscale_client.models.create_cluster_compute import (
    CreateClusterCompute,
)
from anyscale.sdk.anyscale_client.models.production_job import ProductionJob
from anyscale.sdk.anyscale_client.models.ha_job_states import HaJobStates
from anyscale.utils.runtime_env import upload_and_rewrite_working_dir
from anyscale.sdk import anyscale_client
from anyscale.sdk.anyscale_client.api.default_api import DefaultApi
from anyscale.version import __version__ as version
from anyscale.sdk.anyscale_client.models.create_cluster_environment import (
    CreateClusterEnvironment,
)
from anyscale.sdk.anyscale_client.models.create_byod_cluster_environment import (
    CreateBYODClusterEnvironment,
)
from anyscale.sdk.anyscale_client.models.cluster import Cluster
from anyscale.sdk.anyscale_client.models.cluster_state import ClusterState
from anyscale.sdk.anyscale_client.models.update_cluster import UpdateCluster

from anyscale.sdk.anyscale_client.models.cluster_environment_build import (
    ClusterEnvironmentBuild,
)
from anyscale.shared_anyscale_utils.headers import RequestHeaders
from anyscale.cli_logger import BlockLogger, StringLogger, pad_string
from anyscale.cluster_compute import get_selected_cloud_id_or_default
from anyscale.util import get_endpoint
from anyscale.authenticate import AuthenticationBlock
from anyscale_client.models import ClusterEnvironmentBuildStatus
from anyscale.util import get_ray_and_py_version_for_default_cluster_env
from anyscale.project_utils import get_default_project
from anyscale.utils.ray_version_utils import get_correct_name_for_base_image_name

logger = logging.getLogger(__file__)


def _upload_and_rewrite_working_dir_in_create_production_job(create_production_job):
    """
    If a local working_dir and upload_path are specified, this method
    will upload the working_dir and rewrites the working_dir to the remote uri.

    Please note that this method mutates the model in-place.
    """
    if create_production_job.config is not None:
        runtime_env = None
        if isinstance(create_production_job.config, dict):
            runtime_env = create_production_job.config.get("runtime_env")
        elif hasattr(create_production_job.config, "runtime_env"):
            runtime_env = create_production_job.config.runtime_env
        if runtime_env is not None:
            new_runtime_env = upload_and_rewrite_working_dir(runtime_env)
            if isinstance(create_production_job.config, dict):
                create_production_job.config["runtime_env"] = new_runtime_env
            elif hasattr(create_production_job.config, "runtime_env"):
                create_production_job.config.runtime_env = new_runtime_env


def _upload_and_rewrite_working_dir_ray_serve_config(
    ray_serve_config: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    If a local working_dir and upload_path are specified, this method
    will upload the working_dir and rewrites the working_dir to the remote uri.

    Please note that this method returns the new ray_serve_config.
    """
    if not ray_serve_config:
        return ray_serve_config
    res = copy.deepcopy(ray_serve_config)
    if isinstance(res, dict):
        if "applications" in res:
            for ray_serve_app_config in res["applications"]:
                runtime_env = ray_serve_app_config.get("runtime_env")
                if runtime_env is not None:
                    new_runtime_env = upload_and_rewrite_working_dir(runtime_env)
                    ray_serve_app_config["runtime_env"] = new_runtime_env
        else:
            runtime_env = res.get("runtime_env")
            if runtime_env is not None:
                new_runtime_env = upload_and_rewrite_working_dir(runtime_env)
                res["runtime_env"] = new_runtime_env
    return res


def _is_create_byod_cluster_environment(obj: Any) -> bool:
    """
    Returns True if the object's config_json has "docker_image" set.
    The object can be either a dictionary or an object with "config_json"
    as an attribute.
    """
    config_json = None
    if isinstance(obj, dict):
        config_json = obj.get("config_json")
    else:
        config_json = getattr(obj, "config_json", None)
    return isinstance(config_json, dict) and "docker_image" in config_json


# Build states that are in progress
IN_PROGRESS_BUILD_STATES = [
    ClusterEnvironmentBuildStatus.IN_PROGRESS,
    ClusterEnvironmentBuildStatus.PENDING,
]


class AnyscaleSDKDeprecationError(Exception):
    """Raised when a deprecated AnyscaleSDK method is called."""

    pass


class AnyscaleSDK(DefaultApi):  # type: ignore
    def __init__(
        self,
        auth_token: Optional[str] = None,
        host: Optional[str] = None,
    ):
        # Default host to production but respect the ANYSCALE_HOST environment variable.
        if host is None:
            host = os.environ.get("ANYSCALE_HOST", "https://api.anyscale.com")

        # Automatically convert anyscale host to host for anyscale external api
        if host.startswith("https://console."):
            host = host.replace("console.", "api.", 1)

        # Automatically convert anyscale host to host for anyscale external api (premerge)
        if host.startswith("https://ci-"):
            host = host.rstrip("/") + "/ext"

        # Adds base path "v0" for API endpoints
        endpoint_url = host.rstrip("/") + "/v0"
        configuration = anyscale_client.Configuration(host=endpoint_url)
        configuration.proxy = os.environ.get("https_proxy")
        configuration.connection_pool_maxsize = 100
        if auth_token is None:
            auth_token, _ = AuthenticationBlock._load_credentials()
        api_client = anyscale_client.ApiClient(
            configuration, cookie=f"cli_token={auth_token}"
        )
        configure_tcp_keepalive(api_client)
        api_client.set_default_header(RequestHeaders.CLIENT, "SDK")
        api_client.set_default_header(RequestHeaders.CLIENT_VERSION, version)

        self.log: BlockLogger = BlockLogger()
        self._auth_token = auth_token

        super(AnyscaleSDK, self).__init__(api_client)

    def build_cluster_environment(
        self,
        create_cluster_environment: Union[
            CreateClusterEnvironment, CreateBYODClusterEnvironment
        ],
        poll_rate_seconds: int = 15,
        timeout_seconds: Optional[int] = None,
        log_output: bool = False,
    ) -> Optional[ClusterEnvironmentBuild]:
        """
        Creates a new Cluster Environment and waits for build to complete.

        If a Cluster Environment with the same name already exists, this will
        create an updated build of that environment.

        Args:
            create_cluster_environment - CreateClusterEnvironment object
            poll_rate_seconds - seconds to wait when polling build operation status; defaults to 15
            timeout_seconds - maximum number of seconds to wait for build operation to complete before timing out; defaults to no timeout

        Returns:
            Newly created ClusterEnvironmentBuild object

        Raises:
            Exception if building Cluster Environment failed or timed out
        """

        self.log.log_output = log_output
        cluster_environments = self.search_cluster_environments(
            {
                "name": {"equals": create_cluster_environment.name},
                "paging": {"count": 1},
            }
        ).results

        if not cluster_environments:
            self.log.info(
                f"Creating new cluster environment {create_cluster_environment.name}"
            )
            if _is_create_byod_cluster_environment(create_cluster_environment):
                cluster_environment = self.create_byod_cluster_environment(
                    create_cluster_environment
                ).result
            else:
                base_image_name = create_cluster_environment.config_json['base_image']
                correct_base_image_name = get_correct_name_for_base_image_name(base_image_name)
                if base_image_name != correct_base_image_name: # We need to make sure base_image_name after ray 2.7.0, before 2.8.0 have a optimized suffix
                    self.log.error(f"Starting from Ray version 2.7.0 and before version 2.8.0, you might be required to include the suffix 'optimized' in the base_image name for that specific Ray version. Try using {self.log.highlight(correct_base_image_name)}")
                    return  # type: ignore
                cluster_environment = self.create_cluster_environment(
                    create_cluster_environment
                ).result
            build = self.list_cluster_environment_builds(
                cluster_environment.id
            ).results[0]
            build_operation_id = build.id
        else:
            self.log.info(
                f"Building new revision of cluster environment {create_cluster_environment.name}"
            )
            cluster_environment = cluster_environments[0]
            if _is_create_byod_cluster_environment(create_cluster_environment):
                build = self.create_byod_cluster_environment_build(
                    {
                        "cluster_environment_id": cluster_environment.id,
                        "config_json": create_cluster_environment.config_json,
                    }
                ).result
            else:
                base_image_name = create_cluster_environment.config_json['base_image']
                correct_base_image_name = get_correct_name_for_base_image_name(base_image_name)
                if base_image_name != correct_base_image_name: # We need to make sure base_image_name after ray 2.7.0, before 2.8.0 have a optimized suffix
                    self.log.error(f"Starting from Ray version 2.7.0 and before version 2.8.0, you might be required to include the suffix 'optimized' in the base_image name for that specific Ray version. Try using {self.log.highlight(correct_base_image_name)}")
                    return  # type: ignore
                build = self.create_cluster_environment_build(
                    {
                        "cluster_environment_id": cluster_environment.id,
                        "config_json": create_cluster_environment.config_json,
                    }
                ).result
            build_operation_id = build.id

        build = self._wait_for_cluster_environment_build_operation(
            build_operation_id, poll_rate_seconds, timeout_seconds, log_output
        )
        self.log.close_block()

        block_label = "BuildDetails"
        self.log.open_block(block_label=block_label, block_title="Build Details")
        with self.log.indent():
            pad_len = len("Name:")
            self.log.info(
                pad_string("Name:", pad_len),
                BlockLogger.highlight(f"{create_cluster_environment.name}:{build.revision}"),
                block_label=block_label
            )
            self.log.info(
                pad_string("ID:", pad_len),
                BlockLogger.highlight(build.id),
                block_label=block_label
            )
        return build

    def _wait_for_cluster_environment_build_operation(
        self,
        operation_id: str,
        poll_rate_seconds: int = 15,
        timeout_seconds: Optional[int] = None,
        log_output: bool = False,
    ) -> ClusterEnvironmentBuild:
        """
        Waits for a Cluster Environment Build operation to complete.

        Args:
            operation_id - ID of the Cluster Environment Build operation
            poll_rate_seconds - seconds to wait when polling build operation status; defaults to 15
            timeout_seconds - maximum number of seconds to wait for build operation to complete before timing out; defaults to no timeout

        Returns:
            ClusterEnvironmentBuild object

        Raises:
            Exception if building Cluster Environment fails or times out.
        """

        self.log.log_output = log_output
        timeout = time() + timeout_seconds if timeout_seconds else None

        operation = self.get_build(operation_id)
        url = get_endpoint(f"configurations/app-config-details/{operation_id}")
        self.log.info(
            f"Waiting for cluster environment to build. View progress at {url}"
        )
        self.log.info(f"Status: {operation.result.status}")
        while operation.result.status in IN_PROGRESS_BUILD_STATES:
            if timeout and time() > timeout:
                raise Exception(
                    f"Building Cluster Environment timed out after {timeout_seconds} seconds."
                )

            sleep(poll_rate_seconds)
            self.log.info(f"Status: {operation.result.status}")
            operation = self.get_build(operation_id)
        if operation.result.status != ClusterEnvironmentBuildStatus.SUCCEEDED:
            raise Exception(
                f"Failed to build Cluster Environment, you can check the full logs at: {url}."
            )
        else:
            self.log.info("Cluster environment successfully finished building.")
            return self.get_build(operation_id).result

    def launch_cluster(
        self,
        project_id: Optional[str],
        cluster_name: str,
        cluster_environment_build_id: Optional[str] = None,
        cluster_compute_id: Optional[str] = None,
        poll_rate_seconds: int = 15,
        timeout_seconds: Optional[int] = None,
        idle_timeout_minutes: Optional[int] = None,
        cluster_compute_config: Optional[ClusterComputeConfig] = None,
        log_output: bool = False,
    ) -> Cluster:
        """
        Starts a Cluster in the specified Project.
        If a Cluster with the specified name already exists, we will update that Cluster.
        Otherwise, a new Cluster will be created.

        Args:
            project_id - ID of the Project the Cluster belongs to or None to launch cluster without a project.
            cluster_name - Name of the Cluster
            cluster_environment_build_id - Cluster Environment Build to start this Cluster with
                                           If none, uses a default cluster_environment_build
            cluster_compute_id - Cluster Compute to start this Cluster with
                                 If none, it checks if `cluster_compute_config` is specified.
                                 If `cluster_compute_config` is none, it uses a default cluster_compute.
            poll_rate_seconds - seconds to wait when polling Cluster operation status; defaults to 15
            timeout_seconds - maximum number of seconds to wait for Cluster operation to complete before timing out; defaults to no timeout
            idle_timeout_minutes - Idle timeout (in minutes), after which the Cluster is terminated
            cluster_compute_config - One-off Cluster Compute that this Cluster will use.
            log_output - Whether to show verbose logs during execution

        Returns:
            Cluster object

        Raises:
            Exception if starting Cluster fails or times out
        """

        self.log.log_output = log_output
        if project_id:
            search_project_id = project_id
        else:
            parent_cloud_id = get_selected_cloud_id_or_default(
                cluster_compute_id=cluster_compute_id,
                cluster_compute_config=cluster_compute_config,
                cloud_id=None,
                cloud_name=None,
            )
            default_project = get_default_project(parent_cloud_id=parent_cloud_id)
            project_id = default_project.id

        clusters: Optional[List[Cluster]] = self.search_clusters(
            {"project_id": project_id, "name": {"equals": cluster_name}}
        ).results

        if clusters:

            sorted_clusters_list = sorted(clusters, key=lambda x: x.created_at, reverse=True)
            cluster: Cluster = sorted_clusters_list[0]

            if cluster.state in [ClusterState.STARTINGUP, ClusterState.RUNNING, ClusterState.UPDATING, ClusterState.TERMINATING, ClusterState.AWAITINGSTARTUP]:
                raise Exception(
                    f"Cluster {cluster_name} in project {project_id} is in state {cluster.state}. "
                    "Please terminate the cluster or use a new cluster name, then rerun launch_cluster."
                )

            if not cluster_environment_build_id:
                cluster_environment_build_id = cluster.cluster_environment_build_id

            if not cluster_compute_id:
                cluster_compute_id = cluster.cluster_compute_id

            if not idle_timeout_minutes:
                idle_timeout_minutes = cluster.idle_timeout_minutes

            # Update cluster with new configurations
            # Or use configurations from terminated cluster
            update_cluster_config = UpdateCluster(
                cluster_environment_build_id=cluster_environment_build_id,
                cluster_compute_id=cluster_compute_id,
                idle_timeout_minutes=idle_timeout_minutes,
            )

            self.update_cluster(cluster.id, update_cluster_config)

        else:
            if not cluster_environment_build_id:
                # Use default cluster environment build when starting cluster without specifying an id
                (
                    ray_version,
                    py_version,
                ) = get_ray_and_py_version_for_default_cluster_env()
                cluster_environment_build_id = (
                    self.get_default_cluster_environment_build(
                        f"py{py_version}", ray_version
                    ).result.id
                )

            if not cluster_compute_id:
                if cluster_compute_config:
                    cluster_compute_id = self.create_cluster_compute(
                        CreateClusterCompute(
                            config=cluster_compute_config, anonymous=True
                        )
                    ).result.id
                else:
                    # Use default cluster compute when starting cluster without specifying an id
                    cluster_compute_id = self.get_default_cluster_compute().result.id


            create_cluster_payload: Dict[str, Any] = {
                "name": cluster_name,
                "project_id": project_id,
                "cluster_environment_build_id": cluster_environment_build_id,
                "cluster_compute_id": cluster_compute_id,
            }
            if idle_timeout_minutes:
                create_cluster_payload["idle_timeout_minutes"] = idle_timeout_minutes

            # Create cluster DB reference
            cluster = self.create_cluster(create_cluster_payload).result

        # Start cluster
        start_operation = self.start_cluster(
            cluster.id,
            {
                "cluster_environment_build_id": cluster_environment_build_id,
                "cluster_compute_id": cluster_compute_id,
                "idle_timeout_minutes": idle_timeout_minutes,
            },
        ).result

        return self.wait_for_cluster_operation(
            start_operation.id, poll_rate_seconds, timeout_seconds
        )

    def launch_cluster_with_new_cluster_environment(
        self,
        project_id: Optional[str],
        cluster_name: str,
        create_cluster_environment: CreateClusterEnvironment,
        cluster_compute_id: Optional[str] = None,
        poll_rate_seconds: int = 15,
        timeout_seconds: Optional[int] = None,
        idle_timeout_minutes: Optional[int] = None,
        cluster_compute_config: Optional[ClusterComputeConfig] = None,
    ) -> Cluster:
        """
        Builds a new Cluster Environment, then starts a Cluster in the specified Project with the new build.
        If a Cluster with the specified name already exists, we will update that Cluster.
        Otherwise, a new Cluster will be created.

        Args:
            project_id - ID of the Project the Cluster belongs to or None to launch a cluster without a project.
            cluster_name - Name of the Cluster
            create_cluster_environment - CreateClusterEnvironment object
            cluster_compute_id - Cluster Compute to start this Cluster with
                                 If none, it checks if `cluster_compute_config` is specified.
                                 If `cluster_compute_config` is  none, it uses a default cluster_compute.
            poll_rate_seconds - seconds to wait when polling for operations; defaults to 15
            timeout_seconds - maximum number of seconds to wait for each operation to complete before timing out; defaults to no timeout
            idle_timeout_minutes - Idle timeout (in minutes), after which the Cluster is terminated
            cluster_compute_config - One-off Cluster Compute that this Cluster will use.

        Returns:
            Cluster object

        Raises:
            Exception if building the new Cluster Environment fails or starting the Cluster fails.
        """

        cluster_environment_build = self.build_cluster_environment(
            create_cluster_environment, poll_rate_seconds, timeout_seconds
        )

        assert cluster_environment_build is not None
        return self.launch_cluster(
            project_id,
            cluster_name,
            cluster_environment_build.id,
            cluster_compute_id,
            poll_rate_seconds,
            timeout_seconds,
            idle_timeout_minutes,
            cluster_compute_config,
        )

    def wait_for_cluster_operation(
        self,
        operation_id: str,
        poll_rate_seconds: int = 15,
        timeout_seconds: Optional[int] = None,
    ) -> Cluster:
        """
        Waits for a Cluster operation to complete, most commonly used when starting, terminating, or updating a Cluster.

        Args:
            operation_id - ID of the Cluster Operation
            poll_rate_seconds - seconds to wait when polling build operation status; defaults to 15
            timeout_seconds - maximum number of seconds to wait for build operation to complete before timing out; defaults to no timeout

        Returns:
            Cluster object when the operation completes successfully

        Raises:
            Exception if building Cluster operation fails or times out
        """

        timeout = time() + timeout_seconds if timeout_seconds else None

        operation = self.get_cluster_operation(operation_id).result
        while not operation.completed:
            if timeout and time() > timeout:
                raise Exception(
                    f"Cluster start up timed out after {timeout_seconds} seconds."
                )

            sleep(poll_rate_seconds)
            operation = self.get_cluster_operation(operation_id).result

        if operation.result.error:
            raise Exception("Failed to start Cluster", operation.result.error)
        else:
            return self.get_cluster(operation.cluster_id).result

    def fetch_production_job_logs(self, job_id: str) -> str:
        from anyscale.controllers.job_controller import JobController

        """
        Retrieves logs for a job (eg. 'prodjob_123').
        - This will retrieve logs from only the latest job run of the job.

        Args:
            job_id (str): ID of a job

        Returns:
            str: All logs for the job
                (so far, if the job is still running)
        """
        string_logger = StringLogger()
        job_controller = JobController(string_logger, auth_token=self._auth_token)
        job_controller.logs(job_id, no_cluster_journal_events=True)

        return string_logger.out_string

    def create_job(self, create_production_job, **kwargs):
        _upload_and_rewrite_working_dir_in_create_production_job(create_production_job)
        return super().create_job(create_production_job, **kwargs)

    def create_service(self, create_production_service, **kwargs):
        _upload_and_rewrite_working_dir_in_create_production_job(
            create_production_service
        )
        return super().create_service(create_production_service, **kwargs)

    def apply_service(self, create_production_service, **kwargs):
        _upload_and_rewrite_working_dir_in_create_production_job(
            create_production_service
        )
        return super().apply_service(create_production_service, **kwargs)

    def rollout_service_v2(self, apply_production_service_v2_model, **kwargs):
        """
        DEPRECATED. Please use rollout_service.
        """
        if isinstance(apply_production_service_v2_model, dict):
            ray_serve_config = apply_production_service_v2_model.get("ray_serve_config")
            new_ray_serve_config = _upload_and_rewrite_working_dir_ray_serve_config(
                ray_serve_config
            )
            apply_production_service_v2_model["ray_serve_config"] = new_ray_serve_config
        elif hasattr(apply_production_service_v2_model, "ray_serve_config"):
            ray_serve_config = apply_production_service_v2_model.ray_serve_config
            new_ray_serve_config = _upload_and_rewrite_working_dir_ray_serve_config(
                ray_serve_config
            )
            apply_production_service_v2_model.ray_serve_config = new_ray_serve_config

        return super().rollout_service_v2(apply_production_service_v2_model, **kwargs)

    def rollout_service(self, apply_service_model, **kwargs):
        """
        If a local working_dir and upload_path are specified in ray_serve_config,
        this method uploads the local working_dir to the upload_path and rewrites
        the working_dir in ray_serve_config.
        """
        if isinstance(apply_service_model, dict):
            ray_serve_config = apply_service_model.get("ray_serve_config")
            new_ray_serve_config = _upload_and_rewrite_working_dir_ray_serve_config(
                ray_serve_config
            )
            apply_service_model["ray_serve_config"] = new_ray_serve_config
        elif hasattr(apply_service_model, "ray_serve_config"):
            ray_serve_config = apply_service_model.ray_serve_config
            new_ray_serve_config = _upload_and_rewrite_working_dir_ray_serve_config(
                ray_serve_config
            )
            apply_service_model.ray_serve_config = new_ray_serve_config

        return super().rollout_service(apply_service_model, **kwargs)

    def get_cluster_token(self, project_id: str, cluster_name: str) -> str:
        """
        Retrieves cluster token for a Cluster.
        This cluster token can be used to authenticate Anyscale services (eg: serve, jupyter, grafana)

        Args:
            project_id - ID of the Project the Cluster belongs to
            cluster_name - Name of the Cluster

        Returns
            string containing cluster token

        Raises
            Exception if invalid cluster name
        """
        clusters = self.search_clusters(
            {"project_id": project_id, "name": {"equals": cluster_name}}
        ).results

        if clusters:
            cluster = clusters[0]
        else:
            raise Exception(
                (
                    f"Cluster named {cluster_name} does not exist in Project {project_id} "
                    "or this user does not have read permissions for it."
                )
            )
        cluster_token = cluster.access_token
        return cluster_token

    # =========================================================================
    # Deprecated methods - these raise AnyscaleSDKDeprecationError
    # =========================================================================

    # Cloud
    def get_cloud(self, cloud_id, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.get_cloud() is deprecated. "
            "Please use anyscale.cloud.get() instead. "
            "See https://docs.anyscale.com/reference/cloud for details."
        )

    def get_default_cloud(self, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.get_default_cloud() is deprecated. "
            "Please use anyscale.cloud.get_default() instead. "
            "See https://docs.anyscale.com/reference/cloud for details."
        )

    def search_clouds(self, clouds_query, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.search_clouds() is deprecated. "
            "Please use anyscale.cloud.list() instead. "
            "See https://docs.anyscale.com/reference/cloud for details."
        )

    # Cluster Environment
    def create_byod_cluster_environment_build(self, create_byod_cluster_environment_build, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.create_byod_cluster_environment_build() is deprecated. "
            "Please use anyscale.image.register() instead. "
            "See https://docs.anyscale.com/reference/image for details."
        )

    def create_cluster_environment_build(self, create_cluster_environment_build, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.create_cluster_environment_build() is deprecated. "
            "Please use anyscale.image.build() instead. "
            "See https://docs.anyscale.com/reference/image for details."
        )

    def find_cluster_environment_build_by_identifier(self, identifier, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.find_cluster_environment_build_by_identifier() is deprecated. "
            "Please use anyscale.image.get() instead. "
            "See https://docs.anyscale.com/reference/image for details."
        )

    def get_cluster_environment_build(self, cluster_environment_build_id, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.get_cluster_environment_build() is deprecated. "
            "Please use anyscale.image.get() instead. "
            "See https://docs.anyscale.com/reference/image for details."
        )

    def list_cluster_environment_builds(self, cluster_environment_id, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.list_cluster_environment_builds() is deprecated. "
            "Please use anyscale.image.list() instead. "
            "See https://docs.anyscale.com/reference/image for details."
        )

    def create_byod_cluster_environment(self, create_byod_cluster_environment, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.create_byod_cluster_environment() is deprecated. "
            "Please use anyscale.image.register() instead. "
            "See https://docs.anyscale.com/reference/image for details."
        )

    def create_cluster_environment(self, create_cluster_environment, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.create_cluster_environment() is deprecated. "
            "Please use anyscale.image.build() instead. "
            "See https://docs.anyscale.com/reference/image for details."
        )

    def get_cluster_environment(self, cluster_environment_id, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.get_cluster_environment() is deprecated. "
            "Please use anyscale.image.get() instead. "
            "See https://docs.anyscale.com/reference/image for details."
        )

    def search_cluster_environments(self, cluster_environments_query, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.search_cluster_environments() is deprecated. "
            "Please use anyscale.image.list() instead. "
            "See https://docs.anyscale.com/reference/image for details."
        )
        
    # Compute Config
    def create_cluster_compute(self, create_cluster_compute, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.create_cluster_compute() is deprecated. "
            "Please use anyscale.compute_config.create() instead. "
            "See https://docs.anyscale.com/reference/compute-config-api for details."
        )

    def delete_cluster_compute(self, cluster_compute_id, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.delete_cluster_compute() is deprecated. "
            "Please use anyscale.compute_config.archive() instead. "
            "See https://docs.anyscale.com/reference/compute-config-api for details."
        )

    def get_cluster_compute(self, cluster_compute_id, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.get_cluster_compute() is deprecated. "
            "Please use anyscale.compute_config.get() instead. "
            "See https://docs.anyscale.com/reference/compute-config-api for details."
        )

    def get_default_cluster_compute(self, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.get_default_cluster_compute() is deprecated. "
            "Please use anyscale.compute_config.get_default() instead. "
            "See https://docs.anyscale.com/reference/compute-config-api for details."
        )

    def search_cluster_computes(self, cluster_computes_query, **kwargs):
        raise AnyscaleSDKDeprecationError(
            "AnyscaleSDK.search_cluster_computes() is deprecated. "
            "Please use anyscale.compute_config.list() instead. "
            "See https://docs.anyscale.com/reference/compute-config-api for details."
        )
