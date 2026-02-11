import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import click
import tabulate
import yaml

from anyscale.cli_logger import BlockLogger
from anyscale.cloud_utils import get_cloud_id_and_name, get_last_used_cloud
from anyscale.cluster_compute import (
    get_cluster_compute_from_name,
    get_default_cluster_compute,
    get_selected_cloud_id_or_default,
)
from anyscale.cluster_env import (
    get_build_from_cluster_env_identifier,
    get_default_cluster_env_build,
)
from anyscale.controllers.base_controller import BaseController
from anyscale.project_utils import (
    get_and_validate_project_id,
    get_proj_id_from_name,
    get_project_id,
    load_project_or_throw,
    validate_project_id,
)
from anyscale.sdk.anyscale_client import (
    Cluster,
    ClusterEnvironment,
    ClusterEnvironmentBuild,
    ComputeTemplate,
    CreateCluster,
    SessionState,
    StartClusterOptions,
    UpdateCluster,
)
from anyscale.sdk.anyscale_client.models.cluster_compute_config import (
    ClusterComputeConfig,
)
from anyscale.sdk.anyscale_client.models.create_cluster_compute import (
    CreateClusterCompute,
)
from anyscale.shared_anyscale_utils.util import slugify
from anyscale.shared_anyscale_utils.utils.byod import BYOD_DISPLAY_PREFIX, BYODInfo
from anyscale.util import (
    extract_versions_from_image_name,
    get_endpoint,
    PROJECT_NAME_ENV_VAR,
    wait_for_session_start,
)
from anyscale.utils.cluster_debug import debug_cluster
from anyscale.utils.name_utils import gen_valid_name


class ClusterController(BaseController):
    def __init__(
        self, log: Optional[BlockLogger] = None, initialize_auth_api_client: bool = True
    ):
        if log is None:
            log = BlockLogger()

        super().__init__(initialize_auth_api_client=initialize_auth_api_client)
        self.log = log
        self.log.open_block("Output")

    def start(  # noqa: PLR0912, PLR0913
        self,
        cluster_name: Optional[str],
        cluster_id: Optional[str],
        cluster_env_name: Optional[str],
        docker: Optional[str],
        python_version: Optional[str],
        ray_version: Optional[str],
        cluster_compute_name: Optional[str],
        cluster_compute_file: Optional[str],
        cloud_name: Optional[str],
        idle_timeout: Optional[int],
        project_id: Optional[str],
        project_name: Optional[str],
        user_service_access: Optional[str],
    ) -> None:
        if cluster_name is not None and cluster_id is not None:
            raise click.ClickException(
                "`--name` and `--cluster-id` cannot both be specified. Please only provide one"
                "of these two arguments."
            )
        if cluster_compute_name is not None and cluster_compute_file is not None:
            raise click.ClickException(
                "`--compute` and `--compute-file` cannot both be specified. Please only provide one"
                "of these two arguments."
            )
        if (
            cluster_compute_name is not None or cluster_compute_file is not None
        ) and cloud_name is not None:
            self.log.warning(
                f"Because {cluster_compute_name or cluster_compute_file} were specified, the `--cloud_name` "
                f"argument {cloud_name} will be ignored."
            )
        if cluster_env_name is not None and docker is not None:
            raise click.ClickException(
                "`--env` and `--docker` cannot both be specified. Please only provide one"
                "of these two arguments."
            )
        if docker is not None and python_version is None and ray_version is None:
            python_version, ray_version = extract_versions_from_image_name(docker)

        if docker is not None and python_version is None:
            raise click.ClickException(
                "`--python-version` should be specified when `--docker` is used."
            )
        if docker is not None and ray_version is None:
            raise click.ClickException(
                "`--ray-version` should be specified when `--docker` is used."
            )

        # Try to get project id and cluster name of cluster per arguments to
        # see if the cluster already exists. If the cluster doesn't already exist,
        # get the project id and cluster name of a cluster to create.
        cluster_compute_id, cluster_compute_config = None, None
        if cluster_compute_name:
            cluster_compute_id = get_cluster_compute_from_name(
                cluster_compute_name, self.api_client,
            ).id
        elif cluster_compute_file:
            cluster_compute_config = self._load_cluster_compute(cluster_compute_file)
        # parent_cloud_id used to search for existing cluster in default project
        # if cloud isolation is enabled and we are unable to find the cluster with the
        # provided project_id.
        parent_cloud_id = get_selected_cloud_id_or_default(
            self.api_client,
            self.anyscale_api_client,
            cluster_compute_id=cluster_compute_id,
            cluster_compute_config=cluster_compute_config,
            cloud_id=None,
            cloud_name=cloud_name,
        )
        project_id, cluster_name = self._get_project_id_and_cluster_name(
            cluster_id,
            project_id,
            cluster_name,
            project_name,
            cloud_id=parent_cloud_id,
        )

        cluster_list = self.anyscale_api_client.search_clusters(
            {
                "project_id": project_id,
                "name": {"equals": cluster_name},
                "archive_status": "ALL",
            }
        ).results
        existing_cluster = cluster_list[0] if len(cluster_list) > 0 else None

        if docker is not None:
            assert cluster_env_name is None
            assert python_version
            assert ray_version
            cluster_env_name = BYODInfo(
                docker_image_name=docker,
                python_version=python_version,
                ray_version=ray_version,
            ).encode()

        cluster_env, build = self._get_cluster_env_and_build(cluster_env_name)
        cluster_compute = self._get_cluster_compute(
            cluster_compute_name,
            cluster_compute_file,
            cloud_name,
            existing_cluster,
            project_id,
        )
        cloud = self.anyscale_api_client.get_cloud(
            cluster_compute.config.cloud_id
        ).result

        # Whether configurations that could require a restart were passed
        passed_cluster_env = bool(cluster_env_name)
        passed_cluster_compute = bool(cluster_compute_name or cluster_compute_file)
        cluster, needs_start = self._create_or_update_cluster_data(
            existing_cluster,
            cluster_name,
            project_id,
            build,
            cluster_compute,
            passed_cluster_env,
            passed_cluster_compute,
            idle_timeout,
            user_service_access,
        )

        url = get_endpoint(f"/projects/{project_id}/clusters/{cluster.id}")
        if cluster_env.name.startswith(BYOD_DISPLAY_PREFIX):
            # Don't include build.revision or build.id for BYOD images
            cluster_environment_display = f"{cluster_env.name}"
        else:
            cluster_environment_display = (
                f"{cluster_env.name}.{build.revision} (id={build.id})"
            )

        cluster_start_parameters_str = (
            f"\t\t\tCluster environment: {cluster_environment_display}\n"
            f"\t\t\tCluster compute: {cluster_compute.name} (id={cluster_compute.id})\n"
            f"\t\t\tProject: {project_id}\n"
            f"\t\t\tCloud: {cloud.name}\n"
            f"\t\t\tURL: {url}"
        )
        if needs_start:
            self.anyscale_api_client.start_cluster(
                cluster.id,
                StartClusterOptions(
                    cluster_environment_build_id=build.id,
                    cluster_compute_id=cluster_compute.id,
                ),
            )
            self.log.info(
                f"Starting cluster {cluster.name} (id={cluster.id}) with the following parameters:\n"
                f"{cluster_start_parameters_str}"
            )
            wait_for_session_start(project_id, cluster_name, self.api_client)
            self.log.info(f"Cluster {cluster_name} finished starting.")
        else:
            self.log.info(
                f"Cluster {cluster_name} is currently running with the following parameters:\n"
                f"{cluster_start_parameters_str}"
            )
            self.log.info(f"View at {url}")

    def terminate(
        self,
        cluster_name: Optional[str],
        cluster_id: Optional[str],
        project_id: Optional[str],
        project_name: Optional[str],
        cloud_id: Optional[str],
        cloud_name: Optional[str],
    ) -> None:
        self._validate_args_to_find_cluster(
            cluster_name, cluster_id, project_name, project_id, cloud_name, cloud_id
        )
        if cloud_id or cloud_name:
            cloud_id, _ = get_cloud_id_and_name(
                api_client=self.api_client, cloud_id=cloud_id, cloud_name=cloud_name,
            )
        project_id, cluster_name = self._get_project_id_and_cluster_name(
            cluster_id, project_id, cluster_name, project_name, cloud_id=cloud_id
        )

        cluster_list = self.anyscale_api_client.search_clusters(
            {"project_id": project_id, "name": {"equals": cluster_name}}
        ).results
        if len(cluster_list) == 0:
            raise click.ClickException(
                f"No cluster {cluster_name} found in project {project_id}."
            )

        cluster = cluster_list[0]
        self.anyscale_api_client.terminate_cluster(cluster.id, {})

        url = get_endpoint(f"/projects/{project_id}/clusters/{cluster.id}")
        self.log.info(f"Cluster {cluster_name} is terminating. View progress at {url}")

    def archive(
        self,
        cluster_name: Optional[str],
        cluster_id: Optional[str],
        project_id: Optional[str],
        project_name: Optional[str],
        cloud_id: Optional[str],
        cloud_name: Optional[str],
    ) -> None:
        self._validate_args_to_find_cluster(
            cluster_name, cluster_id, project_name, project_id, cloud_name, cloud_id
        )
        if cloud_id or cloud_name:
            cloud_id, _ = get_cloud_id_and_name(
                api_client=self.api_client, cloud_id=cloud_id, cloud_name=cloud_name,
            )
        project_id, cluster_name = self._get_project_id_and_cluster_name(
            cluster_id, project_id, cluster_name, project_name, cloud_id
        )

        cluster_list = self.anyscale_api_client.search_clusters(
            {"project_id": project_id, "name": {"equals": cluster_name}}
        ).results
        if len(cluster_list) == 0:
            raise click.ClickException(
                f"No cluster {cluster_name} found in project {project_id}."
            )

        cluster = cluster_list[0]

        self.anyscale_api_client.archive_cluster(cluster.id)
        self.log.info(f"Cluster {cluster.id} is successfully archived.")

    def list(  # noqa: PLR0912, PLR0913
        self,
        cluster_name: Optional[str],
        cluster_id: Optional[str],
        project_id: Optional[str],
        include_all_projects: bool,
        include_inactive: bool,
        include_archived: bool,
        max_items: int,
        project_name: Optional[str],
        cloud_id: Optional[str],
        cloud_name: Optional[str],
    ) -> None:
        if (
            project_id is not None or project_name is not None
        ) and include_all_projects:
            self.log.warning(
                f"Because `include_all_projects` was specified, the `--project-id` "
                f"argument {project_id} will be ignored."
            )

        # if cluster id is specified --> get
        # if cluster name is specified, find in project and get
        # if not include all proejcts, get current project
        cluster_list = []
        if cluster_id is not None:
            # Get cluster from id if specified
            cluster_list.append(
                self.api_client.get_decorated_cluster_api_v2_decorated_sessions_cluster_id_get(
                    cluster_id
                ).result
            )
        elif cluster_name is not None:
            # Find cluster name in project if cluster_name specified
            self._validate_args_to_find_cluster(
                cluster_name, cluster_id, project_name, project_id, cloud_name, cloud_id
            )
            if cloud_id or cloud_name:
                cloud_id, _ = get_cloud_id_and_name(
                    api_client=self.api_client,
                    cloud_id=cloud_id,
                    cloud_name=cloud_name,
                )
            project_id, cluster_name = self._get_project_id_and_cluster_name(
                cluster_id, project_id, cluster_name, project_name, cloud_id=cloud_id,
            )
            cluster_list.extend(
                self.api_client.list_decorated_clusters_api_v2_decorated_sessions_get(
                    project_id=project_id, name_match=cluster_name
                ).results
            )
        else:
            search_clusters_query: Dict[str, Any] = {}
            if not include_all_projects:
                # Specify project id if not include_all_projects
                if project_id:
                    validate_project_id(project_id, self.api_client)
                elif project_name:
                    project_id = get_proj_id_from_name(project_name, self.api_client)
                else:
                    try:
                        project_definition = load_project_or_throw()
                        project_id = get_project_id(project_definition.root)
                    except click.ClickException:
                        parent_cloud_id = None
                        log_msg = "No project context detected or `--project-id` provided. Continuing without a project."
                        if not (cloud_id or cloud_name):
                            cloud_name = get_last_used_cloud(
                                project_id=None,
                                anyscale_api_client=self.anyscale_api_client,
                            )
                        parent_cloud_id, _ = get_cloud_id_and_name(
                            api_client=self.api_client,
                            cloud_id=cloud_id,
                            cloud_name=cloud_name,
                        )
                        log_msg = f"No project context detected or `--project-id` provided. Continuing without a project assuming cloud {parent_cloud_id}."
                        default_project = self.anyscale_api_client.get_default_project(
                            parent_cloud_id=parent_cloud_id
                        ).result
                        project_id = default_project.id
                        self.log.info(log_msg)
                search_clusters_query["project_id"] = project_id
            if not include_inactive:
                search_clusters_query["state_filter"] = ["Running"]
            if include_archived:
                search_clusters_query["archive_status"] = "ALL"

            # Page through all clusters in response
            cluster_list_resp = self.api_client.list_decorated_clusters_api_v2_decorated_sessions_get(
                **search_clusters_query, count=20
            )
            next_paging_token = cluster_list_resp.metadata.next_paging_token
            cluster_list.extend(cluster_list_resp.results)
            has_more = (next_paging_token is not None) and (
                len(cluster_list) < max_items
            )
            while has_more:
                cluster_list_resp = self.api_client.list_decorated_clusters_api_v2_decorated_sessions_get(
                    **search_clusters_query, paging_token=next_paging_token, count=20
                )
                next_paging_token = cluster_list_resp.metadata.next_paging_token
                cluster_list.extend(cluster_list_resp.results)
                has_more = (next_paging_token is not None) and (
                    len(cluster_list) < max_items
                )
            cluster_list = cluster_list[:max_items]

        clusters_table = [
            [
                cluster.name,
                cluster.id,
                cluster.state,
                cluster.cloud_id,
                get_endpoint(f"/projects/{cluster.project_id}/clusters/{cluster.id}"),
            ]
            for cluster in cluster_list
        ]

        table = tabulate.tabulate(
            clusters_table,
            headers=["NAME", "ID", "STATE", "CLOUD ID", "URL",],
            tablefmt="plain",
        )
        print(f"Clusters:\n{table}")

    def debug_networking(self, cluster_id: Optional[str],) -> bool:
        cluster = self.anyscale_api_client.get_cluster(cluster_id).result
        if cluster.state != SessionState.RUNNING:
            raise click.ClickException(
                f"Cluster {cluster_id} is not running. Cannot debug networking"
            )

        # TODO(ilr) modify messaging based on whether or not the cloud of this cluster uses Private IPs
        url = cluster.head_node_info.url
        self.log.info(
            f"Beginning network diagnostics for cluster {cluster_id} at {url}"
        )

        success = True
        for test_result in debug_cluster(
            self.log, urlparse(url).netloc, cluster.head_node_info.ip_address
        ):
            if test_result.success:
                self.log.info(f"{test_result.name}: ✅")
                if test_result.warnings:
                    self.log.warning(test_result.warnings)
            else:
                self.log.error(
                    f"{test_result.name}: ❌ Error Message: {test_result.error}"
                )
                success = False
        return success

    # Helpers

    def _validate_args_to_find_cluster(
        self,
        cluster_name: Optional[str],
        cluster_id: Optional[str],
        project_name: Optional[str],
        project_id: Optional[str],
        cloud_name: Optional[str],
        cloud_id: Optional[str],
    ):
        """
        Validate arguments to ensure we can find the cluster to take action on
        (eg: for commands like archive and terminate).
        """
        if cluster_name is not None and cluster_id is not None:
            raise click.ClickException(
                "`--name` and `--cluster-id` cannot both be specified. Please only provide one "
                "of these two arguments."
            )
        if cluster_name is None and cluster_id is None:
            raise click.ClickException(
                "Please specity one of `--name` or `--cluster-id`."
            )
        if cloud_name is not None and cloud_id is not None:
            raise click.ClickException(
                "`--cloud-name` and `--cloud-id` cannot both be specified. Please only provide one "
                "of these two arguments."
            )
        if cluster_name is not None and (
            project_id is None
            and project_name is None
            and cloud_id is None
            and cloud_name is None
        ):
            raise click.ClickException(
                "Please specify the project of the cluster through `--project-id` or `--project-name`. If the cluster "
                "doesn't belong to any project, please specify the cloud id with `--cloud-id`."
            )

    def _load_cluster_compute(self, cluster_compute_file: str) -> ClusterComputeConfig:
        with open(cluster_compute_file) as f:
            config_dict: Dict[str, Any] = yaml.safe_load(f)
        cluster_compute_config = ClusterComputeConfig(**config_dict)
        return cluster_compute_config

    def _get_project_id_and_cluster_name(
        self,
        cluster_id: Optional[str],
        project_id: Optional[str],
        cluster_name: Optional[str],
        project_name: Optional[str],
        cloud_id: Optional[str],
    ) -> Tuple[str, str]:
        """
        Get cluster name and project id. If cluster id is specified, any cluster can
        be started (including outside the current or specified project).
        """
        project_name = project_name or os.environ.get(PROJECT_NAME_ENV_VAR)
        if cluster_id:
            try:
                cluster = self.anyscale_api_client.get_cluster(cluster_id).result
                cluster_name = cluster.name
                project_id = cluster.project_id
                assert project_id and cluster_name  # For mypy
            except Exception:  # noqa: BLE001
                raise click.ClickException(f"No cluster exists with id {cluster_id}.")
        else:
            project_id = get_and_validate_project_id(
                project_id=project_id,
                project_name=project_name,
                parent_cloud_id=cloud_id,
                api_client=self.api_client,
                anyscale_api_client=self.anyscale_api_client,
            )
            assert project_id is not None
            cluster_name = self._get_or_generate_cluster_name(project_id, cluster_name)
        return project_id, cluster_name

    def _get_cluster_env_and_build(
        self, cluster_env_name: Optional[str]
    ) -> Tuple[ClusterEnvironment, ClusterEnvironmentBuild]:
        if cluster_env_name:
            build = get_build_from_cluster_env_identifier(
                cluster_env_name, self.anyscale_api_client
            )
        else:
            # Use default cluster environment per ray and python version
            build = get_default_cluster_env_build(
                self.api_client, self.anyscale_api_client
            )
        cluster_env = self.anyscale_api_client.get_cluster_environment(
            build.cluster_environment_id
        ).result
        return cluster_env, build

    def _get_cluster_compute(
        self,
        cluster_compute_name: Optional[str],
        cluster_compute_file: Optional[str],
        cloud_name: Optional[str],
        existing_cluster: Optional[Cluster],
        project_id: Optional[str],
    ) -> ComputeTemplate:
        if cluster_compute_name:
            cluster_compute = get_cluster_compute_from_name(
                cluster_compute_name, self.api_client,
            )
        elif cluster_compute_file:
            cluster_compute_config = self._load_cluster_compute(cluster_compute_file)
            name = gen_valid_name("cli-config")
            cluster_compute = self.anyscale_api_client.create_cluster_compute(
                CreateClusterCompute(name=name, config=cluster_compute_config)
            ).result
        elif not cloud_name and existing_cluster:
            cluster_compute = self.anyscale_api_client.get_cluster_compute(
                existing_cluster.cluster_compute_id
            ).result
        else:
            cluster_compute = get_default_cluster_compute(
                cloud_name, project_id, self.api_client, self.anyscale_api_client
            )
        return cluster_compute

    def _get_or_generate_cluster_name(
        self, project_id: str, cluster_name: Optional[str]
    ) -> str:
        """
        Return slugified cluster name if provided, else generate default cluster name from project id.
        """
        if not cluster_name:
            cluster_name = str(
                self.api_client.get_project_default_session_name_api_v2_projects_project_id_default_session_name_get(
                    project_id=project_id,
                ).result.name
            )
        else:
            cluster_name = slugify(cluster_name)
        assert cluster_name, f"Cluster name {cluster_name} is invalid."
        return cluster_name

    def _create_or_update_cluster_data(  # noqa: PLR0913
        self,
        existing_cluster: Optional[Cluster],
        cluster_name: str,
        project_id: str,
        build: ClusterEnvironmentBuild,
        cluster_compute: ComputeTemplate,
        passed_cluster_env: bool,
        passed_cluster_compute: bool,
        idle_timeout: Optional[int],
        user_service_access: Optional[str],
    ) -> Tuple[Cluster, bool]:
        """
        Creates new cluster with specified parameters if it doesn't already exist. Otherwise
        update the idle timeout of the existing cluster if necessary. Returns Cluster object
        and whether this cluster needs to be started.
        """
        if not existing_cluster:
            # Create a new cluster if there is no existing cluster with the given cluster_name

            create_cluster_data = {
                "name": cluster_name,
                "project_id": project_id,
                "cluster_compute_id": cluster_compute.id,
                "cluster_environment_build_id": build.id,
            }
            if user_service_access:
                create_cluster_data["user_service_access"] = user_service_access
            if idle_timeout:
                create_cluster_data["idle_timeout_minutes"] = idle_timeout

            cluster = self.anyscale_api_client.create_cluster(
                CreateCluster(**create_cluster_data)
            ).result
            needs_start = True
        else:
            # Get the existing session and update the idle_timeout if required
            cluster = existing_cluster
            needs_start = self._check_needs_start(
                cluster,
                build,
                cluster_compute,
                passed_cluster_env,
                passed_cluster_compute,
            )
            if idle_timeout:
                self.anyscale_api_client.update_cluster(
                    cluster.id, UpdateCluster(idle_timeout_minutes=idle_timeout)
                )
                self.log.info(
                    f"Updated idle timeout minutes to {idle_timeout} minutes."
                )

        return (cluster, needs_start)

    def _check_needs_start(
        self,
        current_cluster: Cluster,
        build: ClusterEnvironmentBuild,
        cluster_compute: ComputeTemplate,
        passed_cluster_env: bool,
        passed_cluster_compute: bool,
    ) -> bool:
        """
        Checks if existing cluster needs be restarted based on it's current configuration
        and the configuraiton passed in the command.
        """
        if current_cluster.state != "Running":
            self.log.info(
                f"Cluster {current_cluster.name} will be restarted because it's current state is {current_cluster.state}."
            )
            return True
        if not passed_cluster_env and not passed_cluster_compute:
            self.log.info(
                f"Cluster {current_cluster.name} does not need to be restarted because no configuration values were passed in."
            )
            return False

        needs_start = False
        if (
            passed_cluster_env
            and current_cluster.cluster_environment_build_id != build.id
        ):
            current_cluster_build = self.anyscale_api_client.get_cluster_environment_build(
                current_cluster.cluster_environment_build_id
            ).result
            current_cluster_env = self.anyscale_api_client.get_cluster_environment(
                current_cluster_build.cluster_environment_id
            ).result
            cluster_env = self.anyscale_api_client.get_cluster_environment(
                build.cluster_environment_id
            ).result
            self.log.info(
                f"The cluster is currently using {current_cluster_env.name}:{current_cluster_build.revision} as the cluster env "
                f"and {cluster_env.name}:{build.revision} was provided, so the cluster will be updated to use "
                "the passed cluster environment."
            )
            needs_start = True
        if (
            passed_cluster_compute
            and current_cluster.cluster_compute_id != cluster_compute.id
        ):
            current_cluster_compute = self.anyscale_api_client.get_cluster_compute(
                current_cluster.cluster_compute_id
            ).result
            self.log.info(
                f"The cluster is currently using ({current_cluster_compute.name}) (id={current_cluster_compute.id}) as the cluster "
                f"compute config and {cluster_compute.name} (id={cluster_compute.id}) was provided, so the cluster will "
                "be updated to use the passed cluster compute config."
            )
            needs_start = True
        return needs_start
