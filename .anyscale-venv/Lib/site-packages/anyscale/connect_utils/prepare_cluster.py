from datetime import datetime
from enum import Enum
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from packaging import version

from anyscale.authenticate import get_auth_api_client
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models import Build, CreateComputeTemplate, Session
from anyscale.cluster_compute import (
    get_cluster_compute_from_name,
    get_selected_cloud_id_or_default,
)
from anyscale.conf import MINIMUM_RAY_VERSION
from anyscale.links import DOCS_CLUSTER
from anyscale.sdk.anyscale_client import (
    ComputeTemplateConfig,
    CreateCluster,
    StartClusterOptions,
    UpdateCluster,
)
from anyscale.sdk.anyscale_client.models.compute_template import ComputeTemplate
from anyscale.sdk.anyscale_client.models.session_state import SessionState
from anyscale.util import generate_slug, get_endpoint, wait_for_session_start
from anyscale.utils.connect_helpers import get_cluster, list_entities
from anyscale.utils.name_utils import gen_valid_name


# Max number of auto created clusters.
MAX_CLUSTERS = 40

# The type of the dict that can be passed to create a cluster env.
# e.g., {"base_image": "anyscale/ray-ml:1.1.0-gpu"}
CLUSTER_ENV_DICT_TYPE = Dict[str, Union[str, List[str]]]

# The cluster compute type. It can either be a string, eg my_template or a dict,
# eg, {"cloud_id": "id-123" ...}
CLUSTER_COMPUTE_DICT_TYPE = Dict[str, Any]

# Commands used to build Ray from source. Note that intermediate stages will
# be cached by the app config builder.
BUILD_STEPS = [
    "git clone https://github.com/ray-project/ray.git",
    "curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg",
    "sudo mv bazel.gpg /etc/apt/trusted.gpg.d/",
    'echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list',
    "sudo apt-get update && sudo apt-get install -y bazel=3.2.0",
    'cd ray/python && sudo env "PATH=$PATH" python setup.py develop',
    "pip uninstall -y ray",
]

# Default minutes for autosuspend.
DEFAULT_AUTOSUSPEND_TIMEOUT = 120


# Default docker images to use for connect clusters.
def _get_base_image(image: str, ray_version: str, cpu_or_gpu: str) -> str:
    py_version = "".join(str(x) for x in sys.version_info[0:2])
    if py_version not in ["36", "37", "38"]:
        raise ValueError(f"No default docker image for py{py_version}")
    return f"anyscale/{image}:{ray_version}-py{py_version}-{cpu_or_gpu}"


class PrepareClusterAction(str, Enum):
    START = "START"
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    WAIT = "WAIT"
    NO_OP = "NO_OP"


class PrepareClusterBlock:
    def __init__(  # noqa: PLR0913
        self,
        project_id: str,
        cluster_name: Optional[str],
        autosuspend_timeout: Optional[int],
        allow_public_internet_traffic: Optional[bool],
        needs_update: bool,
        cluster_compute_name: Optional[str],
        cluster_compute_dict: Optional[CLUSTER_COMPUTE_DICT_TYPE],
        cloud_name: Optional[str],
        build_pr: Optional[int],
        force_rebuild: bool,
        build_commit: Optional[str],
        cluster_env_name: Optional[str],
        cluster_env_dict: Optional[CLUSTER_ENV_DICT_TYPE],
        cluster_env_revision: Optional[int],
        ray: Any,
        log_output: bool = True,
    ):
        """
        Class to prepare a cluster (create and start if required) so it can be
        connected to via ray.client. Relevant information about the cluster is
        stored in the instance variables.
        """
        self.project_id = project_id
        self.cluster_name = cluster_name
        self.autosuspend_timeout = autosuspend_timeout
        self.allow_public_internet_traffic = allow_public_internet_traffic
        self.needs_update = needs_update
        self.cluster_compute_name = cluster_compute_name
        self.cluster_compute_dict = cluster_compute_dict
        self.cloud_name = cloud_name
        self.build_pr = build_pr
        self.force_rebuild = force_rebuild
        self.build_commit = build_commit
        self.cluster_env_name = cluster_env_name
        self.cluster_env_dict = cluster_env_dict
        self.cluster_env_revision = cluster_env_revision
        self._ray = ray

        auth_api_client = get_auth_api_client(log_output=log_output)
        self.api_client = auth_api_client.api_client
        self.anyscale_api_client = auth_api_client.anyscale_api_client

        self.log = BlockLogger(log_output=log_output)
        self.block_label = "PrepareCluster"
        self.log.open_block(self.block_label, block_title="Preparing the cluster")

    def prepare(self):
        prepare_cluster_action = self._get_prepare_cluster_action(
            self.project_id, self.cluster_name, self.needs_update
        )

        if prepare_cluster_action in {
            PrepareClusterAction.CREATE,
            PrepareClusterAction.START,
            PrepareClusterAction.UPDATE,
        }:
            existing_terminated_cluster = None
            if self.cluster_name:
                existing_terminated_cluster = get_cluster(
                    self.anyscale_api_client, self.project_id, self.cluster_name
                )

            self.cluster_env_name = self._build_cluster_env_if_needed(
                self.project_id,
                self.build_pr,
                self.build_commit,
                self.cluster_env_dict,
                self.cluster_env_name,
                self.force_rebuild,
            )

            # If the cluster build is not explicitly provided by the user
            # then we use the existing cluster's build id (if the cluster
            # already exists). Otherwise, we get the default build ID.
            if self.cluster_env_name or existing_terminated_cluster is None:
                build_id = self._get_cluster_build(
                    self.cluster_env_name, self.cluster_env_revision
                ).id
            else:
                build_id = existing_terminated_cluster.build_id

            cloud_id = get_selected_cloud_id_or_default(
                api_client=self.api_client,
                anyscale_api_client=self.anyscale_api_client,
                cloud_name=self.cloud_name,
            )
            if (
                (existing_terminated_cluster is not None)
                and (existing_terminated_cluster.cloud_id is not None)
                and (self.cloud_name is not None)  # user explicitly passed cloud var.
                and (
                    cloud_id != existing_terminated_cluster.cloud_id
                )  # cloud_id is the id of self.cloud_name
            ):
                raise ValueError(
                    "Current cluster already has a cloud. Changing it to a new cloud is not allowed. If you need a new cloud, please create a new cluster."
                )

            # If the user explicitly passes a cluster compute (dict or name),
            # Or if the cluster does not exist,
            # Or if the cluster already exist but the user is changing clouds,
            # Then get the updated cluster compute.
            # Otherwise, get the compute template of the existing terminated cluster.
            if (
                self.cluster_compute_name
                or self.cluster_compute_dict
                or existing_terminated_cluster is None
                or (
                    self.cloud_name is not None  # user explicitly passed cloud var.
                    and cloud_id
                    != existing_terminated_cluster.cloud_id  # cloud_id is the id of self._cloud_name
                )
            ):
                compute_template_id = self._get_cluster_compute_id(
                    self.project_id,
                    self.cluster_compute_name,
                    self.cluster_compute_dict,
                    self.cloud_name,
                )
            else:
                compute_template_id = existing_terminated_cluster.compute_template_id

            self._wait_for_app_build(self.project_id, build_id)

            self.cluster_name = self._start_or_create_cluster(
                project_id=self.project_id,
                build_id=build_id,
                compute_template_id=compute_template_id,
                cluster_name=self.cluster_name,
                autosuspend_timeout=self.autosuspend_timeout,
                allow_public_internet_traffic=self.allow_public_internet_traffic,
            )
        else:
            assert (
                self.cluster_name
            ), f"{self.cluster_name} is None, yet _derive_prepare_cluster_action returned {prepare_cluster_action}."
            cluster = get_cluster(
                self.anyscale_api_client, self.project_id, self.cluster_name
            )
            assert (
                cluster
            ), f"Cluster {self.cluster_name} not found, yet _derive_prepare_cluster_action returned {prepare_cluster_action}."
            self._validate_new_cluster_compute_and_env_match_existing_cluster(
                project_id=self.project_id, running_cluster=cluster,
            )
            if prepare_cluster_action == PrepareClusterAction.WAIT:
                cluster = get_cluster(
                    self.anyscale_api_client, self.project_id, self.cluster_name
                )
                wait_for_session_start(
                    self.project_id,
                    cluster.name,
                    self.api_client,
                    log=self.log,
                    block_label=self.block_label,
                )

            self.log.info(
                f"Cluster {BlockLogger.highlight(self.cluster_name)} is currently running.",
                block_label=self.block_label,
            )
            self.log.info("Connecting to this cluster:", block_label=self.block_label)
            url = get_endpoint(f"/projects/{self.project_id}/clusters/{cluster.id}")
            self._log_cluster_configs(
                cluster, cluster.build_id, cluster.compute_template_id, url,
            )

        self.log.close_block(self.block_label)

    def _start_or_create_cluster(
        self,
        project_id: str,
        build_id: str,
        compute_template_id: str,
        cluster_name: Optional[str],
        autosuspend_timeout: Optional[int],
        allow_public_internet_traffic: Optional[bool],
    ) -> str:
        """Create/Start a cluster based on its current state and passed args.

        Args:
            project_id (str): The project to use.
            build_id (str): Build to start cluster with.
            compute_template_id (str): Compute template to start cluster with
            cluster_name (Optional[str]): If specified, the given cluster
                will be created or updated as needed. Otherwise the cluster
                name will be picked automatically.
            autosuspend_timeout (Optional[int]): Autosuspend value of cluster
            allow_public_internet_traffic (bool): Whether to allow public internet
                traffic to serve endpoints
        Returns:
            The name of the cluster to connect to.
        """
        ray_cli = self._ray.util.client.ray
        # If a cluster name is not specified, user wants to start a new cluster.
        did_user_not_specify_cluster_name = not cluster_name
        if did_user_not_specify_cluster_name:
            # Try to generate an auto-incrementing cluster name from the first 100 clusters.
            # If there are more than 40 clusters with, we will generate a random name.
            # If the name is already taken, we will re-try with a different random name.
            used_names: List[str] = []
            has_more = True
            paging_token = None
            while has_more:
                resp = self.anyscale_api_client.search_clusters(
                    clusters_query={
                        "project_id": project_id,
                        "paging": {"count": 50, "paging_token": paging_token},
                    }
                )
                used_names.extend([c.name for c in resp.results])
                paging_token = resp.metadata.next_paging_token
                has_more = paging_token is not None

            self.log.debug("-> Starting a new cluster")
            for i in range(MAX_CLUSTERS):
                name = f"cluster-{i}"
                if name not in used_names:
                    cluster_name = name
                    self.log.debug("Starting cluster %s", cluster_name)
                    break

        self.log.debug(
            f"Updating {cluster_name} to use build id {build_id} and compute template id {compute_template_id}"
        )
        # TODO(ekl): race condition here since "up" breaks the lock.
        if ray_cli.is_connected():
            self._ray.util.disconnect()
        # Update cluster.
        self.log.debug("Starting cluster with sdk and compute config.")

        cluster, start_required = self._create_or_update_session_data(
            cluster_name,
            project_id,
            build_id,
            compute_template_id,
            autosuspend_timeout,
            bool(allow_public_internet_traffic),
            did_user_not_specify_cluster_name,
        )
        return self._start_cluster_if_required(
            cluster,
            start_required,
            project_id,
            build_id,
            compute_template_id,
            allow_public_internet_traffic,
        )

    def _start_cluster_if_required(
        self,
        cluster: Session,
        start_required: bool,
        project_id: str,
        build_id: str,
        compute_template_id: str,
        allow_public_internet_traffic: Optional[bool],
    ) -> str:
        """Start the cluster if required.

        Args:
            cluster (Session): The cluster to start.
            start_required (bool): Whether the cluster.
            project_id (str): The project to use.
            build_id (str): Build to start cluster with.
            compute_template_id (str): Compute template to start cluster with
            allow_public_internet_traffic (bool): Whether to allow public internet
                traffic to serve endpoints
        """
        url = get_endpoint(f"/projects/{project_id}/clusters/{cluster.id}")

        if start_required:
            self.log.debug(
                "Note that restarting the cluster will not change the configurations correctly "
                "in AIOA or GCP."
            )

        self._log_cluster_configs(cluster, build_id, compute_template_id, url)

        if start_required:
            self.anyscale_api_client.start_cluster(
                cluster.id,
                StartClusterOptions(
                    cluster_environment_build_id=build_id,
                    cluster_compute_id=compute_template_id,
                    allow_public_internet_traffic=allow_public_internet_traffic,
                ),
            )

            wait_for_session_start(
                project_id,
                cluster.name,
                self.api_client,
                log=self.log,
                block_label=self.block_label,
            )
            self.log.debug(f"Cluster {cluster.name} finished starting. View at {url}")

        return str(cluster.name)

    def _log_cluster_configs(
        self, cluster: Session, build_id: str, compute_template_id: str, url: str,
    ) -> None:
        """Prints information about the cluster."""

        cluster_env = self.anyscale_api_client.get_build(build_id).result
        # The SDK uses application_template_id but this is really just the cluster_env_name.
        cluster_env_name = (
            cluster_env.application_template_id + ":" + str(cluster_env.revision)
        )

        compute_config: ComputeTemplate = self.anyscale_api_client.get_compute_template(
            compute_template_id
        ).result
        compute_config_name = compute_config.name
        compute_config_config = compute_config.config

        left_pad = " " * 2
        self.log.info(
            f"{left_pad}{'cluster id:': <30}{cluster.id}", block_label=self.block_label
        )
        self.log.info(
            f"{left_pad}{'cluster environment:': <30}{cluster_env_name}",
            block_label=self.block_label,
        )
        self.log.info(
            f"{left_pad}{'cluster environment id:': <30}{build_id}",
            block_label=self.block_label,
        )
        self.log.info(
            f"{left_pad}{'cluster compute:': <30}{compute_config_name}",
            block_label=self.block_label,
        )
        self.log.info(
            f"{left_pad}{'cluster compute id:': <30}{compute_template_id}",
            block_label=self.block_label,
        )

        def get_minutes_output(field: Optional[int]) -> str:
            if field and field > 0:
                return f"{field} minutes"
            return "disabled"

        self.log.info(
            f"{left_pad}{'idle termination:': <30}{get_minutes_output(cluster.idle_timeout)}",
            block_label=self.block_label,
        )
        self.log.info(
            f"{left_pad}{'maximum uptime:': <30}{get_minutes_output(compute_config_config.maximum_uptime_minutes)}",
            block_label=self.block_label,
        )

        self.log.info(f"{left_pad}{'link:': <30}{url}", block_label=self.block_label)

    def _create_or_update_session_data(  # noqa: PLR0913
        self,
        cluster_name: Optional[str],
        project_id: str,
        build_id: str,
        compute_template_id: str,
        idle_timeout: Optional[int],
        allow_public_internet_traffic: bool = False,
        did_user_not_specify_cluster_name: bool = False,
    ) -> Tuple[Session, bool]:
        """
        If cluster name is None, creates a new cluster with a generates a cluster name.

        Creates new cluster with build_id and compute_template_id if cluster
        with `cluster_name` doesn't already exist. Otherwise, update the
        `idle_timeout` of the existing cluster if provided.
        """

        start_required = True

        if not cluster_name:
            # Generate a cluster name
            cluster_name = f"cluster-{generate_slug()}"

        cluster = get_cluster(self.anyscale_api_client, project_id, cluster_name)

        if did_user_not_specify_cluster_name:
            while cluster:
                # User wants to create a new cluster but cluster with name already exists,
                # generate a new name until we have one that's not taken.
                cluster_name = f"cluster-{generate_slug()}"
                cluster = get_cluster(
                    self.anyscale_api_client, project_id, cluster_name
                )

        if not cluster:
            # Create a new cluster if there is no existing cluster with the given cluster_name
            # This will occur if the user did not specify a cluster name, or if a cluster with
            # the user specified cluster_name does not exist.
            self.log.info(
                f"Cluster {BlockLogger.highlight(cluster_name)} does not exist. A new cluster will be created.",
                block_label=self.block_label,
            )
            self.log.info(
                f"Starting cluster {BlockLogger.highlight(cluster_name)}:",
                block_label=self.block_label,
            )
            # User didn't specify idle timeout, use the default autosuspend timeout
            if idle_timeout is None:
                idle_timeout = DEFAULT_AUTOSUSPEND_TIMEOUT
            create_cluster_data = CreateCluster(
                name=cluster_name,
                project_id=project_id,
                cluster_environment_build_id=build_id,
                cluster_compute_id=compute_template_id,
                idle_timeout_minutes=idle_timeout,
                allow_public_internet_traffic=allow_public_internet_traffic,
            )
            self.anyscale_api_client.create_cluster(create_cluster_data)
            cluster = get_cluster(self.anyscale_api_client, project_id, cluster_name)
            assert cluster is not None
        else:
            # Get the existing cluster and update the idle_timeout if required
            if cluster.state == "Running":
                start_required = self._validate_new_cluster_compute_and_env_match_existing_cluster(
                    project_id, cluster, print_warnings=False
                )
                if start_required:
                    self.log.info(
                        f"Cluster {BlockLogger.highlight(cluster_name)} is currently running, "
                        "but the cluster configurations specified in `ray.init()` indicate an "
                        f"update should occur. This will restart cluster {BlockLogger.highlight(cluster_name)}.",
                        block_label=self.block_label,
                    )
                    self.log.info(
                        f"Restarting cluster {BlockLogger.highlight(cluster_name)}:",
                        block_label=self.block_label,
                    )
                else:
                    self.log.info(
                        f"Cluster {BlockLogger.highlight(cluster_name)} is currently running.",
                        block_label=self.block_label,
                    )
                    self.log.info(
                        "Connecting to this cluster:", block_label=self.block_label
                    )
            else:
                self.log.info(
                    f"Cluster {BlockLogger.highlight(cluster_name)} exists but not running. This cluster will be restarted.",
                    block_label=self.block_label,
                )
                self.log.info(
                    f"Restarting cluster {BlockLogger.highlight(cluster_name)}:",
                    block_label=self.block_label,
                )
            if idle_timeout:
                self.anyscale_api_client.update_cluster(
                    cluster.id, UpdateCluster(idle_timeout_minutes=idle_timeout)
                )

        return cluster, start_required

    def _get_prepare_cluster_action(
        self, project_id: str, cluster_name: Optional[str], needs_update: bool
    ) -> PrepareClusterAction:
        """
        Returns the next cluster action based on the cluster state.
        """
        if not cluster_name:
            return PrepareClusterAction.CREATE
        cluster = get_cluster(self.anyscale_api_client, project_id, cluster_name)
        if not cluster:
            # Unconditionally create the cluster if it isn't up.
            return PrepareClusterAction.CREATE
        elif cluster.state == SessionState.STARTINGUP:
            return PrepareClusterAction.WAIT
        elif cluster.state != SessionState.RUNNING:
            # Unconditionally start the cluster if it isn't up.
            return PrepareClusterAction.START
        elif needs_update:
            return PrepareClusterAction.UPDATE
        else:
            return PrepareClusterAction.NO_OP

    def _get_default_cluster_env_build(self) -> Build:
        """
        Get the default cluster env build based on the local python and ray versions.
        """
        major, minor = sys.version_info[:2]
        MIN_PY_VER = (3, 8)
        MAX_PY_VER = (3, 12)

        if not (MIN_PY_VER <= (major, minor) <= MAX_PY_VER):
            raise ValueError(
                f"No default container image for python version {major}.{minor}."
                f"Please use a Python version between {MIN_PY_VER[0]}.{MIN_PY_VER[1]} "
                f"and {MAX_PY_VER[0]}.{MAX_PY_VER[1]}."
            )

        ray_version = self._ray.__version__
        if version.parse(ray_version) < version.parse(MINIMUM_RAY_VERSION):
            raise ValueError(
                f"No default container image for Ray version {ray_version}. Please upgrade "
                f"to a version >= {MINIMUM_RAY_VERSION}."
            )
        if "dev0" in ray_version:
            raise ValueError(
                f"Your locally installed Ray version is {ray_version}. "
                "There is no default container image for nightly versions of Ray."
            )
        try:
            build = self.api_client.get_default_cluster_env_build_api_v2_builds_default_py_version_ray_version_get(
                f"py{major}{minor}", ray_version
            ).result
            return build
        except Exception:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to get default container image for Ray: {ray_version} on Python: py{major}{minor}"
            )

    def _get_cluster_build(
        self, cluster_env_name: Optional[str], cluster_env_revision: Optional[int],
    ) -> Build:
        """Returns the build of the cluster to be created.
        By default we return the default cluster env, unless the user overrides.
        """
        if cluster_env_name:
            build = self._get_cluster_env_build(cluster_env_name, cluster_env_revision)
        else:
            try:
                build = self._get_default_cluster_env_build()
            except ValueError as e:
                # A default cluster env was not found
                raise ValueError(
                    "Because you did not specify a cluster environment, we "
                    "attempted to use a default cluster environment but could "
                    "not find one that matches your requirements. "
                    "To specify a cluster environment, use the cluster_env "
                    "argument to ray.init() or the ANYSCALE_CLUSTER_ENV environment variable. "
                    "If one does not already exist, you can create a new cluster environment "
                    f"with your dependencies by following these instructions: {DOCS_CLUSTER}."
                ) from e

        return build

    def _get_cluster_env_build(
        self, cluster_env_name: str, clust_env_revision: Optional[int]
    ) -> Build:
        """
        Get the build for the specified cluster environment. If the cluster env revsion is
        not provided, use the latest revision.
        """
        app_template_id = None
        cluster_environments = self.anyscale_api_client.search_cluster_environments(
            {"name": {"equals": cluster_env_name},}
        ).results
        for cluster_env in cluster_environments:
            if cluster_env.name == cluster_env_name:
                app_template_id = cluster_env.id
        if not app_template_id:
            raise RuntimeError(
                f"Cluster Environment '{cluster_env_name}' not found. See environments at "
                f'{get_endpoint("/configurations/?tab=cluster-env")}.'
            )
        builds = list_entities(
            self.anyscale_api_client.list_cluster_environment_builds, app_template_id
        )

        build_to_use = None
        if clust_env_revision:
            for build in builds:
                if build.revision == clust_env_revision:
                    build_to_use = build

            if not build_to_use:
                raise RuntimeError(
                    "Revision {} of cluster environment '{}' not found.".format(
                        clust_env_revision, cluster_env_name
                    )
                )
        else:
            latest_build_revision = -1
            for build in builds:
                if build.revision > latest_build_revision:
                    latest_build_revision = build.revision
                    build_to_use = build
            self.log.debug(
                "Using latest revision {} of {}".format(
                    latest_build_revision, cluster_env_name
                )
            )
        assert build_to_use  # for mypy
        return build_to_use

    def _build_cluster_env_if_needed(
        self,
        project_id: str,
        build_pr: Optional[int],
        build_commit: Optional[str],
        cluster_env_dict: Optional[CLUSTER_ENV_DICT_TYPE],
        cluster_env_name: Optional[str],
        force_rebuild: bool,
    ):
        """Builds a new cluster env on the fly if a cluster env dict is provided by the user
        or if the user wants to build ray from source."""
        if build_pr or build_commit:
            cluster_env_name = self._build_app_config_from_source(
                project_id, build_pr, build_commit, force_rebuild
            )

        elif cluster_env_dict:
            # Replacing ":" with "-" because the cluster env name cannot include ":"
            cluster_env_name = (
                cluster_env_name.replace(":", "-")
                if cluster_env_name
                else "anonymous_cluster_env-{}".format(
                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                )
            )
            self.log.info(
                f"Building a new docker image for cluster environment {cluster_env_name} with "
                "the provided cluster environment dictionary.",
                block_label=self.block_label,
            )
            self.anyscale_api_client.create_cluster_environment(
                {
                    "name": cluster_env_name,
                    "project_id": project_id,
                    "config_json": cluster_env_dict,
                }
            )
        return cluster_env_name

    def _build_app_config_from_source(
        self,
        project_id: str,
        build_pr: Optional[int],
        build_commit: Optional[str],
        force_rebuild: bool,
    ) -> str:
        """
        Build a cluster environment from a build PR and/or commit.
        """
        config_name = "ray-build"
        if build_pr:
            config_name += f"-{build_pr}"
        if build_commit:
            config_name += f"-{build_commit}"
        # Force creation of a unique app config.
        if force_rebuild:
            config_name += f"-{int(time.time())}"
        cluster_environments = list_entities(
            self.anyscale_api_client.search_cluster_environments,
            filters={"name": {"equals": config_name}},
        )
        found = any(a.name == config_name for a in cluster_environments)
        if not found:
            build_steps = BUILD_STEPS.copy()
            # Add a unique command to bust the Makisu cache if needed.
            # Otherwise we could end up caching a previous fetch.
            build_steps.append(f"echo UNIQUE_ID={config_name}")
            if build_pr:
                build_steps.append(
                    "cd ray && git fetch origin pull/{}/head:target && "
                    "git checkout target".format(build_pr)
                )
            if build_commit:
                build_steps.append(f"cd ray && git checkout {build_commit}")
            build_steps.append(
                'cd ray/python && sudo env "PATH=$PATH" python setup.py develop'
            )
            self.anyscale_api_client.create_cluster_environment(
                {
                    "name": config_name,
                    "project_id": project_id,
                    "config_json": {
                        "base_image": _get_base_image("ray", "nightly", "cpu"),
                        "debian_packages": ["curl", "unzip", "zip", "gnupg"],
                        "post_build_cmds": build_steps,
                    },
                }
            )
        return config_name

    def _wait_for_app_build(self, project_id: str, build_id: str) -> Build:
        """
        Block until cluster env finishes building.
        """
        has_logged = False
        while True:
            build = self.anyscale_api_client.get_build(build_id).result
            if build.status in ["pending", "in_progress"]:
                if not has_logged:
                    url = get_endpoint(
                        f"projects/{project_id}/app-config-details/{build_id}"
                    )
                    self.log.info(
                        f"Waiting for cluster env to be built (see {url} for progress)...",
                        block_label=self.block_label,
                    )
                    has_logged = True
                time.sleep(10.0)
            elif build.status in ["failed", "pending_cancellation", "canceled"]:
                raise RuntimeError(
                    "Cluster env status is '{}', please select another revision".format(
                        build.status
                    )
                )
            else:
                assert build.status == "succeeded"
                return build

    def _get_cluster_compute_id(
        self,
        project_id: str,
        cluster_compute_name: Optional[str],
        cluster_compute_dict: Optional[CLUSTER_COMPUTE_DICT_TYPE],
        cloud_name: Optional[str],
    ) -> str:
        """Returns the compute template ids of the cluster to be created.
        By default we return the default cluster compute, unless the user overrides.
        """
        # get cluster compute template
        if cluster_compute_name:
            compute_template_id = get_cluster_compute_from_name(
                cluster_compute_name, self.api_client
            ).id
        else:
            # If the user specifies _cluster_compute_dict use it, otherwise
            # use the default cluster compute template.
            if cluster_compute_dict:
                cluster_compute_class = ComputeTemplateConfig(**cluster_compute_dict)
                config_object = cluster_compute_class
            else:
                cloud_id = get_selected_cloud_id_or_default(
                    api_client=self.api_client,
                    anyscale_api_client=self.anyscale_api_client,
                    cloud_name=cloud_name,
                )
                config_object = self.anyscale_api_client.get_default_compute_config(
                    cloud_id
                ).result
            compute_template_id = self._register_compute_template(
                project_id, config_object
            )
        return compute_template_id

    def _register_compute_template(
        self, project_id: str, config_object: ComputeTemplateConfig  # noqa: ARG002
    ) -> str:
        """
        Register compute template with a default name and return the compute template id.
        """
        created_template = self.api_client.create_compute_template_api_v2_compute_templates_post(
            create_compute_template=CreateComputeTemplate(
                name=gen_valid_name("autogenerated-config"),
                # project ID is deprecated in compute config
                project_id=None,
                config=config_object,
                anonymous=True,
            )
        ).result
        compute_template_id = str(created_template.id)
        return compute_template_id

    def _is_equal_cluster_compute(
        self, cluster_compute_1_id: str, cluster_compute_2_id: str,
    ) -> bool:
        """
        Compares config fields of two ComputeTemplate objects.
        """
        try:
            cluster_compute_1 = self.anyscale_api_client.get_compute_template(
                cluster_compute_1_id
            ).result.config
            cluster_compute_2 = self.anyscale_api_client.get_compute_template(
                cluster_compute_2_id
            ).result.config
            return bool(cluster_compute_1 == cluster_compute_2)
        except Exception as e:  # noqa: BLE001
            self.log.debug(f"Error comparing cluster compute: {e}")
            return False

    def _validate_new_cluster_compute_and_env_match_existing_cluster(
        self, project_id: str, running_cluster: Session, print_warnings: bool = True
    ) -> bool:
        """
        Compare the build id, compute template id, and allow_public_internet_traffic values
        for a existing running cluster and the values passed in. Print warnings if incompatible
        values if specified, and return if the specified arguments require a cluster restart.
        """
        update_required = False
        cluster_name = running_cluster.name
        if self.cluster_env_name:
            overriding_build = self._get_cluster_build(
                self.cluster_env_name, self.cluster_env_revision
            )
            if overriding_build.id != running_cluster.build_id:
                current_cluster_build = self.anyscale_api_client.get_build(
                    running_cluster.build_id
                ).result
                current_cluster_env = self.anyscale_api_client.get_cluster_environment(
                    current_cluster_build.application_template_id
                ).result
                if print_warnings:
                    self.log.warning(
                        f"The cluster is currently using {current_cluster_env.name}:{current_cluster_build.revision} as the cluster env, "
                        f"yet {self.cluster_env_name}:{self.cluster_env_revision} was provided. The cluster will not be updated and will "
                        f"still use {current_cluster_env.name}:{current_cluster_build.revision} as the cluster env. To update the cluster, "
                        "please specify `update=True` in the anyscale address, e.g.: "
                        f'`ray.init("anyscale://{cluster_name}", cluster_env="{self.cluster_env_name}:{self.cluster_env_revision}", update=True)`'
                    )
                update_required = True
        if self.cluster_compute_name:
            # Because self.cluster_compute_name exists, this will get the id of an existing
            # cluster compute and not register a new one.
            overriding_compute = self._get_cluster_compute_id(
                project_id,
                self.cluster_compute_name,
                self.cluster_compute_dict,
                self.cloud_name,
            )
            if not self._is_equal_cluster_compute(
                overriding_compute, running_cluster.compute_template_id
            ):
                cluster_compute = self.anyscale_api_client.get_compute_template(
                    running_cluster.compute_template_id
                ).result
                if print_warnings:
                    self.log.warning(
                        f"The cluster is currently using {cluster_compute.name} as the compute config, "
                        f"yet {self.cluster_compute_name} was provided. The cluster will not be updated and will "
                        f"still use {cluster_compute.name} as the compute config. To update the cluster, please specify "
                        "`update=True` in the anyscale address, e.g.: "
                        f'`ray.init("anyscale://{cluster_name}", cluster_compute="{self.cluster_compute_name}", update=True")`'
                    )
                update_required = True
        if self.cluster_env_dict or self.cluster_compute_dict:
            self.log.warning(
                f"The cluster {cluster_name} is already active. To update the cluster please "
                "specify `update=True` in the anyscale address."
            )
            update_required = True

        if (
            self.allow_public_internet_traffic is not None
            and self.allow_public_internet_traffic
            != running_cluster.allow_public_internet_traffic
        ):
            if print_warnings:
                self.log.warning(
                    f"The cluster currently {'allows' if running_cluster.allow_public_internet_traffic else 'does not allow'} "
                    f"public internet traffic to the Serve endpoints, yet `allow_public_internet_traffic={self.allow_public_internet_traffic}` "
                    f"was specified. The cluster will not be updated and will continue {'allowing' if running_cluster.allow_public_internet_traffic else 'not allowing'} "
                    "public internet traffic to the Serve endpoints. To update the cluster, please specify `update=True` in the anyscale "
                    "address, e.g.: "
                    f'`ray.init("anyscale://{cluster_name}", update=True, allow_public_internet_traffic={self.allow_public_internet_traffic})`'
                )
            update_required = True

        return update_required


def create_prepare_cluster_block(  # noqa: PLR0913
    project_id: str,
    cluster_name: Optional[str],
    autosuspend_timeout: Optional[int],
    allow_public_internet_traffic: Optional[bool],
    needs_update: bool,
    cluster_compute_name: Optional[str],
    cluster_compute_dict: Optional[CLUSTER_COMPUTE_DICT_TYPE],
    cloud_name: Optional[str],
    build_pr: Optional[int],
    force_rebuild: bool,
    build_commit: Optional[str],
    cluster_env_name: Optional[str],
    cluster_env_dict: Optional[CLUSTER_ENV_DICT_TYPE],
    cluster_env_revision: Optional[int],
    ray: Any,
    log_output: bool = True,
):
    """
    Function to get PrepareClusterBlock object. The PrepareClusterBlock object
    is not a global variable an will be reinstantiated on each call to
    get_prepare_cluster_block.
    """
    prepare_cluster_block = PrepareClusterBlock(
        project_id=project_id,
        cluster_name=cluster_name,
        autosuspend_timeout=autosuspend_timeout,
        allow_public_internet_traffic=allow_public_internet_traffic,
        needs_update=needs_update,
        cluster_compute_name=cluster_compute_name,
        cluster_compute_dict=cluster_compute_dict,
        cloud_name=cloud_name,
        build_pr=build_pr,
        force_rebuild=force_rebuild,
        build_commit=build_commit,
        cluster_env_name=cluster_env_name,
        cluster_env_dict=cluster_env_dict,
        cluster_env_revision=cluster_env_revision,
        ray=ray,
        log_output=log_output,
    )
    prepare_cluster_block.prepare()
    return prepare_cluster_block
