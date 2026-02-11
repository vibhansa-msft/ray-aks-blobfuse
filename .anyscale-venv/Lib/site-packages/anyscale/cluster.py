import dataclasses
from typing import Any, Dict, Optional

import click

from anyscale.authenticate import get_auth_api_client
from anyscale.cli_logger import BlockLogger
from anyscale.connect import ClientBuilder
from anyscale.connect_utils.prepare_cluster import create_prepare_cluster_block
from anyscale.connect_utils.project import create_project_block
from anyscale.util import PROJECT_NAME_ENV_VAR


log = BlockLogger()  # Anyscale CLI Logger


@dataclasses.dataclass
class ClusterInfo:
    """
    Synchronize with ray.dashboard.modules.job.sdk.ClusterInfo
    """

    address: str
    cookies: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, Any]] = None


def get_job_submission_client_cluster_info(
    address: str,
    create_cluster_if_needed: Optional[bool] = None,
    cookies: Optional[Dict[str, Any]] = None,  # noqa: ARG001
    metadata: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> ClusterInfo:
    """
    Get address and cookies used for ray JobSubmissionClient.

    Args:
        address (str): Address of same form as ray.init address without
            anyscale:// prefix.
        create_cluster_if_needed (bool): Indicates whether the cluster
            of the address returned needs to be running. Raise an error
            if cluster is not running and this is False. Create a cluster
            if cluster is not running and this is True.

    Returns:
        ClusterInfo object consisting of address, cookies, and metadata for
            JobSubmissionClient to use.
    """

    if create_cluster_if_needed:
        # Use ClientBuilder to start cluster if needed because cluster needs to be active for
        # the calling command.
        api_client = get_auth_api_client().api_client
        client_builder = ClientBuilder(
            address=address, log=BlockLogger(log_output=False)
        )
        project_block = create_project_block(
            client_builder._project_dir,  # noqa: SLF001
            client_builder._project_name,  # noqa: SLF001
            cloud_name=client_builder._cloud_name,  # noqa: SLF001
            cluster_compute_name=client_builder._cluster_compute_name,  # noqa: SLF001
            cluster_compute_dict=client_builder._cluster_compute_dict,  # noqa: SLF001
        )
        project_id = project_block.project_id
        project_dir = project_block.project_dir
        if project_dir and not client_builder._project_name:  # noqa: SLF001
            # Warning when project dir provided or found in current directory and project name not provided
            # as input.
            # TODO(nikita): Remove after .anyscale.yaml is no longer supported
            log.warning(
                f"Project directory {project_dir} was detected. Using a project directory "
                "to set the project for the cluster to use with `ray job submit` has been deprecated, and this "
                "functionality will be removed in April 2022. To start an Anyscale cluster "
                "in a particular project, please instead specify the project name as "
                f'RAY_ADDRESS="anyscale://{project_block.project_name}/{client_builder._cluster_name if client_builder._cluster_name else ""}". '  # noqa: SLF001
                "The project name can also be specified by setting the environment variable "
                f'{PROJECT_NAME_ENV_VAR}="{project_block.project_name}". '
                "Otherwise the cluster will not be grouped to a particular project.\n"
            )
        prepare_cluster_block = create_prepare_cluster_block(
            project_id=project_id,
            cluster_name=client_builder._cluster_name,  # noqa: SLF001
            autosuspend_timeout=client_builder._autosuspend_timeout,  # noqa: SLF001
            allow_public_internet_traffic=client_builder._allow_public_internet_traffic,  # noqa: SLF001
            needs_update=client_builder._needs_update,  # noqa: SLF001
            cluster_compute_name=client_builder._cluster_compute_name,  # noqa: SLF001
            cluster_compute_dict=client_builder._cluster_compute_dict,  # noqa: SLF001
            cloud_name=client_builder._cloud_name,  # noqa: SLF001
            build_pr=client_builder._build_pr,  # noqa: SLF001
            force_rebuild=client_builder._force_rebuild,  # noqa: SLF001
            build_commit=client_builder._build_commit,  # noqa: SLF001
            cluster_env_name=client_builder._cluster_env_name,  # noqa: SLF001
            cluster_env_dict=client_builder._cluster_env_dict,  # noqa: SLF001
            cluster_env_revision=client_builder._cluster_env_revision,  # noqa: SLF001
            ray=client_builder._ray,  # noqa: SLF001
        )
        cluster_name = prepare_cluster_block.cluster_name
    else:
        # Calling ClientBuilder to parse address.
        api_client = get_auth_api_client(log_output=False).api_client
        client_builder = ClientBuilder(
            address=address, log=BlockLogger(log_output=False)
        )
        cluster_name = client_builder._cluster_name  # noqa: SLF001
        project_block = create_project_block(
            client_builder._project_dir,  # noqa: SLF001
            client_builder._project_name,  # noqa: SLF001
            log_output=False,
        )
        project_id = project_block.project_id

    user = api_client.get_user_info_api_v2_userinfo_get().result
    metadata = {"creator_id": user.id}

    cluster_list = api_client.list_sessions_api_v2_sessions_get(
        project_id=project_id, active_only=True, name=cluster_name
    ).results

    if len(cluster_list) > 0:
        cluster = cluster_list[0]
        access_token = api_client.get_cluster_access_token_api_v2_authentication_cluster_id_cluster_access_token_get(
            cluster_id=cluster.id
        )
        if cluster.host_name and access_token:
            return ClusterInfo(
                address=cluster.host_name,
                cookies={"anyscale-token": access_token},
                metadata=metadata,
            )
        else:
            raise click.ClickException(
                f"Host name or access token not found for cluster {cluster_name}. Please check the cluster is currently running."
            )
    else:
        raise click.ClickException(
            f"No running cluster found with name {cluster_name} in project {project_id}. Please start "
            "the cluster, or change the project context if the wrong one is being used."
        )
