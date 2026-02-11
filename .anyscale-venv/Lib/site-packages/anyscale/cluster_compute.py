import re
from typing import Optional, Tuple, Union

from anyscale.authenticate import get_auth_api_client
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.api.default_api import DefaultApi
from anyscale.client.openapi_client.models import (
    ComputeTemplateQuery,
    CreateComputeTemplate,
)
from anyscale.cloud_utils import get_cloud_id_and_name, get_last_used_cloud
from anyscale.sdk.anyscale_client import (
    ArchiveStatus,
    ComputeTemplateConfig,
)
from anyscale.sdk.anyscale_client.api.default_api import DefaultApi as SDKDefaultApi
from anyscale.sdk.anyscale_client.models.cluster_compute_config import (
    ClusterComputeConfig,
)
from anyscale.sdk.anyscale_client.models.compute_template import ComputeTemplate
from anyscale.utils.cloud_utils import get_organization_default_cloud


log = BlockLogger()


def get_default_cluster_compute(
    cloud_name: Optional[str],
    project_id: Optional[str],
    api_client: Optional[DefaultApi] = None,
    anyscale_api_client: Optional[Union[DefaultApi, SDKDefaultApi]] = None,
) -> ComputeTemplate:
    if api_client is None:
        api_client = get_auth_api_client().api_client
    if anyscale_api_client is None:
        anyscale_api_client = get_auth_api_client().anyscale_api_client

    if cloud_name is None:
        default_cloud_name = get_organization_default_cloud(api_client)
        cloud_name = default_cloud_name or get_last_used_cloud(
            project_id, anyscale_api_client
        )

    cloud_id, _ = get_cloud_id_and_name(api_client, cloud_name=cloud_name)
    config_object = anyscale_api_client.get_default_compute_config(cloud_id).result  # type: ignore
    compute_template = register_compute_template(config_object, api_client=api_client)
    return compute_template


def parse_cluster_compute_name_version(
    cluster_compute_name_version: str,
) -> Tuple[str, Optional[int]]:

    # Regex pattern to validate the cluster compute name version string:
    # - the name could not contain colon, leading or trailing spaces
    # - the version should be a positive integer
    # - the version is optional
    # - the name and version should be separated by a colon if version is provided
    COMPUTE_CONFIG_NAME_VERSION_VALIDATION_REGEX_PATTERN = (
        r"^[^\s:]+(\s+[^\s:]+)*(:[1-9]+[0-9]*)?$"
    )

    if not re.match(
        COMPUTE_CONFIG_NAME_VERSION_VALIDATION_REGEX_PATTERN,
        cluster_compute_name_version,
    ):
        raise ValueError(
            f"Invalid compute config name and version '{cluster_compute_name_version}'. Please provide name and version in format `my-compute-config:3` or provide compute configuration name `my-compute-config`."
        )

    cluster_compute_name = cluster_compute_name_version
    version = None
    if cluster_compute_name_version.count(":") == 1:
        cluster_compute_name, parsed_version = cluster_compute_name_version.split(":")
        version = int(parsed_version)

    return cluster_compute_name, version


def get_cluster_compute_from_name(
    cluster_compute_name: str,
    api_client: Optional[DefaultApi] = None,
    include_archived: Optional[bool] = False,
    *,
    cloud_name: Optional[str] = None,
) -> ComputeTemplate:
    if api_client is None:
        api_client = get_auth_api_client().api_client

    version = None

    (cluster_compute_name, version,) = parse_cluster_compute_name_version(
        cluster_compute_name
    )

    cloud_id = None
    if cloud_name:
        # Resolve cloud ID when a cloud name is provided to disambiguate configs with the same name.
        cloud_id, _ = get_cloud_id_and_name(api_client, cloud_name=cloud_name)

    cluster_computes = api_client.search_compute_templates_api_v2_compute_templates_search_post(
        ComputeTemplateQuery(
            orgwide=True,
            name={"equals": cluster_compute_name},
            include_anonymous=True,
            archive_status=ArchiveStatus.ALL,
            version=version,
            cloud_id=cloud_id,
        )
    ).results

    if len(cluster_computes) == 0:
        raise ValueError(
            f"The compute config {cluster_compute_name} does not exist, or you don't have sufficient permissions."
        )
    cluster_compute = cluster_computes[0]
    if not include_archived and cluster_compute.archived_at is not None:
        raise ValueError(f"The compute config {cluster_compute_name} is archived.")

    return cluster_compute


def register_compute_template(
    config_object: ComputeTemplateConfig, api_client: Optional[DefaultApi] = None,
) -> ComputeTemplate:
    """
    Register compute template with a default name and return the compute template id."""
    if api_client is None:
        api_client = get_auth_api_client().api_client
    created_template = api_client.create_compute_template_api_v2_compute_templates_post(
        create_compute_template=CreateComputeTemplate(
            config=config_object, anonymous=True,
        )
    ).result
    return created_template


def get_selected_cloud_id_or_default(
    api_client: Optional[DefaultApi] = None,
    anyscale_api_client: Optional[Union[DefaultApi, SDKDefaultApi]] = None,
    cluster_compute_id: Optional[str] = None,
    cluster_compute_config: Optional[ClusterComputeConfig] = None,
    cloud_id: Optional[str] = None,
    cloud_name: Optional[str] = None,
):
    """
    Gets cloud_id that is selected for the current command from the
    arguments `cloud_id`, `cloud_name`, `cluster_compute_id`, or `cluster_compute_config`.
    If the cloud_id is not selected through any of these arguments, get the default cloud
    from the default cluster compute.
    """
    api_client = api_client or get_auth_api_client(log_output=False).api_client
    anyscale_api_client = (
        anyscale_api_client or get_auth_api_client(log_output=False).anyscale_api_client
    )
    if cloud_id or cloud_name:
        parent_cloud_id, _ = get_cloud_id_and_name(
            api_client=api_client, cloud_id=cloud_id, cloud_name=cloud_name,
        )
    elif cluster_compute_id:
        parent_cloud_id = anyscale_api_client.get_cluster_compute(  # type: ignore
            cluster_compute_id
        ).result.config.cloud_id
    elif cluster_compute_config:
        parent_cloud_id = cluster_compute_config.cloud_id
    else:
        parent_cloud_id = (
            anyscale_api_client.get_default_cluster_compute().result.config.cloud_id
        )
    return parent_cloud_id
