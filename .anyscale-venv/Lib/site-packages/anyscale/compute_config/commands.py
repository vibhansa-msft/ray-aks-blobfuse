from typing import Optional

from anyscale._private.sdk import sdk_command
from anyscale.compute_config._private.compute_config_sdk import PrivateComputeConfigSDK
from anyscale.compute_config.models import (
    ComputeConfigListResult,
    ComputeConfigType,
    ComputeConfigVersion,
)


_COMPUTE_CONFIG_SDK_SINGLETON_KEY = "compute_config_sdk"

_CREATE_EXAMPLE = """
import anyscale
from anyscale.compute_config.models import ComputeConfig, HeadNodeConfig, MarketType, WorkerNodeGroupConfig

single_deployment_compute_config = ComputeConfig(
    head_node=HeadNodeConfig(
        instance_type="m5.8xlarge",
    ),
    worker_nodes=[
        WorkerNodeGroupConfig(
            instance_type="m5.8xlarge",
            min_nodes=5,
            max_nodes=5,
        ),
        WorkerNodeGroupConfig(
            instance_type="m5.4xlarge",
            min_nodes=1,
            max_nodes=10,
            market_type=MarketType.SPOT,
        ),
    ],
)
full_name: str = anyscale.compute_config.create(single_deployment_compute_config, name="my-single-deployment-compute-config")

multi_deployment_compute_config = MultiResourceComputeConfig(
    configs=[
        ComputeConfig(
            cloud_resource="vm-aws-us-west-1",
            head_node=HeadNodeConfig(
                instance_type="m5.2xlarge",
            ),
            worker_nodes=[
                WorkerNodeGroupConfig(
                    instance_type="m5.4xlarge",
                    min_nodes=1,
                    max_nodes=10,
                ),
            ],
        ),
        ComputeConfig(
            cloud_resource="vm-aws-us-west-2",
            head_node=HeadNodeConfig(
                instance_type="m5.2xlarge",
            ),
            worker_nodes=[
                WorkerNodeGroupConfig(
                    instance_type="m5.4xlarge",
                    min_nodes=1,
                    max_nodes=10,
                ),
            ],
        )
    ]
)
full_name: str = anyscale.compute_config.create(multi_deployment_compute_config, name="my-multi-deployment-compute-config")
"""

_CREATE_ARG_DOCSTRINGS = {
    "config": "The config options defining the compute config.",
    "name": "The name of the compute config. This should *not* include a version tag. If a name is not provided, one will be automatically generated.",
}


@sdk_command(
    _COMPUTE_CONFIG_SDK_SINGLETON_KEY,
    PrivateComputeConfigSDK,
    doc_py_example=_CREATE_EXAMPLE,
    arg_docstrings=_CREATE_ARG_DOCSTRINGS,
)
def create(
    config: ComputeConfigType,
    *,
    name: Optional[str],
    _private_sdk: Optional[PrivateComputeConfigSDK] = None,
) -> str:
    """Create a new version of a compute config.

    Returns the full name of the registered compute config, including the version.
    """
    full_name, _ = _private_sdk.create_compute_config(config, name=name)  # type: ignore
    return full_name


_GET_EXAMPLE = """
import anyscale
from anyscale.compute_config.models import ComputeConfig

compute_config: ComputeConfig = anyscale.compute_config.get("my-compute-config")
"""
_GET_ARG_DOCSTRINGS = {
    "name": "The name of the compute config. This can inclue an optional version tag, i.e., 'name:version'. If no version tag is provided, the latest version will be returned.",
    "include_archived": "Whether to consider archived compute configs (defaults to False).",
    "cloud": "Cloud name to filter by when resolving the compute config by name. Useful when multiple compute configs with the same name exist across different clouds.",
}


@sdk_command(
    _COMPUTE_CONFIG_SDK_SINGLETON_KEY,
    PrivateComputeConfigSDK,
    doc_py_example=_GET_EXAMPLE,
    arg_docstrings=_GET_ARG_DOCSTRINGS,
)
def get(
    name: Optional[str],
    *,
    include_archived: bool = False,
    cloud: Optional[str] = None,
    _id: Optional[str] = None,
    _private_sdk: Optional[PrivateComputeConfigSDK] = None,
) -> ComputeConfigVersion:
    """Get the compute config with the specified name.

    The name can contain an optional version tag, i.e., 'name:version'.
    If no version is provided, the latest one will be returned.

    Args:
        name: Name of the compute config (can include version tag)
        include_archived: Whether to include archived configs
        cloud: Cloud name to filter by when resolving config name
        _id: Internal parameter for fetching by ID
        _private_sdk: Internal SDK instance
    """
    # NOTE(edoakes): I want to avoid exposing fetching by ID in the public API,
    # but it's needed for parity with the existing CLI. Therefore I am adding it
    # as a hidden private API that can be used like: (`name="", _id=id`).
    return _private_sdk.get_compute_config(  # type: ignore
        name=name or None, id=_id, cloud=cloud, include_archived=include_archived
    )


_ARCHIVE_EXAMPLE = """
import anyscale

anyscale.compute_config.archive(name="my-compute-config")
"""

_ARCHIVE_ARG_DOCSTRINGS = {"name": "Name of the compute config."}


@sdk_command(
    _COMPUTE_CONFIG_SDK_SINGLETON_KEY,
    PrivateComputeConfigSDK,
    doc_py_example=_ARCHIVE_EXAMPLE,
    arg_docstrings=_ARCHIVE_ARG_DOCSTRINGS,
)
def archive(
    name: Optional[str],
    *,
    _id: Optional[str] = None,
    _private_sdk: Optional[PrivateComputeConfigSDK] = None,
):
    """Archive a compute config and all of its versions.

    The name can contain an optional version, e.g., 'name:version'.
    If no version is provided, the latest one will be archived.

    Once a compute config is archived, its name will no longer be usable in the organization.
    """
    # NOTE(edoakes): I want to avoid exposing fetching by ID in the public API,
    # but it's needed for parity with the existing CLI. Therefore I am adding it
    # as a hidden private API that can be used like: (`name="", _id=id`).
    return _private_sdk.archive_compute_config(name=name or None, id=_id)  # type: ignore


_GET_DEFAULT_EXAMPLE = """
import anyscale
from anyscale.compute_config.models import ComputeConfigVersion

# Get the default compute config for the default cloud
default_config: ComputeConfigVersion = anyscale.compute_config.get_default()

# Get the default compute config for a specific cloud
default_config: ComputeConfigVersion = anyscale.compute_config.get_default(cloud="my-cloud")

# Get the default compute config for a specific cloud resource
default_config: ComputeConfigVersion = anyscale.compute_config.get_default(
    cloud="my-cloud",
    cloud_resource="my-cloud-resource"
)
"""

_GET_DEFAULT_ARG_DOCSTRINGS = {
    "cloud": "Name of the cloud. If not provided, uses the organization's default cloud.",
    "cloud_resource": "Name of the cloud resource. If not provided, uses the default cloud resource for the cloud.",
}


@sdk_command(
    _COMPUTE_CONFIG_SDK_SINGLETON_KEY,
    PrivateComputeConfigSDK,
    doc_py_example=_GET_DEFAULT_EXAMPLE,
    arg_docstrings=_GET_DEFAULT_ARG_DOCSTRINGS,
)
def get_default(
    *,
    cloud: Optional[str] = None,
    cloud_resource: Optional[str] = None,
    _private_sdk: Optional[PrivateComputeConfigSDK] = None,
) -> ComputeConfigVersion:
    """Get the default compute config for the specified cloud.

    Returns the default compute configuration that will be used when no compute config
    is explicitly specified for a workload.

    Args:
        cloud: Name of the cloud. If not provided, uses the organization's default cloud.
        cloud_resource: Name of the cloud resource. If not provided, uses the default
            cloud resource for the cloud.
        _private_sdk: Internal SDK instance.

    Returns:
        ComputeConfigVersion containing the default compute config.
    """
    return _private_sdk.get_default_compute_config(  # type: ignore
        cloud=cloud, cloud_resource=cloud_resource
    )


_LIST_EXAMPLE = """
import anyscale

# List all compute configs created by the current user
configs = anyscale.compute_config.list()

# List with filtering and sorting
configs = anyscale.compute_config.list(
    cloud_name="aws-prod",
    sort_by="created_at",
    sort_order="desc",
    max_items=10
)

# Access results and pagination token
for config in configs.results:
    print(f"{config.name} (version {config.version})")

# Fetch next page if available
if configs.next_token:
    next_page = anyscale.compute_config.list(
        cloud_name="aws-prod",
        sort_by="created_at",
        sort_order="desc",
        max_items=10,
        next_token=configs.next_token
    )
"""

_LIST_ARG_DOCSTRINGS = {
    "name": "Filter by compute config name.",
    "cloud_id": "Filter by cloud ID.",
    "cloud_name": "Filter by cloud name.",
    "sort_by": "Field to sort by. Options: 'name', 'created_at', 'last_modified_at'. Default: 'last_modified_at'",
    "sort_order": "Sort order. Options: 'asc' (ascending) or 'desc' (descending). Default: 'asc'",
    "max_items": "Maximum number of items to return per page. Default: 20",
    "next_token": "Pagination token for fetching the next page of results.",
    "include_shared": "Include compute configs shared with the user (not just those created by the user). Default: False",
}


@sdk_command(
    _COMPUTE_CONFIG_SDK_SINGLETON_KEY,
    PrivateComputeConfigSDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_ARG_DOCSTRINGS,
)
def list(  # noqa: A001, PLR0913
    *,
    name: Optional[str] = None,
    cloud_id: Optional[str] = None,
    cloud_name: Optional[str] = None,
    sort_by: str = "last_modified_at",
    sort_order: str = "asc",
    max_items: int = 20,
    next_token: Optional[str] = None,
    include_shared: bool = False,
    _id: Optional[str] = None,
    _private_sdk: Optional[PrivateComputeConfigSDK] = None,
) -> ComputeConfigListResult:
    """List compute configurations with filtering, sorting, and pagination.

    Returns a ComputeConfigListResult object containing:
    - results: List of compute config objects
    - next_token: Token for fetching the next page (None if no more results)
    - count: Number of results in this page

    Args:
        name: Filter by compute config name
        cloud_id: Filter by cloud ID
        cloud_name: Filter by cloud name
        sort_by: Field to sort by ('name', 'created_at', 'last_modified_at')
        sort_order: Sort order ('asc' or 'desc')
        max_items: Maximum number of items to return per page
        next_token: Pagination token for next page
        include_shared: Include configs shared with the user
        _id: Internal parameter for fetching by ID
        _private_sdk: Internal SDK instance

    Returns:
        ComputeConfigListResult with 'results' (list), 'next_token' (str|None), and 'count' (int)

    Raises:
        ValueError: If both cloud_id and cloud_name are provided
    """
    if cloud_id and cloud_name:
        raise ValueError(
            "Only one of cloud_id or cloud_name can be provided, not both."
        )

    return _private_sdk.list_compute_configs(  # type: ignore
        name=name,
        id=_id,
        cloud_id=cloud_id,
        cloud_name=cloud_name,
        sort_by=sort_by,
        sort_order=sort_order,
        max_items=max_items,
        next_token=next_token,
        include_shared=include_shared,
    )
