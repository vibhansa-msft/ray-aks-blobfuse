from typing import Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.compute_config._private.compute_config_sdk import PrivateComputeConfigSDK
from anyscale.compute_config.commands import (
    _ARCHIVE_ARG_DOCSTRINGS,
    _ARCHIVE_EXAMPLE,
    _CREATE_ARG_DOCSTRINGS,
    _CREATE_EXAMPLE,
    _GET_ARG_DOCSTRINGS,
    _GET_DEFAULT_ARG_DOCSTRINGS,
    _GET_DEFAULT_EXAMPLE,
    _GET_EXAMPLE,
    _LIST_ARG_DOCSTRINGS,
    _LIST_EXAMPLE,
    archive as archive,
    create as create,
    get as get,
    get_default as get_default,
    list as list,  # noqa: A004
)
from anyscale.compute_config.models import (
    ComputeConfig as ComputeConfig,
    ComputeConfigListResult,
    ComputeConfigType,
    ComputeConfigVersion,
    HeadNodeConfig as HeadNodeConfig,
    MultiResourceComputeConfig as MultiResourceComputeConfig,
    WorkerNodeGroupConfig as WorkerNodeGroupConfig,
)


class ComputeConfigSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateComputeConfigSDK(
            client=client, logger=logger, timer=timer
        )

    @sdk_docs(
        doc_py_example=_CREATE_EXAMPLE, arg_docstrings=_CREATE_ARG_DOCSTRINGS,
    )
    def create(  # noqa: F811
        self, config: ComputeConfigType, *, name: Optional[str],
    ) -> str:
        """Create a new version of a compute config.

        Returns the full name of the registered compute config, including the version.
        """
        full_name, _ = self._private_sdk.create_compute_config(config, name=name)
        return full_name

    @sdk_docs(
        doc_py_example=_GET_EXAMPLE, arg_docstrings=_GET_ARG_DOCSTRINGS,
    )
    def get(  # noqa: F811
        self, name: str, *, include_archived: bool = False, _id: Optional[str] = None,
    ) -> ComputeConfigVersion:
        """Get the compute config with the specified name.

        The name can contain an optional version tag, i.e., 'name:version'.
        If no version is provided, the latest one will be returned.
        """
        # NOTE(edoakes): I want to avoid exposing fetching by ID in the public API,
        # but it's needed for parity with the existing CLI. Therefore I am adding it
        # as a hidden private API that can be used like: (`name="", _id=id`).
        return self._private_sdk.get_compute_config(
            name=name or None, id=_id, include_archived=include_archived
        )

    @sdk_docs(
        doc_py_example=_ARCHIVE_EXAMPLE, arg_docstrings=_ARCHIVE_ARG_DOCSTRINGS,
    )
    def archive(self, name: str, *, _id: Optional[str] = None):  # noqa: F811
        """Archive a compute config and all of its versions.

        The name can contain an optional version, e.g., 'name:version'.
        If no version is provided, the latest one will be archived.

        Once a compute config is archived, its name will no longer be usable in the organization.
        """
        # NOTE(edoakes): I want to avoid exposing fetching by ID in the public API,
        # but it's needed for parity with the existing CLI. Therefore I am adding it
        # as a hidden private API that can be used like: (`name="", _id=id`).
        return self._private_sdk.archive_compute_config(name=name or None, id=_id,)

    @sdk_docs(
        doc_py_example=_GET_DEFAULT_EXAMPLE, arg_docstrings=_GET_DEFAULT_ARG_DOCSTRINGS,
    )
    def get_default(  # noqa: F811
        self, *, cloud: Optional[str] = None, cloud_resource: Optional[str] = None,
    ) -> ComputeConfigVersion:
        """Get the default compute config for the specified cloud.

        Returns the default compute configuration that will be used when no compute config
        is explicitly specified for a workload.

        Args:
            cloud: Name of the cloud. If not provided, uses the organization's default cloud.
            cloud_resource: Name of the cloud resource. If not provided, uses the default
                cloud resource for the cloud.

        Returns:
            ComputeConfigVersion containing the default compute config.
        """
        return self._private_sdk.get_default_compute_config(
            cloud=cloud, cloud_resource=cloud_resource
        )

    @sdk_docs(
        doc_py_example=_LIST_EXAMPLE, arg_docstrings=_LIST_ARG_DOCSTRINGS,
    )
    def list(  # noqa: F811, A003, PLR0913
        self,
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

        Returns:
            ComputeConfigListResult with 'results' (list), 'next_token' (str|None), and 'count' (int)

        Raises:
            ValueError: If both cloud_id and cloud_name are provided
        """
        if cloud_id and cloud_name:
            raise ValueError(
                "Only one of cloud_id or cloud_name can be provided, not both."
            )

        return self._private_sdk.list_compute_configs(
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
