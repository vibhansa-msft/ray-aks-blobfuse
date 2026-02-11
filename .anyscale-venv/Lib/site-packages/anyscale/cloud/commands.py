from typing import List, Optional

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_command
from anyscale.cloud._private.cloud_sdk import Cloud, PrivateCloudSDK
from anyscale.cloud.models import CreateCloudCollaborator


_CLOUD_SDK_SINGLETON_KEY = "cloud_sdk"

_LIST_EXAMPLE = """
import anyscale

# Example: Get the first 50 clouds
for cloud in anyscale.cloud.list(max_items=50):
    print(cloud.name)
"""

_LIST_ARG_DOCSTRINGS = {
    "cloud_id": "If provided, returns just the cloud with this ID wrapped in a one-page iterator.",
    "name": "Substring or exact name to match against the cloud name.",
    "max_items": "Maximum total number of items to yield (default: iterate all).",
    "page_size": "Number of items to fetch per API request (default: API default).",
}


@sdk_command(
    _CLOUD_SDK_SINGLETON_KEY,
    PrivateCloudSDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_ARG_DOCSTRINGS,
)
def list(  # noqa: A001
    *,
    cloud_id: Optional[str] = None,
    name: Optional[str] = None,
    max_items: Optional[int] = None,
    page_size: Optional[int] = None,
    _private_sdk: Optional[PrivateCloudSDK] = None,
) -> ResultIterator[Cloud]:
    """List clouds or fetch a single cloud by ID/name."""
    return _private_sdk.list(  # type: ignore
        cloud_id=cloud_id, name=name, max_items=max_items, page_size=page_size
    )


_ADD_COLLABORATORS_EXAMPLE = """
import anyscale
from anyscale.cloud.models import CloudPermissionLevel, CreateCloudCollaborator

anyscale.cloud.add_collaborators(
    cloud="cloud_name",
    collaborators=[
        CreateCloudCollaborator(
            email="test1@anyscale.com",
            permission_level=CloudPermissionLevel.WRITE,
        ),
        CreateCloudCollaborator(
            email="test2@anyscale.com",
            permission_level=CloudPermissionLevel.READONLY,
        ),
    ],
)
"""

_ADD_COLLABORATORS_ARG_DOCSTRINGS = {
    "cloud": "The cloud to add users to.",
    "collaborators": "The list of collaborators to add to the cloud.",
}


@sdk_command(
    _CLOUD_SDK_SINGLETON_KEY,
    PrivateCloudSDK,
    doc_py_example=_ADD_COLLABORATORS_EXAMPLE,
    arg_docstrings=_ADD_COLLABORATORS_ARG_DOCSTRINGS,
)
def add_collaborators(
    cloud: str,
    collaborators: List[CreateCloudCollaborator],
    *,
    _private_sdk: Optional[PrivateCloudSDK] = None,
) -> str:
    """
    Batch add collaborators to a cloud.

    :param cloud: The cloud to add users to.
    :param collaborators: The list of collaborators to add to the cloud.
    """
    return _private_sdk.add_collaborators(cloud, collaborators)  # type: ignore


_GET_EXAMPLE = """
import anyscale

# Get a cloud by ID
cloud_by_id = anyscale.cloud.get(id="cloud_id")

# Get a cloud by name
cloud_by_name = anyscale.cloud.get(name="cloud_name")
"""

_GET_ARG_DOCSTRINGS = {
    "id": "The ID of the cloud to retrieve.",
    "name": "The name of the cloud to retrieve.",
}


@sdk_command(
    _CLOUD_SDK_SINGLETON_KEY,
    PrivateCloudSDK,
    doc_py_example=_GET_EXAMPLE,
    arg_docstrings=_GET_ARG_DOCSTRINGS,
)
def get(
    id: Optional[str] = None,  # noqa: A002
    name: Optional[str] = None,
    *,
    _private_sdk: Optional[PrivateCloudSDK] = None,
) -> Optional[Cloud]:
    """
    Get the cloud model for the provided cloud ID or name.

    If neither ID nor name is provided, returns `None`.

    :param id: The ID of the cloud to retrieve.
    :param name: The name of the cloud to retrieve.
    :return: A `Cloud` object if found, otherwise `None`.
    """
    return _private_sdk.get(id=id, name=name)  # type: ignore


_GET_DEFAULT_EXAMPLE = """
import anyscale

# Get the user's default cloud
default_cloud = anyscale.cloud.get_default()
"""


@sdk_command(
    _CLOUD_SDK_SINGLETON_KEY,
    PrivateCloudSDK,
    doc_py_example=_GET_DEFAULT_EXAMPLE,
    arg_docstrings={},
)
def get_default(*, _private_sdk: Optional[PrivateCloudSDK] = None) -> Optional[Cloud]:
    """
    Get the user's default cloud.

    :return: The default `Cloud` object if it exists, otherwise `None`.
    """
    return _private_sdk.get_default()  # type: ignore


_TERMINATE_SYSTEM_CLUSTER_EXAMPLE = """
import anyscale

# Terminate the system cluster for the cloud with the specified ID
anyscale.cloud.terminate_system_cluster(cloud_id="cloud_id", wait=True)
"""

_TERMINATE_SYSTEM_CLUSTER_ARG_DOCSTRINGS = {
    "cloud_id": "The ID of the cloud whose system cluster should be terminated.",
    "wait": "If True, wait for the system cluster to be terminated before returning. Defaults to False.",
}


@sdk_command(
    _CLOUD_SDK_SINGLETON_KEY,
    PrivateCloudSDK,
    doc_py_example=_TERMINATE_SYSTEM_CLUSTER_EXAMPLE,
    arg_docstrings=_TERMINATE_SYSTEM_CLUSTER_ARG_DOCSTRINGS,
)
def terminate_system_cluster(
    cloud_id: str,
    wait: Optional[bool] = False,
    *,
    _private_sdk: Optional[PrivateCloudSDK] = None,
) -> str:
    """
    Terminate the system cluster for the specified cloud.

    :param cloud: The name of the cloud whose system cluster should be terminated.
    :param wait: If True, wait for the system cluster to be terminated before returning. Defaults to False.
    :return: ID of the terminated system cluster.
    """
    return _private_sdk.terminate_system_cluster(cloud_id, wait)  # type: ignore
