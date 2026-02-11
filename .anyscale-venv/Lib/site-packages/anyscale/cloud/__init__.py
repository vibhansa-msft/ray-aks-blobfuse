import inspect
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.cloud._private.cloud_sdk import PrivateCloudSDK
from anyscale.cloud.commands import (
    _ADD_COLLABORATORS_ARG_DOCSTRINGS,
    _ADD_COLLABORATORS_EXAMPLE,
    _GET_ARG_DOCSTRINGS,
    _GET_DEFAULT_EXAMPLE,
    _GET_EXAMPLE,
    _TERMINATE_SYSTEM_CLUSTER_ARG_DOCSTRINGS,
    _TERMINATE_SYSTEM_CLUSTER_EXAMPLE,
    add_collaborators,
    get,
    get_default,
    list,  # noqa: A004 - claude_comment("claude-opus-4-5", "SDK public API re-export")
    terminate_system_cluster,
)
from anyscale.cloud.models import Cloud, CreateCloudCollaborator
from anyscale.connect import ClientBuilder


class CloudSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateCloudSDK(client=client, logger=logger, timer=timer)

    @sdk_docs(
        doc_py_example=_ADD_COLLABORATORS_EXAMPLE,
        arg_docstrings=_ADD_COLLABORATORS_ARG_DOCSTRINGS,
    )
    def add_collaborators(
        self, cloud: str, collaborators: List[CreateCloudCollaborator],
    ) -> None:
        """
        Batch add collaborators to a cloud.

        :param cloud: The cloud to add users to.
        :param collaborators: The list of collaborators to add to the cloud.
        """
        self._private_sdk.add_collaborators(cloud, collaborators)

    @sdk_docs(
        doc_py_example=_GET_EXAMPLE, arg_docstrings=_GET_ARG_DOCSTRINGS,
    )
    def get_cloud(
        self, id: Optional[str], name: Optional[str],  # noqa: A002
    ) -> Optional[Cloud]:
        """
        Retrieve a cloud by its name or ID.

        :param id: The ID of the cloud to retrieve.
        :param name: The name of the cloud to retrieve.
        :return: A ``Cloud`` object if found, otherwise ``None``.
        """
        return self._private_sdk.get(id=id, name=name)

    @sdk_docs(
        doc_py_example=_GET_DEFAULT_EXAMPLE, arg_docstrings={},
    )
    def get_default_cloud(self) -> Optional[Cloud]:
        """
        Get the default cloud for your organization.

        :return: The default ``Cloud`` object if it exists, otherwise ``None``.
        """
        return self._private_sdk.get_default()

    @sdk_docs(
        doc_py_example=_TERMINATE_SYSTEM_CLUSTER_EXAMPLE,
        arg_docstrings=_TERMINATE_SYSTEM_CLUSTER_ARG_DOCSTRINGS,
    )
    def terminate_system_cluster(self, cloud_id: str, wait: bool = False) -> str:
        """
        Terminate the system cluster for a given cloud.

        :return: ID of the terminated system cluster.
        """
        return self._private_sdk.terminate_system_cluster(cloud_id, wait)


# Note: indentation here matches that of connect.py::ClientBuilder.
BUILDER_HELP_FOOTER = """
        See ``anyscale.ClientBuilder`` for full documentation of
        this experimental feature."""


class CloudModule(ModuleType):
    """
    A custom callable module object for `anyscale.cloud`.

    This hack is needed since `anyscale.cloud` is a function for Anyscale connect
    but also a module for the SDK.
    """

    def __init__(self):
        # Expose attributes from the SDK.
        self.CloudSDK = CloudSDK
        self.add_collaborators = add_collaborators
        self.list = list
        self.get = get
        self.get_default = get_default
        self.terminate_system_cluster = terminate_system_cluster

        # Expose Anyscale connect
        self.new_builder = self._new_builder()

    # This code is copied from frontend/cli/anyscale/__init__.py.
    def _new_builder(self) -> Any:
        target = ClientBuilder.cloud

        def new_session_builder(*a: List[Any], **kw: Dict[str, Any]) -> Any:
            builder = ClientBuilder()
            return target(builder, *a, **kw)  # type: ignore

        new_session_builder.__name__ = "cloud"
        new_session_builder.__doc__ = target.__doc__ + BUILDER_HELP_FOOTER  # type: ignore
        new_session_builder.__signature__ = inspect.signature(target)  # type: ignore

        return new_session_builder

    def __call__(self, *args, **kwargs):
        """
        Define the behavior when `anyscale.cloud` is called for Anyscale connect.
        """
        return self.new_builder(*args, **kwargs)


sys.modules[__name__] = CloudModule()
