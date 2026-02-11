from typing import Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_docs
from anyscale.cli_logger import BlockLogger
from anyscale.image._private.image_sdk import PrivateImageSDK
from anyscale.image.commands import (
    _ARCHIVE_ARG_DOCSTRINGS,
    _ARCHIVE_EXAMPLE,
    _BUILD_ARG_DOCSTRINGS,
    _BUILD_EXAMPLE,
    _GET_ARG_DOCSTRINGS,
    _GET_EXAMPLE,
    _LIST_ARG_DOCSTRINGS,
    _LIST_EXAMPLE,
    _REGISTER_ARG_DOCSTRINGS,
    _REGISTER_EXAMPLE,
    archive as archive,
    build as build,
    get as get,
    list as list,  # noqa: A004 - claude_comment("claude-opus-4-5", "SDK public API re-export")
    register as register,
)
from anyscale.image.models import ImageBuild


class ImageSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
    ):
        self._private_sdk = PrivateImageSDK(client=client, logger=logger,)

    @sdk_docs(
        doc_py_example=_BUILD_EXAMPLE, arg_docstrings=_BUILD_ARG_DOCSTRINGS,
    )
    def build(  # noqa: F811
        self, containerfile: str, *, name: str, ray_version: Optional[str] = None
    ) -> str:  # noqa: F811
        """Build an image from a Containerfile.

        Returns the URI of the image.
        """
        return self._private_sdk.build_image_from_containerfile_with_image_uri(
            name, containerfile, ray_version=ray_version
        )

    @sdk_docs(
        doc_py_example=_GET_EXAMPLE, arg_docstrings=_GET_ARG_DOCSTRINGS,
    )
    def get(self, *, name: str) -> ImageBuild:  # noqa: F811
        """The name can contain an optional version tag, i.e., 'name:version'.

        If no version is provided, the latest one will be returned.
        """
        return self._private_sdk.get(name)

    @sdk_docs(
        doc_py_example=_LIST_EXAMPLE, arg_docstrings=_LIST_ARG_DOCSTRINGS,
    )
    def list(  # noqa: A001, F811, PLR0913
        self,
        *,
        image_id: Optional[str] = None,
        name: Optional[str] = None,
        image_name: Optional[str] = None,
        project: Optional[str] = None,
        creator_id: Optional[str] = None,
        include_archived: bool = False,
        include_anonymous: bool = False,
        max_items: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> ResultIterator[ImageBuild]:
        """List images or fetch a single image by ID."""
        return self._private_sdk.list(
            image_id=image_id,
            name=name,
            image_name=image_name,
            project=project,
            creator_id=creator_id,
            include_archived=include_archived,
            include_anonymous=include_anonymous,
            max_items=max_items,
            page_size=page_size,
        )

    @sdk_docs(
        doc_py_example=_REGISTER_EXAMPLE, arg_docstrings=_REGISTER_ARG_DOCSTRINGS,
    )
    def register(  # noqa: F811
        self,
        image_uri: str,
        *,
        name: str,
        ray_version: Optional[str] = None,
        registry_login_secret: Optional[str] = None,
    ) -> str:
        """
        Register a BYOD image with a container image name.
        """
        return self._private_sdk.register_byod_image_with_name(
            image_uri,
            registry_login_secret=registry_login_secret,
            ray_version=ray_version,
            name=name,
        )

    @sdk_docs(
        doc_py_example=_ARCHIVE_EXAMPLE, arg_docstrings=_ARCHIVE_ARG_DOCSTRINGS,
    )
    def archive(self, name: str) -> None:  # noqa: F811
        """Archive an image and all of its versions.

        Once archived, the image name will no longer be usable in the organization.
        Archived images can still be viewed using `include_archived=True` in list().
        """
        return self._private_sdk.archive(name=name)
