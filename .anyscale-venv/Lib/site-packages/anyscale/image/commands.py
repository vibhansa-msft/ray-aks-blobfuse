from typing import Optional

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_command
from anyscale.image._private.image_sdk import PrivateImageSDK
from anyscale.image.models import ImageBuild


_IMAGE_SDK_SINGLETON_KEY = "image_sdk"

_BUILD_EXAMPLE = """
import anyscale

containerfile = '''
FROM anyscale/ray:2.21.0-py39
RUN pip install --no-cache-dir pandas
'''

image_uri: str = anyscale.image.build(containerfile, name="mycoolimage")
"""

_BUILD_ARG_DOCSTRINGS = {
    "name": "The name of the image.",
    "containerfile": "The content of the Containerfile.",
    "ray_version": "The version of Ray to use in the image",
}


@sdk_command(
    _IMAGE_SDK_SINGLETON_KEY,
    PrivateImageSDK,
    doc_py_example=_BUILD_EXAMPLE,
    arg_docstrings=_BUILD_ARG_DOCSTRINGS,
)
def build(
    containerfile: str,
    *,
    name: str,
    ray_version: Optional[str] = None,
    _private_sdk: Optional[PrivateImageSDK] = None,
) -> str:
    """Build an image from a Containerfile.

    Returns the URI of the image.
    """
    return _private_sdk.build_image_from_containerfile_with_image_uri(  # type: ignore
        name, containerfile, ray_version=ray_version
    )


_GET_EXAMPLE = """
import anyscale

image_status = anyscale.image.get(name="mycoolimage")
"""

_GET_ARG_DOCSTRINGS = {
    "name": (
        "Get the details of an image.\n\n"
        "The name can contain an optional version, e.g., 'name:version'. "
        "If no version is provided, the latest one will be used.\n\n"
    )
}


@sdk_command(
    _IMAGE_SDK_SINGLETON_KEY,
    PrivateImageSDK,
    doc_py_example=_GET_EXAMPLE,
    arg_docstrings=_GET_ARG_DOCSTRINGS,
)
def get(*, name: str, _private_sdk: Optional[PrivateImageSDK] = None) -> ImageBuild:
    """The name can contain an optional version tag, i.e., 'name:version'.

    If no version is provided, the latest one will be returned.
    """
    return _private_sdk.get(name)  # type: ignore


_LIST_EXAMPLE = """
import anyscale

for image in anyscale.image.list(max_items=5):
    print(image.name, image.latest_build_revision)
"""

_LIST_ARG_DOCSTRINGS = {
    "image_id": "If provided, returns just the image with this ID wrapped in a one-page iterator.",
    "name": "Substring to match against the image name.",
    "image_name": "Substring to match against the resolved image URI (BYOD images only).",
    "project": "Name of the project that owns the image.",
    "creator_id": "Filter images by the creator user ID.",
    "include_archived": "Include archived images (default: False).",
    "include_anonymous": "Include anonymous (workspace-scoped) images (default: False).",
    "max_items": "Maximum total number of items to yield (default: iterate all).",
    "page_size": "Number of items to fetch per API request (default: API default).",
}


@sdk_command(
    _IMAGE_SDK_SINGLETON_KEY,
    PrivateImageSDK,
    doc_py_example=_LIST_EXAMPLE,
    arg_docstrings=_LIST_ARG_DOCSTRINGS,
)
def list(  # noqa: A001, PLR0913
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
    _private_sdk: Optional[PrivateImageSDK] = None,
) -> ResultIterator[ImageBuild]:
    """List images or fetch a single image by ID."""
    return _private_sdk.list(  # type: ignore
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


_REGISTER_EXAMPLE = """
import anyscale

image_uri: str = anyscale.image.register("docker.io/myuser/myimage:v2", name="mycoolimage")
"""

_REGISTER_ARG_DOCSTRINGS = {
    "image_uri": "The URI of the BYOD image to register.",
    "name": "Name for the container image. If the name already exists, a new version will be built. Otherwise, a new container image will be created.",
    "ray_version": "The Ray version (X.Y.Z) specified for this image specified by either an image URI or a containerfile. If you don't specify a Ray version, Anyscale defaults to the latest Ray version available at the time of the Anyscale CLI/SDK release.",
    "registry_login_secret": "Name or identifier of the secret containing credentials to authenticate to the docker registry hosting the image.",  # pragma: allowlist secret
}


@sdk_command(
    _IMAGE_SDK_SINGLETON_KEY,
    PrivateImageSDK,
    doc_py_example=_REGISTER_EXAMPLE,
    arg_docstrings=_REGISTER_ARG_DOCSTRINGS,
)
def register(
    image_uri: str,
    *,
    name: str,
    ray_version: Optional[str] = None,
    registry_login_secret: Optional[str] = None,
    _private_sdk: Optional[PrivateImageSDK] = None,
) -> str:
    """
    Registers a BYOD image with a container image name.

    Returns the URI of the image.
    """
    return _private_sdk.register_byod_image_with_name(  # type: ignore
        image_uri,
        registry_login_secret=registry_login_secret,
        ray_version=ray_version,
        name=name,
    )


_ARCHIVE_EXAMPLE = """
import anyscale

anyscale.image.archive(name="my-old-image")
"""

_ARCHIVE_ARG_DOCSTRINGS = {
    "name": "Name of the image to archive. Can include an optional version tag (e.g., 'name:version').",
}


@sdk_command(
    _IMAGE_SDK_SINGLETON_KEY,
    PrivateImageSDK,
    doc_py_example=_ARCHIVE_EXAMPLE,
    arg_docstrings=_ARCHIVE_ARG_DOCSTRINGS,
)
def archive(name: str, *, _private_sdk: Optional[PrivateImageSDK] = None,) -> None:
    """Archive an image and all of its versions.

    Once archived, the image name will no longer be usable in the organization.
    Archived images can still be viewed using `include_archived=True` in list().
    """
    return _private_sdk.archive(name=name)  # type: ignore
