from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Dict, Optional, Union

from anyscale._private.models.model_base import ModelBase, ModelEnum


class ImageBuildStatus(ModelEnum):
    """Status of an image build operation."""

    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"

    def __str__(self):
        return self.name

    __docstrings__: ClassVar[Dict[str, str]] = {
        IN_PROGRESS: "The image build is in progress.",
        SUCCEEDED: "The image build succeeded.",
        FAILED: "The image build failed.",
        UNKNOWN: "The CLI/SDK received an unexpected state from the API server. In most cases, this means you need to update the CLI.",
    }


@dataclass(frozen=True)
class ImageBuild(ModelBase):
    __doc_py_example__ = """\
import anyscale
from anyscale.image.models import ImageBuild

first_image: ImageBuild = next(anyscale.image.list(max_items=1), None)
"""

    id: str = field(metadata={"docstring": "Unique identifier of the image."},)

    def _validate_id(self, id: str):  # noqa: A002, A003
        if id is None or not isinstance(id, str):
            raise ValueError("The image id must be a string.")

    name: str = field(
        metadata={
            "docstring": "Human-readable name assigned to the image (cluster environment)."
        },
    )

    def _validate_name(self, name: str):
        if name is None or not isinstance(name, str):
            raise ValueError("The image name must be a string.")

    project_id: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Project identifier that owns this image, if available."
        },
    )

    def _validate_project_id(self, project_id: Optional[str]):
        if project_id is not None and not isinstance(project_id, str):
            raise ValueError("The project id must be a string if provided.")

    creator_id: Optional[str] = field(
        default=None,
        metadata={"docstring": "Identifier of the user who created the image."},
    )

    def _validate_creator_id(self, creator_id: Optional[str]):
        if creator_id is not None and not isinstance(creator_id, str):
            raise ValueError("The creator id must be a string if provided.")

    creator_email: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Email address of the user who created the image, if available."
        },
    )

    def _validate_creator_email(self, creator_email: Optional[str]):
        if creator_email is not None and not isinstance(creator_email, str):
            raise ValueError("The creator email must be a string if provided.")

    is_anonymous: bool = field(
        default=False,
        metadata={
            "docstring": "Whether the image was created as an anonymous (workspace-scoped) resource."
        },
    )

    def _validate_is_anonymous(self, is_anonymous: bool):
        if not isinstance(is_anonymous, bool):
            raise ValueError("The is_anonymous flag must be a boolean.")

    created_at: Optional[datetime] = field(
        default=None, metadata={"docstring": "Timestamp when the image was created."},
    )

    def _validate_created_at(self, created_at: Optional[datetime]):
        if created_at is not None and not isinstance(created_at, datetime):
            raise ValueError("The created_at field must be a datetime if provided.")

    last_modified_at: Optional[datetime] = field(
        default=None,
        metadata={"docstring": "Timestamp when the image was last updated."},
    )

    def _validate_last_modified_at(self, last_modified_at: Optional[datetime]):
        if last_modified_at is not None and not isinstance(last_modified_at, datetime):
            raise ValueError(
                "The last_modified_at field must be a datetime if provided."
            )

    latest_build_id: Optional[str] = field(
        default=None,
        metadata={"docstring": "Identifier of the most recent build for this image."},
    )

    def _validate_latest_build_id(self, latest_build_id: Optional[str]):
        if latest_build_id is not None and not isinstance(latest_build_id, str):
            raise ValueError("The latest build id must be a string if provided.")

    latest_build_revision: Optional[int] = field(
        default=None,
        metadata={
            "docstring": "Revision number of the most recent image build (treated as the latest version)."
        },
    )

    def _validate_latest_build_revision(self, latest_build_revision: Optional[int]):
        if latest_build_revision is not None and not isinstance(
            latest_build_revision, int
        ):
            raise ValueError(
                "The latest build revision must be an integer if provided."
            )

    latest_build_status: Optional[Union[str, ImageBuildStatus]] = field(
        default=None, metadata={"docstring": "Status of the most recent image build."},
    )

    def _validate_latest_build_status(
        self, latest_build_status: Optional[Union[str, ImageBuildStatus]]
    ) -> Optional[ImageBuildStatus]:
        if latest_build_status is None:
            return None
        if isinstance(latest_build_status, ImageBuildStatus):
            return latest_build_status
        return ImageBuildStatus.validate(str(latest_build_status).upper())

    latest_image_uri: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "URI for the latest published image version, if available."
        },
    )

    def _validate_latest_image_uri(self, latest_image_uri: Optional[str]):
        if latest_image_uri is not None and not isinstance(latest_image_uri, str):
            raise ValueError("The latest image URI must be a string if provided.")
