from contextlib import suppress
import os
from types import SimpleNamespace
from typing import cast, ClassVar, Dict, List, Optional

from anyscale._private.anyscale_client.common import AnyscaleClientInterface
from anyscale._private.models.image_uri import ImageURI
from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models.decorated_application_template import (
    DecoratedApplicationTemplate,
)
from anyscale.image.models import ImageBuild, ImageBuildStatus
from anyscale.sdk.anyscale_client import (
    ClusterEnvironmentBuild,
    ClusterEnvironmentBuildStatus,
)


ENABLE_IMAGE_BUILD_FOR_TRACKED_REQUIREMENTS = (
    os.environ.get("ANYSCALE_ENABLE_IMAGE_BUILD_FOR_TRACKED_REQUIREMENTS", "0") == "1"
)


MAX_PAGE_SIZE = 50


class PrivateImageSDK(BaseSDK):
    def __init__(
        self,
        *,
        logger: Optional[BlockLogger] = None,
        client: Optional[AnyscaleClientInterface] = None,
    ):
        super().__init__(logger=logger, client=client)
        self._enable_image_build_for_tracked_requirements = (
            ENABLE_IMAGE_BUILD_FOR_TRACKED_REQUIREMENTS
        )

    @property
    def enable_image_build_for_tracked_requirements(self) -> bool:
        return self._enable_image_build_for_tracked_requirements

    def get_default_image(self) -> str:
        return self.client.get_default_build_id()

    def get_image_build(self, build_id: str) -> Optional[ClusterEnvironmentBuild]:
        return self.client.get_cluster_env_build(build_id)

    _BACKEND_IMAGE_STATUS_TO_IMAGE_BUILD_STATUS: ClassVar[
        Dict[ClusterEnvironmentBuildStatus, ImageBuildStatus]
    ] = {
        ClusterEnvironmentBuildStatus.PENDING: ImageBuildStatus.IN_PROGRESS,
        ClusterEnvironmentBuildStatus.IN_PROGRESS: ImageBuildStatus.IN_PROGRESS,
        ClusterEnvironmentBuildStatus.SUCCEEDED: ImageBuildStatus.SUCCEEDED,
        ClusterEnvironmentBuildStatus.FAILED: ImageBuildStatus.FAILED,
        ClusterEnvironmentBuildStatus.PENDING_CANCELLATION: ImageBuildStatus.FAILED,
        ClusterEnvironmentBuildStatus.CANCELED: ImageBuildStatus.FAILED,
    }

    def _get_image_build_status(
        self, build: ClusterEnvironmentBuild
    ) -> ImageBuildStatus:
        return cast(
            ImageBuildStatus,
            self._BACKEND_IMAGE_STATUS_TO_IMAGE_BUILD_STATUS.get(
                build.status, ImageBuildStatus.UNKNOWN
            ),
        )

    def get(self, name: str) -> ImageBuild:
        """Get an image by name with optional version (name:version format).

        If no version is specified, returns the latest build.
        If a version is specified, searches for that specific revision.
        """
        image_name, version_str = name.split(":", 1) if ":" in name else (name, None)
        version: Optional[int] = None
        if version_str:
            try:
                version = int(version_str)
            except ValueError:
                raise ValueError(
                    f"Invalid version format '{version_str}'. Version must be a number."
                )

        response = self.client.list_application_templates(
            name=image_name,
            image_name=None,
            creator_id=None,
            project=None,
            include_archived=False,
            defaults_first=False,
            count=50,
            paging_token=None,
        )

        # list_application_templates does substring matching, need exact match
        results = response.results if response.results else []
        template = next((t for t in results if t.name == image_name), None)
        if not template:
            raise ValueError(f"Image '{image_name}' not found.")

        if not template.latest_build:
            raise ValueError(f"No builds found for image '{image_name}'.")

        # If specific version requested, verify it exists and get its details
        if version is not None and template.latest_build.revision != version:
            for b in self.client.list_cluster_env_builds(template.id):
                if b.revision == version and b.id:
                    image_uri_obj = self.get_image_uri_from_build_id(
                        b.id, use_image_alias=True
                    )
                    return ImageBuild(
                        id=template.id or "",
                        name=template.name or "",
                        project_id=template.project_id,
                        creator_id=template.creator_id,
                        creator_email=template.creator.email
                        if template.creator
                        else None,
                        is_anonymous=bool(template.anonymous),
                        created_at=template.created_at,
                        last_modified_at=template.last_modified_at,
                        latest_build_id=b.id,
                        latest_build_revision=b.revision,
                        latest_build_status=self._get_image_build_status(b),
                        latest_image_uri=image_uri_obj.image_uri
                        if image_uri_obj
                        else None,
                    )
            raise ValueError(f"Version {version} not found for image '{image_name}'.")

        return self._application_template_to_image_build(template)

    def build_image_from_containerfile(
        self,
        name: str,
        containerfile: str,
        ray_version: Optional[str] = None,
        anonymous: bool = False,
    ) -> str:
        return self.client.get_cluster_env_build_id_from_containerfile(
            cluster_env_name=name,
            containerfile=containerfile,
            anonymous=anonymous,
            ray_version=ray_version,
        )

    def build_image_from_containerfile_with_image_uri(
        self,
        name: str,
        containerfile: str,
        ray_version: Optional[str] = None,
        anonymous: bool = False,
    ) -> str:
        build_id = self.build_image_from_containerfile(
            name=name,
            containerfile=containerfile,
            ray_version=ray_version,
            anonymous=anonymous,
        )
        image_uri = self.get_image_uri_from_build_id(build_id)
        if image_uri:
            return image_uri.image_uri
        raise RuntimeError(
            f"This is a bug! Failed to get image uri for build {build_id} that just created."
        )

    def build_image_from_requirements(
        self, name: str, base_build_id: str, requirements: List[str]
    ):
        if requirements:
            base_build = self.client.get_cluster_env_build(base_build_id)
            if (
                base_build
                and base_build.status == ClusterEnvironmentBuildStatus.SUCCEEDED
                and base_build.docker_image_name
            ):
                self.logger.info(f"Using tracked python packages: {requirements}")
                lines = [
                    "# syntax=docker/dockerfile:1",
                    f"FROM {base_build.docker_image_name}",
                ]
                for requirement in requirements:
                    lines.append(f'RUN pip install "{requirement}"')
                return self.build_image_from_containerfile(
                    name, "\n".join(lines), ray_version=base_build.ray_version
                )
            else:
                raise RuntimeError(
                    f"Base build {base_build_id} is not a successful build."
                )
        else:
            return base_build_id

    def registery_image(
        self,
        image_uri: str,
        registry_login_secret: Optional[str] = None,
        ray_version: Optional[str] = None,
    ) -> str:
        image_uri_checked = ImageURI.from_str(image_uri_str=image_uri)
        return self.client.get_cluster_env_build_id_from_image_uri(
            image_uri=image_uri_checked,
            registry_login_secret=registry_login_secret,
            ray_version=ray_version,
        )

    def register_byod_image_with_name(
        self,
        image_uri_str: str,
        name: str,
        ray_version: Optional[str] = None,
        registry_login_secret: Optional[str] = None,
    ) -> str:
        image_uri_checked = ImageURI.from_str(image_uri_str=image_uri_str)
        if image_uri_checked.is_cluster_env_image():
            raise RuntimeError(
                f"Image URI {image_uri_str} is not a BYOD image. "
                "The 'register' command only works with BYOD images."
            )

        build_id = self.client.get_cluster_env_build_id_from_image_uri(
            image_uri=image_uri_checked,
            registry_login_secret=registry_login_secret,
            ray_version=ray_version,
            name=name,
        )

        fetched_image_uri = self.get_image_uri_from_build_id(
            build_id, use_image_alias=True
        )
        if fetched_image_uri:
            return fetched_image_uri.image_uri
        raise RuntimeError(
            f"This is a bug! Failed to get image uri for build {build_id} that just created."
        )

    def get_image_uri_from_build_id(
        self, build_id: str, use_image_alias: bool = False
    ) -> Optional[ImageURI]:
        return self.client.get_cluster_env_build_image_uri(
            cluster_env_build_id=build_id, use_image_alias=use_image_alias,
        )

    def _application_template_to_image_build(
        self, template: DecoratedApplicationTemplate
    ) -> ImageBuild:
        """Convert a DecoratedApplicationTemplate to an ImageBuild."""
        latest_build = template.latest_build
        latest_build_id = latest_build.id if latest_build else None
        latest_build_revision = latest_build.revision if latest_build else None
        latest_build_status = None
        latest_image_uri = None

        if latest_build:
            if latest_build.status:
                with suppress(ValueError):
                    latest_build_status = ImageBuildStatus.validate(
                        latest_build.status.upper()
                    )

            latest_image_uri = latest_build.docker_image_name
            if not latest_image_uri and latest_build_id:
                image_uri = self.get_image_uri_from_build_id(
                    latest_build_id, use_image_alias=True
                )
                if image_uri:
                    latest_image_uri = image_uri.image_uri

        creator_email = template.creator.email if template.creator else None

        return ImageBuild(
            id=template.id or "",
            name=template.name or "",
            project_id=template.project_id,
            creator_id=template.creator_id,
            creator_email=creator_email,
            is_anonymous=bool(template.anonymous),
            created_at=template.created_at,
            last_modified_at=template.last_modified_at,
            latest_build_id=latest_build_id,
            latest_build_revision=latest_build_revision,
            latest_build_status=latest_build_status,
            latest_image_uri=latest_image_uri,
        )

    def _should_filter_template(
        self,
        template: DecoratedApplicationTemplate,
        include_anonymous: bool,
        include_archived: bool,
    ) -> bool:
        """Return True if template should be filtered out."""
        if not include_anonymous and template.anonymous:
            return True
        return bool(not include_archived and template.archived_at is not None)

    def list(  # noqa: PLR0913, PLR0917
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
        if page_size is not None and not (1 <= page_size <= MAX_PAGE_SIZE):
            raise ValueError(
                f"page_size must be between 1 and {MAX_PAGE_SIZE}, inclusive."
            )

        # Handle single image lookup by ID
        if image_id is not None:
            template = self.client.get_application_template(image_id)

            if template is None or self._should_filter_template(
                template, include_anonymous, include_archived
            ):
                # Return empty iterator
                return ResultIterator(
                    page_token=None,
                    max_items=0,
                    fetch_page=lambda _: SimpleNamespace(
                        results=[], metadata=SimpleNamespace(next_paging_token=None),
                    ),
                    parse_fn=None,
                )

            # Return single-item iterator
            summary = self._application_template_to_image_build(template)
            return ResultIterator(
                page_token=None,
                max_items=1,
                fetch_page=lambda token: SimpleNamespace(
                    results=[summary] if token is None else [],
                    metadata=SimpleNamespace(next_paging_token=None),
                ),
                parse_fn=None,
            )

        # Handle paginated list
        def _fetch_page(token: Optional[str]) -> SimpleNamespace:
            response = self.client.list_application_templates(
                name=name,
                image_name=image_name,
                creator_id=creator_id,
                project=project,
                include_archived=include_archived,
                defaults_first=False,
                count=page_size,
                paging_token=token,
            )

            results = response.results if response.results else []
            filtered_results = [
                template
                for template in results
                if not self._should_filter_template(
                    template, include_anonymous, include_archived
                )
            ]

            next_token = (
                response.metadata.next_paging_token if response.metadata else None
            )

            return SimpleNamespace(
                results=filtered_results,
                metadata=SimpleNamespace(next_paging_token=next_token),
            )

        return ResultIterator(
            page_token=None,
            max_items=max_items,
            fetch_page=_fetch_page,
            parse_fn=self._application_template_to_image_build,
        )

    def archive(
        self, *, name: Optional[str] = None, image_id: Optional[str] = None
    ) -> None:  # noqa: A002
        """Archive an image by name or ID.

        If name is provided, looks up the image ID first.
        Once archived, the image name will no longer be usable in the organization.

        Args:
            name: The name of the image to archive (can include version, e.g., 'name:version').
            image_id: The ID of the image to archive.

        Raises:
            ValueError: If the image is not found or cannot be archived.
        """
        if name is None and image_id is None:
            raise ValueError("Either 'name' or 'image_id' must be provided.")

        if name is not None and image_id is not None:
            raise ValueError(
                "Only one of 'name' or 'image_id' can be provided, not both."
            )

        resolved_id = image_id
        if name is not None:
            # Parse name to extract just the image name (without version)
            image_name = name.split(":", 1)[0] if ":" in name else name

            # Look up the image by name
            response = self.client.list_application_templates(
                name=image_name,
                image_name=None,
                creator_id=None,
                project=None,
                include_archived=False,
                defaults_first=False,
                count=50,
                paging_token=None,
            )

            results = response.results if response.results else []
            template = next((t for t in results if t.name == image_name), None)
            if not template:
                raise ValueError(f"Image '{image_name}' not found.")
            resolved_id = template.id

        self.client.archive_image(image_id=resolved_id)
        self.logger.info("Successfully archived image.")
