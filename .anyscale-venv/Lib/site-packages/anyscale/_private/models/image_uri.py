import re

from anyscale.sdk.anyscale_client.models.cluster_environment import ClusterEnvironment
from anyscale.sdk.anyscale_client.models.cluster_environment_build import (
    ClusterEnvironmentBuild,
)


DEFAULT_APP_CONFIG_ID_PREFIX = "DEFAULT_APP_CONFIG_ID"
IMAGE_URI_PATTERN = "[registry_host/]user_name/repository[:tag][@digest]"
IMAGE_URI_PATTERN_RE = re.compile(
    r"^"
    # Optional registry host: hostname with optional port
    r"((?P<host>[a-zA-Z0-9.-]+)(?::(?P<port>[0-9]+))?/)?"
    # Repository: user_name/repository
    r"(?P<user>[a-zA-Z0-9-_]+/)?"  # user is optional.
    r"(?P<repository>[a-zA-Z0-9-/_.]+)"
    # Optional Tag: version or string after ':'
    r"(:(?P<tag>[a-zA-Z0-9_.-]+))?"
    # Optional Digest: string after '@'
    # Note that when both tag and digest are provided, the tag is ignored.
    r"(@(?P<digest>[a-zA-Z0-9:]+))?"
    r"$"
)

ANYSCALE_CLUSTER_ENV_PREFIX = "anyscale/image/"
DEFAULT_IMAGE_URI_PREFIX = "anyscale/ray"


class ImageURI:
    image_uri: str

    @classmethod
    def from_str(cls, image_uri_str: str):
        if not isinstance(image_uri_str, str):
            raise TypeError("'image_uri' must be a string.")

        if image_uri_str:
            matches = IMAGE_URI_PATTERN_RE.match(image_uri_str)
            if not matches:
                raise ValueError(
                    f"Invalid image URI: '{image_uri_str}'. Must be in the format: '{IMAGE_URI_PATTERN}'."
                )
        instance = super().__new__(cls)
        instance.image_uri = image_uri_str
        return instance

    @classmethod
    def _is_byod_build(cls, build: ClusterEnvironmentBuild) -> bool:
        return build.is_byod or (not build.config_json and not build.containerfile)

    @classmethod
    def _is_default_build(cls, build: ClusterEnvironmentBuild) -> bool:
        return bool(
            isinstance(build.cluster_environment_id, str)
            and build.cluster_environment_id.startswith(DEFAULT_APP_CONFIG_ID_PREFIX)
        )

    @classmethod
    def from_cluster_env_build(
        cls,
        cluster_env: ClusterEnvironment,
        build: ClusterEnvironmentBuild,
        use_image_alias: bool = False,
    ):
        if cls._is_byod_build(build) and not use_image_alias:
            if build.docker_image_name:
                return ImageURI.from_str(build.docker_image_name)
            else:
                return ImageURI.from_str("")
        elif cls._is_default_build(build):
            return ImageURI.from_str(build.docker_image_name)
        else:
            image_uri_str = (
                ANYSCALE_CLUSTER_ENV_PREFIX + f"{cluster_env.name}:{build.revision}"
            )
            return ImageURI.from_str(image_uri_str)

    def to_cluster_env_name(self) -> str:
        """Convert the image URI to a cluster environment name."""
        pattern = re.compile("^[A-Za-z0-9_-]+$")
        # Keep only characters that match the pattern
        escaped = []
        for c in self.image_uri:
            if not pattern.match(c):
                escaped.append("-")
            else:
                escaped.append(c)
        return "".join(escaped)

    def is_cluster_env_image(self) -> bool:
        """Check if the image URI is a cluster environment image."""
        return self.image_uri.startswith(ANYSCALE_CLUSTER_ENV_PREFIX)

    def is_default_image(self) -> bool:
        """Check if the image URI is a default image."""
        return self.image_uri.startswith(DEFAULT_IMAGE_URI_PREFIX)

    def to_cluster_env_identifier(self) -> str:
        assert self.is_cluster_env_image()
        identifier = self.image_uri.split("/")[2]
        return identifier

    def __str__(self):
        return self.image_uri

    def __repr__(self):
        return self.image_uri

    def __eq__(self, other):
        if isinstance(other, ImageURI):
            return self.image_uri == other.image_uri
        return False

    def __hash__(self):
        return hash(self.image_uri)
