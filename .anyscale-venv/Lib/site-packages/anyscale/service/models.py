from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Union

from anyscale._private.models import ModelBase, ModelEnum
from anyscale._private.workload import WorkloadConfig


@dataclass(frozen=True)
class RayGCSExternalStorageConfig(ModelBase):
    """Configuration options for external storage for the Ray Global Control Store (GCS).

    When external storage is enabled for a service, it will be able to continue serving traffic if the
    head node goes down. Without external storage, the entire cluster will be restarted.
    """

    __doc_py_example__ = """\
from anyscale.service.models import ServiceConfig, RayGCSExternalStorageConfig

config = ServiceConfig(
    name="my-service",
    applications=[{"import_path": "main:app"}],
    ray_gcs_external_storage_config=RayGCSExternalStorageConfig(
        enabled=True,
        address="http://redis-address:8888",
        certificate_path="/etc/ssl/certs/ca-certificates.crt",
    ),
)
"""

    __doc_yaml_example__ = """\
name: my-service
applications:
  - import_path: main:app
ray_gcs_external_storage_config:
  enabled: true
  address: http://redis-address:8888
  certificate_path: /etc/ssl/certs/ca-certificates.crt
"""

    enabled: bool = field(
        default=True,
        metadata={
            "docstring": "Enable or disable external storage. When `False`, the other fields are ignored."
        },
    )

    def _validate_enabled(self, enabled: bool):
        if not isinstance(enabled, bool):
            raise TypeError("'enabled' must be a boolean.")

    address: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Address to connect to the redis instance at. Defaults to the cloud-wide configuration."
        },
    )

    def _validate_address(self, address: Optional[str]):
        if address is not None and not isinstance(address, str):
            raise TypeError("'address' must be a string.")

    certificate_path: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Path to the TLS certificate file to use for authentication when using secure connections. Defaults to the cloud-wide configuration."
        },
    )

    def _validate_certificate_path(self, certificate_path: Optional[str]):
        if certificate_path is not None and not isinstance(certificate_path, str):
            raise TypeError("'certificate_path' must be a string.")


@dataclass(frozen=True)
class TracingConfig(ModelBase):
    """Configuration options for tracing.

    Tracing provides a way to collect telemetry on requests that are
    handled by your Serve application. If tracing is enabled, you can instrument
    your Serve application and export traces to a specific backend by configuring the
    `exporter_import_path` option. This feature is currently experimental.
    """

    __doc_py_example__ = """\
from anyscale.service.models import ServiceConfig, TracingConfig

config = ServiceConfig(
    name="my-service",
    applications=[{"import_path": "main:app"}],
    tracing_config=TracingConfig(
        enabled=True,
        exporter_import_path="my_module:custom_tracing_exporter",
        sampling_ratio=1.0,
    ),
)
"""

    __doc_yaml_example__ = """\
name: my-service
applications:
  - import_path: main:app
tracing_config:
  enabled: true
  exporter_import_path: "my_module:custom_tracing_exporter"
  sampling_ratio: 1.0
"""

    enabled: bool = field(
        default=True,
        metadata={
            "docstring": (
                "Flag to enable tracing. "
                "When `True` and no `exporter_import_path` "
                "is defined, then the default tracing exporter will be "
                "used to export traces to serve logs. "
                "When `False`, the other fields are ignored."
            )
        },
    )

    def _validate_enabled(self, enabled: bool):
        if not isinstance(enabled, bool):
            raise TypeError("'enabled' must be a boolean.")

    exporter_import_path: Optional[str] = field(
        default=None,
        metadata={
            "docstring": (
                "Path to exporter function. Should be of the "
                'form "module.submodule_1...submodule_n.'
                'export_tracing". This is equivalent to '
                '"from module.submodule_1...submodule_n import export_tracing. '
                "Exporter function takes no arguments and returns List[SpanProcessor]."
            )
        },
    )

    def _validate_exporter_import_path(self, exporter_import_path: Optional[str]):
        if exporter_import_path is not None and not isinstance(
            exporter_import_path, str
        ):
            raise TypeError("'exporter_import_path' must be a string.")

    sampling_ratio: float = field(
        default=1.0,
        metadata={
            "docstring": "Ratio of traces to record and export. "
            "Reducing the ratio can help with minimizing serving latency and storage overhead. "
        },
    )

    def _validate_sampling_ratio(self, sampling_ratio: float):
        if sampling_ratio is not None and not isinstance(sampling_ratio, float):
            raise TypeError("'sampling_ratio' must be a float.")


@dataclass(frozen=True)
class ServiceConfig(WorkloadConfig):
    """Configuration options for a service."""

    __doc_py_example__ = """\
from anyscale.service.models import ServiceConfig

config = ServiceConfig(
    name="my-service",
    working_dir=".",
    applications=[{"import_path": "main:app"}],
    # An inline `ComputeConfig` can also be provided.
    compute_config="my-compute-config:1",
    # A containerfile path can also be provided.
    image_uri="anyscale/image/my-image:1",
)
"""

    __doc_yaml_example__ = """\
name: my-service
applications:
  - import_path: main:app
image_uri: anyscale/image/my-image:1 # (Optional) Exclusive with `containerfile`.
containerfile: /path/to/Dockerfile # (Optional) Exclusive with `image_uri`.
compute_config: my-compute-config:1 # (Optional) An inline dictionary can also be provided.
working_dir: /path/to/working_dir # (Optional) If a local directory is provided, it will be uploaded automatically.
excludes: # (Optional) List of files to exclude from being packaged up for the service.
  - .git
  - .env
  - .DS_Store
  - __pycache__
requirements: # (Optional) List of requirements files to install. Can also be a path to a requirements.txt.
  - emoji==1.2.0
  - numpy==1.19.5
env_vars: # (Optional) Dictionary of environment variables to set in the service.
  MY_ENV_VAR: my_value
  ANOTHER_ENV_VAR: another_value
py_modules: # (Optional) A list of local directories or remote URIs that will be added to the Python path.
  - /path/to/my_module
  - s3://my_bucket/my_module
cloud: anyscale-prod # (Optional) The name of the Anyscale Cloud.
project: my-project # (Optional) The name of the Anyscale Project.
query_auth_token_enabled: True # (Optional) Whether or not to gate this service behind an auth token.
http_options:
  request_timeout_s: 60 # (Optional) Timeout for HTTP requests in seconds. Default is no timeout.
  keep_alive_timeout_s: 60 # (Optional) Timeout for HTTP keep-alive connections in seconds. Default is 5 seconds.
grpc_options:
  port: 50051 # (Optional) Port to listen on for gRPC requests. Default is 9000.
  grpc_servicer_functions: # (Optional) List of functions to expose as gRPC servicers.
    - my_module:my_function
logging_config: # (Optional) Configuration options for logging.
  encoding: JSON # JSON or TEXT.
  log_level: INFO
  enable_access_log: true
tracing_config: # (Optional) Configuration options for tracing.
  enabled: true
  exporter_import_path: my_module:custom_tracing_exporter
  sampling_ratio: 1.0
tags:
  team: serving
  cost-center: eng
"""

    # Override the `name` field from `WorkloadConfig` so we can document it separately for jobs and services.
    name: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Unique name of the service. When running inside a workspace, this defaults to the name of the workspace, else it is required."
        },
    )

    applications: List[Dict[str, Any]] = field(
        default_factory=list,
        repr=False,
        metadata={
            "docstring": "List of Ray Serve applications to run. At least one application must be specified. For details, see the Ray Serve config file format documentation: https://docs.ray.io/en/latest/serve/production-guide/config.html"
        },
    )

    version_name: Optional[str] = field(
        default=None, metadata={"docstring": "Unique name of the version."},
    )

    def _validate_version_name(self, version_name: Optional[str]):
        # Allow None if optional; enforce non-empty string if required
        if version_name is None:
            return None
        return version_name

    def _validate_applications(self, applications: List[Dict[str, Any]]):
        if not isinstance(applications, list):
            raise TypeError("'applications' must be a list.")
        if len(applications) == 0:
            raise ValueError("'applications' cannot be empty.")

        # Validate import paths.
        for app in applications:
            import_path = app.get("import_path", None)
            if not import_path:
                raise ValueError("Every application must specify an import path.")

            if not isinstance(import_path, str):
                raise TypeError(f"'import_path' must be a string, got: {import_path}")

            if (
                import_path.count(":") != 1
                or import_path.rfind(":") in {0, len(import_path) - 1}
                or import_path.rfind(".") in {0, len(import_path) - 1}
            ):
                raise ValueError(
                    f"'import_path' must be of the form: 'module.optional_submodule:app', but got: '{import_path}'."
                )

    query_auth_token_enabled: bool = field(
        default=True,
        repr=False,
        metadata={
            "docstring": "Whether or not queries to this service is gated behind an authentication token. If `True`, an auth token is generated the first time the service is deployed. You can find the token in the UI or by fetching the status of the service."
        },
    )

    def _validate_query_auth_token_enabled(self, query_auth_token_enabled: bool):
        if not isinstance(query_auth_token_enabled, bool):
            raise TypeError("'query_auth_token_enabled' must be a boolean.")

    http_options: Optional[Dict[str, Any]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "HTTP options that will be passed to Ray Serve. See https://docs.ray.io/en/latest/serve/production-guide/config.html for supported options."
        },
    )

    def _validate_http_options(self, http_options: Optional[Dict[str, Any]]):
        """Validate the `http_options` field.

        This is passed through as part of the Ray Serve config, but some fields are
        disallowed (not valid when deploying Anyscale services).
        """
        if http_options is None:
            return
        elif not isinstance(http_options, dict):
            raise TypeError("'http_options' must be a dict.")

        banned_options = {"host", "port", "root_path"}
        banned_options_passed = {o for o in banned_options if o in http_options}
        if len(banned_options_passed) > 0:
            raise ValueError(
                "The following provided 'http_options' are not permitted "
                f"in Anyscale: {banned_options_passed}."
            )

    grpc_options: Optional[Dict[str, Any]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "gRPC options that will be passed to Ray Serve. See https://docs.ray.io/en/latest/serve/production-guide/config.html for supported options."
        },
    )

    def _validate_grpc_options(self, grpc_options: Optional[Dict[str, Any]]):
        """Validate the `grpc_options` field.

        This is passed through as part of the Ray Serve config, but some fields are
        disallowed (not valid when deploying Anyscale services).
        """
        if grpc_options is None:
            return
        elif not isinstance(grpc_options, dict):
            raise TypeError("'grpc_options' must be a dict.")

        banned_options = {
            "port",
        }
        banned_options_passed = {o for o in banned_options if o in grpc_options}
        if len(banned_options_passed) > 0:
            raise ValueError(
                "The following provided 'grpc_options' are not permitted "
                f"in Anyscale: {banned_options_passed}."
            )

    logging_config: Optional[Dict[str, Any]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Logging options that will be passed to Ray Serve. See https://docs.ray.io/en/latest/serve/production-guide/config.html for supported options."
        },
    )

    def _validate_logging_config(self, logging_config: Optional[Dict[str, Any]]):
        if logging_config is not None and not isinstance(logging_config, dict):
            raise TypeError("'logging_config' must be a dict.")

    ray_gcs_external_storage_config: Union[RayGCSExternalStorageConfig, Dict, None] = (
        field(
            default=None,
            repr=False,
            metadata={
                "docstring": "Configuration options for external storage for the Ray Global Control Store (GCS).",
                "customer_hosted_only": True,
            },
        )
    )

    def _validate_ray_gcs_external_storage_config(
        self,
        ray_gcs_external_storage_config: Union[RayGCSExternalStorageConfig, Dict, None],
    ) -> Optional[RayGCSExternalStorageConfig]:
        if ray_gcs_external_storage_config is None:
            return None

        if isinstance(ray_gcs_external_storage_config, dict):
            ray_gcs_external_storage_config = RayGCSExternalStorageConfig.from_dict(
                ray_gcs_external_storage_config
            )

        if not isinstance(ray_gcs_external_storage_config, RayGCSExternalStorageConfig):
            raise TypeError(
                "'ray_gcs_external_storage_config' must be a RayGCSExternalStorageConfig or corresponding dict."
            )

        return ray_gcs_external_storage_config

    tracing_config: Union[TracingConfig, Dict, None] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Configuration options for tracing.",
            "customer_hosted_only": True,
        },
    )

    def _validate_tracing_config(
        self, tracing_config: Union[TracingConfig, Dict, None],
    ) -> Optional[TracingConfig]:
        if tracing_config is None:
            return None

        if isinstance(tracing_config, dict):
            tracing_config = TracingConfig.from_dict(tracing_config)

        if not isinstance(tracing_config, TracingConfig):
            raise TypeError(
                "'tracing_config' must be a TracingConfig or corresponding dict."
            )

        return tracing_config

    tags: Optional[Dict[str, str]] = field(
        default=None, metadata={"docstring": "Tags to associate with the service."},
    )

    def _validate_tags(self, tags: Optional[Dict[str, str]]):
        if tags is None:
            return
        if not isinstance(tags, dict):
            raise TypeError("'tags' must be a Dict[str, str].")
        for k, v in tags.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError("'tags' must be a Dict[str, str].")


class ServiceState(ModelEnum):
    """Current state of a service."""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    # TODO(edoakes): UPDATING comes up while rolling out and rolling back.
    # This is very unexpected from a customer's point of view, we should fix it.
    UPDATING = "UPDATING"
    ROLLING_OUT = "ROLLING_OUT"
    ROLLING_BACK = "ROLLING_BACK"
    UNHEALTHY = "UNHEALTHY"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"
    UNKNOWN = "UNKNOWN"
    SYSTEM_FAILURE = "SYSTEM_FAILURE"

    __docstrings__: ClassVar[Dict[str, str]] = {
        STARTING: "The service is starting up.",
        RUNNING: "The service is running and healthy.",
        UPDATING: "One of the service versions or the networking stack is updating.",
        ROLLING_OUT: "The service is rolling out to a new canary version.",
        ROLLING_BACK: "The service is rolling back to the primary version.",
        UNHEALTHY: (
            "The service has become unhealthy due to an unexpected issue "
            "such as a node failure. In most cases these issues should "
            "automatically recover."
        ),
        TERMINATING: "The service is terminating.",
        TERMINATED: "The service is terminated.",
        UNKNOWN: (
            "The CLI/SDK received an unexpected state from the API server. "
            "In most cases, this means you need to update the CLI."
        ),
        SYSTEM_FAILURE: (
            "An unexpected error occurred in the Anyscale stack. "
            "Please reach out to customer support immediately."
        ),
    }


class ServiceVersionState(ModelEnum):
    """Current state of a service version."""

    STARTING = "STARTING"
    UPDATING = "UPDATING"
    RUNNING = "RUNNING"
    UNHEALTHY = "UNHEALTHY"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"
    UNKNOWN = "UNKNOWN"
    SYSTEM_FAILURE = "SYSTEM_FAILURE"

    __docstrings__: ClassVar[Dict[str, str]] = {
        STARTING: "The service version is starting up.",
        RUNNING: "The service version is running and healthy.",
        UPDATING: (
            "The application(s) on the service version are updating "
            "due to a config deploy or update."
        ),
        UNHEALTHY: (
            "The service version has become unhealthy due to an unexpected "
            "issue such as a node failure. In most cases these issues should "
            "automatically recover."
        ),
        TERMINATING: "The service version is terminating.",
        TERMINATED: "The service version is terminated.",
        UNKNOWN: (
            "The CLI/SDK received an unexpected state from the API server. "
            "In most cases, this means you need to update the CLI."
        ),
        SYSTEM_FAILURE: (
            "An unexpected error occurred in the Anyscale stack. "
            "Please reach out to customer support immediately."
        ),
    }


@dataclass(frozen=True)
class ServiceVersionStatus(ModelBase):
    """Current status of a single version of a running service."""

    __doc_py_example__ = """\
import anyscale
from anyscale.service.models import ServiceStatus, ServiceVersionStatus

status: ServiceStatus = anyscale.service.status("my-service")
primary_version_status: ServiceVersionStatus = status.primary_version
"""

    __doc_cli_example__ = """\
$ anyscale service status -n my-service
name: my-service
id: service2_uz6l8yhy2as5wrer3shzj6kh67
state: RUNNING
primary_version:
  id: 601bd56c4b
  name: v1
  state: RUNNING
  weight: 100
  created_at: 2025-04-18 17:21:28.323174+00:00
"""

    id: str = field(
        metadata={
            "docstring": "Unique ID of the service _version_ (*not* the service)."
        }
    )

    def _validate_id(self, id: str):  # noqa: A002
        if not isinstance(id, str):
            raise TypeError("'id' must be a string.")

    name: str = field(
        metadata={"docstring": "Human-readable name of the service version."}
    )

    def _validate_name(self, name: str):
        if name is None:
            return None
        return name

    state: Union[ServiceVersionState, str] = field(
        metadata={"docstring": "Current state of the service version."}
    )

    def _validate_state(
        self, state: Union[ServiceVersionState, str]
    ) -> ServiceVersionState:
        if isinstance(state, str):
            # This will raise a ValueError if the state is unrecognized.
            state = ServiceVersionState(state)
        elif not isinstance(state, ServiceVersionState):
            raise TypeError("'state' must be a ServiceVersionState.")

        return state

    weight: int = field(
        metadata={
            "docstring": "Current canary weight of the service version. This is `100` if there is no ongoing rollout."
        }
    )

    def _validate_weight(self, weight: int):
        if not isinstance(weight, int):
            raise TypeError("'weight' must be an int.")

    created_at: datetime = field(
        metadata={"docstring": "Creation time of the service version."},
    )

    def _validate_created_at(self, created_at: datetime):
        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime.")

    config: Union[ServiceConfig, Dict] = field(
        repr=False, metadata={"docstring": "Configuration of this service version."}
    )

    def _validate_config(self, config: Union[ServiceConfig, Dict]) -> ServiceConfig:
        if isinstance(config, dict):
            config = ServiceConfig.from_dict(config)

        if not isinstance(config, ServiceConfig):
            raise TypeError("'config' must be a ServiceConfig or corresponding dict.")

        return config


@dataclass(frozen=True)
class ServiceStatus(ModelBase):
    """Current status of a service."""

    __doc_py_example__ = """\
import anyscale
from anyscale.service.models import ServiceStatus

status: ServiceStatus = anyscale.service.status("my-service")
"""

    __doc_cli_example__ = """\
$ anyscale service status -n my-service
name: my-service
id: service2_uz6l8yhy2as5wrer3shzj6kh67
state: RUNNING
primary_version:
  id: 601bd56c4b
  state: RUNNING
  weight: 100
"""

    name: str = field(metadata={"docstring": "Unique name of the service."})

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")

    id: str = field(
        metadata={
            "docstring": "Unique ID of the service (generated when the service is first deployed)."
        }
    )

    def _validate_id(self, id: str):  # noqa: A002
        if not isinstance(id, str):
            raise TypeError("'id' must be a string.")

    state: Union[ServiceState, str] = field(
        metadata={"docstring": "Current state of the service."}
    )

    def _validate_state(self, state: Union[ServiceState, str]) -> ServiceState:
        if isinstance(state, str):
            # This will raise a ValueError if the state is unrecognized.
            state = ServiceState(state)
        elif not isinstance(state, ServiceState):
            raise TypeError("'state' must be a ServiceState.")

        return state

    query_url: str = field(
        repr=False, metadata={"docstring": "URL used to query the service."}
    )

    def _validate_query_url(self, query_url: str):
        if not isinstance(query_url, str):
            raise TypeError("'query_url' must be a string.")

    creator: Optional[str] = field(
        default=None,
        metadata={"docstring": "Email of the user or entity that created the service."},
    )

    def _validate_creator(self, creator: Optional[str]):
        if creator is not None and not isinstance(creator, str):
            raise TypeError("'creator' must be a string.")

    query_auth_token: Optional[str] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Query auth token required to query the service (if any). Must be passed in as a header like: `Authorization: Bearer <token>`."
        },
    )

    def _validate_query_auth_token(self, query_auth_token: Optional[str]):
        if query_auth_token is not None and not isinstance(query_auth_token, str):
            raise TypeError("'query_auth_token' must be a string.")

    primary_version: Union[ServiceVersionStatus, Dict, None] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Primary version of the service. During a rollout, this is the service version that was running prior to the deploy."
        },
    )

    def _validate_primary_version(
        self, primary_version: Union[ServiceVersionStatus, Dict, None]
    ) -> Optional[ServiceVersionStatus]:
        if primary_version is None:
            return None

        if isinstance(primary_version, dict):
            primary_version = ServiceVersionStatus.from_dict(primary_version)

        if not isinstance(primary_version, ServiceVersionStatus):
            raise TypeError(
                "'primary_version' must be a ServiceVersionStatus or corresponding dict."
            )

        return primary_version

    canary_version: Union[ServiceVersionStatus, Dict, None] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Canary version of the service (the one that is being deployed). When there is no rollout ongoing, this will be `None`."
        },
    )

    def _validate_canary_version(
        self, canary_version: Union[ServiceVersionStatus, Dict, None]
    ) -> Optional[ServiceVersionStatus]:
        if canary_version is None:
            return None

        if isinstance(canary_version, dict):
            canary_version = ServiceVersionStatus.from_dict(canary_version)

        if not isinstance(canary_version, ServiceVersionStatus):
            raise TypeError(
                "'canary_version' must be a ServiceVersionStatus or corresponding dict."
            )

        return canary_version

    project: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Name of the project that this service version belongs to."
        },
    )

    def _validate_project(self, project: Optional[str]):
        if project is not None and not isinstance(project, str):
            raise TypeError("project must be a string.")

    versions: Optional[List[ServiceVersionStatus]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "All active versions of the service. For multi-version services, this contains all versions with their traffic weights."
        },
    )

    def _validate_versions(
        self, versions: Optional[List[ServiceVersionStatus]]
    ) -> Optional[List[ServiceVersionStatus]]:
        if versions is None:
            return None
        if not isinstance(versions, list):
            raise TypeError("'versions' must be a list.")
        for v in versions:
            if not isinstance(v, ServiceVersionStatus):
                raise TypeError("'versions' must be a list of ServiceVersionStatus.")
        return versions


class ServiceSortField(ModelEnum):
    """Fields available for sorting services."""

    STATUS = "STATUS"
    NAME = "NAME"
    CREATED_AT = "CREATED_AT"

    __docstrings__: ClassVar[Dict[str, str]] = {
        STATUS: "Sort by service status (active first by default).",
        NAME: "Sort by service name.",
        CREATED_AT: "Sort by creation timestamp.",
    }


class ServiceLogMode(ModelEnum):
    """Mode to use for getting job logs."""

    HEAD = "HEAD"
    TAIL = "TAIL"

    __docstrings__: ClassVar[Dict[str, str]] = {
        HEAD: "Fetch logs from the start.",
        TAIL: "Fetch logs from the end.",
    }


@dataclass(frozen=True)
class ServiceModel(ModelBase):
    """A model for a service."""

    id: str = field(metadata={"docstring": "Unique ID of the service."})
    name: str = field(metadata={"docstring": "Name of the service."})
    state: ServiceState = field(metadata={"docstring": "Current state of the service."})


class ServiceSortOrder(ModelEnum):
    """Enum for sort order directions."""

    ASC = "ASC"
    DESC = "DESC"

    __docstrings__: ClassVar[Dict[str, str]] = {
        ASC: "Sort in ascending order.",
        DESC: "Sort in descending order.",
    }
