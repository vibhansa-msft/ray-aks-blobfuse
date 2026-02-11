import enum
from typing import Any, Dict, Optional, Union

from anyscale.anyscale_pydantic import Field, root_validator
from anyscale.models.job_model import (
    _validate_and_modify_runtime_env,
    BaseHAJobConfig,
)
from anyscale.service.models import TracingConfig


class UserServiceAccessTypes(str, enum.Enum):
    private = "private"
    public = "public"


class ServiceConfig(BaseHAJobConfig):
    name: str = Field(
        ..., description="Name of service to be submitted.",
    )
    access: UserServiceAccessTypes = Field(
        UserServiceAccessTypes.public,
        description=(
            "Whether user service (eg: serve deployment) can be accessed by public "
            "internet traffic. If public, a user service endpoint can be queried from "
            "the public internet with the provided authentication token. "
            "If private, the user service endpoint can only be queried from within "
            "the same Anyscale cloud and will not require an authentication token."
        ),
    )

    ray_serve_config: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "The Ray Serve config to use for this Production service. It is supported only on v2 clouds. "
            "This config defines your Ray Serve application, and will be passed directly to Ray Serve. "
            "You can learn more about Ray Serve config files here: https://docs.ray.io/en/latest/serve/production-guide/config.html"
        ),
    )

    tracing_config: Union[None, Dict[str, Any], TracingConfig] = Field(
        None,
        description=(
            "Config to enable collecting and exporting traces. "
            "If tracing is enabled, you can instrument "
            "your Serve application and export traces to a specific backend by configuring the "
            "`exporter_import_path` option. This feature is currently experimental."
        ),
    )

    ray_gcs_external_storage_config: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Address to connect to external storage at. "
            "Must be accessible from instances running in the provided cloud. "
            "This is only supported for v2 services."
        ),
    )

    # Version level fields for Services V2
    version: Optional[str] = Field(
        None,
        description="A version string that represents the version for this service. "
        "Will be populated with the hash of the config if not specified.",
    )
    canary_percent: Optional[int] = Field(
        None,
        description="A manual target weight for this service. "
        "If this field is not set, the service will automatically roll out. "
        "If set, this should be a number between 0 and 100 (inclusive). "
        "The newly created version will have weight `canary_percent` "
        "and the existing version will have `100 - canary_percent`.",
    )

    rollout_strategy: Optional[str] = Field(
        None,
        description="Strategy for rollout. "
        "The ROLLOUT strategy will deploy your Ray Serve configuration onto a newly started cluster, and then shift traffic over to the new cluster. "
        "You can manually control the speed of the rollout using the canary_weight configuration.\n"
        "The IN_PLACE strategy will use Ray Serve in place upgrade to update your existing cluster in place. "
        "When using this rollout strategy, you may only change the ray_serve_config field. "
        "You cannot partially shift traffic or rollback an in place upgrade. "
        "In place upgrades are faster and riskier than rollouts, and we recommend only using them for relatively safe changes (for example, increasing the number of replicas on a Ray Serve deployment).\n"
        "Default strategy is ROLLOUT.",
    )

    config: Optional[Dict[str, Any]] = Field(
        None, description="Target Service's configuration",
    )

    auto_complete_rollout: Optional[bool] = Field(
        None,
        description="Flag to indicate whether or not to complete the rollout after the canary version reaches 100%.",
    )

    max_surge_percent: Optional[int] = Field(
        None,
        description="Max amount of excess capacity allocated during the rollout (0-100).",
    )

    @root_validator
    def validates_config(cls, values) -> Dict[str, Any]:
        assert (
            values.get("runtime_env") is None
        ), "runtime_env should not be set for Services on v2 clouds."

        return values

    @root_validator
    def validate_runtime_env_v2(cls: Any, values: Any) -> Any:  # noqa: PLR0912
        ray_serve_config = values.get("ray_serve_config")
        if ray_serve_config is None:
            return values

        if "applications" in ray_serve_config:
            for ray_serve_app_config in ray_serve_config["applications"]:
                runtime_env = ray_serve_app_config.get("runtime_env")
                if runtime_env is not None:
                    _validate_and_modify_runtime_env(runtime_env)
        else:
            runtime_env = ray_serve_config.get("runtime_env")
            if runtime_env is not None:
                _validate_and_modify_runtime_env(runtime_env)

        return values
