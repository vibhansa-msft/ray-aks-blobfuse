import os
from typing import Any, Dict, List, Optional

import click
import yaml

from anyscale import AnyscaleSDK
from anyscale.cli_logger import BlockLogger

# There are two API clients: one that's exposed to customers via the SDK (ExternalApi)
# and one that's only used internally by the UI and CLI (InternalApi).
from anyscale.client.openapi_client.api.default_api import DefaultApi as InternalApi
from anyscale.client.openapi_client.models.decorated_list_service_api_model import (
    DecoratedListServiceAPIModel,
)
from anyscale.controllers.base_controller import BaseController
from anyscale.models.service_model import ServiceConfig
from anyscale.project_utils import infer_project_id
from anyscale.sdk.anyscale_client.models import (
    ApplyServiceModel,
    RollbackServiceModel,
    ServiceModel,
)
from anyscale.tables import ServicesTable
from anyscale.util import (
    get_endpoint,
    populate_unspecified_cluster_configs_from_current_workspace,
)
from anyscale.utils.runtime_env import override_runtime_env_config
from anyscale.utils.workload_types import Workload
from anyscale.utils.workspace_notification import (
    send_workspace_notification,
    WorkspaceNotification,
    WorkspaceNotificationAction,
)


class ServiceController(BaseController):
    def __init__(self, *, sdk: Optional[AnyscaleSDK] = None):
        super().__init__()
        self.log = BlockLogger()
        self.sdk = sdk if sdk is not None else AnyscaleSDK()

    @property
    def internal_api_client(self) -> InternalApi:
        return self.api_client

    def _format_apply_service_model(self, config: ServiceConfig) -> ApplyServiceModel:
        if not config.ray_serve_config:
            raise click.ClickException(
                "ray_serve_config must be provided in the service configuration."
            )

        if "import_path" in config.ray_serve_config:
            self.log.warning(
                "The single application Ray Serve config is deprecated in Ray 2.8 and "
                "removed in Ray 2.9. Please migrate to the multi-application Ray Serve "
                "config format. Please see "
                "https://docs.ray.io/en/latest/serve/multi-app.html#migrating-from-a-single-application-config "
                "for migration instructions."
            )

        return ApplyServiceModel(
            name=config.name,
            description=config.description or "Service updated from CLI",
            project_id=config.project_id,
            version=config.version,
            canary_percent=config.canary_percent,
            ray_serve_config=config.ray_serve_config,
            ray_gcs_external_storage_config=config.ray_gcs_external_storage_config,
            tracing_config=config.tracing_config,
            build_id=config.build_id,
            compute_config_id=config.compute_config_id,
            rollout_strategy=config.rollout_strategy,
            config=config.config,
            auto_complete_rollout=config.auto_complete_rollout,
            max_surge_percent=config.max_surge_percent,
        )

    def get_authenticated_user_id(self) -> str:
        user_info_response = (
            self.internal_api_client.get_user_info_api_v2_userinfo_get()
        )
        return user_info_response.result.id

    def _get_services_by_name(
        self,
        *,
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        created_by_me: bool,
        max_items: int,
    ) -> List[DecoratedListServiceAPIModel]:
        """Makes an API call to get all services matching the provided filters.

        Note that "name" is an *exact match* (different from the REST API semantics).
        """
        creator_id = self.get_authenticated_user_id() if created_by_me else None

        paging_token = None
        services_list: List[DecoratedListServiceAPIModel] = []
        while len(services_list) < max_items:
            resp = self.internal_api_client.list_services_api_v2_services_v2_get(
                project_id=project_id,
                name=name,
                creator_id=creator_id,
                count=10,
                paging_token=paging_token,
            )
            for r in resp.results:
                # The 'name' filter in the list endpoint is not exact match.
                if name is None or r.name == name:
                    services_list.append(r)
                if len(services_list) >= max_items:
                    break

            paging_token = resp.metadata.next_paging_token
            if paging_token is None:
                break

        return services_list[:max_items]

    def _get_service_id_from_name(
        self, service_name: str, project_id: Optional[str]
    ) -> str:
        """Get the ID for a service by name.

        If project_id is specified, filter to that project, else don't filter on project_id and
        instead error if there are multiple services with the name.

        Raises an exception if there are zero or multiple services with the given name.
        """
        results = self._get_services_by_name(
            name=service_name, project_id=project_id, created_by_me=False, max_items=10
        )

        if len(results) == 0:
            raise click.ClickException(
                f"No service with name '{service_name}' was found. "
                "Please verify that this service exists and you have access to it."
            )
        elif len(results) > 1:
            raise click.ClickException(
                f"There are multiple services with name '{service_name}'. "
                "Please filter using --project-id or specify the --service-id instead. "
                f"Services found: \n{ServicesTable(results)}"
            )

        return results[0].id

    def get_service_id(
        self,
        *,
        service_id: Optional[str] = None,
        service_name: Optional[str] = None,
        service_config_file: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """Get the service ID given the ID, name, or config file.

        This is a utility used by multiple CLI commands to standardize mapping these options
        to a service_id.

        The precedence is: service_id > service_name > service_config_file.

        If the project_id is specified directly or via the service_config_file it will be used
        to filter the query. Else we will try to find the service across all projects and error
        if there are multiple.
        """
        if service_id is not None:
            if service_name is not None or service_config_file is not None:
                raise click.ClickException(
                    "Only one of service ID, name, or config file should be specified."
                )
        elif service_name is not None:
            if service_config_file is not None:
                raise click.ClickException(
                    "Only one of service ID, name, or config file should be specified."
                )
            service_id = self._get_service_id_from_name(service_name, project_id)
        elif service_config_file is not None:
            service_config: ServiceConfig = self.parse_service_config_dict(
                self.read_service_config_file(service_config_file)
            )
            # Allow the passed project_id to override the one in the file.
            project_id = project_id or service_config.project_id
            service_id = self._get_service_id_from_name(service_config.name, project_id)
        else:
            raise click.ClickException(
                "Service ID, name, or config file must be specified."
            )

        return service_id

    def override_config_options(
        self,
        service_config: ServiceConfig,
        *,
        auto_complete_rollout: bool,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        max_surge_percent: Optional[int] = None,
        canary_percent: Optional[int] = None,
        rollout_strategy: Optional[str] = None,
    ) -> ServiceConfig:
        """
        Given the input config and user overrides, this method
        constructs the expected service configuration.

        Note that if the command is run from a workspace, some values
        may be overwritten.
        """
        if name:
            service_config.name = name

        if description:
            service_config.description = description

        if version:
            service_config.version = version

        if canary_percent is not None:
            service_config.canary_percent = canary_percent

        if rollout_strategy:
            service_config.rollout_strategy = rollout_strategy

        if service_config.rollout_strategy:
            service_config.rollout_strategy = service_config.rollout_strategy.upper()

        if auto_complete_rollout is not None:
            service_config.auto_complete_rollout = auto_complete_rollout

        if max_surge_percent is not None:
            service_config.max_surge_percent = max_surge_percent

        if (
            service_config.max_surge_percent is not None
            and service_config.rollout_strategy == "IN_PLACE"
        ):
            raise ValueError(
                "--max-surge-percent is not supported for IN_PLACE rollouts."
            )

        return service_config

    def read_service_config_file(self, service_config_file: str) -> Dict[str, Any]:
        if not os.path.exists(service_config_file):
            raise click.ClickException(f"Config file {service_config_file} not found.")

        with open(service_config_file) as f:
            return yaml.safe_load(f)

    def parse_service_config_dict(self, config_dict: Dict[str, Any]) -> ServiceConfig:
        # If running in a workspace, auto-populate unspecified fields.
        config_dict = populate_unspecified_cluster_configs_from_current_workspace(
            config_dict, self.sdk, populate_name=True
        )
        return ServiceConfig.parse_obj(config_dict)

    def rollout(  # noqa: PLR0913
        self,
        config_dict: Dict[str, Any],
        *,
        auto_complete_rollout: bool,
        name: Optional[str] = None,
        version: Optional[str] = None,
        max_surge_percent: Optional[int] = None,
        canary_percent: Optional[int] = None,
        rollout_strategy: Optional[str] = None,
    ):
        """
        Deploys a Service 2.0.
        """
        config = self.override_config_options(
            self.parse_service_config_dict(config_dict),
            name=name,
            version=version,
            max_surge_percent=max_surge_percent,
            canary_percent=canary_percent,
            rollout_strategy=rollout_strategy,
            auto_complete_rollout=auto_complete_rollout,
        )

        config.project_id = infer_project_id(
            self.sdk,
            self.internal_api_client,
            self.log,
            project_id=config.project_id,
            cluster_compute_id=config.compute_config_id,
            cluster_compute=config.compute_config,
            cloud=config.cloud,
        )

        service_config = config.config
        if service_config:
            access_config = service_config.get("access", {})

            if (
                isinstance(access_config, dict)
                and access_config.get("use_bearer_token", True) is False
            ):
                self.log.warning(
                    "Using bearer token for authorization is disabled: no authorization guarding will be used for this service."
                )

        if not config.ray_serve_config:
            config.ray_serve_config = {}

        self._overwrite_runtime_env_in_v2_ray_serve_config(config)

        apply_service_model: ApplyServiceModel = self._format_apply_service_model(
            config
        )
        service: ServiceModel = self.sdk.rollout_service(apply_service_model).result
        self.log.info(f"Service {service.id} rollout initiated.")
        self.log.info(
            f'View the service in the UI at {get_endpoint(f"/services/{service.id}")}'
        )

        url = service.base_url
        auth_token = service.auth_token
        self.log.info(
            "You can query the service endpoint using the curl request below:"
        )
        self.log.info(f"curl -H 'Authorization: Bearer {auth_token}' {url}")

        send_workspace_notification(
            self.sdk,
            WorkspaceNotification(
                body=f"Service {service.name} rollout initiated.",
                action=WorkspaceNotificationAction(
                    type="navigate-service", title="View Service", value=service.id,
                ),
            ),
        )

    def _overwrite_runtime_env_in_v2_ray_serve_config(self, config: ServiceConfig):
        """Modifies config in place."""
        ray_serve_config = config.ray_serve_config
        if ray_serve_config is not None and "applications" in ray_serve_config:
            for ray_serve_app_config in ray_serve_config["applications"]:
                ray_serve_app_config["runtime_env"] = override_runtime_env_config(
                    runtime_env=ray_serve_app_config.get("runtime_env"),
                    anyscale_api_client=self.sdk,
                    api_client=self.internal_api_client,
                    workload_type=Workload.SERVICES,
                    compute_config_id=config.compute_config_id,
                    log=self.log,
                )

        else:
            assert ray_serve_config is not None
            ray_serve_config["runtime_env"] = override_runtime_env_config(
                runtime_env=ray_serve_config.get("runtime_env"),
                anyscale_api_client=self.sdk,
                api_client=self.internal_api_client,
                workload_type=Workload.SERVICES,
                compute_config_id=config.compute_config_id,
                log=self.log,
            )

    def rollback(self, service_id: str, max_surge_percent: Optional[int] = None):
        service: ServiceModel = self.sdk.rollback_service(
            service_id,
            rollback_service_model=RollbackServiceModel(
                max_surge_percent=max_surge_percent
            ),
        ).result

        self.log.info(f"Service {service.id} rollback initiated.")
        self.log.info(
            f'View the service in the UI at {get_endpoint(f"/services/{service.id}")}'
        )
