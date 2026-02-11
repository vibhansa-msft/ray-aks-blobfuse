from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.client.openapi_client.models import (
    CloudDeploymentComputeConfig,
    ComputeNodeType,
    ComputeTemplateConfig,
    DecoratedComputeTemplate,
    DecoratedComputeTemplateConfig,
    PhysicalResources,
    Resources,
    WorkerNodeType,
)
from anyscale.cluster_compute import parse_cluster_compute_name_version
from anyscale.compute_config.models import (
    CloudDeployment,
    ComputeConfig,
    ComputeConfigListResult,
    ComputeConfigType,
    ComputeConfigVersion,
    HeadNodeConfig,
    MarketType,
    MultiResourceComputeConfig,
    PhysicalResources as UserPhysicalResources,
    WorkerNodeGroupConfig,
)
from anyscale.sdk.anyscale_client.models import ClusterComputeConfig


# Used to explicitly make the head node unschedulable.
# We can't leave resources empty because the backend will fill in CPU and GPU
# to match the instance type hardware.
UNSCHEDULABLE_RESOURCES = Resources(cpu=0, gpu=0)

# Label key for accelerator type (used by Ray for GPU/TPU scheduling)
RAY_ACCELERATOR_TYPE_LABEL = "ray.io/accelerator-type"


def _validate_no_tpu_on_vm_stack(
    node_config: Union[HeadNodeConfig, WorkerNodeGroupConfig],
    compute_stack: str,
    node_name: str,
) -> None:
    """Raise error if TPU resources are specified on a VM stack.

    TPU free pod shapes are only supported on Kubernetes (GKE) stacks.
    For VM stacks, users must use GPU or specify explicit instance types.

    Args:
        node_config: The head or worker node configuration to validate.
        compute_stack: The compute stack type ("VM" or "KUBERNETES").
        node_name: Name of the node group for error messages.

    Raises:
        ValueError: If TPU resources are specified on a VM stack.
    """
    if compute_stack != "VM":
        return

    pr = node_config.required_resources
    if pr is None:
        return

    # Check for TPU count or hosts in required_resources
    has_tpu = (pr.TPU is not None and pr.TPU > 0) or (
        pr.tpu_hosts is not None and pr.tpu_hosts > 0
    )

    # Check for TPU accelerator type in accelerator field or labels
    accel = pr.accelerator or ""
    if not accel:
        if node_config.required_labels:
            accel = node_config.required_labels.get(RAY_ACCELERATOR_TYPE_LABEL, "")
        if not accel and node_config.labels:
            accel = node_config.labels.get(RAY_ACCELERATOR_TYPE_LABEL, "")

    has_tpu_accel = accel.upper().startswith("TPU") if accel else False

    if has_tpu or has_tpu_accel:
        raise ValueError(
            f"Node group '{node_name}': TPU free pod shapes are not supported on VM stacks. "
            f"For TPU workloads, please use a Kubernetes (GKE) cloud deployment, or switch to GPU resources."
        )


class PrivateComputeConfigSDK(BaseSDK):
    def _convert_resource_dict_to_api_model(
        self, resource_dict: Optional[Dict[str, float]]
    ) -> Optional[Resources]:
        if resource_dict is None:
            return None

        resource_dict = deepcopy(resource_dict)
        return Resources(
            cpu=resource_dict.pop("CPU", None),
            gpu=resource_dict.pop("GPU", None),
            memory=resource_dict.pop("memory", None),
            object_store_memory=resource_dict.pop("object_store_memory", None),
            custom_resources=resource_dict or None,
        )

    def _validate_tpu_on_vm_stack(
        self, cloud_id: str, compute_config: ComputeConfig
    ) -> None:
        """Validate that TPU resources are not used on VM stacks.

        TPU free pod shapes are only supported on Kubernetes (GKE) cloud deployments.
        This provides early validation before making backend API calls.

        Args:
            cloud_id: The cloud ID to look up the cloud resource.
            compute_config: The compute config to validate.

        Raises:
            ValueError: If TPU resources are specified on a VM stack.
        """
        # Only validate if cloud_resource is specified (free pod shapes require this)
        if not compute_config.cloud_resource:
            return

        # Look up the cloud resource to get its compute stack type
        cloud_resource = self._client.get_cloud_resource_by_name(
            cloud_id=cloud_id, cloud_resource_name=compute_config.cloud_resource
        )
        if not cloud_resource or not cloud_resource.compute_stack:
            return

        compute_stack = str(cloud_resource.compute_stack).upper()

        # Validate head node
        if compute_config.head_node:
            _validate_no_tpu_on_vm_stack(
                compute_config.head_node, compute_stack, "head_node"
            )

        # Validate worker nodes
        if compute_config.worker_nodes:
            for worker in compute_config.worker_nodes:
                node_name = worker.name or worker.instance_type or "worker"
                _validate_no_tpu_on_vm_stack(worker, compute_stack, node_name)

    def _convert_head_node_config_to_api_model(
        self,
        config: Union[None, Dict, HeadNodeConfig],
        *,
        cloud_id: str,
        cloud_resource_name: Optional[str] = None,
        schedulable_by_default: bool,
    ) -> ComputeNodeType:
        if config is None:
            cloud_resource_id = None
            if cloud_resource_name:
                cloud_resource = self._client.get_cloud_resource_by_name(
                    cloud_id=cloud_id, cloud_resource_name=cloud_resource_name
                )
                if cloud_resource:
                    cloud_resource_id = cloud_resource.cloud_resource_id

            # If no head node config is provided, use the cloud/cloud resource default.
            default: ClusterComputeConfig = self._client.get_default_compute_config(
                cloud_id=cloud_id, cloud_resource_id=cloud_resource_id,
            ).config

            api_model = ComputeNodeType(
                name="head-node",
                instance_type=default.head_node_type.instance_type,
                # Let the backend populate the physical resources
                # (regardless of what the default compute config says).
                resources=None if schedulable_by_default else UNSCHEDULABLE_RESOURCES,
                flags=default.flags,
            )
        else:
            # Make mypy happy.
            assert isinstance(config, HeadNodeConfig)

            flags: Dict[str, Any] = deepcopy(config.flags) if config.flags else {}
            if config.cloud_deployment:
                assert isinstance(config.cloud_deployment, CloudDeployment)
                flags["cloud_deployment"] = config.cloud_deployment.to_dict()

            # Convert required_resources to API model (OpenAPI PhysicalResources object)
            # Only pass non-None values to avoid sending null fields to backend
            required_resources_api = None
            if config.required_resources is not None:
                pr_dict = config.required_resources.to_dict(for_api=True)
                # Copy only the fields that are present in the dict
                # To add support for new fields, just add them to this list
                pr_kwargs = {
                    field: pr_dict[field]
                    for field in [
                        "cpu",
                        "gpu",
                        "memory",
                        "accelerator",
                        "tpu",
                        "cpu_architecture",
                    ]
                    if field in pr_dict
                }
                # Handle tpu_hosts which maps to anyscale_tpu_hosts in the API
                if "tpu_hosts" in pr_dict:
                    pr_kwargs["anyscale_tpu_hosts"] = pr_dict["tpu_hosts"]
                required_resources_api = PhysicalResources(**pr_kwargs)

            api_model = ComputeNodeType(
                name="head-node",
                instance_type=config.instance_type,
                resources=self._convert_resource_dict_to_api_model(config.resources)
                if config.resources is not None or schedulable_by_default
                else UNSCHEDULABLE_RESOURCES,
                required_resources=required_resources_api,
                labels=config.labels,
                required_labels=config.required_labels,
                flags=flags or None,
                advanced_configurations_json=config.advanced_instance_config or None,
            )

        return api_model

    def _convert_worker_node_group_configs_to_api_models(
        self, configs: Optional[List[Union[Dict, WorkerNodeGroupConfig]]],
    ) -> Optional[List[WorkerNodeType]]:
        if configs is None:
            return None

        api_models = []
        for config in configs:
            # Make mypy happy.
            assert isinstance(config, WorkerNodeGroupConfig)

            flags: Dict[str, Any] = deepcopy(config.flags) if config.flags else {}
            if config.cloud_deployment:
                assert isinstance(config.cloud_deployment, CloudDeployment)
                flags["cloud_deployment"] = config.cloud_deployment.to_dict()

            # Convert required_resources to API model (OpenAPI PhysicalResources object)
            # Only pass non-None values to avoid sending null fields to backend
            required_resources_api = None
            if config.required_resources is not None:
                pr_dict = config.required_resources.to_dict(for_api=True)
                # Copy only the fields that are present in the dict
                # To add support for new fields, just add them to this list
                pr_kwargs = {
                    field: pr_dict[field]
                    for field in [
                        "cpu",
                        "gpu",
                        "memory",
                        "accelerator",
                        "tpu",
                        "cpu_architecture",
                    ]
                    if field in pr_dict
                }
                # Handle tpu_hosts which maps to anyscale_tpu_hosts in the API
                if "tpu_hosts" in pr_dict:
                    pr_kwargs["anyscale_tpu_hosts"] = pr_dict["tpu_hosts"]
                required_resources_api = PhysicalResources(**pr_kwargs)

            api_model = WorkerNodeType(
                name=config.name,
                instance_type=config.instance_type,
                resources=self._convert_resource_dict_to_api_model(config.resources),
                required_resources=required_resources_api,
                labels=config.labels,
                required_labels=config.required_labels,
                min_workers=config.min_nodes,
                max_workers=config.max_nodes,
                use_spot=config.market_type
                in {MarketType.SPOT, MarketType.PREFER_SPOT},
                fallback_to_ondemand=config.market_type == MarketType.PREFER_SPOT,
                flags=flags or None,
                advanced_configurations_json=config.advanced_instance_config or None,
            )
            api_models.append(api_model)

        return api_models

    def _convert_single_deployment_compute_config_to_api_model(
        self,
        cloud_id: str,
        compute_config: ComputeConfig,
        base_flags: Optional[Dict[str, Any]] = None,
    ) -> CloudDeploymentComputeConfig:
        # We should only make the head node schedulable when it's the *only* node in the cluster.
        # `worker_nodes=None` uses the default serverless config, so this only happens if `worker_nodes`
        # is explicitly set to an empty list.

        # Merge the MultiResourceComputeConfig flags (which apply to all cloud resources) with the individual compute config flags.
        # The individual compute config flags will override the base flags.
        flags = deepcopy(base_flags) if base_flags else {}
        flags.update(compute_config.flags or {})
        flags["allow-cross-zone-autoscaling"] = compute_config.enable_cross_zone_scaling

        if compute_config.min_resources:
            flags["min_resources"] = compute_config.min_resources
        if compute_config.max_resources:
            flags["max_resources"] = compute_config.max_resources

        return CloudDeploymentComputeConfig(
            cloud_deployment=compute_config.cloud_resource,
            allowed_azs=compute_config.zones,
            head_node_type=self._convert_head_node_config_to_api_model(
                compute_config.head_node,
                cloud_id=cloud_id,
                cloud_resource_name=compute_config.cloud_resource,
                schedulable_by_default=(
                    not compute_config.worker_nodes
                    and not compute_config.auto_select_worker_config
                ),
            ),
            worker_node_types=self._convert_worker_node_group_configs_to_api_models(
                compute_config.worker_nodes,
            ),
            auto_select_worker_config=compute_config.auto_select_worker_config,
            flags=flags,
            advanced_configurations_json=compute_config.advanced_instance_config
            or None,
        )

    def create_compute_config(
        self, compute_config: ComputeConfigType, *, name: Optional[str] = None
    ) -> Tuple[str, str]:
        """Register the provided compute config and return its internal ID."""

        if name is not None:
            _, version = parse_cluster_compute_name_version(name)
            if version is not None:
                raise ValueError(
                    "A version tag cannot be provided when creating a compute config. "
                    "The latest version tag will be generated and returned."
                )

        if isinstance(compute_config, MultiResourceComputeConfig):
            return self.create_multi_deployment_compute_config(
                compute_config, name=name
            )
        else:
            assert isinstance(compute_config, ComputeConfig)
            return self.create_single_deployment_compute_config(
                compute_config, name=name
            )

    def create_single_deployment_compute_config(
        self, compute_config: ComputeConfig, *, name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Register the provided single-deployment compute config and return its internal ID."""

        # Returns the default cloud if user-provided cloud is not specified (`None`).
        cloud_id = self.client.get_cloud_id(cloud_name=compute_config.cloud)  # type: ignore

        # Validate TPU not used on VM stacks (early validation before API call)
        self._validate_tpu_on_vm_stack(cloud_id, compute_config)

        deployment_config = self._convert_single_deployment_compute_config_to_api_model(
            cloud_id, compute_config
        )

        compute_config_api_model = ComputeTemplateConfig(
            cloud_id=cloud_id,
            deployment_configs=[deployment_config],
            # For compatibility, continue setting the top-level fields.
            allowed_azs=deployment_config.allowed_azs,
            head_node_type=deployment_config.head_node_type,
            worker_node_types=deployment_config.worker_node_types,
            auto_select_worker_config=deployment_config.auto_select_worker_config,
            flags=deployment_config.flags,
            advanced_configurations_json=deployment_config.advanced_configurations_json
            or None,
        )

        full_name, compute_config_id = self.client.create_compute_config(
            compute_config_api_model, name=name
        )
        self.logger.info(f"Created compute config: '{full_name}'")
        ui_url = self.client.get_compute_config_ui_url(
            compute_config_id, cloud_id=cloud_id
        )
        self.logger.info(f"View the compute config in the UI: '{ui_url}'")
        return full_name, compute_config_id

    def create_multi_deployment_compute_config(
        self, compute_config: MultiResourceComputeConfig, *, name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Register the provided multi-deployment compute config and return its internal ID."""
        # Returns the default cloud if user-provided cloud is not specified (`None`).
        cloud_id = self.client.get_cloud_id(cloud_name=compute_config.cloud)  # type: ignore

        # Validate TPU not used on VM stacks for each config (early validation)
        assert compute_config.configs
        for config in compute_config.configs:
            assert isinstance(config, ComputeConfig)
            self._validate_tpu_on_vm_stack(cloud_id, config)

        # Convert each compute config to the CloudDeploymentComputeConfig API model.
        deployment_configs = []
        for config in compute_config.configs:
            assert isinstance(config, ComputeConfig)
            deployment_configs.append(
                self._convert_single_deployment_compute_config_to_api_model(
                    cloud_id, config, compute_config.flags
                )
            )
        default_config = deployment_configs[0]

        compute_config_api_model = ComputeTemplateConfig(
            cloud_id=cloud_id,
            deployment_configs=deployment_configs,
            # For compatibility, use the first deployment config to set the top-level fields.
            allowed_azs=default_config.allowed_azs,
            head_node_type=default_config.head_node_type,
            worker_node_types=default_config.worker_node_types,
            auto_select_worker_config=default_config.auto_select_worker_config,
            flags=default_config.flags,
            advanced_configurations_json=default_config.advanced_configurations_json
            or None,
        )
        full_name, compute_config_id = self.client.create_compute_config(
            compute_config_api_model, name=name
        )
        self.logger.info(f"Created compute config: '{full_name}'")

        ui_url = self.client.get_compute_config_ui_url(
            compute_config_id, cloud_id=cloud_id
        )
        self.logger.info(f"View the compute config in the UI: '{ui_url}'")

        return full_name, compute_config_id

    def _convert_api_model_to_advanced_instance_config(
        self,
        api_model: Union[
            DecoratedComputeTemplateConfig, ComputeNodeType, WorkerNodeType
        ],
    ) -> Optional[Dict]:
        if api_model.advanced_configurations_json:
            return api_model.advanced_configurations_json

        # Only one of aws_advanced_configurations_json or gcp_advanced_configurations_json will be set.
        if api_model.aws_advanced_configurations_json:
            return api_model.aws_advanced_configurations_json
        if api_model.gcp_advanced_configurations_json:
            return api_model.gcp_advanced_configurations_json

        return None

    def _convert_api_model_to_resource_dict(
        self, resources: Optional[Resources]
    ) -> Optional[Dict[str, float]]:
        # Flatten the resource dict returned by the API and strip `None` values.
        if resources is None:
            return None

        return {
            k: v
            for k, v in {
                "CPU": resources.cpu,
                "GPU": resources.gpu,
                "memory": resources.memory,
                "object_store_memory": resources.object_store_memory,
                **(resources.custom_resources or {}),
            }.items()
            if v is not None
        }

    def _convert_api_model_to_head_node_config(
        self, api_model: ComputeNodeType
    ) -> HeadNodeConfig:
        flags: Dict[str, Any] = deepcopy(api_model.flags) or {}

        cloud_deployment_dict = flags.pop("cloud_deployment", None)
        cloud_deployment = (
            CloudDeployment.from_dict(cloud_deployment_dict)
            if cloud_deployment_dict
            else None
        )

        # Convert required_resources from API model to user-facing model
        required_resources = None
        if api_model.required_resources is not None:
            required_resources = UserPhysicalResources.from_dict(
                api_model.required_resources.to_dict()
            )

        return HeadNodeConfig(
            instance_type=api_model.instance_type,
            resources=self._convert_api_model_to_resource_dict(api_model.resources),
            required_resources=required_resources,
            labels=api_model.labels,
            required_labels=api_model.required_labels,
            advanced_instance_config=self._convert_api_model_to_advanced_instance_config(
                api_model,
            ),
            flags=flags or None,
            cloud_deployment=cloud_deployment,
        )

    def _convert_api_models_to_worker_node_group_configs(
        self, api_models: List[WorkerNodeType]
    ) -> List[WorkerNodeGroupConfig]:
        # TODO(edoakes): support advanced_instance_config.
        configs = []
        for api_model in api_models:
            if api_model.use_spot and api_model.fallback_to_ondemand:
                market_type = MarketType.PREFER_SPOT
            elif api_model.use_spot:
                market_type = MarketType.SPOT
            else:
                market_type = MarketType.ON_DEMAND

            min_nodes = api_model.min_workers
            if min_nodes is None:
                min_nodes = 0

            max_nodes = api_model.max_workers
            if max_nodes is None:
                # TODO(edoakes): this defaulting to 10 seems like really strange
                # behavior here but I'm copying what the UI does. In Shomil's new
                # API let's not make these optional.
                max_nodes = 10

            flags: Dict[str, Any] = deepcopy(api_model.flags) or {}

            cloud_deployment_dict = flags.pop("cloud_deployment", None)
            cloud_deployment = (
                CloudDeployment.from_dict(cloud_deployment_dict)
                if cloud_deployment_dict
                else None
            )

            # Convert required_resources from API model to user-facing model
            required_resources = None
            if api_model.required_resources is not None:
                required_resources = UserPhysicalResources.from_dict(
                    api_model.required_resources.to_dict()
                )

            configs.append(
                WorkerNodeGroupConfig(
                    name=api_model.name,
                    instance_type=api_model.instance_type,
                    resources=self._convert_api_model_to_resource_dict(
                        api_model.resources
                    ),
                    required_resources=required_resources,
                    labels=api_model.labels,
                    required_labels=api_model.required_labels,
                    advanced_instance_config=self._convert_api_model_to_advanced_instance_config(
                        api_model,
                    ),
                    min_nodes=min_nodes,
                    max_nodes=max_nodes,
                    market_type=market_type,
                    flags=flags or None,
                    cloud_deployment=cloud_deployment,
                )
            )

        return configs

    def _convert_cloud_deployment_compute_config_api_model_to_single_resource_compute_config(
        self, cloud_name: str, api_model: CloudDeploymentComputeConfig,
    ) -> ComputeConfig:
        worker_nodes = None
        if not api_model.auto_select_worker_config:
            if api_model.worker_node_types is not None:
                # Convert worker node types when they are present.
                worker_nodes = self._convert_api_models_to_worker_node_group_configs(
                    api_model.worker_node_types
                )
            else:
                # An explicit head-node-only cluster (no worker nodes configured).
                worker_nodes = []

        zones = None
        # NOTE(edoakes): the API returns '["any"]' if no AZs are passed in on the creation path.
        if api_model.allowed_azs not in [["any"], []]:
            zones = api_model.allowed_azs

        enable_cross_zone_scaling = False
        flags: Dict[str, Any] = deepcopy(api_model.flags) or {}
        enable_cross_zone_scaling = flags.pop("allow-cross-zone-autoscaling", False)
        min_resources = flags.pop("min_resources", None)
        max_resources = flags.pop("max_resources", None)
        if max_resources is None:
            max_resources = {}
            max_cpus = flags.pop("max-cpus", None)
            if max_cpus:
                max_resources["CPU"] = max_cpus
            max_gpus = flags.pop("max-gpus", None)
            if max_gpus:
                max_resources["GPU"] = max_gpus

        return ComputeConfig(
            cloud=cloud_name,
            cloud_resource=api_model.cloud_deployment,
            zones=zones,
            advanced_instance_config=api_model.advanced_configurations_json or None,
            enable_cross_zone_scaling=enable_cross_zone_scaling,
            head_node=self._convert_api_model_to_head_node_config(
                api_model.head_node_type
            ),
            worker_nodes=worker_nodes,
            min_resources=min_resources,
            max_resources=max_resources or None,
            flags=flags,
            auto_select_worker_config=api_model.auto_select_worker_config or False,
        )

    def _convert_api_model_to_compute_config_version(
        self, api_model: DecoratedComputeTemplate  # noqa: ARG002
    ) -> ComputeConfigVersion:
        api_model_config: DecoratedComputeTemplateConfig = api_model.config
        cloud = self.client.get_cloud(cloud_id=api_model_config.cloud_id)
        if cloud is None:
            raise RuntimeError(
                f"Cloud with ID '{api_model_config.cloud_id}' not found. "
                "This should never happen; please reach out to Anyscale support."
            )

        configs = None
        if api_model_config.deployment_configs:
            configs = [
                self._convert_cloud_deployment_compute_config_api_model_to_single_resource_compute_config(
                    cloud.name, config
                )
                for config in api_model_config.deployment_configs
            ]
            if len(configs) == 1:
                # If there's only one deployment config, return it directly.
                return ComputeConfigVersion(
                    name=f"{api_model.name}:{api_model.version}",
                    id=api_model.id,
                    config=configs[0],
                )
            return ComputeConfigVersion(
                name=f"{api_model.name}:{api_model.version}",
                id=api_model.id,
                config=MultiResourceComputeConfig(cloud=cloud.name, configs=configs),
            )

        # If there are no deployment configs, this is a compute config for a single cloud deployment - parse the top-level fields.

        worker_nodes = None
        if not api_model_config.auto_select_worker_config:
            if api_model_config.worker_node_types is not None:
                # Convert worker node types when they are present.
                worker_nodes = self._convert_api_models_to_worker_node_group_configs(
                    api_model_config.worker_node_types
                )
            else:
                # An explicit head-node-only cluster (no worker nodes configured).
                worker_nodes = []

        zones = None
        # NOTE(edoakes): the API returns '["any"]' if no AZs are passed in on the creation path.
        if api_model_config.allowed_azs not in [["any"], []]:
            zones = api_model_config.allowed_azs

        enable_cross_zone_scaling = False
        flags: Dict[str, Any] = deepcopy(api_model_config.flags) or {}
        enable_cross_zone_scaling = flags.pop("allow-cross-zone-autoscaling", False)
        min_resources = flags.pop("min_resources", None)
        max_resources = flags.pop("max_resources", None)
        if max_resources is None:
            max_resources = {}
            max_cpus = flags.pop("max-cpus", None)
            if max_cpus:
                max_resources["CPU"] = max_cpus
            max_gpus = flags.pop("max-gpus", None)
            if max_gpus:
                max_resources["GPU"] = max_gpus

        return ComputeConfigVersion(
            name=f"{api_model.name}:{api_model.version}",
            id=api_model.id,
            config=ComputeConfig(
                cloud=cloud.name,
                zones=zones,
                advanced_instance_config=self._convert_api_model_to_advanced_instance_config(
                    api_model_config
                ),
                enable_cross_zone_scaling=enable_cross_zone_scaling,
                head_node=self._convert_api_model_to_head_node_config(
                    api_model_config.head_node_type
                ),
                worker_nodes=worker_nodes,  # type: ignore
                min_resources=min_resources,
                max_resources=max_resources or None,
                auto_select_worker_config=api_model_config.auto_select_worker_config,
                flags=flags,
            ),
        )

    def _resolve_id(
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        if id is not None:
            compute_config_id = id
        elif name is not None:
            compute_config_id = self.client.get_compute_config_id(
                compute_config_name=name, cloud=cloud, include_archived=include_archived
            )
            if compute_config_id is None:
                raise RuntimeError(f"Compute config '{name}' not found.")
        else:
            raise ValueError("Either name or ID must be provided.")

        return compute_config_id

    def get_compute_config(
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        include_archived: bool = False,
    ) -> ComputeConfigVersion:
        """Get the compute config for the provided name.

        The name can contain an optional version, e.g., '<name>:<version>'.
        If no version is provided, the latest one will be returned.
        """
        compute_config_id = self._resolve_id(
            id=id, name=name, cloud=cloud, include_archived=include_archived
        )
        compute_config = self.client.get_compute_config(compute_config_id)
        if compute_config is None:
            raise RuntimeError(
                f"Compute config with ID '{compute_config_id}' not found.'"
            )
        return self._convert_api_model_to_compute_config_version(compute_config)

    def archive_compute_config(
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
    ):
        compute_config_id = self._resolve_id(id=id, name=name, cloud=cloud)
        self.client.archive_compute_config(compute_config_id=compute_config_id)
        self.logger.info("Compute config is successfully archived.")

    def get_default_compute_config(
        self, *, cloud: Optional[str] = None, cloud_resource: Optional[str] = None,
    ) -> ComputeConfigVersion:
        """Get the default compute config for the specified cloud.

        Args:
            cloud: Name of the cloud. If not provided, uses the default cloud.
            cloud_resource: Name of the cloud resource. If not provided, uses the
                default cloud resource for the cloud.

        Returns:
            ComputeConfigVersion containing the default compute config.
        """
        # Resolve cloud name to cloud ID
        cloud_id = self.client.get_cloud_id(cloud_name=cloud)  # type: ignore

        # Resolve cloud resource name to cloud resource ID if provided
        cloud_resource_id = None
        if cloud_resource:
            cloud_resource_obj = self.client.get_cloud_resource_by_name(
                cloud_id=cloud_id, cloud_resource_name=cloud_resource
            )
            if cloud_resource_obj:
                cloud_resource_id = cloud_resource_obj.cloud_resource_id

        # Get the default compute config from the client
        default_config = self.client.get_default_compute_config(
            cloud_id=cloud_id, cloud_resource_id=cloud_resource_id
        )

        # Convert to SDK model
        return self._convert_api_model_to_compute_config_version(default_config)

    def list_compute_configs(  # noqa: PLR0913
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud_id: Optional[str] = None,
        cloud_name: Optional[str] = None,
        sort_by: str = "last_modified_at",
        sort_order: str = "asc",
        max_items: int = 20,
        next_token: Optional[str] = None,
        include_shared: bool = False,
    ):
        """List compute configs with filtering, sorting, and pagination.

        Args:
            name: Filter by compute config name
            id: Filter by specific compute config ID
            cloud_id: Filter by cloud ID
            cloud_name: Filter by cloud name (will be resolved to cloud_id)
            sort_by: Field to sort by (name, created_at, last_modified_at)
            sort_order: Sort order (asc or desc)
            max_items: Maximum number of items to return per page
            next_token: Pagination token for fetching next page
            include_shared: Include configs shared with the user

        Returns:
            ComputeConfigListResult with:
            - results: List of compute config objects
            - next_token: Pagination token for next page (None if no more results)
            - count: Number of results returned
        """

        # If fetching by ID, return single result
        if id:
            compute_config = self.client.get_compute_config(id)
            if compute_config is None:
                raise RuntimeError(f"Compute config with ID '{id}' not found.")
            return ComputeConfigListResult(
                results=[compute_config], next_token=None, count=1
            )

        # Resolve cloud_name to cloud_id if provided
        resolved_cloud_id = cloud_id
        if cloud_name:
            # Directly use the client's get_cloud_id method
            resolved_cloud_id = self.client.get_cloud_id(cloud_name=cloud_name)  # type: ignore

        # Build query for search endpoint
        query: Dict[str, Any] = {"paging": {"count": max_items}}

        # Add name filter if specified
        if name:
            query["name"] = {"equals": name}

        # Add creator_id filter if not including shared configs
        if not include_shared and not name:
            # Use self.client instead of creating new BaseController
            user_info = self.client.get_user_info()  # type: ignore
            query["creator_id"] = user_info["id"] if isinstance(user_info, dict) else user_info.id  # type: ignore

        # Add cloud filter if specified
        if resolved_cloud_id:
            query["cloud_id"] = resolved_cloud_id

        # Add pagination token if provided
        if next_token:
            query["paging"]["paging_token"] = next_token

        # Add server-side sorting
        sort_field_map = {
            "name": "NAME",
            "created_at": "CREATED_AT",
            "last_modified_at": "LAST_MODIFIED_AT",
        }
        query["sort_by_clauses"] = [
            {
                "sort_field": sort_field_map.get(sort_by, "LAST_MODIFIED_AT"),
                "sort_order": sort_order.upper(),
            }
        ]

        # Make API call
        response = self.client.search_cluster_computes(query)  # type: ignore

        # Return structured result
        pagination_token = (
            response.metadata.next_paging_token if response.metadata else None
        )
        return ComputeConfigListResult(
            results=response.results,
            next_token=pagination_token,
            count=len(response.results),
        )
