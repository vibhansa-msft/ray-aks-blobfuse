from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar, DefaultDict, Dict, List, Optional, Union
import warnings

from anyscale._private.models import ModelBase, ModelEnum


ResourceDict = Dict[str, float]
LabelDict = Dict[str, str]
AdvancedInstanceConfigDict = Dict[str, Any]

# CPU architecture constants
CPU_ARCHITECTURE_X86_64 = "x86_64"
CPU_ARCHITECTURE_ARM64 = "arm64"

# TPU configuration constants
# See: https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus
TPU_DOCS_URL = "https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus"
REQUIRED_TPU_NODE_SELECTORS = [
    "cloud.google.com/gke-tpu-topology",
    "cloud.google.com/gke-tpu-accelerator",
]
# Ray labels that can be used to derive TPU node selectors (alternative to explicit node selectors)
RAY_TPU_ACCELERATOR_LABEL = "ray.io/accelerator-type"
RAY_TPU_TOPOLOGY_LABEL = "ray.io/tpu-topology"


def _extract_accelerator_from_labels(
    labels: Optional[LabelDict], required_labels: Optional[LabelDict]
) -> Optional[str]:
    """Extract accelerator type from labels or required_labels.

    Checks for the ray.io/accelerator-type label in required_labels first,
    then falls back to labels. This allows users to specify GPU/TPU type
    via labels as an alternative to the accelerator field in required_resources.

    Args:
        labels: Optional labels dictionary
        required_labels: Optional required_labels dictionary

    Returns:
        The accelerator type string if found, None otherwise.
    """
    # Check required_labels first (takes precedence)
    if required_labels and RAY_TPU_ACCELERATOR_LABEL in required_labels:
        accel = required_labels[RAY_TPU_ACCELERATOR_LABEL]
        if accel:
            return accel

    # Fall back to labels
    if labels and RAY_TPU_ACCELERATOR_LABEL in labels:
        accel = labels[RAY_TPU_ACCELERATOR_LABEL]
        if accel:
            return accel

    return None


def _generate_free_pod_shape_name(
    cpu: Optional[int],
    memory: Optional[Union[str, int]],
    gpu: Optional[int],
    accelerator: Optional[str],
    tpu: Optional[int],
    tpu_hosts: Optional[int],
    cpu_architecture: Optional[str] = None,
) -> str:
    """Generate a unique name for a free pod shape based on physical resources.

    Follows the backend naming convention:
    - "FP-{CPU}CPU-{MEM}GB" for CPU/memory only
    - "FP-{CPU}CPU-{MEM}GB-{GPU}GPU" for GPU without accelerator type
    - "FP-{CPU}CPU-{MEM}GB-{GPU}GPU-{ACCEL}" for GPU with accelerator type
    - "FP-{CPU}CPU-{MEM}GB-{TPU}TPU-{TPUHOST}TPUHost-{ACCEL}" for TPU
    - Append "-ARM64" for arm64 architecture

    Args:
        cpu: Number of CPUs
        memory: Memory in bytes or Kubernetes format string (e.g., '4Gi')
        gpu: Number of GPUs
        accelerator: Accelerator type (e.g., 'T4', 'A100', 'TPU-V6E')
        tpu: Number of TPUs
        tpu_hosts: Number of TPU hosts
        cpu_architecture: CPU architecture type ('x86_64' or 'arm64')

    Returns:
        Generated free pod shape name.
    """
    # Parse memory to bytes if string, then convert to GB
    memory_bytes = _parse_memory_string(memory) if memory else 0
    memory_gb = memory_bytes // (1024 * 1024 * 1024)  # Convert to GB

    parts = [f"FP-{cpu or 0}CPU-{memory_gb}GB"]

    if tpu and tpu > 0:
        parts.append(f"{tpu}TPU")
        if tpu_hosts and tpu_hosts > 0:
            parts.append(f"{tpu_hosts}TPUHost")
        if accelerator:
            parts.append(accelerator)
    elif gpu and gpu > 0:
        parts.append(f"{gpu}GPU")
        if accelerator:
            parts.append(accelerator)

    # Add ARM64 suffix for non-default architecture
    if cpu_architecture and cpu_architecture.lower() == CPU_ARCHITECTURE_ARM64:
        parts.append("ARM64")

    return "-".join(parts)


def _parse_memory_string(memory_str: Union[str, int]) -> int:
    """Parse memory string (e.g., '4Gi', '1024Mi', '1G') to bytes.

    Uses Kubernetes quantity parsing to handle all standard K8s resource formats.

    Args:
        memory_str: Memory value as string (e.g., '4Gi') or int (bytes).

    Returns:
        Memory in bytes as int.
    """
    if isinstance(memory_str, int):
        return memory_str

    if not isinstance(memory_str, str):
        raise TypeError(f"Memory must be a string or int, got: {type(memory_str)}")

    try:
        from kubernetes.utils.quantity import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional Kubernetes dependency for quantity parsing")
            parse_quantity,
        )

        # parse_quantity returns a Decimal in the canonical unit (bytes for memory)
        return int(parse_quantity(memory_str))
    except (ValueError, ImportError, AttributeError) as e:
        raise ValueError(f"Invalid memory format: {memory_str}. Error: {e}") from e


def _format_bytes_to_memory_string(memory_bytes: Union[int, str]) -> str:
    """Convert memory in bytes to human-readable string format (e.g., '8Gi').

    Args:
        memory_bytes: Memory value in bytes (int) or already formatted string.

    Returns:
        Memory as human-readable string (e.g., '8Gi', '512Mi').
    """
    if isinstance(memory_bytes, str):
        return memory_bytes

    if not isinstance(memory_bytes, int):
        return str(memory_bytes)

    # Use GiB if it's a clean multiple, otherwise use MiB or bytes
    gib = 1024 * 1024 * 1024
    mib = 1024 * 1024

    # GiB format (clean multiple or with decimal)
    if memory_bytes >= gib:
        if memory_bytes % gib == 0:
            return f"{memory_bytes // gib}Gi"
        return f"{memory_bytes / gib:.1f}Gi".rstrip("0").rstrip(".")

    # MiB format
    if memory_bytes >= mib:
        return f"{memory_bytes // mib}Mi"

    return str(memory_bytes)


@dataclass(frozen=True)
class PhysicalResources(ModelBase):
    """Physical resources specification for compute nodes.

    Used for custom instance types (free pod shapes) to explicitly define
    CPU, memory, and GPU resources instead of deriving from instance type.
    """

    __doc_py_example__ = """
from anyscale.compute_config.models import PhysicalResources

required_resources = PhysicalResources(
    CPU=4,
    memory="8Gi",
    GPU=1,
    accelerator="T4",
)
"""

    __doc_yaml_example__ = """
required_resources:
  CPU: 4
  memory: 8Gi
  GPU: 1
  accelerator: T4
"""

    CPU: Optional[int] = field(
        default=None, metadata={"docstring": "Number of CPUs to allocate."},
    )

    def _validate_CPU(self, CPU: Optional[int]):
        if CPU is not None:
            if not isinstance(CPU, int):
                raise TypeError("'CPU' must be an int.")
            if CPU < 0:
                raise ValueError("'CPU' must be >= 0.")

    memory: Optional[Union[str, int]] = field(
        default=None,
        metadata={
            "docstring": "Amount of memory to allocate. Can be specified as bytes (int) or as a string with units (e.g., '4Gi', '1024Mi')."
        },
    )

    def _validate_memory(self, memory: Optional[Union[str, int]]):
        if memory is not None:
            try:
                parsed = _parse_memory_string(memory)
                if parsed < 0:
                    raise ValueError("'memory' must be >= 0.")
            except TypeError:
                # Re-raise TypeError as-is for better error messages
                raise
            except ValueError as e:
                raise ValueError(f"Invalid memory value: {e}")

    GPU: Optional[int] = field(
        default=None, metadata={"docstring": "Number of GPUs to allocate."},
    )

    def _validate_GPU(self, GPU: Optional[int]):
        if GPU is not None:
            if not isinstance(GPU, int):
                raise TypeError("'GPU' must be an int.")
            if GPU < 0:
                raise ValueError("'GPU' must be >= 0.")

    accelerator: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Type of accelerator (e.g., 'T4', 'L4', 'A100', 'H100', 'TPUv4'). "
            "Use abbreviated names for readability in generated instance type names."
        },
    )

    def _validate_accelerator(self, accelerator: Optional[str]):
        if accelerator is not None and not isinstance(accelerator, str):
            raise TypeError("'accelerator' must be a string.")

    TPU: Optional[int] = field(
        default=None, metadata={"docstring": "Number of TPUs to allocate."},
    )

    def _validate_TPU(self, TPU: Optional[int]):
        if TPU is not None:
            if not isinstance(TPU, int):
                raise TypeError("'TPU' must be an int.")
            if TPU < 0:
                raise ValueError("'TPU' must be >= 0.")

    tpu_hosts: Optional[int] = field(
        default=None,
        metadata={
            "docstring": "Number of TPU hosts (for anyscale/tpu_hosts custom resource).",
            "alias": "anyscale/tpu_hosts",
        },
    )

    def _validate_tpu_hosts(self, tpu_hosts: Optional[int]):
        if tpu_hosts is not None:
            if not isinstance(tpu_hosts, int):
                raise TypeError("'tpu_hosts' must be an int.")
            if tpu_hosts < 0:
                raise ValueError("'tpu_hosts' must be >= 0.")

    cpu_architecture: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "CPU architecture type. Valid values: 'x86_64' (default), 'arm64'.",
        },
    )

    def _validate_cpu_architecture(self, cpu_architecture: Optional[str]):
        if cpu_architecture is not None:
            if not isinstance(cpu_architecture, str):
                raise TypeError("'cpu_architecture' must be a string.")
            valid_values = [CPU_ARCHITECTURE_X86_64, CPU_ARCHITECTURE_ARM64]
            if cpu_architecture not in valid_values:
                raise ValueError(
                    f"'cpu_architecture' must be one of {valid_values}, got '{cpu_architecture}'."
                )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhysicalResources":
        """Create PhysicalResources from dict, handling case variations.

        Handles conversions from API model (lowercase) to user model (mixed case):
        - cpu -> CPU
        - gpu -> GPU
        - tpu -> TPU
        - Memory -> memory
        - anyscale_tpu_hosts -> tpu_hosts

        Also converts memory from bytes (int) to human-readable format (e.g., '8Gi').
        """
        # Key mapping from API model (lowercase) to user model (mixed case)
        key_mapping = {
            "cpu": "CPU",
            "gpu": "GPU",
            "tpu": "TPU",
            "Memory": "memory",
            "anyscale_tpu_hosts": "tpu_hosts",
        }

        normalized = {}
        for k, v in d.items():
            # Skip None values
            if v is None:
                continue
            # Map to expected key name, or keep original if not in mapping
            normalized_key = key_mapping.get(k, k)
            # Convert memory from bytes to human-readable format
            normalized_value = (
                _format_bytes_to_memory_string(v)
                if normalized_key == "memory" and isinstance(v, int)
                else v
            )
            normalized[normalized_key] = normalized_value

        return cls(**normalized)

    def to_dict(
        self, *, exclude_none: bool = True, for_api: bool = False
    ) -> Dict[str, Any]:
        """Convert to dictionary.

        Args:
            exclude_none: If True, exclude keys with None values.
            for_api: If True, use lowercase keys (cpu, gpu, tpu) and convert memory to bytes.
                     If False, use uppercase keys (CPU, GPU, TPU) and keep memory human-readable.
        """
        result: Dict[str, Any] = {}

        # Key names based on output format
        cpu_key = "cpu" if for_api else "CPU"
        gpu_key = "gpu" if for_api else "GPU"
        tpu_key = "tpu" if for_api else "TPU"

        if self.CPU is not None or not exclude_none:
            result[cpu_key] = self.CPU

        if self.memory is not None:
            if for_api:
                # Convert to bytes for API
                result["memory"] = _parse_memory_string(self.memory)
            # Keep human-readable format for user output
            elif isinstance(self.memory, str):
                result["memory"] = self.memory
            else:
                result["memory"] = _format_bytes_to_memory_string(self.memory)
        elif not exclude_none:
            result["memory"] = None

        if self.GPU is not None or not exclude_none:
            result[gpu_key] = self.GPU

        if self.accelerator is not None or not exclude_none:
            result["accelerator"] = self.accelerator

        if self.TPU is not None or not exclude_none:
            result[tpu_key] = self.TPU

        if self.tpu_hosts is not None or not exclude_none:
            result["tpu_hosts"] = self.tpu_hosts

        if self.cpu_architecture is not None or not exclude_none:
            result["cpu_architecture"] = self.cpu_architecture

        return result


def _validate_resource_dict(r: Optional[ResourceDict], *, field_name: str):
    if r is None:
        return

    if not isinstance(r, dict):
        raise TypeError(f"'{field_name}' must be a Dict[str, float], but got: {r}")

    for k, v in r.items():
        if not isinstance(k, str):
            raise TypeError(f"'{field_name}' keys must be strings, but got: {k}")
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError(
                    f"'{field_name}' values must be >= 0, but got: '{k}: {v}'"
                )
        else:
            raise TypeError(
                f"'{field_name}' values must be floats, but got: '{k}: {v}'"
            )


def _validate_label_dict(labels: Optional[LabelDict]):
    if labels is None:
        return

    # Convert any non-string keys/values to strings to ensure compatibility
    for k, v in labels.items():
        if not isinstance(k, str):
            raise TypeError(f"'labels' keys must be strings, but got: {k}")
        if not isinstance(v, str):
            raise TypeError(f"'labels' values must be strings, but got: {v}")


def _validate_advanced_instance_config_dict(c: Optional[AdvancedInstanceConfigDict]):
    if c is None:
        return

    if not isinstance(c, dict) or not all(isinstance(k, str) for k in c):
        raise TypeError("'advanced_instance_config' must be a Dict[str, Any]")


@dataclass(frozen=True)
class CloudDeployment(ModelBase):
    """Cloud deployment selectors for a node group; one or more selectors may be passed to target a specific deployment from all of a cloud's deployments."""

    __doc_py_example__ = """
from anyscale.compute_config.models import CloudDeployment

cloud_deployment = CloudDeployment(
    provider="aws",
    region="us-west-2",
    machine_pool="machine-pool-name",
    id="cldrsrc_1234567890",
)
"""

    __doc_yaml_example__ = """
cloud_deployment:
  provider: aws
  region: us-west-2
  machine_pool: machine-pool-name
  id: cldrsrc_1234567890
"""

    provider: Optional[str] = field(
        default=None,
        metadata={"docstring": "Cloud provider name, e.g., `aws` or `gcp`."},
    )

    def _validate_provider(self, provider: Optional[str]):
        if provider is not None and not isinstance(provider, str):
            raise TypeError("'provider' must be a string.")

    region: Optional[str] = field(
        default=None,
        metadata={"docstring": "Cloud provider region, e.g., `us-west-2`."},
    )

    def _validate_region(self, region: Optional[str]):
        if region is not None and not isinstance(region, str):
            raise TypeError("'region' must be a string.")

    machine_pool: Optional[str] = field(
        default=None, metadata={"docstring": "Machine pool name."}
    )

    def _validate_machine_pool(self, machine_pool: Optional[str]):
        if machine_pool is not None and not isinstance(machine_pool, str):
            raise TypeError("'machine_pool' must be a string.")

    id: Optional[str] = field(
        default=None, metadata={"docstring": "Cloud deployment ID from cloud setup."}
    )

    def _validate_id(self, id: Optional[str]):  # noqa: A002
        if id is not None and not isinstance(id, str):
            raise TypeError("'id' must be a string.")


@dataclass(frozen=True)
class _NodeConfig(ModelBase):
    instance_type: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Cloud provider instance type, e.g., `m5.2xlarge` on AWS or `n2-standard-8` on GCP. "
            "Defaults to 'custom' when required_resources is provided."
        },
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_NodeConfig":
        """Create node config from dict, checking for deprecated physical_resources field."""
        # Check for deprecated physical_resources field
        if "physical_resources" in d and d["physical_resources"] is not None:
            raise ValueError(
                "'physical_resources' has been renamed to 'required_resources'. "
                "Please update your configuration to use 'required_resources' instead."
            )
        # Remove physical_resources if it's None to avoid passing unknown field
        if "physical_resources" in d:
            d = d.copy()
            del d["physical_resources"]
        return cls(**d)

    def _validate_instance_type(self, instance_type: Optional[str]) -> str:
        # Default to "custom" when required_resources is provided
        if instance_type is None:
            if self.required_resources is not None:
                instance_type = "custom"
            else:
                raise ValueError(
                    "'instance_type' is required when 'required_resources' is not provided."
                )
        if not isinstance(instance_type, str):
            raise TypeError("'instance_type' must be a string.")
        return instance_type

    def to_dict(self, *, exclude_none: bool = True) -> Dict[str, Any]:
        """Convert to dictionary, excluding 'instance_type: custom' for free pod shapes."""
        result = super().to_dict(exclude_none=exclude_none)

        # Remove instance_type if it's "custom" and required_resources is present
        # This makes the output cleaner for free pod shapes
        if (
            result.get("instance_type") == "custom"
            and result.get("required_resources") is not None
        ):
            del result["instance_type"]

        return result

    resources: Optional[ResourceDict] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Logical resources that will be available on this node. Defaults to match the physical resources of the instance type."
        },
    )

    def _validate_resources(self, resources: Optional[ResourceDict]):
        _validate_resource_dict(resources, field_name="resources")

    required_resources: Optional[PhysicalResources] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Physical resources for custom instance types (free pod shapes). Explicitly defines CPU, memory, and GPU resources."
        },
    )

    def _validate_required_resources(
        self, required_resources: Optional[PhysicalResources]
    ) -> Optional[PhysicalResources]:
        if required_resources is None:
            return None
        if isinstance(required_resources, dict):
            # Convert dict to PhysicalResources object
            return PhysicalResources.from_dict(required_resources)
        if isinstance(required_resources, PhysicalResources):
            return required_resources
        raise TypeError(
            "'required_resources' must be a PhysicalResources object or dict."
        )

    labels: Optional[LabelDict] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Labels to associate the node with for scheduling purposes. Defaults to the list of Ray & Anyscale default labels."
        },
    )

    def _validate_labels(self, labels: Optional[LabelDict]):
        _validate_label_dict(labels)

    required_labels: Optional[LabelDict] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Required labels that must be present on the node for scheduling purposes."
        },
    )

    def _validate_required_labels(self, required_labels: Optional[LabelDict]):
        _validate_label_dict(required_labels)
        # Cross-field validation: if accelerator-type is a GPU, GPU must be specified
        self._validate_gpu_accelerator_consistency(required_labels)
        # Cross-field validation: if accelerator-type is a TPU, TPU fields must be specified
        self._validate_tpu_accelerator_consistency(required_labels)

    def _validate_gpu_accelerator_consistency(
        self, required_labels: Optional[LabelDict]
    ):
        """Validate that GPU count is specified when a non-TPU accelerator type is requested.

        If ray.io/accelerator-type is specified and it's not a TPU type,
        then GPU must be specified in required_resources.
        """
        if required_labels is None:
            return

        accelerator_type = required_labels.get(RAY_TPU_ACCELERATOR_LABEL)
        if not accelerator_type:
            return

        # Skip if it's a TPU type (TPU validation is handled separately)
        if accelerator_type.upper().startswith("TPU"):
            return

        # Non-TPU accelerator type specified - assume it's a GPU and require GPU count
        # GPU accelerator type is specified, check if GPU count is provided
        if self.required_resources is None:
            raise ValueError(
                f"When specifying an accelerator type '{accelerator_type}' in "
                f"'required_labels', you must also specify 'GPU' count in 'required_resources'.\n"
                f"Example:\n"
                f"  required_resources:\n"
                f"    CPU: 7\n"
                f"    memory: 12Gi\n"
                f"    GPU: 1  # Required when using accelerator type\n"
                f"  required_labels:\n"
                f"    ray.io/accelerator-type: {accelerator_type}"
            )

        pr = self.required_resources
        gpu_count = (
            pr.get("GPU") or pr.get("gpu") or 0 if isinstance(pr, dict) else pr.GPU or 0
        )

        if not gpu_count or gpu_count <= 0:
            raise ValueError(
                f"When specifying an accelerator type '{accelerator_type}' in "
                f"'required_labels', you must also specify 'GPU' count in 'required_resources'.\n"
                f"Example:\n"
                f"  required_resources:\n"
                f"    CPU: 7\n"
                f"    memory: 12Gi\n"
                f"    GPU: 1  # Required when using accelerator type\n"
                f"  required_labels:\n"
                f"    ray.io/accelerator-type: {accelerator_type}"
            )

    def _validate_tpu_accelerator_consistency(
        self, required_labels: Optional[LabelDict]
    ):
        """Validate that TPU fields are specified when a TPU accelerator type is requested.

        If ray.io/accelerator-type is a TPU type (starts with "TPU"),
        then TPU, tpu_hosts, and ray.io/tpu-topology must be specified.
        """
        if required_labels is None:
            return

        accelerator_type = required_labels.get(RAY_TPU_ACCELERATOR_LABEL)
        if not accelerator_type:
            return

        # Only validate if it's a TPU type
        if not accelerator_type.upper().startswith("TPU"):
            return

        # TPU accelerator type specified - check required fields
        missing_fields = []

        # Check TPU count in required_resources
        tpu_count = 0
        tpu_hosts = 0
        if self.required_resources is not None:
            pr = self.required_resources
            if isinstance(pr, dict):
                tpu_count = pr.get("TPU") or pr.get("tpu") or 0
                tpu_hosts = pr.get("tpu_hosts") or 0
            else:
                tpu_count = pr.TPU or 0
                tpu_hosts = pr.tpu_hosts or 0

        if not tpu_count or tpu_count <= 0:
            missing_fields.append("TPU (in required_resources)")

        if not tpu_hosts or tpu_hosts <= 0:
            missing_fields.append("tpu_hosts (in required_resources)")

        # Check ray.io/tpu-topology in required_labels or labels
        has_topology = RAY_TPU_TOPOLOGY_LABEL in required_labels
        if not has_topology and self.labels:
            has_topology = RAY_TPU_TOPOLOGY_LABEL in self.labels

        if not has_topology:
            missing_fields.append("ray.io/tpu-topology (in required_labels)")

        if missing_fields:
            raise ValueError(
                f"When specifying a TPU accelerator type '{accelerator_type}' in "
                f"'required_labels', you must also specify: {', '.join(missing_fields)}.\n"
                f"Example:\n"
                f"  required_resources:\n"
                f"    CPU: 7\n"
                f"    memory: 12Gi\n"
                f"    TPU: 4\n"
                f"    tpu_hosts: 1\n"
                f"  required_labels:\n"
                f"    ray.io/accelerator-type: {accelerator_type}\n"
                f"    ray.io/tpu-topology: 2x2\n"
                f"For more information, see: {TPU_DOCS_URL}"
            )

    advanced_instance_config: Optional[AdvancedInstanceConfigDict] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Advanced instance configurations that will be passed through to the cloud provider.",
            "customer_hosted_only": True,
        },
    )

    def _validate_advanced_instance_config(
        self, advanced_instance_config: Optional[AdvancedInstanceConfigDict]
    ):
        _validate_advanced_instance_config_dict(advanced_instance_config)
        # Validate TPU-specific node selectors if TPU is configured
        self._validate_tpu_node_selectors(advanced_instance_config)
        # Validate accelerator type consistency between required_labels and nodeSelector
        self._validate_accelerator_consistency(advanced_instance_config)

    def _validate_accelerator_consistency(
        self, advanced_instance_config: Optional[AdvancedInstanceConfigDict]
    ):
        """Validate that accelerator types are consistent between required_labels and nodeSelector.

        If both ray.io/accelerator-type (in required_labels) and cloud.google.com/gke-accelerator
        (in nodeSelector) are specified, they should refer to the same accelerator type.
        """
        if advanced_instance_config is None:
            return

        # Extract nodeSelector from advanced_instance_config
        spec = advanced_instance_config.get("spec", {})
        if not isinstance(spec, dict):
            return
        node_selectors = spec.get("nodeSelector", {})
        if not node_selectors:
            return

        # Get cloud.google.com/gke-accelerator from nodeSelector
        gke_accelerator = node_selectors.get("cloud.google.com/gke-accelerator")
        if not gke_accelerator:
            return

        # Get ray.io/accelerator-type from required_labels or labels
        ray_accelerator = None
        if self.required_labels and RAY_TPU_ACCELERATOR_LABEL in self.required_labels:
            ray_accelerator = self.required_labels[RAY_TPU_ACCELERATOR_LABEL]
        elif self.labels and RAY_TPU_ACCELERATOR_LABEL in self.labels:
            ray_accelerator = self.labels[RAY_TPU_ACCELERATOR_LABEL]

        if not ray_accelerator:
            return

        # Both are specified - check for consistency
        # Normalize gke-accelerator by removing common prefixes and extracting GPU identifier
        # e.g., "nvidia-tesla-t4" -> "t4", "nvidia-a100-80gb" -> "a100"
        gke_normalized = gke_accelerator.lower()
        for prefix in ["nvidia-", "tesla-", "geforce-", "quadro-"]:
            gke_normalized = gke_normalized.replace(prefix, "")
        # Remove common suffixes like "-80gb", "-40gb", "-sxm", "-pcie"
        for suffix in ["-80gb", "-40gb", "-sxm4", "-sxm", "-pcie", "-nvlink"]:
            gke_normalized = gke_normalized.replace(suffix, "")
        gke_normalized = gke_normalized.strip("-").replace("-", "").replace("_", "")

        # Normalize ray accelerator type
        ray_normalized = ray_accelerator.lower().replace("-", "").replace("_", "")

        # Check if there's any overlap - the normalized values should match or one should contain the other
        # This handles cases like:
        #   - ray: "T4" vs gke: "nvidia-tesla-t4" -> both normalize to contain "t4"
        #   - ray: "A100" vs gke: "nvidia-a100-80gb" -> both normalize to contain "a100"
        is_consistent = (
            gke_normalized in ray_normalized
            or ray_normalized in gke_normalized
            or gke_normalized == ray_normalized
        )

        if not is_consistent:
            warnings.warn(
                f"Potentially inconsistent accelerator types: 'ray.io/accelerator-type: {ray_accelerator}' "
                f"may not match 'cloud.google.com/gke-accelerator: {gke_accelerator}' in nodeSelector. "
                f"Please ensure both refer to the same GPU type.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_tpu_node_selectors(
        self, advanced_instance_config: Optional[AdvancedInstanceConfigDict]
    ):
        """Validate that required GKE TPU configuration is present when TPU is specified.

        When using TPUs on GKE, the following are required:
        - tpu_hosts: Specifies single-host (1) or multi-host (>1) TPU configuration
        - Either explicit node selectors OR labels/required_labels that can be used to derive them:
          Option 1 (explicit node selectors):
            - cloud.google.com/gke-tpu-topology: Specifies the TPU topology (e.g., '2x2', '4x4')
            - cloud.google.com/gke-tpu-accelerator: Specifies the TPU type (e.g., 'tpu-v5-lite-podslice')
          Option 2 (labels or required_labels - node selectors will be derived):
            - ray.io/accelerator-type: TPU accelerator type (e.g., 'TPU-V6E')
            - ray.io/tpu-topology: TPU topology (e.g., '2x2')

        Note: TPU labels must not be specified in both labels and required_labels.

        See: https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus
        """
        # Check if TPU is configured in required_resources
        if self.required_resources is None:
            return

        pr = self.required_resources
        if isinstance(pr, dict):
            tpu_count = pr.get("TPU") or pr.get("tpu") or 0
            tpu_hosts = pr.get("tpu_hosts") or 0
        else:
            tpu_count = pr.TPU or 0
            tpu_hosts = pr.tpu_hosts or 0

        if not tpu_count or tpu_count <= 0:
            return

        # TPU is configured, validate tpu_hosts is specified
        if not tpu_hosts or tpu_hosts <= 0:
            raise ValueError(
                f"TPU configuration requires 'tpu_hosts' in 'required_resources' to specify "
                f"single-host (tpu_hosts: 1) or multi-host (tpu_hosts: >1) TPU configuration. "
                f"Example:\n"
                f"  required_resources:\n"
                f"    CPU: 7\n"
                f"    memory: 12Gi\n"
                f"    TPU: 4\n"
                f"    accelerator: TPU-V5E\n"
                f"    tpu_hosts: 4  # Required for TPU\n"
                f"For more information on TPU configuration, see: {TPU_DOCS_URL}"
            )

        # Extract node selectors from advanced_instance_config
        node_selectors = {}
        if advanced_instance_config:
            spec = advanced_instance_config.get("spec", {})
            if isinstance(spec, dict):
                node_selectors = spec.get("nodeSelector", {})

        # Check if labels can be used to derive the node selectors
        labels = self.labels or {}
        required_labels = self.required_labels or {}

        # Helper to check for TPU labels in a label dict
        def has_tpu_labels(label_dict: Dict[str, str]) -> tuple:
            has_accel = RAY_TPU_ACCELERATOR_LABEL in label_dict and label_dict[
                RAY_TPU_ACCELERATOR_LABEL
            ].upper().startswith("TPU")
            has_topo = RAY_TPU_TOPOLOGY_LABEL in label_dict
            return has_accel, has_topo

        labels_accel, labels_topo = has_tpu_labels(labels)
        req_labels_accel, req_labels_topo = has_tpu_labels(required_labels)

        # Check for conflicts: TPU labels should not be in both labels and required_labels
        conflicting_labels = []
        if labels_accel and req_labels_accel:
            conflicting_labels.append(RAY_TPU_ACCELERATOR_LABEL)
        if labels_topo and req_labels_topo:
            conflicting_labels.append(RAY_TPU_TOPOLOGY_LABEL)

        if conflicting_labels:
            conflict_str = ", ".join(f"'{label}'" for label in conflicting_labels)
            raise ValueError(
                f"TPU configuration labels {conflict_str} cannot be specified in both "
                f"'labels' and 'required_labels'. Please specify them in only one place."
            )

        # If both TPU labels are present (in either labels or required_labels),
        # node selectors will be derived by the backend
        has_tpu_accelerator_label = labels_accel or req_labels_accel
        has_tpu_topology_label = labels_topo or req_labels_topo

        if has_tpu_accelerator_label and has_tpu_topology_label:
            return

        # Check for missing node selectors (only if labels are not provided)
        missing_selectors = [
            selector
            for selector in REQUIRED_TPU_NODE_SELECTORS
            if selector not in node_selectors
        ]

        if missing_selectors:
            missing_str = ", ".join(f"'{s}'" for s in missing_selectors)
            raise ValueError(
                f"TPU configuration requires either:\n"
                f"1. Node selectors in 'advanced_instance_config.spec.nodeSelector': {missing_str}\n"
                f"   Example:\n"
                f"     advanced_instance_config:\n"
                f"       spec:\n"
                f"         nodeSelector:\n"
                f"           cloud.google.com/gke-tpu-topology: '4x4'\n"
                f"           cloud.google.com/gke-tpu-accelerator: 'tpu-v5-lite-podslice'\n"
                f"2. OR labels/required_labels that will be used to derive node selectors:\n"
                f"   Example:\n"
                f"     labels:\n"
                f"       ray.io/accelerator-type: TPU-V6E\n"
                f"       ray.io/tpu-topology: 2x2\n"
                f"For more information on TPU configuration, see: {TPU_DOCS_URL}"
            )

    flags: Optional[Dict[str, Any]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Node-level flags specifying advanced or experimental options.",
            "customer_hosted_only": False,
        },
    )

    def _validate_flags(self, flags: Optional[Dict[str, Any]]):
        if flags is None:
            return

        if not isinstance(flags, dict):
            raise TypeError("'flags' must be a dict")

    cloud_deployment: Union[CloudDeployment, Dict[str, str], None] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Cloud deployment selectors for a node group; one or more selectors may be passed to target a specific deployment from all of a cloud's deployments.",
            "customer_hosted_only": False,
        },
    )

    def _validate_cloud_deployment(
        self, cloud_deployment: Union[CloudDeployment, Dict[str, str], None]
    ) -> Optional[CloudDeployment]:
        if cloud_deployment is None:
            return None
        if isinstance(cloud_deployment, dict):
            cloud_deployment = CloudDeployment.from_dict(cloud_deployment)
        if not isinstance(cloud_deployment, CloudDeployment):
            raise TypeError(
                "'cloud_deployment' must be a CloudDeployment or corresponding dict"
            )
        return cloud_deployment


@dataclass(frozen=True)
class HeadNodeConfig(_NodeConfig):
    """Configuration options for the head node of a cluster."""

    __doc_py_example__ = """
from anyscale.compute_config.models import ComputeConfig, HeadNodeConfig

config = ComputeConfig(
    head_node=HeadNodeConfig(
        instance_type="m5.8xlarge",
    ),
)
"""

    __doc_yaml_example__ = """
head_node:
  instance_type: m5.8xlarge
"""


class MarketType(ModelEnum):
    """Market type of instances to use (on-demand vs. spot)."""

    ON_DEMAND = "ON_DEMAND"
    SPOT = "SPOT"
    PREFER_SPOT = "PREFER_SPOT"

    __docstrings__: ClassVar[Dict[str, str]] = {
        ON_DEMAND: "Use on-demand instances only.",
        SPOT: "Use spot instances only.",
        PREFER_SPOT: (
            "Prefer to use spot instances, but fall back to on-demand if necessary. "
            "If on-demand instances are running and spot instances become available, "
            "the on-demand instances will be evicted and replaced with spot instances."
        ),
    }


@dataclass(frozen=True)
class WorkerNodeGroupConfig(_NodeConfig):
    """Configuration options for a worker node group in a cluster.

    Clusters can have multiple worker node groups that use different instance types or configurations.
    """

    __doc_py_example__ = """
from anyscale.compute_config.models import ComputeConfig, MarketType, WorkerNodeGroupConfig

config = ComputeConfig(
    worker_nodes=[
        WorkerNodeGroupConfig(
            instance_type="m5.8xlarge",
            min_nodes=5,
            max_nodes=5,
        ),
        WorkerNodeGroupConfig(
            instance_type="m5.4xlarge",
            min_nodes=1,
            max_nodes=10,
            market_type=MarketType.SPOT,
        ),
    ],
)
"""

    __doc_yaml_example__ = """
worker_nodes:
- instance_type: m5.8xlarge
  min_nodes: 5
  max_nodes: 5
- instance_type: m5.4xlarge
  min_nodes: 1
  max_nodes: 10
  market_type: SPOT
"""

    name: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Unique name of this worker group. Defaults to a human-friendly representation of the instance type."
        },
    )

    def _validate_name(self, name: Optional[str]) -> str:
        # Default name to the instance type if not specified.
        if name is None:
            # For free pod shapes (custom instance types), generate a unique name
            # from required resources to allow multiple custom worker groups
            if self.instance_type == "custom" and self.required_resources is not None:
                pr = self.required_resources
                # Get accelerator from required_resources, or fall back to
                # required_labels/labels if not specified (ray.io/accelerator-type)
                accelerator = pr.accelerator
                if not accelerator:
                    accelerator = _extract_accelerator_from_labels(
                        self.labels, self.required_labels
                    )
                name = _generate_free_pod_shape_name(
                    cpu=pr.CPU,
                    memory=pr.memory,
                    gpu=pr.GPU,
                    accelerator=accelerator,
                    tpu=pr.TPU,
                    tpu_hosts=pr.tpu_hosts,
                    cpu_architecture=pr.cpu_architecture,
                )
            else:
                name = self.instance_type

        if not isinstance(name, str):
            raise TypeError("'name' must be a string")
        if len(name) == 0:
            raise ValueError("'name' cannot be empty")

        return name

    min_nodes: int = field(
        default=0,
        metadata={
            "docstring": "Minimum number of nodes of this type that will be kept running in the cluster."
        },
    )

    def _validate_min_nodes(self, min_nodes: int):
        if not isinstance(min_nodes, int):
            raise TypeError("'min_nodes' must be an int")
        if min_nodes < 0:
            raise ValueError("'min_nodes' must be >= 0")

    max_nodes: int = field(
        default=10,
        metadata={
            "docstring": "Maximum number of nodes of this type that can be running in the cluster."
        },
    )

    def _validate_max_nodes(self, max_nodes: int):
        if not isinstance(max_nodes, int):
            raise TypeError("'max_nodes' must be an int")
        if max_nodes < 1:
            raise ValueError("'max_nodes' must be >= 1")
        if max_nodes < self.min_nodes:
            raise ValueError(f"'max_nodes' must be >= 'min_nodes' ({self.min_nodes})")

    market_type: Union[str, MarketType] = field(
        default=MarketType.ON_DEMAND,
        metadata={
            "docstring": "The type of instances to use (see `MarketType` enum values for details).",
            "customer_hosted_only": True,
        },
    )

    def _validate_market_type(self, market_type: Union[str, MarketType]) -> MarketType:
        if isinstance(market_type, str):
            # This will raise a ValueError if the market_type is unrecognized.
            market_type = MarketType(market_type)
        elif not isinstance(market_type, MarketType):
            raise TypeError("'market_type' must be a MarketType.")

        return market_type


@dataclass(frozen=True)
class ComputeConfig(ModelBase):
    """Compute configuration for instance types and cloud resources for a cluster with a single cloud resource."""

    __doc_py_example__ = """
from anyscale.compute_config.models import (
    ComputeConfig, HeadNodeConfig, MarketType, WorkerNodeGroupConfig
)

config = ComputeConfig(
    cloud="my-cloud",
    head_node=HeadNodeConfig(
        instance_type="m5.8xlarge",
    ),
    worker_nodes=[
        WorkerNodeGroupConfig(
            instance_type="m5.8xlarge",
            min_nodes=5,
            max_nodes=5,
        ),
        WorkerNodeGroupConfig(
            instance_type="m5.4xlarge",
            min_nodes=1,
            max_nodes=10,
            market_type=MarketType.SPOT,
        ),
    ],
)
"""

    __doc_yaml_example__ = """
cloud: my-cloud
zones: # (Optional) Defaults to to all zones in a region.
  - us-west-2a
  - us-west-2b
head_node:
  instance_type: m5.8xlarge
worker_nodes:
- instance_type: m5.8xlarge
  min_nodes: 5
  max_nodes: 5
  market_type: PREFER_SPOT # (Optional) Defaults to ON_DEMAND
- instance_type: g5.4xlarge
  min_nodes: 1
  max_nodes: 10
  market_type: SPOT # (Optional) Defaults to ON_DEMAND
min_resources: # (Optional) Defaults to no minimum.
  CPU: 1
  GPU: 1
  CUSTOM_RESOURCE: 0
max_resources: # (Optional) Defaults to no maximum.
  CPU: 6
  GPU: 10
  CUSTOM_RESOURCE: 10
enable_cross_zone_scaling: true # (Optional) Defaults to false.
advanced_instance_config: # (Optional) Defaults to no advanced configurations.
  # AWS specific configuration example
  BlockDeviceMappings:
    - DeviceName: DEVICE_NAME
      Ebs:
        VolumeSize: VOLUME_SIZE
        DeleteOnTermination: DELETE_ON_TERMINATION
  IamInstanceProfile:
    Arn: IAM_INSTANCE_PROFILE_ARN
  NetworkInterfaces:
    - SubnetId: SUBNET_ID
      Groups:
        - SECURITY_GROUP_ID
      AssociatePublicIpAddress: ASSOCIATE_PUBLIC_IP
  TagSpecifications:
    - ResourceType: RESOURCE_TYPE
      Tags:
        - Key: TAG_KEY
          Value: TAG_VALUE
  # GCP specific configuration example
  instance_properties:
    disks:
      - boot: BOOT_OPTION
        auto_delete: AUTO_DELETE_OPTION
        initialize_params:
          disk_size_gb: DISK_SIZE_GB
    service_accounts:
      - email: SERVICE_ACCOUNT_EMAIL
        scopes:
          - SCOPE_URL
    network_interfaces:
      - subnetwork: SUBNETWORK_URL
        access_configs:
          - type: ACCESS_CONFIG_TYPE
    labels:
      LABEL_KEY: LABEL_VALUE
"""

    cloud: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace)."
        },
    )

    def _validate_cloud(self, cloud: Optional[str]):
        if cloud is not None and not isinstance(cloud, str):
            raise TypeError("'cloud' must be a string")

    cloud_resource: Optional[str] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "The cloud resource to use for this workload. Defaults to the primary cloud resource of the Cloud.",
            "customer_hosted_only": True,
        },
    )

    def _validate_cloud_resource(self, cloud_resource: Optional[str]):
        if cloud_resource is not None and not isinstance(cloud_resource, str):
            raise TypeError("'cloud_resource' must be a string")

    head_node: Union[HeadNodeConfig, Dict, None] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Configuration options for the head node of the cluster. Defaults to the cloud's default head node configuration."
        },
    )

    def _validate_head_node(
        self, head_node: Union[HeadNodeConfig, Dict, None]
    ) -> Optional[HeadNodeConfig]:
        if head_node is None:
            return None

        if isinstance(head_node, dict):
            head_node = HeadNodeConfig.from_dict(head_node)
        if not isinstance(head_node, HeadNodeConfig):
            raise TypeError(
                "'head_node' must be a HeadNodeConfig or corresponding dict"
            )

        return head_node

    worker_nodes: Optional[List[Union[WorkerNodeGroupConfig, Dict]]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Configuration options for the worker nodes of the cluster. If not provided, worker nodes will be automatically selected based on logical resource requests. To use a head-node only cluster, pass `[]` here."
        },
    )

    def _validate_worker_nodes(
        self, worker_nodes: Optional[List[Union[WorkerNodeGroupConfig, Dict]]]
    ) -> Optional[List[WorkerNodeGroupConfig]]:
        if worker_nodes is None:
            return None

        if not isinstance(worker_nodes, list) or not all(
            isinstance(c, (dict, WorkerNodeGroupConfig)) for c in worker_nodes
        ):
            raise TypeError(
                "'worker_nodes' must be a list of WorkerNodeGroupConfigs or corresponding dicts"
            )

        duplicate_names = set()
        name_counts: DefaultDict[str, int] = defaultdict(int)
        worker_node_models: List[WorkerNodeGroupConfig] = []
        for node in worker_nodes:
            parsed_node = (
                WorkerNodeGroupConfig.from_dict(node)
                if isinstance(node, dict)
                else node
            )
            assert isinstance(parsed_node, WorkerNodeGroupConfig)
            worker_node_models.append(parsed_node)
            name = parsed_node.name
            assert name is not None
            name_counts[name] += 1
            if name_counts[name] > 1:
                duplicate_names.add(name)

        if duplicate_names:
            raise ValueError(
                f"'worker_nodes' names must be unique, but got duplicate names: {duplicate_names}"
            )

        return worker_node_models

    min_resources: Optional[ResourceDict] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Total minimum logical resources across all nodes in the cluster. Resources omitted from this field have no minimum.",
            "customer_hosted_only": False,
        },
    )

    def _validate_min_resources(self, min_resources: Optional[ResourceDict]):
        _validate_resource_dict(min_resources, field_name="min_resources")

    max_resources: Optional[ResourceDict] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Total maximum logical resources across all nodes in the cluster. Resources omitted from this field have no maximum.",
            "customer_hosted_only": False,
        },
    )

    def _validate_max_resources(self, max_resources: Optional[ResourceDict]):
        _validate_resource_dict(max_resources, field_name="max_resources")

    zones: Optional[List[str]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Availability zones to consider for this cluster. Defaults to all zones in the cloud's region. By default all instances with user workloads scheduled on them will run in the same zone to save cost, unless `enable_cross_zone_scaling` is set.",
            "customer_hosted_only": True,
        },
    )

    def _validate_zones(self, zones: Optional[List[str]]):
        if zones is None:
            return
        if not isinstance(zones, list) or not all(isinstance(z, str) for z in zones):
            raise TypeError("'zones' must be a List[str]")
        if len(zones) == 0:
            raise ValueError(
                "'zones' must not be an empty list. Set `None` to default to all zones."
            )

    enable_cross_zone_scaling: bool = field(
        default=False,
        repr=False,
        metadata={
            "docstring": "Allow instances in the cluster to be run across multiple zones. This is recommended when running production services (for fault-tolerance in a zone failure scenario). It is not recommended for workloads that have a large amount of inter-zone communication due to the possibility of higher costs and degraded performance. When false, all instances with user workloads scheduled on them (e.g. all worker nodes in multi-node clusters) will run in the same zone to save cost.",
            "customer_hosted_only": True,
        },
    )

    def _validate_enable_cross_zone_scaling(self, enable_cross_zone_scaling: bool):
        if not isinstance(enable_cross_zone_scaling, bool):
            raise TypeError("'enable_cross_zone_scaling' must be a boolean")

    advanced_instance_config: Optional[AdvancedInstanceConfigDict] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Advanced instance configurations that will be passed through to the cloud provider.",
            "customer_hosted_only": True,
        },
    )

    def _validate_advanced_instance_config(
        self, advanced_instance_config: Optional[AdvancedInstanceConfigDict],
    ):
        _validate_advanced_instance_config_dict(advanced_instance_config)

    flags: Optional[Dict[str, Any]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Cluster-level flags specifying advanced or experimental options.",
            "customer_hosted_only": False,
        },
    )

    def _validate_flags(self, flags: Optional[Dict[str, Any]]):
        if flags is None:
            return

        if not isinstance(flags, dict):
            raise TypeError("'flags' must be a dict")

    auto_select_worker_config: bool = field(
        default=False,
        repr=False,
        metadata={
            "docstring": "Allow worker groups to be automatically configured based on the workload's logical resource requests. When false, worker groups must be explicitly configured.",
        },
    )

    def _validate_auto_select_worker_config(self, auto_select_worker_config: bool):
        if not isinstance(auto_select_worker_config, bool):
            raise TypeError("'auto_select_worker_config' must be a boolean")
        if auto_select_worker_config and self.worker_nodes is not None:
            raise ValueError(
                "'auto_select_worker_config' must be false when 'worker_nodes' are provided"
            )


@dataclass(frozen=True)
class MultiResourceComputeConfig(ModelBase):
    """EXPERIMENTAL. Compute configuration for a cluster with multiple possible cloud resources."""

    __doc_py_example__ = """
from anyscale.compute_config.models import (
    MultiResourceComputeConfig, ComputeConfig, HeadNodeConfig, WorkerNodeGroupConfig
)
config = MultiResourceComputeConfig(
    cloud="my-cloud",
    configs=[
        ComputeConfig(
            cloud_resource="vm-aws-us-west-1",
            head_node=HeadNodeConfig(
                instance_type="m5.2xlarge",
            ),
            worker_nodes=[
                WorkerNodeGroupConfig(
                    instance_type="m5.4xlarge",
                    min_nodes=1,
                    max_nodes=10,
                ),
            ],
        ),
        ComputeConfig(
            cloud_resource="vm-aws-us-west-2",
            head_node=HeadNodeConfig(
                instance_type="m5.2xlarge",
            ),
            worker_nodes=[
                WorkerNodeGroupConfig(
                    instance_type="m5.4xlarge",
                    min_nodes=1,
                    max_nodes=10,
                ),
            ],
        )
    ]
)
"""

    __doc_yaml_example__ = """
cloud: my-cloud
configs:
- cloud_resource: vm-aws-us-west-1
  head_node:
    instance_type: m5.2xlarge
  worker_nodes:
  - instance_type: m5.4xlarge
    min_nodes: 1
    max_nodes: 10
- cloud_resource: vm-aws-us-west-2
  head_node:
    instance_type: m5.2xlarge
  worker_nodes:
  - instance_type: m5.4xlarge
    min_nodes: 1
    max_nodes: 10
"""
    cloud: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "The Anyscale Cloud to run this workload on. If not provided, the organization default will be used (or, if running in a workspace, the cloud of the workspace)."
        },
    )

    def _validate_cloud(self, cloud: Optional[str]):
        if cloud is not None and not isinstance(cloud, str):
            raise TypeError("'cloud' must be a string")

    configs: List[Union[ComputeConfig, Dict]] = field(
        default_factory=list,
        repr=False,
        metadata={
            "docstring": "List of compute configurations, one for each cloud resource.",
            "customer_hosted_only": True,
        },
    )

    def _validate_configs(
        self, configs: List[Union[ComputeConfig, Dict]]
    ) -> List[ComputeConfig]:
        if not isinstance(configs, list) or not all(
            isinstance(c, (dict, ComputeConfig)) for c in configs
        ):
            raise TypeError(
                "'configs' must be a list of ComputeConfigs or corresponding dicts"
            )

        config_models: List[ComputeConfig] = []
        unique_clouds = set()
        unique_resources = set()
        for config in configs:
            parsed_config = (
                ComputeConfig.from_dict(config) if isinstance(config, dict) else config
            )
            assert isinstance(parsed_config, ComputeConfig)
            config_models.append(parsed_config)

            if parsed_config.cloud:
                unique_clouds.add(parsed_config.cloud)

            unique_resources.add(parsed_config.cloud_resource)

        if len(unique_clouds) > 1:
            raise ValueError("'cloud' must be the same for all configs.")

        if len(unique_resources) != len(configs):
            raise ValueError(
                "'cloud_resource' must be unique for each compute configuration."
            )

        if len(configs) == 0:
            raise ValueError(
                "'configs' must include at least one compute configuration."
            )

        return config_models

    flags: Optional[Dict[str, Any]] = field(
        default=None,
        repr=False,
        metadata={
            "docstring": "Flags specifying advanced or experimental options that should be applied to all cloud resources. Flags specified in the individual compute configurations will override these flags.",
        },
    )

    def _validate_flags(self, flags: Optional[Dict[str, Any]]):
        if flags is None:
            return

        if not isinstance(flags, dict):
            raise TypeError("'flags' must be a dict")


ComputeConfigType = Union[ComputeConfig, MultiResourceComputeConfig]


def compute_config_type_from_yaml(config_file: str) -> ComputeConfigType:
    """
    Parse a YAML compute config file into either a ComputeConfig or MultiResourceComputeConfig.
    """
    error_message = f"Could not parse config file '{config_file}' as a ComputeConfig or MultiResourceComputeConfig:\n"

    try:
        return ComputeConfig.from_yaml(config_file)
    except Exception as e:  # noqa: BLE001
        error_message += f"ComputeConfig: {e}\n"

    try:
        return MultiResourceComputeConfig.from_yaml(config_file)
    except Exception as e:  # noqa: BLE001
        error_message += f"MultiResourceComputeConfig: {e}\n"

    raise TypeError(error_message.rstrip())


def compute_config_type_from_dict(config_dict: Dict) -> ComputeConfigType:
    """
    Parse a compute config dict into either a ComputeConfig or MultiResourceComputeConfig.
    """
    error_message = f"Could not parse config dict '{config_dict}' as a ComputeConfig or MultiResourceComputeConfig:\n"

    try:
        return ComputeConfig.from_dict(config_dict)
    except Exception as e:  # noqa: BLE001
        error_message += f"ComputeConfig: {e}\n"

    try:
        return MultiResourceComputeConfig.from_dict(config_dict)
    except Exception as e:  # noqa: BLE001
        error_message += f"MultiResourceComputeConfig: {e}\n"

    raise TypeError(error_message.rstrip())


@dataclass(frozen=True)
class ComputeConfigVersion(ModelBase):
    """Details of a created version of a compute config.

    Includes the config options and metadata such as the name, version, and ID.
    """

    __doc_py_example__ = """
import anyscale
from anyscale.compute_config.models import (
    ComputeConfigVersion
)

version: ComputeConfigVersion = anyscale.compute_config.get("my-compute-config")
"""

    __doc_cli_example__ = """\
$ anyscale compute-config get -n my-compute-config
name: my-compute-config:1
id: cpt_r4b4b3621rl3uggg7llj3mvme6
config:
  cloud: my-cloud
  head_node:
    instance_type: m5.8xlarge
  worker_nodes:
  - instance_type: m5.8xlarge
    min_nodes: 5
    max_nodes: 5
  - instance_type: m5.4xlarge
    min_nodes: 1
    max_nodes: 10
    market_type: SPOT
"""

    name: str = field(
        metadata={
            "docstring": "Name of the compute config including the version tag, i.e., 'name:version'."
        }
    )

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")

        if not name.count(":") == 1:
            raise ValueError("'name' must be in the format: '<name>:<version>'.")

    id: str = field(metadata={"docstring": "Unique ID of the compute config."})

    def _validate_id(self, id: str):  # noqa: A002
        if not isinstance(id, str):
            raise TypeError("'id' must be a string.")

    config: Optional[ComputeConfigType] = field(
        default=None, metadata={"docstring": "The compute configuration."},
    )

    def _validate_config(self, config: Optional[ComputeConfigType]):
        if config is not None and not isinstance(
            config, (ComputeConfig, MultiResourceComputeConfig)
        ):
            raise TypeError(
                "'config' must be a ComputeConfig or MultiResourceComputeConfig"
            )


@dataclass(frozen=True)
class ComputeConfigListResult(ModelBase):
    """Result object returned by anyscale.compute_config.list().

    Contains a list of compute configs matching the query, along with pagination metadata.
    """

    __doc_py_example__ = """
import anyscale
from anyscale.compute_config.models import ComputeConfigListResult

# List compute configs with pagination
result: ComputeConfigListResult = anyscale.compute_config.list(
    cloud_name="aws-prod",
    sort_by="created_at",
    max_items=10
)

# Access results
print(f"Found {result.count} compute configs")
for config in result.results:
    print(f"  - {config.name}:{config.version}")

# Handle pagination
if result.next_token:
    next_page = anyscale.compute_config.list(
        cloud_name="aws-prod",
        max_items=10,
        next_token=result.next_token
    )
"""

    results: List[Any] = field(
        default_factory=list,
        metadata={
            "docstring": "List of compute config objects matching the query. Each item is a ClusterCompute object with details about the compute configuration."
        },
    )

    def _validate_results(self, results: List[Any]):
        if not isinstance(results, list):
            raise TypeError("'results' must be a list.")

    next_token: Optional[str] = field(
        default=None,
        metadata={
            "docstring": "Pagination token for fetching the next page of results. If None, there are no more results available."
        },
    )

    def _validate_next_token(self, next_token: Optional[str]):
        if next_token is not None and not isinstance(next_token, str):
            raise TypeError("'next_token' must be a string or None.")

    count: int = field(
        default=0,
        metadata={
            "docstring": "Number of results returned in this response. Useful for checking if there are results without iterating through the list."
        },
    )

    def _validate_count(self, count: int):
        if not isinstance(count, int):
            raise TypeError("'count' must be an integer.")
        if count < 0:
            raise ValueError("'count' must be non-negative.")
