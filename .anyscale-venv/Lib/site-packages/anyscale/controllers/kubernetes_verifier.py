"""
Kubernetes Cloud Deployment Verifier

Handles verification of Kubernetes-based cloud deployments including:
- Operator pod health and connectivity
- File storage (CSI drivers, PVCs, NFS)
- Network connectivity
- Gateway support
- Nginx ingress controller
"""


from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from typing import Dict, List, Optional

import click
import requests

from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models.cloud_deployment import CloudDeployment
from anyscale.client.openapi_client.models.cloud_providers import CloudProviders
from anyscale.client.openapi_client.models.file_storage import FileStorage
from anyscale.client.openapi_client.models.kubernetes_config import (
    KubernetesConfig as OpenAPIKubernetesConfig,
)
from anyscale.controllers.cloud_file_storage_utils import verify_file_storage_exists


# =============================================================================
# CONSTANTS
# =============================================================================

# Operator configuration
OPERATOR_HEALTH_PORT = 2113
OPERATOR_CONFIG_ENDPOINT = "/config"
OPERATOR_HEALTH_ENDPOINT = "/healthz/run"
DEFAULT_OPERATOR_NAMESPACE = "anyscale-operator"

# Network and timing configuration
PORT_FORWARD_WAIT_TIME = 3  # seconds to wait for port forward to establish
HTTP_REQUEST_TIMEOUT = 10  # seconds for HTTP requests to operator
PORT_FORWARD_TERMINATION_TIMEOUT = 5  # seconds to wait for graceful termination

OPERATOR_LABEL_SELECTOR = "app=anyscale-operator"

# Gateway resource types to check
GATEWAY_RESOURCE_TYPES = [
    "gateway.gateway",  # Gateway API v1
    "gateways.gateway.networking.k8s.io",  # Full API path
    "gateway",  # Short name
    "gw",  # Common alias
]

# NGINX ingress controller configurations
NGINX_INGRESS_CONFIGS = [
    {"namespace": "ingress-nginx", "label": "app.kubernetes.io/name=ingress-nginx"},
    {"namespace": "nginx-ingress", "label": "app=nginx-ingress"},
    {"namespace": "kube-system", "label": "app.kubernetes.io/name=ingress-nginx"},
    {"namespace": "default", "label": "app=nginx-ingress"},
]

# Ingress controller name patterns for fallback search
INGRESS_CONTROLLER_KEYWORDS = [
    "ingress",
    "haproxy",
    "traefik",
    "contour",
    "ambassador",
    "istio-gateway",
    "nginx",
]

# kubectl binary search paths
KUBECTL_COMMON_PATHS = [
    "/usr/local/bin/kubectl",
    "/usr/bin/kubectl",
    "/bin/kubectl",
    "/opt/homebrew/bin/kubectl",  # macOS homebrew
    "~/.local/bin/kubectl",  # User local install
]

# Status and result strings
PASSED_STATUS = "PASSED"
FAILED_STATUS = "FAILED"
SKIPPED_STATUS = "SKIPPED"
RUNNING_STATUS = "Running"


# Verification status enum
class VerificationStatus(Enum):
    """Status of a verification check."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# Verification component names (for consistent reporting)
class VerificationComponents:
    OPERATOR_POD_INSTALLED = "Operator Pod Installed"
    OPERATOR_HEALTH = "Operator Health"
    OPERATOR_IDENTITY = "Operator Identity"
    FILE_STORAGE = "File Storage"
    GATEWAY_SUPPORT = "Gateway Support"
    NGINX_INGRESS = "NGINX Ingress"


# =============================================================================
# EXCEPTIONS
# =============================================================================


class KubernetesVerificationError(Exception):
    """Base exception for all Kubernetes verification errors."""


class KubectlError(KubernetesVerificationError):
    """Raised when kubectl commands fail."""

    def __init__(
        self, message: str, command: Optional[str] = None, stderr: Optional[str] = None
    ):
        super().__init__(message)
        self.command = command
        self.stderr = stderr


class KubectlNotFoundError(KubernetesVerificationError):
    """Raised when kubectl binary cannot be found."""


class OperatorPodNotFoundError(KubernetesVerificationError):
    """Raised when the Anyscale operator pod cannot be found."""


class OperatorConnectionError(KubernetesVerificationError):
    """Raised when connection to the operator fails."""

    def __init__(
        self,
        message: str,
        pod_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        super().__init__(message)
        self.pod_name = pod_name
        self.endpoint = endpoint


class PortForwardError(KubernetesVerificationError):
    """Raised when port forwarding to a pod fails."""

    def __init__(
        self, message: str, pod_name: Optional[str] = None, port: Optional[int] = None
    ):
        super().__init__(message)
        self.pod_name = pod_name
        self.port = port


class IdentityVerificationError(KubernetesVerificationError):
    """Raised when operator identity verification fails."""

    def __init__(
        self,
        message: str,
        expected_identity: Optional[str] = None,
        actual_identity: Optional[str] = None,
    ):
        super().__init__(message)
        self.expected_identity = expected_identity
        self.actual_identity = actual_identity


class FileStorageVerificationError(KubernetesVerificationError):
    """Raised when file storage verification fails."""


class GatewayVerificationError(KubernetesVerificationError):
    """Raised when gateway verification fails."""

    def __init__(self, message: str, gateway_name: Optional[str] = None):
        super().__init__(message)
        self.gateway_name = gateway_name


class ResourceNotFoundError(KubernetesVerificationError):
    """Raised when a required Kubernetes resource is not found."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        super().__init__(message)
        self.resource_type = resource_type
        self.resource_name = resource_name
        self.namespace = namespace


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class VerificationResults:
    """Tracks the results of all verification steps."""

    operator_pod_installed: VerificationStatus = VerificationStatus.FAILED
    operator_health: VerificationStatus = VerificationStatus.FAILED
    operator_identity: VerificationStatus = VerificationStatus.FAILED
    file_storage: VerificationStatus = VerificationStatus.FAILED
    gateway_support: VerificationStatus = VerificationStatus.FAILED
    nginx_ingress: VerificationStatus = VerificationStatus.FAILED

    def to_dict(self) -> Dict[str, VerificationStatus]:
        """Convert to dictionary format for reporting."""
        return {
            VerificationComponents.OPERATOR_POD_INSTALLED: self.operator_pod_installed,
            VerificationComponents.OPERATOR_HEALTH: self.operator_health,
            VerificationComponents.OPERATOR_IDENTITY: self.operator_identity,
            VerificationComponents.FILE_STORAGE: self.file_storage,
            VerificationComponents.GATEWAY_SUPPORT: self.gateway_support,
            VerificationComponents.NGINX_INGRESS: self.nginx_ingress,
        }

    @property
    def overall_success(self) -> bool:
        """Return True if all verification steps passed or were skipped."""
        return all(
            status in (VerificationStatus.PASSED, VerificationStatus.SKIPPED)
            for status in [
                self.operator_pod_installed,
                self.operator_health,
                self.operator_identity,
                self.file_storage,
                self.gateway_support,
                self.nginx_ingress,
            ]
        )


@dataclass
class KubernetesConfig:
    """Configuration for Kubernetes cluster access."""

    context: str
    operator_namespace: str

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.context:
            raise ValueError("Kubernetes context cannot be empty")
        if not self.operator_namespace:
            raise ValueError("Operator namespace cannot be empty")


@dataclass
class OperatorHealthData:
    """Data retrieved from operator health endpoint."""

    status_code: int
    response_text: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        """Return True if operator is healthy."""
        return self.status_code == 200


@dataclass
class OperatorConfigData:
    """Data retrieved from operator config endpoint."""

    status_code: int
    response_text: str
    config_data: Optional[Dict] = None
    config_error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Return True if config data is valid."""
        return self.status_code == 200 and self.config_data is not None


@dataclass
class OperatorData:
    """Combined data from operator health and config endpoints."""

    health: OperatorHealthData
    config: OperatorConfigData

    @classmethod
    def from_dict(cls, data: Dict) -> "OperatorData":
        """Create OperatorData from dictionary format used in original code."""
        health = OperatorHealthData(
            status_code=data["health_status"], response_text=data.get("health_response")
        )

        config = OperatorConfigData(
            status_code=data["config_status"],
            response_text=data["config_response"],
            config_data=data.get("config_data"),
            config_error=data.get("config_error"),
        )

        return cls(health=health, config=config)


@dataclass
class GatewayConfig:
    """Gateway configuration from operator."""

    enabled: bool = False
    name: Optional[str] = None

    @classmethod
    def from_operator_config(cls, config_data: Optional[Dict]) -> "GatewayConfig":
        """Extract gateway config from operator configuration."""
        if not config_data:
            return cls()

        gateway_config = config_data.get("gateway", {})
        if not gateway_config:
            return cls()

        return cls(
            enabled=gateway_config.get("enable", False), name=gateway_config.get("name")
        )

    @property
    def requires_verification(self) -> bool:
        """Return True if gateway verification is required."""
        return self.enabled and self.name is not None


# =============================================================================
# KUBECTL OPERATIONS
# =============================================================================


class KubectlOperations:
    """Utility class for executing kubectl commands with consistent error handling."""

    def __init__(self, context: str, logger: BlockLogger):
        self.context = context
        self.log = logger
        self._kubectl_path: Optional[str] = None

    def get_resource(
        self, resource_type: str, name: str, namespace: Optional[str] = None
    ) -> Dict:
        """Get a single Kubernetes resource by name."""
        cmd_args = ["get", resource_type, name, "--context", self.context, "-o", "json"]
        if namespace:
            cmd_args.extend(["-n", namespace])

        try:
            result = self._run_kubectl_command(cmd_args)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            if "not found" in e.stderr.lower():
                raise ResourceNotFoundError(
                    f"{resource_type} '{name}' not found",
                    resource_type=resource_type,
                    resource_name=name,
                    namespace=namespace,
                )
            raise KubectlError(
                f"Failed to get {resource_type} '{name}': {e.stderr}",
                command=" ".join(cmd_args),
                stderr=e.stderr,
            )
        except json.JSONDecodeError as e:
            raise KubectlError(
                f"Invalid JSON response from kubectl: {e}", command=" ".join(cmd_args)
            )

    def list_resources(
        self,
        resource_type: str,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None,
        all_namespaces: bool = False,
    ) -> List[Dict]:
        """List Kubernetes resources with optional filtering."""
        cmd_args = ["get", resource_type, "--context", self.context, "-o", "json"]

        if all_namespaces:
            cmd_args.append("--all-namespaces")
        elif namespace:
            cmd_args.extend(["-n", namespace])

        if label_selector:
            cmd_args.extend(["-l", label_selector])

        try:
            result = self._run_kubectl_command(cmd_args)
            data = json.loads(result.stdout)
            return data.get("items", [])
        except subprocess.CalledProcessError as e:
            raise KubectlError(
                f"Failed to list {resource_type}: {e.stderr}",
                command=" ".join(cmd_args),
                stderr=e.stderr,
            )
        except json.JSONDecodeError as e:
            raise KubectlError(
                f"Invalid JSON response from kubectl: {e}", command=" ".join(cmd_args)
            )

    def get_resource_field(
        self,
        resource_type: str,
        name: str,
        jsonpath: str,
        namespace: Optional[str] = None,
    ) -> str:
        """Get a specific field from a Kubernetes resource using jsonpath."""
        cmd_args = [
            "get",
            resource_type,
            name,
            "--context",
            self.context,
            "-o",
            f"jsonpath={jsonpath}",
        ]
        if namespace:
            cmd_args.extend(["-n", namespace])

        try:
            result = self._run_kubectl_command(cmd_args)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            if "not found" in e.stderr.lower():
                raise ResourceNotFoundError(
                    f"{resource_type} '{name}' not found",
                    resource_type=resource_type,
                    resource_name=name,
                    namespace=namespace,
                )
            raise KubectlError(
                f"Failed to get field from {resource_type} '{name}': {e.stderr}",
                command=" ".join(cmd_args),
                stderr=e.stderr,
            )

    def get_available_contexts(self) -> List[str]:
        """Get list of available kubectl contexts."""
        try:
            result = self._run_kubectl_command(["config", "get-contexts", "-o", "name"])
            contexts = [
                ctx.strip() for ctx in result.stdout.strip().split("\n") if ctx.strip()
            ]
            return contexts
        except subprocess.CalledProcessError as e:
            raise KubectlError(
                f"Failed to get kubectl contexts: {e.stderr}",
                command="kubectl config get-contexts -o name",
                stderr=e.stderr,
            )

    def get_current_context(self) -> Optional[str]:
        """Get the current kubectl context."""
        try:
            result = self._run_kubectl_command(["config", "current-context"])
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            if "current-context is not set" in e.stderr.lower():
                return None
            raise KubectlError(
                f"Failed to get current context: {e.stderr}",
                command="kubectl config current-context",
                stderr=e.stderr,
            )

    def start_port_forward(
        self, pod_name: str, local_port: int, remote_port: int, namespace: str
    ) -> subprocess.Popen:
        """Start port forwarding to a pod."""
        cmd_args = [
            "port-forward",
            "--context",
            self.context,
            "-n",
            namespace,
            pod_name,
            f"{local_port}:{remote_port}",
        ]

        try:
            cmd = self._get_kubectl_cmd(cmd_args)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Create new process group for cleanup
            )
            return process
        except (subprocess.CalledProcessError, OSError) as e:
            raise KubectlError(
                f"Failed to start port forward to {pod_name}: {e}",
                command=" ".join(cmd_args),
            )

    def check_kubectl_available(self) -> bool:
        """Check if kubectl command is available."""
        try:
            self._run_kubectl_command(["version", "--client"])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, KubectlNotFoundError):
            return False

    def get_pod_status(self, pod_name: str, namespace: str) -> str:
        """
        Get pod status phase in specific namespace.

        Args:
            pod_name: Name of the pod
            namespace: Namespace containing the pod

        Returns:
            Pod status phase (e.g., "Running", "Pending") or "unknown" if cannot be determined
        """
        try:
            return self.get_resource_field(
                "pod", pod_name, "{.status.phase}", namespace=namespace
            )
        except (KubectlError, ResourceNotFoundError):
            # Return "unknown" if status cannot be determined
            return "unknown"

    def is_pod_running(self, pod_name: str, namespace: str) -> bool:
        """
        Check if pod is in running state.

        Args:
            pod_name: Name of the pod
            namespace: Namespace containing the pod

        Returns:
            True if pod is running, False otherwise
        """
        try:
            status = self.get_resource_field(
                "pod", pod_name, "{.status.phase}", namespace=namespace
            )
            return status == RUNNING_STATUS
        except (KubectlError, ResourceNotFoundError):
            # Return False if status check fails
            return False

    def _run_kubectl_command(self, args: List[str]) -> subprocess.CompletedProcess:
        """Execute a kubectl command with the given arguments."""
        cmd = self._get_kubectl_cmd(args)
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _get_kubectl_cmd(self, args: List[str]) -> List[str]:
        """Get kubectl command with proper binary path."""
        kubectl_path = self._find_kubectl_binary()
        if not kubectl_path:
            raise KubectlNotFoundError(
                "kubectl command not found. Please install kubectl and ensure it's in your PATH."
            )
        return [kubectl_path] + args

    def _find_kubectl_binary(self) -> Optional[str]:
        """Find kubectl binary in common locations."""
        if self._kubectl_path:
            return self._kubectl_path

        # Try to find kubectl using shutil.which first (respects PATH)
        kubectl_path = shutil.which("kubectl")
        if kubectl_path:
            self._kubectl_path = kubectl_path
            return kubectl_path

        # Try common installation locations
        for path in KUBECTL_COMMON_PATHS:
            expanded_path = os.path.expanduser(path)
            if os.path.isfile(expanded_path) and os.access(expanded_path, os.X_OK):
                self._kubectl_path = expanded_path
                return expanded_path

        return None


# =============================================================================
# OPERATOR VERIFIER
# =============================================================================


class OperatorVerifier:
    """Handles verification of Anyscale operator pod, health, and identity."""

    def __init__(
        self,
        kubectl_ops: KubectlOperations,
        k8s_config: KubernetesConfig,
        logger: BlockLogger,
    ):
        self.kubectl = kubectl_ops
        self.config = k8s_config
        self.log = logger

    def find_operator_pod(self) -> str:
        """Find and verify operator pod is running."""
        try:
            pods = self.kubectl.list_resources(
                "pods",
                namespace=self.config.operator_namespace,
                label_selector=OPERATOR_LABEL_SELECTOR,
            )
        except KubectlError as e:
            raise OperatorPodNotFoundError(f"Failed to list operator pods: {e}")

        if not pods:
            raise OperatorPodNotFoundError(
                "No Anyscale operator pods found. Expected pods with labels like "
                "'app=anyscale-operator'"
            )

        operator_pod = pods[0]["metadata"]["name"]

        if not self.kubectl.is_pod_running(
            operator_pod, self.config.operator_namespace
        ):
            raise OperatorPodNotFoundError(
                f"Operator pod '{operator_pod}' is not running"
            )

        return operator_pod

    def get_operator_data(self, pod_name: str) -> OperatorData:
        """Port forward to operator and fetch both health and config data."""
        try:
            with self._port_forward_to_operator(pod_name) as local_port:
                # Fetch health data
                health_data = self._fetch_health_data(local_port)

                # Fetch config data
                config_data = self._fetch_config_data(local_port)

                return OperatorData(health=health_data, config=config_data)

        except requests.RequestException as e:
            raise OperatorConnectionError(
                f"Cannot connect to operator endpoints: {e}", pod_name=pod_name
            )
        except RuntimeError as e:
            raise PortForwardError(
                f"Port forwarding failed: {e}",
                pod_name=pod_name,
                port=OPERATOR_HEALTH_PORT,
            )

    def verify_operator_health(self, operator_data: OperatorData) -> VerificationStatus:
        """Verify operator health using pre-fetched data."""
        if operator_data.health.is_healthy:
            return VerificationStatus.PASSED
        else:
            self.log.error(
                f"Health check failed - HTTP {operator_data.health.status_code}"
            )
            if operator_data.health.response_text:
                self.log.error(f"Response: {operator_data.health.response_text}")
            return VerificationStatus.FAILED

    def verify_operator_identity(
        self,
        operator_data: OperatorData,
        kubernetes_config: OpenAPIKubernetesConfig,
        cloud_provider: Optional[CloudProviders],
    ) -> VerificationStatus:
        """Verify operator identity using pre-fetched config data."""
        # Validate kubernetes_config contents
        expected_identity = kubernetes_config.anyscale_operator_iam_identity
        if not expected_identity:
            self.log.info(
                "Operator is not using IAM identity - skipping identity verification"
            )
            return VerificationStatus.SKIPPED

        # Validate config response
        if not operator_data.config.is_valid:
            self.log.error(
                f"Config endpoint returned HTTP {operator_data.config.status_code}"
            )
            if operator_data.config.response_text:
                self.log.error(f"Response: {operator_data.config.response_text}")
            return VerificationStatus.FAILED

        # Extract actual identity from config
        if operator_data.config.config_data is None:
            self.log.error("Operator config data is None")
            return VerificationStatus.FAILED

        actual_identity = operator_data.config.config_data.get("iamIdentity")
        if not actual_identity:
            self.log.error("Operator config missing 'iamIdentity' field")
            return VerificationStatus.FAILED

        # Perform identity comparison
        if self._evaluate_identity_match(
            expected_identity, actual_identity, cloud_provider
        ):
            # Get cloud provider string for display
            provider_str = str(cloud_provider) if cloud_provider else "AWS"
            self.log.info(
                f"{provider_str} identity match: Expected identity matches (Expected: {expected_identity})"
            )
            self.log.info("Expected identity matches actual identity")
            return VerificationStatus.PASSED
        else:
            self.log.error("Operator identity mismatch")
            self.log.error(f"Expected: {expected_identity}")
            self.log.error(f"Actual: {actual_identity}")
            return VerificationStatus.FAILED

    @contextmanager
    def _port_forward_to_operator(self, pod_name: str):
        """Context manager that port forwards to operator pod."""
        port_forward_process = None
        local_port = None
        try:
            # Get a free port for port forwarding
            local_port = self._get_free_port()
            self.log.info(f"Using local port {local_port} for port forwarding")

            # Start port forwarding to the pod
            self.log.info(
                f"Starting port forward to pod {pod_name} on port {local_port}:{OPERATOR_HEALTH_PORT}..."
            )

            port_forward_process = self.kubectl.start_port_forward(
                pod_name,
                local_port,
                OPERATOR_HEALTH_PORT,
                self.config.operator_namespace,
            )

            # Wait for port forward to establish
            self.log.info("Waiting for port forward to establish...")
            time.sleep(PORT_FORWARD_WAIT_TIME)

            # Check if port forward process is still running
            if port_forward_process.poll() is not None:
                stderr = (
                    port_forward_process.stderr.read().decode()
                    if port_forward_process.stderr
                    else ""
                )
                raise RuntimeError(f"Port forward failed to start: {stderr}")

            # Yield the local port to the calling function
            yield local_port

        finally:
            # Clean up port forward process
            if port_forward_process and port_forward_process.poll() is None:
                try:
                    # Kill the entire process group to ensure cleanup
                    os.killpg(os.getpgid(port_forward_process.pid), signal.SIGTERM)
                    port_forward_process.wait(timeout=PORT_FORWARD_TERMINATION_TIMEOUT)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    # Force kill if graceful termination fails
                    with suppress(ProcessLookupError):
                        os.killpg(os.getpgid(port_forward_process.pid), signal.SIGKILL)
                except (OSError, ValueError) as e:
                    self.log.warning(f"Port forward cleanup warning: {e}")

    def _get_free_port(self) -> int:
        """Get a random free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _fetch_health_data(self, local_port: int) -> OperatorHealthData:
        """Fetch health data from operator."""
        response = requests.get(
            f"http://localhost:{local_port}{OPERATOR_HEALTH_ENDPOINT}",
            timeout=HTTP_REQUEST_TIMEOUT,
        )

        return OperatorHealthData(
            status_code=response.status_code,
            response_text=response.text if response.status_code != 200 else None,
        )

    def _fetch_config_data(self, local_port: int) -> OperatorConfigData:
        """Fetch config data from operator."""
        max_retries = 6
        retry_delay = 5

        for attempt in range(max_retries):
            response = requests.get(
                f"http://localhost:{local_port}{OPERATOR_CONFIG_ENDPOINT}",
                timeout=HTTP_REQUEST_TIMEOUT,
            )

            config_data = None
            config_error = None

            if response.status_code == 200:
                try:
                    config_data = response.json()

                    # Check if iamIdentity is present in the config
                    # This is necessary because the operator may not have the iamIdentity field
                    # immediately available after startup while it is being bootstrapped (connecting to KCP, storing registry secrets, etc.)
                    if config_data and "iamIdentity" in config_data:
                        return OperatorConfigData(
                            status_code=response.status_code,
                            response_text=response.text,
                            config_data=config_data,
                            config_error=config_error,
                        )

                    # If iamIdentity is missing and we have retries left, wait and retry
                    if attempt < max_retries - 1:
                        self.log.info(
                            f"Operator config endpoint returned 200 but iamIdentity not found. "
                            f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Last attempt and still no iamIdentity, return what we have
                        self.log.warning(
                            f"iamIdentity not found after {max_retries} attempts"
                        )

                except json.JSONDecodeError as e:
                    config_error = str(e)
            else:
                # Non-200 status code, return immediately without retrying
                return OperatorConfigData(
                    status_code=response.status_code,
                    response_text=response.text,
                    config_data=config_data,
                    config_error=config_error,
                )

        # Return the last response (this will have config_data but no iamIdentity)
        return OperatorConfigData(
            status_code=response.status_code,
            response_text=response.text,
            config_data=config_data,
            config_error=config_error,
        )

    def _evaluate_identity_match(
        self,
        expected_identity: str,
        actual_identity: str,
        cloud_provider: Optional[CloudProviders],
    ) -> bool:
        """Evaluate if the operator identity matches expected identity based on cloud provider."""
        if not expected_identity or not actual_identity:
            return False

        # Convert to string for comparison, default to AWS
        cloud_provider_str = str(cloud_provider) if cloud_provider else "AWS"

        # Handle cloud provider specific identity comparison
        if cloud_provider_str == "AWS":
            return self._evaluate_aws_identity(expected_identity, actual_identity)
        elif cloud_provider_str == "GCP":
            return self._evaluate_gcp_identity(expected_identity, actual_identity)
        elif cloud_provider_str == "AZURE":
            return self._evaluate_azure_identity(expected_identity, actual_identity)
        else:
            # For unknown providers, fall back to exact string comparison
            self.log.warning(
                f"Unknown cloud provider '{cloud_provider}', using exact string comparison"
            )
            return expected_identity == actual_identity

    def _evaluate_aws_identity(
        self, expected_identity: str, actual_identity: str
    ) -> bool:
        """Evaluate AWS IAM identity comparison."""
        try:
            # If they're exactly equal, that's fine
            if expected_identity == actual_identity:
                return True

            # Check if actual is an assumed role version of expected role
            if self._is_aws_assumed_role(actual_identity):
                # Extract the role name from both ARNs
                expected_role = self._extract_aws_role_name(expected_identity)
                actual_role = self._extract_aws_role_name_from_assumed_role(
                    actual_identity
                )

                if expected_role and actual_role and expected_role == actual_role:
                    # Also check account ID matches
                    expected_account = self._extract_aws_account_id(expected_identity)
                    actual_account = self._extract_aws_account_id(actual_identity)

                    if expected_account == actual_account:
                        self.log.info(
                            f"AWS identity match: Role '{expected_role}' (account: {expected_account})"
                        )
                        return True

            return False

        except (ValueError, IndexError, AttributeError) as e:
            self.log.error(f"Error evaluating AWS identity: {e}")
            return False

    def _evaluate_gcp_identity(
        self, expected_identity: str, actual_identity: str
    ) -> bool:
        """Evaluate GCP identity comparison."""
        return expected_identity == actual_identity

    def _evaluate_azure_identity(
        self, expected_identity: str, actual_identity: str
    ) -> bool:
        """Evaluate Azure identity comparison."""
        return expected_identity == actual_identity

    def _is_aws_assumed_role(self, arn: str) -> bool:
        """Check if ARN is an assumed role ARN."""
        return arn.startswith("arn:aws:sts:") and ":assumed-role/" in arn

    def _extract_aws_role_name(self, role_arn: str) -> Optional[str]:
        """Extract role name from IAM role ARN."""
        try:
            if ":role/" in role_arn:
                return role_arn.split(":role/")[-1]
            return None
        except (ValueError, IndexError):
            return None

    def _extract_aws_role_name_from_assumed_role(
        self, assumed_role_arn: str
    ) -> Optional[str]:
        """Extract role name from assumed role ARN."""
        try:
            if ":assumed-role/" in assumed_role_arn:
                parts = assumed_role_arn.split(":assumed-role/")[-1].split("/")
                if len(parts) >= 1:
                    return parts[0]  # Role name is first part after assumed-role/
            return None
        except (ValueError, IndexError):
            return None

    def _extract_aws_account_id(self, arn: str) -> Optional[str]:
        """Extract AWS account ID from any ARN."""
        try:
            # ARN format: arn:partition:service:region:account-id:resource
            parts = arn.split(":")
            if len(parts) >= 5:
                return parts[4]
            return None
        except (ValueError, IndexError):
            return None


# =============================================================================
# STORAGE VERIFIER
# =============================================================================


class StorageVerifier:
    """Handles verification of file storage components for Kubernetes deployments."""

    def __init__(
        self,
        kubectl_ops: KubectlOperations,
        k8s_config: KubernetesConfig,
        logger: BlockLogger,
    ):
        self.kubectl = kubectl_ops
        self.config = k8s_config
        self.log = logger

    def verify_file_storage(
        self, file_storage: FileStorage, cloud_deployment: CloudDeployment
    ) -> VerificationStatus:
        """Verify file storage configuration (non-functional checks only).

        Returns:
            VerificationStatus enum value
        """
        self.log.info("Verifying file storage configuration...")
        verification_results = []

        if getattr(file_storage, "csi_ephemeral_volume_driver", None):
            driver_name = file_storage.csi_ephemeral_volume_driver
            if driver_name:
                self.log.info(f"Checking CSI driver: {driver_name}")
                result = self._verify_csi_driver(driver_name)
                verification_results.append(("CSI driver", result))

        if getattr(file_storage, "persistent_volume_claim", None):
            pvc_name = file_storage.persistent_volume_claim
            if pvc_name:
                self.log.info(f"Checking PVC: {pvc_name}")
                result = self._verify_pvc(pvc_name)
                verification_results.append(("PVC", result))

        if getattr(file_storage, "file_storage_id", None):
            self.log.info("Checking NFS file storage exists via cloud provider APIs...")
            try:
                nfs_exists = verify_file_storage_exists(
                    file_storage, cloud_deployment, logger=self.log
                )
                verification_results.append(("NFS", nfs_exists))
            except (ValueError, KeyError, TypeError, ImportError) as e:
                self.log.error(
                    f"Cloud provider API error while verifying file storage: {e}"
                )
                raise RuntimeError(
                    f"Cloud provider API error while verifying file storage: {e}"
                ) from e

        # Return overall status
        if verification_results:
            if all(result for _, result in verification_results):
                return VerificationStatus.PASSED
            else:
                return VerificationStatus.FAILED
        else:
            self.log.info("No file storage components found to verify")
            return VerificationStatus.SKIPPED

    def _verify_csi_driver(self, driver_name: str) -> bool:
        """Check if CSI driver exists on cluster."""
        try:
            driver_info = self.kubectl.get_resource("csidriver", driver_name)

            # Parse driver details for logging
            driver_spec = driver_info.get("spec", {})
            self.log.info(f"CSI driver '{driver_name}' is available")
            self.log.info(
                f"Attach required: {driver_spec.get('attachRequired', 'unknown')}"
            )
            self.log.info(
                f"Pod info on mount: {driver_spec.get('podInfoOnMount', 'unknown')}"
            )
            return True

        except ResourceNotFoundError:
            self.log.error(f"CSI driver '{driver_name}' not found")
            self.log.error("Available CSI drivers:")
            self._list_available_csi_drivers()
            return False

        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to query CSI driver: {e}")
            raise RuntimeError(
                f"kubectl error while verifying CSI driver '{driver_name}': {e}"
            ) from e

    def _verify_pvc(self, pvc_name: str) -> bool:
        """Check if PVC exists and is bound in operator namespace."""
        try:
            pvc_data = self.kubectl.get_resource(
                "pvc", pvc_name, namespace=self.config.operator_namespace
            )

            status = pvc_data.get("status", {})
            phase = status.get("phase")
            capacity = status.get("capacity", {})
            storage_class = pvc_data.get("spec", {}).get("storageClassName")

            if phase == "Bound":
                self.log.info(f"PVC '{pvc_name}' is bound")
                self.log.info(f"Capacity: {capacity.get('storage', 'unknown')}")
                self.log.info(f"Storage class: {storage_class or 'default'}")
                return True
            else:
                self.log.error(
                    f"FAILED: PVC '{pvc_name}' is not bound (status: {phase})"
                )
                return False

        except ResourceNotFoundError:
            self.log.error(
                f"FAILED: PVC '{pvc_name}' not found in namespace '{self.config.operator_namespace}'"
            )
            self.log.error("Available PVCs in namespace:")
            self._list_available_pvcs()
            return False

        except Exception as e:  # noqa: BLE001
            self.log.error(f"FAILED: Failed to check PVC '{pvc_name}': {e}")
            raise RuntimeError(
                f"kubectl error while verifying PVC '{pvc_name}': {e}"
            ) from e

    def _list_available_csi_drivers(self) -> None:
        """List available CSI drivers for troubleshooting."""
        try:
            drivers = self.kubectl.list_resources("csidrivers")
            if drivers:
                for driver in drivers:
                    name = driver.get("metadata", {}).get("name", "unknown")
                    self.log.error(f"  - {name}")
            else:
                self.log.error("  (no CSI drivers found in cluster)")
        except Exception:  # noqa: BLE001
            self.log.error("  (failed to list CSI drivers)")

    def _list_available_pvcs(self) -> None:
        """List available PVCs for troubleshooting."""
        try:
            pvcs = self.kubectl.list_resources(
                "pvcs", namespace=self.config.operator_namespace
            )
            if pvcs:
                for pvc in pvcs:
                    name = pvc.get("metadata", {}).get("name", "unknown")
                    self.log.error(f"  - {name}")
            else:
                self.log.error(
                    f"  (no PVCs found in namespace '{self.config.operator_namespace}')"
                )
        except Exception:  # noqa: BLE001
            self.log.error("  (failed to list PVCs)")


# =============================================================================
# GATEWAY VERIFIER
# =============================================================================


class GatewayVerifier:
    """Handles verification of gateway and ingress components for Kubernetes deployments."""

    def __init__(
        self,
        kubectl_ops: KubectlOperations,
        k8s_config: KubernetesConfig,
        logger: BlockLogger,
    ):
        self.kubectl = kubectl_ops
        self.config = k8s_config
        self.log = logger

    def verify_gateway_support(self, operator_data: OperatorData) -> VerificationStatus:
        """Verify gateway support using pre-fetched config data.

        Returns:
            VerificationStatus enum value
        """
        if not operator_data.config.is_valid:
            self.log.info(
                "Could not retrieve operator configuration - skipping gateway verification"
            )
            return VerificationStatus.SKIPPED

        # Extract gateway configuration from operator data
        gateway_config = GatewayConfig.from_operator_config(
            operator_data.config.config_data
        )

        if not gateway_config.enabled:
            self.log.info(
                "Gateway support is not enabled - skipping gateway verification"
            )
            return VerificationStatus.SKIPPED

        if not gateway_config.requires_verification:
            self.log.error(
                "Gateway is enabled but no gateway name found in operator configuration"
            )
            return VerificationStatus.FAILED

        # Verify gateway exists in cluster
        assert (
            gateway_config.name is not None
        )  # guaranteed by requires_verification check
        if self._verify_gateway_exists(gateway_config.name):
            return VerificationStatus.PASSED
        else:
            return VerificationStatus.FAILED

    def verify_nginx_ingress(self) -> VerificationStatus:
        """Check for NGINX ingress controller (warning only)."""
        try:
            self.log.info("Checking for NGINX ingress controller...")

            # Try different NGINX ingress controller configurations
            for config_dict in NGINX_INGRESS_CONFIGS:
                nginx_pod = self._find_nginx_pod(
                    config_dict["namespace"], config_dict["label"]
                )
                if nginx_pod:
                    if self.kubectl.is_pod_running(nginx_pod, config_dict["namespace"]):
                        self.log.info(
                            f"PASSED: Found running NGINX ingress controller: {nginx_pod} "
                            f"(namespace: {config_dict['namespace']})"
                        )
                        return VerificationStatus.PASSED
                    else:
                        pod_status = self.kubectl.get_pod_status(
                            nginx_pod, config_dict["namespace"]
                        )
                        self.log.warning(
                            f"WARNING: Found NGINX ingress controller '{nginx_pod}' "
                            f"but it's not running (status: {pod_status})"
                        )

            # Try fallback search by name patterns
            if self._find_nginx_by_name_pattern():
                return VerificationStatus.PASSED

            # No NGINX ingress controller found
            self.log.warning("No NGINX ingress controller found")
            self.log.warning("This may impact ingress routing capabilities")
            self.log.warning("Available ingress controllers:")
            self._list_available_ingress_controllers()
            return VerificationStatus.FAILED

        except (KubectlError, ResourceNotFoundError) as e:
            self.log.warning(f"WARNING: Could not verify NGINX ingress controller: {e}")
            raise RuntimeError(
                f"kubectl error during NGINX ingress verification: {e}"
            ) from e

    def _verify_gateway_exists(self, gateway_name: str) -> bool:
        """Verify that the specified gateway exists in the cluster."""
        try:
            # Try to find gateway in common Gateway API resource types
            for resource_type in GATEWAY_RESOURCE_TYPES:
                if self._check_gateway_resource(resource_type, gateway_name):
                    return True

            # If not found in operator namespace, try cluster-wide search
            self.log.info(
                f"Gateway '{gateway_name}' not found in operator namespace, "
                "searching cluster-wide..."
            )
            for resource_type in GATEWAY_RESOURCE_TYPES:
                if self._check_gateway_resource_cluster_wide(
                    resource_type, gateway_name
                ):
                    return True

            self.log.error(f"FAILED: Gateway '{gateway_name}' not found in cluster")
            self.log.error("Available gateways:")
            self._list_available_gateways()
            return False

        except (KubectlError, ResourceNotFoundError) as e:
            self.log.error(f"FAILED: Failed to verify gateway '{gateway_name}': {e}")
            raise RuntimeError(
                f"kubectl error while verifying gateway '{gateway_name}': {e}"
            ) from e

    def _check_gateway_resource(self, resource_type: str, gateway_name: str) -> bool:
        """Check for gateway resource in operator namespace."""
        try:
            gateway_data = self.kubectl.get_resource(
                resource_type, gateway_name, namespace=self.config.operator_namespace
            )

            self.log.info(
                f"PASSED: Gateway '{gateway_name}' found in namespace '{self.config.operator_namespace}'"
            )

            # Log gateway status if available
            status = gateway_data.get("status", {})
            conditions = status.get("conditions", [])
            for condition in conditions:
                if (
                    condition.get("type") == "Ready"
                    and condition.get("status") == "True"
                ):
                    self.log.info("  Status: Ready")
                    break

            return True

        except ResourceNotFoundError:
            return False

    def _check_gateway_resource_cluster_wide(
        self, resource_type: str, gateway_name: str
    ) -> bool:
        """Check for gateway resource cluster-wide."""
        try:
            gateways = self.kubectl.list_resources(resource_type, all_namespaces=True)

            for gateway in gateways:
                if gateway.get("metadata", {}).get("name") == gateway_name:
                    namespace = gateway.get("metadata", {}).get("namespace", "unknown")
                    self.log.info(
                        f"PASSED: Gateway '{gateway_name}' found in namespace '{namespace}'"
                    )
                    return True

            return False

        except Exception:  # noqa: BLE001
            # Broad exception handling for fallback case
            return False

    def _find_nginx_pod(self, namespace: str, label_selector: str) -> Optional[str]:
        """Find NGINX ingress pod by label selector in specific namespace."""
        try:
            pods = self.kubectl.list_resources(
                "pods", namespace=namespace, label_selector=label_selector
            )

            if pods:
                return pods[0]["metadata"]["name"]
            return None

        except Exception:  # noqa: BLE001
            # Broad exception handling for fallback pod discovery
            return None

    def _find_nginx_by_name_pattern(self) -> bool:
        """Find NGINX ingress controller by name pattern across all namespaces."""
        try:
            pods = self.kubectl.list_resources("pods", all_namespaces=True)

            # Look for pods with names containing NGINX and ingress keywords
            for pod in pods:
                metadata = pod.get("metadata", {})
                name = metadata.get("name", "").lower()
                namespace = metadata.get("namespace", "")
                status_phase = pod.get("status", {}).get("phase", "")

                if "nginx" in name and "ingress" in name:
                    if status_phase == RUNNING_STATUS:
                        self.log.info(
                            f"PASSED: Found NGINX ingress controller by name pattern: "
                            f"{metadata['name']} (namespace: {namespace})"
                        )
                        return True
                    else:
                        self.log.warning(
                            f"WARNING: Found NGINX ingress controller '{metadata['name']}' "
                            f"but it's not running (status: {status_phase})"
                        )

            return False

        except Exception:  # noqa: BLE001
            # Broad exception handling for fallback case
            return False

    def _list_available_gateways(self) -> None:
        """List available gateways for troubleshooting."""
        try:
            for resource_type in GATEWAY_RESOURCE_TYPES:
                gateways = self.kubectl.list_resources(
                    resource_type, all_namespaces=True
                )

                if gateways:
                    self.log.error(f"Available {resource_type}:")
                    for gw in gateways:
                        name = gw.get("metadata", {}).get("name", "unknown")
                        self.log.error(f"  - {name}")
                    return

            self.log.error("  (no gateways found in cluster)")

        except Exception:  # noqa: BLE001
            # Broad exception handling for troubleshooting helper
            self.log.error("  (failed to list gateways)")

    def _list_available_ingress_controllers(self) -> None:
        """List available ingress controllers for troubleshooting."""
        try:
            pods = self.kubectl.list_resources("pods", all_namespaces=True)

            ingress_controllers = []
            for pod in pods:
                metadata = pod.get("metadata", {})
                name = metadata.get("name", "").lower()
                namespace = metadata.get("namespace", "")

                # Look for common ingress controller name patterns
                if any(keyword in name for keyword in INGRESS_CONTROLLER_KEYWORDS):
                    ingress_controllers.append(
                        f"{metadata['name']} (namespace: {namespace})"
                    )

            if ingress_controllers:
                for controller in ingress_controllers:
                    self.log.warning(f"  - {controller}")
            else:
                self.log.warning("  (no ingress controllers found)")

        except Exception:  # noqa: BLE001
            # Broad exception handling for troubleshooting helper
            self.log.warning("  (failed to list ingress controllers)")


# =============================================================================
# MAIN VERIFIER CLASS
# =============================================================================


class KubernetesCloudDeploymentVerifier:
    """Verifies Kubernetes-based cloud deployments with comprehensive checks"""

    def __init__(self, logger: BlockLogger, api_client):
        self.log = logger
        self.api_client = api_client
        self.k8s_config: Optional[KubernetesConfig] = None
        self.results = VerificationResults()

    def verify(self, cloud_deployment: CloudDeployment) -> bool:
        """
        Main verification workflow for Kubernetes cloud deployments.

        Performs comprehensive checks including operator health, identity verification,
        file storage, networking, and gateway configuration.

        Args:
            cloud_deployment: The cloud deployment configuration
        """
        deployment_name = cloud_deployment.name or cloud_deployment.cloud_resource_id
        self.log.info(f"Starting Kubernetes verification for: {deployment_name}")

        if cloud_deployment.file_storage is not None and isinstance(
            cloud_deployment.file_storage, dict
        ):
            cloud_deployment.file_storage = FileStorage(**cloud_deployment.file_storage)

        if cloud_deployment.kubernetes_config is not None and isinstance(
            cloud_deployment.kubernetes_config, dict
        ):
            cloud_deployment.kubernetes_config = OpenAPIKubernetesConfig(
                **cloud_deployment.kubernetes_config
            )

        try:
            return self._run_verification_steps(cloud_deployment)

        except click.ClickException:
            # Re-raise ClickExceptions as they contain user-friendly messages
            raise
        except requests.RequestException as e:
            self.log.error(f"Network error during verification: {e}")
            return False
        except (subprocess.CalledProcessError, OSError) as e:
            self.log.error(f"System error during verification: {e}")
            return False
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.log.error(f"Data parsing error during verification: {e}")
            return False

    @contextmanager
    def _verification_step(self, step_name: str):
        """Context manager for verification steps that indents detailed output."""
        self.log.info(f"{step_name}...")
        with self.log.indent():
            yield

    def _run_verification_steps(self, cloud_deployment: CloudDeployment) -> bool:
        """Execute the verification steps in sequence."""
        # Step 1: Configure kubectl
        with self._verification_step("Configuring kubectl access"):
            self._get_kubectl_config()

        # k8s_config is guaranteed to be set by _get_kubectl_config()
        assert self.k8s_config is not None

        # Initialize utility classes
        kubectl_ops = KubectlOperations(self.k8s_config.context, self.log)
        operator_verifier = OperatorVerifier(kubectl_ops, self.k8s_config, self.log)
        storage_verifier = StorageVerifier(kubectl_ops, self.k8s_config, self.log)
        gateway_verifier = GatewayVerifier(kubectl_ops, self.k8s_config, self.log)

        # Step 2: Find and verify operator pod
        with self._verification_step("Finding operator pod"):
            try:
                operator_pod = operator_verifier.find_operator_pod()
                self.results.operator_pod_installed = VerificationStatus.PASSED
            except OperatorPodNotFoundError as e:
                self.log.error(
                    "Failed to find operator pod, please make sure the operator is running"
                )
                self.log.error(f"Error: {e}")
                return False

        # Step 3: Port forward and fetch operator data (health + config)
        with self._verification_step("Verifying operator status"):
            try:
                operator_data = operator_verifier.get_operator_data(operator_pod)
            except (OperatorConnectionError, PortForwardError) as e:
                self.log.error(
                    "Failed to connect to operator, please make sure the operator is running version >= 0.7.0 and has status reporting enabled"
                )
                self.log.error(f"Error: {e}")
                return False

            self.log.info("Verifying operator health...")
            self.results.operator_health = operator_verifier.verify_operator_health(
                operator_data
            )
            self.log.info(f"Operator Health: {self.results.operator_health.value}")

            self.log.info("Verifying operator identity...")
            if cloud_deployment.kubernetes_config is None:
                self.log.error(
                    "Kubernetes configuration is missing from cloud deployment"
                )
                self.results.operator_identity = VerificationStatus.FAILED
            else:
                self.results.operator_identity = operator_verifier.verify_operator_identity(
                    operator_data,
                    cloud_deployment.kubernetes_config,
                    cloud_deployment.provider,
                )
            self.log.info(f"Operator Identity: {self.results.operator_identity.value}")

        # Step 4: Check file storage
        with self._verification_step("Checking file storage"):
            if cloud_deployment.file_storage is None:
                self.log.info(
                    "No file storage configured - skipping file storage verification"
                )
                self.results.file_storage = VerificationStatus.SKIPPED
            else:
                self.results.file_storage = storage_verifier.verify_file_storage(
                    cloud_deployment.file_storage, cloud_deployment
                )

            self.log.info(f"File Storage: {self.results.file_storage.value}")

        # Step 5: Verify gateway support
        with self._verification_step("Checking gateway support"):
            self.results.gateway_support = gateway_verifier.verify_gateway_support(
                operator_data
            )
            self.log.info(f"Gateway Support: {self.results.gateway_support.value}")

        # Step 6: Check NGINX ingress (warning only)
        with self._verification_step("Checking NGINX ingress controller"):
            self.results.nginx_ingress = gateway_verifier.verify_nginx_ingress()
            self.log.info(f"NGINX Ingress: {self.results.nginx_ingress.value}")

        self._show_verification_summary()

        if self.results.overall_success:
            self.log.info(
                "Kubernetes cloud deployment verification completed successfully"
            )
        else:
            self.log.error("Kubernetes cloud deployment verification failed")

        return self.results.overall_success

    def _show_verification_summary(self):
        """Show verification results summary in the same format as VM verification."""
        verification_result_summary = ["Verification result:"]

        for component, result in self.results.to_dict().items():
            verification_result_summary.append(f"{component}: {result.value}")

        self.log.info("\n".join(verification_result_summary))

    def _get_kubectl_config(self):
        """Get kubectl context and operator namespace from user"""
        # If k8s_config is already set, skip prompting
        if self.k8s_config is not None:
            self.log.info(
                f"Using configured context='{self.k8s_config.context}', "
                f"namespace='{self.k8s_config.operator_namespace}'"
            )
            return

        # Check if kubectl is available
        temp_kubectl = KubectlOperations("", self.log)
        if not temp_kubectl.check_kubectl_available():
            raise click.ClickException(
                "kubectl command not found. Please install kubectl and ensure it's in your PATH."
            )

        # Get available contexts
        contexts = temp_kubectl.get_available_contexts()
        if not contexts:
            raise click.ClickException(
                "No kubectl contexts found. Please configure kubectl to access your Kubernetes cluster."
            )

        # Prompt for context selection
        if len(contexts) > 1:
            self.log.info("Available kubectl contexts:")
            for i, ctx in enumerate(contexts):
                current_marker = (
                    " (current)" if ctx == temp_kubectl.get_current_context() else ""
                )
                self.log.info(f"  {i+1}. {ctx}{current_marker}")

            choice = click.prompt(
                "Select context number",
                type=click.IntRange(1, len(contexts)),
                default=1,
            )
            kubectl_context = contexts[choice - 1]
        else:
            kubectl_context = contexts[0]
            self.log.info(f"Using kubectl context: {kubectl_context}")

        # Prompt for operator namespace
        operator_namespace = click.prompt(
            "Enter the Anyscale operator namespace",
            default=DEFAULT_OPERATOR_NAMESPACE,
            type=str,
            show_default=True,
        )

        self.k8s_config = KubernetesConfig(
            context=kubectl_context, operator_namespace=operator_namespace
        )

        self.log.info(
            f"Configured: context='{self.k8s_config.context}', "
            f"namespace='{self.k8s_config.operator_namespace}'"
        )
