"""
Azure Resource Manager (ARM) utility functions for deploying templates.

This module provides utilities for ARM template deployment operations
including creation, waiting, and output retrieval.
"""

import time
from typing import Any, Dict, Optional, Tuple

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import (
    Deployment,
    DeploymentMode,
    DeploymentProperties,
)
from click import ClickException

from anyscale.cli_logger import BlockLogger


class ARMTemplateUtils:
    """Utility class for Azure Resource Manager template operations."""

    def __init__(self, subscription_id: str, logger: Optional[BlockLogger] = None):
        """
        Initialize ARM template utilities for a specific subscription.

        Args:
            subscription_id: Azure subscription ID
            logger: Optional BlockLogger for output
        """
        self.log = logger or BlockLogger()
        self.subscription_id = subscription_id
        self._credential = None
        self._client = None

    def _get_credential(self) -> DefaultAzureCredential:
        """Get or create Azure credential."""
        if self._credential is None:
            self._credential = DefaultAzureCredential()
        return self._credential

    def _get_client(self) -> ResourceManagementClient:
        """Get or create Resource Management client."""
        if self._client is None:
            credential = self._get_credential()
            self._client = ResourceManagementClient(credential, self.subscription_id)
        return self._client

    def deploy_template(
        self,
        deployment_name: str,
        resource_group: str,
        template: Dict[str, Any],
        parameters: Dict[str, Any],
        timeout_seconds: int = 600,
        mode: DeploymentMode = DeploymentMode.INCREMENTAL,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Deploy an ARM template and wait for completion.

        Args:
            deployment_name: Name of the deployment
            resource_group: Resource group name
            template: ARM template as dictionary
            parameters: Template parameters
            timeout_seconds: Timeout in seconds (default: 600)
            mode: Deployment mode (default: Incremental)

        Returns:
            Tuple containing dictionary of output key-value pairs and portal URL

        Raises:
            ClickException: If deployment fails or times out
        """
        client = self._get_client()

        # Format parameters for ARM (ARM expects {"key": {"value": "val"}})
        arm_parameters = {k: {"value": v} for k, v in parameters.items()}

        deployment_properties = DeploymentProperties(
            mode=mode, template=template, parameters=arm_parameters
        )

        self.log.info(f"Starting ARM deployment: {deployment_name}")

        try:
            # Begin deployment
            deployment_operation = client.deployments.begin_create_or_update(
                resource_group_name=resource_group,
                deployment_name=deployment_name,
                parameters=Deployment(properties=deployment_properties),
            )

            # Wait for completion
            return self._wait_for_deployment(
                client,
                resource_group,
                deployment_name,
                deployment_operation,
                timeout_seconds,
            )

        except HttpResponseError as e:
            raise ClickException(f"Failed to start ARM deployment: {e.message}")

    def _wait_for_deployment(
        self,
        client: ResourceManagementClient,
        resource_group: str,
        deployment_name: str,
        deployment_operation,
        timeout_seconds: int,
    ) -> Tuple[Dict[str, Any], str]:
        """Wait for ARM deployment to complete with progress tracking."""
        portal_url = (
            f"https://portal.azure.com/#blade/HubsExtension/DeploymentDetailsBlade/"
            f"overview/id/%2Fsubscriptions%2F{self.subscription_id}%2F"
            f"resourceGroups%2F{resource_group}%2Fproviders%2F"
            f"Microsoft.Resources%2Fdeployments%2F{deployment_name}"
        )
        self.log.info(f"Track progress at: {portal_url}")

        with self.log.spinner("Deploying ARM template..."):
            start_time = time.time()
            end_time = start_time + timeout_seconds

            while time.time() < end_time:
                try:
                    # Check if operation is done
                    if deployment_operation.done():
                        result = deployment_operation.result()

                        if result.properties.provisioning_state == "Succeeded":
                            self.log.info(
                                f"ARM deployment {deployment_name} completed successfully"
                            )
                            return self._extract_outputs(result), portal_url

                        # Deployment failed
                        error_msg = self._get_deployment_errors(
                            client, resource_group, deployment_name
                        )
                        raise ClickException(
                            f"ARM deployment failed with state: "
                            f"{result.properties.provisioning_state}. "
                            f"Errors: {error_msg}. "
                            f"Check the deployment at: {portal_url}"
                        )

                    # Still in progress, wait
                    time.sleep(5)

                except HttpResponseError as e:
                    raise ClickException(
                        f"Error checking deployment status: {e.message}"
                    )

            # Timeout
            raise ClickException(
                f"ARM deployment {deployment_name} timed out after "
                f"{timeout_seconds} seconds. Check Azure Portal: {portal_url}"
            )

    def _extract_outputs(self, deployment: Deployment) -> Dict[str, Any]:
        """Extract outputs from a completed deployment."""
        if not deployment.properties or not deployment.properties.outputs:
            return {}

        # ARM outputs have format: {"key": {"type": "string", "value": "..."}}
        # We want to return: {"key": "value"}
        outputs = {}
        for key, output_obj in deployment.properties.outputs.items():
            if isinstance(output_obj, dict) and "value" in output_obj:
                outputs[key] = output_obj["value"]
            else:
                outputs[key] = output_obj

        return outputs

    def _get_deployment_errors(
        self,
        client: ResourceManagementClient,
        resource_group: str,
        deployment_name: str,
    ) -> str:
        """Get detailed error messages from deployment operations."""
        try:
            operations = client.deployment_operations.list(
                resource_group_name=resource_group, deployment_name=deployment_name
            )

            errors = []
            for operation in operations:
                if (
                    operation.properties
                    and operation.properties.status_message
                    and operation.properties.provisioning_state == "Failed"
                ):
                    msg = operation.properties.status_message
                    if hasattr(msg, "error"):
                        error_detail = f"{msg.error.code}: {msg.error.message}"
                    else:
                        error_detail = str(msg)

                    resource_name = "Unknown"
                    if (
                        operation.properties.target_resource
                        and operation.properties.target_resource.resource_name
                    ):
                        resource_name = (
                            operation.properties.target_resource.resource_name
                        )
                    errors.append(f"Resource: {resource_name}, Error: {error_detail}")

            return "; ".join(errors) if errors else "No detailed error information"

        except Exception as e:  # noqa: BLE001
            return f"Could not retrieve error details: {e}"

    def get_deployment_outputs(
        self, deployment_name: str, resource_group: str
    ) -> Dict[str, Any]:
        """
        Get outputs from an existing ARM deployment.

        Args:
            deployment_name: Name of the deployment
            resource_group: Resource group name

        Returns:
            Dictionary of output key-value pairs

        Raises:
            ClickException: If deployment not found
        """
        client = self._get_client()

        try:
            deployment = client.deployments.get(
                resource_group_name=resource_group, deployment_name=deployment_name
            )
            return self._extract_outputs(deployment)

        except ResourceNotFoundError:
            raise ClickException(
                f"Deployment {deployment_name} not found in resource group {resource_group}"
            )
        except HttpResponseError as e:
            raise ClickException(f"Failed to get deployment outputs: {e.message}")
