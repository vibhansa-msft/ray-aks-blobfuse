"""
CloudFormation utility functions for creating and managing stacks.

This module provides reusable utilities for CloudFormation stack operations
including creation, waiting, and output retrieval.
"""

import time
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError
from click import ClickException

from anyscale.cli_logger import BlockLogger


class CloudFormationUtils:
    """Utility class for CloudFormation operations."""

    def __init__(self, logger: Optional[BlockLogger] = None):
        self.log = logger or BlockLogger()

    def create_and_wait_for_stack(  # noqa: PLR0913
        self,
        stack_name: str,
        template_body: str,
        parameters: list,
        region: str,
        capabilities: Optional[list] = None,
        timeout_seconds: int = 600,
        boto3_session: Optional[boto3.Session] = None,
        update_if_exists: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a CloudFormation stack and wait for it to complete.

        Args:
            stack_name: Name of the CloudFormation stack
            template_body: CloudFormation template body
            parameters: Stack parameters
            region: AWS region
            capabilities: Stack capabilities (default: CAPABILITY_NAMED_IAM, CAPABILITY_AUTO_EXPAND)
            timeout_seconds: Timeout in seconds (default: 600)
            boto3_session: Optional boto3 session (creates new if not provided)
            update_if_exists: Whether to update stack if it already exists (default: False)

        Returns:
            CloudFormation stack information

        Raises:
            ClickException: If stack creation fails or times out
        """
        if boto3_session is None:
            boto3_session = boto3.Session(region_name=region)

        cfn_client = boto3_session.client("cloudformation", region_name=region)

        if capabilities is None:
            capabilities = ["CAPABILITY_NAMED_IAM", "CAPABILITY_AUTO_EXPAND"]

        # Create the stack
        self.log.info(f"Creating CloudFormation stack: {stack_name}")
        try:
            cfn_client.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=capabilities,
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "AlreadyExistsException" and update_if_exists:
                self.log.warning(f"Stack {stack_name} already exists, updating...")
                try:
                    cfn_client.update_stack(
                        StackName=stack_name,
                        TemplateBody=template_body,
                        Parameters=parameters,
                        Capabilities=capabilities,
                    )
                except ClientError as update_error:
                    raise ClickException(
                        f"Failed to update existing stack: {update_error}"
                    )
            else:
                raise ClickException(f"Failed to create CloudFormation stack: {e}")

        # Wait for stack completion
        return self._wait_for_stack_completion(
            cfn_client, stack_name, region, timeout_seconds
        )

    def _wait_for_stack_completion(
        self, cfn_client, stack_name: str, region: str, timeout_seconds: int
    ) -> Dict[str, Any]:
        """Wait for CloudFormation stack to complete with timeout."""
        stack_url = f"https://{region}.console.aws.amazon.com/cloudformation/home?region={region}#/stacks/stackinfo?stackId={stack_name}"
        self.log.info(f"Track progress at: {stack_url}")

        with self.log.spinner("Creating cloud resources through CloudFormation..."):
            start_time = time.time()
            end_time = start_time + timeout_seconds

            while time.time() < end_time:
                try:
                    stacks = cfn_client.describe_stacks(StackName=stack_name)
                    cfn_stack = stacks["Stacks"][0]
                    stack_status = cfn_stack["StackStatus"]

                    if stack_status in (
                        "CREATE_FAILED",
                        "ROLLBACK_COMPLETE",
                        "ROLLBACK_IN_PROGRESS",
                        "UPDATE_FAILED",
                    ):
                        # Clean up resources if stack failed
                        self._cleanup_failed_stack(cfn_client, stack_name, region)

                        # Get error details
                        error_details = self._get_stack_error_details(
                            cfn_client, stack_name
                        )
                        raise ClickException(
                            f"CloudFormation stack failed: {stack_status}. "
                            f"Error details: {error_details}. "
                            f"Check the stack at: {stack_url}"
                        )

                    if stack_status in ("CREATE_COMPLETE", "UPDATE_COMPLETE"):
                        self.log.info(
                            f"CloudFormation stack {stack_name} completed successfully"
                        )
                        return cfn_stack

                    # Still in progress
                    time.sleep(5)

                except ClientError as e:
                    if "does not exist" in str(e):
                        raise ClickException(
                            f"CloudFormation stack {stack_name} not found"
                        )
                    raise ClickException(f"Error checking CloudFormation stack: {e}")

            # Timeout
            raise ClickException(
                f"CloudFormation stack {stack_name} timed out after {timeout_seconds} seconds. "
                f"Check the stack at: {stack_url}"
            )

    def _cleanup_failed_stack(self, cfn_client, stack_name: str, region: str) -> None:
        """Clean up resources from a failed CloudFormation stack."""
        try:
            # Try to get stack outputs to find resources to clean up
            stacks = cfn_client.describe_stacks(StackName=stack_name)
            stack = stacks["Stacks"][0]

            # Look for S3 bucket in outputs
            for output in stack.get("Outputs", []):
                if "Bucket" in output.get("OutputKey", ""):
                    bucket_name = output.get("OutputValue")
                    if bucket_name:
                        self._cleanup_s3_bucket(bucket_name, region)

        except Exception as e:  # noqa: BLE001
            self.log.warning(f"Failed to cleanup resources from failed stack: {e}")

    def _cleanup_s3_bucket(self, bucket_name: str, region: str) -> None:
        """Clean up S3 bucket created by failed stack."""
        try:
            s3_client = boto3.client("s3", region_name=region)
            s3_client.delete_bucket(Bucket=bucket_name)
            self.log.info(f"Successfully deleted S3 bucket: {bucket_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchBucket":
                self.log.error(f"Failed to delete S3 bucket {bucket_name}: {e}")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to delete S3 bucket {bucket_name}: {e}")

    def _get_stack_error_details(self, cfn_client, stack_name: str) -> str:
        """Get detailed error information from CloudFormation stack events."""
        try:
            events = cfn_client.describe_stack_events(StackName=stack_name)
            error_events = [
                event
                for event in events.get("StackEvents", [])
                if event.get("ResourceStatus", "").endswith("_FAILED")
            ]

            if error_events:
                latest_error = error_events[0]
                return (
                    f"Resource: {latest_error.get('LogicalResourceId', 'Unknown')}, "
                    f"Status: {latest_error.get('ResourceStatus', 'Unknown')}, "
                    f"Reason: {latest_error.get('ResourceStatusReason', 'Unknown')}"
                )
            else:
                return "No detailed error information available"

        except Exception as e:  # noqa: BLE001
            return f"Failed to get error details: {e}"

    def get_stack_outputs(
        self,
        stack_name: str,
        region: str,
        boto3_session: Optional[boto3.Session] = None,
    ) -> Dict[str, str]:
        """
        Get all outputs from a CloudFormation stack.

        Args:
            stack_name: Name of the CloudFormation stack
            region: AWS region
            boto3_session: Optional boto3 session (creates new if not provided)

        Returns:
            Dictionary of output key-value pairs

        Raises:
            ClickException: If stack not found or error retrieving outputs
        """
        if boto3_session is None:
            boto3_session = boto3.Session(region_name=region)

        cfn_client = boto3_session.client("cloudformation", region_name=region)

        try:
            stacks = cfn_client.describe_stacks(StackName=stack_name)
            stack = stacks["Stacks"][0]
            outputs = {}

            for output in stack.get("Outputs", []):
                outputs[output["OutputKey"]] = output["OutputValue"]

            return outputs
        except ClientError as e:
            if "does not exist" in str(e):
                raise ClickException(f"CloudFormation stack {stack_name} not found")
            raise ClickException(f"Failed to get CloudFormation outputs: {e}")

    def get_stack_output(
        self,
        stack_name: str,
        output_key: str,
        region: str,
        boto3_session: Optional[boto3.Session] = None,
    ) -> str:
        """
        Get a specific output value from a CloudFormation stack.

        Args:
            stack_name: Name of the CloudFormation stack
            output_key: Key of the output to retrieve
            region: AWS region
            boto3_session: Optional boto3 session (creates new if not provided)

        Returns:
            Output value

        Raises:
            ClickException: If stack not found or output key not found
        """
        outputs = self.get_stack_outputs(stack_name, region, boto3_session)

        if output_key not in outputs:
            available_keys = list(outputs.keys())
            raise ClickException(
                f"Output key '{output_key}' not found in stack {stack_name}. "
                f"Available keys: {available_keys}"
            )

        return outputs[output_key]

    def delete_stack(
        self,
        stack_name: str,
        region: str,
        boto3_session: Optional[boto3.Session] = None,
    ) -> None:
        """
        Delete a CloudFormation stack.

        Args:
            stack_name: Name of the CloudFormation stack
            region: AWS region
            boto3_session: Optional boto3 session (creates new if not provided)

        Raises:
            ClickException: If stack deletion fails
        """
        if boto3_session is None:
            boto3_session = boto3.Session(region_name=region)

        cfn_client = boto3_session.client("cloudformation", region_name=region)

        try:
            self.log.info(f"Deleting CloudFormation stack: {stack_name}")
            cfn_client.delete_stack(StackName=stack_name)

            # Wait for deletion to complete
            with self.log.spinner("Deleting CloudFormation stack..."):
                start_time = time.time()
                timeout_seconds = 300  # 5 minutes for deletion
                end_time = start_time + timeout_seconds

                while time.time() < end_time:
                    try:
                        cfn_client.describe_stacks(StackName=stack_name)
                        time.sleep(5)  # Still exists, wait
                    except ClientError as e:
                        if "does not exist" in str(e):
                            self.log.info(
                                f"CloudFormation stack {stack_name} deleted successfully"
                            )
                            return
                        raise ClickException(f"Error checking stack deletion: {e}")

                raise ClickException(
                    f"Stack deletion timed out after {timeout_seconds} seconds"
                )

        except ClientError as e:
            if "does not exist" in str(e):
                self.log.info(f"Stack {stack_name} does not exist")
                return
            raise ClickException(f"Failed to delete CloudFormation stack: {e}")


# Convenience functions for backward compatibility
def create_cloudformation_stack(
    stack_name: str,
    template_body: str,
    parameters: list,
    region: str,
    logger: Optional[BlockLogger] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to create a CloudFormation stack."""
    utils = CloudFormationUtils(logger)
    return utils.create_and_wait_for_stack(
        stack_name, template_body, parameters, region, **kwargs
    )


def get_cloudformation_outputs(
    stack_name: str, region: str, logger: Optional[BlockLogger] = None, **kwargs
) -> Dict[str, str]:
    """Convenience function to get CloudFormation stack outputs."""
    utils = CloudFormationUtils(logger)
    return utils.get_stack_outputs(stack_name, region, **kwargs)


def get_cloudformation_output(
    stack_name: str,
    output_key: str,
    region: str,
    logger: Optional[BlockLogger] = None,
    **kwargs,
) -> str:
    """Convenience function to get a specific CloudFormation stack output."""
    utils = CloudFormationUtils(logger)
    return utils.get_stack_output(stack_name, output_key, region, **kwargs)
