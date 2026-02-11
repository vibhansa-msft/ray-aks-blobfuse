"""
Kubernetes Cloud Setup Command

This module provides a streamlined command for setting up Anyscale on Kubernetes clusters.
It handles infrastructure provisioning, cloud registration, and operator installation.
"""

from dataclasses import dataclass
import json
import os
import random
import re
import subprocess
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional

import click
import yaml

from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models import (
    AWSConfig,
    CloudDeployment,
    CloudProviders,
    ComputeStack,
    KubernetesConfig as OpenAPIKubernetesConfig,
    ObjectStorage,
)
from anyscale.controllers.cloud_controller import CloudController
from anyscale.controllers.kubernetes_verifier import (
    KubernetesCloudDeploymentVerifier,
    KubernetesConfig,
)
from anyscale.shared_anyscale_utils.conf import ANYSCALE_CORS_ORIGIN, ANYSCALE_HOST


@dataclass
class ClusterInfo:
    """Information about the target Kubernetes cluster."""

    context: str
    namespace: str
    provider: str
    region: str
    cluster_name: str
    project_id: Optional[str] = None
    oidc_provider: Optional[str] = None
    resource_group: Optional[str] = None  # Azure resource group


@dataclass
class InfrastructureResources:
    """Resources created during infrastructure setup."""

    bucket_name: str  # For Azure: ABFSS URL format
    iam_role_arn: str  # For Azure: Managed Identity Principal ID (for Anyscale API)
    region: str
    project_id: Optional[str] = None
    resource_group: Optional[str] = None  # Azure resource group
    workload_identity_client_id: Optional[
        str
    ] = None  # Azure: Client ID (for Helm values)
    blob_endpoint: Optional[
        str
    ] = None  # Azure: Blob endpoint URL (for object_storage.endpoint)
    deployment_url: Optional[
        str
    ] = None  # Cloud provider deployment URL (AWS CloudFormation, Azure ARM)


class KubernetesCloudSetupCommand:
    """Command to setup Kubernetes cloud."""

    def __init__(self, logger: Optional[BlockLogger] = None, debug: bool = False):
        self.log = logger or BlockLogger()
        self.cloud_controller = CloudController(log=self.log)
        self.skip_confirmation = False
        self.debug = debug or os.environ.get("ANYSCALE_DEBUG") == "1"

    def run(  # noqa: PLR0913
        self,
        provider: str,
        region: str,
        name: str,
        cluster_name: str,
        namespace: str,
        project_id: Optional[str],
        resource_group: Optional[str],
        functional_verify: bool,
        yes: bool,
        values_file: Optional[str] = None,
        operator_chart: Optional[str] = None,
        cloud_id: Optional[str] = None,
        resource_name: Optional[str] = None,
    ) -> None:
        """
        Main entry point for Kubernetes cloud setup.

        This method handles both:
        1. Creating a new cloud (when cloud_id is None)
        2. Adding a resource to an existing cloud (when cloud_id is provided)

        Args:
            provider: Cloud provider (aws, gcp, azure)
            region: Cloud region
            name: Name for the Anyscale cloud. If cloud_id is not provided, this will be used to create a new cloud.
            cluster_name: Kubernetes cluster name/context
            namespace: Namespace for the Anyscale operator
            project_id: GCP project ID (required for GCP)
            resource_group: Resource group for Azure clouds (required for Azure)
            functional_verify: Whether to run functional verification
            yes: Skip confirmation prompts
            values_file: Optional custom path for Helm values file
            operator_chart: Optional path to operator chart (skips helm repo add/update)
            cloud_id: Optional cloud ID for the Anyscale cloud to add the resource to
            resource_name: Optional name for the cloud resource (will be auto-generated if not provided)
        """
        # Determine if we're creating a new cloud or adding to existing
        create_cloud = cloud_id is None

        # Validate cloud_id is provided when adding to existing cloud
        if not create_cloud:
            assert (
                cloud_id
            ), "cloud_id is required when adding a resource to an existing cloud"

        # Set up logging message based on mode
        if create_cloud:
            setup_message = (
                f"Setting up Kubernetes cloud '{name}' on {provider.upper()}"
            )
        else:
            setup_message = f"Setting up Kubernetes cloud resource for '{name}' on {provider.upper()}"

        self.log.open_block("Setup", setup_message)

        # Set confirmation flag
        self.skip_confirmation = yes

        # Track what resources were created for cleanup messaging
        infrastructure = None
        cluster_info = None
        cloud_resource_id = None

        try:
            self._check_required_tools(provider)

            # Step 1: Prompt for namespace BEFORE infrastructure setup
            # This is needed because the IAM role trust relationship depends on the namespace
            final_namespace = self._prompt_for_namespace(
                namespace, skip_confirmation=yes
            )

            # Step 2: Discover and validate cluster
            cluster_info = self._discover_cluster(
                cluster_name,
                final_namespace,
                provider,
                region,
                project_id,
                resource_group,
            )

            # Step 3: Set up cloud infrastructure
            infrastructure = self._setup_infrastructure(
                provider, region, name, cluster_info
            )

            # Step 4: Register cloud OR create cloud resource
            if create_cloud:
                # Register new cloud with Anyscale
                cloud_id = self._register_cloud(
                    name, provider, region, infrastructure, cluster_info
                )

                # Get the cloud resource ID from the newly registered cloud
                cloud_resources = self.cloud_controller.get_decorated_cloud_resources(
                    cloud_id
                )
                if not cloud_resources:
                    raise click.ClickException(
                        "No cloud resources found after registration"
                    )
                cloud_resource_id = cloud_resources[0].cloud_resource_id
            else:
                # Should have been validated earlier, but just in case
                assert (
                    cloud_id
                ), "cloud_id is required when adding a resource to an existing cloud"

                # Create cloud resource in existing cloud
                cloud_resource_id = self._create_cloud_resource(
                    cloud_id,
                    provider,
                    region,
                    infrastructure,
                    cluster_info,
                    resource_name,
                )

            # Step 5: Install Anyscale operator
            self._install_operator(
                cloud_resource_id,
                provider,
                region,
                final_namespace,
                infrastructure,
                values_file,
                operator_chart,
                skip_confirmation=yes,
            )

            # Step 6: Verify installation
            if functional_verify:
                self._verify_installation(
                    cloud_id, final_namespace, cluster_info, cloud_resource_id
                )

            self.log.close_block("Setup")
            if create_cloud:
                self.log.info(
                    f"Kubernetes cloud '{name}' setup completed successfully!"
                )
            else:
                self.log.info(
                    f"Kubernetes cloud resource setup for '{name}' completed successfully!"
                )
        except Exception:  # noqa: BLE001
            self.log.close_block("Setup")
            self._handle_setup_failure(
                provider,
                infrastructure,
                cloud_id,
                name,
                is_cloud_resource_setup=not create_cloud,
            )
            raise

    def _debug(self, *msg: str) -> None:
        """Log debug messages only when debug mode is enabled."""
        if self.debug:
            self.log.debug(*msg)

    def _check_required_tools(self, provider: str) -> None:
        """Check that required CLI tools are installed."""
        # Common tools required for all providers
        required_tools = ["kubectl", "helm"]

        # Provider-specific tools
        if provider == "aws":
            required_tools.append("aws")
        elif provider == "gcp":
            required_tools.extend(["gcloud", "gsutil"])
        elif provider == "azure":
            required_tools.append("az")

        self._debug(f"Checking for required tools: {', '.join(required_tools)}")

        missing_tools = []
        for tool in required_tools:
            if not self._check_command_available(tool):
                missing_tools.append(tool)

        if missing_tools:
            error_msg = f"Missing required CLI tools: {', '.join(missing_tools)}\n\n"
            raise click.ClickException(error_msg.rstrip())

        self.log.info(
            f"Required CLI tools are installed ({', '.join(required_tools)})",
            block_label="Setup",
        )

    def _check_command_available(self, command: str) -> bool:
        """Check if a command is available in the system PATH."""
        try:
            result = subprocess.run(
                ["which", command], capture_output=True, text=True, check=False
            )
            return result.returncode == 0
        except Exception:  # noqa: BLE001
            return False

    def _discover_cluster(
        self,
        cluster_name: str,
        namespace: str,
        provider: str,
        region: str,
        project_id: Optional[str],
        resource_group: Optional[str] = None,
    ) -> ClusterInfo:
        """Discover and validate the target Kubernetes cluster using cloud provider APIs."""
        self.log.info(
            f"Discovering {provider.upper()} cluster: {cluster_name}",
            block_label="Setup",
        )

        if provider == "aws":
            return self._discover_aws_cluster(cluster_name, namespace, region)
        elif provider == "gcp":
            if not project_id:
                raise click.ClickException(
                    "GCP project ID is required. Please provide --project-id"
                )
            return self._discover_gcp_cluster(
                cluster_name, namespace, region, project_id
            )
        elif provider == "azure":
            if not resource_group:
                raise click.ClickException(
                    "Azure resource group is required. Please provide --resource-group"
                )
            return self._discover_azure_cluster(
                cluster_name, namespace, region, resource_group
            )
        else:
            raise click.ClickException(f"Unsupported provider: {provider}")

    def _discover_aws_cluster(
        self, cluster_name: str, namespace: str, region: str
    ) -> ClusterInfo:
        """Discover AWS EKS cluster details and configure kubeconfig."""

        try:
            self._debug("Fetching OIDC provider information...")
            oidc_provider = self._get_eks_oidc_provider(cluster_name, region)
            self._debug(f"OIDC Provider: {oidc_provider}")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to get OIDC provider: {e}")
            raise click.ClickException(
                f"Failed to get OIDC provider for cluster {cluster_name}: {e}"
            )

        try:
            self._debug("Configuring kubeconfig for EKS cluster...")
            self._configure_aws_kubeconfig(cluster_name, region)
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to configure kubeconfig: {e}")
            raise click.ClickException(
                f"Failed to configure kubeconfig for EKS cluster: {e}"
            )

        try:
            self._debug("Verifying kubeconfig configuration...")
            self._verify_kubeconfig()
            current_context = self._get_current_kubectl_context()
            self.log.info(f"Cluster discovered: {current_context}", block_label="Setup")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to verify kubeconfig: {e}")
            raise click.ClickException(f"Failed to verify kubeconfig: {e}")

        return ClusterInfo(
            context=current_context,
            namespace=namespace,
            provider="aws",
            region=region,
            cluster_name=cluster_name,
            oidc_provider=oidc_provider,
        )

    def _discover_gcp_cluster(
        self, cluster_name: str, namespace: str, region: str, project_id: str
    ) -> ClusterInfo:
        """Discover GCP GKE cluster details and configure kubeconfig."""

        try:
            self._debug("Configuring kubeconfig for GKE cluster...")
            self._configure_gcp_kubeconfig(cluster_name, region, project_id)
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to configure kubeconfig: {e}")
            raise click.ClickException(
                f"Failed to configure kubeconfig for GKE cluster: {e}"
            )

        try:
            self._debug("Verifying kubeconfig configuration...")
            self._verify_kubeconfig()
            current_context = self._get_current_kubectl_context()
            self.log.info(f"Cluster discovered: {current_context}", block_label="Setup")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to verify kubeconfig: {e}")
            raise click.ClickException(f"Failed to verify kubeconfig: {e}")

        return ClusterInfo(
            context=current_context,
            namespace=namespace,
            provider="gcp",
            region=region,
            cluster_name=cluster_name,
            project_id=project_id,
        )

    def _discover_azure_cluster(
        self, cluster_name: str, namespace: str, region: str, resource_group: str
    ) -> ClusterInfo:
        """Discover Azure AKS cluster details and configure kubeconfig."""
        try:
            # Get AKS cluster info to extract OIDC issuer
            self._debug("Fetching AKS cluster information...")
            cluster_info = self._get_aks_cluster_info(cluster_name, resource_group)
            oidc_issuer = cluster_info.get("oidcIssuerProfile", {}).get("issuerUrl")

            if not oidc_issuer:
                raise click.ClickException(
                    "OIDC issuer not found. Please ensure AKS cluster has OIDC issuer enabled."
                )

            self._debug(f"OIDC Issuer: {oidc_issuer}")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to get AKS cluster info: {e}")
            raise click.ClickException(
                f"Failed to get OIDC issuer for cluster {cluster_name}: {e}"
            )

        try:
            self._debug("Configuring kubeconfig for AKS cluster...")
            self._configure_azure_kubeconfig(cluster_name, resource_group)
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to configure kubeconfig: {e}")
            raise click.ClickException(
                f"Failed to configure kubeconfig for AKS cluster: {e}"
            )

        try:
            self._debug("Verifying kubeconfig configuration...")
            self._verify_kubeconfig()
            current_context = self._get_current_kubectl_context()
            self.log.info(f"Cluster discovered: {current_context}", block_label="Setup")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to verify kubeconfig: {e}")
            raise click.ClickException(f"Failed to verify kubeconfig: {e}")

        return ClusterInfo(
            context=current_context,
            namespace=namespace,
            provider="azure",
            region=region,
            cluster_name=cluster_name,
            resource_group=resource_group,
            oidc_provider=oidc_issuer,
        )

    def _setup_infrastructure(
        self, provider: str, region: str, name: str, cluster_info: ClusterInfo,
    ) -> InfrastructureResources:
        """Set up cloud infrastructure (S3/GCS bucket, IAM roles, etc.)."""
        self.log.info(
            f"Setting up {provider.upper()} infrastructure...", block_label="Setup"
        )

        if provider == "aws":
            return self._setup_aws_infrastructure(region, name, cluster_info)
        elif provider == "gcp":
            return self._setup_gcp_infrastructure(region, name, cluster_info)
        elif provider == "azure":
            return self._setup_azure_infrastructure(region, name, cluster_info)
        else:
            raise click.ClickException(f"Unsupported provider: {provider}")

    def _setup_aws_infrastructure(  # noqa: PLR0912
        self, region: str, name: str, cluster_info: ClusterInfo,
    ) -> InfrastructureResources:
        """Set up AWS infrastructure for Kubernetes using CloudFormation."""
        try:
            import boto3  # noqa: PLC0415 - codex_reason("gpt5.2", "optional cloud deps loaded lazily")

            from anyscale.utils.cloudformation_utils import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional cloud deps loaded lazily")
                CloudFormationUtils,
            )
        except ImportError as e:  # noqa: BLE001
            self.log.error(f"Failed to import required modules: {e}")
            raise click.ClickException(f"Failed to import required modules: {e}")

        try:
            # Generate a unique cloud ID (sanitize name to replace _ with -)
            sanitized_name = name.replace("_", "-")
            cloud_id = f"k8s-{sanitized_name}-{os.urandom(4).hex()}"
            stack_name = cloud_id.lower()
            self._debug(f"Generated cloud ID: {cloud_id}")
            self._debug(f"CloudFormation stack name: {stack_name}")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to generate cloud ID: {e}")
            raise click.ClickException(f"Failed to generate cloud ID: {e}")

        try:
            # Generate CloudFormation template for Kubernetes setup with actual OIDC provider
            if not cluster_info.oidc_provider:
                raise click.ClickException(
                    "OIDC provider information not found. Please ensure the EKS cluster has OIDC provider enabled."
                )
            self._debug("Generating CloudFormation template...")
            self._debug(
                f"Using namespace: {cluster_info.namespace} with service account: anyscale-operator"
            )
            cfn_template_body = self._generate_aws_cloudformation_template(
                cloud_id, cluster_info.oidc_provider, cluster_info.namespace,
            )
            self._debug("CloudFormation template generated successfully")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to generate CloudFormation template: {e}")
            raise click.ClickException(
                f"Failed to generate CloudFormation template: {e}"
            )

        try:
            self._debug("Preparing CloudFormation parameters...")
            parameters = [{"ParameterKey": "CloudID", "ParameterValue": cloud_id}]
            self._debug(f"Prepared {len(parameters)} CloudFormation parameters")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to prepare CloudFormation parameters: {e}")
            raise click.ClickException(
                f"Failed to prepare CloudFormation parameters: {e}"
            )

        try:
            with self.log.indent():
                self.log.info(
                    "Creating CloudFormation stack (this may take a few minutes)...",
                    block_label="Setup",
                )
                boto3_session = boto3.Session(region_name=region)
                cfn_utils = CloudFormationUtils(self.log)
                cfn_utils.create_and_wait_for_stack(
                    stack_name=stack_name,
                    template_body=cfn_template_body,
                    parameters=parameters,
                    region=region,
                    boto3_session=boto3_session,
                    timeout_seconds=600,
                )
            self.log.info("CloudFormation stack created", block_label="Setup")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to create CloudFormation stack: {e}")
            raise click.ClickException(f"Failed to create CloudFormation stack: {e}")

        try:
            self._debug("Retrieving CloudFormation stack outputs...")
            stack_outputs = cfn_utils.get_stack_outputs(
                stack_name, region, boto3_session
            )
            bucket_name = stack_outputs.get("S3BucketName", f"anyscale-{cloud_id}")
            iam_role_arn = stack_outputs.get("AnyscaleCrossAccountIAMRoleArn")

            if not iam_role_arn:
                raise click.ClickException(
                    "Failed to get IAM role ARN from CloudFormation stack"
                )

            self._debug(f"S3 Bucket: {bucket_name}")
            self._debug(f"IAM Role ARN: {iam_role_arn}")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to get CloudFormation outputs: {e}")
            raise click.ClickException(f"Failed to get CloudFormation outputs: {e}")

        return InfrastructureResources(
            bucket_name=bucket_name, iam_role_arn=iam_role_arn, region=region
        )

    def _generate_aws_cloudformation_template(
        self, cloud_id: str, oidc_provider_arn: str, namespace: str,
    ) -> str:
        """Generate CloudFormation template for AWS Kubernetes setup."""
        # Extract OIDC provider URL from ARN for the condition
        # ARN format: arn:aws:iam::ACCOUNT:oidc-provider/oidc.eks.REGION.amazonaws.com/id/XXXXXX
        # We need: oidc.eks.REGION.amazonaws.com/id/XXXXXX
        if "oidc-provider/" not in oidc_provider_arn:
            raise click.ClickException(
                f"Invalid OIDC provider ARN format: {oidc_provider_arn}"
            )
        oidc_provider_url = oidc_provider_arn.split("oidc-provider/")[-1]

        service_account_name = "anyscale-operator"

        # Use ANYSCALE_CORS_ORIGIN from shared config
        # This respects the ANYSCALE_HOST environment variable
        allowed_origin = ANYSCALE_CORS_ORIGIN

        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"Anyscale Kubernetes Cloud Infrastructure for {cloud_id}",
            "Parameters": {
                "CloudID": {
                    "Type": "String",
                    "Description": "Cloud ID for resource naming",
                }
            },
            "Resources": {
                "AnyscaleBucket": {
                    "Type": "AWS::S3::Bucket",
                    "Properties": {
                        "BucketName": {"Fn::Sub": "anyscale-${CloudID}"},
                        "VersioningConfiguration": {"Status": "Enabled"},
                        "PublicAccessBlockConfiguration": {
                            "BlockPublicAcls": True,
                            "BlockPublicPolicy": True,
                            "IgnorePublicAcls": True,
                            "RestrictPublicBuckets": True,
                        },
                        "CorsConfiguration": {
                            "CorsRules": [
                                {
                                    "AllowedHeaders": ["*"],
                                    "AllowedMethods": [
                                        "GET",
                                        "PUT",
                                        "POST",
                                        "HEAD",
                                        "DELETE",
                                    ],
                                    "AllowedOrigins": [allowed_origin],
                                    "ExposedHeaders": [
                                        "Accept-Ranges",
                                        "Content-Range",
                                        "Content-Length",
                                    ],
                                    "MaxAge": 3600,
                                },
                            ]
                        },
                    },
                },
                "AnyscaleOperatorRole": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "RoleName": {"Fn::Sub": "${CloudID}-anyscale-operator-role"},
                        "AssumeRolePolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Principal": {"Federated": oidc_provider_arn},
                                    "Action": "sts:AssumeRoleWithWebIdentity",
                                    "Condition": {
                                        "StringEquals": {
                                            f"{oidc_provider_url}:sub": f"system:serviceaccount:{namespace}:{service_account_name}"
                                        }
                                    },
                                }
                            ],
                        },
                        "Policies": [
                            {
                                "PolicyName": "AnyscaleS3AccessPolicy",
                                "PolicyDocument": {
                                    "Version": "2012-10-17",
                                    "Statement": [
                                        {
                                            "Effect": "Allow",
                                            "Action": [
                                                "s3:GetObject",
                                                "s3:PutObject",
                                                "s3:DeleteObject",
                                                "s3:ListBucket",
                                            ],
                                            "Resource": [
                                                {
                                                    "Fn::GetAtt": [
                                                        "AnyscaleBucket",
                                                        "Arn",
                                                    ]
                                                },
                                                {"Fn::Sub": "${AnyscaleBucket.Arn}/*"},
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                },
            },
            "Outputs": {
                "S3BucketName": {
                    "Value": {"Ref": "AnyscaleBucket"},
                    "Description": "Name of the S3 bucket",
                },
                "AnyscaleCrossAccountIAMRoleArn": {
                    "Value": {"Fn::GetAtt": ["AnyscaleOperatorRole", "Arn"]},
                    "Description": "ARN of the Anyscale operator IAM role",
                },
            },
        }

        return json.dumps(template, indent=2)

    def _setup_gcp_infrastructure(  # noqa: PLR0912
        self, region: str, name: str, cluster_info: ClusterInfo,
    ) -> InfrastructureResources:
        """Set up GCP infrastructure for Kubernetes using GCP Python SDK.

        Note: Deployment Manager is deprecated so it is unused here.
        Infrastructure Manager was tried but did not work well, so we rely
        on the GCP Python SDK instead.
        """
        try:
            from anyscale.utils.gcp_utils import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional cloud deps loaded lazily")
                get_google_cloud_client_factory,
            )
        except ImportError as e:
            self.log.error(f"Failed to import required modules: {e}")
            raise click.ClickException(f"Failed to import required modules: {e}")

        try:
            # Generate a unique cloud ID
            cloud_id = f"k8s-{name}-{os.urandom(4).hex()}"
            deployment_name = cloud_id.replace("_", "-").lower()
            self._debug(f"Generated cloud ID: {cloud_id}")
            self._debug(f"Infrastructure Manager deployment name: {deployment_name}")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to generate cloud ID: {e}")
            raise click.ClickException(f"Failed to generate cloud ID: {e}")

        try:
            # Get Google Cloud client factory
            factory = get_google_cloud_client_factory(self.log, cluster_info.project_id)
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to initialize GCP client: {e}")
            raise click.ClickException(f"Failed to initialize GCP client: {e}")

        try:
            with self.log.indent():
                self.log.warning(
                    "NOTE: GCP resources (bucket and service account) created by this command are not managed by Anyscale.",
                )
                self.log.warning(
                    "You will need to manually delete these resources when the cloud is no longer needed.",
                )
                self.log.info(
                    "Creating GCP resources (bucket, service account, IAM bindings)...",
                )

                # Calculate resource names
                # Service account name: anyscale-operator-<random 8 chars>
                # Max length for GCP service account is 30 characters
                random_suffix = os.urandom(4).hex()  # 8 hex chars
                anyscale_service_account_name = f"anyscale-operator-{random_suffix}"
                bucket_name = f"anyscale-{cloud_id.replace('_', '-').lower()}"

                # Create GCS bucket
                self._debug(f"Creating GCS bucket: {bucket_name}")
                storage_client = factory.storage.Client()
                bucket = storage_client.bucket(bucket_name)
                bucket.location = region
                bucket.storage_class = "REGIONAL"
                bucket.iam_configuration.uniform_bucket_level_access_enabled = True
                bucket.iam_configuration.public_access_prevention = "enforced"
                bucket.versioning_enabled = True
                bucket.labels = {"anyscale-cloud-id": cloud_id.replace("-", "_")}

                # Set CORS
                # Use ANYSCALE_CORS_ORIGIN from shared config
                # This respects the ANYSCALE_HOST environment variable
                allowed_origin = ANYSCALE_CORS_ORIGIN
                bucket.cors = [
                    {
                        "origin": [allowed_origin],
                        "responseHeader": ["*"],
                        "method": ["GET", "PUT", "POST", "HEAD", "DELETE"],
                        "maxAgeSeconds": 3600,
                    }
                ]

                storage_client.create_bucket(bucket, location=region)
                self.log.info(f"Created GCS bucket: {bucket_name}", block_label="Setup")

                # Create service account
                self._debug(
                    f"Creating service account: {anyscale_service_account_name}"
                )
                iam_client = factory.build("iam", "v1")
                service_account_body = {
                    "accountId": anyscale_service_account_name,
                    "serviceAccount": {
                        "displayName": f"{cloud_id} Anyscale operator service account",
                        "description": "Service account for Anyscale Kubernetes operator",
                    },
                }

                service_account = (
                    iam_client.projects()
                    .serviceAccounts()
                    .create(
                        name=f"projects/{cluster_info.project_id}",
                        body=service_account_body,
                    )
                    .execute()
                )

                service_account_email = service_account["email"]
                self.log.info(
                    f"Created service account: {service_account_email}",
                    block_label="Setup",
                )

                # Wait for service account to propagate through GCP systems
                self._debug("Waiting 10 seconds for service account to propagate...")
                time.sleep(10)

                # Grant Workload Identity binding
                self._debug("Setting up Workload Identity binding")

                # The K8s service account needs:
                # 1. workloadIdentityUser role - to impersonate the GCP service account
                # 2. serviceAccountTokenCreator - to generate tokens (for getOpenIdToken)

                policy_body = {
                    "policy": {
                        "bindings": [
                            {
                                "role": "roles/iam.workloadIdentityUser",
                                "members": [
                                    f"serviceAccount:{cluster_info.project_id}.svc.id.goog[{cluster_info.namespace}/anyscale-operator]"
                                ],
                            },
                            {
                                "role": "roles/iam.serviceAccountTokenCreator",
                                "members": [f"serviceAccount:{service_account_email}"],
                            },
                        ]
                    }
                }

                iam_client.projects().serviceAccounts().setIamPolicy(
                    resource=f"projects/{cluster_info.project_id}/serviceAccounts/{service_account_email}",
                    body=policy_body,
                ).execute()

                self.log.info(
                    "Configured Workload Identity binding", block_label="Setup"
                )

                # Grant storage admin role to service account for the bucket
                # Note: There's often a propagation delay after service account creation
                # We need to retry with exponential backoff
                self._debug("Granting storage permissions")

                max_retries = 5
                retry_delay = 2  # Start with 2 seconds

                for attempt in range(max_retries):
                    try:
                        bucket_policy = bucket.get_iam_policy(
                            requested_policy_version=3
                        )
                        bucket_policy.bindings.append(
                            {
                                "role": "roles/storage.admin",
                                "members": {f"serviceAccount:{service_account_email}"},
                            }
                        )
                        bucket.set_iam_policy(bucket_policy)
                        break  # Success!
                    except Exception as e:  # noqa: BLE001
                        if "does not exist" in str(e) and attempt < max_retries - 1:
                            self._debug(
                                f"Service account not yet propagated, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            raise  # Re-raise if it's not a propagation issue or we're out of retries

                self.log.info(
                    "Granted storage permissions to service account",
                    block_label="Setup",
                )

            self.log.info("GCP resources created successfully", block_label="Setup")
            self.log.warning(
                f"REMINDER: To clean up when no longer needed, delete GCS bucket '{bucket_name}' and service account '{service_account_email}'"
            )
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to create GCP resources: {e}")
            raise click.ClickException(f"Failed to create GCP resources: {e}")

        # Resources were created in the try block above
        # bucket_name and service_account_email are already set
        self._debug(f"GCS Bucket: {bucket_name}")
        self._debug(f"Service Account Email: {service_account_email}")

        return InfrastructureResources(
            bucket_name=bucket_name,
            iam_role_arn=service_account_email,  # For GCP, we use service account email
            region=region,
            project_id=cluster_info.project_id,
        )

    def _setup_azure_infrastructure(  # noqa: PLR0912
        self, region: str, name: str, cluster_info: ClusterInfo,
    ) -> InfrastructureResources:
        """Set up Azure infrastructure for Kubernetes using ARM template."""
        try:
            from anyscale.utils.arm_utils import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional cloud deps loaded lazily")
                ARMTemplateUtils,
            )
        except ImportError as e:
            self.log.error(f"Failed to import required modules: {e}")
            raise click.ClickException(f"Failed to import required modules: {e}")

        # Validate inputs
        if not name or not isinstance(name, str):
            raise click.ClickException("Cloud name must be a non-empty string")

        # Generate unique resource names
        # Use 8-char hex for global uniqueness (storage accounts must be globally unique)
        random_suffix = os.urandom(4).hex()  # 8 hex characters

        # Storage account: 3-24 chars, lowercase alphanumeric only
        # "anyscale" (8) + random (8) = 16 chars
        storage_account_name = f"anyscale{random_suffix}"

        # Workload identity: up to 128 chars, alphanumeric, hyphens, underscores
        # Human-readable format: "anyscale-<cloud_name>-operator"
        sanitized_cloud_name = re.sub(r"[^a-zA-Z0-9-]", "-", name).strip("-").lower()
        # Truncate cloud name to fit within 128 char limit: "anyscale-" (9) + "-operator" (9) = 18 chars overhead
        max_cloud_name_len = 128 - 18
        truncated_cloud_name = sanitized_cloud_name[:max_cloud_name_len]
        workload_identity_name = f"anyscale-{truncated_cloud_name}-operator"

        self._debug(f"Azure storage account name: {storage_account_name}")
        self._debug(f"Azure workload identity name: {workload_identity_name}")

        try:
            # Generate ARM template
            self._debug("Generating ARM template...")
            # Validate oidc_provider is set (should be validated earlier in Azure flow)
            assert (
                cluster_info.oidc_provider is not None
            ), "OIDC provider must be set for Azure"
            arm_template_body = self._generate_azure_arm_template(
                oidc_issuer=cluster_info.oidc_provider,
                namespace=cluster_info.namespace,
                cloud_name=name,
            )
            self._debug("ARM template generated successfully")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to generate ARM template: {e}")
            raise click.ClickException(f"Failed to generate ARM template: {e}")

        try:
            # Get Azure subscription ID
            subscription_id = self._get_azure_subscription_id()
            self._debug(f"Using subscription: {subscription_id}")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to get Azure subscription: {e}")
            raise click.ClickException(f"Failed to get Azure subscription: {e}")

        outputs: dict = {}
        portal_url: str = ""
        try:
            with self.log.indent():
                self.log.info(
                    "Creating ARM deployment (this may take a few minutes)...",
                    block_label="Setup",
                )
                arm_utils = ARMTemplateUtils(subscription_id, self.log)
                # Use storage account name as deployment name (unique identifier)
                outputs, portal_url = arm_utils.deploy_template(
                    deployment_name=storage_account_name,
                    resource_group=cluster_info.resource_group,
                    template=json.loads(arm_template_body),
                    parameters={
                        "location": region,
                        "storageAccountName": storage_account_name,
                        "workloadIdentityName": workload_identity_name,
                    },
                    timeout_seconds=600,
                )
            self.log.info("ARM deployment completed", block_label="Setup")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to create ARM deployment: {e}")
            raise click.ClickException(f"Failed to create ARM deployment: {e}")

        try:
            self._debug("Retrieving ARM deployment outputs...")
            abfss_url = outputs.get("abfssUrl")
            blob_endpoint = outputs.get("blobEndpoint")
            workload_identity_principal_id = outputs.get("workloadIdentityPrincipalId")
            workload_identity_client_id = outputs.get("workloadIdentityClientId")

            if (
                not abfss_url
                or not blob_endpoint
                or not workload_identity_principal_id
                or not workload_identity_client_id
            ):
                raise click.ClickException(
                    "Failed to get required outputs from ARM deployment"
                )

            self._debug(f"ABFSS URL: {abfss_url}")
            self._debug(f"Blob Endpoint: {blob_endpoint}")
            self._debug(
                f"Workload Identity Principal ID: {workload_identity_principal_id}"
            )
            self._debug(f"Workload Identity Client ID: {workload_identity_client_id}")
        except Exception as e:  # noqa: BLE001
            self.log.error(f"Failed to get ARM deployment outputs: {e}")
            raise click.ClickException(f"Failed to get ARM deployment outputs: {e}")

        return InfrastructureResources(
            bucket_name=abfss_url,  # ABFSS URL: abfss://container@account.dfs.core.windows.net
            iam_role_arn=workload_identity_principal_id,  # Principal ID for Anyscale API
            region=region,
            resource_group=cluster_info.resource_group,
            workload_identity_client_id=workload_identity_client_id,  # Client ID for Helm values
            blob_endpoint=blob_endpoint,  # Blob endpoint URL for object_storage.endpoint
            deployment_url=portal_url,
        )

    def _get_gke_cluster_info(
        self, cluster_name: str, region: str, project_id: str
    ) -> Dict[str, Any]:
        """Get GKE cluster information using gcloud CLI."""
        try:
            # Try regional cluster first
            result = subprocess.run(
                [
                    "gcloud",
                    "container",
                    "clusters",
                    "describe",
                    cluster_name,
                    f"--region={region}",
                    f"--project={project_id}",
                    "--format=json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)

            # Try zonal cluster
            # Assuming zone 'a' if regional fails
            zone = f"{region}-a"
            result = subprocess.run(
                [
                    "gcloud",
                    "container",
                    "clusters",
                    "describe",
                    cluster_name,
                    f"--zone={zone}",
                    f"--project={project_id}",
                    "--format=json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Failed to get GKE cluster info: {e.stderr}")
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Failed to parse GKE cluster info: {e}")

    def _get_gke_zones(
        self, cluster_name: str, region: str, project_id: str
    ) -> List[str]:
        """Get zones where the GKE cluster's node pools are located."""
        try:
            cluster_info = self._get_gke_cluster_info(cluster_name, region, project_id)

            # Extract zones from node pools
            zones = []
            node_pools = cluster_info.get("nodePools", [])

            for pool in node_pools:
                # For zonal clusters, each node pool has locations
                pool_locations = pool.get("locations", [])
                zones.extend(pool_locations)

            # If no zones found from node pools, try cluster-level locations
            if not zones:
                cluster_locations = cluster_info.get("locations", [])
                if cluster_locations:
                    zones = cluster_locations

            # Remove duplicates and sort
            if zones:
                unique_zones = sorted(set(zones))
                self._debug(f"Discovered zones: {', '.join(unique_zones)}")
                return unique_zones
            else:
                # Fallback to default zones
                self._debug(
                    "No zones found in cluster info, falling back to default zones"
                )
                return [region + "-a", region + "-b", region + "-c"]

        except Exception as e:  # noqa: BLE001
            self._debug(f"Failed to get zones: {e}, using default zones")
            return [region + "-a", region + "-b", region + "-c"]

    def _configure_gcp_kubeconfig(
        self, cluster_name: str, region: str, project_id: str
    ) -> None:
        """Configure kubeconfig for GCP GKE cluster."""
        self.log.info(f"Configuring kubeconfig for GKE cluster: {cluster_name}")

        try:
            # Try regional cluster first
            result = subprocess.run(
                [
                    "gcloud",
                    "container",
                    "clusters",
                    "get-credentials",
                    cluster_name,
                    f"--region={region}",
                    f"--project={project_id}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self.log.info("GKE kubeconfig configured successfully")
                return

            # Try zonal cluster
            zone = f"{region}-a"
            subprocess.run(
                [
                    "gcloud",
                    "container",
                    "clusters",
                    "get-credentials",
                    cluster_name,
                    f"--zone={zone}",
                    f"--project={project_id}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.log.info("GKE kubeconfig configured successfully")
        except subprocess.CalledProcessError as e:
            raise click.ClickException(
                f"Failed to configure GKE kubeconfig: {e.stderr}"
            )

    def _get_eks_cluster_info(self, cluster_name: str, region: str) -> Dict[str, Any]:
        """Get EKS cluster information using AWS CLI."""
        try:
            result = subprocess.run(
                [
                    "aws",
                    "eks",
                    "describe-cluster",
                    "--name",
                    cluster_name,
                    "--region",
                    region,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            cluster_data = json.loads(result.stdout)
            return cluster_data.get("cluster", {})
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Failed to get EKS cluster info: {e.stderr}")

    def _get_eks_availability_zones(self, cluster_name: str, region: str) -> List[str]:
        """Get availability zones where the EKS cluster's subnets are located."""
        try:
            cluster_info = self._get_eks_cluster_info(cluster_name, region)
            subnet_ids = cluster_info.get("resourcesVpcConfig", {}).get("subnetIds", [])

            if not subnet_ids:
                raise click.ClickException(
                    f"No subnets found for EKS cluster '{cluster_name}'. "
                    "Cannot determine availability zones."
                )

            # Get subnet details to find their availability zones
            result = subprocess.run(
                [
                    "aws",
                    "ec2",
                    "describe-subnets",
                    "--subnet-ids",
                    *subnet_ids,
                    "--region",
                    region,
                    "--query",
                    "Subnets[*].AvailabilityZone",
                    "--output",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            zones = json.loads(result.stdout)
            # Remove duplicates and sort
            unique_zones = sorted(set(zones))

            if unique_zones:
                self._debug(f"Discovered availability zones: {', '.join(unique_zones)}")
                return unique_zones
            else:
                raise click.ClickException(
                    f"No availability zones found for EKS cluster '{cluster_name}'. "
                    "Please ensure the cluster subnets are properly configured."
                )

        except click.ClickException:
            raise
        except Exception as e:  # noqa: BLE001
            raise click.ClickException(
                f"Failed to get availability zones for EKS cluster '{cluster_name}': {e}"
            )

    def _get_eks_oidc_provider(self, cluster_name: str, region: str) -> str:
        """Get EKS OIDC provider URL for IRSA."""
        cluster_info = self._get_eks_cluster_info(cluster_name, region)
        identity = cluster_info.get("identity", {})
        oidc_issuer = identity.get("oidc", {}).get("issuer", "")

        if not oidc_issuer:
            raise click.ClickException(
                "Could not find OIDC issuer for EKS cluster. IRSA setup requires OIDC provider."
            )

        # Extract OIDC provider ARN
        # OIDC issuer URL format: https://oidc.eks.region.amazonaws.com/id/EXAMPLED539D4633E53CE8D
        if "oidc.eks." in oidc_issuer and ".amazonaws.com/id/" in oidc_issuer:
            oidc_id = oidc_issuer.split("/id/")[-1]
            account_id = self._get_aws_account_id()
            oidc_provider_arn = f"arn:aws:iam::{account_id}:oidc-provider/oidc.eks.{region}.amazonaws.com/id/{oidc_id}"
            return oidc_provider_arn

        raise click.ClickException(
            f"Could not parse OIDC provider from issuer URL: {oidc_issuer}"
        )

    def _get_aws_account_id(self) -> str:
        """Get AWS account ID."""
        try:
            result = subprocess.run(
                [
                    "aws",
                    "sts",
                    "get-caller-identity",
                    "--query",
                    "Account",
                    "--output",
                    "text",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Failed to get AWS account ID: {e.stderr}")

    def _configure_aws_kubeconfig(self, cluster_name: str, region: str) -> None:
        """Configure kubeconfig for AWS EKS cluster."""
        self.log.info(f"Configuring kubeconfig for EKS cluster: {cluster_name}")

        try:
            subprocess.run(
                [
                    "aws",
                    "eks",
                    "update-kubeconfig",
                    "--region",
                    region,
                    "--name",
                    cluster_name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.log.info("EKS kubeconfig configured successfully")
        except subprocess.CalledProcessError as e:
            raise click.ClickException(
                f"Failed to configure EKS kubeconfig: {e.stderr}"
            )

    def _configure_azure_kubeconfig(
        self, cluster_name: str, resource_group: str
    ) -> None:
        """Configure kubeconfig for Azure AKS cluster."""
        self.log.info(f"Configuring kubeconfig for AKS cluster: {cluster_name}")

        try:
            subprocess.run(
                [
                    "az",
                    "aks",
                    "get-credentials",
                    "--resource-group",
                    resource_group,
                    "--name",
                    cluster_name,
                    "--overwrite-existing",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.log.info("AKS kubeconfig configured successfully")
        except subprocess.CalledProcessError as e:
            raise click.ClickException(
                f"Failed to configure AKS kubeconfig: {e.stderr}"
            )

    def _get_aks_cluster_info(
        self, cluster_name: str, resource_group: str
    ) -> Dict[str, Any]:
        """Get AKS cluster information using Azure CLI."""
        try:
            result = subprocess.run(
                [
                    "az",
                    "aks",
                    "show",
                    "--name",
                    cluster_name,
                    "--resource-group",
                    resource_group,
                    "--output",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Failed to get AKS cluster info: {e.stderr}")
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Failed to parse AKS cluster info: {e}")

    def _get_aks_availability_zones(
        self, cluster_name: str, resource_group: str
    ) -> List[str]:
        """Get availability zones where the AKS cluster's node pools are located."""
        try:
            cluster_info = self._get_aks_cluster_info(cluster_name, resource_group)

            # Extract zones from agent pool profiles
            zones: List[str] = []
            agent_pools = cluster_info.get("agentPoolProfiles") or []

            for pool in agent_pools:
                pool_zones = pool.get("availabilityZones") or []
                zones.extend(pool_zones)

            # Remove duplicates and sort
            if len(zones) > 0:
                unique_zones = sorted(set(zones))
                self._debug(f"Discovered zones: {', '.join(unique_zones)}")
                return unique_zones
            else:
                raise click.ClickException(
                    f"No availability zones found for AKS cluster '{cluster_name}'. "
                    "Please ensure the cluster node pools have availability zones configured."
                )

        except click.ClickException:
            raise
        except Exception as e:  # noqa: BLE001
            raise click.ClickException(
                f"Failed to get availability zones for AKS cluster '{cluster_name}': {e}"
            )

    def _get_azure_subscription_id(self) -> str:
        """Get Azure subscription ID from Azure CLI."""
        try:
            result = subprocess.run(
                ["az", "account", "show", "--query", "id", "--output", "tsv",],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise click.ClickException(
                f"Failed to get Azure subscription ID: {e.stderr}"
            )

    def _get_azure_tenant_id(self) -> str:
        """Get Azure tenant ID from Azure CLI."""
        try:
            result = subprocess.run(
                ["az", "account", "show", "--query", "tenantId", "--output", "tsv",],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Failed to get Azure tenant ID: {e.stderr}")

    def _generate_azure_arm_template(
        self, oidc_issuer: str, namespace: str, cloud_name: str,
    ) -> str:
        """
        Generate ARM template for Azure Kubernetes setup.

        Creates:
        - Storage Account for object storage
        - Blob Container for data
        - Managed Identity for Workload Identity
        - Federated Identity Credential linking K8s SA to Managed Identity
        - Role Assignment granting storage permissions to Managed Identity

        The template expects these parameters to be passed at deployment time:
        - storageAccountName: Name for the storage account (3-24 chars, lowercase alphanumeric only)
        - workloadIdentityName: Name for the managed identity (up to 128 chars)

        Args:
            oidc_issuer: AKS OIDC issuer URL
            namespace: Kubernetes namespace for the service account
            cloud_name: User-provided cloud name (used in tags)
        """
        service_account_name = "anyscale-operator"

        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "location": {
                    "type": "string",
                    "defaultValue": "[resourceGroup().location]",
                    "metadata": {"description": "The location for all resources"},
                },
                "storageAccountName": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 24,
                    "metadata": {
                        "description": "Name for the storage account (3-24 chars, lowercase alphanumeric only)"
                    },
                },
                "workloadIdentityName": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                    "metadata": {
                        "description": "Name for the managed identity (up to 128 chars)"
                    },
                },
                "tags": {
                    "type": "object",
                    "defaultValue": {
                        "anyscale-cloud-name": cloud_name,
                        "created-by": "anyscale-cli",
                        "workload": "anyscale-operator",
                        "purpose": "anyscale-kubernetes-integration",
                    },
                    "metadata": {"description": "Tags to apply to all resources"},
                },
            },
            "variables": {
                "storageAccountName": "[parameters('storageAccountName')]",
                "blobContainerName": "anyscale-data",
                "workloadIdentityName": "[parameters('workloadIdentityName')]",
                "federatedCredentialName": "[concat(parameters('workloadIdentityName'), '-fedcred')]",
                "storageBlobDataOwnerRoleId": "[subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')]",
                "storageAccountResourceId": "[resourceId('Microsoft.Storage/storageAccounts', variables('storageAccountName'))]",
                "workloadIdentityResourceId": "[resourceId('Microsoft.ManagedIdentity/userAssignedIdentities', variables('workloadIdentityName'))]",
            },
            "resources": [
                {
                    "type": "Microsoft.Storage/storageAccounts",
                    "apiVersion": "2023-01-01",
                    "name": "[variables('storageAccountName')]",
                    "location": "[parameters('location')]",
                    "tags": "[parameters('tags')]",
                    "sku": {"name": "Standard_LRS"},
                    "kind": "StorageV2",
                    "properties": {
                        "accessTier": "Hot",
                        "supportsHttpsTrafficOnly": True,
                        "minimumTlsVersion": "TLS1_2",
                        "allowBlobPublicAccess": False,
                        "allowSharedKeyAccess": True,
                        "isHnsEnabled": True,
                        "networkAcls": {
                            "bypass": "AzureServices",
                            "defaultAction": "Allow",
                        },
                    },
                },
                {
                    "type": "Microsoft.Storage/storageAccounts/blobServices",
                    "apiVersion": "2023-01-01",
                    "name": "[concat(variables('storageAccountName'), '/default')]",
                    "dependsOn": ["[variables('storageAccountResourceId')]"],
                    "properties": {
                        "cors": {
                            "corsRules": [
                                {
                                    "allowedOrigins": [ANYSCALE_CORS_ORIGIN],
                                    "allowedMethods": ["GET", "HEAD", "PUT"],
                                    "maxAgeInSeconds": 3600,
                                    "exposedHeaders": ["*"],
                                    "allowedHeaders": ["*"],
                                }
                            ]
                        }
                    },
                },
                {
                    "type": "Microsoft.Storage/storageAccounts/blobServices/containers",
                    "apiVersion": "2023-01-01",
                    "name": "[concat(variables('storageAccountName'), '/default/', variables('blobContainerName'))]",
                    "dependsOn": [
                        "[variables('storageAccountResourceId')]",
                        "[resourceId('Microsoft.Storage/storageAccounts/blobServices', variables('storageAccountName'), 'default')]",
                    ],
                    "properties": {"publicAccess": "None"},
                },
                {
                    "type": "Microsoft.ManagedIdentity/userAssignedIdentities",
                    "apiVersion": "2023-01-31",
                    "name": "[variables('workloadIdentityName')]",
                    "location": "[parameters('location')]",
                    "tags": "[parameters('tags')]",
                },
                {
                    "type": "Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials",
                    "apiVersion": "2023-01-31",
                    "name": "[concat(variables('workloadIdentityName'), '/', variables('federatedCredentialName'))]",
                    "dependsOn": ["[variables('workloadIdentityResourceId')]"],
                    "properties": {
                        "audiences": ["api://AzureADTokenExchange"],
                        "issuer": oidc_issuer,
                        "subject": f"system:serviceaccount:{namespace}:{service_account_name}",
                    },
                },
                {
                    "type": "Microsoft.Storage/storageAccounts/providers/roleAssignments",
                    "apiVersion": "2022-04-01",
                    "name": "[concat(variables('storageAccountName'), '/Microsoft.Authorization/', guid(variables('storageAccountResourceId'), variables('workloadIdentityResourceId'), variables('storageBlobDataOwnerRoleId')))]",
                    "dependsOn": [
                        "[variables('storageAccountResourceId')]",
                        "[variables('workloadIdentityResourceId')]",
                    ],
                    "properties": {
                        "roleDefinitionId": "[variables('storageBlobDataOwnerRoleId')]",
                        "principalId": "[reference(variables('workloadIdentityResourceId'), '2023-01-31').principalId]",
                        "principalType": "ServicePrincipal",
                    },
                },
            ],
            "outputs": {
                "abfssUrl": {
                    "type": "string",
                    "value": "[concat('abfss://', variables('blobContainerName'), '@', variables('storageAccountName'), '.dfs.core.windows.net')]",
                    "metadata": {
                        "description": "ABFSS URL for bucket_name (abfss://<container>@<account>.dfs.core.windows.net)"
                    },
                },
                "blobEndpoint": {
                    "type": "string",
                    "value": "[substring(reference(resourceId('Microsoft.Storage/storageAccounts', variables('storageAccountName')), '2023-01-01').primaryEndpoints.blob, 0, sub(length(reference(resourceId('Microsoft.Storage/storageAccounts', variables('storageAccountName')), '2023-01-01').primaryEndpoints.blob), 1))]",
                    "metadata": {
                        "description": "Blob endpoint URL for object_storage.endpoint (https://<account>.blob.core.windows.net)"
                    },
                },
                "workloadIdentityPrincipalId": {
                    "type": "string",
                    "value": "[reference(variables('workloadIdentityResourceId'), '2023-01-31').principalId]",
                    "metadata": {
                        "description": "Principal ID (Object ID) of the managed identity - used for Anyscale API"
                    },
                },
                "workloadIdentityClientId": {
                    "type": "string",
                    "value": "[reference(variables('workloadIdentityResourceId'), '2023-01-31').clientId]",
                    "metadata": {
                        "description": "Client ID (Application ID) of the managed identity - used for Helm annotations"
                    },
                },
            },
        }

        return json.dumps(template, indent=2)

    def _verify_kubeconfig(self) -> None:
        """Verify that kubeconfig is working correctly."""
        self.log.info("Verifying kubeconfig configuration...")

        try:
            subprocess.run(
                ["kubectl", "cluster-info"], capture_output=True, text=True, check=True
            )
            self.log.info("Kubeconfig verification successful")
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Kubeconfig verification failed: {e.stderr}")

    def _get_current_kubectl_context(self) -> str:
        """Get the current kubectl context."""
        try:
            result = subprocess.run(
                ["kubectl", "config", "current-context"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise click.ClickException(
                f"Failed to get current kubectl context: {e.stderr}"
            )

    def _register_cloud(  # noqa: PLR0912
        self,
        name: str,
        provider: str,
        region: str,
        infrastructure: InfrastructureResources,
        cluster_info: ClusterInfo,
    ) -> str:
        """Register the cloud with Anyscale."""
        self.log.info("Registering cloud with Anyscale...", block_label="Setup")

        if provider == "aws":
            # Dynamically determine availability zones from the EKS cluster
            zones = self._get_eks_availability_zones(cluster_info.cluster_name, region)

            cloud_deployment = CloudDeployment(
                name=name,
                provider=CloudProviders.AWS,
                region=region,
                compute_stack=ComputeStack.K8S,
                object_storage=ObjectStorage(
                    bucket_name=infrastructure.bucket_name, region=region
                ),
                aws_config=AWSConfig(),
                kubernetes_config=OpenAPIKubernetesConfig(
                    anyscale_operator_iam_identity=infrastructure.iam_role_arn,
                    zones=zones,
                ),
            )
        elif provider == "gcp":
            assert infrastructure.project_id, "Project ID is required for GCP"

            from anyscale.client.openapi_client.models import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional cloud deps loaded lazily")
                GCPConfig,
            )

            # Dynamically determine zones from the GKE cluster
            zones = self._get_gke_zones(
                cluster_info.cluster_name, region, infrastructure.project_id
            )

            cloud_deployment = CloudDeployment(
                name=name,
                provider=CloudProviders.GCP,
                region=region,
                compute_stack=ComputeStack.K8S,
                object_storage=ObjectStorage(
                    bucket_name=infrastructure.bucket_name, region=region
                ),
                gcp_config=GCPConfig(project_id=infrastructure.project_id,),
                kubernetes_config=OpenAPIKubernetesConfig(
                    anyscale_operator_iam_identity=infrastructure.iam_role_arn,
                    zones=zones,
                ),
            )
        elif provider == "azure":
            assert infrastructure.resource_group, "Resource group is required for Azure"

            from anyscale.client.openapi_client.models import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional cloud deps loaded lazily")
                AzureConfig,
            )

            # Dynamically determine zones from the AKS cluster
            zones = self._get_aks_availability_zones(
                cluster_info.cluster_name, infrastructure.resource_group
            )

            # Get Azure tenant ID
            tenant_id = self._get_azure_tenant_id()

            cloud_deployment = CloudDeployment(
                name=name,
                provider=CloudProviders.AZURE,
                region=region,
                compute_stack=ComputeStack.K8S,
                object_storage=ObjectStorage(
                    bucket_name=infrastructure.bucket_name,
                    region=region,
                    endpoint=infrastructure.blob_endpoint,  # Blob endpoint URL
                ),
                azure_config=AzureConfig(tenant_id=tenant_id),
                kubernetes_config=OpenAPIKubernetesConfig(
                    anyscale_operator_iam_identity=infrastructure.iam_role_arn,  # Principal ID
                    zones=zones,
                ),
            )
        else:
            raise click.ClickException(f"Unsupported provider: {provider}")

        # Register the cloud
        try:
            self._debug("Cloud deployment details:")
            self._debug(f"  Name: {cloud_deployment.name}")
            self._debug(f"  Provider: {cloud_deployment.provider}")
            self._debug(f"  Region: {cloud_deployment.region}")
            self._debug(f"  Compute Stack: {cloud_deployment.compute_stack}")
            self._debug(f"  Bucket Name: {cloud_deployment.object_storage.bucket_name}")
            self._debug(
                f"  IAM Identity: {cloud_deployment.kubernetes_config.anyscale_operator_iam_identity}"
            )
            if cloud_deployment.aws_config:
                self._debug("  AWS Config:")
                self._debug(
                    f"    IAM Role ID: {cloud_deployment.aws_config.anyscale_iam_role_id}"
                )

            # Temporarily suppress cloud controller logging to avoid Helm command output
            original_log_info = self.cloud_controller.log.info
            self.cloud_controller.log.info = lambda *_args, **_kwargs: None

            try:
                if provider == "aws":
                    self.log.info("Registering AWS cloud...")
                    self.cloud_controller.register_aws_cloud(
                        name=name,
                        cloud_resource=cloud_deployment,
                        functional_verify=None,
                        yes=True,
                        skip_verifications=True,
                        auto_add_user=True,
                    )
                elif provider == "gcp":
                    self.log.info("Registering GCP cloud...")
                    self.cloud_controller.register_gcp_cloud(
                        name=name,
                        cloud_resource=cloud_deployment,
                        functional_verify=None,
                        yes=True,
                        skip_verifications=True,
                        auto_add_user=True,
                    )
                elif provider == "azure":
                    self.log.info("Registering Azure cloud...")
                    self.cloud_controller.register_azure_or_generic_cloud(
                        name=name,
                        provider="azure",
                        cloud_resource=cloud_deployment,
                        auto_add_user=True,
                    )
                else:
                    raise click.ClickException(f"Unsupported provider: {provider}")
            finally:
                # Restore the original log.info method
                self.cloud_controller.log.info = original_log_info

            self._debug("Cloud registration completed, fetching cloud ID...")
            # Use get_cloud_id_and_name helper to fetch the registered cloud
            from anyscale.cloud_utils import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional cloud deps loaded lazily")
                get_cloud_id_and_name,
            )

            try:
                cloud_id, _ = get_cloud_id_and_name(
                    self.cloud_controller.api_client, cloud_name=name
                )
            except Exception as e:  # noqa: BLE001
                raise click.ClickException(f"Failed to find registered cloud: {e}")

            if not cloud_id:
                raise click.ClickException(
                    "Failed to get cloud ID from registered cloud"
                )

            self.log.info(f"Cloud registered with ID: {cloud_id}", block_label="Setup")

            return cloud_id

        except Exception as e:  # noqa: BLE001
            self.log.error(f"Cloud registration failed with error: {e}")
            self.log.error(f"Error type: {type(e).__name__}")
            if hasattr(e, "response"):
                self.log.error(f"Response details: {getattr(e, 'response', 'N/A')}")
            if hasattr(e, "args"):
                self.log.error(f"Error args: {e.args}")
            self.log.error(f"Full traceback: {traceback.format_exc()}")
            raise click.ClickException(f"Failed to register cloud: {e}")

    def _install_operator(  # noqa: PLR0913
        self,
        cloud_resource_id: str,
        provider: str,
        region: str,
        namespace: str,
        infrastructure: InfrastructureResources,
        values_file: Optional[str] = None,
        operator_chart: Optional[str] = None,
        skip_confirmation: bool = False,
    ) -> None:
        """Install the Anyscale operator using Helm."""
        self.log.info("Installing Anyscale operator...", block_label="Setup")

        release_name = "anyscale-operator"

        # Prompt user about nginx ingress installation
        install_nginx = self._prompt_for_nginx_ingress(skip_confirmation)

        # Generate Helm command and extract --set-string flags from it
        self._debug("Generating Helm command to extract parameters...")
        helm_command = self.cloud_controller._generate_helm_upgrade_command(  # noqa: SLF001
            provider=provider,
            cloud_deployment_id=cloud_resource_id,
            region=region,
            operator_iam_identity=infrastructure.iam_role_arn,
        )

        set_string_values = self._extract_set_string_values(helm_command)
        self._debug(f"Extracted {len(set_string_values)} --set-string parameters")

        values_file_path = self._generate_helm_values_file(
            provider=provider,
            cloud_deployment_id=cloud_resource_id,
            region=region,
            namespace=namespace,
            infrastructure=infrastructure,
            custom_path=values_file,
            additional_values=set_string_values,
            install_nginx_ingress=install_nginx,
        )

        # Determine chart reference based on operator_chart parameter
        if operator_chart:
            # Use the provided chart path directly
            self._debug(f"Using operator chart from: {operator_chart}")
            chart_reference = operator_chart
        else:
            # Add Helm repo before installing
            self._debug("Adding Anyscale Helm repository...")
            self._add_helm_repo()
            chart_reference = "anyscale/anyscale-operator"

        # Build a simple Helm command that only uses the values file
        self._debug("Generating Helm command...")
        helm_command = (
            f"helm upgrade {release_name} {chart_reference} "
            f"--values {values_file_path} "
            f"--namespace {namespace} "
            f"--create-namespace "
            f"--wait "
            f"-i"
        )

        self._execute_helm_command(helm_command)

    def _add_helm_repo(self) -> None:
        """Add and update the Anyscale Helm repository."""
        try:
            # Add the Anyscale Helm repository
            self.log.info("Adding Anyscale Helm repository...", block_label="Setup")
            subprocess.run(
                [
                    "helm",
                    "repo",
                    "add",
                    "anyscale",
                    "https://anyscale.github.io/helm-charts",
                ],
                capture_output=True,
                text=True,
                check=False,  # Don't fail if repo already exists
            )

            # Update the Helm repository
            self.log.info("Updating Helm repositories...", block_label="Setup")
            subprocess.run(
                ["helm", "repo", "update", "anyscale"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.log.info(
                "Helm repository configured successfully", block_label="Setup"
            )
        except subprocess.CalledProcessError as e:
            self.log.error(f"Failed to configure Helm repository: {e.stderr}")
            raise click.ClickException(
                f"Failed to configure Helm repository: {e.stderr}"
            )

    def _extract_set_string_values(self, helm_command: str) -> Dict[str, str]:
        """
        Extract all --set-string key=value pairs from a Helm command.

        Args:
            helm_command: The Helm command string to parse

        Returns:
            Dictionary of key-value pairs from --set-string flags
        """
        set_string_values = {}

        # Pattern to match --set-string key=value
        pattern = r"--set-string\s+(\S+?)=(\S+)"

        matches = re.findall(pattern, helm_command)
        for key, value in matches:
            set_string_values[key] = value

        return set_string_values

    def _set_nested_value(self, d: Dict[str, Any], key_path: str, value: Any) -> None:
        """
        Set a value in a nested dictionary using a dotted key path.

        Args:
            d: The dictionary to modify
            key_path: Dotted key path (e.g., "workloads.serviceaccount.name")
            value: The value to set

        Example:
            _set_nested_value({}, "workloads.serviceaccount.name", "my-sa")
            # Results in: {"workloads": {"serviceaccount": {"name": "my-sa"}}}
        """
        keys = key_path.split(".")
        current = d

        # Navigate/create the nested structure
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # If the key exists but isn't a dict, we have a conflict
                # In this case, we'll overwrite it with a dict
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    def _prompt_for_namespace(
        self, default_namespace: str, skip_confirmation: bool = False
    ) -> str:
        """Prompt user for namespace confirmation."""
        final_namespace = default_namespace or "anyscale-operator"

        if skip_confirmation:
            self.log.info(f"Using namespace: {final_namespace}", block_label="Setup")
            return final_namespace

        self.log.info("Configuring Kubernetes namespace...")
        self.log.info(
            f"Specify the namespace to use for the Anyscale operator (leave blank for default: {final_namespace})."
        )
        self.log.info("If the namespace does not exist, it will be created.")
        self.log.info("Enter your namespace:")

        final_namespace = click.prompt("", default=final_namespace, show_default=True)

        # Validate namespace (Kubernetes DNS-1123 label requirements)
        # Must be lowercase alphanumeric or hyphens, start and end with alphanumeric, max 63 chars

        if not final_namespace:
            raise click.ClickException("Namespace cannot be empty")
        if len(final_namespace) > 63:
            raise click.ClickException("Namespace must be 63 characters or less")
        if not re.match(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", final_namespace):
            raise click.ClickException(
                "Namespace must consist of lowercase alphanumeric characters or hyphens, "
                "and must start and end with an alphanumeric character"
            )

        self.log.info(f"Using namespace: {final_namespace}")

        return final_namespace

    def _prompt_for_nginx_ingress(self, skip_confirmation: bool = False) -> bool:
        """Prompt user whether to install nginx ingress subchart."""
        if skip_confirmation:
            self.log.info("Using default: nginx ingress subchart will be installed")
            return True

        self.log.info(
            "The Anyscale operator can install an nginx ingress controller as part of the setup.",
            block_label="Setup",
        )
        self.log.info(
            "If you already have an ingress controller installed, you may want to skip this.",
            block_label="Setup",
        )

        response = click.confirm(
            "Do you want to install the nginx ingress subchart?", default=True
        )

        if response:
            self.log.info(
                "nginx ingress subchart will be installed", block_label="Setup"
            )
        else:
            self.log.info(
                "nginx ingress subchart will NOT be installed", block_label="Setup"
            )

        return response

    def _generate_helm_values_file(  # noqa: PLR0913
        self,
        provider: str,
        cloud_deployment_id: str,
        region: str,
        namespace: str,
        infrastructure: InfrastructureResources,
        custom_path: Optional[str] = None,
        additional_values: Optional[Dict[str, str]] = None,
        install_nginx_ingress: bool = True,
    ) -> str:
        """Generate Helm values file and save it locally."""
        self.log.info("Generating Helm values file...")

        # Start with an empty dictionary to build up values
        values: Dict[str, Any] = {}

        # First, parse and merge additional_values with nested keys
        if additional_values:
            for key, value in additional_values.items():
                self._set_nested_value(values, key, value)

        # Now overlay our constants on top (these take precedence)
        # Use _set_nested_value to ensure proper nesting
        self._set_nested_value(values, "global.cloudDeploymentId", cloud_deployment_id)
        self._set_nested_value(values, "global.cloudProvider", provider)

        # For Azure, use Client ID for Helm; for others, use iam_role_arn
        if provider == "azure":
            iam_identity = infrastructure.workload_identity_client_id
        else:
            iam_identity = infrastructure.iam_role_arn

        self._set_nested_value(values, "global.auth.iamIdentity", iam_identity)
        self._set_nested_value(values, "ingress-nginx.enabled", install_nginx_ingress)

        # Add ingress-nginx annotations based on cloud provider if nginx ingress is enabled
        if install_nginx_ingress:
            self._set_ingress_nginx_annotations(values, provider)

        # Add region for AWS only (using global.aws.region)
        # Region field is deprecated for other providers
        if provider == "aws":
            self._set_nested_value(values, "global.aws.region", region)

        # Add control plane URL from ANYSCALE_HOST environment variable
        if ANYSCALE_HOST:
            self._set_nested_value(values, "global.controlPlaneURL", ANYSCALE_HOST)
            self.log.info(f"Using control plane URL: {ANYSCALE_HOST}")

        if custom_path:
            values_file_path = custom_path
        else:
            # Create filename with random suffix for uniqueness
            random_suffix = random.randint(1000, 9999)
            filename = (
                f"anyscale-helm-values-{provider}-{namespace}-{random_suffix}.yaml"
            )
            values_file_path = os.path.join(os.getcwd(), filename)

        with open(values_file_path, "w") as f:
            yaml.dump(values, f, default_flow_style=False, sort_keys=False)

        self.log.info(f"Generated Helm values file: {values_file_path}")

        return values_file_path

    def _set_ingress_nginx_annotations(
        self, values: Dict[str, Any], provider: str
    ) -> None:

        self.log.info("Setting ingress-nginx annotations...")
        self.log.warning(
            "The ingress-nginx controller is in maintenance mode and will be retired as of March 2026."
        )

        # Build annotations dictionary based on provider
        annotations = {}
        if provider == "aws":
            # AWS NLB configuration for ingress-nginx
            annotations = {
                "service.beta.kubernetes.io/aws-load-balancer-scheme": "internet-facing",
                "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled": "true",
                "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
            }
        elif provider == "gcp":
            # GCP external load balancer configuration for ingress-nginx
            annotations = {
                "cloud.google.com/load-balancer-type": "External",
            }
        elif provider == "azure":
            # Azure health probe configuration for ingress-nginx
            annotations = {
                "service.beta.kubernetes.io/azure-load-balancer-health-probe-request-path": "/healthz",
            }

        # Use _set_nested_value to create the structure, then set annotations dict
        if annotations:
            self._set_nested_value(
                values, "ingress-nginx.controller.service.annotations", annotations
            )

    def _execute_helm_command(self, helm_command: str) -> None:
        """Execute the helm command."""
        # Convert multi-line command to single line and execute
        single_line_command = helm_command.replace(" \\\n", " ").replace("\n", " ")

        self.log.info(f"Executing: {single_line_command}")

        try:
            subprocess.run(
                single_line_command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
            self.log.info("Helm installation completed successfully")
        except subprocess.CalledProcessError as e:
            self.log.error(f"Helm installation failed: {e.stderr}")
            raise click.ClickException(
                f"Failed to install Anyscale operator: {e.stderr}"
            )

    def _verify_installation(
        self,
        cloud_id: str,
        namespace: str,
        cluster_info: ClusterInfo,
        cloud_resource_id: Optional[str] = None,
    ) -> None:
        """Verify the Kubernetes installation."""
        self.log.info("Verifying installation...")

        # Get the cloud deployment
        cloud_resources = self.cloud_controller.get_cloud_resources(cloud_id)

        if not cloud_resources:
            raise click.ClickException("No cloud resources found for verification")

        # Find the specific cloud resource if cloud_resource_id is provided
        cloud_deployment = None
        if cloud_resource_id:
            for resource in cloud_resources:
                if resource.cloud_resource_id == cloud_resource_id:
                    cloud_deployment = resource
                    break
            if not cloud_deployment:
                raise click.ClickException(
                    f"Could not find cloud resource with ID {cloud_resource_id}"
                )
        else:
            # Fallback to first resource for backward compatibility
            cloud_deployment = cloud_resources[0]

        # Use the existing Kubernetes verifier
        verifier = KubernetesCloudDeploymentVerifier(
            self.log, self.cloud_controller.api_client
        )

        # Set up kubectl config for verification using the discovered context
        verifier.k8s_config = KubernetesConfig(
            context=cluster_info.context,  # Use the discovered context to avoid re-prompting
            operator_namespace=namespace,
        )

        # Run verification
        success = verifier.verify(cloud_deployment)

        if success:
            self.log.info("Verification completed successfully")
        else:
            self.log.error("Verification failed - please check the logs above")
            raise click.ClickException("Installation verification failed")

    def _create_cloud_resource(  # noqa: PLR0912
        self,
        cloud_id: str,
        provider: str,
        region: str,
        infrastructure: InfrastructureResources,
        cluster_info: ClusterInfo,
        resource_name: Optional[str],
    ) -> str:
        """
        Create a cloud resource in an existing cloud and return the cloud_resource_id.

        Args:
            cloud_id: ID of the existing cloud
            provider: Cloud provider (aws, gcp, azure)
            region: Cloud region
            infrastructure: Infrastructure resources created
            cluster_info: Cluster information
            resource_name: Name for the cloud resource (optional, will be auto-generated if not provided)

        Returns:
            The cloud_resource_id of the created resource
        """
        self.log.info("Creating cloud resource in Anyscale...", block_label="Setup")
        if resource_name:
            self.log.info(f"Using resource name: {resource_name}", block_label="Setup")
        else:
            self.log.info("Resource name will be auto-generated", block_label="Setup")

        if provider == "aws":
            # Dynamically determine availability zones from the EKS cluster
            zones = self._get_eks_availability_zones(cluster_info.cluster_name, region)

            cloud_deployment = CloudDeployment(
                name=resource_name,
                provider=CloudProviders.AWS,
                region=region,
                compute_stack=ComputeStack.K8S,
                object_storage=ObjectStorage(
                    bucket_name=infrastructure.bucket_name, region=region
                ),
                aws_config=AWSConfig(),
                kubernetes_config=OpenAPIKubernetesConfig(
                    anyscale_operator_iam_identity=infrastructure.iam_role_arn,
                    zones=zones,
                ),
            )
        elif provider == "gcp":
            assert infrastructure.project_id, "Project ID is required for GCP"

            from anyscale.client.openapi_client.models import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional cloud deps loaded lazily")
                GCPConfig,
            )

            # Dynamically determine zones from the GKE cluster
            zones = self._get_gke_zones(
                cluster_info.cluster_name, region, infrastructure.project_id
            )

            cloud_deployment = CloudDeployment(
                name=resource_name,
                provider=CloudProviders.GCP,
                region=region,
                compute_stack=ComputeStack.K8S,
                object_storage=ObjectStorage(
                    bucket_name=infrastructure.bucket_name, region=region
                ),
                gcp_config=GCPConfig(project_id=infrastructure.project_id),
                kubernetes_config=OpenAPIKubernetesConfig(
                    anyscale_operator_iam_identity=infrastructure.iam_role_arn,
                    zones=zones,
                ),
            )
        else:
            raise click.ClickException(f"Unsupported provider: {provider}")

        # Create cloud resource using the API
        try:
            self._debug("Cloud deployment details:")
            self._debug(f"  Provider: {cloud_deployment.provider}")
            self._debug(f"  Region: {cloud_deployment.region}")
            self._debug(f"  Compute Stack: {cloud_deployment.compute_stack}")
            self._debug(f"  Bucket Name: {cloud_deployment.object_storage.bucket_name}")
            self._debug(
                f"  IAM Identity: {cloud_deployment.kubernetes_config.anyscale_operator_iam_identity}"
            )

            # Save cloud deployment to a temporary file and use create_cloud_resource
            self._debug("Saving cloud deployment to temporary file...")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as temp_file:
                # Convert CloudDeployment to dict for YAML serialization
                deployment_dict = cloud_deployment.to_dict()
                yaml.dump(deployment_dict, temp_file, default_flow_style=False)
                temp_file_path = temp_file.name

            try:
                self._debug(f"Created temporary spec file: {temp_file_path}")
                self._debug("Calling cloud_controller.create_cloud_resource...")

                # Use cloud_controller's create_cloud_resource method which now returns the cloud_resource_id
                cloud_resource_id = self.cloud_controller.create_cloud_resource(
                    cloud=None,
                    cloud_id=cloud_id,
                    spec_file=temp_file_path,
                    skip_verification=True,  # We will do verification in the _verify_installation method
                    yes=True,  # Skip confirmation prompts
                )
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file_path)
                    self._debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:  # noqa: BLE001
                    self._debug(f"Failed to clean up temporary file: {e}")

            self.log.info(
                f"Cloud resource created with ID: {cloud_resource_id}",
                block_label="Setup",
            )

            return cloud_resource_id

        except Exception as e:  # noqa: BLE001
            self.log.error(f"Cloud resource creation failed with error: {e}")
            self.log.error(f"Error type: {type(e).__name__}")
            if hasattr(e, "response"):
                self.log.error(f"Response details: {getattr(e, 'response', 'N/A')}")
            if hasattr(e, "args"):
                self.log.error(f"Error args: {e.args}")
            self.log.error(f"Full traceback: {traceback.format_exc()}")
            raise click.ClickException(f"Failed to create cloud resource: {e}")

    def _handle_setup_failure(
        self,
        provider: str,
        infrastructure: Optional[InfrastructureResources],
        cloud_id: Optional[str],
        name: str,
        is_cloud_resource_setup: bool = False,
    ) -> None:
        """Handle setup failure by providing cleanup instructions to the user."""
        self.log.error("")
        self.log.error("=" * 80)
        self.log.error("SETUP FAILED - MANUAL CLEANUP REQUIRED")
        self.log.error("=" * 80)
        self.log.error("")

        if is_cloud_resource_setup:
            self.log.error(
                "The Kubernetes cloud resource setup failed, leaving resources in an incomplete state."
            )
        else:
            self.log.error(
                "The Kubernetes cloud setup failed, leaving resources in an incomplete state."
            )

        self.log.error(
            "You must manually clean up the following resources to avoid charges:"
        )
        self.log.error("")

        if provider == "aws":
            self._log_aws_cleanup_instructions(
                infrastructure, cloud_id, name, is_cloud_resource_setup
            )
        elif provider == "gcp":
            self._log_gcp_cleanup_instructions(
                infrastructure, cloud_id, name, is_cloud_resource_setup
            )
        elif provider == "azure":
            self._log_azure_cleanup_instructions(
                infrastructure, cloud_id, name, is_cloud_resource_setup
            )

        self.log.error("")
        self.log.error("=" * 80)

    def _log_aws_cleanup_instructions(
        self,
        infrastructure: Optional[InfrastructureResources],
        cloud_id: Optional[str],
        name: str,
        is_cloud_resource_setup: bool = False,
    ) -> None:
        """Log AWS-specific cleanup instructions."""
        self.log.error("AWS Resources to clean up:")
        self.log.error("")

        if infrastructure:
            self.log.error("1. CloudFormation Stack:")
            # Stack name pattern: k8s-{name}-{random} with underscores replaced by hyphens
            stack_name_pattern = f"k8s-{name}-*".replace("_", "-").lower()
            self.log.error(
                f"   - Find and delete the CloudFormation stack matching pattern: {stack_name_pattern}"
            )
            self.log.error(f"   - Region: {infrastructure.region}")
            self.log.error(
                "   - AWS Console: CloudFormation > Stacks > Select stack > Delete"
            )
            self.log.error(
                f"   - AWS CLI: aws cloudformation delete-stack --stack-name <stack-name> --region {infrastructure.region}"
            )
            self.log.error("")
            self.log.error("   This will automatically delete:")
            self.log.error(f"   - S3 Bucket: {infrastructure.bucket_name}")
            self.log.error(f"   - IAM Role: {infrastructure.iam_role_arn}")
            self.log.error("")

        if cloud_id:
            if is_cloud_resource_setup:
                self.log.error("2. Anyscale Cloud Resource:")
                self.log.error(
                    f"   - Delete the cloud resource from cloud '{name}' (ID: {cloud_id})"
                )
                self.log.error(
                    f"   - CLI: anyscale cloud resource delete --cloud '{name}' --resource <resource-name>"
                )
                self.log.error(
                    "   - To find the resource name, run: anyscale cloud get --name '{name}'"
                )
            else:
                self.log.error("2. Anyscale Cloud Registration:")
                self.log.error(
                    f"   - Delete the cloud '{name}' (ID: {cloud_id}) from Anyscale"
                )
                self.log.error(f"   - CLI: anyscale cloud delete --name '{name}'")
            self.log.error("")

        if not infrastructure:
            self.log.error(
                "No infrastructure resources were created before the failure."
            )
            self.log.error("")

    def _log_gcp_cleanup_instructions(
        self,
        infrastructure: Optional[InfrastructureResources],
        cloud_id: Optional[str],
        name: str,
        is_cloud_resource_setup: bool = False,
    ) -> None:
        """Log GCP-specific cleanup instructions."""
        self.log.error("GCP Resources to clean up:")
        self.log.error("")

        if infrastructure:
            self.log.error("1. GCS Bucket:")
            self.log.error(f"   - Bucket: {infrastructure.bucket_name}")
            self.log.error(f"   - Project: {infrastructure.project_id}")
            self.log.error(
                "   - GCP Console: Cloud Storage > Buckets > Select bucket > Delete"
            )
            self.log.error(
                f"   - gcloud CLI: gsutil rm -r gs://{infrastructure.bucket_name}"
            )
            self.log.error("")

            self.log.error("2. Service Account:")
            self.log.error(f"   - Service Account: {infrastructure.iam_role_arn}")
            self.log.error(f"   - Project: {infrastructure.project_id}")
            self.log.error(
                "   - GCP Console: IAM & Admin > Service Accounts > Select account > Delete"
            )
            self.log.error(
                f"   - gcloud CLI: gcloud iam service-accounts delete {infrastructure.iam_role_arn} --project={infrastructure.project_id}"
            )
            self.log.error("")

        if cloud_id:
            if is_cloud_resource_setup:
                self.log.error("3. Anyscale Cloud Resource:")
                self.log.error(
                    f"   - Delete the cloud resource from cloud '{name}' (ID: {cloud_id})"
                )
                self.log.error(
                    f"   - CLI: anyscale cloud resource delete --cloud '{name}' --resource <resource-name>"
                )
                self.log.error(
                    "   - To find the resource name, run: anyscale cloud get --name '{name}'"
                )
                self.log.error(
                    f"   - Console: {ANYSCALE_HOST}/clouds (if using custom host)"
                )
            else:
                self.log.error("3. Anyscale Cloud Registration:")
                self.log.error(
                    f"   - Delete the cloud '{name}' (ID: {cloud_id}) from Anyscale"
                )
                self.log.error(f"   - CLI: anyscale cloud delete --name '{name}'")

    def _log_azure_cleanup_instructions(
        self,
        infrastructure: Optional[InfrastructureResources],
        cloud_id: Optional[str],
        name: str,
        is_cloud_resource_setup: bool = False,
    ) -> None:
        """Log Azure-specific cleanup instructions."""
        self.log.error("Azure Resources to clean up:")
        self.log.error("")

        if infrastructure:
            self.log.error("1. ARM Deployment:")
            if infrastructure.deployment_url:
                self.log.error(
                    f"   - View and delete the ARM deployment at: {infrastructure.deployment_url}"
                )
            else:
                self.log.error(
                    f"   - Find ARM deployments in resource group: {infrastructure.resource_group}"
                )
                self.log.error(
                    f"   - Azure Portal: Resource Groups > {infrastructure.resource_group} > Deployments"
                )
            self.log.error("")

        if cloud_id:
            if is_cloud_resource_setup:
                self.log.error("2. Anyscale Cloud Resource:")
                self.log.error(
                    f"   - Delete the cloud resource from cloud '{name}' (ID: {cloud_id})"
                )
                self.log.error(
                    f"   - CLI: anyscale cloud resource delete --cloud '{name}' --resource <resource-name>"
                )
                self.log.error(
                    f"   - To find the resource name, run: anyscale cloud get --name '{name}'"
                )
                self.log.error(
                    f"   - Console: {ANYSCALE_HOST}/clouds (if using custom host)"
                )
            else:
                self.log.error("2. Anyscale Cloud Registration:")
                self.log.error(
                    f"   - Delete the cloud '{name}' (ID: {cloud_id}) from Anyscale"
                )
                self.log.error(f"   - CLI: anyscale cloud delete --name '{name}'")
                self.log.error(
                    f"   - Console: {ANYSCALE_HOST}/clouds (if using custom host)"
                )
            self.log.error("")

        if not infrastructure:
            self.log.error(
                "No infrastructure resources were created before the failure."
            )
            self.log.error("")


def setup_kubernetes_cloud(  # noqa: PLR0913
    provider: str,
    region: str,
    name: str,
    cluster_name: str,
    namespace: str = "anyscale-operator",
    project_id: Optional[str] = None,
    resource_group: Optional[str] = None,
    functional_verify: bool = False,
    yes: bool = False,
    values_file: Optional[str] = None,
    debug: bool = False,
    operator_chart: Optional[str] = None,
) -> None:
    """
    Set up Anyscale on a Kubernetes cluster.

    This function can be called from multiple CLI commands and provides
    the core K8s setup functionality.

    Args:
        provider: Cloud provider (aws, gcp, azure)
        region: Cloud region
        name: Name for the Anyscale cloud
        cluster_name: Kubernetes cluster name
        namespace: Namespace for Anyscale operator (default: anyscale-operator)
        project_id: GCP project ID (optional, for future GCP support)
        functional_verify: Whether to run functional verification
        yes: Skip confirmation prompts
        values_file: Optional path for Helm values file
        debug: Enable debug logging
        operator_chart: Optional path to operator chart (skips helm repo add/update)
    """
    cmd = KubernetesCloudSetupCommand(debug=debug)

    try:
        cmd.run(
            provider=provider,
            region=region,
            name=name,
            cluster_name=cluster_name,
            namespace=namespace,
            project_id=project_id,
            resource_group=resource_group,
            functional_verify=functional_verify,
            yes=yes,
            values_file=values_file,
            operator_chart=operator_chart,
        )
    except Exception as e:  # noqa: BLE001
        click.echo(f"Setup failed: {e}", err=True)
        raise click.Abort()


def setup_kubernetes_cloud_resource(  # noqa: PLR0913
    provider: str,
    region: str,
    cloud_name: Optional[str],
    cloud_id: Optional[str],
    cluster_name: str,
    resource_name: Optional[str],
    namespace: str = "anyscale-operator",
    project_id: Optional[str] = None,
    resource_group: Optional[str] = None,
    functional_verify: bool = False,
    yes: bool = False,
    values_file: Optional[str] = None,
    debug: bool = False,
    operator_chart: Optional[str] = None,
) -> None:
    """
    Set up cloud resources for an existing Anyscale cloud on a Kubernetes cluster.

    This function sets up infrastructure and installs the operator without
    registering a new cloud.

    Args:
        provider: Cloud provider (aws, gcp, azure)
        region: Cloud region
        cloud_name: Name of existing Anyscale cloud
        cloud_id: ID of existing Anyscale cloud
        cluster_name: Kubernetes cluster name
        resource_name: Name for the cloud resource (optional, will be auto-generated if not provided)
        namespace: Namespace for Anyscale operator (default: anyscale-operator)
        project_id: GCP project ID (optional, for GCP)
        resource_group: Resource group for Azure clouds (optional, for Azure)
        functional_verify: Whether to run functional verification
        yes: Skip confirmation prompts
        values_file: Optional path for Helm values file
        debug: Enable debug logging
        operator_chart: Optional path to operator chart (skips helm repo add/update)
    """
    cmd = KubernetesCloudSetupCommand(debug=debug)

    # Preprocessing: Fetch full cloud info to ensure cloud exists and get both name and ID
    if not cloud_id and not cloud_name:
        click.echo("Either cloud_name or cloud_id must be provided", err=True)
        raise click.Abort()

    if cloud_id and cloud_name:
        click.echo("Only one of cloud_name or cloud_id can be provided", err=True)
        raise click.Abort()

    # Use get_cloud_id_and_name to validate cloud exists and get both ID and name
    try:
        from anyscale.cloud_utils import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional cloud deps loaded lazily")
            get_cloud_id_and_name,
        )

        if cloud_id:
            cloud_id, cloud_name = get_cloud_id_and_name(
                cmd.cloud_controller.api_client, cloud_id=cloud_id
            )
        else:
            cloud_id, cloud_name = get_cloud_id_and_name(
                cmd.cloud_controller.api_client, cloud_name=cloud_name
            )

    except Exception as e:  # noqa: BLE001
        click.echo(f"Failed to fetch cloud information: {e}", err=True)
        raise click.Abort()

    if not cloud_id or not cloud_name:
        click.echo("Could not find cloud with provided name or ID", err=True)
        raise click.Abort()

    try:
        # Use the unified run method with cloud_id to indicate resource-only mode
        cmd.run(
            provider=provider,
            region=region,
            name=cloud_name,
            cluster_name=cluster_name,
            namespace=namespace,
            project_id=project_id,
            resource_group=resource_group,
            functional_verify=functional_verify,
            yes=yes,
            values_file=values_file,
            operator_chart=operator_chart,
            cloud_id=cloud_id,
            resource_name=resource_name,
        )
    except Exception as e:  # noqa: BLE001
        click.echo(f"Setup failed: {e}", err=True)
        raise click.Abort()
