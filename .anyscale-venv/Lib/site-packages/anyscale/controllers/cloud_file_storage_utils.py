"""
Cloud File Storage Utilities

Common functions for verifying AWS EFS and GCP Filestore resources
across different cloud verification contexts.
"""

from typing import Dict, Optional

import boto3

from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models.cloud_deployment import CloudDeployment
from anyscale.client.openapi_client.models.cloud_providers import CloudProviders
from anyscale.client.openapi_client.models.file_storage import FileStorage
from anyscale.util import _get_aws_efs_mount_target_ip


class GCPFilestoreInfo:
    """Container for GCP Filestore information"""

    def __init__(
        self,
        filestore_id: str,
        exists: bool,
        mount_target_ip: Optional[str] = None,
        root_dir: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        self.filestore_id = filestore_id
        self.exists = exists
        self.mount_target_ip = mount_target_ip
        self.root_dir = root_dir
        self.details = details or {}


def _get_gcp_client_factory(logger, cloud_deployment: CloudDeployment):
    """Helper function to get GCP client factory."""
    from anyscale.cli_logger import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP deps loaded only when needed")
        CloudSetupLogger,
    )
    from anyscale.utils.gcp_utils import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP deps loaded only when needed")
        get_google_cloud_client_factory,
    )

    gcp_config = cloud_deployment.gcp_config or {}
    project_id = gcp_config.get("project_id")

    if not project_id:
        raise ValueError("No GCP project_id found in cloud deployment config")

    # Create GCP client factory - need to handle logger type mismatch
    if isinstance(logger, CloudSetupLogger):
        return get_google_cloud_client_factory(logger, project_id), project_id
    else:
        # Create a CloudSetupLogger wrapper for compatibility
        cloud_setup_logger = CloudSetupLogger()
        return (
            get_google_cloud_client_factory(cloud_setup_logger, project_id),
            project_id,
        )


def _get_detailed_filestore_config(
    factory, vpc_name: Optional[str], filestore_id: str, logger: BlockLogger
):
    """Helper function to get detailed filestore configuration."""
    if not vpc_name:
        return None, None

    try:
        from anyscale.cli_logger import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP deps loaded only when needed")
            CloudSetupLogger,
        )
        from anyscale.utils import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP deps loaded only when needed")
            gcp_utils,
        )

        cloud_setup_logger = CloudSetupLogger()
        filestore_config = gcp_utils.get_gcp_filestore_config_from_full_name(
            factory, vpc_name, filestore_id, cloud_setup_logger
        )
        if filestore_config:
            return filestore_config.mount_target_ip, filestore_config.root_dir
    except (ImportError, AttributeError, KeyError) as e:
        logger.warning(f"Could not retrieve detailed filestore config: {e}")

    return None, None


def get_gcp_filestore_info(
    filestore_id: str,
    cloud_deployment: CloudDeployment,
    logger: Optional[BlockLogger] = None,
) -> GCPFilestoreInfo:
    """
    Common function to fetch GCP Filestore information.

    Args:
        filestore_id: The GCP Filestore instance ID
        cloud_deployment: CloudDeployment object containing GCP config
        logger: Optional logger for output

    Returns:
        GCPFilestoreInfo object containing Filestore details and existence status
    """
    if logger is None:
        logger = BlockLogger()

    try:
        from anyscale.cli_logger import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP deps loaded only when needed")
            CloudSetupLogger,
        )
        from anyscale.gcp_verification import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP deps loaded only when needed")
            GCPLogger,
            verify_filestore,
        )

        # Get GCP config and client factory
        factory, project_id = _get_gcp_client_factory(logger, cloud_deployment)

        gcp_config = cloud_deployment.gcp_config or {}
        vpc_name = gcp_config.get("vpc_name")

        if not vpc_name:
            logger.warning("No VPC name found - may affect connectivity checks")

        # Create GCP logger for filestore verification
        gcp_logger = GCPLogger(CloudSetupLogger(), project_id, None, True)

        # Use existing filestore verification logic
        region = cloud_deployment.region or ""
        verification_result = verify_filestore(
            factory=factory,
            file_store_instance_name=filestore_id,
            vpc_name=vpc_name,
            cloud_region=region,
            logger=gcp_logger,
            strict=False,
        )

        # Try to get detailed filestore config if available
        mount_target_ip, root_dir = _get_detailed_filestore_config(
            factory, vpc_name, filestore_id, logger
        )

        if verification_result:
            logger.info(f"GCP Filestore {filestore_id} exists and is accessible")
            return GCPFilestoreInfo(
                filestore_id=filestore_id,
                exists=True,
                mount_target_ip=mount_target_ip,
                root_dir=root_dir,
            )
        else:
            logger.error(f"GCP Filestore {filestore_id} verification failed")
            return GCPFilestoreInfo(filestore_id=filestore_id, exists=False)

    except ImportError as e:
        logger.warning(f"GCP verification dependencies not available: {e}")
        # Don't fail verification if GCP deps are missing
        return GCPFilestoreInfo(filestore_id=filestore_id, exists=True)
    except (ValueError, AttributeError, KeyError) as e:
        logger.error(f"Configuration error checking GCP Filestore {filestore_id}: {e}")
        return GCPFilestoreInfo(filestore_id=filestore_id, exists=False)


def verify_file_storage_exists(
    file_storage: FileStorage,
    cloud_deployment: CloudDeployment,
    logger: Optional[BlockLogger] = None,
) -> bool:
    """
    Generic function to verify file storage exists based on cloud provider.

    Args:
        file_storage: File storage configuration object
        cloud_deployment: CloudDeployment object
        logger: Optional logger for output

    Returns:
        bool: True if file storage exists, False otherwise
    """
    if logger is None:
        logger = BlockLogger()

    file_storage_id = getattr(file_storage, "file_storage_id", None)
    if not file_storage_id:
        logger.warning("No file_storage_id provided - skipping existence check")
        return True

    provider_value = (
        getattr(cloud_deployment.provider, "value", str(cloud_deployment.provider))
        if cloud_deployment.provider
        else None
    )

    if provider_value == CloudProviders.AWS:
        region = cloud_deployment.region
        if not region:
            logger.error("No region found for AWS EFS verification")
            return False

        boto3_session = boto3.Session(region_name=region)
        return _get_aws_efs_mount_target_ip(boto3_session, file_storage_id) is not None

    elif provider_value == CloudProviders.GCP:
        filestore_info = get_gcp_filestore_info(
            file_storage_id, cloud_deployment, logger
        )
        return filestore_info.exists

    else:
        logger.warning(
            f"File storage existence check not supported for provider: {provider_value}"
        )
        return True
