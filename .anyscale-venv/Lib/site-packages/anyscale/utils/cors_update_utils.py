"""CORS update utilities for cloud storage buckets.

This module provides utilities to check and update CORS configurations
on cloud storage buckets (S3, GCS, Azure Blob) to support Anyscale UI
features like the file viewer.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    import boto3

from anyscale.cloud_resource import check_aws_cors_rules
from anyscale.shared_anyscale_utils.conf import (
    ANYSCALE_CORS_EXPOSE_HEADERS,
    ANYSCALE_CORS_ORIGIN,
)


def check_aws_cors_needs_update(
    bucket_name: str, region: str, boto3_session: Optional["boto3.Session"] = None
) -> Tuple[bool, str]:
    """
    Check if AWS S3 bucket CORS needs update for file viewer support.

    Args:
        bucket_name: The S3 bucket name (without s3:// prefix)
        region: AWS region
        boto3_session: Optional boto3 session to use

    Returns:
        Tuple of (needs_update, reason)
    """
    try:
        import boto3 as boto3_module  # noqa: PLC0415
        from botocore.exceptions import ClientError  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "AWS SDK is not installed. Please install it with: pip install boto3"
        ) from e

    if boto3_session is None:
        boto3_session = boto3_module.Session()

    s3 = boto3_session.resource("s3", region_name=region)
    bucket = s3.Bucket(bucket_name)

    try:
        cors_rules = bucket.Cors().cors_rules
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchCORSConfiguration":
            return True, "No CORS configuration exists"
        raise

    is_correct, reason = check_aws_cors_rules(cors_rules)
    return not is_correct, reason


def update_aws_cors(
    bucket_name: str, region: str, boto3_session: Optional["boto3.Session"] = None
) -> None:
    """
    Update AWS S3 bucket CORS configuration.

    Args:
        bucket_name: The S3 bucket name (without s3:// prefix)
        region: AWS region
        boto3_session: Optional boto3 session to use
    """
    try:
        import boto3 as boto3_module  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "AWS SDK is not installed. Please install it with: pip install boto3"
        ) from e

    if boto3_session is None:
        boto3_session = boto3_module.Session()

    s3_client = boto3_session.client("s3", region_name=region)
    s3_client.put_bucket_cors(
        Bucket=bucket_name,
        CORSConfiguration={
            "CORSRules": [
                {
                    "AllowedHeaders": ["*"],
                    "AllowedMethods": ["GET", "PUT", "POST", "HEAD", "DELETE"],
                    "AllowedOrigins": [ANYSCALE_CORS_ORIGIN],
                    "ExposeHeaders": ANYSCALE_CORS_EXPOSE_HEADERS,
                    "MaxAgeSeconds": 3600,
                }
            ]
        },
    )


def check_gcp_cors_needs_update(
    bucket_name: str, project_id: Optional[str]
) -> Tuple[bool, str]:
    """
    Check if GCP GCS bucket CORS needs update.

    Args:
        bucket_name: The GCS bucket name (without gs:// prefix)
        project_id: GCP project ID

    Returns:
        Tuple of (needs_update, reason)
    """
    try:
        from google.cloud import storage  # noqa: PLC0415

        from anyscale.gcp_verification import check_gcp_cors_rules  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Google Cloud SDK is not installed. Please install it with: "
            "pip install google-cloud-storage"
        ) from e

    client = storage.Client(project=project_id)
    bucket = client.get_bucket(bucket_name)

    is_correct, reason = check_gcp_cors_rules(bucket.cors or [])
    return not is_correct, reason


def update_gcp_cors(bucket_name: str, project_id: Optional[str]) -> None:
    """
    Update GCP GCS bucket CORS configuration.

    Args:
        bucket_name: The GCS bucket name (without gs:// prefix)
        project_id: GCP project ID
    """
    try:
        from google.cloud import storage  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Google Cloud SDK is not installed. Please install it with: "
            "pip install google-cloud-storage"
        ) from e

    client = storage.Client(project=project_id)
    bucket = client.get_bucket(bucket_name)
    bucket.cors = [
        {
            "origin": [ANYSCALE_CORS_ORIGIN],
            "responseHeader": ["*"],
            "method": ["GET", "PUT", "POST", "HEAD", "DELETE"],
            "maxAgeSeconds": 3600,
        }
    ]
    bucket.patch()


def check_azure_cors_rules(cors_rules: List) -> Tuple[bool, str]:
    """
    Check if Azure Blob storage CORS rules are correctly configured for Anyscale.

    This is a shared helper used by both CORS update utilities and cloud verification.

    Args:
        cors_rules: List of CorsRule objects from Azure Blob storage

    Returns:
        Tuple of (is_correct, reason)
    """
    if not cors_rules:
        return False, "No CORS configuration exists"

    for rule in cors_rules:
        if ANYSCALE_CORS_ORIGIN in rule.allowed_origins and "*" in rule.exposed_headers:
            return True, "CORS already configured correctly"

    return False, "CORS missing required exposedHeaders for file viewer"


def check_azure_cors_needs_update(storage_account_name: str) -> Tuple[bool, str]:
    """
    Check if Azure Blob storage CORS needs update.

    Args:
        storage_account_name: Azure storage account name

    Returns:
        Tuple of (needs_update, reason)
    """
    try:
        from azure.identity import DefaultAzureCredential  # noqa: PLC0415
        from azure.storage.blob import BlobServiceClient  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Azure SDK is not installed. Please install it with: "
            "pip install azure-identity azure-storage-blob"
        ) from e

    credential = DefaultAzureCredential()
    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)

    props = blob_service.get_service_properties()
    cors_rules = props.get("cors", [])

    is_correct, reason = check_azure_cors_rules(cors_rules)
    return not is_correct, reason


def update_azure_cors(storage_account_name: str) -> None:
    """
    Update Azure Blob storage CORS configuration.

    Args:
        storage_account_name: Azure storage account name
    """
    try:
        from azure.identity import DefaultAzureCredential  # noqa: PLC0415
        from azure.storage.blob import BlobServiceClient, CorsRule  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Azure SDK is not installed. Please install it with: "
            "pip install azure-identity azure-storage-blob"
        ) from e

    credential = DefaultAzureCredential()
    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)

    cors_rule = CorsRule(
        allowed_origins=[ANYSCALE_CORS_ORIGIN],
        allowed_methods=["GET", "HEAD", "PUT"],
        allowed_headers=["*"],
        exposed_headers=["*"],
        max_age_in_seconds=3600,
    )
    blob_service.set_service_properties(cors=[cors_rule])
