"""Azure Blob Storage adapter for S3 proxy."""
import logging
from typing import List, Optional, BinaryIO
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError

from .config import ProxyConfig

logger = logging.getLogger(__name__)


class AzureBlobAdapter:
    """Adapter to translate S3 operations to Azure Blob Storage REST API."""
    
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.account_url = f"https://{config.storage_account_name}.blob.core.windows.net"
        
        # Initialize Azure Blob Service Client
        self.blob_service_client = BlobServiceClient(
            account_url=self.account_url,
            credential=config.storage_account_key
        )
        
    def _get_container_client(self, bucket_name: str) -> ContainerClient:
        """Get container client for the given bucket."""
        container_name = self.config.get_container_name(bucket_name)
        return self.blob_service_client.get_container_client(container_name)
    
    def _get_blob_client(self, bucket_name: str, key: str) -> BlobClient:
        """Get blob client for the given bucket and key."""
        container_name = self.config.get_container_name(bucket_name)
        return self.blob_service_client.get_blob_client(
            container=container_name,
            blob=key
        )
    
    def put_object(self, bucket_name: str, key: str, data: BinaryIO, 
                   content_type: Optional[str] = None) -> dict:
        """Upload object to Azure Blob Storage (S3 PUT)."""
        try:
            blob_client = self._get_blob_client(bucket_name, key)
            
            # Upload blob
            content_settings = None
            if content_type:
                from azure.storage.blob import ContentSettings
                content_settings = ContentSettings(content_type=content_type)
            
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=content_settings
            )
            
            logger.info(f"Uploaded blob: {bucket_name}/{key}")
            return {"ETag": blob_client.get_blob_properties().etag}
            
        except Exception as e:
            logger.error(f"Error uploading blob {bucket_name}/{key}: {str(e)}")
            raise
    
    def get_object(self, bucket_name: str, key: str) -> tuple:
        """Download object from Azure Blob Storage (S3 GET)."""
        try:
            blob_client = self._get_blob_client(bucket_name, key)
            
            # Download blob
            download_stream = blob_client.download_blob()
            properties = blob_client.get_blob_properties()
            
            metadata = {
                "ContentLength": properties.size,
                "ContentType": properties.content_settings.content_type or "application/octet-stream",
                "ETag": properties.etag,
                "LastModified": properties.last_modified
            }
            
            logger.info(f"Downloaded blob: {bucket_name}/{key}")
            return download_stream.readall(), metadata
            
        except ResourceNotFoundError:
            logger.warning(f"Blob not found: {bucket_name}/{key}")
            raise
        except Exception as e:
            logger.error(f"Error downloading blob {bucket_name}/{key}: {str(e)}")
            raise
    
    def delete_object(self, bucket_name: str, key: str) -> None:
        """Delete object from Azure Blob Storage (S3 DELETE)."""
        try:
            blob_client = self._get_blob_client(bucket_name, key)
            blob_client.delete_blob()
            logger.info(f"Deleted blob: {bucket_name}/{key}")
            
        except ResourceNotFoundError:
            logger.warning(f"Blob not found for deletion: {bucket_name}/{key}")
            # S3 DELETE is idempotent, so we don't raise error
        except Exception as e:
            logger.error(f"Error deleting blob {bucket_name}/{key}: {str(e)}")
            raise
    
    def list_objects(self, bucket_name: str, prefix: str = "", 
                    max_keys: int = 1000) -> dict:
        """List objects in Azure Blob Storage (S3 LIST)."""
        try:
            container_client = self._get_container_client(bucket_name)
            
            # List blobs with prefix
            blobs = container_client.list_blobs(
                name_starts_with=prefix if prefix else None
            )
            
            contents = []
            count = 0
            for blob in blobs:
                if count >= max_keys:
                    break
                    
                contents.append({
                    "Key": blob.name,
                    "Size": blob.size,
                    "LastModified": blob.last_modified,
                    "ETag": blob.etag
                })
                count += 1
            
            logger.info(f"Listed {count} objects in {bucket_name} with prefix '{prefix}'")
            return {
                "Contents": contents,
                "IsTruncated": count >= max_keys,
                "MaxKeys": max_keys,
                "Prefix": prefix
            }
            
        except Exception as e:
            logger.error(f"Error listing blobs in {bucket_name}: {str(e)}")
            raise
    
    def head_object(self, bucket_name: str, key: str) -> dict:
        """Get object metadata from Azure Blob Storage (S3 HEAD)."""
        try:
            blob_client = self._get_blob_client(bucket_name, key)
            properties = blob_client.get_blob_properties()
            
            metadata = {
                "ContentLength": properties.size,
                "ContentType": properties.content_settings.content_type or "application/octet-stream",
                "ETag": properties.etag,
                "LastModified": properties.last_modified
            }
            
            logger.info(f"Got blob metadata: {bucket_name}/{key}")
            return metadata
            
        except ResourceNotFoundError:
            logger.warning(f"Blob not found: {bucket_name}/{key}")
            raise
        except Exception as e:
            logger.error(f"Error getting blob metadata {bucket_name}/{key}: {str(e)}")
            raise
