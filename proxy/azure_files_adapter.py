"""Azure Files adapter for S3 proxy."""
import logging
from typing import List, Optional, BinaryIO
from azure.storage.fileshare import ShareServiceClient, ShareClient, ShareFileClient
from azure.core.exceptions import ResourceNotFoundError
import os

from .config import ProxyConfig

logger = logging.getLogger(__name__)


class AzureFilesAdapter:
    """Adapter to translate S3 operations to Azure Files REST API."""
    
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.account_url = f"https://{config.storage_account_name}.file.core.windows.net"
        
        # Initialize Azure Files Service Client
        self.file_service_client = ShareServiceClient(
            account_url=self.account_url,
            credential=config.storage_account_key
        )
        
    def _get_share_client(self, bucket_name: str) -> ShareClient:
        """Get share client for the given bucket (mapped to file share)."""
        share_name = self.config.get_container_name(bucket_name)
        return self.file_service_client.get_share_client(share_name)
    
    def _get_file_client(self, bucket_name: str, key: str) -> ShareFileClient:
        """Get file client for the given bucket and key."""
        share_name = self.config.get_container_name(bucket_name)
        
        # Parse directory path and file name from key
        # S3 key format: "path/to/file.txt" -> directory: "path/to", file: "file.txt"
        directory_path = os.path.dirname(key)
        file_name = os.path.basename(key)
        
        share_client = self.file_service_client.get_share_client(share_name)
        
        if directory_path:
            directory_client = share_client.get_directory_client(directory_path)
            return directory_client.get_file_client(file_name)
        else:
            return share_client.get_file_client(file_name)
    
    def _ensure_directory(self, bucket_name: str, key: str) -> None:
        """Ensure directory structure exists for the given key."""
        directory_path = os.path.dirname(key)
        if not directory_path:
            return
            
        share_client = self._get_share_client(bucket_name)
        
        # Create nested directories
        path_parts = directory_path.split('/')
        current_path = ""
        
        for part in path_parts:
            if current_path:
                current_path = f"{current_path}/{part}"
            else:
                current_path = part
                
            try:
                directory_client = share_client.get_directory_client(current_path)
                directory_client.create_directory()
            except Exception as e:
                # Directory might already exist - check for specific error
                if "ResourceExists" not in str(type(e).__name__):
                    # Re-raise if it's not a "directory already exists" error
                    logger.debug(f"Directory creation returned: {e}")
                pass
    
    def put_object(self, bucket_name: str, key: str, data: BinaryIO,
                   content_type: Optional[str] = None) -> dict:
        """Upload file to Azure Files (S3 PUT)."""
        try:
            # Ensure directory structure exists
            self._ensure_directory(bucket_name, key)
            
            file_client = self._get_file_client(bucket_name, key)
            
            # Read data
            file_data = data.read() if hasattr(data, 'read') else data
            
            # Upload file
            file_client.upload_file(
                file_data,
                metadata={"ContentType": content_type} if content_type else None
            )
            
            logger.info(f"Uploaded file: {bucket_name}/{key}")
            
            # Get properties for ETag
            properties = file_client.get_file_properties()
            return {"ETag": properties.etag}
            
        except Exception as e:
            logger.error(f"Error uploading file {bucket_name}/{key}: {str(e)}")
            raise
    
    def get_object(self, bucket_name: str, key: str) -> tuple:
        """Download file from Azure Files (S3 GET)."""
        try:
            file_client = self._get_file_client(bucket_name, key)
            
            # Download file
            download_stream = file_client.download_file()
            properties = file_client.get_file_properties()
            
            metadata = {
                "ContentLength": properties.size,
                "ContentType": properties.metadata.get("ContentType", "application/octet-stream") if properties.metadata else "application/octet-stream",
                "ETag": properties.etag,
                "LastModified": properties.last_modified
            }
            
            logger.info(f"Downloaded file: {bucket_name}/{key}")
            return download_stream.readall(), metadata
            
        except ResourceNotFoundError:
            logger.warning(f"File not found: {bucket_name}/{key}")
            raise
        except Exception as e:
            logger.error(f"Error downloading file {bucket_name}/{key}: {str(e)}")
            raise
    
    def delete_object(self, bucket_name: str, key: str) -> None:
        """Delete file from Azure Files (S3 DELETE)."""
        try:
            file_client = self._get_file_client(bucket_name, key)
            file_client.delete_file()
            logger.info(f"Deleted file: {bucket_name}/{key}")
            
        except ResourceNotFoundError:
            logger.warning(f"File not found for deletion: {bucket_name}/{key}")
            # S3 DELETE is idempotent, so we don't raise error
        except Exception as e:
            logger.error(f"Error deleting file {bucket_name}/{key}: {str(e)}")
            raise
    
    def list_objects(self, bucket_name: str, prefix: str = "",
                    max_keys: int = 1000) -> dict:
        """List files in Azure Files (S3 LIST)."""
        try:
            share_client = self._get_share_client(bucket_name)
            
            contents = []
            count = 0
            
            # List files recursively
            def list_directory(directory_client, current_prefix=""):
                nonlocal count
                
                for item in directory_client.list_directories_and_files():
                    if count >= max_keys:
                        break
                        
                    item_path = f"{current_prefix}/{item.name}" if current_prefix else item.name
                    
                    # Check if matches prefix filter
                    if prefix and not item_path.startswith(prefix):
                        continue
                    
                    if item.is_directory:
                        # Recursively list subdirectory
                        subdir_client = directory_client.get_subdirectory_client(item.name)
                        list_directory(subdir_client, item_path)
                    else:
                        # Add file to results
                        contents.append({
                            "Key": item_path,
                            "Size": item.size,
                            "LastModified": item.last_modified if hasattr(item, 'last_modified') else None
                        })
                        count += 1
            
            # Start listing from root or specified prefix directory
            if prefix and '/' in prefix:
                # Get directory part of prefix
                prefix_dir = os.path.dirname(prefix)
                try:
                    directory_client = share_client.get_directory_client(prefix_dir)
                    list_directory(directory_client, prefix_dir)
                except ResourceNotFoundError:
                    # Directory doesn't exist, return empty list
                    pass
            else:
                root_directory = share_client.get_directory_client("")
                list_directory(root_directory)
            
            logger.info(f"Listed {count} files in {bucket_name} with prefix '{prefix}'")
            return {
                "Contents": contents,
                "IsTruncated": count >= max_keys,
                "MaxKeys": max_keys,
                "Prefix": prefix
            }
            
        except Exception as e:
            logger.error(f"Error listing files in {bucket_name}: {str(e)}")
            raise
    
    def head_object(self, bucket_name: str, key: str) -> dict:
        """Get file metadata from Azure Files (S3 HEAD)."""
        try:
            file_client = self._get_file_client(bucket_name, key)
            properties = file_client.get_file_properties()
            
            metadata = {
                "ContentLength": properties.size,
                "ContentType": properties.metadata.get("ContentType", "application/octet-stream") if properties.metadata else "application/octet-stream",
                "ETag": properties.etag,
                "LastModified": properties.last_modified
            }
            
            logger.info(f"Got file metadata: {bucket_name}/{key}")
            return metadata
            
        except ResourceNotFoundError:
            logger.warning(f"File not found: {bucket_name}/{key}")
            raise
        except Exception as e:
            logger.error(f"Error getting file metadata {bucket_name}/{key}: {str(e)}")
            raise
