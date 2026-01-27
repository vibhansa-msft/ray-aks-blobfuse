"""Configuration module for S3 to Azure Storage proxy."""
import os
from enum import Enum
from typing import Optional


class StorageBackend(str, Enum):
    """Supported Azure storage backends."""
    BLOB = "blob"
    FILES = "files"


class ProxyConfig:
    """Configuration for the S3 to Azure proxy service."""
    
    def __init__(self):
        # Proxy settings
        self.proxy_host = os.getenv("PROXY_HOST", "0.0.0.0")
        self.proxy_port = int(os.getenv("PROXY_PORT", "5000"))
        
        # Storage backend selection
        backend_str = os.getenv("STORAGE_BACKEND", "blob").lower()
        self.storage_backend = StorageBackend(backend_str)
        
        # Azure credentials
        self.storage_account_name = os.getenv("SA", os.getenv("STORAGE_ACCOUNT_NAME", ""))
        self.storage_account_key = os.getenv("STORAGE_ACCOUNT_KEY", "")
        self.container_name = os.getenv("CONTAINER", "datasets")
        
        # Azure Files specific
        self.file_share_name = os.getenv("FILE_SHARE_NAME", "datasets")
        
        # S3 to Azure mapping
        self.bucket_to_container_map = self._parse_bucket_mapping()
        
    def _parse_bucket_mapping(self) -> dict:
        """Parse S3 bucket to Azure container/share mapping from environment."""
        mapping = {}
        mapping_str = os.getenv("BUCKET_MAPPING", "")
        
        if mapping_str:
            # Format: "bucket1:container1,bucket2:container2"
            for pair in mapping_str.split(","):
                if ":" in pair:
                    bucket, container = pair.split(":", 1)
                    mapping[bucket.strip()] = container.strip()
        
        # Default mapping: use the default container for all buckets
        if not mapping:
            mapping["*"] = self.container_name
            
        return mapping
    
    def get_container_name(self, bucket_name: str) -> str:
        """Get Azure container/share name for S3 bucket."""
        # Check for exact match
        if bucket_name in self.bucket_to_container_map:
            return self.bucket_to_container_map[bucket_name]
        
        # Check for wildcard mapping
        if "*" in self.bucket_to_container_map:
            return self.bucket_to_container_map["*"]
        
        # Default to bucket name
        return bucket_name
    
    def is_blob_backend(self) -> bool:
        """Check if using Azure Blob Storage backend."""
        return self.storage_backend == StorageBackend.BLOB
    
    def is_files_backend(self) -> bool:
        """Check if using Azure Files backend."""
        return self.storage_backend == StorageBackend.FILES
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.storage_account_name:
            raise ValueError("Storage account name is required (SA or STORAGE_ACCOUNT_NAME env var)")
        
        if not self.storage_account_key:
            raise ValueError("Storage account key is required (STORAGE_ACCOUNT_KEY env var)")
