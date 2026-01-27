"""Unit tests for proxy configuration."""
import os
import pytest
from proxy.config import ProxyConfig, StorageBackend


def test_default_config():
    """Test default configuration values."""
    # Clear environment
    for key in ["PROXY_HOST", "PROXY_PORT", "STORAGE_BACKEND", "SA", "STORAGE_ACCOUNT_NAME"]:
        os.environ.pop(key, None)
    
    # Set minimal required config
    os.environ["STORAGE_ACCOUNT_NAME"] = "testaccount"
    os.environ["STORAGE_ACCOUNT_KEY"] = "testkey"
    
    config = ProxyConfig()
    
    assert config.proxy_host == "0.0.0.0"
    assert config.proxy_port == 5000
    assert config.storage_backend == StorageBackend.BLOB
    assert config.storage_account_name == "testaccount"
    assert config.storage_account_key == "testkey"


def test_blob_backend():
    """Test blob backend configuration."""
    os.environ["STORAGE_BACKEND"] = "blob"
    os.environ["STORAGE_ACCOUNT_NAME"] = "testaccount"
    os.environ["STORAGE_ACCOUNT_KEY"] = "testkey"
    
    config = ProxyConfig()
    
    assert config.is_blob_backend() is True
    assert config.is_files_backend() is False
    assert config.storage_backend == StorageBackend.BLOB


def test_files_backend():
    """Test files backend configuration."""
    os.environ["STORAGE_BACKEND"] = "files"
    os.environ["STORAGE_ACCOUNT_NAME"] = "testaccount"
    os.environ["STORAGE_ACCOUNT_KEY"] = "testkey"
    
    config = ProxyConfig()
    
    assert config.is_blob_backend() is False
    assert config.is_files_backend() is True
    assert config.storage_backend == StorageBackend.FILES


def test_bucket_mapping():
    """Test bucket to container mapping."""
    os.environ["STORAGE_ACCOUNT_NAME"] = "testaccount"
    os.environ["STORAGE_ACCOUNT_KEY"] = "testkey"
    os.environ["BUCKET_MAPPING"] = "bucket1:container1,bucket2:container2"
    
    config = ProxyConfig()
    
    assert config.get_container_name("bucket1") == "container1"
    assert config.get_container_name("bucket2") == "container2"
    # When no wildcard and no match, returns bucket name itself
    assert config.get_container_name("unknown") == "unknown"


def test_wildcard_mapping():
    """Test wildcard bucket mapping."""
    os.environ["STORAGE_ACCOUNT_NAME"] = "testaccount"
    os.environ["STORAGE_ACCOUNT_KEY"] = "testkey"
    os.environ["BUCKET_MAPPING"] = ""
    os.environ["CONTAINER"] = "mycontainer"
    
    config = ProxyConfig()
    
    # Wildcard should map any bucket to default container
    assert config.get_container_name("anybucket") == "mycontainer"


def test_validate_missing_account_name():
    """Test validation fails without account name."""
    # Clear environment
    os.environ.pop("STORAGE_ACCOUNT_NAME", None)
    os.environ.pop("SA", None)
    os.environ["STORAGE_ACCOUNT_KEY"] = "testkey"
    
    config = ProxyConfig()
    
    with pytest.raises(ValueError, match="Storage account name is required"):
        config.validate()


def test_validate_missing_account_key():
    """Test validation fails without account key."""
    os.environ["STORAGE_ACCOUNT_NAME"] = "testaccount"
    os.environ.pop("STORAGE_ACCOUNT_KEY", None)
    
    config = ProxyConfig()
    
    with pytest.raises(ValueError, match="Storage account key is required"):
        config.validate()


def test_custom_port():
    """Test custom proxy port configuration."""
    os.environ["PROXY_PORT"] = "8080"
    os.environ["STORAGE_ACCOUNT_NAME"] = "testaccount"
    os.environ["STORAGE_ACCOUNT_KEY"] = "testkey"
    
    config = ProxyConfig()
    
    assert config.proxy_port == 8080


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
