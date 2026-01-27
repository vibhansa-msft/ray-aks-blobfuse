"""Unit tests for S3 handler."""
import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import Request
from fastapi.responses import Response
from proxy.s3_handler import S3Handler
import io


@pytest.fixture
def mock_storage_adapter():
    """Create a mock storage adapter."""
    adapter = Mock()
    adapter.put_object = Mock(return_value={"ETag": '"abc123"'})
    adapter.get_object = Mock(return_value=(
        b"test data",
        {
            "ContentType": "text/plain",
            "ContentLength": 9,
            "ETag": '"abc123"',
            "LastModified": "2024-01-01T00:00:00Z"
        }
    ))
    adapter.delete_object = Mock()
    adapter.head_object = Mock(return_value={
        "ContentType": "text/plain",
        "ContentLength": 9,
        "ETag": '"abc123"',
        "LastModified": "2024-01-01T00:00:00Z"
    })
    adapter.list_objects = Mock(return_value={
        "Contents": [
            {"Key": "file1.txt", "Size": 100, "LastModified": "2024-01-01T00:00:00Z"},
            {"Key": "file2.txt", "Size": 200, "LastModified": "2024-01-02T00:00:00Z"}
        ],
        "IsTruncated": False,
        "MaxKeys": 1000,
        "Prefix": ""
    })
    return adapter


@pytest.fixture
def s3_handler(mock_storage_adapter):
    """Create S3 handler with mock adapter."""
    return S3Handler(mock_storage_adapter)


@pytest.mark.asyncio
async def test_handle_put_object(s3_handler, mock_storage_adapter):
    """Test PUT object request."""
    # Create mock request
    request = Mock(spec=Request)
    request.method = "PUT"
    request.body = AsyncMock(return_value=b"test data")
    request.headers = {"Content-Type": "text/plain"}
    
    # Handle request
    response = await s3_handler._handle_put_object(request, "mybucket", "test.txt")
    
    # Verify
    assert response.status_code == 200
    assert "ETag" in response.headers
    mock_storage_adapter.put_object.assert_called_once()


@pytest.mark.asyncio
async def test_handle_get_object(s3_handler, mock_storage_adapter):
    """Test GET object request."""
    response = await s3_handler._handle_get_object("mybucket", "test.txt")
    
    assert response.status_code == 200
    assert response.body == b"test data"
    assert response.headers["Content-Type"] == "text/plain"
    mock_storage_adapter.get_object.assert_called_once_with("mybucket", "test.txt")


@pytest.mark.asyncio
async def test_handle_delete_object(s3_handler, mock_storage_adapter):
    """Test DELETE object request."""
    response = await s3_handler._handle_delete_object("mybucket", "test.txt")
    
    assert response.status_code == 204
    mock_storage_adapter.delete_object.assert_called_once_with("mybucket", "test.txt")


@pytest.mark.asyncio
async def test_handle_head_object(s3_handler, mock_storage_adapter):
    """Test HEAD object request."""
    response = await s3_handler._handle_head_object("mybucket", "test.txt")
    
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/plain"
    assert response.headers["Content-Length"] == "9"
    mock_storage_adapter.head_object.assert_called_once_with("mybucket", "test.txt")


@pytest.mark.asyncio
async def test_handle_list_objects(s3_handler, mock_storage_adapter):
    """Test LIST objects request."""
    # Create mock request
    request = Mock(spec=Request)
    request.query_params = {"prefix": "test/", "max-keys": "100"}
    
    response = await s3_handler._handle_list_objects(request, "mybucket")
    
    assert response.status_code == 200
    assert "ListBucketResult" in response.body.decode()
    mock_storage_adapter.list_objects.assert_called_once_with("mybucket", "test/", 100)


@pytest.mark.asyncio
async def test_handle_get_object_not_found(s3_handler, mock_storage_adapter):
    """Test GET object when file not found."""
    mock_storage_adapter.get_object.side_effect = Exception("not found")
    
    with pytest.raises(Exception):
        await s3_handler._handle_get_object("mybucket", "nonexistent.txt")


def test_build_list_objects_xml(s3_handler):
    """Test XML generation for list objects."""
    result = {
        "Contents": [
            {"Key": "file1.txt", "Size": 100},
            {"Key": "file2.txt", "Size": 200}
        ],
        "IsTruncated": False,
        "MaxKeys": 1000,
        "Prefix": "test/"
    }
    
    xml = s3_handler._build_list_objects_xml("mybucket", result)
    
    assert "ListBucketResult" in xml
    assert "mybucket" in xml
    assert "file1.txt" in xml
    assert "file2.txt" in xml
    assert "test/" in xml


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
