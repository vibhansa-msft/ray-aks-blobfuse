"""S3 request handler for the proxy."""
import logging
from typing import Optional, BinaryIO
from fastapi import Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import xml.etree.ElementTree as ET
from datetime import datetime
import io

logger = logging.getLogger(__name__)


class S3Handler:
    """Handler for S3-compatible API requests."""
    
    def __init__(self, storage_adapter):
        """Initialize with a storage adapter (Blob or Files)."""
        self.storage_adapter = storage_adapter
    
    async def handle_request(self, request: Request, bucket: str, key: Optional[str] = None) -> Response:
        """Route S3 request to appropriate handler based on method and path."""
        method = request.method
        
        try:
            if method == "PUT" and key:
                return await self._handle_put_object(request, bucket, key)
            elif method == "GET" and key:
                return await self._handle_get_object(bucket, key)
            elif method == "GET" and not key:
                return await self._handle_list_objects(request, bucket)
            elif method == "DELETE" and key:
                return await self._handle_delete_object(bucket, key)
            elif method == "HEAD" and key:
                return await self._handle_head_object(bucket, key)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported operation: {method} on bucket={bucket}, key={key}")
                
        except Exception as e:
            logger.error(f"Error handling S3 request: {str(e)}")
            return self._create_error_response(str(e))
    
    async def _handle_put_object(self, request: Request, bucket: str, key: str) -> Response:
        """Handle S3 PUT object request."""
        try:
            # Read request body
            body = await request.body()
            data = io.BytesIO(body)
            
            # Get content type
            content_type = request.headers.get("Content-Type", "application/octet-stream")
            
            # Upload to storage
            result = self.storage_adapter.put_object(bucket, key, data, content_type)
            
            # Return success response
            return Response(
                status_code=200,
                headers={
                    "ETag": result.get("ETag", ""),
                    "x-amz-request-id": "proxy-request"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in PUT object: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_get_object(self, bucket: str, key: str) -> Response:
        """Handle S3 GET object request."""
        try:
            # Download from storage
            data, metadata = self.storage_adapter.get_object(bucket, key)
            
            # Return object data
            return Response(
                content=data,
                status_code=200,
                headers={
                    "Content-Type": metadata.get("ContentType", "application/octet-stream"),
                    "Content-Length": str(metadata.get("ContentLength", len(data))),
                    "ETag": metadata.get("ETag", ""),
                    "Last-Modified": self._format_http_date(metadata.get("LastModified")),
                    "x-amz-request-id": "proxy-request"
                }
            )
            
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail="NoSuchKey")
            logger.error(f"Error in GET object: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_delete_object(self, bucket: str, key: str) -> Response:
        """Handle S3 DELETE object request."""
        try:
            # Delete from storage
            self.storage_adapter.delete_object(bucket, key)
            
            # Return success response
            return Response(
                status_code=204,
                headers={"x-amz-request-id": "proxy-request"}
            )
            
        except Exception as e:
            logger.error(f"Error in DELETE object: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_head_object(self, bucket: str, key: str) -> Response:
        """Handle S3 HEAD object request."""
        try:
            # Get metadata from storage
            metadata = self.storage_adapter.head_object(bucket, key)
            
            # Return headers only
            return Response(
                status_code=200,
                headers={
                    "Content-Type": metadata.get("ContentType", "application/octet-stream"),
                    "Content-Length": str(metadata.get("ContentLength", 0)),
                    "ETag": metadata.get("ETag", ""),
                    "Last-Modified": self._format_http_date(metadata.get("LastModified")),
                    "x-amz-request-id": "proxy-request"
                }
            )
            
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail="NoSuchKey")
            logger.error(f"Error in HEAD object: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_list_objects(self, request: Request, bucket: str) -> Response:
        """Handle S3 LIST objects request."""
        try:
            # Parse query parameters
            prefix = request.query_params.get("prefix", "")
            max_keys = int(request.query_params.get("max-keys", "1000"))
            
            # List objects from storage
            result = self.storage_adapter.list_objects(bucket, prefix, max_keys)
            
            # Build S3 XML response
            xml_response = self._build_list_objects_xml(bucket, result)
            
            return Response(
                content=xml_response,
                status_code=200,
                media_type="application/xml",
                headers={"x-amz-request-id": "proxy-request"}
            )
            
        except Exception as e:
            logger.error(f"Error in LIST objects: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _build_list_objects_xml(self, bucket: str, result: dict) -> str:
        """Build S3 ListBucketResult XML response."""
        root = ET.Element("ListBucketResult")
        root.set("xmlns", "http://s3.amazonaws.com/doc/2006-03-01/")
        
        ET.SubElement(root, "Name").text = bucket
        ET.SubElement(root, "Prefix").text = result.get("Prefix", "")
        ET.SubElement(root, "MaxKeys").text = str(result.get("MaxKeys", 1000))
        ET.SubElement(root, "IsTruncated").text = str(result.get("IsTruncated", False)).lower()
        
        for obj in result.get("Contents", []):
            contents_elem = ET.SubElement(root, "Contents")
            ET.SubElement(contents_elem, "Key").text = obj.get("Key", "")
            ET.SubElement(contents_elem, "Size").text = str(obj.get("Size", 0))
            
            if obj.get("LastModified"):
                ET.SubElement(contents_elem, "LastModified").text = self._format_iso_date(obj["LastModified"])
            
            if obj.get("ETag"):
                ET.SubElement(contents_elem, "ETag").text = obj["ETag"]
        
        return ET.tostring(root, encoding="unicode")
    
    def _format_http_date(self, dt) -> str:
        """Format datetime for HTTP headers."""
        if dt is None:
            return ""
        if isinstance(dt, str):
            return dt
        # Convert to HTTP date format: "Wed, 21 Oct 2015 07:28:00 GMT"
        return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    
    def _format_iso_date(self, dt) -> str:
        """Format datetime for ISO 8601 (S3 XML)."""
        if dt is None:
            return ""
        if isinstance(dt, str):
            return dt
        # Convert to ISO format: "2015-10-21T07:28:00.000Z"
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    def _create_error_response(self, message: str, code: str = "InternalError") -> Response:
        """Create S3-style error XML response."""
        root = ET.Element("Error")
        ET.SubElement(root, "Code").text = code
        ET.SubElement(root, "Message").text = message
        ET.SubElement(root, "RequestId").text = "proxy-request"
        
        xml_content = ET.tostring(root, encoding="unicode")
        
        return Response(
            content=xml_content,
            status_code=500,
            media_type="application/xml"
        )
