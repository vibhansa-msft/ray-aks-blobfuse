"""S3 to Azure Storage Proxy Server."""
import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

from .config import ProxyConfig
from .azure_blob_adapter import AzureBlobAdapter
from .azure_files_adapter import AzureFilesAdapter
from .s3_handler import S3Handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="S3 to Azure Storage Proxy")

# Global configuration and handler
config = None
s3_handler = None


@app.on_event("startup")
async def startup_event():
    """Initialize proxy on startup."""
    global config, s3_handler
    
    try:
        # Load configuration
        config = ProxyConfig()
        config.validate()
        
        logger.info(f"Starting S3 to Azure Storage Proxy")
        logger.info(f"Storage Backend: {config.storage_backend.value}")
        logger.info(f"Storage Account: {config.storage_account_name}")
        
        # Initialize appropriate storage adapter
        if config.is_blob_backend():
            logger.info("Using Azure Blob Storage adapter")
            storage_adapter = AzureBlobAdapter(config)
        else:
            logger.info("Using Azure Files adapter")
            storage_adapter = AzureFilesAdapter(config)
        
        # Initialize S3 handler
        s3_handler = S3Handler(storage_adapter)
        
        logger.info(f"Proxy server started successfully on {config.proxy_host}:{config.proxy_port}")
        
    except Exception as e:
        logger.error(f"Failed to start proxy server: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "backend": config.storage_backend.value if config else "unknown"
    }


@app.api_route("/{bucket}/{key:path}", methods=["GET", "PUT", "DELETE", "HEAD"])
async def handle_object_request(request: Request, bucket: str, key: str):
    """Handle S3 object requests (GET, PUT, DELETE, HEAD)."""
    logger.info(f"Received {request.method} request for bucket={bucket}, key={key}")
    return await s3_handler.handle_request(request, bucket, key)


@app.api_route("/{bucket}", methods=["GET"])
async def handle_bucket_request(request: Request, bucket: str):
    """Handle S3 bucket requests (LIST)."""
    logger.info(f"Received {request.method} request for bucket={bucket}")
    return await s3_handler.handle_request(request, bucket, None)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )


def main():
    """Run the proxy server."""
    # Load config to get host and port
    temp_config = ProxyConfig()
    
    uvicorn.run(
        app,
        host=temp_config.proxy_host,
        port=temp_config.proxy_port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
