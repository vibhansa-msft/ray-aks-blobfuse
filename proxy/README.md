# S3 to Azure Storage Proxy

This proxy service translates S3 API calls to Azure Blob Storage or Azure Files REST APIs, allowing applications that use S3 APIs to seamlessly work with Azure storage backends.

## Features

- **S3 API Compatibility**: Supports common S3 operations (PUT, GET, DELETE, HEAD, LIST)
- **Dual Backend Support**: 
  - Azure Blob Storage (default)
  - Azure Files
- **Easy Configuration**: Switch between backends via environment variables
- **Kubernetes Ready**: Includes deployment manifests for AKS

## Supported Operations

| S3 Operation | Azure Blob Storage | Azure Files | Status |
|--------------|-------------------|-------------|---------|
| PUT Object   | ✅ PutBlob         | ✅ UploadFile | Supported |
| GET Object   | ✅ GetBlob         | ✅ DownloadFile | Supported |
| DELETE Object| ✅ DeleteBlob      | ✅ DeleteFile | Supported |
| HEAD Object  | ✅ GetBlobProperties | ✅ GetFileProperties | Supported |
| LIST Objects | ✅ ListBlobs       | ✅ ListFilesAndDirectories | Supported |

## Configuration

The proxy is configured via environment variables:

### Required Variables

- `STORAGE_ACCOUNT_NAME` or `SA`: Azure storage account name
- `STORAGE_ACCOUNT_KEY`: Azure storage account key

### Optional Variables

- `STORAGE_BACKEND`: Choose `blob` (default) or `files`
- `PROXY_HOST`: Proxy server host (default: `0.0.0.0`)
- `PROXY_PORT`: Proxy server port (default: `5000`)
- `CONTAINER`: Default Azure container name (default: `datasets`)
- `FILE_SHARE_NAME`: Azure Files share name when using Files backend (default: `datasets`)
- `BUCKET_MAPPING`: Map S3 buckets to Azure containers/shares (format: `bucket1:container1,bucket2:container2`)

### Example: Using Azure Blob Storage

```bash
export STORAGE_BACKEND=blob
export STORAGE_ACCOUNT_NAME=mystorageaccount
export STORAGE_ACCOUNT_KEY=your-account-key
export CONTAINER=datasets
```

### Example: Using Azure Files

```bash
export STORAGE_BACKEND=files
export STORAGE_ACCOUNT_NAME=mystorageaccount
export STORAGE_ACCOUNT_KEY=your-account-key
export FILE_SHARE_NAME=datasets
```

## Running Locally

### Prerequisites

- Python 3.11+
- Azure Storage Account with either Blob Storage or Files enabled

### Installation

```bash
cd proxy
pip install -r requirements.txt
```

### Run the Proxy

```bash
# Set required environment variables
export STORAGE_ACCOUNT_NAME=mystorageaccount
export STORAGE_ACCOUNT_KEY=your-account-key
export STORAGE_BACKEND=blob  # or "files"

# Start the proxy server
python -m proxy.proxy_server
```

The proxy will start on `http://localhost:5000` by default.

### Testing

You can test the proxy with S3 clients like `boto3`:

```python
import boto3
from botocore.client import Config

# Configure S3 client to use the proxy
s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:5000',
    aws_access_key_id='dummy',  # Not validated by proxy
    aws_secret_access_key='dummy',  # Not validated by proxy
    config=Config(signature_version='s3v4')
)

# Upload a file
s3.put_object(Bucket='mybucket', Key='test.txt', Body=b'Hello World')

# Download a file
response = s3.get_object(Bucket='mybucket', Key='test.txt')
print(response['Body'].read())

# List objects
response = s3.list_objects_v2(Bucket='mybucket', Prefix='')
for obj in response.get('Contents', []):
    print(obj['Key'])

# Delete a file
s3.delete_object(Bucket='mybucket', Key='test.txt')
```

**Full example script**: See [example_usage.py](example_usage.py)

## Kubernetes Deployment

### Prerequisites

- AKS cluster with kubectl configured
- Azure Storage Account
- Storage account key

### Quick Deploy (Recommended)

Use the deployment helper script:

```bash
cd k8s
./deploy-proxy.sh
```

The script will:
- Prompt for Azure storage credentials
- Create necessary secrets and ConfigMaps
- Deploy the proxy service
- Display connection information

### Manual Deploy the Proxy

1. **Create the secret with your storage credentials:**

```bash
kubectl create secret generic azure-storage-secret \
  --from-literal=storage-account-name=mystorageaccount \
  --from-literal=storage-account-key=your-account-key
```

2. **Create ConfigMap with proxy code:**

```bash
kubectl create configmap proxy-code \
  --from-file=proxy/
```

3. **Deploy the proxy:**

```bash
# For Azure Blob Storage (default)
kubectl apply -f k8s/s3-proxy-deployment.yaml

# For Azure Files, edit the deployment and change STORAGE_BACKEND to "files"
```

4. **Verify the deployment:**

```bash
kubectl get pods -l app=s3-azure-proxy
kubectl get svc s3-azure-proxy
```

5. **Configure your applications to use the proxy:**

Set `S3_ENDPOINT_URL=http://s3-azure-proxy:5000` in your application pods.

### Switching Between Blob and Files

Edit the deployment and change the `STORAGE_BACKEND` environment variable:

```bash
kubectl edit deployment s3-azure-proxy
```

Change:
```yaml
- name: STORAGE_BACKEND
  value: "files"  # Change from "blob" to "files"
```

Then restart the deployment:
```bash
kubectl rollout restart deployment s3-azure-proxy
```

## Architecture

```
┌─────────────────┐
│   S3 Client     │
│  (boto3, etc.)  │
└────────┬────────┘
         │ S3 API requests
         ▼
┌─────────────────┐
│  S3 Handler     │
│  (FastAPI)      │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────────┐
│  Blob   │ │    Files     │
│ Adapter │ │   Adapter    │
└────┬────┘ └──────┬───────┘
     │             │
     ▼             ▼
┌─────────────────────────┐
│   Azure Storage         │
│  (Blob or Files)        │
└─────────────────────────┘
```

## Troubleshooting

### Proxy fails to start

- Check that `STORAGE_ACCOUNT_NAME` and `STORAGE_ACCOUNT_KEY` are set correctly
- Verify network connectivity to Azure Storage
- Check logs: `kubectl logs -l app=s3-azure-proxy`

### 404 Errors

- Ensure the Azure container/share exists
- Check bucket-to-container mapping configuration
- Verify object keys match Azure blob/file paths

### Authentication Errors

- Verify storage account key is correct
- Check that the storage account has Blob or Files enabled
- Ensure network access is allowed (firewall rules)

## Performance Considerations

- The proxy adds network latency between client and Azure Storage
- For high-throughput scenarios, consider:
  - Running multiple proxy replicas
  - Using horizontal pod autoscaling
  - Placing proxy pods close to client applications (same node/zone)

## Security

- Store Azure Storage credentials in Kubernetes Secrets
- Use RBAC to restrict access to secrets
- Consider using Managed Identity instead of account keys (future enhancement)
- Run proxy in isolated namespace with network policies

## Limitations

- Currently supports basic S3 operations (PUT, GET, DELETE, HEAD, LIST)
- Does not support multipart uploads (future enhancement)
- Does not support S3 presigned URLs (future enhancement)
- No authentication/authorization on proxy endpoints (add API gateway if needed)

## Future Enhancements

- [ ] Azure Managed Identity support
- [ ] Multipart upload support
- [ ] Presigned URL support
- [ ] Caching layer
- [ ] Metrics and monitoring
- [ ] Rate limiting
- [ ] Authentication on proxy endpoints
