#!/bin/bash
# Deploy S3 to Azure Storage Proxy to Kubernetes

set -e

echo "=== S3 to Azure Storage Proxy Deployment ==="
echo ""

# Check if required tools are installed
command -v kubectl >/dev/null 2>&1 || { echo "Error: kubectl is required but not installed. Aborting." >&2; exit 1; }

# Get configuration from environment or prompt
read -p "Enter Azure Storage Account Name [${SA:-}]: " STORAGE_ACCOUNT
STORAGE_ACCOUNT=${STORAGE_ACCOUNT:-$SA}

if [ -z "$STORAGE_ACCOUNT" ]; then
    echo "Error: Storage account name is required"
    exit 1
fi

read -sp "Enter Azure Storage Account Key [${STORAGE_ACCOUNT_KEY:-}]: " STORAGE_KEY
echo ""
STORAGE_KEY=${STORAGE_KEY:-$STORAGE_ACCOUNT_KEY}

if [ -z "$STORAGE_KEY" ]; then
    echo "Error: Storage account key is required"
    exit 1
fi

read -p "Enter Storage Backend (blob/files) [blob]: " BACKEND
BACKEND=${BACKEND:-blob}

echo ""
echo "Configuration:"
echo "  Storage Account: $STORAGE_ACCOUNT"
echo "  Backend: $BACKEND"
echo ""

# Create namespace (if it doesn't exist)
kubectl create namespace s3-proxy --dry-run=client -o yaml | kubectl apply -f -

# Create secret
echo "Creating Azure storage secret..."
kubectl create secret generic azure-storage-secret \
  --from-literal=storage-account-name="$STORAGE_ACCOUNT" \
  --from-literal=storage-account-key="$STORAGE_KEY" \
  --namespace=s3-proxy \
  --dry-run=client -o yaml | kubectl apply -f -

# Create ConfigMap with proxy code
echo "Creating ConfigMap with proxy code..."
kubectl create configmap proxy-code \
  --from-file=../proxy/ \
  --namespace=s3-proxy \
  --dry-run=client -o yaml | kubectl apply -f -

# Update deployment with backend configuration
echo "Deploying proxy service..."
cat s3-proxy-deployment.yaml | \
  sed "s/value: \"blob\"/value: \"$BACKEND\"/" | \
  kubectl apply -n s3-proxy -f -

# Wait for deployment
echo "Waiting for proxy to be ready..."
kubectl wait --for=condition=available --timeout=120s \
  deployment/s3-azure-proxy -n s3-proxy

# Get service endpoint
SERVICE_IP=$(kubectl get svc s3-azure-proxy -n s3-proxy -o jsonpath='{.spec.clusterIP}')

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Proxy service is running at: http://${SERVICE_IP}:5000"
echo ""
echo "To use the proxy in your applications, set:"
echo "  S3_ENDPOINT_URL=http://s3-azure-proxy.s3-proxy.svc.cluster.local:5000"
echo ""
echo "To test the proxy:"
echo "  kubectl port-forward -n s3-proxy svc/s3-azure-proxy 5000:5000"
echo "  python proxy/example_usage.py"
echo ""
echo "To view logs:"
echo "  kubectl logs -n s3-proxy -l app=s3-azure-proxy -f"
echo ""
