<#
.SYNOPSIS
    Setup Azure Kubernetes Service (AKS) cluster with CPU or GPU configuration
.DESCRIPTION
    This script creates and configures an AKS cluster based on the GPU flag.
    It handles:
    - AKS cluster creation (CPU or GPU infrastructure)
    - Azure Blob CSI driver enablement
    - NVIDIA device plugin installation (GPU only)
    - Credentials configuration
.PARAMETER ResourceGroup
    Azure Resource Group name
.PARAMETER ClusterName
    AKS cluster name
.PARAMETER Location
    Azure location
.PARAMETER EnableGpu
    Enable GPU support (boolean)
.PARAMETER NodeCount
    Number of nodes
.PARAMETER VmType
    VM size
.PARAMETER StorageAccountName
    Storage account name for role assignment
.PARAMETER StorageAccountResourceGroup
    Resource group of the storage account
#>

param(
    [string]$ResourceGroup,
    [string]$ClusterName,
    [string]$Location,
    [bool]$EnableGpu = $false,
    [int]$NodeCount = 2,
    [string]$VmType = "Standard_D4_v2",
    [string]$StorageAccountName = "",
    [string]$StorageAccountResourceGroup = ""
)

Write-Host "========== AKS CLUSTER SETUP ==========" -ForegroundColor Cyan

# ===== Check if AKS cluster already exists =====
Write-Host "Checking if AKS cluster $ClusterName exists..."
$aksExists = $false
try {
    $aksResult = az aks show -g $ResourceGroup -n $ClusterName --query "name" -o tsv 2>$null
    if ($aksResult) {
        $aksExists = $true
        Write-Host "AKS cluster $ClusterName already exists. Skipping creation." -ForegroundColor Yellow
    }
} catch {
    # AKS cluster doesn't exist, will create it
}

if (-not $aksExists) {
    # ===== Create AKS Cluster with appropriate infrastructure =====
    Write-Host "Creating AKS cluster $ClusterName with ($NodeCount x $VmType)..." -ForegroundColor Green
    az aks create `
        -g $ResourceGroup `
        -n $ClusterName `
        --node-count $NodeCount `
        --node-vm-size $VmType `
        --enable-managed-identity `
        --enable-blob-driver | Out-Null
        
    # Wait for cluster to be fully ready
    Write-Host "Waiting for AKS cluster to be ready (this may take several minutes)..."
    $clusterReady = $false
    $retryCount = 0
    $maxRetries = 60  # 30 minutes with 30-second intervals
    
    while (-not $clusterReady -and $retryCount -lt $maxRetries) {
        $provisioningState = az aks show -g $ResourceGroup -n $ClusterName --query "provisioningState" -o tsv 2>$null
        if ($provisioningState -eq "Succeeded") {
            $clusterReady = $true
            Write-Host "AKS cluster is ready!" -ForegroundColor Green
        } else {
            $retryCount++
            Write-Host "Cluster status: $provisioningState (attempt $retryCount/$maxRetries)..."
            Start-Sleep -Seconds 30
        }
    }
    
    if (-not $clusterReady) {
        Write-Host "ERROR: AKS cluster failed to reach ready state after 30 minutes." -ForegroundColor Red
        exit 1
    }
}

# ===== Get AKS Credentials =====
Write-Host "Retrieving kubeconfig credentials..."
try {
    az aks get-credentials -g $ResourceGroup -n $ClusterName --overwrite-existing 2>&1 | Out-Null
    Write-Host "Kubeconfig retrieved successfully." -ForegroundColor Green
} catch {
    Write-Host "Warning: Failed to retrieve kubeconfig. Continuing..." -ForegroundColor Yellow
}

# ===== GPU-specific Configuration =====
if ($EnableGpu) {
    # Check if NVIDIA device plugin is already deployed
    Write-Host "Checking if NVIDIA device plugin is already deployed..."
    $nvidiaDeploy = $false
    try {
        $nvidiaResult = kubectl get deployment -n kube-system nvidia-device-plugin-daemonset 2>$null
        if ($nvidiaResult) {
            $nvidiaDeploy = $true
            Write-Host "NVIDIA device plugin already deployed. Skipping installation." -ForegroundColor Yellow
        }
    } catch {
        # NVIDIA plugin not deployed
    }
    
    if (-not $nvidiaDeploy) {
        Write-Host "Installing NVIDIA device plugin..."
        kubectl apply -f k8s/nvidia-device-plugin.yaml | Out-Null
    }
}

# ===== Create StorageClass and PVC for Blobfuse2 =====
Write-Host "Deleting existing StorageClasses (if any) to apply new configuration..."
try {
    kubectl delete storageclass azureblob-fuse2 2>$null | Out-Null
    Write-Host "Previous StorageClass 'azureblob-fuse2' deleted." -ForegroundColor Yellow
} catch {
    # StorageClass doesn't exist, proceed
}

try {
    kubectl delete storageclass blobfuse-training-data 2>$null | Out-Null
    Write-Host "Previous StorageClass 'blobfuse-training-data' deleted." -ForegroundColor Yellow
} catch {
    # StorageClass doesn't exist, proceed
}

try {
    kubectl delete storageclass blobfuse-checkpoints 2>$null | Out-Null
    Write-Host "Previous StorageClass 'blobfuse-checkpoints' deleted." -ForegroundColor Yellow
} catch {
    # StorageClass doesn't exist, proceed
}

Write-Host "Creating StorageClasses for Blobfuse2..."
$scYaml = Get-Content k8s/storageclass-blobfuse2.yaml -Raw
$scYaml = $scYaml -replace "__STORAGE_ACCOUNT__", $StorageAccountName -replace "__CONTAINER__", "dataset" -replace "__RESOURCE_GROUP__", $StorageAccountResourceGroup
$scYaml | kubectl apply -f - | Out-Null

Write-Host "Deleting existing PVCs (if any) to apply new configuration..."
try {
    kubectl delete pvc blob-pvc 2>$null | Out-Null
    Write-Host "Previous PVC 'blob-pvc' deleted." -ForegroundColor Yellow
} catch {
    # PVC doesn't exist, proceed
}

try {
    kubectl delete pvc blob-pvc-checkpoint 2>$null | Out-Null
    Write-Host "Previous PVC 'blob-pvc-checkpoint' deleted." -ForegroundColor Yellow
} catch {
    # PVC doesn't exist, proceed
}

try {
    kubectl delete pvc blob-pvc-dataset 2>$null | Out-Null
    Write-Host "Previous PVC 'blob-pvc-dataset' deleted." -ForegroundColor Yellow
} catch {
    # PVC doesn't exist, proceed
}

Write-Host "Creating PVC for Blobfuse2 (checkpoints with read-write access)..."
$pvcCheckpointsCreated = $false
$retryCount = 0
$maxRetries = 10

while (-not $pvcCheckpointsCreated -and $retryCount -lt $maxRetries) {
    try {
        kubectl apply -f k8s/pvc-checkpoint.yaml 2>&1 | Out-Null
        $pvcCheckpointsCreated = $true
        Write-Host "Checkpoints PVC created successfully." -ForegroundColor Green
    } catch {
        $retryCount++
        if ($retryCount -lt $maxRetries) {
            Write-Host "PVC creation attempt $retryCount failed. Retrying in 10 seconds..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10
        } else {
            Write-Host "ERROR: Failed to create checkpoints PVC after $maxRetries attempts." -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host "Creating PVC for Blobfuse2 (dataset with read-only access)..."
$pvcDatasetCreated = $false
$retryCount = 0

while (-not $pvcDatasetCreated -and $retryCount -lt $maxRetries) {
    try {
        kubectl apply -f k8s/pvc-training-data.yaml 2>&1 | Out-Null
        $pvcDatasetCreated = $true
        Write-Host "Dataset PVC created successfully." -ForegroundColor Green
    } catch {
        $retryCount++
        if ($retryCount -lt $maxRetries) {
            Write-Host "Dataset PVC creation attempt $retryCount failed. Retrying in 10 seconds..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10
        } else {
            Write-Host "ERROR: Failed to create dataset PVC after $maxRetries attempts." -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host "AKS cluster setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
