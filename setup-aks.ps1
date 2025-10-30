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
.PARAMETER GpuNodeCount
    Number of GPU nodes (if GPU enabled)
.PARAMETER VmType
    GPU VM size (if GPU enabled)
.PARAMETER CpuNodeCount
    Number of CPU nodes
.PARAMETER VmType
    CPU VM size
#>

param(
    [string]$ResourceGroup,
    [string]$ClusterName,
    [string]$Location,
    [bool]$EnableGpu = $false,
    [int]$NodeCount = 2,
    [string]$VmType = "Standard_D4_v2"
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
        
    # Wait for cluster to be ready
    Write-Host "Waiting for AKS cluster to be ready..."
    Start-Sleep -Seconds 30
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
Write-Host "Deleting existing StorageClass 'azureblob-fuse2' (if any) to apply new configuration..."
try {
    kubectl delete storageclass azureblob-fuse2 2>$null | Out-Null
    Write-Host "Previous StorageClass deleted." -ForegroundColor Yellow
} catch {
    # StorageClass doesn't exist, proceed
}

Write-Host "Creating StorageClass for Blobfuse2..."
$scYaml = Get-Content k8s/storageclass-blobfuse2.yaml -Raw
$scYaml = $scYaml -replace "__STORAGE_ACCOUNT__", $SA -replace "__CONTAINER__", $Container -replace "__RESOURCE_GROUP__", $RG
$scYaml | kubectl apply -f - | Out-Null

Write-Host "Deleting existing PVC 'blob-pvc' (if any) to apply new configuration..."
try {
    kubectl delete pvc blob-pvc 2>$null | Out-Null
    Write-Host "Previous PVC deleted." -ForegroundColor Yellow
} catch {
    # PVC doesn't exist, proceed
}

Write-Host "Creating PVC for Blobfuse2..."
kubectl apply -f k8s/pvc-blob.yaml | Out-Null

Write-Host "AKS cluster setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan


