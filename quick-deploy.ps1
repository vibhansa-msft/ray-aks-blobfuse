<#
Quick Deploy Script for Ray HPO
This script quickly redeploys StorageClasses, PVCs, and RayJob without infrastructure checks
Use this when infrastructure (AKS, Storage Account, KubeRay) is already set up

Usage: powershell -ExecutionPolicy Bypass -File .\quick-deploy.ps1
#>

# ========== SETUP AND INITIALIZATION ==========

# Import utility functions and load parameters
. "$PSScriptRoot\utils.ps1"
Load-EnvParameters

# ========== CLEANUP RUNNING JOBS ==========

Write-Host "Cleaning up any existing RayJobs..."
try {
    kubectl delete rayjob --all 2>$null | Out-Null
    Write-Host "Existing RayJobs deleted." -ForegroundColor Yellow
} catch {
    Write-Host "No existing RayJobs found." -ForegroundColor Gray
}

# ========== STORAGE CLASS AND PVC SETUP ==========
Write-Host "Setting up StorageClasses and PVCs..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

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
$scYaml = $scYaml -replace "__STORAGE_ACCOUNT__", $SA -replace "__CONTAINER__", $CONTAINER -replace "__RESOURCE_GROUP__", $RG
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

# ========== APPLICATION CODE AND CONFIG ==========

Write-Host ""
Write-Host "Preparing application code..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Create ConfigMap for application code
Write-Host "Creating ConfigMap for application code..."
try {
    kubectl delete configmap hpo-app-code 2>$null | Out-Null
} catch {}

kubectl create configmap hpo-app-code --from-file=app/ | Out-Null
Write-Host "ConfigMap 'hpo-app-code' created successfully." -ForegroundColor Green

# ========== RAY JOB DEPLOYMENT ==========

Write-Host ""
Write-Host "Deploying RayJob..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Determine which RayJob to deploy based on GPU flag
$job = if ($GPU -eq "true" -or $GPU -eq $true) { "hpo-job-gpu" } else { "hpo-job-cpu" }

# Prepare RayJob YAML
$rayjobFile = if ($GPU -eq "true" -or $GPU -eq $true) { "k8s/rayjob-gpu.yaml" } else { "k8s/rayjob-cpu.yaml" }
$rayjobYaml = Get-Content $rayjobFile -Raw

# Substitute all placeholder tokens with actual paths (from .env.example)
$rayjobYaml = $rayjobYaml -replace "__DATA_DIR__", $DATA_DIR -replace "__CHECKPOINT_DIR__", $CHECKPOINT_DIR -replace "__APP_DIR__", $APP_DIR -replace "__CACHE_DIR__", $CACHE_DIR -replace "__NUM_WORKERS__", $NUM_WORKERS -replace "__WORKER_REPLICAS__", $WORKER_REPLICAS

# Submit RayJob
Write-Host "Submitting RayJob '$job'..."
$rayjobYaml | kubectl apply -f - | Out-Null
Write-Host "RayJob '$job' submitted successfully." -ForegroundColor Green

# Deploy Ray Dashboard Service
Write-Host "Deploying Ray Dashboard service..."
kubectl apply -f k8s/ray-dashboard-service.yaml | Out-Null
Write-Host "Ray Dashboard service deployed successfully." -ForegroundColor Green

# Dashboard service is deployed above - use open-dashboard.ps1 to access it

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Quick Deploy Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Wait for Ray head pod to be ready, then open dashboard
Write-Host "Waiting for Ray head pod to be ready..." -ForegroundColor Cyan
$headReady = $false
$waitTime = 0
$maxWait = 120

while (-not $headReady -and $waitTime -lt $maxWait) {
    $headPod = kubectl get pod -l "ray.io/node-type=head" -o jsonpath="{.items[0].metadata.name}" 2>$null
    if ($headPod) {
        $podStatus = kubectl get pod $headPod -o jsonpath="{.status.phase}" 2>$null
        if ($podStatus -eq "Running") {
            $headReady = $true
            Write-Host "Ray head pod is ready. Opening dashboard..." -ForegroundColor Green
            & "$PSScriptRoot\open-dashboard.ps1"
            break
        }
    }
    
    if (-not $headReady) {
        Start-Sleep -Seconds 10
        $waitTime += 10
        Write-Host "  Waiting... ($waitTime/$maxWait seconds)" -ForegroundColor Gray
    }
}

if (-not $headReady) {
    Write-Host "Ray head pod not ready after $maxWait seconds. Open dashboard manually:" -ForegroundColor Yellow
    Write-Host "  .\open-dashboard.ps1" -ForegroundColor Gray
}

# Start monitoring the Ray job in background
& "$PSScriptRoot\monitor-jobs.ps1" -JobName $job

