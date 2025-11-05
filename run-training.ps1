# Distributed Model Training Job Deployment
# Deploys Ray job for distributed model training on parquet datasets

# Import utility functions and load parameters
. "$PSScriptRoot\utils.ps1"
Load-EnvParameters

Write-Host ""
Write-Host "Ray Distributed Model Training Job Deployment" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Clean up any existing training jobs
Write-Host "Cleaning up existing training jobs..."
kubectl delete rayjob --all --ignore-not-found=true 2>$null | Out-Null

# Ensure ConfigMap has the latest code
Write-Host "Updating application code ConfigMap..."
kubectl delete configmap hpo-app-code --ignore-not-found=true 2>$null | Out-Null
kubectl create configmap hpo-app-code --from-file=app/ --from-file=data/ | Out-Null
Write-Host "ConfigMap updated with latest code." -ForegroundColor Green

# Deploy the training RayJob
Write-Host "Deploying distributed training RayJob..."

# Read training job YAML
$trainingYaml = Get-Content "k8s/rayjob-training.yaml" -Raw

# Substitute placeholders with values from environment variables
$trainingYaml = $trainingYaml -replace "__DATA_DIR__", $DATA_DIR `
    -replace "__CHECKPOINT_DIR__", $CHECKPOINT_DIR `
    -replace "__APP_DIR__", $APP_DIR `
    -replace "__CACHE_DIR__", $CACHE_DIR `
    -replace "__CHECKPOINT_DIR__", $CHECKPOINT_CACHE `
    -replace "__WORKER_REPLICAS__", $WORKER_REPLICAS `
    -replace "__NUM_WORKERS__", $NUM_WORKERS `
    -replace "__FILES_PER_WORKER__", 3 `
    -replace "__BATCH_SIZE__", 64 `
    -replace "__DATASET__", "openwebtext" `
    -replace "__HF_TOKEN__", $HF_TOKEN


# Apply the processed YAML
$trainingYaml | kubectl apply -f - | Out-Null
Write-Host "Model training job submitted successfully." -ForegroundColor Green

# Deploy dashboard service 
Write-Host "Ensuring Ray Dashboard service is available..."
kubectl apply -f k8s/ray-dashboard-service.yaml | Out-Null

Write-Host ""
Write-Host "Model Training Job Deployed!" -ForegroundColor Green
Write-Host "============================" -ForegroundColor Green
Write-Host ""

# ========== MONITORING DEPLOYMENT ==========

Write-Host "Setting up monitoring..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Deploy monitoring stack
Deploy-MonitoringStack | Out-Null

# ========== DASHBOARD ACCESS ==========

# Wait for Ray head pod to be ready
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
            Write-Host "Ray head pod is ready!" -ForegroundColor Green
            break
        }
    }
    
    if (-not $headReady) {
        Start-Sleep -Seconds 10
        $waitTime += 10
        Write-Host "  Waiting... ($waitTime/$maxWait seconds)" -ForegroundColor Gray
    }
}

if ($headReady) {
    Write-Host ""
    Open-AllDashboards
} else {
    Write-Host "Ray head pod not ready after $maxWait seconds. Open dashboards manually:" -ForegroundColor Yellow
    Write-Host "  .\open-dashboard.ps1" -ForegroundColor Gray
}

# Start monitoring
Write-Host ""
Write-Host "Starting job monitoring..." -ForegroundColor Cyan
& "$PSScriptRoot\monitor-jobs.ps1" -JobName "model-training-job"
