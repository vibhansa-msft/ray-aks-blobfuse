# Data Preparation Job Deployment
# Deploys Ray job for various dataset preparation tasks

# Import utility functions and load parameters
. "$PSScriptRoot\utils.ps1"
Load-EnvParameters

Write-Host ""
Write-Host "Ray Data Preparation Job Deployment" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Clean up any existing data prep jobs
Write-Host "Cleaning up existing data preparation jobs..."
kubectl delete rayjob data-prep-job --ignore-not-found=true 2>$null | Out-Null

# Ensure ConfigMap has the latest code (including new data scripts)
Write-Host "Updating application code ConfigMap..."
kubectl delete configmap hpo-app-code --ignore-not-found=true 2>$null | Out-Null
kubectl create configmap hpo-app-code --from-file=app/ --from-file=data/ | Out-Null
Write-Host "ConfigMap updated with latest code." -ForegroundColor Green

# Deploy the data preparation RayJob
Write-Host "Deploying data preparation RayJob..."

# Prepare data preparation job YAML with path substitution
$dataprepYaml = Get-Content "k8s/rayjob-dataprep.yaml" -Raw

# Substitute placeholder tokens with actual paths (from .env.example)
$dataprepYaml = $dataprepYaml -replace "__DATA_DIR__", $DATA_DIR -replace "__CHECKPOINT_DIR__", $CHECKPOINT_DIR -replace "__APP_DIR__", $APP_DIR -replace "__CACHE_DIR__", $CACHE_DIR -replace "__WORKER_REPLICAS__", $WORKER_REPLICAS -replace "__NUM_WORKERS__", $NUM_WORKERS

# Apply the processed YAML
$dataprepYaml | kubectl apply -f - | Out-Null
Write-Host "Data preparation job submitted successfully." -ForegroundColor Green

# Deploy dashboard service 
Write-Host "Ensuring Ray Dashboard service is available..."
kubectl apply -f k8s/ray-dashboard-service.yaml | Out-Null

Write-Host ""
Write-Host "Data Preparation Job Deployed!" -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green
Write-Host ""

# ========== MONITORING DEPLOYMENT ==========

Write-Host "Setting up monitoring..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Deploy monitoring stack (Grafana + Prometheus)
Deploy-MonitoringStack | Out-Null

# ========== DASHBOARD ACCESS ==========

# Wait for Ray head pod to be ready, then open both dashboards
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
& "$PSScriptRoot\monitor-jobs.ps1" -JobName "data-prep-job"