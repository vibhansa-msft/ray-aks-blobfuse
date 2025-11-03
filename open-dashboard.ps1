# Open Both Ray and Grafana Dashboards
# Opens both Ray dashboard and Grafana monitoring dashboards

# Import utility functions
. "$PSScriptRoot\utils.ps1"

Write-Host "[DASHBOARD] Opening Ray and Grafana Dashboards" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Ray head pod exists
$headPod = kubectl get pod -l "ray.io/node-type=head" -o jsonpath="{.items[0].metadata.name}" 2>$null

if (-not $headPod) {
    Write-Host "[ERROR] No Ray head pod found. Run quick-deploy.ps1 or run-dataprep.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "[SUCCESS] Ray cluster detected" -ForegroundColor Green

# Check if monitoring is deployed, if not, deploy it
$monitoringPods = kubectl get pods -n monitoring --no-headers 2>$null
if (-not $monitoringPods -or ($monitoringPods | Where-Object { $_ -match "Running" }).Count -lt 2) {
    Write-Host "[DEPLOY] Deploying monitoring stack..." -ForegroundColor Yellow
    Deploy-MonitoringStack | Out-Null
}

# Open both dashboards using the utility function
Open-AllDashboards