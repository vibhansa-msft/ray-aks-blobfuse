# Ray Monitoring Stack Deployment
# Deploys Grafana + Prometheus for comprehensive Ray cluster monitoring

# Import utility functions and load parameters
. "$PSScriptRoot\utils.ps1"
Load-EnvParameters

Write-Host ""
Write-Host "Ray Monitoring Stack Deployment" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

# Deploy the monitoring stack
Write-Host "Deploying Grafana + Prometheus monitoring stack..." -ForegroundColor Yellow
kubectl apply -f k8s/monitoring/monitoring-stack.yaml | Out-Null

Write-Host "Waiting for monitoring services to be ready..." -ForegroundColor Gray

# Wait for Prometheus to be ready
Write-Host "  Checking Prometheus..." -ForegroundColor Gray
$prometheusReady = $false
$waitTime = 0
$maxWait = 120

while (-not $prometheusReady -and $waitTime -lt $maxWait) {
    $prometheusPod = kubectl get pod -l "app=prometheus" -n monitoring -o jsonpath="{.items[0].status.phase}" 2>$null
    if ($prometheusPod -eq "Running") {
        $prometheusReady = $true
        Write-Host "  ✅ Prometheus is ready" -ForegroundColor Green
    } else {
        Start-Sleep -Seconds 10
        $waitTime += 10
    }
}

# Wait for Grafana to be ready
Write-Host "  Checking Grafana..." -ForegroundColor Gray
$grafanaReady = $false
$waitTime = 0

while (-not $grafanaReady -and $waitTime -lt $maxWait) {
    $grafanaPod = kubectl get pod -l "app=grafana" -n monitoring -o jsonpath="{.items[0].status.phase}" 2>$null
    if ($grafanaPod -eq "Running") {
        $grafanaReady = $true
        Write-Host "  ✅ Grafana is ready" -ForegroundColor Green
    } else {
        Start-Sleep -Seconds 10
        $waitTime += 10
    }
}

# Get Grafana service details
Write-Host ""
Write-Host "Getting Grafana access information..." -ForegroundColor Cyan
$grafanaIP = kubectl get service grafana-service -n monitoring -o jsonpath="{.status.loadBalancer.ingress[0].ip}" 2>$null

if ($grafanaIP) {
    Write-Host "✅ Grafana External IP: $grafanaIP" -ForegroundColor Green
    $grafanaURL = "http://${grafanaIP}:3000"
} else {
    Write-Host "⏳ External IP pending... Checking again in 30 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    $grafanaIP = kubectl get service grafana-service -n monitoring -o jsonpath="{.status.loadBalancer.ingress[0].ip}" 2>$null
    if ($grafanaIP) {
        $grafanaURL = "http://${grafanaIP}:3000"
    } else {
        $grafanaURL = "http://localhost:3000 (use port forwarding)"
    }
}

Write-Host ""
Write-Host "=======================================" -ForegroundColor Green
Write-Host "Monitoring Stack Deployed Successfully!" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 Access Information:" -ForegroundColor Yellow
Write-Host "   Grafana URL: $grafanaURL" -ForegroundColor White
Write-Host "   Username: admin" -ForegroundColor Gray
Write-Host "   Password: raymonitoring123" -ForegroundColor Gray
Write-Host ""

if ($grafanaURL -like "*localhost*") {
    Write-Host "🔧 Port Forwarding Setup:" -ForegroundColor Yellow
    Write-Host "   kubectl port-forward svc/grafana-service 3000:3000 -n monitoring" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "📊 Pre-configured Data Sources:" -ForegroundColor Cyan
Write-Host "   • Prometheus (Ray metrics, K8s metrics)" -ForegroundColor White
Write-Host "   • Ray Dashboard metrics on port 8080" -ForegroundColor White
Write-Host ""

Write-Host "📈 Available Metrics:" -ForegroundColor Cyan
Write-Host "   • Ray cluster health and resource usage" -ForegroundColor White
Write-Host "   • Job execution progress and performance" -ForegroundColor White
Write-Host "   • Memory, CPU, and GPU utilization" -ForegroundColor White
Write-Host "   • Task scheduling and worker status" -ForegroundColor White
Write-Host ""

Write-Host ""
Write-Host "Dashboards are pre-configured and auto-provisioned in Grafana." -ForegroundColor Green
Write-Host "No manual import needed - they will appear automatically in the Dashboards section." -ForegroundColor Green
Write-Host ""

Write-Host "🚀 Next Steps:" -ForegroundColor Green
Write-Host "   1. Access Grafana dashboard" -ForegroundColor Gray
Write-Host "   2. View imported Ray monitoring dashboard" -ForegroundColor Gray
Write-Host "   3. Run Ray jobs to see live monitoring data" -ForegroundColor Gray
Write-Host ""

# Offer to open Grafana automatically
$openGrafana = Read-Host "Open Grafana in browser now? (y/n)"
if ($openGrafana -eq "y" -or $openGrafana -eq "Y") {
    if ($grafanaURL -notlike "*localhost*") {
        Start-Process $grafanaURL
    } else {
        Write-Host "Setting up port forwarding and opening Grafana..." -ForegroundColor Cyan
        Start-Process -FilePath "kubectl" -ArgumentList "port-forward", "svc/grafana-service", "3000:3000", "-n", "monitoring" -WindowStyle Hidden
        Start-Sleep -Seconds 5
        Start-Process "http://localhost:3000"
    }
}