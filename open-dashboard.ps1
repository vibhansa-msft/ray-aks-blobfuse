# Ray Dashboard - Simple Port Forwarding
# Starts port forwarding directly to Ray head pod and opens browser

# Stop existing port forwarding
Get-Process kubectl -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*port-forward*8265*" } | Stop-Process -Force 2>$null

# Get Ray head pod name
$headPod = kubectl get pod -l "ray.io/node-type=head" -o jsonpath="{.items[0].metadata.name}" 2>$null

if (-not $headPod) {
    Write-Host "No Ray head pod found. Run quick-deploy.ps1 first." -ForegroundColor Red
    exit 1
}

# Start port forwarding to head pod directly
Write-Host "Starting Ray Dashboard port forwarding..." -ForegroundColor Cyan
Start-Process kubectl -ArgumentList "port-forward", "pod/$headPod", "8265:8265" -WindowStyle Hidden

# Wait and test connection
Start-Sleep -Seconds 8
try {
    Invoke-WebRequest -Uri "http://localhost:8265" -TimeoutSec 5 -ErrorAction Stop | Out-Null
    Start-Process "http://localhost:8265"
} catch {
    Write-Host "Dashboard not accessible. Try again in a few moments." -ForegroundColor Red
}