
<#
Deploy Script for Ray HPO Training Pipeline
This is a convenience wrapper that guides users to use the modular scripts:
- infra-setup.ps1:  One-time infrastructure setup (AKS, Storage, KubeRay)
- quick-deploy.ps1: Fast redeploy of StorageClasses, PVCs, and RayJob

For one-time infrastructure setup:
  powershell -ExecutionPolicy Bypass -File .\infra-setup.ps1

For quick redeploy (after infrastructure is set up):
  powershell -ExecutionPolicy Bypass -File .\quick-deploy.ps1

For full deployment (infrastructure + app):
  powershell -ExecutionPolicy Bypass -File .\deploy.ps1
#>

# ========== SETUP AND INITIALIZATION ==========

# Import utility functions and load parameters
. "$PSScriptRoot\utils.ps1"
Load-EnvParameters

# Check for required CLI tools
foreach ($t in @("az","kubectl","helm","docker","python")) { Need $t }

# ========== MAIN DEPLOYMENT FLOW ==========

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ray HPO Deployment Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Infrastructure setup
Write-Host "Step 1: Running infrastructure setup..." -ForegroundColor Yellow
& "$PSScriptRoot\infra-setup.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Infrastructure setup failed" -ForegroundColor Red
    exit 1
}

# Step 2: Quick deploy (PVCs, StorageClasses, RayJob)
Write-Host ""
Write-Host "Step 2: Running quick deploy..." -ForegroundColor Yellow
& "$PSScriptRoot\quick-deploy.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Quick deploy failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Full Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Tip: For future deployments, use:" -ForegroundColor Yellow
Write-Host "  powershell -ExecutionPolicy Bypass -File .\quick-deploy.ps1" -ForegroundColor Gray
Write-Host ""

