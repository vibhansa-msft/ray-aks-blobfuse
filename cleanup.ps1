<#
.SYNOPSIS
    Cleanup script to delete all Azure resources created by the deployment
.DESCRIPTION
    This script deletes:
    - AKS cluster
    - Storage account
    - Resource group (which removes everything)
    Requires Azure CLI and proper permissions
#>

# Load environment parameters
. "$PSScriptRoot\utils.ps1"
Load-EnvParameters

Write-Host "========== AZURE CLEANUP ==========" -ForegroundColor Cyan
Write-Host ""

# Verify Azure login
Write-Host "Verifying Azure authentication..." -ForegroundColor Yellow
$loginAttempt = 0
$maxAttempts = 3

while ($loginAttempt -lt $maxAttempts) {
    try {
        $accountInfo = az account show 2>&1
        if ($LASTEXITCODE -eq 0) {
            $currentUser = az account show --query "user.name" -o tsv
            $currentSub = az account show --query "name" -o tsv
            Write-Host "✅ Logged in as: $currentUser" -ForegroundColor Green
            Write-Host "   Subscription: $currentSub" -ForegroundColor Green
            break
        } else {
            throw "Not logged in"
        }
    } catch {
        $loginAttempt++
        if ($loginAttempt -lt $maxAttempts) {
            Write-Host ""
            Write-Host "⚠️  Not logged in to Azure (Attempt $loginAttempt/$maxAttempts)" -ForegroundColor Yellow
            Write-Host "Running: az login" -ForegroundColor Cyan
            az login | Out-Null
            Write-Host ""
        } else {
            Write-Host ""
            Write-Host "❌ Failed to authenticate with Azure after $maxAttempts attempts" -ForegroundColor Red
            Write-Host "Please manually run: az login" -ForegroundColor Yellow
            exit 1
        }
    }
}

Write-Host ""
Write-Host "Resources to be deleted:" -ForegroundColor Yellow
Write-Host "  - Resource Group: $RG" -ForegroundColor White
Write-Host "  - AKS Cluster: $AKS" -ForegroundColor White
Write-Host "  - Storage Account: $SA" -ForegroundColor White
Write-Host "  - All associated resources (VNets, disks, etc.)" -ForegroundColor White
Write-Host ""

# Confirmation
Write-Host "⚠️  WARNING: This will DELETE all resources!" -ForegroundColor Red
$confirmation = Read-Host "Type 'yes' to confirm deletion"

if ($confirmation -ne "yes") {
    Write-Host "Cleanup cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Starting cleanup..." -ForegroundColor Cyan
Write-Host ""

# Delete AKS cluster first (can take a few minutes)
Write-Host "Deleting AKS cluster: $AKS..." -ForegroundColor Yellow
try {
    $aksExists = az aks show -g $RG -n $AKS --query "name" -o tsv 2>$null
    if ($aksExists) {
        az aks delete -g $RG -n $AKS --yes 2>&1 | Out-Null
        Write-Host "✅ AKS cluster deleted" -ForegroundColor Green
    } else {
        Write-Host "⚠️  AKS cluster not found (already deleted)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Warning deleting AKS cluster: $_" -ForegroundColor Yellow
}

Write-Host ""

# Delete Storage Account
Write-Host "Deleting Storage Account: $SA..." -ForegroundColor Yellow
try {
    $saExists = az storage account show -g $RG -n $SA --query "name" -o tsv 2>$null
    if ($saExists) {
        az storage account delete -g $RG -n $SA --yes 2>&1 | Out-Null
        Write-Host "✅ Storage account deleted" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Storage account not found (already deleted)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Warning deleting storage account: $_" -ForegroundColor Yellow
}

Write-Host ""

# Delete Resource Group (this will delete everything including networking, disks, etc.)
Write-Host "Deleting Resource Group: $RG..." -ForegroundColor Yellow
Write-Host "(This may take a few minutes...)" -ForegroundColor Gray
try {
    $rgExists = az group exists -n $RG
    if ($rgExists) {
        az group delete -n $RG --yes 2>&1 | Out-Null
        Write-Host "✅ Resource group deleted" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Resource group not found (already deleted)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Warning deleting resource group: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========== CLEANUP COMPLETE ==========" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor White
Write-Host "✅ All Azure resources have been deleted" -ForegroundColor Green
Write-Host ""

# Validate that resource group is deleted
Write-Host "Validating resource group deletion..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Remaining Resource Groups:" -ForegroundColor Cyan
try {
    $remainingRGs = az group list --query "[].name" -o table 2>&1
    if ($remainingRGs -match "No resources found") {
        Write-Host "✅ No resource groups found - Clean slate!" -ForegroundColor Green
    } else {
        Write-Host $remainingRGs -ForegroundColor White
        if ($remainingRGs -match $RG) {
            Write-Host "⚠️  WARNING: Resource group '$RG' still exists!" -ForegroundColor Red
        } else {
            Write-Host "✅ Target resource group '$RG' successfully deleted" -ForegroundColor Green
        }
    }
} catch {
    Write-Host "Error retrieving resource groups: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "Run deploy.ps1 to redeploy everything fresh" -ForegroundColor Gray
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
