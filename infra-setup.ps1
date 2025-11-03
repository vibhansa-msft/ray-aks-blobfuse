<#
Infrastructure Setup Script for Ray HPO
Sets up Azure resources: Resource Group, Storage Account, AKS Cluster, KubeRay
Run this once when first setting up the infrastructure

Usage: powershell -ExecutionPolicy Bypass -File .\infra-setup.ps1
#>

# ========== SETUP AND INITIALIZATION ==========

# Import utility functions and load parameters
. "$PSScriptRoot\utils.ps1"
Load-EnvParameters

# Check for required CLI tools
foreach ($t in @("az","kubectl","helm","docker","python")) { Need $t }

# ========== AZURE AUTHENTICATION ==========

# Check and perform Azure authentication if needed
Write-Host "Checking Azure authentication..."
$loginCheck = az account show 2>&1
if ($LASTEXITCODE -ne 0 -or -not $loginCheck) {
  Write-Host "Not logged in. Running az login..."
  az login | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Azure login failed" -ForegroundColor Red
    exit 1
  }
} else {
  Write-Host "Already logged in to Azure."
}

Write-Host "Setting Azure subscription to $SUBSCRIPTION..."
az account set --subscription $SUBSCRIPTION
if ($LASTEXITCODE -ne 0) {
  Write-Host "ERROR: Failed to set subscription $SUBSCRIPTION" -ForegroundColor Red
  exit 1
}

# ========== AZURE RESOURCES PROVISIONING ==========

# ===== Resource Group =====
Write-Host "Checking if resource group $RG exists..." -ForegroundColor Cyan
$rgExists = az group exists -n $RG | ConvertFrom-Json
if ($rgExists) {
    Write-Host "Resource group $RG already exists. Skipping creation." -ForegroundColor Green
} else {
    Write-Host "Creating resource group $RG..."
    az group create -n $RG -l $LOC | Out-Null
    Write-Host "Resource group created successfully." -ForegroundColor Green
}

# ===== AKS Cluster =====
Write-Host ""
Write-Host "Setting up AKS cluster..." -ForegroundColor Cyan

# Convert GPU to boolean if it's a string
$gpuEnabled = if ($GPU -is [bool]) { $GPU } else { [System.Convert]::ToBoolean($GPU) }

# Call the dedicated AKS setup script
& "$PSScriptRoot\aks-setup.ps1" `
    -ResourceGroup $RG `
    -ClusterName $AKS `
    -Location $LOC `
    -EnableGpu $gpuEnabled `
    -NodeCount $NodeCount `
    -VmType $VmType `
    -StorageAccountName $SA `
    -StorageAccountResourceGroup $

# Check if setup-aks.ps1 failed
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: AKS setup script failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

# ===== Storage Account =====
Write-Host ""
Write-Host "Setting up Storage Account..." -ForegroundColor Cyan
Write-Host "Checking if Storage Account $SA exists..."
$saExists = $false
try {
    $saResult = az storage account show -g $RG -n $SA --query "name" -o tsv 2>$null
    if ($saResult) {
        $saExists = $true
        Write-Host "Storage Account $SA already exists. Skipping creation." -ForegroundColor Green
    }
} catch {
    # Storage Account doesn't exist, will create it
}

if (-not $saExists) {
    Write-Host "Creating Storage Account $SA..."
    az storage account create -g $RG -n $SA -l $LOC --sku Standard_LRS | Out-Null
    Write-Host "Storage Account created successfully." -ForegroundColor Green
}

# ===== Blob Container =====
Write-Host "Checking if container $Container exists in Storage Account $SA..."
$containerExists = $false
try {
    $containerResult = az storage container exists -n $Container --account-name $SA --query "exists" -o tsv 2>$null
    if ($containerResult -eq "true") {
        $containerExists = $true
        Write-Host "Container $Container already exists. Skipping creation." -ForegroundColor Green
    }
} catch {
    # Container check failed, will attempt to create
}

if (-not $containerExists) {
    Write-Host "Creating container $Container..."
    az storage container create --name $Container --account-name $SA --auth-mode login | Out-Null
    Write-Host "Container created successfully." -ForegroundColor Green
}

# ========== KUBERNETES OPERATORS ==========

# ===== Install KubeRay Operator =====
Write-Host ""
Write-Host "Installing KubeRay Operator..." -ForegroundColor Cyan
Write-Host "Checking if KubeRay operator is already deployed..."
$kuberayDeploy = $false
try {
    $kuberayResult = kubectl get deployment -n kuberay-operator kuberay-operator 2>$null
    if ($kuberayResult) {
        $kuberayDeploy = $true
        Write-Host "KubeRay operator already deployed. Skipping installation." -ForegroundColor Green
    }
} catch {
    # KubeRay not deployed yet, check if helm release exists
    try {
        $helmRelease = helm list -n kuberay-operator 2>$null | Select-String "kuberay-operator"
        if ($helmRelease) {
            $kuberayDeploy = $true
            Write-Host "KubeRay operator Helm release already exists. Skipping installation." -ForegroundColor Green
        }
    } catch {
        # Helm release doesn't exist
    }
}

if (-not $kuberayDeploy) {
    Write-Host "Installing KubeRay operator (Helm)..."
    try {
        helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 2>$null | Out-Null
        helm repo update 2>$null | Out-Null
        helm install kuberay-operator kuberay/kuberay-operator --version 1.4.2 2>$null | Out-Null
        kubectl rollout status deploy/kuberay-operator -n kuberay-operator 2>$null | Out-Null
        Write-Host "KubeRay operator installed successfully." -ForegroundColor Green
    } catch {
        Write-Host "Warning: Failed to install KubeRay operator. Continuing..." -ForegroundColor Yellow
    }
}

# ========== DATASET PREPARATION ==========

# Create and activate virtual environment
Write-Host ""
Write-Host "Preparing dataset..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv | Out-Null
}

& ".\.venv\Scripts\Activate.ps1"

# Check if dataset is already uploaded to Azure Blob Storage
Write-Host "Checking if AG News dataset already exists in Azure Blob Storage..."
$datasetExists = $false
$datasetLocation = ""

# Step 1: Check storage account first
try {
    $blobList = @(az storage blob list --account-name $SA --container-name $Container --prefix "ag_news" --query "[].name" -o tsv 2>$null)
    $blobCount = ($blobList | Measure-Object).Count
    if ($blobCount -gt 0) {
        $datasetExists = $true
        $datasetLocation = "storage account"
        Write-Host "✅ AG News dataset found in storage account ($blobCount blobs). Skipping download and upload." -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Could not query storage account. Checking locally..." -ForegroundColor Yellow
}

# Step 2: If not in storage account, check locally
if (-not $datasetExists) {
    $localParquetFiles = @(Get-ChildItem -Path "ag_news_parquet" -Filter "*.parquet" -ErrorAction SilentlyContinue)
    $localParquetCount = ($localParquetFiles | Measure-Object).Count
    
    if ($localParquetCount -gt 0) {
        $datasetExists = $true
        $datasetLocation = "local"
        Write-Host "✅ AG News dataset found locally ($localParquetCount files). Uploading to storage account..." -ForegroundColor Green
    }
}

# Step 3: If not found locally or in storage, will download
if (-not $datasetExists) {
    Write-Host "❌ AG News dataset not found in storage or locally. Will download and prepare..." -ForegroundColor Yellow
}

# Upload dataset to Azure Blob Storage based on where it's located
if ($datasetLocation -eq "storage account") {
    # Dataset already in storage account - skip everything
    Write-Host "Skipping dataset operations (already in storage account)." -ForegroundColor Green
} elseif ($datasetLocation -eq "local") {
    # Dataset is local - just upload to storage account
    Write-Host "Uploading local dataset to Azure Blob Storage..."
    try {
        az storage blob upload-batch --account-name $SA -d "$Container/ag_news" -s "ag_news_parquet" 2>&1 | Out-Null
        Write-Host "Dataset uploaded successfully." -ForegroundColor Green
    } catch {
        Write-Host "Warning: Dataset upload may have failed. Continuing..." -ForegroundColor Yellow
    }
} else {
    # Dataset not found anywhere - download, prepare, and upload
    Write-Host "Downloading and preparing AG News dataset locally and uploading to Azure Blob..."

    # Install Python dependencies
    Write-Host "Installing Python dependencies..."
    try {
        # Install dependencies without upgrading pip (to avoid permission issues)
        pip install datasets pyarrow pandas 2>&1 | Out-Null
        pip install -r app/requirements.txt 2>&1 | Out-Null
        Write-Host "Python dependencies installed successfully." -ForegroundColor Green
    } catch {
        Write-Host "Warning: Some Python dependencies may have failed to install. Continuing..." -ForegroundColor Yellow
    }

    # Prepare dataset and upload to storage
    Write-Host "Preparing AG News dataset..."
    try {
        python data/prepare_ag_news.py 2>&1 | Out-Null
        Write-Host "Dataset prepared successfully." -ForegroundColor Green
    } catch {
        Write-Host "Warning: Dataset preparation may have failed. Continuing..." -ForegroundColor Yellow
    }

    Write-Host "Uploading dataset to Azure Blob Storage..."
    try {
        az storage blob upload-batch --account-name $SA -d "$Container/ag_news" -s "ag_news_parquet" 2>&1 | Out-Null
        Write-Host "Dataset uploaded successfully." -ForegroundColor Green
    } catch {
        Write-Host "Warning: Dataset upload may have failed. Continuing..." -ForegroundColor Yellow
    }
}

# Deactivate Python virtual environment
deactivate

if (-not $saExists) {
    # Terminate the script now and ask user to manually assign the roles
    Write-Host ""
    Write-Host "IMPORTANT: Please assign the following roles to the AKS managed identity:" -ForegroundColor Yellow
    Write-Host "  - Storage Blob Data Contributor" -ForegroundColor Yellow
    Write-Host "  - Storage Account Contributor" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After assigning roles, run quick-deploy.ps1 to deploy the RayJob." -ForegroundColor Yellow
    exit 1
}

# ========== COMPLETION ==========

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Infrastructure Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your infrastructure is now ready. Use quick-deploy.ps1 to redeploy:" -ForegroundColor Yellow
Write-Host "  powershell -ExecutionPolicy Bypass -File .\quick-deploy.ps1" -ForegroundColor Gray
Write-Host ""
