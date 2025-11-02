# =====================================================
# Utility functions and parameter loading for deploy.ps1
# =====================================================
# This file contains:
# - Load-EnvParameters: Load deployment parameters from .env.example
# - Need: Check and install missing CLI tools (kubectl, helm, docker, python)
# =====================================================

# =====================================================
# Load Environment Parameters from .env.example
# =====================================================
# If .env.example exists, load all parameters from it
# Otherwise, use hardcoded defaults
function Load-EnvParameters {
    # Load parameters from .env.example if present
    if (Test-Path ".env.example") {
        Get-Content .env.example | ForEach-Object {
            if ($_ -match "^([A-Z_]+)=(.*)$") {
                $name = $matches[1]
                $value = $matches[2] -replace '^"|"$', ''  # Remove surrounding quotes
                Set-Variable -Name $name -Value $value -Scope Script
            }
        }
    }
    
    # Set default values for parameters not in .env.example
    if (-not $SUBSCRIPTION) { $SUBSCRIPTION = "68e5d74d-cc6b-4be6-9606-cd1c77fa55f0" }
    if (-not $RG) { $RG = "vibhansa-ray-aks-rg" }
    if (-not $AKS) { $AKS = "vibhansa-ray-aks" }
    if (-not $SA) { $SA = "vibhansaraystorage" }
    if (-not $LOC) { $LOC = "eastus" }
    if (-not $CONTAINER) { $CONTAINER = "dataset" }
    if (-not $GPU) { $GPU = $false }
    if (-not $NodeCount) { $NodeCount = 2 }
    if (-not $VmType) { $VmType = "Standard_D4_v2" }
    if (-not $WORKER_REPLICAS) { $WORKER_REPLICAS = 4 }
    if (-not $NUM_WORKERS) { $NUM_WORKERS = 4 }
    
    # Mount path defaults (if not in .env.example)
    if (-not $DATA_DIR) { $DATA_DIR = "/mnt/blob/datasets" }
    if (-not $CHECKPOINT_DIR) { $CHECKPOINT_DIR = "/mnt/blob/checkpoints" }
    if (-not $APP_DIR) { $APP_DIR = "/app" }
    if (-not $CACHE_DIR) { $CACHE_DIR = "/mnt/blobfusecache" }
}

# =====================================================
# Check and Install Required CLI Tools
# =====================================================
# This function ensures required tools are available
# Supported tools: kubectl, helm, docker, python
# If a tool is not found, it will be installed automatically
function Need {
    param([string]$cmd)
    
    # First, check if command already exists in PATH
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        Write-Host "$cmd is already installed. Skipping installation."
        return
    }
    
    # Check if previously installed in standard locations
    $helmTarget = "$env:USERPROFILE\.azure-helm"
    $kubectlPath = "$env:USERPROFILE\.azure-kubectl"
    
    if ($cmd -eq "helm" -and (Test-Path "$helmTarget\helm.exe")) {
        Write-Host "Helm found in standard location. Adding to PATH..."
        $env:PATH = "$helmTarget;" + $env:PATH
        return
    }
    
    if ($cmd -eq "kubectl" -and (Test-Path "$kubectlPath\kubectl.exe")) {
        Write-Host "kubectl found in standard location. Adding to PATH..."
        $env:PATH = "$kubectlPath;" + $env:PATH
        return
    }
    
    # If not found in PATH or standard locations, proceed with installation based on command type
    # kubectl: Install via Azure CLI
    if ($cmd -eq "kubectl") {
        $kubectlPath = "$env:USERPROFILE\.azure-kubectl"
        $kubectlExe = Join-Path $kubectlPath "kubectl.exe"
        $foundKubectl = $false
        
        # Check if kubectl is already in PATH or installed locally
        if (Get-Command kubectl -ErrorAction SilentlyContinue) {
            $foundKubectl = $true
        } elseif (Test-Path $kubectlExe) {
            # Found in default install location, add to PATH
            $env:PATH = "$kubectlPath;" + $env:PATH
            $foundKubectl = $true
        }

        if (-not $foundKubectl) {
            # Install kubectl via Azure CLI
            Write-Host "Installing kubectl via az..."
            az aks install-cli | Out-Null
            if (Test-Path $kubectlExe) {
                # Add to current session PATH
                $env:PATH = "$kubectlPath;" + $env:PATH

                # Add to user PATH permanently for future sessions
                $currentUserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
                if ($currentUserPath -notlike "*$kubectlPath*") {
                    [Environment]::SetEnvironmentVariable("PATH", "$kubectlPath;" + $currentUserPath, "User")
                    Write-Host "Added $kubectlPath to user PATH. You may need to restart your terminal for changes to take effect."
                }
            }

            if (-not (Get-Command kubectl -ErrorAction SilentlyContinue) -and -not (Test-Path $kubectlExe)) {
                Write-Error "Failed to install kubectl automatically. Please install manually."
                exit 1
            }
        }

    # helm: Download from official release, extract, and add to PATH
    } elseif ($cmd -eq "helm") {
        # Download Helm binary
        Write-Host "Installing Helm..."
        $helmUrl = "https://get.helm.sh/helm-v3.14.2-windows-amd64.zip"
        $zipPath = "$env:TEMP\helm.zip"
        $extractPath = "$env:TEMP\helm-extract"
        
        try {
            Write-Host "Downloading Helm from $helmUrl..."
            Invoke-WebRequest -Uri $helmUrl -OutFile $zipPath -ErrorAction Stop
            Write-Host "Extracting Helm..."
            Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

            # Copy to standard location and add to PATH
            $helmExe = "$extractPath\windows-amd64\helm.exe"
            $helmTarget = "$env:USERPROFILE\.azure-helm"
            if (!(Test-Path $helmTarget)) { New-Item -ItemType Directory -Path $helmTarget | Out-Null }
            Copy-Item $helmExe -Destination "$helmTarget\helm.exe" -Force
            $env:PATH = "$helmTarget;" + $env:PATH

            # Cleanup temporary files
            Remove-Item $zipPath -Force
            Remove-Item $extractPath -Recurse -Force

            Write-Host "Helm installed successfully."
        } catch {
            Write-Error "Failed to download/install Helm: $_"
            exit 1
        }

    # docker: Install via winget and start Docker Desktop
    } elseif ($cmd -eq "docker") {
        # Install Docker Desktop
        Write-Host "Installing Docker Desktop via winget..."
        winget install -e --id Docker.DockerDesktop

        # Start Docker Desktop
        Write-Host "Attempting to start Docker Desktop..."
        Start-Process -FilePath "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe" -WindowStyle Minimized

        # Wait for Docker daemon to be ready (up to 2 minutes)
        $maxWait = 120
        $waited = 0
        while (-not (docker info 2>$null)) {
            Start-Sleep -Seconds 5
            $waited += 5
            if ($waited -ge $maxWait) {
                Write-Error "Docker Desktop did not start within $maxWait seconds. Please start it manually and ensure it is running."
                exit 1
            }
        }

        Write-Host "Docker Desktop is running."

    # python: Install via winget
    } elseif ($cmd -eq "python") {
        # Install Python 3.11
        Write-Host "Installing Python 3.11 via winget..."
        winget install -e --id Python.Python.3.11

        if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
            Write-Error "Failed to install Python automatically. Please install manually."
            exit 1
        }

    # Unknown command
    } else {
        Write-Error "Missing '$cmd' in PATH."
        exit 1
    }
}