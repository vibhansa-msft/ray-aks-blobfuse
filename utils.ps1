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
    
    if (-not $MAX_PREPROCESS_TASK_CONCURRENCY) { $MAX_PREPROCESS_TASK_CONCURRENCY = 3 }
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

# =====================================================
# Deploy Monitoring Stack (Grafana + Prometheus)
# =====================================================
# Deploys Grafana monitoring stack if not already running
function Deploy-MonitoringStack {
    Write-Host "[MONITOR] Checking monitoring stack..." -ForegroundColor Cyan
    
    # Check if monitoring namespace exists and has running pods
    $monitoringPods = kubectl get pods -n monitoring --no-headers 2>$null
    if ($LASTEXITCODE -eq 0 -and $monitoringPods) {
        $runningPods = $monitoringPods | Where-Object { $_ -match "Running" }
        if ($runningPods.Count -ge 2) {
            Write-Host "[SUCCESS] Monitoring stack already running" -ForegroundColor Green
            return $true
        }
    }
    
    Write-Host "[DEPLOY] Deploying monitoring stack..." -ForegroundColor Yellow
    
    # Deploy monitoring stack
    kubectl apply -f k8s/monitoring/monitoring-stack.yaml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to deploy monitoring stack" -ForegroundColor Red
        return $false
    }
    
    # Wait for pods to be ready
    Write-Host "[WAIT] Waiting for monitoring pods to be ready..." -ForegroundColor Yellow
    
    $timeout = 300  # 5 minutes
    $elapsed = 0
    do {
        Start-Sleep -Seconds 10
        $elapsed += 10
        
        $prometheusPod = kubectl get pods -n monitoring -l app=prometheus --no-headers 2>$null | Where-Object { $_ -match "Running" }
        $grafanaPod = kubectl get pods -n monitoring -l app=grafana --no-headers 2>$null | Where-Object { $_ -match "Running" }
        
        if ($prometheusPod -and $grafanaPod) {
            Write-Host "[SUCCESS] Monitoring stack deployed successfully!" -ForegroundColor Green
            
            # Import Ray dashboard if available
            if (Test-Path "grafana-ray-dashboard.json") {
                Import-RayDashboard
            }
            return $true
        }
        
        Write-Host "   Still waiting... ($elapsed/$timeout seconds)" -ForegroundColor Gray
    } while ($elapsed -lt $timeout)
    
    Write-Host "Warning: Monitoring deployment timeout. Check logs manually." -ForegroundColor Yellow
    return $false
}

# =====================================================
# Import Ray Dashboard to Grafana
# =====================================================
function Import-RayDashboard {
    Write-Host "[IMPORT] Importing Ray dashboard..." -ForegroundColor Yellow
    
    try {
        # Get Grafana service details
        $grafanaService = kubectl get svc grafana-service -n monitoring -o json | ConvertFrom-Json
        $grafanaURL = "http://localhost:3000"  # Use port forwarding by default
        
        # Start port forwarding in background if needed
        $portForwardJob = Start-Job -ScriptBlock {
            kubectl port-forward svc/grafana-service 3000:3000 -n monitoring
        }
        Start-Sleep -Seconds 5
        
        # Load and prepare dashboard
        $dashboardJson = Get-Content "grafana-ray-dashboard.json" -Raw
        $importPayload = @{
            dashboard = ($dashboardJson | ConvertFrom-Json).dashboard
            overwrite = $true
            inputs = @()
        } | ConvertTo-Json -Depth 20
        
        $headers = @{
            'Authorization' = 'Basic ' + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:raymonitoring123"))
            'Content-Type' = 'application/json'
        }
        
        # Import dashboard
        Invoke-RestMethod -Uri "$grafanaURL/api/dashboards/import" -Method POST -Headers $headers -Body $importPayload | Out-Null
        Write-Host "[SUCCESS] Ray dashboard imported successfully!" -ForegroundColor Green
        
        # Clean up port forward job
        Stop-Job $portForwardJob -ErrorAction SilentlyContinue
        Remove-Job $portForwardJob -ErrorAction SilentlyContinue
    }
    catch {
        Write-Host "[WARNING] Dashboard import failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# =====================================================
# Open Both Ray and Grafana Dashboards
# =====================================================
function Open-AllDashboards {
    Write-Host "[DASHBOARD] Opening Ray and Grafana dashboards..." -ForegroundColor Cyan
    
    # Get Ray dashboard URL - always use port forwarding for reliability
    Write-Host "[INFO] Setting up Ray dashboard port forwarding..." -ForegroundColor Yellow
    $rayURL = "http://localhost:8265"
    
    # Stop any existing port forwarding jobs
    Get-Job -Name "RayPortForward*" -ErrorAction SilentlyContinue | Stop-Job | Remove-Job
    
    # Start Ray port forwarding to head service (more reliable than external service)
    $headPod = kubectl get pod -l "ray.io/node-type=head" -o jsonpath="{.items[0].metadata.name}" 2>$null
    if ($headPod) {
        Start-Job -Name "RayPortForward" -ScriptBlock {
            param($pod)
            kubectl port-forward pod/$pod 8265:8265
        } -ArgumentList $headPod | Out-Null
        Start-Sleep -Seconds 5
    } else {
        Write-Host "[WARNING] No Ray head pod found for port forwarding" -ForegroundColor Yellow
    }
    
    # Get Grafana URL
    $grafanaExternalIP = kubectl get svc grafana-service -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>$null
    if (-not $grafanaExternalIP -or $grafanaExternalIP -eq "") {
        Write-Host "[INFO] Setting up Grafana port forwarding..." -ForegroundColor Yellow
        $grafanaURL = "http://localhost:3000"
        
        # Stop any existing Grafana port forwarding
        Get-Job -Name "GrafanaPortForward*" -ErrorAction SilentlyContinue | Stop-Job | Remove-Job
        
        # Start Grafana port forwarding
        Start-Job -Name "GrafanaPortForward" -ScriptBlock {
            kubectl port-forward svc/grafana-service 3000:3000 -n monitoring
        } | Out-Null
        Start-Sleep -Seconds 3
    } else {
        Write-Host "[INFO] Using Grafana external IP: $grafanaExternalIP" -ForegroundColor Yellow
        $grafanaURL = "http://${grafanaExternalIP}:3000"
    }
    
    Write-Host "[INFO] Dashboard URLs:" -ForegroundColor Green
    Write-Host "   Ray Dashboard: $rayURL" -ForegroundColor White
    Write-Host "   Grafana: $grafanaURL" -ForegroundColor White
    Write-Host "   Grafana Credentials: admin / raymonitoring123" -ForegroundColor Yellow
    Write-Host ""
    
    # Automatically open both dashboards
    Write-Host "[BROWSER] Opening Ray dashboard..." -ForegroundColor Cyan
    Start-Process $rayURL
    
    Start-Sleep -Seconds 2
    
    Write-Host "[BROWSER] Opening Grafana dashboard..." -ForegroundColor Cyan
    Start-Process $grafanaURL
    
    Write-Host "[SUCCESS] Both dashboards opened!" -ForegroundColor Green
    Write-Host "[INFO] Use credentials for Grafana: admin / raymonitoring123" -ForegroundColor Yellow
    
    # Import Ray dashboard to Grafana if it exists
    if (Test-Path "grafana-ray-dashboard.json") {
        Write-Host "[IMPORT] Importing Ray dashboard to Grafana..." -ForegroundColor Cyan
        Start-Sleep -Seconds 5  # Wait for Grafana to be ready
        try {
            $dashboardJson = Get-Content "grafana-ray-dashboard.json" -Raw
            $importPayload = @{
                dashboard = ($dashboardJson | ConvertFrom-Json).dashboard
                overwrite = $true
                inputs = @()
            } | ConvertTo-Json -Depth 20
            
            $headers = @{
                'Authorization' = 'Basic ' + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:raymonitoring123"))
                'Content-Type' = 'application/json'
            }
            
            $response = Invoke-RestMethod -Uri "$grafanaURL/api/dashboards/import" -Method POST -Headers $headers -Body $importPayload
            Write-Host "[SUCCESS] Ray dashboard imported! Dashboard ID: $($response.dashboardId)" -ForegroundColor Green
            Write-Host "[INFO] Navigate to Dashboards -> Browse -> Ray Cluster Monitoring in Grafana" -ForegroundColor Yellow
        }
        catch {
            Write-Host "[WARNING] Dashboard import failed: $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "[INFO] You can manually import grafana-ray-dashboard.json via Grafana UI" -ForegroundColor Gray
        }
    }
}