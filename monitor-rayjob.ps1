# Ray Job Monitoring Script - Simplified
# Checks Ray job status every 30 seconds until completion or failure

param(
    [string]$JobName = "hpo-job-gpu",
    [int]$CheckInterval = 30,
    [int]$MaxInitializationChecks = 25
)

Write-Host "========== RAY JOB MONITORING ==========" -ForegroundColor Cyan
Write-Host "Job Name: $JobName" -ForegroundColor Green
Write-Host "Check Interval: $CheckInterval seconds" -ForegroundColor Green
Write-Host "Max Initialization Checks: $MaxInitializationChecks" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$startTime = Get-Date
$checkCount = 0
$jobComplete = $false
$initializationChecks = 0

Write-Host "Starting job monitoring at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
Write-Host ""

# Verify job exists
Write-Host "Verifying Ray job exists..." -ForegroundColor Yellow
$initialCheck = kubectl get rayjob $JobName -o yaml 2>$null
if (-not $initialCheck) {
    Write-Host "ERROR: Ray job '$JobName' not found" -ForegroundColor Red
    exit 1
}

Write-Host "Ray job found. Starting monitoring..." -ForegroundColor Green
Write-Host ""

while (-not $jobComplete) {
    $checkCount++
    $elapsedTime = (Get-Date) - $startTime
    $elapsedMinutes = [int]($elapsedTime.TotalSeconds / 60)
    $elapsedSeconds = [int]($elapsedTime.TotalSeconds % 60)
    
    Write-Host "[$($elapsedMinutes)m $($elapsedSeconds)s] Check #$checkCount" -ForegroundColor Cyan
    
    # Get RayJob status
    $rayJobOutput = kubectl get rayjob $JobName -o yaml 2>$null
    
    if (-not $rayJobOutput) {
        Write-Host "  ERROR: Could not retrieve RayJob status" -ForegroundColor Red
        $jobComplete = $true
        break
    }
    
    # Extract deployment status
    $deployStatus = ($rayJobOutput | Select-String "jobDeploymentStatus:" | Select-Object -First 1) -split ':' | Select-Object -Last 1
    $deployStatus = $deployStatus.Trim()
    
    Write-Host "  Status: $deployStatus" -ForegroundColor White
    
    # Extract succeeded/failed counts
    $succeeded = ($rayJobOutput | Select-String "succeeded:" | Select-Object -First 1) -split ':' | Select-Object -Last 1 | ForEach-Object { $_.Trim() }
    $failed = ($rayJobOutput | Select-String "failed:" | Select-Object -First 1) -split ':' | Select-Object -Last 1 | ForEach-Object { $_.Trim() }
    
    if ($succeeded) { Write-Host "  Succeeded: $succeeded" -ForegroundColor Green }
    if ($failed) { Write-Host "  Failed: $failed" -ForegroundColor Yellow }
    
    # Handle Initializing status
    if ($deployStatus -eq "Initializing") {
        $initializationChecks++
        Write-Host "  Initializing: $initializationChecks/$MaxInitializationChecks" -ForegroundColor Yellow
        
        if ($initializationChecks -ge $MaxInitializationChecks) {
            Write-Host ""
            Write-Host "ERROR: Job stuck in Initializing state for $MaxInitializationChecks checks" -ForegroundColor Red
            Write-Host "Killing job and showing logs..." -ForegroundColor Yellow
            Write-Host ""
            
            try {
                kubectl delete rayjob $JobName 2>$null | Out-Null
            } catch {}
            
            $jobComplete = $true
            break
        }
    } else {
        $initializationChecks = 0  # Reset if not initializing
    }
    
    # Show worker pods if job is running
    if ($deployStatus -eq "Running") {
        Write-Host "  Checking worker pods..." -ForegroundColor White
        try {
            $pods = kubectl get pods -l ray.io/job-name=$JobName --no-headers 2>&1 | Where-Object { $_ -notmatch "No resources found" }
            if ($pods) {
                $podCount = @($pods).Count
                $running = ($pods | Select-String "Running" | Measure-Object).Count
                $pending = ($pods | Select-String "Pending" | Measure-Object).Count
                
                Write-Host "    Pods - Total: $podCount, Running: $running, Pending: $pending" -ForegroundColor Cyan
            }
        } catch {}
    }
    
    # Check if job is complete or failed
    if ($deployStatus -eq "Complete") {
        Write-Host ""
        Write-Host "JOB COMPLETED SUCCESSFULLY" -ForegroundColor Green
        $jobComplete = $true
    } elseif ($deployStatus -eq "Failed") {
        Write-Host ""
        Write-Host "JOB FAILED" -ForegroundColor Red
        $jobComplete = $true
    }
    
    if (-not $jobComplete) {
        Write-Host ""
        Start-Sleep -Seconds $CheckInterval
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

# Calculate final timing
$finalTime = (Get-Date) - $startTime
$totalMinutes = [int]($finalTime.TotalSeconds / 60)
$totalSeconds = [int]($finalTime.TotalSeconds % 60)

Write-Host "Total Time: $($totalMinutes)m $($totalSeconds)s" -ForegroundColor White
Write-Host ""

# Get final status
$finalStatus = kubectl get rayjob $JobName -o yaml 2>$null
if ($finalStatus) {
    $finalDeployStatus = ($finalStatus | Select-String "jobDeploymentStatus:" | Select-Object -First 1) -split ':' | Select-Object -Last 1 | ForEach-Object { $_.Trim() }
    $finalSucceeded = ($finalStatus | Select-String "succeeded:" | Select-Object -First 1) -split ':' | Select-Object -Last 1 | ForEach-Object { $_.Trim() }
    $finalFailed = ($finalStatus | Select-String "failed:" | Select-Object -First 1) -split ':' | Select-Object -Last 1 | ForEach-Object { $_.Trim() }
    
    Write-Host "Final Status: $finalDeployStatus" -ForegroundColor Cyan
    if ($finalSucceeded) { Write-Host "Completed Tasks: $finalSucceeded" -ForegroundColor Green }
    if ($finalFailed) { Write-Host "Failed Tasks: $finalFailed" -ForegroundColor Red }
}

# Show logs if job failed or couldn't initialize
if ($deployStatus -ne "Complete" -or $initializationChecks -ge $MaxInitializationChecks) {
    Write-Host ""
    Write-Host "========== JOB LOGS ==========" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        kubectl logs -l ray.io/job-name=$JobName --all-containers=true --timestamps=true 2>&1 | Select-Object -Last 100
    } catch {
        Write-Host "Unable to retrieve logs" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
