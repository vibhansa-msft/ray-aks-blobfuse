# Ray Job Monitoring Script
# Continuously checks Ray job status every 30 seconds until completion

param(
    [string]$JobName = "hpo-job-gpu",
    [int]$CheckInterval = 30,
    [int]$MaxWaitMinutes = 120
)

Write-Host "========== RAY JOB MONITORING ==========" -ForegroundColor Cyan
Write-Host "Job Name: $JobName" -ForegroundColor Green
Write-Host "Check Interval: $CheckInterval seconds" -ForegroundColor Green
Write-Host "Max Wait Time: $MaxWaitMinutes minutes" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$maxWaitSeconds = $MaxWaitMinutes * 60
$startTime = Get-Date
$jobComplete = $false
$checkCount = 0
$failureCount = 0
$maxFailures = 3

Write-Host "Starting job monitoring at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
Write-Host ""

# Initial check to verify job exists
Write-Host "Verifying Ray job exists..." -ForegroundColor Yellow
$initialCheck = kubectl get rayjob $JobName -o yaml 2>$null
if (-not $initialCheck) {
    Write-Host ""
    Write-Host "ERROR: Unable to retrieve Ray job details" -ForegroundColor Red
    Write-Host "Job name: $JobName" -ForegroundColor White
    Write-Host ""
    Write-Host "Possible reasons:" -ForegroundColor Yellow
    Write-Host "  1. Job does not exist - verify with: kubectl get rayjob" -ForegroundColor Gray
    Write-Host "  2. kubeconfig is not configured - check: kubectl get nodes" -ForegroundColor Gray
    Write-Host "  3. Kubernetes cluster is not accessible" -ForegroundColor Gray
    Write-Host ""
    exit 1
}

Write-Host "Ray job found. Starting continuous monitoring..." -ForegroundColor Green
Write-Host ""

while (-not $jobComplete) {
    $checkCount++
    $elapsedTime = (Get-Date) - $startTime
    $elapsedSeconds = [int]$elapsedTime.TotalSeconds
    $elapsedMinutes = [int]($elapsedSeconds / 60)
    
    Write-Host "[$($elapsedMinutes)m $($elapsedSeconds % 60)s] Check #$checkCount" -ForegroundColor Cyan
    
    # Get RayJob status
    $rayJobOutput = kubectl get rayjob $JobName -o yaml 2>$null
    
    if (-not $rayJobOutput) {
        $failureCount++
        Write-Host "  ERROR: Could not retrieve RayJob status" -ForegroundColor Red
        Write-Host "  Consecutive failures: $failureCount / $maxFailures" -ForegroundColor Yellow
        
        if ($failureCount -ge $maxFailures) {
            Write-Host ""
            Write-Host "FATAL: Unable to retrieve job status after $maxFailures consecutive attempts" -ForegroundColor Red
            Write-Host "Stopping monitoring..." -ForegroundColor Yellow
            $jobComplete = $true
            break
        }
    } else {
        $failureCount = 0  # Reset failure counter on successful retrieval
    }
    
    if ($rayJobOutput) {
        # Extract and display deployment status
        $deployStatusLine = $rayJobOutput | Select-String "jobDeploymentStatus:" | Select-Object -First 1
        if ($deployStatusLine) {
            $deployStatus = ($deployStatusLine -split ':')[1].Trim()
            Write-Host "  Deployment Status: $deployStatus" -ForegroundColor White
        }
        
        # ===== CHECK IF JOB IS STUCK IN INITIALIZING =====
        # After 25 checks (~12 minutes), if still initializing, kill all RayJobs
        if ($deployStatus -eq "Initializing" -and $checkCount -eq 25) {
            Write-Host ""
            Write-Host "WARNING: RayJob still in Initializing state after 25 checks (approx 12 minutes)" -ForegroundColor Red
            Write-Host "This indicates a resource scheduling problem (likely not enough capacity)" -ForegroundColor Yellow
            Write-Host "Killing all RayJobs to free resources..." -ForegroundColor Red
            Write-Host ""
            
            try {
                kubectl delete rayjob --all 2>$null | Out-Null
                Write-Host "All RayJobs deleted successfully" -ForegroundColor Green
                Write-Host ""
                Write-Host "Recommendations:" -ForegroundColor Yellow
                Write-Host "  1. Reduce WORKER_REPLICAS in .env.example" -ForegroundColor Gray
                Write-Host "  2. Reduce NUM_WORKERS in .env.example" -ForegroundColor Gray
                Write-Host "  3. Increase NodeCount in .env.example" -ForegroundColor Gray
                Write-Host "  4. Check node resource availability: kubectl top nodes" -ForegroundColor Gray
                Write-Host ""
                $jobComplete = $true
                break
            } catch {
                Write-Host "Failed to delete RayJobs: $_" -ForegroundColor Red
                $jobComplete = $true
                break
            }
        }
        
        # Extract and display counts
        $succeededLine = $rayJobOutput | Select-String "succeeded:" | Select-Object -First 1
        $failedLine = $rayJobOutput | Select-String "failed:" | Select-Object -First 1
        
        if ($succeededLine) {
            $succeededCount = ($succeededLine -split ':')[1].Trim()
            Write-Host "  Succeeded: $succeededCount" -ForegroundColor Green
        }
        if ($failedLine) {
            $failedCount = ($failedLine -split ':')[1].Trim()
            Write-Host "  Failed: $failedCount" -ForegroundColor Yellow
        }
        
        # Check pod status
        Write-Host "  Checking Ray cluster pods..." -ForegroundColor White
        try {
            $pods = kubectl get pods -l ray.io/job-name=$JobName --no-headers 2>&1 | Where-Object { $_ -notmatch "No resources found" }
            if ($pods) {
                $podArray = @($pods)
                $totalPods = $podArray.Count
                Write-Host "    Total: $totalPods pods" -ForegroundColor White
                
                $running = ($pods | Select-String "Running" | Measure-Object).Count
                $pending = ($pods | Select-String "Pending" | Measure-Object).Count
                $failed = ($pods | Select-String "Failed" | Measure-Object).Count
                $succeeded = ($pods | Select-String "Succeeded" | Measure-Object).Count
                
                if ($running -gt 0) { Write-Host "    Running: $running" -ForegroundColor Green }
                if ($pending -gt 0) { Write-Host "    Pending: $pending" -ForegroundColor Yellow }
                if ($failed -gt 0) { Write-Host "    Failed: $failed" -ForegroundColor Red }
                if ($succeeded -gt 0) { Write-Host "    Succeeded: $succeeded" -ForegroundColor Green }
            } else {
                Write-Host "    No pods found yet (job still initializing)" -ForegroundColor Gray
            }
        } catch {
            Write-Host "    Unable to fetch pod status" -ForegroundColor Gray
        }
        
        # Check if job is complete
        if ($deployStatus -eq "Complete" -or $deployStatus -eq "Failed") {
            $jobComplete = $true
            Write-Host ""
            if ($deployStatus -eq "Complete") {
                Write-Host "JOB COMPLETED SUCCESSFULLY" -ForegroundColor Green
            } else {
                Write-Host "JOB FAILED" -ForegroundColor Red
            }
        }
    } else {
        Write-Host "  Could not retrieve RayJob status" -ForegroundColor Yellow
    }
    
    # Check timeout
    if ($elapsedSeconds -ge $maxWaitSeconds) {
        Write-Host ""
        Write-Host "TIMEOUT: Max wait time reached" -ForegroundColor Yellow
        $jobComplete = $true
    } else {
        if (-not $jobComplete) {
            Write-Host ""
            Start-Sleep -Seconds $CheckInterval
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Monitoring Complete" -ForegroundColor Green
Write-Host "Total elapsed time: $([int]$elapsedTime.TotalMinutes)m $($elapsedTime.Seconds)s" -ForegroundColor White
Write-Host ""

# Final status
Write-Host "Final Job Status:" -ForegroundColor Cyan
$finalStatus = kubectl get rayjob $JobName -o yaml 2>$null | Select-String "jobDeploymentStatus:|succeeded:|failed:"
if ($finalStatus) {
    $finalStatus
} else {
    Write-Host "Unable to retrieve final job status" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Job may have been deleted or cluster is unreachable." -ForegroundColor Gray
    Write-Host "Use these commands to verify:" -ForegroundColor White
    Write-Host "  kubectl get rayjob" -ForegroundColor Gray
    Write-Host "  kubectl get nodes" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Useful Commands:" -ForegroundColor Yellow
Write-Host "  kubectl get rayjob $JobName -o yaml" -ForegroundColor Gray
Write-Host "  kubectl logs -l ray.io/job-name=$JobName -f" -ForegroundColor Gray
Write-Host "  kubectl get pods -l ray.io/job-name=$JobName" -ForegroundColor Gray
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
