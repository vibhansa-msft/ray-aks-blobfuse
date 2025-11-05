# Validate pod distribution
Write-Host "====================================="
Write-Host "Pod Distribution Validation"
Write-Host "====================================="
Write-Host ""

# Get JSON output
$pods = kubectl get pods -o json | ConvertFrom-Json

# Filter worker pods
$workers = $pods.items | Where-Object { $_.metadata.name -like "*worker*" }

Write-Host "Total worker pods: $($workers.Count)"
Write-Host ""

# Group by node
$podsByNode = @{}
foreach ($worker in $workers) {
    $nodeName = $worker.spec.nodeName
    $podName = $worker.metadata.name
    
    if (-not $podsByNode.ContainsKey($nodeName)) {
        $podsByNode[$nodeName] = @()
    }
    $podsByNode[$nodeName] += $podName
}

# Display results
Write-Host "Pod Distribution by Node:"
Write-Host "========================="

$violations = 0
foreach ($node in ($podsByNode.Keys | Sort-Object)) {
    $count = $podsByNode[$node].Count
    $status = if ($count -eq 1) { "OK" } else { "VIOLATION" }
    Write-Host "$node : $count pod(s) [$status]"
    if ($count -ne 1) {
        $violations++
        foreach ($pod in $podsByNode[$node]) {
            Write-Host "  - $pod"
        }
    }
}

Write-Host ""
Write-Host "====================================="
if ($violations -eq 0) {
    Write-Host "PASS: Exactly 1 worker per node"
    Write-Host "Nodes used: $($podsByNode.Count) / 25"
} else {
    Write-Host "FAIL: $violations node(s) with multiple workers"
}
Write-Host "====================================="
