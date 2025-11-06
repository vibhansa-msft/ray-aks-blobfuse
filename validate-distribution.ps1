# Import utility functions and load parameters
. "$PSScriptRoot\utils.ps1"
Load-EnvParameters

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

foreach ($node in ($podsByNode.Keys | Sort-Object)) {
    $count = $podsByNode[$node].Count
    Write-Host "$node : $count pod(s)"
    foreach ($pod in $podsByNode[$node]) {
        Write-Host "  - $pod"
    }
}

Write-Host ""
Write-Host "====================================="
Write-Host "Nodes used: $($podsByNode.Count) / $NodeCount"
Write-Host "====================================="
