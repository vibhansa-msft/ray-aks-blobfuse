# Script to run the Anyscale customer-hosted Azure/K8s cloud setup flow using values from .env.example.
# - Loads .env.example for subscription, resource group, cluster, storage, and Anyscale settings.
# - Ensures Azure context and kubeconfig are set for the AKS cluster.
# - Runs `anyscale cloud setup --stack k8s` (creates the cloud and emits Helm values).
# - Optionally installs/updates the Anyscale operator via Helm using the generated values file.
# - Optionally runs `anyscale cloud register` to attach PVC + storage metadata (only if requested).

param(
    [string]$EnvFile = ".env.example",
    [string]$PersistentVolumeClaim = "blob-pvc-checkpoint",
    [string]$KubernetesZones = "1,2,3",
    [string]$OperatorIdentity = "",  # Optional: federated identity for the Anyscale operator service account
    [string]$Namespace = "anyscale-system",
    [string]$ValuesFile = "anyscale-operator-values.yaml",
    [string]$HelmRepo = "https://charts.anyscale.com",
    [string]$HelmRelease = "anyscale-operator",
    [string]$CloudDeploymentId = "",
    [string]$ComputeConfigFile = "anyscale-compute-config.yaml",
    [string]$ComputeConfigName = "",
    [string]$OperatorValuesFile = "anyscale-operator-config.yaml",
    [string]$CpuValuesFile = "anyscale-cpu-config.yaml",
    [string]$DoSetup = "false",          # anyscale cloud setup (flaky on Windows CLI; default off)
    [string]$InstallOperator = "true",
    [string]$RegisterCloud = "true"      # default on so users get a usable cloud
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Load-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) { throw "Env file not found: $Path" }
    $vars = @{}
    Get-Content $Path | ForEach-Object {
        if ($_ -match "^([A-Z_]+)=(.*)$") {
            $name = $matches[1]
            $value = $matches[2] -replace '^"|"$', ''
            $vars[$name] = $value
        }
    }
    return $vars
}

function Require-Var {
    param([hashtable]$Vars, [string]$Name)
    if (-not $Vars[$Name]) { throw "Missing $Name in $EnvFile" }
    return $Vars[$Name]
}

function Get-CloudResourceId {
    param([string]$CloudName)
    $anyscaleExe = "anyscale"
    $venvExe = Join-Path $PSScriptRoot ".anyscale-venv\Scripts\anyscale.exe"
    if (Test-Path $venvExe) { $anyscaleExe = $venvExe }
    try {
        $info = & $anyscaleExe cloud get -n $CloudName 2>&1
        $matches = ($info | Select-String -Pattern "cloud_resource_id:\s*([a-zA-Z0-9_]+)" -AllMatches).Matches
        if ($matches) { return $matches[0].Groups[1].Value }
    } catch {}
    return $null
}

function Get-HelmExe {
    $helmExe = "$env:USERPROFILE\\.azure-helm\\helm.exe"
    if (Test-Path $helmExe) { return $helmExe }
    return "helm"
}

function Normalize-Bool {
    param([string]$Value)
    if (-not $Value) { return $false }
    $v = $Value.ToString()
    # Strip inline comments and surrounding quotes/whitespace
    $v = ($v -split '#')[0].Trim().Trim('"', "'")
    return ($v.ToLower() -in @("1","true","yes","y","on"))
}

function Parse-ComputeConfigList {
    param([string[]]$Lines, [string]$Cloud)
    $results = @()
    foreach ($line in $Lines) {
        $match = [regex]::Match($line, "^(cpt_[a-z0-9]+)\s+([^\s]+)\s+$Cloud\b", 'IgnoreCase')
        if ($match.Success) {
            $results += [pscustomobject]@{ Id = $match.Groups[1].Value; Name = $match.Groups[2].Value }
        }
    }
    return $results
}

$envVars = Load-DotEnv -Path $EnvFile

$subscription = Require-Var $envVars "SUBSCRIPTION"
$resourceGroup = Require-Var $envVars "RG"
$location = Require-Var $envVars "LOC"
$aksName = Require-Var $envVars "AKS"
$storageAccount = Require-Var $envVars "SA"
$container = Require-Var $envVars "CONTAINER"
$anyscaleToken = Require-Var $envVars "ANYSCALE_CLI_TOKEN"
$cloudName = Require-Var $envVars "ANYSCALE_CLOUD_NAME"

# Optional overrides from .env.example
if ($envVars["ANYSCALE_NAMESPACE"]) { $Namespace = $envVars["ANYSCALE_NAMESPACE"] }
if ($envVars["ANYSCALE_VALUES_FILE"]) { $ValuesFile = $envVars["ANYSCALE_VALUES_FILE"] }
if ($envVars["ANYSCALE_HELM_REPO"]) { $HelmRepo = $envVars["ANYSCALE_HELM_REPO"] }
if ($envVars["ANYSCALE_HELM_RELEASE"]) { $HelmRelease = $envVars["ANYSCALE_HELM_RELEASE"] }
if ($envVars["ANYSCALE_PVC"]) { $PersistentVolumeClaim = $envVars["ANYSCALE_PVC"] }
if ($envVars["ANYSCALE_K8S_ZONES"]) { $KubernetesZones = $envVars["ANYSCALE_K8S_ZONES"] }
if ($envVars["ANYSCALE_OPERATOR_IDENTITY"]) { $OperatorIdentity = $envVars["ANYSCALE_OPERATOR_IDENTITY"] }
if ($envVars["ANYSCALE_INSTALL_OPERATOR"]) {
    $InstallOperator = $envVars["ANYSCALE_INSTALL_OPERATOR"]
}
if ($envVars["ANYSCALE_REGISTER_CLOUD"]) {
    $RegisterCloud = $envVars["ANYSCALE_REGISTER_CLOUD"]
}
if ($envVars["ANYSCALE_DO_SETUP"]) {
    $DoSetup = $envVars["ANYSCALE_DO_SETUP"]
}
if ($envVars["ANYSCALE_CLOUD_DEPLOYMENT_ID"]) { $CloudDeploymentId = $envVars["ANYSCALE_CLOUD_DEPLOYMENT_ID"] }
if ($envVars["ANYSCALE_COMPUTE_CONFIG_FILE"]) { $ComputeConfigFile = $envVars["ANYSCALE_COMPUTE_CONFIG_FILE"] }
if ($envVars["ANYSCALE_OPERATOR_VALUES_FILE"]) { $OperatorValuesFile = $envVars["ANYSCALE_OPERATOR_VALUES_FILE"] }
if ($envVars["ANYSCALE_CPU_VALUES_FILE"]) { $CpuValuesFile = $envVars["ANYSCALE_CPU_VALUES_FILE"] }
if ($envVars["ANYSCALE_COMPUTE_CONFIG_NAME"]) { $ComputeConfigName = $envVars["ANYSCALE_COMPUTE_CONFIG_NAME"] }

# Normalize string toggles to bools
$DoSetupFlag = Normalize-Bool $DoSetup
$InstallOperatorFlag = Normalize-Bool $InstallOperator
$RegisterCloudFlag = Normalize-Bool $RegisterCloud
$cloudId = $CloudDeploymentId

if (Test-Path "./utils.ps1") {
    . ./utils.ps1
    Need "kubectl"
    Need "helm"
}

# Ensure expected tool locations are on PATH for the current process (some CLIs are installed outside the default PATH)
$extraPaths = @(
    "$env:USERPROFILE\.azure-helm",
    "$env:USERPROFILE\.azure-kubectl",
    "C:\\Program Files\\Docker\\Docker\\resources\\bin",
    "C:\\Program Files\\Microsoft SDKs\\Azure\\CLI2\\wbin"
)
foreach ($p in $extraPaths) {
    if ($p -and (Test-Path $p) -and ($env:PATH -notlike "*$p*")) { $env:PATH = "$p;" + $env:PATH }
}

function Assert-Tool {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Required CLI missing or not on PATH: $Name" }
    return $cmd.Source
}

Write-Host "[CHECK] Verifying CLI tools are reachable on PATH" -ForegroundColor Cyan
$kubectlPath = Assert-Tool "kubectl"
$helmPath = Assert-Tool "helm"
$azPath = Assert-Tool "az"
Write-Host "[CHECK] kubectl: $kubectlPath" -ForegroundColor Gray
Write-Host "[CHECK] helm:    $helmPath" -ForegroundColor Gray
Write-Host "[CHECK] az:      $azPath" -ForegroundColor Gray

Write-Host "[AZ] Setting subscription $subscription" -ForegroundColor Cyan
az account set --subscription $subscription | Out-Null

Write-Host "[AZ] Fetching tenant ID" -ForegroundColor Cyan
$tenantId = az account show --query tenantId -o tsv

Write-Host "[AZ] Getting kubeconfig for $aksName" -ForegroundColor Cyan
az aks get-credentials --resource-group $resourceGroup --name $aksName --overwrite-existing | Out-Null

$bucketName = "abfss://$container@$storageAccount.dfs.core.windows.net"
$bucketUrl = "https://$storageAccount.blob.core.windows.net"
Write-Host "[INFO] Using cloud storage bucket $bucketUrl" -ForegroundColor Green

Write-Host "[ANYSCALE] Setting CLI token from .env (no auth login command in this CLI version)" -ForegroundColor Cyan
$env:ANYSCALE_CLI_TOKEN = $anyscaleToken

Write-Host "[FLAGS] DoSetupRaw=$DoSetup InstallOperatorRaw=$InstallOperator RegisterCloudRaw=$RegisterCloud" -ForegroundColor Gray
Write-Host "[FLAGS] DoSetup=$DoSetupFlag InstallOperator=$InstallOperatorFlag RegisterCloud=$RegisterCloudFlag" -ForegroundColor Gray

$setupArgs = @(
    "cloud", "setup",
    "--provider", "azure",
    "--stack", "k8s",
    "--name", $cloudName,
    "--region", $location,
    "--cluster-name", $aksName,
    "--resource-group", $resourceGroup,
    "--namespace", $Namespace,
    "--values-file", $ValuesFile,
    "--yes"
)

if ($DoSetupFlag) {
    Write-Host "[ANYSCALE] Running cloud setup for '$cloudName'" -ForegroundColor Cyan
    Write-Host "anyscale $($setupArgs -join ' ')" -ForegroundColor Gray
    try {
        anyscale @setupArgs
    } catch {
        Write-Host "[WARN] anyscale cloud setup failed (continuing to register): $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "[SKIP] Skipping anyscale cloud setup (DoSetup=false)" -ForegroundColor Yellow
}

if ($RegisterCloudFlag) {
    $registerArgs = @(
        "cloud", "register",
        "--provider", "azure",
        "--region", $location,
        "--name", $cloudName,
        "--compute-stack", "k8s",
        # Register bucket name without scheme so the CLI handles Azure correctly
        "--cloud-storage-bucket-name", $bucketName,
        "--cloud-storage-bucket-region", $location,
        "--azure-tenant-id", $tenantId,
        "--persistent-volume-claim", $PersistentVolumeClaim,
        "--kubernetes-zones", $KubernetesZones,
        "--yes"
    )

    if ($OperatorIdentity) {
        $registerArgs += @("--anyscale-operator-iam-identity", $OperatorIdentity)
    }

    Write-Host "[ANYSCALE] Registering cloud '$cloudName' (PVC/storage)" -ForegroundColor Cyan
    Write-Host "anyscale $($registerArgs -join ' ')" -ForegroundColor Gray
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $registerOutput = & anyscale @registerArgs 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    $registerOutput | ForEach-Object { Write-Host $_ }

    $matchResult = $registerOutput | Select-String -Pattern "cloudDeploymentId=([a-zA-Z0-9_]+)" -AllMatches
    if ($matchResult) {
        $cloudId = $matchResult.Matches | ForEach-Object { $_.Groups[1].Value } | Select-Object -First 1
    }

    $alreadyExists = ($registerOutput -join " `n") -match "already exists"
    if (-not $cloudId -and ($exitCode -ne 0 -or $alreadyExists)) {
        Write-Host "[WARN] Register returned exit code $exitCode (exists=$alreadyExists); attempting to look up existing cloud resource id..." -ForegroundColor Yellow
        $cloudId = Get-CloudResourceId $cloudName
    }

    if (-not $cloudId -and $exitCode -ne 0 -and -not $alreadyExists) {
        throw "anyscale cloud register failed with exit code $exitCode and no resource id detected"
    }
    if (-not $cloudId) {
        Write-Host "[INFO] No cloudDeploymentId parsed; attempting lookup via anyscale cloud get..." -ForegroundColor Yellow
        $cloudId = Get-CloudResourceId $cloudName
    }

    if (-not $cloudId) {
        Write-Host "[INFO] No cloudDeploymentId parsed; attempting lookup via anyscale cloud get..." -ForegroundColor Yellow
        $cloudId = Get-CloudResourceId $cloudName
    }

    if ($cloudId) {
        $helmCmdLines = @(
            "helm upgrade anyscale-operator anyscale/anyscale-operator",
            "  --set-string global.cloudDeploymentId=$cloudId",
            "  --set-string global.cloudProvider=azure",
            "  --set-string global.auth.audience=api://086bc555-6989-4362-ba30-fded273e432b/.default",
            "  --set-string workloads.serviceAccount.name=anyscale-operator",
            "  --namespace $Namespace",
            "  --create-namespace",
            "  --wait",
            "  -i"
        )
        $helmCmd = $helmCmdLines -join "`n"
        Write-Host "[INFO] Detected cloudDeploymentId=$cloudId" -ForegroundColor Green
        Write-Host "[INFO] Helm install command (requires charts.anyscale.com DNS/connectivity):" -ForegroundColor Yellow
        Write-Host $helmCmd -ForegroundColor Gray
        $helmCmd | Out-File -FilePath "anyscale-helm-install.txt" -Encoding ASCII
        Write-Host "[INFO] Saved to anyscale-helm-install.txt" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Could not detect cloudDeploymentId from register output; check CLI output above." -ForegroundColor Yellow
    }
}

if ($InstallOperatorFlag) {
    $helmExe = Get-HelmExe
    Write-Host "[HELM] Adding/updating repo and installing operator (using $helmExe)" -ForegroundColor Cyan
    if (-not $cloudId) {
        Write-Host "[INFO] Resolving cloudDeploymentId for Helm install..." -ForegroundColor Gray
        $cloudId = Get-CloudResourceId $cloudName
    }
    if (-not $cloudId) {
        Write-Host "[WARN] No cloudDeploymentId available; operator install will continue but instance types will not register without it." -ForegroundColor Yellow
    } else {
        Write-Host "[INFO] Using cloudDeploymentId=$cloudId for operator install" -ForegroundColor Green
    }
    try {
        & $helmExe repo add anyscale $HelmRepo --force-update
        & $helmExe repo update
        $helmArgs = @(
            "upgrade", "--install", $HelmRelease, "anyscale/anyscale-operator",
            "-n", $Namespace,
            "--create-namespace",
            "--version", "1.3.2"
        )

        if (Test-Path $OperatorValuesFile) {
            $helmArgs += @("-f", $OperatorValuesFile)
        } elseif (Test-Path $ValuesFile) {
            $helmArgs += @("-f", $ValuesFile)
        } else {
            Write-Host "[WARN] Values file '$OperatorValuesFile' (or fallback '$ValuesFile') not found; proceeding with inline settings." -ForegroundColor Yellow
        }

        if ($cloudId) {
            $helmArgs += @(
                "--set-string", "global.cloudDeploymentId=$cloudId",
                "--set-string", "global.cloudProvider=azure",
                "--set-string", "global.auth.audience=api://086bc555-6989-4362-ba30-fded273e432b/.default",
                "--set-string", "workloads.serviceAccount.name=anyscale-operator"
            )
        }

        & $helmExe @helmArgs

        # Apply CPU-only overrides if file present (reuse values)
        $cpuValuesPath = Join-Path $PSScriptRoot $CpuValuesFile
        if (Test-Path $cpuValuesPath) {
            Write-Host "[HELM] Applying CPU overrides from $CpuValuesFile with --reuse-values" -ForegroundColor Cyan
            & $helmExe upgrade $HelmRelease anyscale/anyscale-operator -n $Namespace -f $cpuValuesPath --reuse-values --install
        } else {
            Write-Host "[WARN] CPU values file '$CpuValuesFile' not found; skipping CPU override Helm upgrade." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[WARN] Helm operator install failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Create compute config (best effort) and emit sample job command
$computePath = Join-Path $PSScriptRoot $ComputeConfigFile
if (Test-Path $computePath) {
    Write-Host "[ANYSCALE] Creating compute config from $ComputeConfigFile" -ForegroundColor Cyan
    if (-not $ComputeConfigName) { $ComputeConfigName = "$cloudName-compute" }
    $createArgs = @("compute-config", "create", "-n", $ComputeConfigName, $computePath)
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $createOutput = & anyscale @createArgs 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    $createOutput | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) {
        Write-Host "[WARN] anyscale compute-config create returned exit code $exitCode (name=$ComputeConfigName)" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARN] Compute config file '$ComputeConfigFile' not found; skipping compute-config create." -ForegroundColor Yellow
}

# Always list and surface the latest compute config for copy/paste
$listOutput = @()
try {
    $listOutput = & anyscale compute-config list --cloud-name $cloudName 2>&1
    $listOutput | ForEach-Object { Write-Host $_ }
} catch {
    Write-Host "[WARN] Unable to list compute configs: $($_.Exception.Message)" -ForegroundColor Yellow
}

$configs = @()
if ($listOutput) { $configs = @(Parse-ComputeConfigList -Lines $listOutput -Cloud $cloudName) }
if ($configs.Count -gt 0) {
    $latest = $configs[-1]
    Write-Host "[INFO] Latest compute config detected: name=$($latest.Name) id=$($latest.Id)" -ForegroundColor Green
    $jobCmd = "anyscale job submit --compute-config $($latest.Name) --cloud $cloudName --working-dir . -- python app/train_new.py"
    Write-Host "[INFO] Sample job submit command (copy/paste then swap entrypoint if needed):" -ForegroundColor Yellow
    Write-Host "       $jobCmd" -ForegroundColor Gray
    $jobCmd | Out-File -FilePath "anyscale-job-submit.txt" -Encoding ASCII
    Write-Host "[INFO] Saved sample to anyscale-job-submit.txt" -ForegroundColor Green
} else {
    Write-Host "[WARN] No compute configs found for cloud '$cloudName'." -ForegroundColor Yellow
}

Write-Host "[DONE] Flow finished." -ForegroundColor Green