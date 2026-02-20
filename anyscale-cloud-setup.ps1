# Script to run the Anyscale customer-hosted Azure/K8s cloud setup flow using values from .env.example.
# - Loads .env.example for subscription, resource group, cluster, storage, and Anyscale settings.
# - Ensures Azure context and kubeconfig are set for the AKS cluster.
# - Runs `anyscale cloud setup --stack k8s` (creates the cloud and emits Helm values).
# - Optionally installs/updates the Anyscale operator via Helm using the generated values file.
# - Optionally runs `anyscale cloud register` to attach PVC + storage metadata (only if requested).

# powershell -ExecutionPolicy Bypass -File .\anyscale-cloud-setup.ps1 -DoSetup true

param(
    [string]$EnvFile = ".env.example",
    [string]$PersistentVolumeClaim = "blob-pvc-checkpoint",
    [string]$KubernetesZones = "1,2,3",
    [string]$OperatorIdentity = "",  # Optional: federated identity for the Anyscale operator service account
    [string]$Namespace = "anyscale-system",
    [string]$ServiceAccountName = "anyscale-operator",
    [string]$WorkspaceServiceAccountName = "default",
    [string]$ValuesFile = "anyscale-operator-values.yaml",
    [string]$HelmRepo = "https://charts.anyscale.com",
    [string]$HelmRelease = "anyscale-operator",
    [string]$CloudDeploymentId = "",
    [string]$ComputeConfigFile = "anyscale-compute-config.yaml",
    [string]$ComputeConfigName = "",
    [string]$OperatorValuesFile = "anyscale-operator-config.yaml",
    [string]$CpuValuesFile = "anyscale-cpu-config.yaml",
    [string]$OperatorChartVersion = "1.4.0",
    [string]$NewCloudName = "",          # optional: create/register under a different cloud name
    [string]$DeleteExistingCloud = "false", # optional: delete an existing cloud before creating
    [string]$DeleteCloudName = "",       # optional: explicit cloud name to delete (defaults to original env name)
    [string]$DoSetup = "false",          # anyscale cloud setup (flaky on Windows CLI; default off)
    [string]$SetupFailureMode = "auto",  # auto: ignore known Windows CLI false-negative only; strict: fail; ignore: always continue
    [string]$StrictLifecycle = "true",   # true: block on delete/register/cloud-id failures
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
            $value = $matches[2].Trim()
            if ($value -match '^"(.*?)"\s*(#.*)?$') {
                $value = $matches[1]
            } elseif ($value -match "^'(.*?)'\s*(#.*)?$") {
                $value = $matches[1]
            } else {
                $value = ($value -split '#')[0].Trim()
            }
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

function Parse-CloudDeploymentIdFromText {
    param([string[]]$Lines)
    if (-not $Lines) { return $null }
    $text = ($Lines -join "`n")
    $patterns = @(
        "cloudDeploymentId\s*[=:]\s*([a-zA-Z0-9_]+)",
        "cloud_resource_id\s*[=:]\s*([a-zA-Z0-9_]+)",
        "\b(cldrsrc_[a-zA-Z0-9]+)\b"
    )
    foreach ($pattern in $patterns) {
        $match = [regex]::Match($text, $pattern, 'IgnoreCase')
        if ($match.Success) { return $match.Groups[1].Value }
    }
    return $null
}

function Get-CloudResourceId {
    param([string]$CloudName)
    $venvExe = Join-Path $PSScriptRoot ".anyscale-venv\Scripts\anyscale.exe"

    $candidates = @()
    if (Test-Path $venvExe) { $candidates += $venvExe }
    $candidates += "anyscale"

    foreach ($anyscaleExe in $candidates) {
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $info = & $anyscaleExe cloud get -n $CloudName 2>&1
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP

        $parsed = Parse-CloudDeploymentIdFromText -Lines $info
        if ($parsed) {
            Write-Host "[INFO] Resolved cloudDeploymentId via '$anyscaleExe'" -ForegroundColor Gray
            return $parsed
        }

        if ($exitCode -ne 0) {
            Write-Host "[WARN] '$anyscaleExe cloud get -n $CloudName' returned exit code $exitCode" -ForegroundColor Yellow
        } else {
            Write-Host "[WARN] Unable to parse cloudDeploymentId from '$anyscaleExe cloud get -n $CloudName' output." -ForegroundColor Yellow
        }
    }

    return $null
}

function Get-HelmExe {
    $helmExe = "$env:USERPROFILE\\.azure-helm\\helm.exe"
    if (Test-Path $helmExe) { return $helmExe }
    return "helm"
}

function Persist-OperatorCloudDeploymentId {
    param(
        [string]$FilePath,
        [string]$CloudId,
        [bool]$Strict = $true
    )

    if (-not $CloudId) {
        if ($Strict) { throw "Cannot persist empty cloudDeploymentId." }
        Write-Host "[WARN] cloudDeploymentId is empty; skipping operator values update." -ForegroundColor Yellow
        return
    }

    if (-not (Test-Path $FilePath)) {
        if ($Strict) { throw "Operator values file '$FilePath' not found; cannot persist cloudDeploymentId." }
        Write-Host "[WARN] Operator values file '$FilePath' not found; skipping cloudDeploymentId update." -ForegroundColor Yellow
        return
    }

    $current = Get-Content $FilePath -Raw
    $updated = $current
    if ($current -match "(?m)^\s*cloudDeploymentId:") {
        $updated = $current -replace "(?m)^\s*cloudDeploymentId:.*$", "  cloudDeploymentId: $CloudId"
    } elseif ($current -match "(?m)^global:\s*$") {
        $updated = $current -replace "(?m)^global:\s*$", "global:`n  cloudDeploymentId: $CloudId"
    } else {
        $updated = "global:`n  cloudDeploymentId: $CloudId`n" + $current
    }

    Set-Content -Path $FilePath -Value $updated -Encoding ASCII
    $verify = Get-Content $FilePath -Raw
    if ($verify -notmatch [regex]::Escape("cloudDeploymentId: $CloudId")) {
        throw "Verification failed after updating cloudDeploymentId in '$FilePath'."
    }
}

function Normalize-Bool {
    param([string]$Value)
    if (-not $Value) { return $false }
    $v = $Value.ToString()
    # Strip inline comments and surrounding quotes/whitespace
    $v = ($v -split '#')[0].Trim().Trim('"', "'")
    return ($v.ToLower() -in @("1","true","yes","y","on"))
}

function Ensure-PvcForNamespace {
    param(
        [string]$Namespace,
        [string]$PvcManifest = "k8s/pvc-checkpoint.yaml",
        [string]$PvcName = "blob-pvc-checkpoint"
    )

    if (-not (Test-Path $PvcManifest)) {
        Write-Host "[PVC] Manifest not found at $PvcManifest; skipping." -ForegroundColor Yellow
        return
    }

    $existing = kubectl get pvc $PvcName -n $Namespace -o name 2>$null
    if ($existing) {
        Write-Host "[PVC] $PvcName already exists in namespace $Namespace" -ForegroundColor Green
        return
    }

    Write-Host "[PVC] Creating $PvcName in namespace $Namespace from $PvcManifest" -ForegroundColor Cyan
    kubectl apply -f $PvcManifest -n $Namespace | Out-Null
}

function Ensure-PatchesConfig {
    param(
        [string]$Namespace = "anyscale-system",
        [string]$ClientId = ""
    )

    if (-not $ClientId) {
        Write-Host "[PATCHES] ClientId not provided; skipping workload identity patch injection." -ForegroundColor Yellow
        return
    }

    $cmJsonRaw = kubectl get configmap patches -n $Namespace -o json 2>$null
    if (-not $cmJsonRaw) {
        Write-Host "[PATCHES] ConfigMap 'patches' not found in $Namespace; skipping." -ForegroundColor Yellow
        return
    }

    $cm = $cmJsonRaw | ConvertFrom-Json
    $patches = $cm.data."patches.yaml"
    if (-not $patches) {
        Write-Host "[PATCHES] patches.yaml missing in ConfigMap; skipping." -ForegroundColor Yellow
        return
    }

    $malformedPattern = '(\s*- op:\s*add\r?\n\s*path:\s*/metadata/annotations/azure\.workload\.identity~1proxy-sidecar-port\r?\n\s*value:\s*"10000"\r?\n\s{8,}- op:\s*add\r?\n\s*path:\s*/metadata/annotations/azure\.workload\.identity~1client-id\r?\n\s*value:\s*"[^"]+"\r?\n)'
    if ($patches -match $malformedPattern) {
        $replacement = "    - op: add`n      path: /metadata/annotations/azure.workload.identity~1proxy-sidecar-port`n      value: `"10000`"`n    - op: add`n      path: /metadata/annotations/azure.workload.identity~1client-id`n      value: `"$ClientId`"`n"
        $patches = [regex]::Replace($patches, $malformedPattern, $replacement, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        $cm.data."patches.yaml" = $patches
        $cm | ConvertTo-Json -Depth 50 | kubectl apply -f - | Out-Null
        Write-Host "[PATCHES] Repaired malformed workload identity client-id patch block." -ForegroundColor Green
        return
    }

    if ($patches -match "azure\.workload\.identity~1client-id") {
        Write-Host "[PATCHES] Workload identity client-id annotation already present." -ForegroundColor Green
        return
    }

    $insertion = "    - op: add`n      path: /metadata/annotations/azure.workload.identity~1client-id`n      value: `"$ClientId`"`n"
    $anchorPattern = "(\s*path:\s*/metadata/annotations/azure\.workload\.identity~1proxy-sidecar-port\r?\n\s*value:\s*`"10000`"\r?\n)"
    $updated = [regex]::Replace($patches, $anchorPattern, "`$1$insertion", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if ($updated -eq $patches) {
        Write-Host "[PATCHES] Could not find proxy-sidecar-port value anchor to insert client-id; skipping." -ForegroundColor Yellow
        return
    }

    $cm.data."patches.yaml" = $updated
    $cm | ConvertTo-Json -Depth 50 | kubectl apply -f - | Out-Null
    Write-Host "[PATCHES] Injected azure.workload.identity/client-id into patches ConfigMap." -ForegroundColor Green
}

function Ensure-NamespaceLabel {
    param(
        [string]$Namespace,
        [string]$Label,
        [string]$Value
    )

    $existing = kubectl get namespace $Namespace -o jsonpath="{.metadata.labels['$Label']}" 2>$null
    if ($existing -and $existing -eq $Value) {
        Write-Host "[NS] Namespace $Namespace already labeled $Label=$Value" -ForegroundColor Green
        return
    }

    Write-Host "[NS] Labeling namespace $Namespace with $Label=$Value" -ForegroundColor Cyan
    kubectl label namespace $Namespace "$Label=$Value" --overwrite | Out-Null
}

function Ensure-ServiceAccountClientId {
    param(
        [string]$Namespace,
        [string]$ServiceAccount,
        [string]$ClientId
    )

    if (-not $ClientId) {
        Write-Host "[SA] ClientId not provided; skipping annotation for $ServiceAccount in $Namespace." -ForegroundColor Yellow
        return
    }

    $saJson = kubectl get sa $ServiceAccount -n $Namespace -o json 2>$null
    if (-not $saJson) {
        Write-Host "[SA] ServiceAccount $ServiceAccount not found in $Namespace; skipping." -ForegroundColor Yellow
        return
    }

    $existing = kubectl get sa $ServiceAccount -n $Namespace -o jsonpath="{.metadata.annotations['azure.workload.identity/client-id']}" 2>$null
    if ($existing -and $existing -eq $ClientId) {
        Write-Host "[SA] $ServiceAccount already annotated with azure.workload.identity/client-id." -ForegroundColor Green
        return
    }

    Write-Host "[SA] Annotating $ServiceAccount in $Namespace with azure.workload.identity/client-id" -ForegroundColor Cyan
    kubectl annotate serviceaccount $ServiceAccount -n $Namespace "azure.workload.identity/client-id=$ClientId" --overwrite | Out-Null
}

function Remove-CloudIfExists {
    param([string]$CloudName)

    if (-not $CloudName) { return $true }
    Write-Host "[ANYSCALE] Checking for existing cloud '$CloudName' to delete" -ForegroundColor Cyan
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $getOutput = & anyscale cloud get -n $CloudName 2>&1
    $getExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($getExit -ne 0) {
        Write-Host "[ANYSCALE] Cloud '$CloudName' not found or get failed (skip delete)." -ForegroundColor Yellow
        return $true
    }

    Write-Host "[ANYSCALE] Deleting cloud '$CloudName'" -ForegroundColor Cyan
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $deleteOutput = & anyscale cloud delete -n $CloudName --yes 2>&1
    $deleteExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    $deleteOutput | ForEach-Object { Write-Host $_ }
    if ($deleteExit -ne 0) {
        Write-Host "[WARN] anyscale cloud delete returned exit code $deleteExit" -ForegroundColor Yellow
        return $false
    } else {
        Write-Host "[INFO] Deleted cloud '$CloudName'" -ForegroundColor Green
        return $true
    }
}

function Resolve-UserAssignedIdentityByClientId {
    param([string]$ClientId)
    $raw = az identity list --query "[?clientId=='$ClientId']" -o json
    if (-not $raw) { return $null }
    $items = $raw | ConvertFrom-Json
    if (-not $items) { return $null }
    return $items[0]
}

function Ensure-FederatedCredential {
    param(
        [pscustomobject]$Identity,
        [string]$Issuer,
        [string]$Subject,
        [string]$Audience = "api://AzureADTokenExchange",
        [string]$Name = "anyscale-operator"
    )

    if (-not $Identity -or -not $Identity.name -or -not $Identity.resourceGroup) {
        throw "Identity metadata incomplete; cannot create federated credential"
    }

    $ficList = az identity federated-credential list --identity-name $Identity.name --resource-group $Identity.resourceGroup | ConvertFrom-Json
    $existing = $ficList | Where-Object { $_.issuer -eq $Issuer -and $_.subject -eq $Subject -and ($_.audiences -contains $Audience) }
    if ($existing) {
        Write-Host "[AZ] Federated credential already present for issuer=$Issuer subject=$Subject" -ForegroundColor Green
        return
    }

    Write-Host "[AZ] Creating federated credential for $($Identity.name) (issuer=$Issuer subject=$Subject)" -ForegroundColor Cyan
    az identity federated-credential create `
        --name $Name `
        --identity-name $Identity.name `
        --resource-group $Identity.resourceGroup `
        --issuer $Issuer `
        --subject $Subject `
        --audiences $Audience | Out-Null
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

function Get-OperatorDeploymentCloudId {
    param(
        [string]$Namespace,
        [string]$DeploymentName = "anyscale-operator"
    )

    $raw = kubectl get deploy $DeploymentName -n $Namespace -o json 2>$null
    if (-not $raw) { return $null }
    $obj = $raw | ConvertFrom-Json
    if (-not $obj.spec.template.spec.containers) { return $null }
    foreach ($container in $obj.spec.template.spec.containers) {
        if (-not $container.args) { continue }
        foreach ($arg in $container.args) {
            if ($arg -match "--cloud-deployment-id=([a-zA-Z0-9_]+)") {
                return $matches[1]
            }
        }
    }
    return $null
}

$envVars = Load-DotEnv -Path $EnvFile

$doSetupOverridden = $PSBoundParameters.ContainsKey("DoSetup")

$subscription = Require-Var $envVars "SUBSCRIPTION"
$resourceGroup = Require-Var $envVars "RG"
$location = Require-Var $envVars "LOC"
$aksName = Require-Var $envVars "AKS"
$storageAccount = Require-Var $envVars "SA"
$container = Require-Var $envVars "CONTAINER"
$anyscaleToken = Require-Var $envVars "ANYSCALE_CLI_TOKEN"
$cloudName = Require-Var $envVars "ANYSCALE_CLOUD_NAME"
$originalCloudName = $cloudName

# Optional overrides from .env.example
if ($envVars["ANYSCALE_NAMESPACE"]) { $Namespace = $envVars["ANYSCALE_NAMESPACE"] }
if ($envVars["ANYSCALE_WORKSPACE_SERVICE_ACCOUNT"]) { $WorkspaceServiceAccountName = $envVars["ANYSCALE_WORKSPACE_SERVICE_ACCOUNT"] }
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
if (-not $doSetupOverridden -and $envVars["ANYSCALE_DO_SETUP"]) {
    $DoSetup = $envVars["ANYSCALE_DO_SETUP"]
}
if ($envVars["ANYSCALE_CLOUD_DEPLOYMENT_ID"]) { $CloudDeploymentId = $envVars["ANYSCALE_CLOUD_DEPLOYMENT_ID"] }
if ($envVars["ANYSCALE_COMPUTE_CONFIG_FILE"]) { $ComputeConfigFile = $envVars["ANYSCALE_COMPUTE_CONFIG_FILE"] }
if ($envVars["ANYSCALE_OPERATOR_VALUES_FILE"]) { $OperatorValuesFile = $envVars["ANYSCALE_OPERATOR_VALUES_FILE"] }
if ($envVars["ANYSCALE_CPU_VALUES_FILE"]) { $CpuValuesFile = $envVars["ANYSCALE_CPU_VALUES_FILE"] }
if ($envVars["ANYSCALE_OPERATOR_CHART_VERSION"]) { $OperatorChartVersion = $envVars["ANYSCALE_OPERATOR_CHART_VERSION"] }
if ($envVars["ANYSCALE_COMPUTE_CONFIG_NAME"]) { $ComputeConfigName = $envVars["ANYSCALE_COMPUTE_CONFIG_NAME"] }
if ($envVars["ANYSCALE_NEW_CLOUD_NAME"]) { $NewCloudName = $envVars["ANYSCALE_NEW_CLOUD_NAME"] }
if ($envVars["ANYSCALE_DELETE_EXISTING_CLOUD"]) { $DeleteExistingCloud = $envVars["ANYSCALE_DELETE_EXISTING_CLOUD"] }
if ($envVars["ANYSCALE_DELETE_CLOUD_NAME"]) { $DeleteCloudName = $envVars["ANYSCALE_DELETE_CLOUD_NAME"] }
if ($envVars["ANYSCALE_STRICT_LIFECYCLE"]) { $StrictLifecycle = $envVars["ANYSCALE_STRICT_LIFECYCLE"] }

if ($NewCloudName) {
    Write-Host "[INFO] Overriding cloud name: $originalCloudName -> $NewCloudName" -ForegroundColor Cyan
    $cloudName = $NewCloudName
}

# Normalize string toggles to bools
$DoSetupFlag = Normalize-Bool $DoSetup
$InstallOperatorFlag = Normalize-Bool $InstallOperator
$RegisterCloudFlag = Normalize-Bool $RegisterCloud
$DeleteExistingCloudFlag = Normalize-Bool $DeleteExistingCloud
$StrictLifecycleFlag = Normalize-Bool $StrictLifecycle
$setupFailureModeNormalized = (($SetupFailureMode -split '#')[0].Trim().Trim('"', "'"))
if (-not $setupFailureModeNormalized) { $setupFailureModeNormalized = "auto" }
$setupFailureModeNormalized = $setupFailureModeNormalized.ToLower()
if ($setupFailureModeNormalized -notin @("auto", "strict", "ignore")) {
    throw "Invalid SetupFailureMode '$SetupFailureMode'. Supported values: auto, strict, ignore"
}
$cloudId = $CloudDeploymentId

# Optionally delete an existing cloud before creating/registering
if ($DeleteExistingCloudFlag) {
    $cloudToDelete = if ($DeleteCloudName) { $DeleteCloudName } elseif ($originalCloudName) { $originalCloudName } else { $cloudName }
    $deleteOk = Remove-CloudIfExists -CloudName $cloudToDelete
    if (-not $deleteOk -and $StrictLifecycleFlag) {
        throw "DeleteExistingCloud=true but cloud '$cloudToDelete' could not be deleted. Aborting due to StrictLifecycle=true."
    }
}

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

# Surface tool paths to anyscale CLI to avoid Windows PATH detection issues
$kubectlDir = Split-Path $kubectlPath -Parent
$helmDir = Split-Path $helmPath -Parent
$azDir = Split-Path $azPath -Parent

foreach ($dir in @($kubectlDir, $helmDir, $azDir)) {
    if ($dir -and ($env:PATH -notlike "*$dir*")) { $env:PATH = "$dir;" + $env:PATH }
}

$env:KUBECTL = $kubectlPath
$env:HELM = $helmPath
$env:AZ = $azPath
$env:KUBECTL_PATH = $kubectlPath
$env:HELM_PATH = $helmPath
$env:AZ_PATH = $azPath

Write-Host "[AZ] Setting subscription $subscription" -ForegroundColor Cyan
az account set --subscription $subscription | Out-Null

Write-Host "[AZ] Fetching tenant ID" -ForegroundColor Cyan
$tenantId = az account show --query tenantId -o tsv

Write-Host "[AZ] Fetching AKS OIDC issuer URL" -ForegroundColor Cyan
$oidcIssuer = az aks show --resource-group $resourceGroup --name $aksName --query "oidcIssuerProfile.issuerUrl" -o tsv
if (-not $oidcIssuer) { throw "Unable to resolve AKS OIDC issuer for workload identity" }
Write-Host "[AZ] OIDC issuer: $oidcIssuer" -ForegroundColor Gray

Write-Host "[AZ] Getting kubeconfig for $aksName" -ForegroundColor Cyan
az aks get-credentials --resource-group $resourceGroup --name $aksName --overwrite-existing | Out-Null

# Prepare namespace/service account for workload identity if clientId provided (after kubeconfig is present)
if ($OperatorIdentity) {
    Ensure-NamespaceLabel -Namespace $Namespace -Label "azure.workload.identity/use" -Value "true"
    Ensure-ServiceAccountClientId -Namespace $Namespace -ServiceAccount "default" -ClientId $OperatorIdentity
}

# Ensure shared PVC exists in the operator namespace to avoid Pending pods on workspace creation
Ensure-PvcForNamespace -Namespace $Namespace -PvcManifest "k8s/pvc-checkpoint.yaml" -PvcName $PersistentVolumeClaim

$bucketName = "abfss://$container@$storageAccount.dfs.core.windows.net"
$bucketEndpoint = "https://$storageAccount.dfs.core.windows.net"
$bucketUrl = $bucketName
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
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $setupOutput = & anyscale @setupArgs 2>&1
    $setupExitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP

    if ($setupExitCode -eq 0) {
        $setupOutput | ForEach-Object { Write-Host $_ }
    } else {
        $setupText = ($setupOutput | Out-String)
        $isKnownWindowsCliFalseNegative = $setupText -match "Missing required CLI tools:\s*kubectl,\s*helm,\s*az"

        if ($setupFailureModeNormalized -eq "ignore") {
            Write-Host "[WARN] anyscale cloud setup returned exit code $setupExitCode, but SetupFailureMode=ignore so continuing." -ForegroundColor Yellow
        } elseif ($setupFailureModeNormalized -eq "auto" -and $isKnownWindowsCliFalseNegative) {
            Write-Host "[WARN] anyscale cloud setup reported missing kubectl/helm/az (known Windows false-negative). Continuing with cloud register + operator install." -ForegroundColor Yellow
        } else {
            $setupOutput | ForEach-Object { Write-Host $_ }
            throw "anyscale cloud setup failed with exit code $setupExitCode (SetupFailureMode=$setupFailureModeNormalized)."
        }
    }
} else {
    Write-Host "[SKIP] Skipping anyscale cloud setup (DoSetup=false)" -ForegroundColor Yellow
}

if ($RegisterCloudFlag) {
    if ($OperatorIdentity) {
        Write-Host "[AZ] Resolving operator identity by clientId $OperatorIdentity" -ForegroundColor Cyan
        $identity = Resolve-UserAssignedIdentityByClientId -ClientId $OperatorIdentity
        if (-not $identity) { throw "Managed identity with clientId $OperatorIdentity not found in subscription $subscription" }
        $operatorSubject = "system:serviceaccount:${Namespace}:${ServiceAccountName}"
        $workspaceSubject = "system:serviceaccount:${Namespace}:${WorkspaceServiceAccountName}"
        Ensure-FederatedCredential -Identity $identity -Issuer $oidcIssuer -Subject $operatorSubject -Name "anyscale-operator"
        Ensure-FederatedCredential -Identity $identity -Issuer $oidcIssuer -Subject $workspaceSubject -Name "anyscale-default"
    }

    $registerArgs = @(
        "cloud", "register",
        "--provider", "azure",
        "--region", $location,
        "--name", $cloudName,
        "--compute-stack", "k8s",
        "--cloud-storage-bucket-name", $bucketName,
        "--cloud-storage-bucket-endpoint", $bucketEndpoint,
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

    $parsedFromRegister = Parse-CloudDeploymentIdFromText -Lines $registerOutput
    if ($parsedFromRegister) { $cloudId = $parsedFromRegister }

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

        $operatorValuesPath = Join-Path $PSScriptRoot $OperatorValuesFile
        try {
            Persist-OperatorCloudDeploymentId -FilePath $operatorValuesPath -CloudId $cloudId -Strict $StrictLifecycleFlag
            Write-Host "[INFO] Updated $OperatorValuesFile with cloudDeploymentId=$cloudId" -ForegroundColor Green
        } catch {
            if ($StrictLifecycleFlag) {
                throw "Failed to update $OperatorValuesFile with cloudDeploymentId=${cloudId}: $($_.Exception.Message)"
            }
            Write-Host "[WARN] Failed to update $OperatorValuesFile with cloudDeploymentId: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    } else {
        if ($StrictLifecycleFlag) {
            throw "Could not detect cloudDeploymentId for cloud '$cloudName'. Aborting due to StrictLifecycle=true."
        }
        Write-Host "[WARN] Could not detect cloudDeploymentId from register output; check CLI output above." -ForegroundColor Yellow
    }
}
elseif ($OperatorIdentity) {
    Write-Host "[AZ] RegisterCloud skipped but operator identity provided; ensuring federated credential" -ForegroundColor Yellow
    $identity = Resolve-UserAssignedIdentityByClientId -ClientId $OperatorIdentity
    if (-not $identity) { throw "Managed identity with clientId $OperatorIdentity not found in subscription $subscription" }
    $operatorSubject = "system:serviceaccount:${Namespace}:${ServiceAccountName}"
    $workspaceSubject = "system:serviceaccount:${Namespace}:${WorkspaceServiceAccountName}"
    Ensure-FederatedCredential -Identity $identity -Issuer $oidcIssuer -Subject $operatorSubject -Name "anyscale-operator"
    Ensure-FederatedCredential -Identity $identity -Issuer $oidcIssuer -Subject $workspaceSubject -Name "anyscale-default"
}

# Always re-resolve the cloud deployment id from the cloud object before Helm apply,
# then persist it to operator values so Helm cannot apply stale ids.
if ($InstallOperatorFlag -or $RegisterCloudFlag) {
    $resolvedCloudId = Get-CloudResourceId $cloudName
    if ($resolvedCloudId) {
        if ($cloudId -and $cloudId -ne $resolvedCloudId) {
            Write-Host "[INFO] Reconciled cloudDeploymentId from cloud object: $cloudId -> $resolvedCloudId" -ForegroundColor Cyan
        }
        $cloudId = $resolvedCloudId
    }

    if ($cloudId) {
        $operatorValuesPath = Join-Path $PSScriptRoot $OperatorValuesFile
        Persist-OperatorCloudDeploymentId -FilePath $operatorValuesPath -CloudId $cloudId -Strict $StrictLifecycleFlag
        Write-Host "[INFO] Ensured $OperatorValuesFile contains cloudDeploymentId=$cloudId before Helm apply" -ForegroundColor Green
    } elseif ($StrictLifecycleFlag) {
        throw "cloudDeploymentId unresolved for cloud '$cloudName' before Helm apply."
    }
}

if ($InstallOperatorFlag) {
    $helmExe = Get-HelmExe
    Write-Host "[HELM] Adding/updating repo and installing operator (using $helmExe)" -ForegroundColor Cyan
    if (-not $cloudId) {
        Write-Host "[INFO] Resolving cloudDeploymentId for Helm install..." -ForegroundColor Gray
        $cloudId = Get-CloudResourceId $cloudName
    }
    if (-not $cloudId -and $StrictLifecycleFlag) {
        throw "InstallOperator=true but cloudDeploymentId is unresolved for cloud '$cloudName'. Aborting due to StrictLifecycle=true."
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
            "--create-namespace"
        )

        if ($OperatorChartVersion) {
            $helmArgs += @("--version", $OperatorChartVersion)
        }

        if (Test-Path $OperatorValuesFile) {
            $helmArgs += @("-f", $OperatorValuesFile)
        } elseif (Test-Path $ValuesFile) {
            $helmArgs += @("-f", $ValuesFile)
        } else {
            Write-Host "[WARN] Values file '$OperatorValuesFile' (or fallback '$ValuesFile') not found; proceeding with inline settings." -ForegroundColor Yellow
        }

        $cpuValuesPath = Join-Path $PSScriptRoot $CpuValuesFile
        if (Test-Path $cpuValuesPath) {
            Write-Host "[HELM] Including CPU overrides from $CpuValuesFile" -ForegroundColor Cyan
            $helmArgs += @("-f", $cpuValuesPath)
        } else {
            Write-Host "[WARN] CPU values file '$CpuValuesFile' not found; skipping CPU overrides." -ForegroundColor Yellow
        }

        if ($cloudId) {
            $helmArgs += @(
                "--set-string", "global.cloudDeploymentId=$cloudId",
                "--set-string", "global.cloudProvider=azure",
                "--set-string", "global.auth.audience=api://086bc555-6989-4362-ba30-fded273e432b/.default",
                "--set-string", "workloads.serviceAccount.name=$ServiceAccountName"
            )
        }

        $helmSuccess = $false
        try {
            & $helmExe @helmArgs
            $helmSuccess = $true
        } catch {
            $msg = $_.Exception.Message
            Write-Host "[WARN] Helm upgrade/install failed (will retry clean): $msg" -ForegroundColor Yellow
            if ($msg -match "instance-types" -or $msg -match "cannot patch") {
                Write-Host "[HELM] Uninstalling release and retrying fresh install..." -ForegroundColor Cyan
                & $helmExe uninstall $HelmRelease -n $Namespace -q
                & $helmExe @helmArgs
                $helmSuccess = $true
            } else {
                throw
            }
        }
        if (-not $helmSuccess) { throw "Helm install did not complete." }

        if ($cloudId) {
            $liveCloudId = Get-OperatorDeploymentCloudId -Namespace $Namespace -DeploymentName $HelmRelease
            if (-not $liveCloudId) {
                if ($StrictLifecycleFlag) {
                    throw "Unable to verify live operator cloudDeploymentId after Helm install."
                }
                Write-Host "[WARN] Could not verify live operator cloudDeploymentId from deployment args." -ForegroundColor Yellow
            } elseif ($liveCloudId -ne $cloudId) {
                if ($StrictLifecycleFlag) {
                    throw "Operator deployment cloudDeploymentId mismatch. expected=$cloudId actual=$liveCloudId"
                }
                Write-Host "[WARN] Operator deployment cloudDeploymentId mismatch. expected=$cloudId actual=$liveCloudId" -ForegroundColor Yellow
            } else {
                Write-Host "[INFO] Verified live operator cloudDeploymentId=$liveCloudId" -ForegroundColor Green
            }
        }

        # Ensure patches ConfigMap carries the workload identity client-id so DefaultAzureCredential works in workspace pods
        if ($OperatorIdentity) {
            Ensure-ServiceAccountClientId -Namespace $Namespace -ServiceAccount $WorkspaceServiceAccountName -ClientId $OperatorIdentity
            Ensure-ServiceAccountClientId -Namespace $Namespace -ServiceAccount $ServiceAccountName -ClientId $OperatorIdentity
            Ensure-PatchesConfig -Namespace $Namespace -ClientId $OperatorIdentity
        }
    } catch {
        if ($StrictLifecycleFlag) {
            throw "Helm operator install failed: $($_.Exception.Message)"
        }
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

# anyscale cloud config update --cloud $cloudName  --enable-log-ingestion
# Log ingestion is not supported on Azure — the error says: "Log ingestion to Anyscale Control Plane is not yet supported on Anyscale First-Party Offering on Azure."

Write-Host "[DONE] Flow finished." -ForegroundColor Green