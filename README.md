# Ray on Azure Kubernetes Service (AKS) with BlobFuse

A complete solution for running distributed Ray workloads on Azure Kubernetes Service with Azure Blob Storage integration and comprehensive monitoring.

## 🚀 Quick Start

```powershell
# Deploy entire infra (Azure resource group, AKS, Storagge account, Container etc followed by a sample job)
.\deploy.ps1

# Deploy and run HPO training job with monitoring
.\quick-deploy.ps1

# Or run data preparation job
.\run-dataprep.ps1

# Open monitoring dashboards only
.\open-dashboard.ps1
```

## 📁 Project Structure

### 🔧 Core Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| **`quick-deploy.ps1`** | Deploy HPO training job with full monitoring | Main deployment script |
| **`run-dataprep.ps1`** | Run data preparation job with monitoring | Data processing workflow |
| **`deploy.ps1`** | Legacy full infrastructure deployment | Advanced setup |
| **`open-dashboard.ps1`** | Open Ray and Grafana dashboards | Monitoring access |
| **`utils.ps1`** | Utility functions and monitoring setup | Shared functions |

### 🏗️ Infrastructure Files

| File | Description |
|------|-------------|
| **`.env.example`** | Environment variables template |
| **`docker/Dockerfile`** | Ray application container image |
| **`app/requirements.txt`** | Python dependencies |
| **`app/train_hpo.py`** | Hyperparameter optimization training code |
| **`data/prepare_ag_news.py`** | Data preparation script |

### ☸️ Kubernetes Manifests (`k8s/`)

| File | Purpose |
|------|---------|
| **`rayjob-cpu.yaml`** | CPU-based Ray job definition |
| **`rayjob-gpu.yaml`** | GPU-based Ray job definition |
| **`storageclass-blobfuse2.yaml`** | Azure Blob storage classes |
| **`pvc-blob.yaml`** | Persistent volume claims for datasets |
| **`nvidia-device-plugin.yaml`** | NVIDIA GPU support |
| **`grafana-monitoring.yaml`** | Grafana + Prometheus monitoring stack |

### 📊 Monitoring Configuration

| File | Description |
|------|-------------|
| **`grafana-ray-dashboard.json`** | Ray cluster monitoring dashboard |
| **`NETWORK-MONITORING.md`** | Network monitoring documentation |

## 🎯 Available Scripts

### 1. **Quick Deploy** - `quick-deploy.ps1`

**Purpose:** Deploy HPO training job with automatic monitoring

**What it does:**
- Cleans up existing Ray jobs
- Sets up Azure Blob storage (StorageClasses & PVCs)  
- Creates application code ConfigMap
- Deploys Ray HPO training job
- Sets up Grafana + Prometheus monitoring
- Opens Ray and Grafana dashboards
- Monitors job execution with real-time status

**Usage:**
```powershell
.\quick-deploy.ps1
```

### 2. **Data Preparation** - `run-dataprep.ps1`

**Purpose:** Run data preparation workflow

**What it does:**
- Deploys data preparation Ray job
- Sets up monitoring infrastructure
- Opens monitoring dashboards
- Processes AG News dataset for training

**Usage:**
```powershell
.\run-dataprep.ps1
```

### 3. **Dashboard Access** - `open-dashboard.ps1`

**Purpose:** Open monitoring dashboards without deploying jobs

**What it does:**
- Sets up port forwarding for Ray dashboard
- Opens Ray dashboard (http://localhost:8265)
- Opens Grafana dashboard (external IP)
- Imports Ray monitoring dashboard to Grafana

**Usage:**
```powershell
.\open-dashboard.ps1
```

### 4. **Full Infrastructure** - `deploy.ps1`

**Purpose:** Complete AKS cluster and Ray infrastructure setup

**What it does:**
- Creates Azure resource group and AKS cluster
- Sets up RBAC and service accounts
- Installs Helm charts and operators
- Configures storage and networking
- Deploys Ray cluster

**Usage:**
```powershell
.\deploy.ps1
```

## 📊 Monitoring Features

### Ray Dashboard
- **URL:** http://localhost:8265
- **Features:** Job status, task execution, cluster health, logs

### Grafana Dashboard  
- **URL:** http://<external-ip>:3000
- **Credentials:** admin / raymonitoring123
- **Panels:**
  - Ray Cluster Status & Active Nodes
  - CPU & Memory Utilization
  - Object Store Usage
  - Task Throughput
  - **Network Monitoring:**
    - Network Bandwidth (Sent/Received)
    - Total Network Usage
    - Real-time Network Speed
    - Per-Node Network Utilization

### Available Metrics
- **Performance:** CPU, Memory, Task execution times
- **Network:** Bandwidth usage, transfer rates, total data moved
- **Storage:** Object store utilization, data spilling
- **Jobs:** Active tasks, completion status, failure rates

## 🔧 Configuration

### Environment Setup
Copy and customize environment variables:
```powershell
cp .env.example .env
# Edit .env with your Azure subscription and resource details
```

### Key Parameters
- **Subscription:** Azure subscription ID
- **Resource Group:** AKS cluster resource group
- **Cluster Name:** AKS cluster name
- **Storage Account:** Azure Blob storage account
- **Worker Replicas:** Number of Ray worker nodes
- **VM Type:** Azure VM size for nodes

## 🚦 Workflow Examples

### Development Workflow
```powershell
# 1. Deploy training job with monitoring
.\quick-deploy.ps1

# 2. Monitor progress in dashboards (auto-opens)
# - Ray Dashboard: Job execution details
# - Grafana: Performance metrics and network usage

# 3. Run data preparation for next iteration
.\run-dataprep.ps1

# 4. Re-access dashboards anytime
.\open-dashboard.ps1
```

### Production Workflow
```powershell
# 1. Set up infrastructure (one-time)
.\deploy.ps1

# 2. Deploy jobs with monitoring
.\quick-deploy.ps1

# 3. Monitor via Grafana for production metrics
# Access: http://<grafana-ip>:3000
```

## 🎛️ Monitoring Integration

### Automatic Setup
All job deployment scripts automatically:
1. Deploy Grafana + Prometheus monitoring stack
2. Import Ray monitoring dashboard 
3. Set up port forwarding for Ray dashboard
4. Open both dashboards in browser
5. Configure network bandwidth monitoring

### Manual Monitoring
```powershell
# Set up monitoring stack only
. .\utils.ps1
Deploy-MonitoringStack
Import-RayDashboard
Open-AllDashboards
```

## 📋 Prerequisites

- **Azure CLI** - Azure authentication and resource management
- **kubectl** - Kubernetes cluster management
- **PowerShell 5.1+** - Script execution environment
- **Docker** (optional) - For building custom images
- **AKS Cluster** - Running Kubernetes cluster with Ray operator

## 🔍 Troubleshooting

### Common Issues

**Ray Job Fails:**
- Check pod logs: `kubectl logs <job-pod-name>`
- Verify storage PVC status: `kubectl get pvc`
- Check resource limits and node capacity

**Monitoring Not Working:**
- Verify monitoring pods: `kubectl get pods -n monitoring`
- Check port forwarding: `kubectl port-forward --help`
- Re-import dashboard: `. .\utils.ps1; Import-RayDashboard`

**Network Metrics Missing:**
- Ensure Ray head pod is running: `kubectl get pods -l ray.io/node-type=head`
- Check Prometheus targets: Access Prometheus UI at http://localhost:9090
- Verify Ray metrics endpoint: Port forward 8080 and check http://localhost:8080/metrics

### Useful Commands
```powershell
# Check Ray jobs
kubectl get rayjobs

# Monitor pods
kubectl get pods -w

# Check monitoring stack
kubectl get pods -n monitoring

# View job logs
kubectl logs -f <pod-name>

# Access services
kubectl get svc --all-namespaces
```

## 🏷️ Tags
`ray` `kubernetes` `azure` `machine-learning` `distributed-computing` `monitoring` `grafana` `prometheus` `hyperparameter-optimization` `blob-storage`