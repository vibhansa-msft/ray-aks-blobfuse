# Ray HPO on AKS with Azure Blob



A complete end-to-end hyperparameter optimization (HPO) pipeline for training DistilBERT on the AG News dataset using Ray Tune, running on Azure Kubernetes Service (AKS) with Azure Blob Storage integration.This project deploys a fully automated Ray cluster on **Azure Kubernetes Service (AKS)** with **KubeRay**, mounts **Azure Blob Storage** via **Blobfuse2 CSI driver**, stages the **AG News** dataset, and submits a **RayJob** that performs **DistilBERT fine-tuning** with **Ray Tune (ASHA scheduler)** for hyperparameter optimization.



## Project Overview## Key Features

- ✅ **Fully Automated**: Single command deploys everything from infrastructure to training job

This project automates the deployment of a distributed machine learning training pipeline that:- ✅ **CPU & GPU Support**: Flexible compute configuration (CPU-only or GPU-accelerated)

- ✅ **Distributed Training**: Ray framework enables distributed training across multiple nodes

- **Orchestrates** Ray clusters on AKS using KubeRay operator- ✅ **Hyperparameter Optimization**: Ray Tune with ASHA scheduler for efficient HPO

- **Stores** training datasets in Azure Blob Storage with Blobfuse2 mounting- ✅ **Cloud Storage Integration**: Azure Blob Storage via Blobfuse2 for scalable dataset access

- **Trains** DistilBERT models for text classification (AG News)- ✅ **Cross-Platform**: PowerShell on Windows, Bash on Linux/macOS

- **Optimizes** hyperparameters using Ray Tune with ASHA scheduler

- **Persists** trained models and checkpoints to blob storage---



## Architecture## Prerequisites

- Azure CLI (`az`), `kubectl`, `helm`, `docker`, `python3`

```- Logged in to Azure: `az login`

┌─────────────────────────────────────────────────────────────────┐- Quota for chosen VM sizes (`Standard_D4s_v5`, `Standard_NC6s_v3` by default)

│                     Azure Cloud                                 │

├─────────────────────────────────────────────────────────────────┤---

│                                                                 │

│  ┌──────────────────────────────────────────────────────────┐   │## How to Run the Deploy Script

│  │              Azure Kubernetes Service (AKS)              │   │

│  │          (5 nodes: Standard_D4_v2 CPU)                   │   │To start the deployment, use the following commands depending on your operating system:

│  │                                                          │   │

│  │  ┌────────────────────────────────────────────────────┐  │   │**Windows PowerShell:**

│  │  │  KubeRay Operator (v1.4.2)                         │  │   │```powershell

│  │  └────────────────────────────────────────────────────┘  │   │powershell -ExecutionPolicy Bypass -File .\deploy.ps1

│  │                                                          │   │```

│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │

│  │  │ Ray Head Pod │  │ Ray Worker 1 │  │ Ray Worker N │   │   │**Linux/macOS Bash:**

│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │```bash

│  │        ↓                  ↓                   ↓           │   │bash ./deploy.sh

│  │  ┌────────────────────────────────────────────────────┐  │   │```

│  │  │  Blobfuse2 Mount (/mnt/blob)                      │  │   │

│  │  │  ├── ag_news/ (13 Parquet files)                  │  │   │Set the required environment variables before running the script, as shown in the sample commands below.

│  │  │  └── checkpoints/ (model_checkpoint.pt)           │  │   │

│  │  └────────────────────────────────────────────────────┘  │   │---

│  │                                                          │   │

│  └──────────────────────────────────────────────────────────┘   │## Quick Start - Sample Commands

│                           ↕                                    │

│  ┌──────────────────────────────────────────────────────────┐   │### Windows PowerShell

│  │  Azure Blob Storage                                     │   │

│  │  (Container: dataset)                                   │   │#### CPU Deployment

│  │  ├── ag_news/*.parquet (13 files)                       │   │```powershell

│  │  └── checkpoints/model_checkpoint.pt                    │   │$env:GPU = "false"

│  └──────────────────────────────────────────────────────────┘   │$env:NodeCount = "2"

│                                                                 │$env:VmType = "Standard_D4_v2"

└─────────────────────────────────────────────────────────────────┘$env:WORKER_REPLICAS = "6"

```$env:NUM_WORKERS = "8"

powershell -ExecutionPolicy Bypass -File .\deploy.ps1

## Prerequisites```



### Local Machine#### GPU Deployment

- **PowerShell** 5.1 or later (Windows)```powershell

- **Azure CLI** (`az` command)$env:GPU = "true"

- **kubectl** (Kubernetes command-line tool)$env:NodeCount = "2"

- **Helm** (Kubernetes package manager)$env:VmType = "Standard_NC6s_v3"

- **Docker** (for building container images)$env:WORKER_REPLICAS = "4"

- **Python** 3.9+ (for dataset preparation)$env:NUM_WORKERS = "4"

powershell -ExecutionPolicy Bypass -File .\deploy.ps1

### Azure Subscription```

- Active Azure subscription with sufficient quota

- Permissions to create:---

  - Resource Groups

  - AKS clusters### Linux/macOS Bash

  - Storage Accounts

  - Virtual Networks#### CPU Deployment

```bash

### Installationexport GPU="false"

```powershellexport NodeCount="2"

# Install Azure CLIexport VmType="Standard_D4_v2"

# Visit: https://learn.microsoft.com/en-us/cli/azure/install-azure-cliexport WORKER_REPLICAS="6"

export NUM_WORKERS="8"

# Install kubectlbash ./deploy.sh

az aks install-cli```



# Install Helm#### GPU Deployment

# Visit: https://helm.sh/docs/intro/install/```bash

export GPU="true"

# Verify installationsexport NodeCount="2"

az --versionexport VmType="Standard_NC6s_v3"

kubectl version --clientexport WORKER_REPLICAS="4"

helm versionexport NUM_WORKERS="4"

```bash ./deploy.sh

```

## Project Structure

---

```

vibhansa-ray-aks/This project deploys a Ray cluster on **AKS** with **KubeRay**, mounts Azure Blob via the **Azure Blob CSI driver (Blobfuse2)**, stages a **public AG News** dataset, and submits a **RayJob** that performs **DistilBERT fine‑tuning** with **Ray Tune (ASHA)**.

├── deploy.ps1                    # Main deployment orchestrator

├── setup-aks.ps1                 # AKS cluster setup- **Python 3.11** - For data preparation

├── monitor-rayjob.ps1            # Ray job monitoring

├── utils.ps1                      # Utility functions---

├── .env.example                   # Configuration template

├── README.md                      # This file### Azure Resources

├── app/

│   ├── train_hpo.py             # Ray training script- Active Azure subscription with sufficient quota for selected VM sizes

│   └── requirements.txt          # Python dependencies- CPU nodes: `Standard_D4s_v5` (4 vCPU, 16 GB RAM)

├── data/- GPU nodes: `Standard_NC6s_v3` (6 vCPU, 56 GB RAM, 1x K80 GPU)

│   └── prepare_ag_news.py        # Dataset preparation- Logged in to Azure: `az login`

├── docker/
│   └── Dockerfile               # Container image definition
├── k8s/
│   ├── rayjob-cpu.yaml          # RayJob specification (CPU)
│   ├── rayjob-gpu.yaml          # RayJob specification (GPU)
│   ├── storageclass-blobfuse2.yaml  # Blobfuse2 StorageClass
│   ├── pvc-blob.yaml            # Persistent Volume Claim
│   └── nvidia-device-plugin.yaml # GPU support (optional)
└── deploy.sh                     # Bash deployment script (Linux/macOS)
```

## Configuration

### 1. Set Up Environment Variables

Copy and configure the environment file:

```powershell
cp .env.example .env
```

Edit `.env.example` with your Azure settings:

```bash
# Azure Configuration
SUBSCRIPTION="your-subscription-id"
RESOURCE_GROUP="your-resource-group"
LOCATION="eastus"
STORAGE_ACCOUNT="yourstoragename"

# AKS Configuration
AKS_CLUSTER="your-cluster-name"
NODE_COUNT=5
VM_TYPE="Standard_D4_v2"
GPU=false

# Ray Configuration
WORKER_REPLICAS=3
NUM_WORKERS=3
NUM_SAMPLES=12
```

### 2. Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NODE_COUNT` | 5 | Number of AKS nodes |
| `VM_TYPE` | Standard_D4_v2 | Azure VM type (4 vCPU, 16GB RAM) |
| `GPU` | false | Enable GPU support |
| `WORKER_REPLICAS` | 3 | Number of Ray worker pod replicas |
| `NUM_WORKERS` | 3 | Number of Ray workers per replica |
| `BATCH_SIZE` | 2 | Training batch size |
| `EPOCHS` | 1 | Number of training epochs |

## Deployment

### Prerequisites Check

Before deploying, ensure:

1. **Azure authentication:**
   ```powershell
   az login
   az account show
   ```

2. **Kubectl access:**
   ```powershell
   kubectl get nodes
   ```

### Running the Deployment

```powershell
# Navigate to project directory
cd c:\Users\vibhansa\Documents\Projects\vibhansa-ray-aks

# Run the deployment script
powershell -ExecutionPolicy Bypass -File .\deploy.ps1
```

### What the Deploy Script Does

The `deploy.ps1` script automates:

1. **Cleanup**
   - Deletes any existing RayJobs (fresh start)
   
2. **Azure Resources**
   - Verifies Resource Group
   - Creates/verifies Storage Account
   - Creates/verifies Blob Container
   - Assigns managed identity roles to AKS cluster
   
3. **Kubernetes Setup**
   - Creates/configures AKS cluster
   - Deletes and recreates StorageClass (applies config changes)
   - Deletes and recreates PVC for blob storage
   
4. **Ray Setup**
   - Installs KubeRay operator via Helm
   - Checks for GPU support (if enabled)
   
5. **Dataset**
   - Checks blob storage for dataset (13 Parquet files)
   - Falls back to local directory or downloads if needed
   
6. **Job Submission**
   - Creates ConfigMap with application code
   - Substitutes configuration parameters
   - Submits RayJob to Kubernetes
   
7. **Monitoring**
   - Continuously monitors job status every 30 seconds
   - Reports progress and resource utilization
   - Auto-kills jobs stuck in "Initializing" state (after 25 checks)

### Deployment Output

```
========================================
Ray Job Deployment Complete
========================================

RayJob 'hpo-job-cpu' has been submitted successfully.
Starting continuous monitoring...

[0m 2s] Check #1
  Deployment Status: Initializing
  Succeeded: 0
  Failed: 0

...

[5m 26s] Check #11
  Deployment Status: Complete
  Succeeded: 1
  Failed: 0

JOB COMPLETED SUCCESSFULLY

========================================
Monitoring Complete
Total elapsed time: 5m 26s
========================================
```

## Training Script

The training script (`app/train_hpo.py`) implements:

### Features
- **Data Loading:** Loads Parquet files from Azure Blob Storage
- **Model:** DistilBERT for sequence classification (4 labels)
- **Training:** Single epoch, configurable batch size
- **Checkpoint:** Saves trained model to `/mnt/blob/checkpoints/`
- **Logging:** Detailed progress and error reporting

### Configuration (Environment Variables)
```python
BLOB_DIR = "/mnt/blob/ag_news"      # Dataset location
NUM_WORKERS = 3                      # Ray workers
NUM_SAMPLES = 12                     # HPO trials
BATCH_SIZE = 2                       # Training batch size
EPOCHS = 1                           # Number of epochs
```

### Output
- Model checkpoint: `/mnt/blob/checkpoints/model_checkpoint.pt` (includes state dict, optimizer, loss)
- Training logs: Ray job logs (accessible via `kubectl logs`)
- Metrics: Loss reported to Ray Tune

## Monitoring and Debugging

### View Job Status
```powershell
# Get RayJob status
kubectl get rayjob hpo-job-cpu

# Get detailed RayJob info
kubectl get rayjob hpo-job-cpu -o yaml

# View job logs
kubectl logs -l ray.io/job-name=hpo-job-cpu -f

# View head pod logs
kubectl logs <head-pod-name> -f

# View worker pod logs
kubectl logs <worker-pod-name> -f
```

### Troubleshooting

#### Job Stuck in "Initializing"
**Problem:** Job stays in "Initializing" state for too long
**Solution:**
- Check node capacity: `kubectl top nodes`
- Reduce `WORKER_REPLICAS` or `NUM_WORKERS`
- Check pod events: `kubectl describe pods -l ray.io/job-name=hpo-job-cpu`

#### Blobfuse Mount Issues
**Problem:** Pods can't access `/mnt/blob`
**Solution:**
- Check PVC status: `kubectl get pvc blob-pvc`
- Check StorageClass: `kubectl get storageclass azureblob-fuse2`
- View blobfuse logs: `kubectl exec <pod-name> -- cat /root/blobfuse.log`

#### Dataset Not Found
**Problem:** "Must provide at least one path" error
**Solution:**
- Verify dataset in storage: `az storage blob list --account-name <storage-account> --container-name dataset --prefix "ag_news"`
- Check blobfuse mount: `kubectl exec <pod-name> -- ls -la /mnt/blob/ag_news/`

#### Memory Issues
**Problem:** OOM (Out of Memory) errors
**Solution:**
- Reduce `BATCH_SIZE` (default: 2)
- Reduce `NUM_WORKERS` (default: 3)
- Use larger node types (`Standard_D8_v2`, etc.)

## Managing Resources

### Delete All Resources
```powershell
# Delete AKS cluster
az aks delete -g <resource-group> -n <cluster-name> --yes

# Delete storage account
az storage account delete -g <resource-group> -n <storage-account> --yes

# Delete resource group (deletes everything)
az group delete -g <resource-group> --yes
```

### Clean Up Running Jobs
```powershell
# Delete all RayJobs
kubectl delete rayjob --all

# Delete specific RayJob
kubectl delete rayjob hpo-job-cpu
```

### Scale Configuration
Adjust in `.env.example`:

| Goal | Changes |
|------|---------|
| Faster training | Reduce `NODE_COUNT`, increase batch size |
| More parallelism | Increase `WORKER_REPLICAS`, `NUM_WORKERS` |
| Test quickly | Set `EPOCHS=1`, `BATCH_SIZE=2` (default) |
| Full HPO | Increase `NUM_SAMPLES`, adjust `WORKER_REPLICAS` |

## Storage Management

### Access Checkpoints
```powershell
# List checkpoints in blob storage
az storage blob list --account-name <storage-account> --container-name dataset --prefix "checkpoints" -o table

# Download checkpoint
az storage blob download --account-name <storage-account> --container-name dataset --name "checkpoints/model_checkpoint.pt" --file "model_checkpoint.pt"
```

### Dataset Details
- **Format:** Parquet
- **Files:** 13 training data files
- **Size:** ~150 MB total
- **Location:** `dataset/ag_news/` in blob storage
- **Access:** Automatically mounted at `/mnt/blob/ag_news` in pods

## Performance Notes

### Typical Execution Times
- **Cluster startup:** 2-3 minutes
- **Job initialization:** 2-3 minutes
- **Single batch training:** 1-2 minutes (batch_size=2)
- **Checkpoint save:** <1 minute

### Resource Utilization
- **Head pod:** ~2 GB RAM, 1 vCPU
- **Worker pod:** ~4-6 GB RAM, 2-3 vCPU (depending on batch size)
- **Blobfuse cache:** ~120 seconds timeout, automatic cleanup

## Advanced Usage

### Running Full HPO
Modify `app/train_hpo.py` to uncomment HPO:
```python
# Change from:
train_ds = load_parquet_ds(f"{BLOB_DIR}/train_00010.parquet")

# To:
train_ds = load_parquet_ds(f"{BLOB_DIR}/train_*.parquet").random_shuffle()

# And enable Ray Tune instead of simple training.fit()
```

### Adding GPU Support
```bash
# Update .env.example
GPU=true

# Update WORKER_REPLICAS and NUM_WORKERS for GPU capacity
WORKER_REPLICAS=2
NUM_WORKERS=2
```

### Custom Models
Replace in `app/train_hpo.py`:
```python
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Change model here
    num_labels=4
)
```

## Support & Troubleshooting

### Common Issues

1. **"Command not found" errors**
   - Ensure all prerequisites are installed and in PATH
   - Restart PowerShell after installation

2. **Azure authentication failures**
   - Run `az login` and select correct subscription
   - Check permissions: `az account show`

3. **Kubectl connection errors**
   - Verify kubeconfig: `kubectl config current-context`
   - Retrieve credentials: `az aks get-credentials -g <rg> -n <cluster>`

4. **RayJob stuck in "Initializing"**
   - Monitor automatically kills after 25 checks (~12 minutes)
   - Check: `kubectl describe pods -l ray.io/job-name=hpo-job-cpu`

### Useful Commands Reference

```powershell
# Cluster management
kubectl get nodes                          # List cluster nodes
kubectl top nodes                          # Node resource usage
kubectl get pods -A                        # All pods in cluster

# Job management
kubectl get rayjob                         # List all RayJobs
kubectl describe rayjob hpo-job-cpu        # Detailed RayJob info
kubectl delete rayjob hpo-job-cpu          # Delete specific job

# Logs
kubectl logs <pod-name>                    # Pod logs
kubectl logs -l ray.io/job-name=hpo-job-cpu # All job-related logs
kubectl exec <pod-name> -- bash            # Execute in pod

# Storage
kubectl get pvc                            # Persistent Volume Claims
kubectl get storageclass                   # Storage Classes
```

## Project Files Reference

| File | Purpose |
|------|---------|
| `deploy.ps1` | Main deployment orchestrator |
| `setup-aks.ps1` | AKS cluster creation & configuration |
| `monitor-rayjob.ps1` | Real-time job monitoring |
| `utils.ps1` | Shared utility functions |
| `app/train_hpo.py` | Ray training script (main logic) |
| `app/requirements.txt` | Python dependencies |
| `data/prepare_ag_news.py` | Dataset preparation script |
| `k8s/rayjob-cpu.yaml` | CPU-only RayJob specification |
| `k8s/rayjob-gpu.yaml` | GPU RayJob specification |
| `k8s/storageclass-blobfuse2.yaml` | Blobfuse2 configuration |
| `k8s/pvc-blob.yaml` | Persistent Volume Claim |
| `.env.example` | Configuration template |

## License

This project is provided as-is for educational and research purposes.

## Contributing

For issues or improvements, please refer to the project documentation and Azure/Ray community resources.

---

**Last Updated:** October 30, 2025
**Version:** 1.0
**Status:** Production Ready ✅
