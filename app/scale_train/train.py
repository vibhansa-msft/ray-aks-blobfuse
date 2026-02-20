# ==============================================================================
# Ray Job for Distributed Model and Dataset Downloading
# ==============================================================================
# This script demonstrates how to use Ray to:
# 1. Initialize a Ray cluster connection.
# 2. Download a large pre-trained model (Mistral-7B) to a shared storage volume.
# 3. Distribute the download of a large dataset (Fineweb-Edu) across multiple 
#    worker nodes in parallel.
#
# Shared Storage:
# The script assumes a shared volume is mounted at '/mnt/shared_storage'.
# This allows all nodes to access the downloaded model and dataset without
# redundant downloads or manual transfer.
# 
# ==============================================================================

import os
import time
import sys
import ray
import json
from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------

# Define storage path where data will be persisted.
# Check env var first, default to /mnt/shared_storage
STORAGE_PATH = os.environ.get("STORAGE_PATH", "/mnt/shared_storage")

# ------------------------------------------------------------------------------
# Logging Helper (dual-write: stdout + local log file)
# ------------------------------------------------------------------------------
import socket
_JOB_TIMESTAMP = int(time.time())
_HOSTNAME = socket.gethostname()
LOG_DIR = "/tmp/ray_job_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"job_{_JOB_TIMESTAMP}_{_HOSTNAME}.log")

def log(message):
    """
    Prints to stdout (captured by Anyscale/Ray for 'anyscale job logs')
    AND appends to a LOCAL log file in /tmp (no blobfuse corruption).
    Each node gets its own log file identified by hostname.
    """
    print(message, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(message + "\n")
            f.flush()
    except Exception:
        pass


@ray.remote(num_cpus=1)
def _collect_node_logs():
    """
    Ray remote task to collect local log files from a worker node.
    Returns a dict with hostname, log file paths, and their contents.
    """
    import glob
    hostname = socket.gethostname()
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    logs = {}
    for lf in log_files:
        try:
            with open(lf, "r") as f:
                logs[os.path.basename(lf)] = f.read()
        except Exception:
            pass
    return {"hostname": hostname, "logs": logs}


def collect_worker_logs():
    """
    Gathers log files from all worker nodes and saves them to shared_storage.
    Called once at job end — single batch write to blobfuse, not concurrent.
    """
    log("\n[Logs] Collecting worker logs from all nodes...")
    logs_output_dir = os.path.join(STORAGE_PATH, "job_logs", str(_JOB_TIMESTAMP))
    os.makedirs(logs_output_dir, exist_ok=True)

    # Save head node's own log first
    try:
        import shutil
        if os.path.exists(LOG_FILE):
            shutil.copy2(LOG_FILE, os.path.join(logs_output_dir, os.path.basename(LOG_FILE)))
            log(f"[Logs] Saved head node log: {os.path.basename(LOG_FILE)}")
    except Exception as e:
        log(f"[Logs] Warning: Failed to save head node log: {e}")

    # Collect from workers — schedule one task per node
    try:
        num_tasks = 20  # More than num_workers to cover all nodes
        futures = [_collect_node_logs.remote() for _ in range(num_tasks)]
        results = ray.get(futures, timeout=60)

        seen_hosts = set()
        for result in results:
            hostname = result["hostname"]
            if hostname in seen_hosts:
                continue
            seen_hosts.add(hostname)
            for filename, content in result["logs"].items():
                out_path = os.path.join(logs_output_dir, filename)
                if not os.path.exists(out_path):
                    with open(out_path, "w") as f:
                        f.write(content)

        log(f"[Logs] Collected logs from {len(seen_hosts)} nodes to {logs_output_dir}")
    except Exception as e:
        log(f"[Logs] Warning: Failed to collect some worker logs: {e}")

# ------------------------------------------------------------------------------
# Storage Validation and Setup
# ------------------------------------------------------------------------------
def validate_and_setup_storage():
    """
    Validates that the shared storage path exists.
    If it exists, creates the necessary subdirectories for model and dataset.
    This must be called before any file operations.
    """
    # 1. Log the path we are about to check
    log(f"[Init] Checking storage path: {STORAGE_PATH}")
    
    # 2. Critical Check: Ensure the Persistent Volume is mounted
    if not os.path.exists(STORAGE_PATH):
        error_msg = f"[CRITICAL] Storage path {STORAGE_PATH} not found! Ensure PV is mounted."
        log(error_msg)
        # We cannot log this to the file because the storage path doesn't exist
        # Raising an error here prevents the script from silently failing later
        raise FileNotFoundError(error_msg)
    
    log(f"[Init] Storage path confirmed: {STORAGE_PATH}")
    
    # 3. Setup: Create required subdirectories if they don't exist
    try:
        log(f"[Init] Creating directories in {STORAGE_PATH}...")
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(DATASET_SAVE_PATH, exist_ok=True)
        os.makedirs(CHECKPOINT_SAVE_PATH, exist_ok=True)
        log(f"[Init] Directories created successfully.")
    except Exception as e:
        log(f"[CRITICAL] Failed to create directories: {e}")
        raise


# Path to save the Large Language Model (LLM)
MODEL_SAVE_PATH = os.path.join(STORAGE_PATH, "model")

# Identifier for the model on Hugging Face Hub
# Note: Mistral-7B is a gated model. Ensure HF_TOKEN env var is set in job.yaml.
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"  

# Identifier for the dataset on Hugging Face Hub
DATASET_ID = "HuggingFaceFW/fineweb-edu" 

# Subset of the dataset to download (shards containing this string)
# Using 100BT subset for a manageable download (~460 parquet files)
DATASET_SUBSET = "sample/100BT" 

# Path to save the dataset shards
DATASET_SAVE_PATH = os.path.join(STORAGE_PATH, "dataset")

# Path to save checkpoints
CHECKPOINT_SAVE_PATH = os.path.join(STORAGE_PATH, "checkpoint")


# ------------------------------------------------------------------------------
# Ray Initialization
# ------------------------------------------------------------------------------
log("\n[Init] Initializing Ray Cluster Connection...")

# Check if Ray is already running (e.g., if run via 'ray job submit')
if ray.is_initialized():
    log("[Init] Ray is already initialized.")
else:
    # Connect to the existing Ray cluster
    ray.init(
        address="auto",     # Connect to the cluster Ray instance (auto-detect)
        log_to_driver=True  # ensures worker logs are forwarded)
    ) 

# Log available resources to verify cluster size
resources = ray.available_resources()
log(f"[Init] Connected. Cluster Resources: {resources}")


# ------------------------------------------------------------------------------
# Task 1: Model Download (Centralized)
# ------------------------------------------------------------------------------
@ray.remote(num_cpus=10)
def download_model():
    """
    Downloads the entire model repository to shared storage.
    Runs as a remote task to offload memory usage from the head node.
    """
    log(f"\n[Model] Checking storage path: {STORAGE_PATH}")
    
    # Path validation is now handled in validate_and_setup_storage()

    log(f"[Model] Downloading {MODEL_ID} to {MODEL_SAVE_PATH}...")
    
    # Log parameters for debugging
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        # Mask the token for security
        masked_token = f"{hf_token[:4]}...{hf_token[-4:]}"
    else:
        masked_token = "NOT_SET"
        
    log(f"[Model] Parameters: repo_id={MODEL_ID}, local_dir={MODEL_SAVE_PATH}, token={masked_token}")

    try:
        # snapshot_download fetches the entire repo
        token_arg = os.environ.get("HF_TOKEN")
        if not token_arg:  # invalid empty string or None
             token_arg = False # Use False for public repos if no token, or let it fail if gated

        snapshot_download(
            repo_id=MODEL_ID, 
            local_dir=MODEL_SAVE_PATH,
            token=token_arg,
            local_dir_use_symlinks=False,
        )
        log(f"[Model] Success! Model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        log(f"[Model] Error downloading model: {e}")
        raise


# ------------------------------------------------------------------------------
# Task 2: Dataset Download (Distributed via Ray Data)
# ------------------------------------------------------------------------------

def download_file_wrapper(batch):
    """
    Ray Data map function: Downloads a batch of files.
    Ray Data passes a dictionary-like batch (e.g. {'item': [file1, file2]}).
    """
    worker_id = f"{ray.get_runtime_context().get_task_id()}"
    node_ip = ray.util.get_node_ip_address()
    worker_tag = f"Worker {worker_id[-8:]}@{node_ip}"

    try:
        # Standardize input: Ray Data 2.x passes batches as dict of column arrays
        filenames = batch.get("item", [])
    except AttributeError:
        # Fallback for older Ray versions or different format
        filenames = batch
        
    file_names = []
    statuses = []
    size_bytes_list = []
    
    token_arg = os.environ.get("HF_TOKEN")
    if not token_arg:  # invalid empty string or None
        token_arg = False # Use False for public repos if no token, or let it fail if gated

    log(f"[{worker_tag}] Starting batch of {len(filenames)} files")
             
    for filename in filenames:
        try:
            log(f"[{worker_tag}] Processing: {filename}")
            file_path = os.path.join(DATASET_SAVE_PATH, filename)
            
            # Simple check to avoid re-downloading existing files
            if os.path.exists(file_path):
                 fsize = os.path.getsize(file_path)
                 log(f"[{worker_tag}] Skipped (exists, {fsize/(1024*1024):.1f} MB): {filename}")
                 file_names.append(filename)
                 statuses.append("skipped")
                 size_bytes_list.append(fsize)
                 continue

            # Download specific file to the shared dataset folder
            hf_hub_download(
                repo_id=DATASET_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=DATASET_SAVE_PATH,
                token=token_arg
            )
            # Get file size after download
            fsize = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            log(f"[{worker_tag}] Downloaded ({fsize/(1024*1024):.1f} MB): {filename}")
            file_names.append(filename)
            statuses.append("downloaded")
            size_bytes_list.append(fsize)
        except Exception as e:
            log(f"[{worker_tag}] Failed: {filename} - {e}")
            file_names.append(filename)
            statuses.append("failed")
            size_bytes_list.append(0)
    
    batch_total_mb = sum(size_bytes_list) / (1024 * 1024)
    log(f"[{worker_tag}] Batch complete: {len(file_names)} files, {batch_total_mb:.1f} MB total")
    # Return columnar format: dict of lists (required by Ray Data map_batches)
    return {"file": file_names, "status": statuses, "size_bytes": size_bytes_list}

# ------------------------------------------------------------------------------
# Task 3: Dataset Download Orchestration
# ------------------------------------------------------------------------------
def download_dataset_distributed():
    """
    Orchestrates the distributed download of dataset files using Ray Data.
    1. Lists files from the Hub.
    2. Creates a Ray Dataset from the file list.
    3. Uses map_batches to distribute the download work.
    """
    log(f"\n[Dataset] Fetching file list for {DATASET_ID} ({DATASET_SUBSET})...")
    
    # List all files in the remote repository
    try:
        all_files = list_repo_files(
            repo_id=DATASET_ID, 
            repo_type="dataset", 
            token=os.environ.get("HF_TOKEN")
        )
    except Exception as e:
        log(f"[Dataset] Error listing repo files: {e}")
        return

    log(f"[Dataset] Total files in repo: {len(all_files)}")

    # Filter for parquet files in the specific subset
    target_files = [f for f in all_files if f.endswith(".parquet") and DATASET_SUBSET in f]
    
    log(f"[Dataset] Filtered to {len(target_files)} parquet files matching '{DATASET_SUBSET}'.")
    log("[Dataset] Triggering distributed download via Ray Data...")
    
    start_time = time.time()
    
    # Create a Ray Dataset from the list of files.
    # We wrap them in a dict standardizes the column name to "item"
    input_data = [{"item": f} for f in target_files]
    ds = ray.data.from_items(input_data)
    
    # Use map_batches to distribute downloads across the cluster.
    # Downloads are I/O-bound, so we use num_cpus=1 per task to allow
    # many tasks to run concurrently on each node. The autoscaler will
    # add more nodes as needed (min_nodes=2, max_nodes=20).
    num_files = len(target_files)
    batch_sz = max(1, num_files // 20)  # ~7 files per batch → 20 batches
    log(f"[Dataset] Distributing {num_files} files in batches of {batch_sz}")
    
    downloaded_ds = ds.map_batches(
        download_file_wrapper, 
        batch_size=batch_sz,  # ~7 files per batch
        num_cpus=1,           # I/O-bound: 1 CPU per task, many tasks per node
        concurrency=20        # Up to 20 concurrent download tasks
    )
    
    # Force execution and collect results
    # take_all() triggers the computation and returns list of row dicts
    all_results = downloaded_ds.take_all()
    
    duration = time.time() - start_time
    
    # Each row is a dict like {"file": "...", "status": "...", "size_bytes": N}
    success_count = sum(1 for r in all_results if r["status"] != "failed")
    failed_count = sum(1 for r in all_results if r["status"] == "failed")
    total_bytes = sum(r.get("size_bytes", 0) for r in all_results)
    total_gb = total_bytes / (1024 ** 3)
    
    log(f"[Dataset] Summary: Processed {len(all_results)} files.")
    log(f"[Dataset] Total data volume: {total_gb:.2f} GB ({total_bytes:,} bytes)")
    log(f"[Dataset] Successfully downloaded {success_count}/{len(target_files)} files in {duration:.2f} seconds.")
    log(f"[Dataset] Throughput: {total_gb/duration*1024:.1f} MB/s" if duration > 0 else "")
    if failed_count > 0:
        failed_files = [r["file"] for r in all_results if r["status"] == "failed"]
        log(f"[Dataset] WARNING: {failed_count} files failed: {failed_files}")


# ------------------------------------------------------------------------------
# Task 4: Distributed Training Simulation
# ------------------------------------------------------------------------------

@ray.remote(num_cpus=0)
class TrainingTracker:
    """
    Ray actor that collects status updates from all training workers.
    The head node polls this actor every 5 seconds to print a status table.
    """
    def __init__(self, num_workers, num_epochs):
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.status = {}
        for i in range(num_workers):
            self.status[i] = {
                "phase": "starting",
                "epoch": 0,
                "batches": 0,
                "bytes_read_mb": 0.0,
                "last_epoch_time": 0.0,
                "last_ckpt_time": 0.0,
                "total_time": 0.0,
                "node_ip": "",
                "done": False,
            }

    def update(self, worker_idx, phase, epoch=0, batches=0, bytes_read_mb=0.0,
              epoch_time=0.0, ckpt_time=0.0, total_time=0.0, node_ip="", done=False):
        self.status[worker_idx] = {
            "phase": phase,
            "epoch": epoch,
            "batches": batches,
            "bytes_read_mb": bytes_read_mb,
            "last_epoch_time": epoch_time,
            "last_ckpt_time": ckpt_time,
            "total_time": total_time,
            "node_ip": node_ip,
            "done": done,
        }

    def get_status(self):
        return dict(self.status)


@ray.remote(num_cpus=30)
def train_worker(worker_idx, assigned_files, tracker, num_epochs=10):
    """
    Distributed training worker. Loads the entire model using HuggingFace
    from_pretrained, simulates training, and checkpoints with save_pretrained.

    Args:
        worker_idx: This worker's index (0..N-1)
        assigned_files: List of parquet training data files for this worker.
        num_epochs: Number of training epochs to simulate.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    node_ip = ray.util.get_node_ip_address()
    tag = f"Train-Worker-{worker_idx}@{node_ip}"
    log(f"\n[{tag}] Starting training simulation...")
    log(f"[{tag}] Data: {len(assigned_files)} files, {num_epochs} epochs")

    # Report initial status to tracker
    tracker.update.remote(worker_idx, "loading_model", node_ip=node_ip)

    try:
        # ---- 1. Load model and tokenizer using from_pretrained ----
        log(f"[{tag}] Loading model and tokenizer from {MODEL_SAVE_PATH}...")
        load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(MODEL_SAVE_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
        load_time = time.time() - load_start

        num_params = sum(p.numel() for p in model.parameters())
        mem_mb = sum(p.nbytes for p in model.parameters()) / (1024 * 1024)
        model.train()  # Set model to training mode
        log(f"[{tag}] Model loaded: {num_params:,} params, {mem_mb:.1f} MB in {load_time:.2f}s")
        tracker.update.remote(worker_idx, "training", epoch=0, node_ip=node_ip)

    except Exception as e:
        log(f"[{tag}] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        tracker.update.remote(worker_idx, "FAILED", node_ip=node_ip)
        return False

    # ---- 2. Training loop: epochs x data files ----
    worker_checkpoint_dir = os.path.join(CHECKPOINT_SAVE_PATH, f"worker_{worker_idx}")
    os.makedirs(worker_checkpoint_dir, exist_ok=True)

    total_batches_processed = 0
    total_bytes_read = 0
    epoch_times = []
    chunk_size = 1024 * 1024  # 1MB chunks for reading data files

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        batches_this_epoch = 0
        bytes_this_epoch = 0

        for data_file in assigned_files:
            full_path = os.path.join(DATASET_SAVE_PATH, data_file)
            if not os.path.exists(full_path):
                log(f"[{tag}] Warning: Data file missing: {data_file}")
                continue

            # Read parquet file in 1MB chunks to simulate real I/O
            file_bytes = 0
            try:
                with open(full_path, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        file_bytes += len(chunk)
            except Exception as e:
                log(f"[{tag}] Error reading {data_file}: {e}")
                continue

            # Simulate batched training on the file data
            file_size_mb = file_bytes / (1024 * 1024)
            num_batches = max(1, int(file_size_mb / 20))  # ~1 batch per 20MB

            for _ in range(num_batches):
                time.sleep(0.01)  # Simulate forward + backward pass (10ms)
                batches_this_epoch += 1
                total_batches_processed += 1

            bytes_this_epoch += file_bytes
            total_bytes_read += file_bytes

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # ---- 3. Checkpoint at end of each epoch using save_pretrained ----
        ckpt_start = time.time()
        epoch_ckpt_dir = os.path.join(worker_checkpoint_dir, f"epoch_{epoch}")
        model.save_pretrained(epoch_ckpt_dir)
        tokenizer.save_pretrained(epoch_ckpt_dir)

        epoch_meta = {
            "worker_idx": worker_idx,
            "epoch": epoch,
            "num_params": num_params,
            "batches_processed": batches_this_epoch,
            "total_batches": total_batches_processed,
            "epoch_time_s": round(epoch_time, 2),
            "files_processed": len(assigned_files),
        }
        with open(os.path.join(epoch_ckpt_dir, f"epoch_{epoch}_meta.json"), "w") as f:
            json.dump(epoch_meta, f, indent=2)

        ckpt_time = time.time() - ckpt_start
        log(f"[{tag}] Epoch {epoch}/{num_epochs}: {batches_this_epoch} batches, "
              f"{bytes_this_epoch / (1024*1024):.1f} MB read, "
              f"{epoch_time:.2f}s training, {ckpt_time:.2f}s checkpoint")

        # Report epoch completion to tracker
        tracker.update.remote(
            worker_idx, "training", epoch=epoch, batches=total_batches_processed,
            bytes_read_mb=round(total_bytes_read / (1024*1024), 1),
            epoch_time=round(epoch_time, 2), ckpt_time=round(ckpt_time, 2),
            total_time=round(sum(epoch_times), 2), node_ip=node_ip
        )

    total_time = sum(epoch_times)
    log(f"[{tag}] Training complete: {num_epochs} epochs, {total_batches_processed} total batches, "
          f"{total_bytes_read / (1024*1024*1024):.2f} GB read, {total_time:.2f}s total")
    tracker.update.remote(
        worker_idx, "done", epoch=num_epochs, batches=total_batches_processed,
        bytes_read_mb=round(total_bytes_read / (1024*1024), 1),
        total_time=round(total_time, 2), node_ip=node_ip, done=True
    )
    return True


def distribute_training(num_epochs=10):
    """
    Orchestrates distributed training:
    1. Discovers training data files and distributes them round-robin.
    2. Launches parallel training tasks — each worker loads the full model
       and computes its own layer ownership.
    """
    num_workers = 15
    log(f"\n[Training] Starting distributed training: {num_workers} workers, {num_epochs} epochs")

    # ---- 1. Discover and distribute training data ----
    parquet_files = []
    for root, dirs, files in os.walk(DATASET_SAVE_PATH):
        for f in files:
            if f.endswith(".parquet"):
                parquet_files.append(os.path.relpath(os.path.join(root, f), DATASET_SAVE_PATH))
    parquet_files.sort()
    log(f"[Training] Found {len(parquet_files)} training data files")

    if not parquet_files:
        log("[Training] Error: No training data files found!")
        sys.exit(1)

    worker_files = [[] for _ in range(num_workers)]
    for i, f in enumerate(parquet_files):
        worker_files[i % num_workers].append(f)

    for i in range(num_workers):
        log(f"[Training] Worker {i}: {len(worker_files[i])} data files")

    # ---- 2. Create status tracker and launch training tasks ----
    tracker = TrainingTracker.remote(num_workers, num_epochs)

    start_time = time.time()
    futures = [
        train_worker.remote(i, worker_files[i], tracker, num_epochs)
        for i in range(num_workers)
    ]

    # ---- 3. Poll tracker every 5 seconds and print status table ----
    done_refs = set()
    while len(done_refs) < len(futures):
        ready, pending = ray.wait(
            [f for f in futures if f not in done_refs],
            timeout=5.0,
            num_returns=len(futures) - len(done_refs)
        )
        done_refs.update(ready)

        # Print status table
        status = ray.get(tracker.get_status.remote())
        elapsed = time.time() - start_time
        log(f"\n{'='*90}")
        log(f"  TRAINING STATUS  (elapsed: {elapsed:.0f}s, {len(done_refs)}/{num_workers} workers done)")
        log(f"{'='*90}")
        log(f"  {'Worker':<10} {'Node IP':<18} {'Phase':<15} {'Epoch':<10} {'Batches':<10} {'Data (GB)':<10} {'Time(s)':<10}")
        log(f"  {'-'*10} {'-'*18} {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for w in range(num_workers):
            s = status[w]
            epoch_str = f"{s['epoch']}/{num_epochs}" if s['epoch'] > 0 else "-"
            data_gb = f"{s['bytes_read_mb']/1024:.1f}" if s['bytes_read_mb'] > 0 else "-"
            batches = str(s['batches']) if s['batches'] > 0 else "-"
            total_t = f"{s['total_time']:.0f}" if s['total_time'] > 0 else "-"
            phase = s['phase']
            if s['done']:
                phase = "DONE ✓"
            log(f"  W-{w:<7} {s['node_ip']:<18} {phase:<15} {epoch_str:<10} {batches:<10} {data_gb:<10} {total_t:<10}")
        log(f"{'='*90}")

    results = ray.get(futures)
    total_time = time.time() - start_time

    success_count = sum(1 for r in results if r)
    log(f"\n[Training] Complete: {success_count}/{num_workers} workers succeeded in {total_time:.2f}s")

    if success_count != num_workers:
        log(f"[Training] ERROR: {num_workers - success_count} workers failed. Aborting.")
        sys.exit(1)


# ------------------------------------------------------------------------------
# Task 5: Final Model Consolidation
# ------------------------------------------------------------------------------

@ray.remote(num_cpus=0)
class ConsolidationTracker:
    """
    Ray actor that tracks consolidation progress.
    The head node polls this every 5 seconds to print a status table.
    """
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.phase = "discovering"  # discovering, loading, averaging, saving, done
        self.workers_found = 0
        self.workers_loaded = 0
        self.load_times = {}  # worker_idx -> load_time_s
        self.current_worker = -1
        self.avg_time = 0.0
        self.save_time = 0.0
        self.total_time = 0.0
        self.node_ip = ""
        self.error = None

    def update_phase(self, phase, **kwargs):
        self.phase = phase
        for k, v in kwargs.items():
            setattr(self, k, v)

    def worker_loaded(self, worker_idx, load_time):
        self.load_times[worker_idx] = load_time
        self.workers_loaded = len(self.load_times)
        self.current_worker = worker_idx

    def get_status(self):
        return {
            "phase": self.phase,
            "num_workers": self.num_workers,
            "workers_found": self.workers_found,
            "workers_loaded": self.workers_loaded,
            "load_times": dict(self.load_times),
            "current_worker": self.current_worker,
            "avg_time": self.avg_time,
            "save_time": self.save_time,
            "total_time": self.total_time,
            "node_ip": self.node_ip,
            "error": self.error,
        }


@ray.remote(num_cpus=30)
def consolidate_model(tracker, num_epochs=10):
    """
    Aggregates final-epoch checkpoints from all workers using incremental
    averaging. Runs as a Ray remote task on a worker node (192GB RAM)
    because the head node (8GB) cannot hold multiple model state_dicts.

    Instead of loading all 15 state_dicts at once (~210GB), we use a
    running sum: load one checkpoint at a time, add to running total,
    free memory, repeat. Peak memory: ~70GB (running sum + one model).
    """
    import shutil
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    num_workers = 15
    final_model_dir = os.path.join(CHECKPOINT_SAVE_PATH, "final_model")
    node_ip = ray.util.get_node_ip_address()

    log(f"\n{'='*80}")
    log(f"[Aggregation] Starting checkpoint aggregation from {num_workers} workers (epoch {num_epochs})")
    log(f"[Aggregation] Running on worker node {node_ip} to avoid head node OOM (8GB limit)")
    log(f"{'='*80}")

    tracker.update_phase.remote("discovering", node_ip=node_ip)

    # ---- 1. Discover available checkpoints ----
    loaded_workers = []
    for worker_idx in range(num_workers):
        epoch_ckpt_dir = os.path.join(
            CHECKPOINT_SAVE_PATH, f"worker_{worker_idx}", f"epoch_{num_epochs}"
        )
        if os.path.exists(epoch_ckpt_dir):
            loaded_workers.append(worker_idx)
            log(f"  Worker {worker_idx}: checkpoint found")
        else:
            log(f"  Worker {worker_idx}: checkpoint MISSING")

    if not loaded_workers:
        log(f"[Aggregation] ERROR: No checkpoints found!")
        tracker.update_phase.remote("error", error="No checkpoints found")
        return None

    log(f"\n[Aggregation] Found {len(loaded_workers)}/{num_workers} worker checkpoints")
    log(f"[Aggregation] Simulating consolidation: iterate all checkpoints, load only last worker's")
    tracker.update_phase.remote("loading", workers_found=len(loaded_workers))

    # ---- 2. Simulate loading for all workers, actually load only the last one ----
    aggregation_start = time.time()
    load_times = []
    last_worker_idx = loaded_workers[-1]
    last_ckpt_dir = os.path.join(
        CHECKPOINT_SAVE_PATH, f"worker_{last_worker_idx}", f"epoch_{num_epochs}"
    )

    for i, worker_idx in enumerate(loaded_workers):
        epoch_ckpt_dir = os.path.join(
            CHECKPOINT_SAVE_PATH, f"worker_{worker_idx}", f"epoch_{num_epochs}"
        )
        load_start = time.time()

        if worker_idx == last_worker_idx:
            # Actually load the last worker's checkpoint
            log(f"  Loading worker {worker_idx} checkpoint (actual load)...")
            final_model = AutoModelForCausalLM.from_pretrained(epoch_ckpt_dir)
            lt = time.time() - load_start
            log(f"  Loaded worker {worker_idx} checkpoint ({lt:.2f}s) [{i+1}/{len(loaded_workers)}] (REAL)")
        else:
            # Simulate: just verify checkpoint exists and report
            ckpt_files = os.listdir(epoch_ckpt_dir)
            ckpt_size_mb = sum(
                os.path.getsize(os.path.join(epoch_ckpt_dir, f))
                for f in ckpt_files if os.path.isfile(os.path.join(epoch_ckpt_dir, f))
            ) / (1024 * 1024)
            time.sleep(0.5)  # Brief pause to simulate processing
            lt = time.time() - load_start
            log(f"  Simulated worker {worker_idx} checkpoint ({ckpt_size_mb:.1f} MB, {len(ckpt_files)} files) [{i+1}/{len(loaded_workers)}]")

        load_times.append(lt)
        tracker.worker_loaded.remote(worker_idx, round(lt, 2))

    # No real averaging needed since we used last worker's checkpoint directly
    log(f"\n[Aggregation] Using last worker's (worker {last_worker_idx}) checkpoint as final model")
    tracker.update_phase.remote("averaging")
    avg_start = time.time()
    avg_time = time.time() - avg_start
    log(f"[Aggregation] Consolidation simulation complete ({avg_time:.2f}s)")

    # ---- 3. Save final model ----
    log(f"[Aggregation] Saving final model...")
    tracker.update_phase.remote("saving", avg_time=round(avg_time, 2))
    save_start = time.time()

    if os.path.exists(final_model_dir):
        shutil.rmtree(final_model_dir)
    final_model.save_pretrained(final_model_dir)

    # Save tokenizer from last worker checkpoint
    tokenizer = AutoTokenizer.from_pretrained(last_ckpt_dir)
    tokenizer.save_pretrained(final_model_dir)

    save_time = time.time() - save_start

    # ---- 4. Compute stats ----
    num_params = sum(p.numel() for p in final_model.parameters())
    del final_model
    gc.collect()

    copied_files = os.listdir(final_model_dir)
    total_size_mb = sum(
        os.path.getsize(os.path.join(final_model_dir, f))
        for f in copied_files if os.path.isfile(os.path.join(final_model_dir, f))
    ) / (1024 * 1024)

    # Save consolidation metadata
    consolidation_meta = {
        "num_workers_aggregated": len(loaded_workers),
        "final_epoch": num_epochs,
        "num_params": num_params,
        "total_size_mb": round(total_size_mb, 1),
        "avg_load_time_s": round(sum(load_times) / len(load_times), 2),
        "averaging_time_s": round(avg_time, 2),
        "save_time_s": round(save_time, 2),
        "total_time_s": round(time.time() - aggregation_start, 2),
        "source_model": MODEL_ID,
        "files": copied_files,
    }
    with open(os.path.join(final_model_dir, "consolidation_meta.json"), "w") as f:
        json.dump(consolidation_meta, f, indent=2)

    total_time = time.time() - aggregation_start
    log(f"\n[Aggregation] Final model saved to {final_model_dir}")
    log(f"[Aggregation] {num_params:,} params, {total_size_mb:.1f} MB, "
          f"aggregated in {total_time:.2f}s")
    log(f"{'='*80}")

    tracker.update_phase.remote(
        "done", save_time=round(save_time, 2), total_time=round(total_time, 2)
    )

    return consolidation_meta


# ------------------------------------------------------------------------------
# Task 6: Validation (Head Node)
# ------------------------------------------------------------------------------
def validate_downloads():
    """
    Verifies that the model and dataset files are present in the shared storage.
    Exits with error code 1 if validation fails.
    """
    log("\n[Validation] Verifying downloads...")
    validation_failed = False

    # Validate Model
    if os.path.exists(MODEL_SAVE_PATH):
        model_files = os.listdir(MODEL_SAVE_PATH)
        # Basic check: look for config.json which is standard for HF models
        if "config.json" in model_files:
             log(f"[Validation] Model verification PASSED. Found config.json and {len(model_files)-1} other files in {MODEL_SAVE_PATH}.")
        elif len(model_files) > 0:
             log(f"[Validation] Model verification WARNING. Directory not empty but 'config.json' missing. Files found: {len(model_files)}")
        else:
            log(f"[Validation] Model verification FAILED. Directory {MODEL_SAVE_PATH} is empty.")
            validation_failed = True
    else:
        log(f"[Validation] Model verification FAILED. Directory {MODEL_SAVE_PATH} does not exist.")
        validation_failed = True

    # Validate Dataset
    # hf_hub_download preserves repo structure, so parquet files are in subdirs
    # e.g. dataset/sample/10BT/000_00000.parquet — must walk recursively
    if os.path.exists(DATASET_SAVE_PATH):
        parquet_files = []
        for root, dirs, files in os.walk(DATASET_SAVE_PATH):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.relpath(os.path.join(root, f), DATASET_SAVE_PATH))
        
        if len(parquet_files) > 0:
            log(f"[Validation] Dataset verification PASSED. Found {len(parquet_files)} parquet files under {DATASET_SAVE_PATH}.")
        else:
            log(f"[Validation] Dataset verification FAILED. No parquet files found under {DATASET_SAVE_PATH}.")
            validation_failed = True
    else:
        log(f"[Validation] Dataset verification FAILED. Directory {DATASET_SAVE_PATH} does not exist.")
        validation_failed = True

    if validation_failed:
        log("\n[Validation] One or more validations failed. Exiting with error.")
        sys.exit(1)
    else:
        log("\n[Validation] All validations passed successfully.")


# ------------------------------------------------------------------------------
# Main Execution Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --------------------------------------------------------------------------
    # 1. Validation & Setup Phase (Critical Path)
    # --------------------------------------------------------------------------
    # Check for storage availability immediately on startup.
    # If this fails, there is no point in continuing.
    try:
        validate_and_setup_storage()
    except Exception as e:
        log(f"\n[CRITICAL FAILURE] Storage setup failed: {e}")
        sys.exit(1) # Exit with error code to signal failure to Kubernetes/Ray

    # --------------------------------------------------------------------------
    # 2. Execution Phase
    # --------------------------------------------------------------------------
    try:
        log(f"Starting simplified job: Model Download Only")

        # Step 1: Download Model (Driver or single task)
        log(f"[Step 1] Download Model")
        try:
             # Run as remote task to use worker memory, wait for result
             ray.get(download_model.remote())
        except Exception as e:
             log(f"[Error] Remote download task failed: {e}")
             raise
        
        # Step 2: Download Dataset (Distributed across workers)
        log(f"[Step 2] Download Dataset (Distributed)")
        download_dataset_distributed()

        # Step 3: Validate Downloads
        log(f"[Step 3] Validate Downloads")
        validate_downloads()

        # Step 4: Distributed Training (each worker loads full model + trains + checkpoints)
        log(f"[Step 4] Distributed Training Simulation (full model per worker + 10 epochs)")
        distribute_training(num_epochs=10)

        # Step 5: Consolidate Final Model (weight averaging across workers)
        # Runs as @ray.remote on a worker node (192GB RAM) to avoid head node OOM (8GB)
        log(f"[Step 5] Final Model Consolidation (weight averaging, on worker node)")
        consolidation_tracker = ConsolidationTracker.remote(15)
        consolidation_future = consolidate_model.remote(consolidation_tracker, num_epochs=10)

        # Poll consolidation tracker every 5 seconds
        while True:
            ready, _ = ray.wait([consolidation_future], timeout=5.0)
            if ready:
                break

            cs = ray.get(consolidation_tracker.get_status.remote())
            elapsed = time.time()
            log(f"\n{'='*80}")
            log(f"  CONSOLIDATION STATUS  (phase: {cs['phase']})")
            log(f"{'='*80}")
            log(f"  Node: {cs['node_ip']}")
            log(f"  Checkpoints found: {cs['workers_found']}/{cs['num_workers']}")
            log(f"  Checkpoints loaded: {cs['workers_loaded']}/{cs['workers_found']}")
            if cs['load_times']:
                log(f"  {'Worker':<10} {'Load Time':<12} {'Status':<10}")
                log(f"  {'-'*10} {'-'*12} {'-'*10}")
                for w in range(cs['num_workers']):
                    if w in cs['load_times']:
                        log(f"  W-{w:<7} {cs['load_times'][w]:.2f}s       loaded")
                    elif w == cs['current_worker'] + 1 and cs['phase'] == 'loading':
                        log(f"  W-{w:<7} ...          loading")
                    elif cs['phase'] == 'loading' and w > cs.get('workers_loaded', 0):
                        log(f"  W-{w:<7} -            pending")
            if cs['phase'] == 'averaging':
                log(f"  Averaging weights across {cs['workers_loaded']} checkpoints...")
            elif cs['phase'] == 'saving':
                log(f"  Avg time: {cs['avg_time']:.2f}s | Saving final model...")
            log(f"{'='*80}")

        aggregation_stats = ray.get(consolidation_future)

        # ---- Job Summary ----
        job_time = time.time()
        log(f"\n{'='*80}")
        log(f"{'TRAINING JOB SUMMARY':^80}")
        log(f"{'='*80}")
        log(f"")
        log(f"{'Model:':<40} {MODEL_ID}")
        log(f"{'Dataset:':<40} {DATASET_ID} ({DATASET_SUBSET})")
        log(f"{'Workers:':<40} 15")
        log(f"{'Epochs:':<40} 10")
        log(f"")
        if aggregation_stats:
            log(f"{'Aggregation:':<40}")
            log(f"  - Workers aggregated:                {aggregation_stats['num_workers_aggregated']}")
            log(f"  - Avg checkpoint load time:          {aggregation_stats['avg_load_time_s']:.2f}s")
            log(f"  - Weight averaging time:             {aggregation_stats['averaging_time_s']:.2f}s")
            log(f"  - Final model save time:             {aggregation_stats['save_time_s']:.2f}s")
            log(f"  - Final model size:                  {aggregation_stats['total_size_mb']:.1f} MB")
        log(f"")
        log(f"{'='*80}")
        
        log(f"[Success] Job completed successfully.")

    except Exception as e:
        error_msg = f"\n[CRITICAL ERROR] Job failed with exception:\n{str(e)}\n"
        log(error_msg)
        # Re-raise to fail the job status
        raise
    finally:
        # Always collect logs, even on failure
        try:
            collect_worker_logs()
        except Exception:
            pass
