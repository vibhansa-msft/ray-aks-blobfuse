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
# The script assumes a shared volume is mounted at '/mnt/cluster_storage'.
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
# Check env var first, default to /mnt/cluster_storage
STORAGE_PATH = os.environ.get("STORAGE_PATH", "/mnt/cluster_storage")

# ------------------------------------------------------------------------------
# Logging Helper & Stdout Redirection
# ------------------------------------------------------------------------------
LOG_FILE = os.path.join(STORAGE_PATH, f"job_execution_{int(time.time())}.log")

class FileLogger(object):
    """
    Writes ONLY to the shared log file, suppressing console output.
    Delegates fileno/isatty/etc to the original stream so that libraries
    like faulthandler (used by Ray) still work.
    """
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.log_file = LOG_FILE

    def write(self, message):
        # Write to BOTH original stream (Anyscale portal) and shared log file
        try:
            self.original_stream.write(message)
            self.original_stream.flush()
        except Exception:
            pass
        try:
            with open(self.log_file, "a") as f:
                f.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return False

# Redirect stdout and stderr
sys.stdout = FileLogger(sys.stdout)
sys.stderr = FileLogger(sys.stderr)

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
    print(f"[Init] Checking storage path: {STORAGE_PATH}")
    
    # 2. Critical Check: Ensure the Persistent Volume is mounted
    if not os.path.exists(STORAGE_PATH):
        error_msg = f"[CRITICAL] Storage path {STORAGE_PATH} not found! Ensure PV is mounted."
        print(error_msg)
        # We cannot log this to the file because the storage path doesn't exist
        # Raising an error here prevents the script from silently failing later
        raise FileNotFoundError(error_msg)
    
    print(f"[Init] Storage path confirmed: {STORAGE_PATH}")
    
    # 3. Setup: Create required subdirectories if they don't exist
    try:
        print(f"[Init] Creating directories in {STORAGE_PATH}...")
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(DATASET_SAVE_PATH, exist_ok=True)
        os.makedirs(CHECKPOINT_SAVE_PATH, exist_ok=True)
        print(f"[Init] Directories created successfully.")
    except Exception as e:
        print(f"[CRITICAL] Failed to create directories: {e}")
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
print("\n[Init] Initializing Ray Cluster Connection...")

# Check if Ray is already running (e.g., if run via 'ray job submit')
if ray.is_initialized():
    print("[Init] Ray is already initialized.")
else:
    # Connect to the existing Ray cluster
    ray.init(
        address="auto",     # Connect to the cluster Ray instance (auto-detect)
        log_to_driver=True  # ensures worker logs are forwarded)
    ) 

# Log available resources to verify cluster size
resources = ray.available_resources()
print(f"[Init] Connected. Cluster Resources: {resources}")


# ------------------------------------------------------------------------------
# Task 1: Model Download (Centralized)
# ------------------------------------------------------------------------------
@ray.remote(num_cpus=10)
def download_model():
    """
    Downloads the entire model repository to shared storage.
    Runs as a remote task to offload memory usage from the head node.
    """
    print(f"\n[Model] Checking storage path: {STORAGE_PATH}")
    
    # Path validation is now handled in validate_and_setup_storage()

    print(f"[Model] Downloading {MODEL_ID} to {MODEL_SAVE_PATH}...")
    
    # Log parameters for debugging
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        # Mask the token for security
        masked_token = f"{hf_token[:4]}...{hf_token[-4:]}"
    else:
        masked_token = "NOT_SET"
        
    print(f"[Model] Parameters: repo_id={MODEL_ID}, local_dir={MODEL_SAVE_PATH}, token={masked_token}")

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
        print(f"[Model] Success! Model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"[Model] Error downloading model: {e}")
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

    print(f"[{worker_tag}] Starting batch of {len(filenames)} files")
             
    for filename in filenames:
        try:
            print(f"[{worker_tag}] Processing: {filename}")
            file_path = os.path.join(DATASET_SAVE_PATH, filename)
            
            # Simple check to avoid re-downloading existing files
            if os.path.exists(file_path):
                 fsize = os.path.getsize(file_path)
                 print(f"[{worker_tag}] Skipped (exists, {fsize/(1024*1024):.1f} MB): {filename}")
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
            print(f"[{worker_tag}] Downloaded ({fsize/(1024*1024):.1f} MB): {filename}")
            file_names.append(filename)
            statuses.append("downloaded")
            size_bytes_list.append(fsize)
        except Exception as e:
            print(f"[{worker_tag}] Failed: {filename} - {e}")
            file_names.append(filename)
            statuses.append("failed")
            size_bytes_list.append(0)
    
    batch_total_mb = sum(size_bytes_list) / (1024 * 1024)
    print(f"[{worker_tag}] Batch complete: {len(file_names)} files, {batch_total_mb:.1f} MB total")
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
    print(f"\n[Dataset] Fetching file list for {DATASET_ID} ({DATASET_SUBSET})...")
    
    # List all files in the remote repository
    try:
        all_files = list_repo_files(
            repo_id=DATASET_ID, 
            repo_type="dataset", 
            token=os.environ.get("HF_TOKEN")
        )
    except Exception as e:
        print(f"[Dataset] Error listing repo files: {e}")
        return

    print(f"[Dataset] Total files in repo: {len(all_files)}")

    # Filter for parquet files in the specific subset
    target_files = [f for f in all_files if f.endswith(".parquet") and DATASET_SUBSET in f]
    
    print(f"[Dataset] Filtered to {len(target_files)} parquet files matching '{DATASET_SUBSET}'.")
    print("[Dataset] Triggering distributed download via Ray Data...")
    
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
    print(f"[Dataset] Distributing {num_files} files in batches of {batch_sz}")
    
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
    
    print(f"[Dataset] Summary: Processed {len(all_results)} files.")
    print(f"[Dataset] Total data volume: {total_gb:.2f} GB ({total_bytes:,} bytes)")
    print(f"[Dataset] Successfully downloaded {success_count}/{len(target_files)} files in {duration:.2f} seconds.")
    print(f"[Dataset] Throughput: {total_gb/duration*1024:.1f} MB/s" if duration > 0 else "")
    if failed_count > 0:
        failed_files = [r["file"] for r in all_results if r["status"] == "failed"]
        print(f"[Dataset] WARNING: {failed_count} files failed: {failed_files}")


# ------------------------------------------------------------------------------
# Task 4: Distributed Model Sharding & Checkpoint (Pipeline-Parallel Simulation)
# ------------------------------------------------------------------------------

@ray.remote(num_cpus=30) # Reserve 30 vCPUs per task to ensure one task per node
def load_and_checkpoint_worker(node_idx, num_workers, total_layers=32):
    """
    Simulates pipeline-parallel distributed training:
    Each worker loads only its assigned model layers (shard) from safetensors,
    then saves the shard as a checkpoint.

    Mistral-7B has 32 transformer layers + embedding + norm + lm_head.
    With 20 workers: 12 workers get 2 layers, 8 workers get 1 layer.
    Worker 0 also gets embed_tokens. Last worker gets model.norm + lm_head.
    """
    import json
    from safetensors.torch import load_file, save_file

    node_ip = ray.util.get_node_ip_address()
    tag = f"Shard-Worker-{node_idx}@{node_ip}"
    print(f"\n[{tag}] Starting pipeline-parallel shard loading...")

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[{tag}] Error: Model path {MODEL_SAVE_PATH} does not exist.")
        return False

    try:
        # ---- 1. Calculate layer assignment (even distribution) ----
        layers_per_worker = total_layers // num_workers
        remainder = total_layers % num_workers

        if node_idx < remainder:
            start = node_idx * (layers_per_worker + 1)
            count = layers_per_worker + 1
        else:
            start = remainder * (layers_per_worker + 1) + (node_idx - remainder) * layers_per_worker
            count = layers_per_worker

        layer_ids = list(range(start, start + count))
        print(f"[{tag}] Assigned layers: {layer_ids} ({count} layers)")

        # ---- 2. Read safetensors weight index ----
        index_file = os.path.join(MODEL_SAVE_PATH, "model.safetensors.index.json")
        if not os.path.exists(index_file):
            print(f"[{tag}] Error: Weight index not found at {index_file}")
            return False

        with open(index_file) as f:
            weight_index = json.load(f)

        weight_map = weight_index["weight_map"]

        # ---- 3. Identify weight keys for our assigned layers ----
        our_keys = set()
        for key in weight_map:
            for layer_id in layer_ids:
                if f"model.layers.{layer_id}." in key:
                    our_keys.add(key)
                    break

        # Worker 0 also gets the embedding layer
        if node_idx == 0:
            for key in weight_map:
                if "embed_tokens" in key:
                    our_keys.add(key)

        # Last worker gets model.norm and lm_head
        if node_idx == num_workers - 1:
            for key in weight_map:
                if "model.norm" in key or "lm_head" in key:
                    our_keys.add(key)

        print(f"[{tag}] Loading {len(our_keys)} weight tensors...")

        # ---- 4. Load only our weights from safetensors files ----
        # Identify which safetensors files contain our keys
        our_files = set(weight_map[k] for k in our_keys)
        print(f"[{tag}] Reading from {len(our_files)} safetensors file(s): {sorted(our_files)}")

        load_start = time.time()
        our_weights = {}
        for sf_file in sorted(our_files):
            full_path = os.path.join(MODEL_SAVE_PATH, sf_file)
            all_tensors = load_file(full_path)
            for key in our_keys:
                if key in all_tensors:
                    our_weights[key] = all_tensors[key]
            del all_tensors  # Free memory for tensors we don't need

        load_time = time.time() - load_start
        num_params = sum(t.numel() for t in our_weights.values())
        mem_mb = sum(t.nbytes for t in our_weights.values()) / (1024 * 1024)
        print(f"[{tag}] Loaded {len(our_weights)} tensors ({num_params:,} params, {mem_mb:.1f} MB) in {load_time:.2f}s")

        # ---- 5. Save shard checkpoint ----
        shard_path = os.path.join(CHECKPOINT_SAVE_PATH, f"shard_{node_idx}")
        os.makedirs(shard_path, exist_ok=True)

        save_start = time.time()
        save_file(our_weights, os.path.join(shard_path, f"model_shard_{node_idx}.safetensors"))

        # Save shard metadata
        shard_info = {
            "node_idx": node_idx,
            "node_ip": node_ip,
            "layers": layer_ids,
            "num_tensors": len(our_weights),
            "num_params": num_params,
            "size_mb": round(mem_mb, 1),
            "weight_keys": sorted(our_weights.keys()),
            "source_files": sorted(our_files),
        }
        with open(os.path.join(shard_path, "shard_info.json"), "w") as f:
            json.dump(shard_info, f, indent=2)

        save_time = time.time() - save_start
        print(f"[{tag}] Saved shard checkpoint to {shard_path} in {save_time:.2f}s")
        print(f"[{tag}] Done. Layers={layer_ids}, Params={num_params:,}, Load={load_time:.2f}s, Save={save_time:.2f}s")
        return True

    except Exception as e:
        print(f"[{tag}] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def distribute_model_loading():
    """
    Launches 20 parallel tasks (one per D48 node) to load model shards
    and save checkpoint shards. Simulates pipeline-parallel model distribution.
    """
    num_workers = 15
    total_layers = 32  # Mistral-7B has 32 transformer layers
    print(f"\n[Checkpoint] Distributing {total_layers}-layer model across {num_workers} workers (pipeline-parallel)...")
    print(f"[Checkpoint] Each worker loads its assigned layers from safetensors and saves a shard checkpoint.")

    futures = [load_and_checkpoint_worker.remote(i, num_workers, total_layers) for i in range(num_workers)]

    results = ray.get(futures)
    success_count = sum(results)

    if success_count == num_workers:
        print(f"[Checkpoint] All {num_workers} workers successfully loaded and saved shard checkpoints.")
    else:
        print(f"[Checkpoint] Only {success_count}/{num_workers} workers succeeded. Failing job.")
        sys.exit(1)


# ------------------------------------------------------------------------------
# Task 5: Validation (Head Node)
# ------------------------------------------------------------------------------
def validate_downloads():
    """
    Verifies that the model and dataset files are present in the shared storage.
    Exits with error code 1 if validation fails.
    """
    print("\n[Validation] Verifying downloads...")
    validation_failed = False

    # Validate Model
    if os.path.exists(MODEL_SAVE_PATH):
        model_files = os.listdir(MODEL_SAVE_PATH)
        # Basic check: look for config.json which is standard for HF models
        if "config.json" in model_files:
             print(f"[Validation] Model verification PASSED. Found config.json and {len(model_files)-1} other files in {MODEL_SAVE_PATH}.")
        elif len(model_files) > 0:
             print(f"[Validation] Model verification WARNING. Directory not empty but 'config.json' missing. Files found: {len(model_files)}")
        else:
            print(f"[Validation] Model verification FAILED. Directory {MODEL_SAVE_PATH} is empty.")
            validation_failed = True
    else:
        print(f"[Validation] Model verification FAILED. Directory {MODEL_SAVE_PATH} does not exist.")
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
            print(f"[Validation] Dataset verification PASSED. Found {len(parquet_files)} parquet files under {DATASET_SAVE_PATH}.")
        else:
            print(f"[Validation] Dataset verification FAILED. No parquet files found under {DATASET_SAVE_PATH}.")
            validation_failed = True
    else:
        print(f"[Validation] Dataset verification FAILED. Directory {DATASET_SAVE_PATH} does not exist.")
        validation_failed = True

    if validation_failed:
        print("\n[Validation] One or more validations failed. Exiting with error.")
        sys.exit(1)
    else:
        print("\n[Validation] All validations passed successfully.")


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
        sys.stderr.write(f"\n[CRITICAL FAILURE] Storage setup failed: {e}\n")
        sys.exit(1) # Exit with error code to signal failure to Kubernetes/Ray

    # --------------------------------------------------------------------------
    # 2. Execution Phase
    # --------------------------------------------------------------------------
    try:
        print(f"Starting simplified job: Model Download Only")

        # Step 1: Download Model (Driver or single task)
        print(f"[Step 1] Download Model")
        try:
             # Run as remote task to use worker memory, wait for result
             ray.get(download_model.remote())
        except Exception as e:
             print(f"[Error] Remote download task failed: {e}")
             raise
        
        # Step 2: Download Dataset (Distributed across workers)
        print(f"[Step 2] Download Dataset (Distributed)")
        download_dataset_distributed()

        # Step 3: Validate Downloads
        print(f"[Step 3] Validate Downloads")
        validate_downloads()

        # Step 4: Distributed Load & Checkpoint (10 workers)
        print(f"[Step 4] Distributed Load & Checkpoint")
        distribute_model_loading()
        
        print(f"[Success] Job completed successfully.")

    except Exception as e:
        error_msg = f"\n[CRITICAL ERROR] Job failed with exception:\n{str(e)}\n"
        print(error_msg)
        # Re-raise to fail the job status
        raise
