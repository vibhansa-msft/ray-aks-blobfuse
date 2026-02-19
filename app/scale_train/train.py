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
        # Write ONLY to shared log file
        try:
            with open(self.log_file, "a") as f:
                f.write(message)
        except Exception:
            pass

    def flush(self):
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
    ray.init(address="auto")

# Log available resources to verify cluster size
resources = ray.available_resources()
print(f"[Init] Connected. Cluster Resources: {resources}")


# ------------------------------------------------------------------------------
# Task 1: Model Download (Centralized)
# ------------------------------------------------------------------------------
@ray.remote(num_cpus=1)
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
                 print(f"[{worker_tag}] Skipped (exists): {filename}")
                 file_names.append(filename)
                 statuses.append("skipped")
                 continue

            # Download specific file to the shared dataset folder
            hf_hub_download(
                repo_id=DATASET_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=DATASET_SAVE_PATH,
                token=token_arg
            )
            print(f"[{worker_tag}] Downloaded: {filename}")
            file_names.append(filename)
            statuses.append("downloaded")
        except Exception as e:
            print(f"[{worker_tag}] Failed: {filename} - {e}")
            file_names.append(filename)
            statuses.append("failed")
    
    print(f"[{worker_tag}] Batch complete: {len(file_names)} files processed")
    # Return columnar format: dict of lists (required by Ray Data map_batches)
    return {"file": file_names, "status": statuses}

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
    
    # Use map_batches to process files in parallel. 
    # With ~460 files, batch_size=20 yields ~23 batches.
    # concurrency=20 ensures 20 tasks run in parallel across 10+ nodes.
    # num_cpus=4 per task → max 12 tasks per D48 node (48 vCPUs),
    # so 20 concurrent tasks need at least ~2 nodes, but Ray scheduler
    # spreads them across available nodes for better I/O parallelism.
    downloaded_ds = ds.map_batches(
        download_file_wrapper, 
        batch_size=20,      # Process 20 files per task for bigger batches
        num_cpus=4,          # Reserve 4 CPUs per task
        concurrency=20       # Run 20 tasks in parallel across the cluster
    )
    
    # Force execution and collect results
    # take_all() triggers the computation and returns list of row dicts
    all_results = downloaded_ds.take_all()
    
    duration = time.time() - start_time
    
    # Each row is a dict like {"file": "...", "status": "downloaded|skipped|failed"}
    success_count = sum(1 for r in all_results if r["status"] != "failed")
    failed_count = sum(1 for r in all_results if r["status"] == "failed")
    
    print(f"[Dataset] Summary: Processed {len(all_results)} files.")
    print(f"[Dataset] Successfully downloaded {success_count}/{len(target_files)} files in {duration:.2f} seconds.")
    if failed_count > 0:
        failed_files = [r["file"] for r in all_results if r["status"] == "failed"]
        print(f"[Dataset] WARNING: {failed_count} files failed: {failed_files}")


# # ------------------------------------------------------------------------------
# # Task 4: Load & Checkpoint (Distributed)
# # ------------------------------------------------------------------------------

# @ray.remote(num_cpus=4) # Reserve enough CPU to load model in memory without OOM
# def load_and_checkpoint_worker(node_idx):
#     """
#     Worker task: Loads the model from shared storage and saves a checkpoint
#     to a unique node-specific directory.
#     """
#     node_ip = ray.util.get_node_ip_address()
#     print(f"\n[Checkpoint-Worker-{node_idx}] Starting on Node IP: {node_ip}")

#     # Unique checkpoint path for this worker to avoid write conflicts
#     node_checkpoint_path = os.path.join(CHECKPOINT_SAVE_PATH, f"node_{node_idx}_{node_ip}")

#     if not os.path.exists(MODEL_SAVE_PATH):
#         print(f"[Checkpoint-Worker-{node_idx}] Error: Model path {MODEL_SAVE_PATH} does not exist.")
#         return False

#     try:
#         print(f"[Checkpoint-Worker-{node_idx}] Loading model from {MODEL_SAVE_PATH}...")
#         start_time = time.time()
        
#         # Load model to CPU
#         # low_cpu_mem_usage=True helps with loading large models
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_SAVE_PATH, 
#             torch_dtype=torch.float32, 
#             low_cpu_mem_usage=True,
#             device_map="cpu",
#             token=os.environ.get("HF_TOKEN")
#         )
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH, token=os.environ.get("HF_TOKEN"))
#         load_time = time.time() - start_time
#         print(f"[Checkpoint-Worker-{node_idx}] Loaded successfully in {load_time:.2f}s.")

#         # Save checkpoint
#         print(f"[Checkpoint-Worker-{node_idx}] Saving to {node_checkpoint_path}...")
#         fs_start_time = time.time()
        
#         os.makedirs(node_checkpoint_path, exist_ok=True)
#         model.save_pretrained(node_checkpoint_path)
#         tokenizer.save_pretrained(node_checkpoint_path)
        
#         save_time = time.time() - fs_start_time
#         print(f"[Checkpoint-Worker-{node_idx}] Saved successfully in {save_time:.2f}s.")
#         return True
        
#     except Exception as e:
#         print(f"[Checkpoint-Worker-{node_idx}] Failed: {e}")
#         return False

# def distribute_model_loading():
#     """
#     Launches 10 parallel tasks to load and checkpoint the model.
#     Ray will schedule these across available nodes.
#     """
#     print("\n[Checkpoint] Triggering distributed model load on 10 workers...")
    
#     # Launch 10 tasks.
#     # To ensure they run on different nodes, we rely on Ray's scheduler 
#     # and the resource requirements (num_cpus=4).
#     # If using placement groups is strictly required for "exactly one per node", 
#     # we can add that, but standard scheduling usually spreads robustly with high CPU reqs.
    
#     futures = [load_and_checkpoint_worker.remote(i) for i in range(10)]
    
#     results = ray.get(futures)
#     success_count = sum(results)
    
#     if success_count == 10:
#         print(f"[Checkpoint] All 10 workers successfully loaded and saved checkpoints.")
#     else:
#         print(f"[Checkpoint] Only {success_count}/10 workers succeeded. Failing job.")
#         sys.exit(1)


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
            for pf in parquet_files:
                print(f"[Validation]   - {pf}")
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
        print(f"[Step 4] SKIPPED: Distributed Load & Checkpoint")
        # distribute_model_loading()
        
        print(f"[Success] Job completed successfully.")

    except Exception as e:
        error_msg = f"\n[CRITICAL ERROR] Job failed with exception:\n{str(e)}\n"
        print(error_msg)
        # Re-raise to fail the job status
        raise
