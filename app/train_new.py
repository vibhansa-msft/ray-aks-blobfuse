import os
import time
import torch
import glob
import logging
import sys
import ray
from ray.air import session
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import ray.data

# Disable Ray's internal logging before any Ray initialization
os.environ["RAY_logging_level"] = "CRITICAL"
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"

# Configure logging to suppress verbose output
logging.basicConfig(level=logging.CRITICAL, format='%(message)s', stream=sys.stdout)
logging.disable(logging.CRITICAL)

# Suppress all Ray and other library logging
for logger_name in ["ray", "ray.tune", "ray.air", "ray.train", "ray.data", 
                     "transformers", "fsspec", "urllib3", "pyarrow", "parquet"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

# --- Configuration from Environment Variables ---
DATA_DIR = os.getenv("DATA_DIR", "/mnt/blob/datasets")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/mnt/blob/checkpoints")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))

# -----------------------------------------------------
# model_type="gpt2"
# MODEL_NAME = "openai-community/" + model_type
# -----------------------------------------------------
# model_type="gpt2-large"
# MODEL_NAME = "openai-community/" + model_type
# -----------------------------------------------------
model_type="Mistral-7B-Instruct-v0.2"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# -----------------------------------------------------

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Construct paths
PARQUET_PATH = os.path.join(DATA_DIR.rstrip("/"), "c4/*.parquet")
# PARQUET_PATH = os.path.join(DATA_DIR.rstrip("/"), "openwebtext/*.parquet")
WORKER_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR.rstrip("/"), "worker_checkpoints", model_type)
TRAINED_MODEL_DIR = os.path.join(CHECKPOINT_DIR.rstrip("/"), "model", model_type)
MODEL_CACHE_DIR = os.path.join(CHECKPOINT_DIR.rstrip("/"), "base_models", model_type)

# For reproducibility
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# ============= CHECKPOINT SHARDING HELPERS =============

def _save_sharded_checkpoint(state_dict, checkpoint_path, shard_size_gb=5):
    """
    Save state dict as sharded checkpoint files (one file per shard).
    Instead of saving one 30GB file, saves multiple ~5GB files.
    
    Args:
        state_dict: Model state dict to save
        checkpoint_path: Base path (without extension, will add _shard_0.pt, _shard_1.pt, etc)
        shard_size_gb: Target size per shard in GB
    """
    import pickle
    import os
    
    shard_size_bytes = shard_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    
    # Serialize state_dict to get estimated size
    serialized = pickle.dumps(state_dict)
    estimated_size = len(serialized)
    
    # Calculate number of shards needed
    num_shards = max(1, (estimated_size + shard_size_bytes - 1) // shard_size_bytes)
    
    if num_shards == 1:
        # Small model, save as single file
        torch.save(state_dict, checkpoint_path)
        print(f"    Saved as single file: {checkpoint_path}", flush=True)
        return
    
    # Split state_dict into shards
    print(f"    Sharding checkpoint into {num_shards} files (~{shard_size_gb}GB each)...", flush=True)
    
    keys = list(state_dict.keys())
    keys_per_shard = max(1, (len(keys) + num_shards - 1) // num_shards)
    
    total_saved = 0
    for shard_idx in range(num_shards):
        # Get slice of keys for this shard
        start_key_idx = shard_idx * keys_per_shard
        end_key_idx = min((shard_idx + 1) * keys_per_shard, len(keys))
        shard_keys = keys[start_key_idx:end_key_idx]
        
        # Create shard dict
        shard_dict = {k: state_dict[k] for k in shard_keys}
        
        # Save shard
        shard_path = checkpoint_path.replace('.pt', f'_shard_{shard_idx}.pt')
        torch.save(shard_dict, shard_path)
        
        shard_size = os.path.getsize(shard_path) / (1024 * 1024 * 1024)
        total_saved += shard_size
        print(f"    ✓ Shard {shard_idx}: {shard_path} ({shard_size:.2f} GB)", flush=True)
    
    print(f"    Total checkpoint size: {total_saved:.2f} GB across {num_shards} shards", flush=True)


def _load_sharded_checkpoint(checkpoint_path, device="cpu"):
    """
    Load sharded checkpoint files and reconstruct the state dict.
    
    Args:
        checkpoint_path: Base path (will look for _shard_0.pt, _shard_1.pt, etc)
        device: Device to load to
        
    Returns:
        Reconstructed state dict
    """
    import glob
    
    # Try to find sharded files
    base_path = checkpoint_path.replace('.pt', '')
    shard_pattern = f"{base_path}_shard_*.pt"
    shard_files = sorted(glob.glob(shard_pattern))
    
    if not shard_files:
        # Try original path (single file)
        if os.path.exists(checkpoint_path):
            print(f"    Loading single checkpoint file: {checkpoint_path}", flush=True)
            return torch.load(checkpoint_path, map_location=device)
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or shards at {shard_pattern}")
    
    # Load and reconstruct from shards
    print(f"    Loading {len(shard_files)} checkpoint shards...", flush=True)
    state_dict = {}
    
    for shard_file in shard_files:
        shard = torch.load(shard_file, map_location=device)
        state_dict.update(shard)
        shard_size = os.path.getsize(shard_file) / (1024 * 1024 * 1024)
        print(f"    ✓ Loaded {os.path.basename(shard_file)} ({shard_size:.2f} GB)", flush=True)
    
    return state_dict

def train_func(config):
    # Model & tokenizer paths
    model_path = MODEL_CACHE_DIR
    tokenizer_path = MODEL_CACHE_DIR

    # Use AutoTokenizer and AutoModelForCausalLM for generic model loading
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(DEVICE)
    model.train()

    # Get Ray Data dataset shard for this worker from session
    ds_shard = session.get_dataset_shard("train")
    
    # Get worker rank and world size
    world_rank = session.get_world_rank()
    # print(f"[Worker {world_rank}/{world_size}] Starting training with Ray Data shard", flush=True)
    
    # Import os at the beginning of the function
    import os
    
    # Batch size configuration
    batch_size = config.get("batch_size", 2 if DEVICE == "cpu" else 8)
    
    # Helper function to tokenize a pandas batch
    def tokenize_batch(batch):
        """Tokenize a batch from Ray Data (pandas DataFrame)"""
        texts = batch["text"].tolist() if "text" in batch else []
        if not texts:
            return None
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    simulate = config.get("simulate_training", False) and DEVICE == "cpu"
    checkpoint_dir = config.get("checkpoint_dir", "/tmp/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config.get("epochs", 1)):
        # print(f"[Worker {world_rank}] Epoch {epoch + 1}/{config.get('epochs', 1)}", flush=True)
        total_loss = 0.0
        num_batches = 0
        
        if simulate:
            print(f"[Worker {world_rank}] Simulating training - reading 2 random parquet files from Azure storage...", flush=True)
            
            import random
            
            files_read = 0
            bytes_read = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            
            try:
                # Get all parquet files from the dataset directory
                all_parquet_files = sorted(glob.glob("/mnt/blob/datasets/c4/*.parquet"))
                
                if not all_parquet_files:
                    print(f"[Worker {world_rank}] No parquet files found in /mnt/blob/datasets/c4/", flush=True)
                else:
                    # Randomly select 2 files for this worker
                    num_files_to_read = min(2, len(all_parquet_files))
                    worker_files = random.sample(all_parquet_files, num_files_to_read)
                    
                    for parquet_file in worker_files:
                        if os.path.exists(parquet_file):
                            print(f"[Worker {world_rank}] Reading: {os.path.basename(parquet_file)}...", flush=True)
                            # Open and read file in chunks
                            with open(parquet_file, 'rb') as f:
                                while True:
                                    chunk = f.read(chunk_size)
                                    if not chunk:
                                        break
                                    bytes_read += len(chunk)
                            files_read += 1
                            print(f"[Worker {world_rank}] ✓ Read {os.path.basename(parquet_file)} ({bytes_read / (1024*1024):.2f} MB total)", flush=True)
                        else:
                            print(f"[Worker {world_rank}] File not found: {parquet_file}", flush=True)
                
                print(f"[Worker {world_rank}] Simulation complete - read {files_read} files ({bytes_read / (1024*1024):.2f} MB)", flush=True)
            except Exception as e:
                print(f"[Worker {world_rank}] Error during simulation: {str(e)}", flush=True)
                import traceback
                traceback.print_exc()
            
            epoch_loss = 0.0
        else:
            # print(f"[Worker {world_rank}] Processing data batches...", flush=True)
            # Iterate over Ray Data batches
            for batch_idx, batch in enumerate(ds_shard.iter_batches(batch_size=batch_size, batch_format="pandas")):
                try:
                    tokenized = tokenize_batch(batch)
                    if tokenized is None or "input_ids" not in tokenized:
                        continue
                    
                    inputs = {k: v.to(DEVICE) for k, v in tokenized.items()}
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # if (batch_idx + 1) % 10 == 0:
                    #     print(f"[Worker {world_rank}] Batch {batch_idx + 1} - Loss: {loss.item():.6f}", flush=True)
                except Exception as e:
                    print(f"[Worker {world_rank}] ERROR processing batch {batch_idx}: {str(e)}", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue
            
            epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
            # print(f"[Worker {world_rank}] Epoch {epoch + 1} completed - Avg Loss: {epoch_loss:.6f}", flush=True)
        
        # Save checkpoint every epoch
        checkpoint_dir_epoch = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        
        # Ensure directory exists with proper error handling for BlobFuse
        try:
            os.makedirs(checkpoint_dir_epoch, exist_ok=True)
            # Verify directory was created
            if not os.path.exists(checkpoint_dir_epoch):
                print(f"[Worker {world_rank}] WARNING: Directory {checkpoint_dir_epoch} not accessible after makedirs", flush=True)
        except Exception as e:
            print(f"[Worker {world_rank}] ERROR creating checkpoint directory {checkpoint_dir_epoch}: {str(e)}", flush=True)
            raise
        
        # Save model state dict - sharded into 5GB chunks
        checkpoint_path = os.path.join(checkpoint_dir_epoch, f"worker_{world_rank}_model.pt")
        try:
            print(f"[Worker {world_rank}] Taking checkpoint", flush=True)
            
            # Save sharded checkpoint (5GB per shard)
            state_dict = model.state_dict()
            _save_sharded_checkpoint(state_dict, checkpoint_path, shard_size_gb=5)
            
            print(f"[Worker {world_rank}] Checkpoint saved (sharded) to {checkpoint_dir_epoch}", flush=True)
        except Exception as e:
            print(f"[Worker {world_rank}] ERROR saving checkpoint to {checkpoint_path}: {str(e)}", flush=True)
            print(f"[Worker {world_rank}] Directory exists: {os.path.exists(checkpoint_dir_epoch)}", flush=True)
            print(f"[Worker {world_rank}] Directory contents: {os.listdir(checkpoint_dir_epoch) if os.path.exists(checkpoint_dir_epoch) else 'N/A'}", flush=True)
            raise
        
        # Report metrics WITHOUT checkpoint (Ray checkpoint storage is causing issues)
        # The checkpoints are already saved to BlobFuse directly
        session.report({"loss": epoch_loss, "epoch": epoch + 1})

def aggregate_checkpoints(checkpoint_dir, num_workers, last_epoch, output_path):
    """
    Aggregate checkpoints from all workers into a final model.
    Each worker saves its checkpoint as either:
    - Single file: checkpoint_dir/epoch_{epoch}/worker_{rank}_model.pt (< 5GB)
    - Sharded files: checkpoint_dir/epoch_{epoch}/worker_{rank}_model_shard_0.pt, _shard_1.pt, etc (>= 5GB)
    """
    print(f"\n{'='*80}", flush=True)
    print(f"[Aggregation] Starting checkpoint aggregation", flush=True)
    print(f"{'='*80}", flush=True)
    
    # Find all available worker checkpoints (single or sharded)
    loaded_workers = []
    for rank in range(num_workers):
        worker_ckpt_base = os.path.join(checkpoint_dir, f"epoch_{last_epoch}", f"worker_{rank}_model.pt")
        worker_ckpt_base_no_ext = worker_ckpt_base.replace('.pt', '')
        
        # Check for single file
        if os.path.exists(worker_ckpt_base):
            loaded_workers.append(rank)
            print(f"  ✓ Worker {rank}: Single checkpoint found", flush=True)
        else:
            # Check for sharded files
            shard_pattern = f"{worker_ckpt_base_no_ext}_shard_0.pt"
            if os.path.exists(shard_pattern):
                loaded_workers.append(rank)
                # Count shards
                shard_count = 0
                shard_idx = 0
                while os.path.exists(f"{worker_ckpt_base_no_ext}_shard_{shard_idx}.pt"):
                    shard_count += 1
                    shard_idx += 1
                print(f"  ✓ Worker {rank}: Sharded checkpoint found ({shard_count} shards)", flush=True)
            else:
                print(f"  ✗ Worker {rank}: Checkpoint not found", flush=True)
    
    if not loaded_workers:
        print(f"\n[Aggregation] ERROR: No checkpoints found for aggregation!", flush=True)
        return None
    
    print(f"\n[Aggregation] Found {len(loaded_workers)} worker checkpoints", flush=True)
    print(f"[Aggregation] Loaded from workers: {loaded_workers}\n", flush=True)
    
    # Track aggregation timing
    aggregation_start_time = time.time()
    load_times = []
    
    if DEVICE == "cpu":
        # CPU mode: simulate aggregation by iterating through checkpoints
        print(f"[Aggregation] CPU mode - simulating aggregation process...", flush=True)
        for idx, rank in enumerate(loaded_workers):
            worker_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{last_epoch}", f"worker_{rank}_model.pt")
            load_start = time.time()
            print(f"  Processing checkpoint {idx + 1}/{len(loaded_workers)} from worker {rank}...", flush=True)
            time.sleep(2)  # Simulate aggregation work
            load_times.append(time.time() - load_start)
        
        print(f"[Aggregation] Simulated aggregation complete!", flush=True)
        
        # Load first checkpoint as the "aggregated" model for CPU simulation
        print(f"[Aggregation] Generating final model...", flush=True)
        first_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{last_epoch}", f"worker_{loaded_workers[0]}_model.pt")
        aggregated = _load_sharded_checkpoint(first_ckpt_path, device="cpu")
    else:
        # GPU mode: load all checkpoints and perform actual averaging
        print(f"[Aggregation] GPU mode - loading all checkpoints and averaging weights...", flush=True)
        state_dicts = []
        for rank in loaded_workers:
            worker_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{last_epoch}", f"worker_{rank}_model.pt")
            load_start = time.time()
            state_dict = _load_sharded_checkpoint(worker_ckpt_path, device="cpu")
            load_times.append(time.time() - load_start)
            state_dicts.append(state_dict)
            print(f"  Loaded checkpoint from worker {rank} ({load_times[-1]:.2f}s)", flush=True)
        
        print(f"\n[Aggregation] Averaging weights across {len(state_dicts)} workers...", flush=True)
        # Average weights across all workers
        aggregated = {}
        for key in state_dicts[0].keys():
            tensors = [sd[key] for sd in state_dicts]
            stacked = torch.stack(tensors)
            aggregated[key] = torch.mean(stacked, dim=0)
        
        print(f"[Aggregation] Weight averaging complete!", flush=True)
    
    # Save final aggregated model with timing - also sharded
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[Aggregation] Saving final model (sharded)...", flush=True)
    save_start = time.time()
    _save_sharded_checkpoint(aggregated, output_path, shard_size_gb=5)
    save_time = time.time() - save_start
    
    aggregation_total_time = time.time() - aggregation_start_time
    avg_load_time = sum(load_times) / len(load_times) if load_times else 0
    
    # print(f"\n[Aggregation] ✓ Final aggregated model saved to: {output_path}", flush=True)
    # print(f"[Aggregation] File size: {os.path.getsize(output_path) / (1024**3):.2f} GB", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    return {
        "num_checkpoints_loaded": len(loaded_workers),
        "avg_load_time": avg_load_time,
        "save_time": save_time,
        "total_aggregation_time": aggregation_total_time
    }

def main():
    job_start_time = time.time()
    ray.init()

    # Discover parquet files from the glob pattern
    print(f"Discovering parquet files from: {PARQUET_PATH}", flush=True)
    parquet_files = sorted(glob.glob(PARQUET_PATH))
    
    if not parquet_files:
        print(f"ERROR: No parquet files found at {PARQUET_PATH}", flush=True)
        ray.shutdown()
        exit(1)
    
    print(f"Found {len(parquet_files)} parquet files", flush=True)
    
    # Set up worker and epoch configuration early
    num_workers = NUM_WORKERS
    epochs = 2
    
    # Calculate total data size
    total_data_size_mb = 0
    for file in parquet_files:
        total_data_size_mb += os.path.getsize(file) / (1024 * 1024)
    total_data_size_gb = total_data_size_mb / 1024
    
    checkpoint_dir = WORKER_CHECKPOINT_DIR

    config = {
        "batch_size": 2 if DEVICE == "cpu" else 8,
        "checkpoint_dir": checkpoint_dir,
        "model_path": MODEL_CACHE_DIR,
        "tokenizer_path": MODEL_CACHE_DIR,
        "simulate_training": not USE_GPU,  # set to True for CPU, False for GPU/real
        "epochs": epochs,
    }

    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=USE_GPU)

    # Load parquet files with Ray Data (automatically sharded across workers)
    # Use parallelism to speed up metadata reading
    print(f"Loading parquet files...", flush=True)
    ds = ray.data.read_parquet(
        parquet_files,
        parallelism=min(num_workers, len(parquet_files)),
        ignore_missing_paths=True
    ).repartition(num_workers)
    print(f"Ray Data dataset created with {len(parquet_files)} files ({total_data_size_gb:.2f} GB)", flush=True)

    
    # Simplified RunConfig - workers save checkpoints directly to BlobFuse
    # No need for Ray's checkpoint persistence which was causing issues
    run_config = RunConfig()

    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": ds}
    )

    # Run distributed training job
    # print(f"Starting distributed training with {num_workers} workers...", flush=True)
    training_start = time.time()
    results = trainer.fit()
    training_time = time.time() - training_start
    # print("Training results:", results.metrics, flush=True)

    # Count total worker checkpoints created
    total_checkpoints = 0
    checkpoint_times = []
    for epoch in range(epochs):
        epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        if os.path.exists(epoch_dir):
            for worker_id in range(num_workers):
                ckpt_file = os.path.join(epoch_dir, f"worker_{worker_id}_model.pt")
                if os.path.exists(ckpt_file):
                    total_checkpoints += 1
    
    avg_checkpoint_write_time = training_time / total_checkpoints if total_checkpoints > 0 else 0

    # Aggregate checkpoints for final model
    print(f"Aggregating {num_workers} worker checkpoints...", flush=True)
    final_model_path = os.path.join(TRAINED_MODEL_DIR, f"{model_type.replace('/', '_')}_finetuned.pt")
    aggregation_stats = aggregate_checkpoints(checkpoint_dir, num_workers, last_epoch=epochs-1, output_path=final_model_path)
    
    job_total_time = time.time() - job_start_time
    
    # Print comprehensive summary
    print(f"\n{'='*80}", flush=True)
    print(f"{'TRAINING JOB SUMMARY':^80}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    print(f"{'Data Statistics:':<40}", flush=True)
    print(f"  - Parquet files loaded:              {len(parquet_files)}", flush=True)
    print(f"  - Total data size:                   {total_data_size_gb:.2f} GB ({total_data_size_mb:.0f} MB)", flush=True)
    
    print(f"\n{'Training Configuration:':<40}", flush=True)
    print(f"  - Model:                             {MODEL_NAME}", flush=True)
    print(f"  - Number of workers:                 {num_workers}", flush=True)
    print(f"  - Number of epochs:                  {epochs}", flush=True)
    
    print(f"\n{'Training Execution:':<40}", flush=True)
    print(f"  - Training duration:                 {training_time:.2f}s", flush=True)
    print(f"  - Total worker checkpoints created:  {total_checkpoints}", flush=True)
    print(f"  - Avg time per checkpoint write:     {avg_checkpoint_write_time:.2f}s", flush=True)
    
    if aggregation_stats:
        print(f"\n{'Aggregation Stage:':<40}", flush=True)
        print(f"  - Checkpoints loaded:                {aggregation_stats['num_checkpoints_loaded']}", flush=True)
        print(f"  - Avg checkpoint load time:          {aggregation_stats['avg_load_time']:.2f}s", flush=True)
        print(f"  - Final model save time:             {aggregation_stats['save_time']:.2f}s", flush=True)
        final_size_gb = os.path.getsize(final_model_path) / (1024**3)
        print(f"  - Final model size:                  {final_size_gb:.2f} GB", flush=True)
    
    print(f"\n{'='*80}\n", flush=True)
    
    ray.shutdown()

if __name__ == "__main__":
    main()
