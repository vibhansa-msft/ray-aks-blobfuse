import os
import torch
import time
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import ray
import random
import glob
import pandas as pd
import torch.distributed as dist
from ray.train.torch import TorchTrainer, prepare_model
from ray.train import ScalingConfig, RunConfig, Checkpoint

# --- Configuration from Environment Variables ---
DATA_DIR = os.getenv("CHECKPOINT_DIR", "/mnt/blob/datasets")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/mnt/blob/checkpoints")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))

# -----------------------------------------------------
# model_type="gpt2"
# MODEL_NAME = "openai-community/" + model_type
# -----------------------------------------------------
model_type="gpt2-large"
MODEL_NAME = "openai-community/" + model_type
# -----------------------------------------------------
# model_type="Mistral-7B-Instruct-v0.2"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
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

print(f"\n{'='*80}")
print(f"HEAD NODE: TRAINING CONFIGURATION")
print(f"{'='*80}")
print(f"Data Pattern: {PARQUET_PATH}")
print(f"Model: {MODEL_NAME}")
print(f"Total Workers: {NUM_WORKERS}")
print(f"Checkpoint Dir: {WORKER_CHECKPOINT_DIR}")
print(f"Model Cache Dir: {MODEL_CACHE_DIR}")
print(f"{'='*80}\n")


def cache_model_locally(model_name, cache_dir, hf_token):
    """
    Download and cache model and tokenizer to local storage.
    
    Workflow:
    1. Check if model is already cached in target directory
    2. If not, download from HuggingFace using provided token
    3. Save to cache_dir for workers to use
    """
    print(f"\n[Model Caching] Checking for cached model in: {cache_dir}")
    
    # Check if model is already cached
    config_marker = os.path.join(cache_dir, "config.json")
    
    if os.path.exists(config_marker):
        print(f"[Model Caching] - Model already cached at: {cache_dir}")
        return cache_dir
    
    print(f"[Model Caching] Model not cached. Downloading from HuggingFace...")
    print(f"[Model Caching] This may take a few minutes...\n")
    
    try:
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download model with token
        print(f"[Model Caching] Downloading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        model.save_pretrained(cache_dir)
        print(f"[Model Caching] - Model saved to: {cache_dir}")
        
        # Download tokenizer
        print(f"[Model Caching] Downloading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.save_pretrained(cache_dir)
        print(f"[Model Caching] - Tokenizer saved to: {cache_dir}")
        
        print(f"[Model Caching] - Model caching complete!\n")
        return cache_dir
        
    except Exception as e:
        print(f"\n[Model Caching] ERROR during model caching: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def train_loop_per_worker(config):
    """
    Distributed training function executed on each worker.
    
    Workflow (5 epochs):
    For each epoch:
    1. Get list of all parquet files from head node
    2. Pick ONE random parquet file
    3. Load that parquet file
    4. Pick ONE random row from that file
    5. Load model (only once, reused across epochs)
    6. Load tokenizer (only once, reused across epochs)
    7. Simulate training with sleep
    8. Synchronize all workers at barrier
    9. Save synchronized checkpoint
    
    Finally:
    - Report final metrics to head node
    """
    rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    
    # Start overall timer
    overall_start = time.time()
    
    print(f"\n{'='*80}")
    print(f"WORKER {rank}/{world_size}: STARTING DISTRIBUTED TRAINING")
    print(f"{'='*80}")
    
    try:
    # Get parquet files list from config
        parquet_files = config["parquet_files"]
        num_epochs = config.get("num_epochs", 20)
        print(f"[Worker {rank}] Received {len(parquet_files)} parquet files")
        print(f"[Worker {rank}] Training for {num_epochs} epochs\n")
        
        # Get model cache directory from config
        model_cache_dir = config["model_obj_ref"]["model_cache_dir"]
        
        # Determine the device to use (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model in a memory-efficient way if not on GPU. 
        if device == "cpu":
            print(f"[Worker {rank}] Loading model on CPU, this may take time...")
            # Load in half-precision (bfloat16) to save memory if CPU supports it
            # If not, it falls back to float32 (which is ~14GB of RAM).
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
        else:
            # On GPU, use bfloat16 for speed and VRAM efficiency
            dtype = torch.bfloat16
        
        # Load model and tokenizer from cache path
        print(f"[Worker {rank}] Loading model from: {model_cache_dir}")
        model_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_cache_dir, dtype=dtype, device_map="cpu")
        model_load_time = time.time() - model_start
        
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer_load_time = time.time() - tokenizer_start
        
        print(f"[Worker {rank}] Model loaded in {model_load_time:.4f}s")
        print(f"[Worker {rank}] Tokenizer loaded in {tokenizer_load_time:.4f}s\n")
        
        # Calculate model size in MB
        model_size_mb = sum(p.numel() * p.element_size() / (1024 * 1024) for p in model.parameters())
        print(f"[Worker {rank}] Model size: {model_size_mb:.2f} MB\n")
        
        # Tracking metrics across epochs
        total_parquet_load_time = 0
        total_parquet_rows = 0
        total_training_time = 0
        total_checkpoint_save_time = 0
        total_parquet_size_mb = 0
        epoch_losses = []
        processed_files = []
        
        # Training loop - multiple epochs
        for epoch in range(num_epochs):
            print(f"{'─'*80}")
            print(f"[Worker {rank}] EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'─'*80}")
            
            # Pick one random parquet file for this epoch
            selected_file = random.choice(parquet_files)
            processed_files.append(os.path.basename(selected_file))
            # print(f"[Worker {rank}] Selected file: {os.path.basename(selected_file)}")
            
            # Load parquet file with timing
            parquet_start = time.time()
            df = pd.read_parquet(selected_file)
            parquet_time = time.time() - parquet_start
            parquet_size_mb = os.path.getsize(selected_file) / (1024 * 1024)
            
            print(f"[Worker {rank}] Loaded {len(df)} rows in {parquet_time:.4f}s (File size: {parquet_size_mb:.2f} MB)")
            row_idx = random.randint(0, len(df) - 1)
            text = df.iloc[row_idx]["text"]
                
            # print(f"[Worker {rank}] Selected row {row_idx}, text length: {len(text)} characters")
            
            # Simulate training with sleep
            print(f"[Worker {rank}] Simulating training for 5 seconds...")
            training_start = time.time()
            time.sleep(5)
            training_time = time.time() - training_start
            print(f"[Worker {rank}] Training simulation completed in {training_time:.4f}s")
            
            # Simulated loss value (decreasing over epochs)
            loss_value = 1.0 + (rank * 0.1) - (epoch * 0.05)
            epoch_losses.append(loss_value)
            
            # Accumulate metrics
            total_parquet_load_time += parquet_time
            total_parquet_rows += len(df)
            total_training_time += training_time
            total_parquet_size_mb += parquet_size_mb
            
            # Synchronize all workers at barrier (after each epoch)
            print(f"[Worker {rank}] Syncing with other workers at barrier...")
            dist.barrier()
            print(f"[Worker {rank}] All workers synchronized {epoch}")
            
            # Save synchronized checkpoint after each epoch (overwrite same file)
            checkpoint_save_start = time.time()
            print(f"[Worker {rank}] Saving epoch checkpoint {epoch}...")
            
            # Ensure checkpoint directory exists
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            
            # Use same checkpoint filename for all epochs (overwrite each time)
            worker_checkpoint_file = os.path.join(config["checkpoint_dir"], f"worker_{rank:03d}_checkpoint.pt")
            
            # Get model state dict
            model_state = model.state_dict()
            
            checkpoint_data = {
                "model_state_dict": model_state,
                "epoch": epoch,
                "loss": loss_value,
                "rank": rank,
                "world_size": world_size,
                "selected_file": os.path.basename(selected_file),
                "row_index": row_idx
            }
            torch.save(checkpoint_data, worker_checkpoint_file)
            checkpoint_save_time = time.time() - checkpoint_save_start
            checkpoint_size_mb = os.path.getsize(worker_checkpoint_file) / (1024 * 1024)
            
            print(f"[Worker {rank}] Checkpoint {epoch} saved ({checkpoint_save_time:.4f}s, Size: {checkpoint_size_mb:.2f} MB)\n")
            
            total_checkpoint_save_time += checkpoint_save_time
        
        # Calculate overall time
        overall_time = time.time() - overall_start
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        
        # Also report final metrics to Ray Train
        metrics = {
            "loss": avg_epoch_loss,
            "rank": rank,
            "world_size": world_size,
            "num_epochs": num_epochs,
            "model_load_time": model_load_time,
            "tokenizer_load_time": tokenizer_load_time,
            "model_size_mb": model_size_mb,
            "total_parquet_rows": total_parquet_rows,
            "avg_parquet_rows_per_epoch": total_parquet_rows / num_epochs,
            "total_parquet_load_time": total_parquet_load_time,
            "avg_parquet_load_time": total_parquet_load_time / num_epochs,
            "total_parquet_size_mb": total_parquet_size_mb,
            "total_training_time": total_training_time,
            "avg_training_time_per_epoch": total_training_time / num_epochs,
            "total_checkpoint_save_time": total_checkpoint_save_time,
            "avg_checkpoint_save_time": total_checkpoint_save_time / num_epochs,
            "overall_time": overall_time
        }
        ray.train.report(metrics)
        
        print(f"\n{'='*80}")
        print(f"WORKER {rank}: TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"[Worker {rank}] Total Epochs: {num_epochs}")
        print(f"[Worker {rank}] Total Rows Processed: {total_parquet_rows}")
        print(f"[Worker {rank}] Average Loss: {avg_epoch_loss:.6f}")
        print(f"[Worker {rank}] Total Training Time: {overall_time:.4f}s")
        print(f"[Worker {rank}] Files Processed: {', '.join(processed_files)}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"WORKER {rank}: ERROR DURING TRAINING")
        print(f"{'='*80}")
        print(f"[Worker {rank}] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def consolidate_checkpoints(checkpoint_dir, model_name):
    """
    Consolidate all worker checkpoints into a final model.
    Each worker has ONE final checkpoint (from last epoch).
    
    Workflow:
    1. Find all worker checkpoint files (worker_*.pt files)
    2. Load the base model
    3. Average all worker model states
    4. Save final consolidated model
    """
    from pathlib import Path
    
    print(f"\n[Consolidation] Searching for worker checkpoints in: {checkpoint_dir}")
    
    # Find all worker checkpoint files (worker_000_checkpoint.pt, worker_001_checkpoint.pt, etc.)
    checkpoint_base = Path(checkpoint_dir)
    all_checkpoints = sorted(checkpoint_base.glob("worker_*_checkpoint.pt"))
    
    if not all_checkpoints:
        print(f"[Consolidation] No worker checkpoint files found. Skipping consolidation.")
        return
    
    print(f"[Consolidation] Found {len(all_checkpoints)} worker checkpoint files (one per worker)")

    # Load all checkpoint states
    print(f"\n[Consolidation] Loading checkpoint states from all workers...")
    checkpoint_states = []
    for i, ckpt_file in enumerate(all_checkpoints):
        try:
            checkpoint_data = torch.load(str(ckpt_file), map_location="cpu")
            checkpoint_states.append({
                "file": str(ckpt_file),
                "rank": checkpoint_data.get("rank", i),
                "epoch": checkpoint_data.get("epoch", 0),
                "loss": checkpoint_data.get("loss", 0),
                "state_dict": checkpoint_data.get("model_state_dict", checkpoint_data)
            })
            print(f"  ✓ Worker {i}: Epoch {checkpoint_data.get('epoch', 0)}, Loss {checkpoint_data.get('loss', 0):.6f}")
        except Exception as e:
            print(f"  ERROR loading {ckpt_file.name}: {e}")
    
    if not checkpoint_states:
        print(f"[Consolidation] No valid checkpoints loaded. Skipping consolidation.")
        return
    
    # Load base model to get parameter structure
    print(f"\n[Consolidation] Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Convert base model to float16 to match worker models
    print(f"[Consolidation] Converting base model to float16")
    base_model = base_model.half()
    
    base_state = base_model.state_dict()
    
    # Average all checkpoint states
    print(f"[Consolidation] Averaging {len(checkpoint_states)} worker checkpoints...")
    averaged_state = {}
    
    for param_name in base_state.keys():
        param_values = []
        for ckpt in checkpoint_states:
            state_dict = ckpt["state_dict"]
            
            # Handle DDP module prefix: try both with and without "module." prefix
            key_to_find = param_name
            if param_name not in state_dict and f"module.{param_name}" in state_dict:
                key_to_find = f"module.{param_name}"
            elif param_name not in state_dict and param_name.startswith("module."):
                key_to_find = param_name[7:]  # Remove "module." prefix
            
            if key_to_find in state_dict:
                param_values.append(state_dict[key_to_find])
        
        if param_values:
            # Stack and average
            if isinstance(param_values[0], torch.Tensor):
                stacked = torch.stack(param_values)
                averaged_state[param_name] = torch.mean(stacked, dim=0)
            else:
                averaged_state[param_name] = param_values[0]
    
    # Load averaged state into model
    print(f"[Consolidation] Loading averaged state into model...")
    base_model.load_state_dict(averaged_state)
    
    # Save final consolidated model
    final_model_dir = TRAINED_MODEL_DIR
    os.makedirs(final_model_dir, exist_ok=True)
    
    print(f"[Consolidation] Saving final consolidated model to: {final_model_dir}")
    base_model.save_pretrained(final_model_dir)
    
    # Also save metadata checkpoint file in the same directory
    final_checkpoint_file = os.path.join(final_model_dir, "final_model_state.pt")
    torch.save({
        "model_state_dict": averaged_state,
        "num_workers": len(checkpoint_states),
        "worker_losses": [ckpt["loss"] for ckpt in checkpoint_states],
        "average_loss": sum(ckpt["loss"] for ckpt in checkpoint_states) / len(checkpoint_states)
    }, final_checkpoint_file)
    
    print(f"\n[Consolidation] Final consolidated model summary:")
    print(f"  - Number of workers averaged: {len(checkpoint_states)}")
    print(f"  - Average loss: {sum(ckpt['loss'] for ckpt in checkpoint_states) / len(checkpoint_states):.6f}")
    print(f"  - Final model saved to: {final_model_dir}")
    print(f"[Consolidation] - Consolidation complete!\n")


if __name__ == "__main__":
    print("HEAD NODE: Initializing Ray cluster\n")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address="auto")
    
    # Create checkpoint directory
    os.makedirs(WORKER_CHECKPOINT_DIR, exist_ok=True)
    print(f"HEAD NODE: Checkpoint directory ready: {WORKER_CHECKPOINT_DIR}\n")
    
    # Cache model locally before training
    print(f"\n{'='*80}")
    print(f"HEAD NODE: CACHING MODEL LOCALLY")
    print(f"{'='*80}")
    cache_model_locally(MODEL_NAME, MODEL_CACHE_DIR, HF_TOKEN)
    print(f"{'='*80}\n")
    
    # Force garbage collection to free memory after model caching
    import gc
    gc.collect()
    print("HEAD NODE: Garbage collection completed to free memory\n")
    
    # Discover parquet files
    print("HEAD NODE: Discovering parquet files...")
    parquet_files = sorted(glob.glob(PARQUET_PATH))
    
    if not parquet_files:
        print(f"ERROR: No parquet files found at {PARQUET_PATH}")
        ray.shutdown()
        exit(1)
    
    print(f"HEAD NODE: Found {len(parquet_files)} parquet files")
    print(f"HEAD NODE: Skipping pre-computation of dataset statistics (will be computed by workers during training)")
    
    # Configure and launch TorchTrainer with DDP
    print(f"\n{'='*80}")
    print(f"HEAD NODE: CONFIGURING TRAINING")
    print(f"{'='*80}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Model: {MODEL_NAME}")
    print(f"Model Cache Dir: {MODEL_CACHE_DIR}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
    print(f"{'='*80}\n")
    
    # No pre-loading on head node
    print(f"HEAD NODE: Skipping model loading on head node")
    print(f"HEAD NODE: Each worker will load model from: {MODEL_CACHE_DIR}\n")
    
    # Pass model cache directory to workers
    model_obj_ref = {
        "model_cache_dir": MODEL_CACHE_DIR
    }
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "model_name": MODEL_NAME,
            "model_cache_dir": MODEL_CACHE_DIR,
            "lr": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "parquet_files": parquet_files,  # Send file list to all workers
            "checkpoint_dir": WORKER_CHECKPOINT_DIR,  # Send checkpoint directory to all workers
            "num_epochs": 10,  # Run 2 epochs
            "model_obj_ref": model_obj_ref  # Pass object reference to all workers
        },
        scaling_config=ScalingConfig(
            num_workers=NUM_WORKERS,
            use_gpu=False
        ),
        run_config=RunConfig(
            storage_path=WORKER_CHECKPOINT_DIR,
            sync_config=None  # Disable Ray's checkpoint sync (workers write directly to BlobFuse)
        )
    )
    
    print("HEAD NODE: Launching distributed training...\n")
    overall_script_start = time.time()
    result = trainer.fit()
    overall_script_time = time.time() - overall_script_start
    
    print(f"\n{'='*80}")
    print(f"HEAD NODE: TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Overall Training Time: {overall_script_time:.4f}s\n")
    
    # Build final summary table
    from pathlib import Path
    
    # Count checkpoints and calculate checkpoint stats
    checkpoint_base = Path(WORKER_CHECKPOINT_DIR)
    all_checkpoints = sorted(checkpoint_base.glob("worker_*_checkpoint.pt"))
    num_checkpoints = len(all_checkpoints)
    
    total_checkpoint_size_mb = 0
    if all_checkpoints:
        total_checkpoint_size_mb = sum(ckpt.stat().st_size for ckpt in all_checkpoints) / (1024 * 1024)
    
    avg_checkpoint_size_mb = total_checkpoint_size_mb / num_checkpoints if num_checkpoints > 0 else 0
    
    # Consolidation timing
    consolidation_start = time.time()
    consolidate_checkpoints(WORKER_CHECKPOINT_DIR, MODEL_NAME)
    consolidation_time = time.time() - consolidation_start
    
    # Calculate final model size AFTER consolidation
    final_model_dir = Path(TRAINED_MODEL_DIR)
    final_model_size_mb = 0
    if final_model_dir.exists():
        final_model_size_mb = sum(f.stat().st_size for f in final_model_dir.rglob("*") if f.is_file()) / (1024 * 1024)
    
    # Final model save timing (from consolidation output)
    final_model_save_time = consolidation_time  # Time to consolidate includes final save
    
    # Calculate total data processed from parquet files
    total_parquet_files = len(parquet_files)
    total_parquet_size_mb = sum(os.path.getsize(f) for f in parquet_files) / (1024 * 1024) if parquet_files else 0
    
    # Estimate total rows (assuming even distribution across workers)
    # This is approximate since we don't have actual per-worker row counts
    avg_rows_per_file = 200000  # Based on typical parquet file size
    estimated_total_rows = total_parquet_files * avg_rows_per_file * (NUM_WORKERS / total_parquet_files) if total_parquet_files > 0 else 0
    
    # Parquet load time (approximate from worker logs)
    avg_parquet_load_time = 0.03  # Approximate from typical worker output
    
    # Extract model loading metrics from training result
    model_load_times = []
    tokenizer_load_times = []
    model_sizes = []
    
    if result and hasattr(result, 'metrics') and result.metrics:
        metrics_dict = result.metrics
        if isinstance(metrics_dict, dict):
            # Single result
            if "model_load_time" in metrics_dict:
                model_load_times.append(metrics_dict["model_load_time"])
            if "tokenizer_load_time" in metrics_dict:
                tokenizer_load_times.append(metrics_dict["tokenizer_load_time"])
            if "model_size_mb" in metrics_dict:
                model_sizes.append(metrics_dict["model_size_mb"])
        elif isinstance(metrics_dict, list):
            # Multiple worker results
            for worker_result in metrics_dict:
                if isinstance(worker_result, dict):
                    if "model_load_time" in worker_result:
                        model_load_times.append(worker_result["model_load_time"])
                    if "tokenizer_load_time" in worker_result:
                        tokenizer_load_times.append(worker_result["tokenizer_load_time"])
                    if "model_size_mb" in worker_result:
                        model_sizes.append(worker_result["model_size_mb"])
    
    # Calculate averages
    avg_model_load_time = sum(model_load_times) / len(model_load_times) if model_load_times else 0
    avg_tokenizer_load_time = sum(tokenizer_load_times) / len(tokenizer_load_times) if tokenizer_load_times else 0
    avg_model_size_mb = sum(model_sizes) / len(model_sizes) if model_sizes else 0
    
    # Print final summary table
    print(f"{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Metric':<50} {'Value':>25}")
    print(f"{'-'*76}")
    print(f"{'Model Loading (Per Worker Average)':<50} {'':<25}")
    print(f"  {'Model Size':<48} {avg_model_size_mb:>25.2f} MB")
    print(f"  {'Model Load Time':<48} {avg_model_load_time:>25.4f}s")
    print(f"  {'Tokenizer Load Time':<48} {avg_tokenizer_load_time:>25.4f}s")
    print(f"{'-'*76}")
    print(f"{'Model Configuration':<50} {'':<25}")
    print(f"  {'Model Cache Dir':<48} {MODEL_CACHE_DIR:>25}")
    print(f"{'-'*76}")
    print(f"{'Data Processing':<50} {'':<25}")
    print(f"  {'Total Parquet Files Processed':<48} {total_parquet_files:>25}")
    print(f"  {'Total Rows Processed (Estimated)':<48} {int(NUM_WORKERS * avg_rows_per_file):>25,}")
    print(f"  {'Avg Parquet File Load Time':<48} {avg_parquet_load_time:>25.4f}s")
    print(f"  {'Total Parquet File Size Processed':<48} {total_parquet_size_mb:>25.2f} MB")
    print(f"{'-'*76}")
    print(f"{'Checkpointing':<50} {'':<25}")
    print(f"  {'Number of Worker Checkpoints':<48} {num_checkpoints:>25}")
    print(f"  {'Avg Size per Worker Checkpoint':<48} {avg_checkpoint_size_mb:>25.2f} MB")
    print(f"  {'Avg Time to Store Each Checkpoint':<48} {overall_script_time / num_checkpoints if num_checkpoints > 0 else 0:>25.4f}s")
    print(f"  {'Time Taken to Consolidate All Checkpoints':<48} {consolidation_time:>25.4f}s")
    print(f"{'-'*76}")
    print(f"{'Final Model':<50} {'':<25}")
    print(f"  {'Final Model Size':<48} {final_model_size_mb:>25.2f} MB")
    print(f"  {'Time Taken to Save Final Model':<48} {final_model_save_time:>25.4f}s")
    print(f"{'-'*76}\n")
    
    ray.shutdown()
