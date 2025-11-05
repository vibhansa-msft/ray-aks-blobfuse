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
import tempfile

# --- Configuration from Environment Variables ---
DATA_DIR = os.getenv("DATA_DIR", "/mnt/blob/datasets")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/mnt/blob/checkpoints")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))

mode_type="gpt2"

MODEL_NAME = os.getenv("MODEL_NAME", "openai-community/") + mode_type
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Construct paths
PARQUET_PATH = os.path.join(DATA_DIR.rstrip("/"), "openwebtext/*.parquet")
WORKER_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR.rstrip("/"), "worker_checkpoints", mode_type)
TRAINED_MODEL_DIR = os.path.join(CHECKPOINT_DIR.rstrip("/"), "model", mode_type)
MODEL_CACHE_DIR = os.path.join(CHECKPOINT_DIR.rstrip("/"), "base_models", mode_type)

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
    Training function executed on each worker with detailed timing instrumentation.
    
    Workflow:
    1. Get list of all parquet files from head node
    2. Pick ONE random parquet file
    3. Load that parquet file
    4. Pick ONE random row from that file
    5. Load model with DDP wrapper
    6. Load tokenizer
    7. Tokenize and fine-tune on that single row
    8. Save checkpoint with model state
    9. Sync all workers at barrier
    """
    rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    
    # Start overall timer
    overall_start = time.time()
    
    print(f"\n{'='*80}")
    print(f"WORKER {rank}/{world_size}: STARTING TRAINING")
    print(f"{'='*80}")
    
    try:
        # Step 1: Get parquet files list from config
        parquet_files = config["parquet_files"]
        print(f"[Worker {rank}] Received {len(parquet_files)} parquet files to choose from")
        
        # Step 2: Pick one random file
        selected_file = random.choice(parquet_files)
        print(f"[Worker {rank}] Randomly selected: {os.path.basename(selected_file)}")
        
        # Step 3: Load parquet file with timing
        parquet_start = time.time()
        print(f"[Worker {rank}] Loading parquet file...")
        df = pd.read_parquet(selected_file)
        parquet_time = time.time() - parquet_start
        print(f"[Worker {rank}] Loaded {len(df)} rows from parquet in {parquet_time:.4f}s")
        
        # Step 4: Pick one random row
        row_idx = random.randint(0, len(df) - 1)
        text = df.iloc[row_idx]["text"]
        print(f"[Worker {rank}] Selected row {row_idx}, text length: {len(text)} characters")
        
        # Step 5: Load model with DDP wrapper with timing
        model_load_start = time.time()
        device = torch.device("cpu")
        print(f"[Worker {rank}] Loading model from cache: {config['model_cache_dir']}")
        model = AutoModelForCausalLM.from_pretrained(config["model_cache_dir"])
        
        # Convert to float16 to reduce memory footprint by ~50%
        print(f"[Worker {rank}] Converting model to float16 for memory optimization")
        model = model.half()
        
        # Enable gradient checkpointing to save memory
        print(f"[Worker {rank}] Enabling gradient checkpointing for memory optimization")
        model.gradient_checkpointing_enable()
        
        model.to(device)
        print(f"[Worker {rank}] Model loaded to device (float16)")
        
        print(f"[Worker {rank}] Wrapping model with DDP for distributed training")
        model = prepare_model(model)
        print(f"[Worker {rank}] Model wrapped with DDP")
        model_load_time = time.time() - model_load_start
        print(f"[Worker {rank}] Model loading completed in {model_load_time:.4f}s")
        
        # Step 6: Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model_cache_dir"])
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[Worker {rank}] Tokenizer ready")
        
        # Step 7: Tokenize and fine-tune with timing
        tokenize_start = time.time()
        print(f"[Worker {rank}] Tokenizing text sample...")
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config["max_seq_length"],
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()
        tokenize_time = time.time() - tokenize_start
        print(f"[Worker {rank}] Tokenized input shape: {input_ids.shape} in {tokenize_time:.4f}s")
        
        # Create optimizer
        optimizer = AdamW(model.parameters(), lr=config["lr"])
        print(f"[Worker {rank}] Optimizer created with lr={config['lr']}")
        
        # Single forward pass with timing
        forward_start = time.time()
        print(f"[Worker {rank}] Starting forward pass...")
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        forward_time = time.time() - forward_start
        print(f"[Worker {rank}] Loss: {loss.item():.6f} (forward pass: {forward_time:.4f}s)")
        
        # Single backward pass with timing
        backward_start = time.time()
        print(f"[Worker {rank}] Starting backward pass...")
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start
        print(f"[Worker {rank}] Training step completed (backward pass: {backward_time:.4f}s)")
        
        # Step 9: Synchronize all workers
        print(f"[Worker {rank}] Synchronizing with other workers at barrier...")
        dist.barrier()
        print(f"[Worker {rank}] All workers synchronized")
        
        # Step 8: Save checkpoint with timing
        checkpoint_save_start = time.time()
        print(f"[Worker {rank}] Saving checkpoint to shared checkpoint directory...")
        
        # Ensure checkpoint directory exists
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        
        # Create worker-specific checkpoint filename
        worker_checkpoint_file = os.path.join(config["checkpoint_dir"], f"worker_{rank:03d}_checkpoint.pt")
        
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "loss": loss.item(),
            "rank": rank,
            "world_size": world_size,
            "selected_file": os.path.basename(selected_file),
            "row_index": row_idx
        }
        torch.save(checkpoint_data, worker_checkpoint_file)
        checkpoint_save_time = time.time() - checkpoint_save_start
        print(f"[Worker {rank}] Checkpoint saved to: {worker_checkpoint_file} ({checkpoint_save_time:.4f}s)")
        
        # Calculate overall time
        overall_time = time.time() - overall_start
        
        # Also report metrics to Ray Train
        metrics = {
            "loss": loss.item(),
            "rank": rank,
            "world_size": world_size,
            "parquet_load_time": parquet_time,
            "model_load_time": model_load_time,
            "tokenize_time": tokenize_time,
            "forward_pass_time": forward_time,
            "backward_pass_time": backward_time,
            "checkpoint_save_time": checkpoint_save_time,
            "overall_time": overall_time
        }
        ray.train.report(metrics)
        
        print(f"{'='*80}")
        print(f"WORKER {rank}: TRAINING COMPLETED SUCCESSFULLY")
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
    
    Workflow:
    1. Load the base model
    2. Find all worker checkpoint files (worker_*.pt files)
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
    
    print(f"[Consolidation] Found {len(all_checkpoints)} worker checkpoint files:")

    # Load all checkpoint states
    print(f"\n[Consolidation] Loading all checkpoint states...")
    checkpoint_states = []
    for i, ckpt_file in enumerate(all_checkpoints):
        try:
            checkpoint_data = torch.load(str(ckpt_file), map_location="cpu")
            checkpoint_states.append({
                "file": str(ckpt_file),
                "rank": checkpoint_data.get("rank", i),
                "loss": checkpoint_data.get("loss", 0),
                "state_dict": checkpoint_data.get("model_state_dict", checkpoint_data)
            })
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
    print(f"[Consolidation] Averaging {len(checkpoint_states)} checkpoint states...")
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
    
    # Configure and launch TorchTrainer with DDP
    print(f"{'='*80}")
    print(f"HEAD NODE: CONFIGURING TRAINING")
    print(f"{'='*80}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Model: {MODEL_NAME}")
    print(f"Model Cache Dir: {MODEL_CACHE_DIR}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
    print(f"{'='*80}\n")
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "model_name": MODEL_NAME,
            "model_cache_dir": MODEL_CACHE_DIR,
            "lr": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "parquet_files": parquet_files,  # Send file list to all workers
            "checkpoint_dir": WORKER_CHECKPOINT_DIR  # Send checkpoint directory to all workers
        },
        scaling_config=ScalingConfig(
            num_workers=NUM_WORKERS,
            use_gpu=False
        ),
        run_config=RunConfig(
            storage_path=WORKER_CHECKPOINT_DIR
        )
    )
    
    print("HEAD NODE: Launching distributed training...\n")
    overall_script_start = time.time()
    result = trainer.fit()
    overall_script_time = time.time() - overall_script_start
    
    print(f"\n{'='*80}")
    print(f"HEAD NODE: TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Overall Training Time: {overall_script_time:.4f}s")
    print(f"Checkpoints saved to: {WORKER_CHECKPOINT_DIR}\n")
    
    # Extract and aggregate timing metrics from all workers
    print(f"{'='*80}")
    print(f"HEAD NODE: AGGREGATING WORKER METRICS")
    print(f"{'='*80}\n")
    
    if hasattr(result, 'metrics') and result.metrics:
        metrics = result.metrics
        
        # Collect timing information
        timings = {
            "parquet_load_time": [],
            "model_load_time": [],
            "tokenize_time": [],
            "forward_pass_time": [],
            "backward_pass_time": [],
            "checkpoint_save_time": [],
            "overall_time": []
        }
        losses = []
        
        # Check if metrics is a list or dict
        if isinstance(metrics, dict):
            # Single worker case
            for key in timings.keys():
                if key in metrics:
                    timings[key].append(metrics[key])
            if "loss" in metrics:
                losses.append(metrics["loss"])
        elif isinstance(metrics, list):
            # Multiple workers
            for metric in metrics:
                if isinstance(metric, dict):
                    for key in timings.keys():
                        if key in metric:
                            timings[key].append(metric[key])
                    if "loss" in metric:
                        losses.append(metric["loss"])
        
        # Print aggregated statistics
        print(f"Worker Timing Statistics (Average):")
        print(f"{'─'*60}")
        print(f"  Parquet Load Time:      {sum(timings['parquet_load_time'])/len(timings['parquet_load_time']) if timings['parquet_load_time'] else 0:.4f}s (min: {min(timings['parquet_load_time']) if timings['parquet_load_time'] else 0:.4f}s, max: {max(timings['parquet_load_time']) if timings['parquet_load_time'] else 0:.4f}s)")
        print(f"  Model Load Time:        {sum(timings['model_load_time'])/len(timings['model_load_time']) if timings['model_load_time'] else 0:.4f}s (min: {min(timings['model_load_time']) if timings['model_load_time'] else 0:.4f}s, max: {max(timings['model_load_time']) if timings['model_load_time'] else 0:.4f}s)")
        print(f"  Tokenization Time:      {sum(timings['tokenize_time'])/len(timings['tokenize_time']) if timings['tokenize_time'] else 0:.4f}s (min: {min(timings['tokenize_time']) if timings['tokenize_time'] else 0:.4f}s, max: {max(timings['tokenize_time']) if timings['tokenize_time'] else 0:.4f}s)")
        print(f"  Forward Pass Time:      {sum(timings['forward_pass_time'])/len(timings['forward_pass_time']) if timings['forward_pass_time'] else 0:.4f}s (min: {min(timings['forward_pass_time']) if timings['forward_pass_time'] else 0:.4f}s, max: {max(timings['forward_pass_time']) if timings['forward_pass_time'] else 0:.4f}s)")
        print(f"  Backward Pass Time:     {sum(timings['backward_pass_time'])/len(timings['backward_pass_time']) if timings['backward_pass_time'] else 0:.4f}s (min: {min(timings['backward_pass_time']) if timings['backward_pass_time'] else 0:.4f}s, max: {max(timings['backward_pass_time']) if timings['backward_pass_time'] else 0:.4f}s)")
        print(f"  Checkpoint Save Time:   {sum(timings['checkpoint_save_time'])/len(timings['checkpoint_save_time']) if timings['checkpoint_save_time'] else 0:.4f}s (min: {min(timings['checkpoint_save_time']) if timings['checkpoint_save_time'] else 0:.4f}s, max: {max(timings['checkpoint_save_time']) if timings['checkpoint_save_time'] else 0:.4f}s)")
        print(f"  {'─'*56}")
        print(f"  Overall Time per Worker: {sum(timings['overall_time'])/len(timings['overall_time']) if timings['overall_time'] else 0:.4f}s (min: {min(timings['overall_time']) if timings['overall_time'] else 0:.4f}s, max: {max(timings['overall_time']) if timings['overall_time'] else 0:.4f}s)")
        
        print(f"\nWorker Loss Statistics:")
        print(f"{'─'*60}")
        print(f"  Average Loss:           {sum(losses)/len(losses) if losses else 0:.6f}")
        print(f"  Min Loss:               {min(losses) if losses else 0:.6f}")
        print(f"  Max Loss:               {max(losses) if losses else 0:.6f}")
        
        print(f"\n{'='*80}")
        print(f"Script Execution Summary:")
        print(f"{'='*80}")
        print(f"  Total Script Time:      {overall_script_time:.4f}s")
        print(f"{'='*80}\n")
    
    print(f"{'='*80}\n")
    
    # Consolidate all worker checkpoints into final model
    print(f"\n{'='*80}")
    print(f"HEAD NODE: CONSOLIDATING WORKER CHECKPOINTS")
    print(f"{'='*80}")
    
    consolidate_checkpoints(WORKER_CHECKPOINT_DIR, MODEL_NAME)
    
    print(f"{'='*80}")
    print(f"HEAD NODE: CONSOLIDATION COMPLETE")
    print(f"{'='*80}\n")
    
    ray.shutdown()
