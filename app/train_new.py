import os
import time
import torch
import glob
import ray
from ray.air import session
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import ray.data

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

# For reproducibility
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

def train_func(config):
    # Model & tokenizer paths
    model_path = MODEL_CACHE_DIR
    tokenizer_path = MODEL_CACHE_DIR

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(DEVICE)
    model.train()

    # Get Ray Data dataset shard for this worker from session
    ds_shard = session.get_dataset_shard("train")
    
    # Get worker rank and world size
    world_rank = session.get_world_rank()
    world_size = session.get_world_size()
    print(f"[Worker {world_rank}/{world_size}] Starting training with Ray Data shard")
    
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
        print(f"[Worker {world_rank}] Epoch {epoch + 1}/{config.get('epochs', 1)}")
        total_loss = 0.0
        num_batches = 0
        
        if simulate:
            print(f"[Worker {world_rank}] Simulating training...")
            print(f"[Worker {world_rank}] Dataset shard assigned - batch size: {batch_size}")
            
            # Sleep to represent actual training time
            print(f"[Worker {world_rank}] Sleeping 10 seconds to simulate training...")
            time.sleep(10)
            epoch_loss = 0.0
        else:
            print(f"[Worker {world_rank}] Processing data batches...")
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
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"[Worker {world_rank}] Batch {batch_idx + 1} - Loss: {loss.item():.6f}")
                except Exception as e:
                    print(f"[Worker {world_rank}] ERROR processing batch {batch_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"[Worker {world_rank}] Epoch {epoch + 1} completed - Avg Loss: {epoch_loss:.6f}")
        
        # Save checkpoint every epoch
        checkpoint_dir_epoch = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir_epoch, exist_ok=True)
        
        # Save model state dict
        checkpoint_path = os.path.join(checkpoint_dir_epoch, f"worker_{world_rank}_model.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Worker {world_rank}] Checkpoint saved to {checkpoint_path}")
        
        # Report metrics WITHOUT checkpoint (Ray checkpoint storage is causing issues)
        # The checkpoints are already saved to BlobFuse directly
        session.report({"loss": epoch_loss, "epoch": epoch + 1})

def aggregate_checkpoints(checkpoint_dir, num_workers, last_epoch, output_path):
    """
    Aggregate checkpoints from all workers into a final model.
    Each worker saves its checkpoint in checkpoint_dir/epoch_{epoch}/worker_{rank}_model.pt
    """
    print(f"\n{'='*80}")
    print(f"[Aggregation] Starting checkpoint aggregation")
    print(f"{'='*80}")
    print(f"[Aggregation] Looking for checkpoints in {checkpoint_dir}")
    print(f"[Aggregation] Expected to find {num_workers} worker checkpoints\n")
    
    # Find all available worker checkpoints
    loaded_workers = []
    for rank in range(num_workers):
        worker_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{last_epoch}", f"worker_{rank}_model.pt")
        if os.path.exists(worker_ckpt_path):
            loaded_workers.append(rank)
            print(f"  ✓ Worker {rank}: Checkpoint found")
        else:
            print(f"  ✗ Worker {rank}: Checkpoint not found")
    
    if not loaded_workers:
        print(f"\n[Aggregation] ERROR: No checkpoints found for aggregation!")
        return
    
    print(f"\n[Aggregation] Found {len(loaded_workers)} worker checkpoints")
    print(f"[Aggregation] Loaded from workers: {loaded_workers}\n")
    
    if DEVICE == "cpu":
        # CPU mode: simulate aggregation by iterating through checkpoints
        print(f"[Aggregation] CPU mode - simulating aggregation process...")
        for idx, rank in enumerate(loaded_workers):
            worker_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{last_epoch}", f"worker_{rank}_model.pt")
            print(f"  Processing checkpoint {idx + 1}/{len(loaded_workers)} from worker {rank}...")
            time.sleep(5)  # Simulate aggregation work
        
        print(f"[Aggregation] Simulated aggregation complete!")
        
        # Load first checkpoint as the "aggregated" model for CPU simulation
        first_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{last_epoch}", f"worker_{loaded_workers[0]}_model.pt")
        aggregated = torch.load(first_ckpt_path, map_location="cpu")
    else:
        # GPU mode: load all checkpoints and perform actual averaging
        print(f"[Aggregation] GPU mode - loading all checkpoints and averaging weights...")
        state_dicts = []
        for rank in loaded_workers:
            worker_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{last_epoch}", f"worker_{rank}_model.pt")
            state_dict = torch.load(worker_ckpt_path, map_location="cpu")
            state_dicts.append(state_dict)
            print(f"  Loaded checkpoint from worker {rank}")
        
        print(f"\n[Aggregation] Averaging weights across {len(state_dicts)} workers...")
        # Average weights across all workers
        aggregated = {}
        for key in state_dicts[0].keys():
            tensors = [sd[key] for sd in state_dicts]
            stacked = torch.stack(tensors)
            aggregated[key] = torch.mean(stacked, dim=0)
        
        print(f"[Aggregation] Weight averaging complete!")
    
    # Save final aggregated model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(aggregated, output_path)
    print(f"\n[Aggregation] ✓ Final aggregated model saved to: {output_path}")
    print(f"[Aggregation] File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")
    print(f"{'='*80}\n")

def main():
    ray.init()

    # Discover parquet files from the glob pattern
    print(f"Discovering parquet files from: {PARQUET_PATH}")
    parquet_files = sorted(glob.glob(PARQUET_PATH))
    
    if not parquet_files:
        print(f"ERROR: No parquet files found at {PARQUET_PATH}")
        ray.shutdown()
        exit(1)
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load parquet files with Ray Data (automatically sharded across workers)
    ds = ray.data.read_parquet(parquet_files).repartition(NUM_WORKERS)
    print(f"Ray Data dataset created with {len(parquet_files)} files")

    checkpoint_dir = WORKER_CHECKPOINT_DIR
    num_workers = NUM_WORKERS
    epochs = 2

    config = {
        "batch_size": 2 if DEVICE == "cpu" else 8,
        "checkpoint_dir": checkpoint_dir,
        "model_path": MODEL_CACHE_DIR,
        "tokenizer_path": MODEL_CACHE_DIR,
        "simulate_training": not USE_GPU,  # set to True for CPU, False for GPU/real
        "epochs": epochs,
    }

    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=USE_GPU)
    
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
    print(f"Starting distributed training with {num_workers} workers...")
    results = trainer.fit()
    print("Training results:", results.metrics)

    # Aggregate checkpoints for final model
    print(f"Aggregating {num_workers} worker checkpoints...")
    final_model_path = os.path.join(TRAINED_MODEL_DIR, "{model_type}_finetuned.pt")
    aggregate_checkpoints(checkpoint_dir, num_workers, last_epoch=epochs-1, output_path=final_model_path)
    
    print(f"Training complete! Final model: {final_model_path}")
    ray.shutdown()

if __name__ == "__main__":
    main()
