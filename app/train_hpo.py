"""
Ray HPO Training Script for AG News Dataset

This script implements hyperparameter optimization (HPO) for a DistilBERT sequence
classification model trained on the AG News dataset. It uses Ray Tune for distributed
HPO with ASHA scheduling and Ray Train for distributed training across a Ray cluster.

Key components:
  - Data loading from Azure Blob Storage via Parquet files
  - DistilBERT model training with PyTorch
  - Ray Tune hyperparameter search with ASHA scheduler
  - Distributed training via Ray TorchTrainer

"""

# ========== IMPORTS ==========

import os
import glob
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
import transformers
import torch



# ========== CONFIGURATION ==========

# Directory for blob storage - mount point for Azure Blob storage (training data - read-only)
DATA_DIR = os.getenv("DATA_DIR", "/mnt/blob/datasets")

# Directory for checkpoints - mount point for storing model checkpoints (read-write)
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/mnt/blob/checkpoints")

# ========== DATA LOADING ==========

def load_parquet_ds(pattern: str):
    """
    Load parquet files matching a pattern from blob storage.
    
    Args:
        pattern: Glob pattern to match parquet files (e.g., "/mnt/blob/datasets/train_0001*.parquet")
    
    Returns:
        Ray Dataset materialized in memory for efficient access
    """
    files = glob.glob(pattern)
    print(f"[DEBUG] Glob pattern: {pattern}")
    print(f"[DEBUG] Files found: {len(files)}")
    print(f"[DEBUG] Files: {files}")
    
    # If glob returns no files, print directory listing for debugging
    if len(files) == 0:
        base_dir = os.path.dirname(pattern)
        print(f"[DEBUG] No files found! Listing directory: {base_dir}")
        try:
            dir_contents = os.listdir(base_dir)
            print(f"[DEBUG] Directory contents: {dir_contents}")
            # List subdirectories too
            for item in dir_contents[:20]:  # Limit to first 20
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    try:
                        subdir_contents = os.listdir(item_path)
                        print(f"[DEBUG] {item}/ contains {len(subdir_contents)} items: {subdir_contents[:5]}")
                    except Exception as e:
                        print(f"[DEBUG] Error listing {item}/: {e}")
        except Exception as e:
            print(f"[DEBUG] Error listing {base_dir}: {e}")
    
    # Materialize dataset to ensure all data is available for training
    ds = ray.data.read_parquet(files, parallelism=min(200, max(1, len(files)))).materialize()
    return ds

# ========== TRAINING LOOP ==========

def train_loop(config):
    """
    Training loop executed on each Ray worker during distributed training.
    
    This function is called by TorchTrainer and executed on each worker.
    It loads the model, optimizes it for the given batch, and reports metrics
    back to Ray Tune for hyperparameter search.
    
    Args:
        config: Dictionary containing hyperparameter values:
            - lr: Learning rate for AdamW optimizer
            - batch_size: Training batch size
            - epochs: Number of training epochs
    """
    from ray import train
    import socket
    
    # ===== Model Setup =====
    
    # Load tokenizer and model from HuggingFace
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    )
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # ===== Hyperparameter Configuration =====
    
    # Extract hyperparameters from config with defaults
    lr = config.get("lr", 3e-5)
    batch_size = config.get("batch_size", 2)
    epochs = config.get("epochs", 1)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # ===== Data Access =====
    
    # Get distributed dataset shard for this worker
    shard = train.get_dataset_shard("train")
    
    # ===== Training Loop =====
    
    # Iterate through epochs (set to 1 for single iteration)
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        batch_count = 0
        
        # Process batches from dataset shard
        for batch in shard.iter_batches(batch_size=batch_size):
            batch_count += 1
            print(f"Processing batch {batch_count}")
            
            # Extract text and labels from batch
            # batch is a dictionary with 'text' and 'label' keys
            texts = batch["text"] if isinstance(batch["text"], list) else batch["text"].tolist()
            labels_raw = batch["label"] if isinstance(batch["label"], list) else batch["label"].tolist()
            
            # Tokenize batch texts and move to device
            enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            labels = torch.tensor(labels_raw)
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            
            # Forward pass through model
            out = model(**enc, labels=labels)
            loss = out.loss
            
            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Batch {batch_count} loss: {loss.item():.4f}")
            
            # Stop after first batch only
            print("Stopping after first batch as configured")
            break
        
        # Report loss to Ray Tune for metric tracking and scheduling decisions
        print(f"Epoch {epoch + 1} completed. Reporting loss to Ray Tune.")
        train.report({"loss": float(loss.item())})
    
    # ===== Save Checkpoint =====
    
    # Create checkpoint directory if it doesn't exist (read-write mount point)
    # Checkpoints are saved to a subdirectory within the checkpoints mount
    checkpoint_subdir = os.path.join(CHECKPOINT_DIR, "model_checkpoints")
    os.makedirs(checkpoint_subdir, exist_ok=True)
    
    # Save model checkpoint with timestamp for versioning
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_subdir, f"model_checkpoint_{timestamp}.pt")
    
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float(loss.item())
        }, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        raise

# ========== HYPERPARAMETER OPTIMIZATION ==========

def main():
    """
    Main entry point for Ray HPO job.
    
    Sets up Ray cluster connection, loads training data, configures distributed
    training, and runs hyperparameter optimization using Ray Tune with ASHA scheduler.
    """
    
    # ===== Ray Cluster Setup =====
    
    # Connect to Ray cluster (running on AKS via RayJob)
    ray.init(address="auto")
    
    # ===== Verify Mount Points =====
    
    # Check if data directory exists and is readable
    print(f"\n=== Verifying Mount Points ===")
    print(f"DATA_DIR: {DATA_DIR}")
    if os.path.exists(DATA_DIR):
        print(f"✓ DATA_DIR exists")
        try:
            contents = os.listdir(DATA_DIR)
            print(f"  Contents: {contents}")
        except Exception as e:
            print(f"  Error listing contents: {e}")
    else:
        print(f"✗ DATA_DIR does not exist")
    
    print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    if os.path.exists(CHECKPOINT_DIR):
        print(f"✓ CHECKPOINT_DIR exists")
        try:
            contents = os.listdir(CHECKPOINT_DIR)
            print(f"  Contents: {contents}")
        except Exception as e:
            print(f"  Error listing contents: {e}")
    else:
        print(f"✗ CHECKPOINT_DIR does not exist")
    
    # ===== Data Loading =====
    
    # Load training dataset from blob storage (read-only mount point)
    # The datasets PVC mounts the dataset container which has ag_news subdirectory
    print(f"Loading dataset from {DATA_DIR}/ag_news/...")
    print(f"[DEBUG] Full mount point path: {DATA_DIR}")
    print(f"[DEBUG] Looking for pattern: {DATA_DIR}/ag_news/train_*.parquet")
    
    # List what's at the mount point
    try:
        mount_contents = os.listdir(DATA_DIR)
        print(f"[DEBUG] Mount point contents: {mount_contents}")
    except Exception as e:
        print(f"[DEBUG] Error listing mount point: {e}")
    
    try:
        train_ds = load_parquet_ds(f"{DATA_DIR}/ag_news/train_*.parquet")
        print(f"Dataset loaded successfully with {train_ds.count()} records")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # ===== Trainer Configuration =====
    
    # Configure distributed Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        scaling_config=ScalingConfig(
            num_workers=int(os.getenv("NUM_WORKERS", "1")),
            use_gpu=torch.cuda.is_available()
        ),
        datasets={"train": train_ds},
        run_config=RunConfig(name="agnews-distilbert-training"),
    )
    
    # ===== Single Training Run =====
    
    # Run one training iteration with default hyperparameters
    print("Starting single training iteration...")
    result = trainer.fit()
    print("Training completed successfully!")
    print(f"Training result: {result}")


# ========== ENTRY POINT ==========

if __name__ == "__main__":
    print("New file with storage account")
    main()

