import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import ray
from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model, get_device
from ray.train import ScalingConfig, RunConfig, Checkpoint
import ray.data
import pandas as pd
from functools import partial
import torch.distributed as dist
import tempfile
import glob

# --- Configuration from Environment Variables ---
DATA_DIR = os.getenv("DATA_DIR", "/mnt/blob/datasets")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/mnt/blob/checkpoints")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "40"))

MODEL_NAME = os.getenv("MODEL_NAME", "openai-community/gpt2-large")

BATCH_SIZE_PER_WORKER = int(os.getenv("BATCH_SIZE_PER_WORKER", "1"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))

# Construct paths
PARQUET_PATH = os.path.join(DATA_DIR.rstrip("/"), "openwebtext/*.parquet")
CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR.rstrip("/"), "checkpoint/openwebtext/")

print(f"--- Head Node: Configuration ---")
print(f"Data Pattern: {PARQUET_PATH}")
print(f"Model: {MODEL_NAME}")
print(f"Total Workers: {NUM_WORKERS}")
print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
print()

# --------------------------------------------------
# Tokenization Function
# --------------------------------------------------
def tokenize_function(batch, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    output = tokenizer(
        batch["text"].tolist(), 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    output["labels"] = output["input_ids"].clone() 
    return output

# --------------------------------------------------
# Training Function (executed on each worker process)
# --------------------------------------------------
def train_loop_per_worker(config):
    rank = ray.train.get_context().get_world_rank()
    print(f"[Worker {rank}]: Starting CPU training function.")

    try:
        device = torch.device("cpu")
        model_name = config.get("model_name", MODEL_NAME)
        print(f"[Worker {rank}]: Loading model {model_name}...")
        
        # Try to load from local cache first, fall back to downloading if needed
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
            print(f"[Worker {rank}]: Model loaded from local cache.")
        except Exception as e:
            print(f"[Worker {rank}]: Local cache miss, attempting to download: {e}")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            print(f"[Worker {rank}]: Model downloaded from Hugging Face Hub.")
        
        model.to(device)
        print(f"[Worker {rank}]: Model loaded successfully.")

        # Prepare model for distributed training (uses DDP for CPU backend)
        model = prepare_model(model)
        optimizer = AdamW(model.parameters(), lr=config["lr"])
        
        # Get the specific shard of the dataset assigned to this worker
        print(f"[Worker {rank}]: Getting dataset shard...")
        dataset_shard = ray.train.get_dataset_shard("train")
        print(f"[Worker {rank}]: Dataset shard obtained.")

        print(f"[Worker {rank}]: Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            print(f"[Worker {rank}]: Tokenizer loaded from local cache.")
        except Exception as e:
            print(f"[Worker {rank}]: Local tokenizer cache miss, attempting to download: {e}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"[Worker {rank}]: Tokenizer downloaded from Hugging Face Hub.")
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        print(f"[Worker {rank}]: Tokenizer loaded.")

        for epoch in range(config["num_epochs"]):
            print(f"[Worker {rank}]: Epoch {epoch+1} started.")
            
            # Iterate over batches in the dataset shard
            # dataset_shard is a StreamSplitDataIterator, so we iterate directly
            batch_idx = 0
            for batch in dataset_shard.iter_batches(batch_size=config["batch_size_per_worker"], prefetch_batches=2):
                print(f"[Worker {rank}]: Processing batch {batch_idx}...")
                
                # Tokenize batch on the worker (lazy tokenization)
                tokenized_batch = tokenize_function(batch, tokenizer)
                
                # Convert to tensors
                input_ids = torch.tensor(tokenized_batch["input_ids"], dtype=torch.long).to(device)
                attention_mask = torch.tensor(tokenized_batch["attention_mask"], dtype=torch.long).to(device)
                labels = torch.tensor(tokenized_batch["labels"], dtype=torch.long).to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                batch_idx += 1
                
                # Report metrics every N batches
                if batch_idx % 10 == 0:
                    metrics = {"loss": loss.item(), "epoch": epoch, "batch_index": batch_idx}
                    if rank == 0:
                        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                            torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
                            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                            ray.train.report(metrics, checkpoint=checkpoint)
                    else:
                        ray.train.report(metrics)
                    print(f"[Worker {rank}]: Checkpoint for batch {batch_idx} reported.")
                    
                break
            
            # Final checkpoint at end of epoch
            print(f"[Worker {rank}]: Epoch {epoch+1} complete. Final sync...")
            dist.barrier()
            
            final_metrics = {"loss": loss.item(), "epoch": epoch, "batch_index": batch_idx, "epoch_complete": True}
            if rank == 0:
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    ray.train.report(final_metrics, checkpoint=checkpoint)
            else:
                ray.train.report(final_metrics)
            
            print(f"[Worker {rank}]: Final checkpoint for epoch {epoch+1} reported.")
    except Exception as e:
        print(f"[Worker {rank}]: ERROR in training loop: {e}")
        import traceback
        traceback.print_exc()
        raise

# --------------------------------------------------
# 5. Main Execution Script (runs on the head node driver)
# --------------------------------------------------
if __name__ == "__main__":
    import tempfile # Make sure tempfile is imported

    print("--- Head Node: Driver Script Started ---")
    
    if not ray.is_initialized():
        ray.init(address="auto") 

    # Resolve parquet files from glob pattern
    print(f"Head Node: Resolving parquet files from pattern: {PARQUET_PATH}")
    parquet_files = sorted(glob.glob(PARQUET_PATH))
    print(f"Head Node: Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        print(f"ERROR: No parquet files found matching pattern: {PARQUET_PATH}")
        ray.shutdown()
        exit(1)

    # Data Ingestion - use directory path instead of glob pattern
    parquet_dir = os.path.dirname(parquet_files[0])
    print(f"Head Node: Creating lazy Ray Dataset plan from directory: {parquet_dir}")
    ray_dataset = ray.data.read_parquet(parquet_dir)
    print(f"Head Node: Dataset plan created. Ray will automatically parallelize reads across workers.")

    # Preprocessing - Tokenization happens on workers, not on head
    # This avoids loading the entire dataset into memory on head node
    print("Head Node: Skipping head-node tokenization. Tokenization will happen on workers.")

    # Configure and Launch Training
    print("Head Node: Configuring TorchTrainer for CPU-based DDP distributed training...")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "model_name": MODEL_NAME,
            "lr": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "batch_size_per_worker": BATCH_SIZE_PER_WORKER
        },
        scaling_config=ScalingConfig(
            num_workers=NUM_WORKERS,
            use_gpu=False # Explicitly set to False for CPU cluster
        ),
        datasets={"train": ray_dataset},
        run_config=RunConfig(
            storage_path=CHECKPOINT_DIR
        )
    )

    print("Head Node: Launching the distributed fine-tuning job on CPUs with file-based checkpointing...")
    result = trainer.fit()
    print("Head Node: Training finished!")
    print(f"Head Node: Training results: {result.metrics}")
