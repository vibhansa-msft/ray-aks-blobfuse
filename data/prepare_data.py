import ray
import os 
import ray.data

@ray.remote
def download_dataset(dataset, raw_dataset_path):
    from datasets import load_dataset
    
    # Load dataset (from Hugging Face) and save to given path
    print(f"Downloading {dataset} from Hugging Face...")
    hf_dataset = load_dataset(dataset, split="train", streaming=False, trust_remote_code=True)
    
    print(f"Saving {dataset} to {raw_dataset_path} ...")
    hf_dataset.save_to_disk(raw_dataset_path)
    print("Download finished and saved!")
    
    return "Completed Download !!"

@ray.remote
def preprocess_data(raw_dataset_path, output_path):
    from datasets import load_from_disk

    print(f"Loading cached dataset from: {raw_dataset_path}")
    # Load the cached HuggingFace dataset and convert to Ray Dataset
    hf_dataset = load_from_disk(raw_dataset_path)
    ds = ray.data.from_huggingface(hf_dataset)
    
    # Get record count without loading all data into memory
    print("Dataset loaded successfully. Starting processing...")
    
    # Inspect a small sample, if desired
    print("Sample data:", ds.take(1))
    
    # Use smaller batch size and more partitions to reduce memory usage
    # OpenWebText is ~8M records, so 100+ partitions helps with memory
    ds = ds.repartition(100)
    
    # Write to Parquet in parallel with memory-efficient settings
    print(f"Writing data to {output_path} ...")
    ds.write_parquet(
        output_path,
        # Use smaller row groups to reduce memory per batch
        row_group_size=50000,  # Smaller batches 
    )
    print("Preprocessing finished!")
    
    return "Completed SAving !!"


# Initialize Ray (for local or cluster use; omit address for single machine)
ray.init(address="auto")

# Choose the data and output path (use CHECKPOINT_DIR which is read-write)
dataset = "openwebtext"
output_path = os.path.join(os.getenv("CHECKPOINT_DIR", "/mnt/blob/checkpoints"), dataset)
raw_dataset_path = os.path.join(os.getenv("CHECKPOINT_DIR", "/mnt/blob/checkpoints"), "raw", dataset)

# Check if raw dataset already exists to avoid re-downloading
if os.path.exists(raw_dataset_path) and os.listdir(raw_dataset_path):
    print(f"Raw dataset already exists at {raw_dataset_path}. Skipping download.")
    result = "Skipped Download - Data Already Exists"
else:
    # Run the method to download the data and save it to storage account
    print(f"Raw dataset not found. Starting download...")
    result = ray.get(download_dataset.remote(dataset, raw_dataset_path))
print(result)

# Run the method to preprocess the data and save it to storage account in parquet format
result = ray.get(preprocess_data.remote(raw_dataset_path, output_path))
print(result)




