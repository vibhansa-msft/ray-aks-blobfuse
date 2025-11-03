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
def process_arrow_file(arrow_file_path, output_path):
    """Convert a single arrow file to parquet format - runs on a worker node"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path
    import os
    import time
    
    arrow_filename = Path(arrow_file_path).stem
    print(f"Worker processing arrow file: {Path(arrow_file_path).name}")
    
    start_time = time.time()
    
    # Read arrow file using pyarrow.memory_map and deserialize
    # HuggingFace uses Arrow streaming format, not the IPC format
    try:
        # Try reading as streaming format first
        with open(str(arrow_file_path), 'rb') as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()
    except:
        # Fallback: try as IPC file format
        table = pa.ipc.open_file(pa.memory_map(str(arrow_file_path), 'r')).read_all()
    
    # Write to parquet
    output_file = os.path.join(output_path, f"{arrow_filename}.parquet")
    pq.write_table(table, output_file)
    
    elapsed_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"Converted {Path(arrow_file_path).name} ({len(table)} rows, {file_size_mb:.2f}MB) in {elapsed_time:.2f}s\n")
    
    return {
        "file": Path(arrow_file_path).name,
        "rows": len(table),
        "size_mb": file_size_mb,
        "time_seconds": elapsed_time
    }

@ray.remote
def preprocess_data(raw_dataset_path, output_path):
    import pyarrow.parquet as pq
    import pyarrow as pa
    from pathlib import Path
    import time
    
    print(f"Reading cached dataset from: {raw_dataset_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory ready: {output_path}")
    
    # Find all arrow files in the dataset directory
    arrow_files = sorted(Path(raw_dataset_path).glob("*.arrow"))
    print(f"Found {len(arrow_files)} arrow files")
    
    if not arrow_files:
        raise FileNotFoundError(f"No arrow files found in {raw_dataset_path}")
    
    print(f"Submitting {len(arrow_files)} arrow file processing tasks for PARALLEL execution...")
    print(f"With {int(ray.cluster_resources().get('CPU', 1))} available CPUs, tasks will run in parallel")
    
    # Record total start time
    total_start_time = time.time()
    
    # Submit all tasks at once - Ray will schedule them across available workers
    # Each arrow file is processed by a different worker in PARALLEL
    task_refs = [
        process_arrow_file.remote(str(arrow_file), output_path)
        for arrow_file in arrow_files
    ]
    
    # Now wait for all to complete
    results = ray.get(task_refs)
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Print statistics
    print("\n" + "="*80)
    print("CONVERSION STATISTICS")
    print("="*80)
    
    total_rows = 0
    total_size_mb = 0
    
    for i, result in enumerate(results, 1):
        rows = result["rows"]
        size_mb = result["size_mb"]
        time_sec = result["time_seconds"]
        throughput_mb_per_sec = size_mb / time_sec if time_sec > 0 else 0
        
        print(f"{i:2d}. {result['file']:25s} | {rows:>8,} rows | {size_mb:>8.2f}MB | {time_sec:>6.2f}s | {throughput_mb_per_sec:>7.2f}MB/s")
        
        total_rows += rows
        total_size_mb += size_mb
    
    avg_time_per_file = total_time / len(results) if results else 0
    overall_throughput = total_size_mb / total_time if total_time > 0 else 0
    
    print("="*80)
    print(f"TOTAL: {len(results)} files | {total_rows:,} rows | {total_size_mb:.2f}MB | {total_time:.2f}s | {overall_throughput:.2f}MB/s")
    print(f"Average per file: {avg_time_per_file:.2f}s")
    print("="*80 + "\n")
    
    print("Preprocessing finished! Dataset converted from Arrow to Parquet format.")
    
    return f"Completed: {len(results)} files converted in {total_time:.2f}s"


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




