import ray
import os 
import ray.data
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pathlib



dataset = "c4"
dataset_config = "en"  # Config name for C4 dataset (e.g., 'en', 'realnewslike', 'en.noblocklist', 'en.noclean')
NUM_EXAMPLES = 30000000  # Total dataset has 364,613,570 rows. We choose only 30M to conserve space. Total size on disk is ~400GB data

# dataset = "openwebtext"
# dataset_config = ""
# NUM_EXAMPLES = None  # Total dataset has 8M rows and roughly 80 * 800 = 15GB worth data

# Choose the data and output path
# Read from DATA_DIR (input location), write to CHECKPOINT_DIR (output location)
data_dir = os.getenv("DATA_DIR", "/mnt/blob/data")
checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/mnt/blob/checkpoints")

raw_dataset_path = os.path.join(data_dir, "raw", dataset, dataset_config)
output_path = os.path.join(checkpoint_dir, dataset)

@ray.remote
def download_dataset(dataset, raw_dataset_path):
    from datasets import load_dataset
    import pyarrow as pa
    
    # Load dataset (from Hugging Face) and save to given path
    print(f"Downloading {dataset} from Hugging Face...")
    
    if dataset_config and NUM_EXAMPLES is not None:
        print(f"Using config: {dataset_config} - Streaming with manual partitioning")
        
        start_time = time.time()
        
        # Load dataset from HuggingFace using datasets library (streaming mode)
        print(f"Loading dataset with streaming=True for memory efficiency...")
        hf_dataset = load_dataset(
            dataset, 
            dataset_config, 
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        # Stream and partition data into Arrow files
        rows_per_file = 200000
        current_buffer = []
        file_count = 0
        total_rows = 0
        
        print(f"Streaming data and partitioning into files (~{rows_per_file:,} rows per file)...")
        
        for i, example in enumerate(hf_dataset):
            # Stop at NUM_EXAMPLES
            if i >= NUM_EXAMPLES:
                break
            
            current_buffer.append(example)
            total_rows += 1
            
            # Write file when buffer is full
            if len(current_buffer) >= rows_per_file:
                table = pa.Table.from_pylist(current_buffer)
                file_path = os.path.join(raw_dataset_path, f"data_shard-{file_count:05d}.arrow")
                
                # Write using PyArrow IPC format (streaming format)
                with open(file_path, 'wb') as f:
                    writer = pa.ipc.new_stream(f, table.schema)
                    writer.write_table(table)
                    writer.close()
                
                current_buffer = []
                file_count += 1
                
                print(f"  Written {file_count} files ({total_rows:,} rows so far)...\n")
        
        # Write remaining data
        if current_buffer:
            table = pa.Table.from_pylist(current_buffer)
            file_path = os.path.join(raw_dataset_path, f"data_shard-{file_count:05d}.arrow")
            
            with open(file_path, 'wb') as f:
                writer = pa.ipc.new_stream(f, table.schema)
                writer.write_table(table)
                writer.close()
            
            file_count += 1
        
        end_time = time.time()
        
        # Count the Arrow files created
        arrow_files = sorted(list(pathlib.Path(raw_dataset_path).glob("*.arrow")))
        num_files = len(arrow_files)
        
        # Calculate file statistics
        total_size_mb = sum(f.stat().st_size for f in arrow_files) / (1024 * 1024)
        avg_file_size_mb = total_size_mb / num_files if num_files > 0 else 0
        
        print(f"\n=======================================================")
        print(f"Successfully downloaded {total_rows:,} rows using streaming")
        print(f"Total Arrow files written: {num_files}")
        print(f"Total data size: {total_size_mb:.2f} MB")
        print(f"Average file size: {avg_file_size_mb:.2f} MB per file")
        print(f"Files saved in: {raw_dataset_path}")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"=======================================================")
    else:
        # Fallback for datasets without config or when no limit is set
        print(f"Using direct HuggingFace API (non-streaming approach)")
        hf_dataset = load_dataset(dataset, split="train", streaming=False, trust_remote_code=True)

        # As we are not in streaming mode, everything is already loaded in memory so lets just dump it
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
    
    # Get source arrow file size
    arrow_file_size_mb = os.path.getsize(arrow_file_path) / (1024 * 1024)
    
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
    parquet_file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    compression_ratio = (1 - parquet_file_size_mb / arrow_file_size_mb) * 100 if arrow_file_size_mb > 0 else 0
    
    print(f"Converted {Path(arrow_file_path).name} ({len(table)} rows, Arrow: {arrow_file_size_mb:.2f}MB -> Parquet: {parquet_file_size_mb:.2f}MB, Compression: {compression_ratio:.1f}%) in {elapsed_time:.2f}s")
    
    return {
        "file": Path(arrow_file_path).name,
        "rows": len(table),
        "arrow_size_mb": arrow_file_size_mb,
        "parquet_size_mb": parquet_file_size_mb,
        "compression_ratio": compression_ratio,
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
    print("\n" + "="*100)
    print("CONVERSION STATISTICS")
    print("="*100)
    
    total_rows = 0
    total_arrow_mb = 0
    total_parquet_mb = 0
    total_processing_time = 0  # Sum of individual file processing times
    
    for i, result in enumerate(results, 1):
        rows = result["rows"]
        arrow_mb = result["arrow_size_mb"]
        parquet_mb = result["parquet_size_mb"]
        time_sec = result["time_seconds"]
        compression = result["compression_ratio"]
        
        print(f"{i:2d}. {result['file']:25s} | {rows:>8,} rows | Arrow: {arrow_mb:>7.2f}MB | Parquet: {parquet_mb:>7.2f}MB | Compression: {compression:>5.1f}% | {time_sec:>6.2f}s")
        
        total_rows += rows
        total_arrow_mb += arrow_mb
        total_parquet_mb += parquet_mb
        total_processing_time += time_sec
    
    avg_time_per_file = total_processing_time / len(results) if results else 0
    total_compression = (1 - total_parquet_mb / total_arrow_mb) * 100 if total_arrow_mb > 0 else 0
    overall_throughput = total_parquet_mb / total_time if total_time > 0 else 0
    
    print("="*100)
    print(f"TOTAL: {len(results)} files | {total_rows:,} rows | Arrow: {total_arrow_mb:.2f}MB | Parquet: {total_parquet_mb:.2f}MB | Compression: {total_compression:.1f}%")
    print(f"Wall-clock time (cluster execution): {total_time:.2f}s")
    print(f"Sum of individual processing times: {total_processing_time:.2f}s (represents total CPU work across all workers)")
    print(f"Average per file: {avg_time_per_file:.2f}s")
    print(f"Overall throughput: {overall_throughput:.2f}MB/s")
    print("="*100 + "\n")
    
    print("Preprocessing finished! Dataset converted from Arrow to Parquet format.")
    
    return f"Completed: {len(results)} files converted in {total_time:.2f}s"


# Initialize Ray (for local or cluster use; omit address for single machine)
ray.init(address="auto")

# Check if raw dataset already exists to avoid re-downloading
if os.path.exists(raw_dataset_path) and os.listdir(raw_dataset_path):
    print(f"Raw dataset already exists at {raw_dataset_path}. Skipping download.")
    result = "Skipped Download - Data Already Exists"
else:
    # Run the method to download the data and save it to storage account
    print(f"Raw dataset not found. Starting download...")
    raw_dataset_path = os.path.join(checkpoint_dir, "raw", dataset, dataset_config)
    os.makedirs(raw_dataset_path, exist_ok=True)
    result = ray.get(download_dataset.remote(dataset, raw_dataset_path))
    
print(result)

# Run the method to preprocess the data and save it to storage account in parquet format
result = ray.get(preprocess_data.remote(raw_dataset_path, output_path))
print(result)




