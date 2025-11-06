"""
Data Preparation Script: Arrow to Parquet Conversion
======================================================

This script converts Arrow IPC format files stored in Azure Blob Storage
to Parquet format using Ray Data and adlfs. Specifically, it will:

1. Connect to Azure Blob Storage using adlfs
2. Read binary Arrow files from the source prefix in Blob Storage
3. Convert each Arrow file to PyArrow tables using streaming IPC reader
4. Write the converted data to Parquet format in the destination prefix in Blob Storage

Expected Data Location:
-----------------------
Input Files (must be pre-uploaded to Azure Blob Storage):
  - Storage Account: Specified by AZURE_STORAGE_ACCOUNT_NAME environment variable
  - Container: "datasets"
  - Path: datasets/openwebtext/raw/
  - Format: Apache Arrow IPC files (.arrow extension)
  - Example: datasets/openwebtext/raw/data-00000-of-00080.arrow

  NOTE: This script assumes the Arrow files are already uploaded to Azure Blob Storage.
        It does not handle uploading the source data. Ensure all .arrow files exist
        in the input path before running this script.

Output Files:
  - Storage Account: Same as input
  - Container: "datasets"
  - Path: datasets/openwebtext/raydata_parquet/
  - Format: Apache Parquet files

Environment Variables:
----------------------
  - AZURE_STORAGE_ACCOUNT_NAME (required): Azure Storage account name
  - AZURE_STORAGE_ACCOUNT_KEY (required): Azure Storage account access key
  - MAX_PREPROCESS_TASK_CONCURRENCY (optional): Max concurrent tasks per stage (default: 2)
"""

import os
import ray
import time
from pathlib import Path
from adlfs import AzureBlobFileSystem
import pyarrow as pa
import pyarrow.parquet as pq


# dataset = "c4"
# dataset_config = "en"  # Config name for C4 dataset (e.g., 'en', 'realnewslike', 'en.noblocklist', 'en.noclean')
# NUM_EXAMPLES = 100000000  # Total dataset has 364,613,570 rows. We choose only 100M to conserve space. Total size on disk is ~400GB data

# Using openwebtext dataset - small dataset for testing
dataset = "openwebtext"
dataset_config = ""  # openwebtext does not have a config
NUM_EXAMPLES = None  # Load entire dataset (8M rows, ~15GB)

AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "")
AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY", "")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "datasets")

# Validate required environment variables
if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
    raise ValueError(
        "Missing required environment variables:\n"
        "  - AZURE_STORAGE_ACCOUNT_NAME\n"
        "  - AZURE_STORAGE_ACCOUNT_KEY\n"
        "Please set these before running the script."
    )


def get_blob_filesystem():
    return AzureBlobFileSystem(
        account_name=AZURE_STORAGE_ACCOUNT_NAME,
        account_key=AZURE_STORAGE_ACCOUNT_KEY
    )


# ==== CONVERT ONE ARROW FILE TO PARQUET IN ADLS ====
@ray.remote
def convert_single_arrow_to_parquet_adlfs(arrow_file, parquet_path, worker_num):
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    fs = get_blob_filesystem()
    base_name = Path(arrow_file).stem
    output_file = parquet_path + base_name + ".parquet"
    
    start_time = time.time()
    print(f"[Worker {worker_num}] Converting {arrow_file} -> {output_file}")
    
    # Read Arrow file size before processing
    arrow_size_mb = fs.size(arrow_file) / (1024 * 1024) if fs.exists(arrow_file) else 0
    
    # Read Arrow with IPC streaming format
    with fs.open(arrow_file, "rb") as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    
    # Write Parquet
    with fs.open(output_file, "wb") as pf:
        pq.write_table(table, pf)
    
    parquet_size_mb = fs.size(output_file) / (1024 * 1024) if fs.exists(output_file) else 0
    compression_ratio = (1 - parquet_size_mb / arrow_size_mb) * 100 if arrow_size_mb > 0 else 0
    elapsed_time = time.time() - start_time
    
    print(f"[Worker {worker_num}] Done {base_name}: {len(table):,} rows | Arrow: {arrow_size_mb:.2f}MB -> Parquet: {parquet_size_mb:.2f}MB | Compression: {compression_ratio:.1f}% | {elapsed_time:.2f}s")
    
    return {
        "arrow": arrow_file,
        "parquet": output_file,
        "rows": len(table),
        "arrow_size_mb": arrow_size_mb,
        "parquet_size_mb": parquet_size_mb,
        "compression_ratio": compression_ratio,
        "time_seconds": elapsed_time,
        "worker_num": worker_num
    }


# ==== MAIN ENTRYPOINT ====
if __name__ == "__main__":
    ray.init(address="auto")
    print(f"ADLFS Data preprocesing booting up...")

    
    # ---- STEP 1: GET ADLFS FILE SYSTEM ----
    fs = get_blob_filesystem()
    
    # ADLFS paths (container/path format, NOT local filesystem paths)
    # These paths are within Azure Blob Storage, not local mounts
    raw_dataset_path = f"{CONTAINER_NAME}/raw/{dataset}/"
    if dataset_config:
        raw_dataset_path = f"{CONTAINER_NAME}/raw/{dataset}/{dataset_config}/"
        
    output_path = f"{CONTAINER_NAME}/{dataset}_adlfs/"

    print(f"Input path (ADLFS): {raw_dataset_path}")
    print(f"Output path (ADLFS): {output_path}")

    # ---- STEP 2: LIST ARROW FILES FROM ADLFS ----
    print(f"\nListing all files in {raw_dataset_path}...")
    try:
        all_files = fs.ls(raw_dataset_path)
        # Filter for .arrow files
        arrow_files = [f for f in all_files if f.endswith('.arrow')]
        arrow_files.sort()
        
        print(f"Found {len(arrow_files)} Arrow files in ADLFS:")
        if not arrow_files:
            print(f"ERROR: No Arrow files found in {raw_dataset_path}")
            exit(1)
    except Exception as e:
        print(f"ERROR listing files: {e}")
        exit(1)

    # ---- STEP 3: DISTRIBUTED CONVERSION ----
    print(f"\nSubmitting Ray tasks for Arrow -> Parquet conversion (distributed)...")
    print(f"Processing {len(arrow_files)} files with Ray workers...\n")
    convert_jobs = [
        convert_single_arrow_to_parquet_adlfs.remote(arrow_file, output_path, worker_num)
        for worker_num, arrow_file in enumerate(arrow_files, 1)
    ]

    # Collect Results
    results = ray.get(convert_jobs)
    
    # Print comprehensive statistics
    print("\n" + "="*120)
    print("CONVERSION STATISTICS")
    print("="*120)
    
    total_rows = 0
    total_arrow_mb = 0
    total_parquet_mb = 0
    total_processing_time = 0
    
    for i, r in enumerate(results, 1):
        rows = r["rows"]
        arrow_mb = r["arrow_size_mb"]
        parquet_mb = r["parquet_size_mb"]
        time_sec = r["time_seconds"]
        compression = r["compression_ratio"]
        
        print(f"{i:2d}. {Path(r['arrow']).name:30s} | {rows:>10,} rows | Arrow: {arrow_mb:>8.2f}MB | Parquet: {parquet_mb:>8.2f}MB | Compression: {compression:>5.1f}% | {time_sec:>7.2f}s")
        
        total_rows += rows
        total_arrow_mb += arrow_mb
        total_parquet_mb += parquet_mb
        total_processing_time += time_sec
    
    avg_time_per_file = total_processing_time / len(results) if results else 0
    total_compression = (1 - total_parquet_mb / total_arrow_mb) * 100 if total_arrow_mb > 0 else 0
    
    print("="*120)
    print(f"TOTAL: {len(results)} files | {total_rows:,} rows | Arrow: {total_arrow_mb:.2f}MB | Parquet: {total_parquet_mb:.2f}MB | Compression: {total_compression:.1f}%")
    print(f"Total processing time: {total_processing_time:.2f}s")
    print(f"Average per file: {avg_time_per_file:.2f}s")
    print("="*120 + "\n")
    
    print("Conversion completed for all files.")