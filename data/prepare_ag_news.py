"""
AG News Dataset Preparation Script

This script downloads the AG News dataset from HuggingFace, converts it to Parquet
format, and shards it for distributed processing. The resulting Parquet files are
uploaded to Azure Blob Storage and consumed by the Ray training jobs.

The AG News dataset contains news articles classified into 4 categories:
  - World (0)
  - Sports (1)
  - Business (2)
  - Science/Technology (3)

Output structure:
  - ag_news_parquet/
    ├── train_00000.parquet
    ├── train_00001.parquet
    ├── ...
    ├── test_00000.parquet
    └── test_00001.parquet
"""

# ========== IMPORTS ==========

from datasets import load_dataset
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os

# ========== CONFIGURATION ==========

# Output directory for sharded Parquet files
OUTPUT_DIR = "ag_news_parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shard size: number of rows per Parquet file
# Smaller shards enable better parallelism in Ray jobs
SHARD_SIZE = 10000

# ========== DATASET PREPARATION ==========

# Check if Parquet files already exist
print(f"Checking if {OUTPUT_DIR} already has Parquet files...")
existing_files = []
if os.path.exists(OUTPUT_DIR):
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.parquet')]

if existing_files:
    print(f"Found {len(existing_files)} existing Parquet files in {OUTPUT_DIR}")
    print("Parquet files already prepared. Skipping dataset preparation.")
else:
    print(f"No existing Parquet files found. Preparing AG News dataset...")
    
    # Process both train and test splits of AG News dataset
    for split in ["train", "test"]:
        print(f"Processing {split} split...")
        
        # Load dataset from HuggingFace
        ds = load_dataset("ag_news", split=split)
        
        # Convert to Pandas DataFrame for manipulation
        df = pd.DataFrame({
            "text": ds["text"],    # Article text
            "label": ds["label"]   # Category label (0-3)
        })
        
        # ===== Sharding =====
        
        # Write data in shards for scalable distributed processing
        for i in range(0, len(df), SHARD_SIZE):
            # Create PyArrow table from Pandas DataFrame shard
            table = pa.Table.from_pandas(df.iloc[i:i+SHARD_SIZE])
            
            # Write to Parquet file with zero-padded shard index
            shard_idx = i // SHARD_SIZE
            output_file = f"{OUTPUT_DIR}/{split}_{shard_idx:05d}.parquet"
            pq.write_table(table, output_file)

# ========== COMPLETION ==========

print(f"Wrote Parquet files to {OUTPUT_DIR}")
print(f"Ready to upload to Azure Blob Storage")

