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

import adlfs
import ray.data
import pyarrow

STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
# Sets max concurrency for each Ray task. This is important to help
# limit the amount of memory being pulled in as the .arrow files are +200MB
# and are processed as a single task for each part of its pipeline. So too
# many of these tasks running at once can exhaust memory.
MAX_TASK_CONCURRENCY = int(os.environ.get("MAX_PREPROCESS_TASK_CONCURRENCY", 2))
ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

DATASET_NAME = "openwebtext"
CONTAINER_NAME = "datasets"


def validate_env_vars():
    if not STORAGE_ACCOUNT_NAME:
        raise ValueError(
            "AZURE_STORAGE_ACCOUNT_NAME environment variable is not set. Please set it to proceed."
        )
    if not ACCOUNT_KEY:
        raise ValueError(
            "AZURE_STORAGE_ACCOUNT_KEY environment variable is not set. Please set it to proceed."
        )


def get_blob_filesystem():
    fsspec_fs = adlfs.AzureBlobFileSystem(
        account_name=STORAGE_ACCOUNT_NAME,
        account_key=ACCOUNT_KEY,
    )
    return fsspec_fs


def convert_arrow_binary_to_table(batch):
    with pyarrow.ipc.open_stream(batch["bytes"][0]) as reader:
        return reader.read_all()


def main():
    validate_env_vars()

    ray.init(address="auto")
    # ray.init() # Uncomment for local testing

    filesystem = get_blob_filesystem()

    raw_path = f"{CONTAINER_NAME}/{DATASET_NAME}/raw/"
    output_path = f"{CONTAINER_NAME}/{DATASET_NAME}/raydata_parquet/"

    print(f"Processing {DATASET_NAME} dataset with Ray Data...")
    print(f"Available Ray resources: {ray.cluster_resources()}")
    print(f"Input path: {raw_path}")
    print(f"Output path: {output_path}")

    print("Loading dataset from Azure Blob Storage...")
    ds = ray.data.read_binary_files(
        raw_path,
        filesystem=filesystem,
        concurrency=MAX_TASK_CONCURRENCY,
    )

    print("Converting binary Arrow data to PyArrow table...")
    ds_table = ds.map_batches(
        convert_arrow_binary_to_table,
        batch_size=1,
        concurrency=MAX_TASK_CONCURRENCY,
    )

    print("Writing to Parquet format in Azure Blob Storage...")
    ds_table.write_parquet(
        output_path,
        filesystem=filesystem,
        concurrency=MAX_TASK_CONCURRENCY,
    )

    print(f"✅ Dataset successfully converted and saved to: {output_path}")
    ray.shutdown()


if __name__ == "__main__":
    main()
