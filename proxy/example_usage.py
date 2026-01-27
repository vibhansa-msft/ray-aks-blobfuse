#!/usr/bin/env python3
"""
Example usage of the S3 to Azure Storage Proxy.

This script demonstrates how to use the proxy with boto3 to interact
with Azure Blob Storage or Azure Files using S3 API.
"""
import boto3
from botocore.client import Config
import os


def main():
    # Configure S3 client to use the proxy
    # In production, set S3_ENDPOINT_URL environment variable
    endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:5000")
    
    print(f"Connecting to S3 proxy at: {endpoint_url}")
    
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id='dummy',  # Not validated by proxy
        aws_secret_access_key='dummy',  # Not validated by proxy
        config=Config(signature_version='s3v4')
    )
    
    # Example bucket name (maps to Azure container/share)
    bucket = 'mybucket'
    
    print("\n=== S3 to Azure Storage Proxy Example ===\n")
    
    # 1. Upload a file (PUT)
    print("1. Uploading a file...")
    key = 'test/example.txt'
    content = b'Hello from S3 API to Azure Storage!'
    
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=content)
        print(f"   ✓ Uploaded: {bucket}/{key}")
    except Exception as e:
        print(f"   ✗ Upload failed: {e}")
    
    # 2. Download a file (GET)
    print("\n2. Downloading the file...")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read()
        print(f"   ✓ Downloaded: {data.decode('utf-8')}")
    except Exception as e:
        print(f"   ✗ Download failed: {e}")
    
    # 3. Get file metadata (HEAD)
    print("\n3. Getting file metadata...")
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
        print(f"   ✓ Size: {response['ContentLength']} bytes")
        print(f"   ✓ Type: {response['ContentType']}")
        print(f"   ✓ ETag: {response['ETag']}")
    except Exception as e:
        print(f"   ✗ Metadata failed: {e}")
    
    # 4. List objects (LIST)
    print("\n4. Listing objects in bucket...")
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix='test/')
        if 'Contents' in response:
            print(f"   ✓ Found {len(response['Contents'])} object(s):")
            for obj in response['Contents']:
                print(f"     - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print("   ✓ No objects found")
    except Exception as e:
        print(f"   ✗ List failed: {e}")
    
    # 5. Delete a file (DELETE)
    print("\n5. Deleting the file...")
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        print(f"   ✓ Deleted: {bucket}/{key}")
    except Exception as e:
        print(f"   ✗ Delete failed: {e}")
    
    print("\n=== Example completed ===\n")
    print("Note: Make sure the proxy is running and configured correctly.")
    print("Set STORAGE_BACKEND to 'blob' or 'files' to choose Azure backend.")


if __name__ == "__main__":
    main()
