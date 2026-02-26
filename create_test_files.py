"""Create 10 x 1MB test files in /mnt/cluster_storage to validate shared storage mount."""
import os

SHARED_DIR = "/mnt/cluster_storage"
NUM_FILES = 10
FILE_SIZE = 1024 * 1024  # 1 MB

os.makedirs(SHARED_DIR, exist_ok=True)

for i in range(NUM_FILES):
    path = os.path.join(SHARED_DIR, f"testfile_{i:02d}.bin")
    with open(path, "wb") as f:
        f.write(os.urandom(FILE_SIZE))
    size_kb = os.path.getsize(path) / 1024
    print(f"Created {path} ({size_kb:.0f} KB)")

# List all files
print(f"\n--- Contents of {SHARED_DIR} ---")
for entry in sorted(os.listdir(SHARED_DIR)):
    full = os.path.join(SHARED_DIR, entry)
    if os.path.isfile(full):
        print(f"  {entry}  ({os.path.getsize(full) / 1024:.0f} KB)")

total = sum(os.path.getsize(os.path.join(SHARED_DIR, f)) for f in os.listdir(SHARED_DIR) if os.path.isfile(os.path.join(SHARED_DIR, f)))
print(f"\nTotal: {total / (1024*1024):.1f} MB across {len(os.listdir(SHARED_DIR))} files")
print("SUCCESS")
