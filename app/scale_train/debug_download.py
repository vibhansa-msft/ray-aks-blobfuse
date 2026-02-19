import os
from huggingface_hub import snapshot_download

# Use a small test model first to avoid downloading huge files if it works
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_SAVE_PATH = "./debug_output_2"

# Token from the log
HF_TOKEN = ""
masked_token = f"{HF_TOKEN[:4]}...{HF_TOKEN[-4:]}" if HF_TOKEN else "None"

print(f"Attempting download of {MODEL_ID} with token {masked_token}...")

try:
    path = snapshot_download(
        repo_id=MODEL_ID, 
        local_dir=MODEL_SAVE_PATH, 
        # local_dir_use_symlinks=False, # Deprecated
        token=HF_TOKEN,
        max_workers=4 # Limit concurrency for stability
    )
    print(f"Download successful to {path}!")
except Exception as e:
    print(f"Download failed: {e}")
