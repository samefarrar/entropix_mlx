from pathlib import Path
from huggingface_hub import snapshot_download

# Specify the checkpoint
checkpoint = "meta-llama/Llama-3.2-1B-Instruct"

if not Path("weights/1B-Instruct").exists():
    print(f"Can't find the weights for {checkpoint}. Downloading...")
    model_path = snapshot_download(
        repo_id=checkpoint, local_dir=Path("weights/1B-Instruct")
    )
else:
    print(f"Weights for {checkpoint} already exist at weights/1B-Instruct.")

print(f"Weights for {checkpoint} are downloaded.")
