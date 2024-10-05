from huggingface_hub import snapshot_download

# Specify the checkpoint
checkpoint = "meta-llama/Llama-3.2-1B-Instruct"

if not Path("weights/1B-Instruct").exists():
    model_path = snapshot_download(
        repo_id=checkpoint, local_dir=Path("weights/1B-Instruct")
    )
