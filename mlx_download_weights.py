from pathlib import Path
from huggingface_hub import snapshot_download
import argparse
from itertools import starmap
import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument("--weight_dir", default="weights", help="Directory to store the weights")
    parser.add_argument("--checkpoint", default="meta-llama/Llama-3.2-1B-Instruct", help="Hugging Face model checkpoint to use")
    parser.add_argument("--convert_to_mlx", action="store_true", default=True, help="Convert the weights to MLX format")
    args = parser.parse_args()

    model_name = (args.checkpoint).split("/")[1]
    checkpoint = args.checkpoint

    if not Path(f"weights/{model_name}").exists():
        print(f"Can't find the weights for {checkpoint}. Downloading...")
        model_path = snapshot_download(
            repo_id=checkpoint, local_dir=Path(f"weights/{model_name}")
        )
    else:
        print(f"Weights for {checkpoint} already exist at weights/{model_name}.")
    print(f"Weights for {checkpoint} are downloaded.")
