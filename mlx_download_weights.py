from pathlib import Path
from huggingface_hub import snapshot_download
import argparse
from itertools import starmap
import numpy as np
import torch

def map_torch_to_mlx(key, value):
    if "tok_embedding" in key:
        key = "embedding.weight"

    elif "norm" in key:
        key = key.replace("attention_norm", "norm1").replace("ffn_norm", "norm2")

    elif "wq" in key or "wk" in key or "wv" in key or "wo" in key:
        key = key.replace("wq", "query_proj")
        key = key.replace("wk", "key_proj")
        key = key.replace("wv", "value_proj")
        key = key.replace("wo", "out_proj")

    elif "w1" in key or "w2" in key or "w3" in key:
        # The FFN is a separate submodule in PyTorch
        key = key.replace("feed_forward.w1", "linear1")
        key = key.replace("feed_forward.w3", "linear2")
        key = key.replace("feed_forward.w2", "linear3")

    elif "output" in key:
        key = key.replace("output", "out_proj")

    elif "rope" in key:
        return None, None

    return key, value.to(torch.float32).numpy()

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


    print(f"Loading weights from weights/{model_name}/original/consolidated.00.pth")
    torch_weights = Path(f"weights/{model_name}/original/consolidated.00.pth")
    state = torch.load(torch_weights, weights_only=True, map_location="cpu")
    output_file = Path(f"converted/{model_name}.npz")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        **{k: v for k, v in starmap(map_torch_to_mlx, state.items()) if k is not None}
    )
