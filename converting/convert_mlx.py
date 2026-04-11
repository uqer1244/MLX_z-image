import os
import json
import torch
import numpy as np
import mlx.core as mx
from safetensors.torch import load_file as load_pt_file


def map_key_and_convert(key, tensor):
    """
    Converts a PyTorch tensor to an MLX array and maps the key names.
    The data type is set to BF16 for the resulting MLX array.
    """
    if isinstance(tensor, torch.Tensor):
        val = tensor.detach().cpu().float().numpy()
    else:
        val = tensor

    new_key = key

    # Key mapping logic to align with the MLX model structure
    if "t_embedder.mlp.0" in key:
        new_key = key.replace("t_embedder.mlp.0", "t_embedder.linear1")
    elif "t_embedder.mlp.2" in key:
        new_key = key.replace("t_embedder.mlp.2", "t_embedder.linear2")
    elif "all_x_embedder.2-1" in key:
        new_key = key.replace("all_x_embedder.2-1", "x_embedder")
    elif "cap_embedder.0" in key:
        new_key = key.replace("cap_embedder.0", "cap_embedder.layers.0")
    elif "cap_embedder.1" in key:
        new_key = key.replace("cap_embedder.1", "cap_embedder.layers.1")
    elif "all_final_layer.2-1" in key:
        new_key = key.replace("all_final_layer.2-1", "final_layer")

    if "adaLN_modulation.1" in new_key:
        new_key = new_key.replace("adaLN_modulation.1", "adaLN_modulation.layers.1")
    elif "attention.to_out.0" in key:
        new_key = key.replace("attention.to_out.0", "attention.to_out")
    elif "adaLN_modulation.0" in key and "final" not in key:
        new_key = key.replace("adaLN_modulation.0", "adaLN_modulation")
    elif "adaLN_modulation.1" in key and "final" not in key:
        new_key = key.replace("adaLN_modulation.1", "adaLN_modulation")

    # Convert to MLX Array with bfloat16 precision
    return new_key, mx.array(val).astype(mx.bfloat16)


def main():
    # Path configuration (modify as needed)
    src_path = "Z-Image-Turbo/transformer"
    dest_path = "Z-Image-Turbo-mlx-Transformer-BF16"

    print(f"Starting Sharded Conversion: {src_path} -> {dest_path}")
    os.makedirs(dest_path, exist_ok=True)

    # 1. Configuration file (config.json) processing
    config_src = os.path.join(src_path, "config.json")
    if os.path.exists(config_src):
        with open(config_src, "r") as f:
            config = json.load(f)

        # Update key names for MLX compatibility
        if "n_heads" in config and "nheads" not in config:
            config["nheads"] = config["n_heads"]
        config["t_scale"] = config.get("t_scale", 1000.0)

        with open(os.path.join(dest_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        print("Config file updated and saved.")
    else:
        print("Warning: config.json not found in source path.")

    # 2. Load index file and identify weight shards
    index_path = os.path.join(src_path, "diffusion_pytorch_model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"Error: Index file not found at {index_path}")
        return

    with open(index_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    files_to_process = sorted(list(set(weight_map.values())))

    print(f"Found {len(files_to_process)} shards to process.")

    new_weight_map = {}

    # 3. Sequential conversion of each shard file
    for i, filename in enumerate(files_to_process):
        src_file_path = os.path.join(src_path, filename)

        # Standardize filename to model-xxxxx-of-xxxxx.safetensors format
        dest_filename = filename.replace("diffusion_pytorch_model", "model")
        dest_file_path = os.path.join(dest_path, dest_filename)

        print(f"[{i + 1}/{len(files_to_process)}] Converting {filename}...")

        # Load PyTorch weights and perform conversion
        pt_weights = load_pt_file(src_file_path)
        mlx_shard = {}

        for k, v in pt_weights.items():
            new_k, new_v = map_key_and_convert(k, v)
            mlx_shard[new_k] = new_v
            new_weight_map[new_k] = dest_filename

        # Save to MLX safetensors format
        mx.save_safetensors(dest_file_path, mlx_shard)

        # Explicit memory cleanup
        del pt_weights
        del mlx_shard
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    # 4. Generate the new index file for MLX
    new_index_data = {
        "metadata": index_data.get("metadata", {}),
        "weight_map": new_weight_map
    }

    with open(os.path.join(dest_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index_data, f, indent=4)

    print("Conversion completed successfully.")
    print(f"Results saved to: {dest_path}")


if __name__ == "__main__":
    main()