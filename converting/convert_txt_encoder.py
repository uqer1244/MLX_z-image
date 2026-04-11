import os
import json
import torch
import mlx.core as mx
import mlx.nn as nn
from safetensors.torch import load_file as load_pt_file
import numpy as np
import shutil

# Attempt to import the specific MLX text encoder class
try:
    from mlx_text_encoder import TextEncoderMLX
except ImportError:
    print("Error: mlx_text_encoder module not found.")
    exit(1)


def main():
    # Configuration: Modify these values directly to change paths or settings
    src_path = "Z-Image-Turbo/text_encoder"
    dest_path = "Z-Image-Turbo-MLX-TextEncoder-4bit"
    group_size = 32  # Recommended: 32 for quality, 64 for size

    print("Starting 4-bit Quantization Conversion for Text Encoder")
    print(f"Source: {src_path}")
    print(f"Target: {dest_path}")
    print(f"Group Size: {group_size}")

    # Create the output directory if it does not exist
    os.makedirs(dest_path, exist_ok=True)

    # 1. Load and process the model configuration
    config_path = os.path.join(src_path, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    print("Loading Configuration...")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Initialize the MLX model with the loaded configuration
    model = TextEncoderMLX(config)
    print("Model initialized.")

    # 2. Weight Loading and Conversion
    print("Loading and converting weights...")
    index_path = os.path.join(src_path, "model.safetensors.index.json")
    collected_weights = {}

    # Check if weights are split into multiple shards
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)

        # Get unique shard filenames and sort them
        shard_files = sorted(list(set(index_data["weight_map"].values())))

        for i, filename in enumerate(shard_files):
            file_path = os.path.join(src_path, filename)
            print(f"Processing shard {i + 1}/{len(shard_files)}: {filename}...")

            pt_weights = load_pt_file(file_path)
            for k, v in pt_weights.items():
                # Convert PyTorch Tensor to Numpy, then to MLX Array (BF16)
                if isinstance(v, torch.Tensor):
                    val_np = v.float().numpy()
                else:
                    val_np = v
                collected_weights[k] = mx.array(val_np).astype(mx.bfloat16)

            # Explicit memory management
            del pt_weights
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
    else:
        # Handle case where weights are in a single file
        single_path = os.path.join(src_path, "model.safetensors")
        print("Processing single file: model.safetensors...")
        if not os.path.exists(single_path):
            print(f"Error: Weight file not found at {single_path}")
            return

        pt_weights = load_pt_file(single_path)
        for k, v in pt_weights.items():
            if isinstance(v, torch.Tensor):
                val_np = v.float().numpy()
            else:
                val_np = v
            collected_weights[k] = mx.array(val_np).astype(mx.bfloat16)

    # 3. Quantization
    print(f"Quantizing to 4-bit with Group Size: {group_size}...")

    # Load weights into the model and evaluate parameters before quantization
    model.load_weights(list(collected_weights.items()))
    del collected_weights
    mx.eval(model.parameters())

    # Perform 4-bit quantization
    nn.quantize(model, bits=4, group_size=group_size)
    print("Quantization applied successfully.")

    # 4. Saving the Results
    print("Saving quantized model...")
    weights_path = os.path.join(dest_path, "model.safetensors")
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")

    # Save the configuration file to the destination
    with open(os.path.join(dest_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print("Config saved.")

    # 5. Tokenizer File Copying
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json"
    ]

    copied_count = 0
    for t_file in tokenizer_files:
        src = os.path.join(src_path, t_file)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_path, t_file))
            copied_count += 1

    if copied_count > 0:
        print(f"Copied {copied_count} tokenizer files.")

    print(f"Conversion complete. Model saved to: {dest_path}")


if __name__ == "__main__":
    main()