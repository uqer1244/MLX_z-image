import argparse
import os
import json
import torch
import mlx.core as mx
import numpy as np
from diffusers import AutoencoderKL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--dest_path", type=str, default="Z-Image-Turbo-MLX-VAE")
    args = parser.parse_args()

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)

    print(f"📥 Loading PyTorch VAE from {args.model_id}...")
    # Load original VAE to get config and state_dict
    vae_pt = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")

    # Save Config
    config = vae_pt.config
    with open(os.path.join(args.dest_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("🔄 Converting Weights...")
    pt_state_dict = vae_pt.state_dict()
    mlx_weights = {}

    for k, v in pt_state_dict.items():
        # We only need the decoder for generation
        if not k.startswith("decoder"):
            continue

        # Convert PyTorch tensor to Numpy
        val = v.detach().cpu().numpy().astype(np.float32)

        # PyTorch (N, C, H, W) -> MLX (N, H, W, C) for Conv2d
        if len(val.shape) == 4:
            val = val.transpose(0, 2, 3, 1)

        mlx_weights[k] = mx.array(val)

    # Save to safe tensors
    save_path = os.path.join(args.dest_path, "vae_model.safetensors")
    mx.save_safetensors(save_path, mlx_weights)
    print(f"✅ VAE Converted & Saved to {save_path}")


if __name__ == "__main__":
    main()