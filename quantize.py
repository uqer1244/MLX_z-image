import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import json
import os
import argparse
from mlx_z_image import ZImageTransformerMLX

def main():
    parser = argparse.ArgumentParser(description="Quantize MLX model to 4-bit")
    parser.add_argument("--model_path", type=str, default="mlx_model", help="Path to FP16 MLX model")
    parser.add_argument("--dest_path", type=str, default="mlx_model_4bit", help="Path to save quantized model")
    parser.add_argument("--group_size", type=int, default=32, help="Group size for quantization")
    args = parser.parse_args()

    print(f"🚀 Starting 4-bit Quantization (Group Size: {args.group_size})")

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)

    config_path = os.path.join(args.model_path, "config.json")
    if not os.path.exists(config_path):
        print("❌ config.json not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
    with open(os.path.join(args.dest_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("📥 Loading FP16 Model...")
    model = ZImageTransformerMLX(config)
    model.load_weights(os.path.join(args.model_path, "model.safetensors"))

    print(f"🔨 Quantizing (bits=4, group_size={args.group_size})...")
    nn.quantize(model, bits=4, group_size=args.group_size)

    save_path = os.path.join(args.dest_path, "model.safetensors")
    print(f"💾 Saving quantized model to {save_path}...")

    weights = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(save_path, weights)

    print("🎉 Quantization Complete!")

if __name__ == "__main__":
    main()