import argparse
import os
import torch
from safetensors import safe_open


def analyze_safetensors(path):
    print(f"🔍 Analyzing Safetensors: {path}")

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata()

        print(f"Total Keys: {len(keys)}")
        if metadata:
            print(f"Metadata: {metadata}")

        rank = "Unknown"
        for k in keys:
            if "lora_down" in k or "down" in k:
                tensor = f.get_tensor(k)
                shape = tensor.shape
                rank = min(shape)
                print(f"📐 Detected Rank: {rank} (based on {k} -> {shape})")
                break

        print("\nKey Structure Sample (First 20):")
        for k in keys[:20]:
            print(f"   - {k}")

        print("\nCompatibility Check:")
        sample_key = keys[0]
        if "lora_unet" in sample_key:
            print("   Type: SD1.5 / SDXL (Uses 'lora_unet' prefix)")
        elif "transformer.layers" in sample_key or "model.layers" in sample_key:
            print("   Type: Transformer-based (Likely Z-Image or SD3)")
        elif "double_blocks" in sample_key:
            print("   Type: FLUX.1 (Uses 'double_blocks')")
        else:
            print("   Type: Unknown / Custom")


def analyze_torch(path):
    print(f"Analyzing PyTorch (.pt/.bin): {path}")
    state_dict = torch.load(path, map_location="cpu")
    keys = list(state_dict.keys())

    print(f"📊 Total Keys: {len(keys)}")
    print("\nKey Structure Sample (First 20):")
    for k in keys[:20]:
        print(f"   - {k}")

def main():
    parser = argparse.ArgumentParser(description="lora Structure Inspector")
    parser.add_argument("--path", type=str, help="Path to .safetensors or .pt file")
    args = parser.parse_args()

    analyze_safetensors("lora/Z-blackgold-style.safetensors")

if __name__ == "__main__":
    main()