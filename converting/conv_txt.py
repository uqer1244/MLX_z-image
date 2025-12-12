import argparse
import os
import json
import torch
import mlx.core as mx
import mlx.nn as nn
import mlx.utils  # utils 추가
import numpy as np
from transformers import AutoModel
from mlx_text_encoder import TextEncoderMLX


def main():
    # 경로 설정
    src_path = "Z-Image-Turbo/text_encoder"
    dest_path = "Z-Image-Turbo-MLX-TextEncoder"

    print(f"🚀 Starting Conversion: {src_path} -> {dest_path}")

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # 1. Config 로드
    config_src = os.path.join(src_path, "config.json")
    if os.path.exists(config_src):
        with open(config_src, "r") as f:
            config = json.load(f)

        with open(os.path.join(dest_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        print(f"✅ Config Loaded: Hidden={config['hidden_size']}, HeadDim={config['head_dim']}")
    else:
        print("❌ config.json not found in source path.")
        return

    print("📥 Loading PyTorch Model...")
    try:
        pt_model = AutoModel.from_pretrained(src_path, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        print(f"❌ Failed to load PyTorch model: {e}")
        return

    print("🏗️ Building MLX Model...")
    mlx_model = TextEncoderMLX(config)

    print("🔄 Converting Weights & Mapping Keys...")
    pt_state_dict = pt_model.state_dict()
    mlx_weights = {}

    for k, v in pt_state_dict.items():
        val = v.detach().cpu().numpy().astype(np.float32)

        # Linear Transpose 제거 (1:1 매핑)

        new_key = k
        if not k.startswith("model."):
            new_key = f"model.{k}"

        mlx_weights[new_key] = mx.array(val)

    try:
        mlx_model.load_weights(list(mlx_weights.items()))
        print("✅ Weights Loaded Successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    print("🔨 Quantizing to 4-bit (Group Size: 32)...")
    nn.quantize(mlx_model, bits=4, group_size=32)

    save_file = os.path.join(dest_path, "model.safetensors")
    print(f"💾 Saving to {save_file}...")

    # [수정] 안전한 저장 로직: tree_flatten을 사용하여 확실하게 평탄화
    # dict(mlx_model.parameters()) 대신 아래 방식을 사용하면 bad_cast 방지 가능
    weights = dict(mlx.utils.tree_flatten(mlx_model.parameters()))

    mx.save_safetensors(save_file, weights)

    print("🎉 Conversion Complete!")


if __name__ == "__main__":
    main()