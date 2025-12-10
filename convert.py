import os
import torch
import numpy as np
import mlx.core as mx
import json
import argparse
from safetensors.torch import load_file
from huggingface_hub import snapshot_download


def map_key_and_convert(key, tensor):
    val = tensor.detach().cpu().numpy().astype(np.float32)
    new_key = key

    # === 매핑 규칙 ===
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

    return new_key, mx.array(val)


def main():
    parser = argparse.ArgumentParser(description="Convert Z-Image-Turbo weights to MLX format")
    parser.add_argument("--model_id", type=str, default="Z-Image-Turbo/Z-Image-Turbo",
                        help="HuggingFace Model ID or local path")
    parser.add_argument("--dest_path", type=str, default="mlx_model", help="Directory to save MLX weights")
    args = parser.parse_args()

    print(f"🚀 Starting Conversion: {args.model_id} -> MLX FP16")

    src_path = args.model_id
    # 로컬 경로가 아니면 HuggingFace에서 다운로드
    if not os.path.exists(src_path):
        print(f"📥 Downloading model from HuggingFace Hub: {src_path}")
        try:
            src_path = snapshot_download(repo_id=src_path, allow_patterns=["*.safetensors", "*.json"])
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)

    # 2. Config 복사 & 수정
    config_path = os.path.join(src_path, "transformer")  # 구조에 따라 경로 조정 필요할 수 있음
    if not os.path.exists(os.path.join(config_path, "config.json")):
        config_path = src_path  # transformer 폴더가 없는 경우 대비

    try:
        with open(os.path.join(config_path, "config.json"), "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ config.json not found in {config_path}")
        return

    # 누락된 키 보정
    if "n_heads" in config and "nheads" not in config:
        config["nheads"] = config["n_heads"]
    config["t_scale"] = config.get("t_scale", 1000.0)

    with open(os.path.join(args.dest_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print(f"✅ Config saved to {args.dest_path}/config.json")

    # 3. PyTorch 가중치 로드
    print("📥 Loading PyTorch weights...")
    pt_state_dict = {}

    # transformer 폴더가 있는지 확인
    weight_path = os.path.join(src_path, "transformer")
    if not os.path.exists(weight_path):
        weight_path = src_path

    files = [f for f in os.listdir(weight_path) if f.endswith(".safetensors") and "diffusion_pytorch_model" in f]
    files.sort()

    if not files:
        print(f"❌ No safetensors found in {weight_path}")
        return

    for f in files:
        print(f"  - Reading {f}...")
        pt_state_dict.update(load_file(os.path.join(weight_path, f), device="cpu"))

    # 4. 변환 (FP16으로)
    print("🔄 Converting weights & Mapping keys...")
    mlx_state_dict = {}
    for k, v in pt_state_dict.items():
        new_k, new_v = map_key_and_convert(k, v)
        mlx_state_dict[new_k] = new_v.astype(mx.float16)  # FP16 변환

    # 5. Pad Token 추가
    dim = config['dim']
    if 'x_pad_token' not in mlx_state_dict:
        mlx_state_dict['x_pad_token'] = mx.zeros((1, dim), dtype=mx.float16)
    if 'cap_pad_token' not in mlx_state_dict:
        mlx_state_dict['cap_pad_token'] = mx.zeros((1, dim), dtype=mx.float16)

    # 6. 저장
    save_file = os.path.join(args.dest_path, "model.safetensors")
    print(f"💾 Saving to {save_file}...")
    mx.save_safetensors(save_file, mlx_state_dict)

    print(f"🎉 Conversion Done! Saved to {args.dest_path}")


if __name__ == "__main__":
    main()