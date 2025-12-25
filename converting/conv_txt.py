import argparse
import os
import json
import torch
import mlx.core as mx
import mlx.nn as nn
from safetensors.torch import load_file as load_pt_file
import numpy as np
import shutil

# ❗ 중요: 같은 폴더에 있는 mlx_text_encoder.py에서 클래스를 가져옵니다.
try:
    from mlx_text_encoder import TextEncoderMLX
except ImportError:
    print("❌ Error: 'mlx_text_encoder.py' not found. Please place it in the same directory.")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert & Quantize Text Encoder to MLX (4-bit)")
    parser.add_argument("--src_path", type=str, default="Z-Image-Turbo/text_encoder",
                        help="Path to PyTorch model folder")
    parser.add_argument("--dest_path", type=str, default="Z-Image-Turbo-MLX-TextEncoder-4bit",
                        help="Output path")
    parser.add_argument("--group_size", type=int, default=32,
                        help="Quantization group size (Recommended: 32 for quality, 64 for size)")
    args = parser.parse_args()

    print(f"🚀 Starting 4-bit Quantization Conversion")
    print(f"   Source: {args.src_path}")
    print(f"   Target: {args.dest_path}")
    print(f"   Group Size: {args.group_size}")

    os.makedirs(args.dest_path, exist_ok=True)

    # 1. Config 로드 및 모델 초기화
    config_path = os.path.join(args.src_path, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ Error: Config not found at {config_path}")
        return

    print("\n[1/4] Loading Configuration...")
    with open(config_path, "r") as f:
        config = json.load(f)

    # 모델 인스턴스 생성 (구조만 생성됨)
    model = TextEncoderMLX(config)
    print("   ✅ Model initialized.")

    # 2. PyTorch 가중치 로드 및 통합
    print("\n[2/4] Loading & Converting Weights (This may take memory)...")

    index_path = os.path.join(args.src_path, "model.safetensors.index.json")
    collected_weights = {}

    if os.path.exists(index_path):
        # 샤딩된 가중치 로드
        with open(index_path, "r") as f:
            index_data = json.load(f)
        shard_files = sorted(list(set(index_data["weight_map"].values())))

        for i, filename in enumerate(shard_files):
            file_path = os.path.join(args.src_path, filename)
            print(f"   Processing shard {i + 1}/{len(shard_files)}: {filename}...")

            pt_weights = load_pt_file(file_path)
            for k, v in pt_weights.items():
                # PyTorch Tensor -> Numpy -> MLX Array
                if isinstance(v, torch.Tensor):
                    # bfloat16 to float32 conversion for numpy compatibility
                    val_np = v.float().numpy()
                else:
                    val_np = v

                # MLX로 변환 (아직은 FP16/BF16 상태)
                collected_weights[k] = mx.array(val_np).astype(mx.bfloat16)

            del pt_weights  # 메모리 확보
            if hasattr(mx, "clear_cache"): mx.clear_cache()
    else:
        # 단일 파일 로드
        single_path = os.path.join(args.src_path, "model.safetensors")
        print(f"   Processing single file: model.safetensors...")
        pt_weights = load_pt_file(single_path)
        for k, v in pt_weights.items():
            if isinstance(v, torch.Tensor):
                val_np = v.float().numpy()
            else:
                val_np = v
            collected_weights[k] = mx.array(val_np).astype(mx.bfloat16)

    # 3. 모델에 가중치 로드 및 양자화
    print(f"\n[3/4] Quantizing to 4-bit (Group Size: {args.group_size})...")

    # 수집한 가중치를 모델에 주입
    model.load_weights(list(collected_weights.items()))
    del collected_weights  # 메모리 해제
    mx.eval(model.parameters())

    # 🔥 핵심: MLX 내장 양자화 함수 실행
    # Linear 레이어들을 QuantizedLinear로 교체하고 가중치를 압축함
    nn.quantize(model, bits=4, group_size=args.group_size)

    print("   ✅ Quantization applied successfully.")

    # 4. 저장
    print("\n[4/4] Saving Quantized Model...")

    # 가중치 저장 (MLX 포맷 - safetensors)
    weights_path = os.path.join(args.dest_path, "model.safetensors")
    model.save_weights(weights_path)
    print(f"   ✅ Weights saved to {weights_path}")

    # Config 복사
    with open(os.path.join(args.dest_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print("   ✅ Config saved.")

    # (옵션) Tokenizer 파일들이 있다면 복사
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
    copied_count = 0
    for t_file in tokenizer_files:
        src = os.path.join(args.src_path, t_file)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.dest_path, t_file))
            copied_count += 1

    if copied_count > 0:
        print(f"   ✅ Copied {copied_count} tokenizer files.")

    print(f"\n🎉 All Done! Model saved to: {args.dest_path}")
    print(f"💡 Usage in Pipeline: remove 'nn.quantize(...)' and just load this model.")


if __name__ == "__main__":
    main()