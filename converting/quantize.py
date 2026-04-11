import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import json
import os
import glob
from mlx_z_image import ZImageTransformerMLX

def main():
    # 설정 (경로 및 세팅)
    model_path = "Z-Image-Turbo-mlx-Transformer-BF16"
    dest_path = "Z-Image-Turbo-mlx-Transformer-4bit"
    group_size = 32

    print(f"Starting 4-bit Quantization (Group Size: {group_size})")
    print(f"Source: {model_path}")
    print(f"Destination: {dest_path}")

    # 목적지 디렉토리가 없으면 생성
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # 1. 설정 파일 처리
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        print("Error: config.json not found in the model path.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # 목적지 폴더에 config 저장
    with open(os.path.join(dest_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print("Configuration file copied.")

    # 2. 모델 초기화 및 가중치 로드
    print("Initializing model...")
    model = ZImageTransformerMLX(config)

    print("Loading sharded weights...")
    # .safetensors 확장자를 가진 모든 파일을 찾습니다 (model-00001-of-00003.safetensors 등)
    weight_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    if not weight_files:
        print(f"Error: No .safetensors files found in {model_path}")
        return

    # 모든 샤드 파일을 로드하여 하나의 딕셔너리로 합칩니다.
    weights = {}
    for wf in weight_files:
        print(f"  Loading: {os.path.basename(wf)}")
        weights.update(mx.load(wf))

    # 로드된 가중치를 모델에 주입합니다.
    # tree_unflatten을 통해 평탄화된 딕셔너리를 모델의 트리 구조에 맞게 변환합니다.
    model.update(mlx.utils.tree_unflatten(list(weights.items())))
    print("All weights loaded and model updated.")

    # 3. 4비트 양자화 적용
    print(f"Applying quantization (bits: 4, group_size: {group_size})...")
    # 터보퀀트 알고리즘이 적용된 선형 레이어들을 4비트로 변환합니다.
    nn.quantize(model, bits=4, group_size=group_size)

    # 4. 양자화된 가중치 저장
    save_path = os.path.join(dest_path, "model.safetensors")
    print(f"Saving quantized model to {save_path}...")

    # 모델 파라미터를 다시 딕셔너리 형태로 추출하여 저장합니다.
    quantized_weights = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(save_path, quantized_weights)

    print("Quantization process completed successfully.")

if __name__ == "__main__":
    main()