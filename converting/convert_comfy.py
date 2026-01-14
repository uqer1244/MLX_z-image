import os
import argparse
import json
import torch
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from safetensors.torch import load_file as load_pt_file
from mlx_z_image import ZImageTransformerMLX
from tqdm import tqdm

# From https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/transformer/config.json
config = {
    "_class_name": "ZImageTransformer2DModel",
    "_diffusers_version": "0.36.0.dev0",
    "all_f_patch_size": [1],
    "all_patch_size": [2],
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
    "cap_feat_dim": 2560,
    "dim": 3840,
    "in_channels": 16,
    "n_heads": 30,
    "n_kv_heads": 30,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "norm_eps": 1e-05,
    "qk_norm": True,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "nheads": 30,
}


# Helpers functions to revert ComfyUI single file model to diffusers format state_dict
# Some keys are already in the target naming, but i'll revert them nonetheless to use
# the original code as-is. Undo what is done here:
# https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/z_image_convert_original_to_comfy.py
def remap_qkv(key, state_dict):
    weight = state_dict.pop(key)
    to_q, to_k, to_v = weight.chunk(3, dim=0)
    state_dict[key.replace(".qkv.", ".to_q.")] = to_q
    state_dict[key.replace(".qkv.", ".to_k.")] = to_k
    state_dict[key.replace(".qkv.", ".to_v.")] = to_v


replace_keys = {
    "final_layer.": "all_final_layer.2-1.",
    "x_embedder.": "all_x_embedder.2-1.",
    ".attention.out.bias": ".attention.to_out.0.bias",
    ".attention.k_norm.weight": ".attention.norm_k.weight",
    ".attention.q_norm.weight": ".attention.norm_q.weight",
    ".attention.out.weight": ".attention.to_out.0.weight",
}


def remap_keys(key, state_dict):
    new_key = key
    for r, rr in replace_keys.items():
        new_key = new_key.replace(r, rr)
    state_dict[new_key] = state_dict.pop(key)


# Torch to MLX converter function from https://github.com/uqer1244/MLX_z-image/blob/master/converting/convert.py
def map_key_and_convert(key, tensor):
    # PyTorch Tensor -> Numpy (Float32)
    # BF16 변환은 나중에 MLX array 생성 시 수행
    if isinstance(tensor, torch.Tensor):
        val = tensor.detach().cpu().float().numpy()
    else:
        val = tensor

    new_key = key

    # 키 매핑 로직 (기존과 동일)
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

    # Changed to a tuple from original code to allow loading without saving to disk
    # also remove the "model.diffusion_model." prefix
    return (
        new_key.replace("model.diffusion_model.", ""),
        mx.array(val).astype(mx.bfloat16),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert and Quantize ZIT AIO safetensors to MLX model in 4-bit"
    )
    parser.add_argument(
        "--src_model",
        type=str,
        default="comfy.safetensors",
        help="Path to ZIT model in ComfyUI format",
    )
    parser.add_argument(
        "--dst_model",
        type=str,
        default="mlx.safetensors",
        help="Path to save quantized model in mlx format",
    )
    parser.add_argument(
        "--group_size", type=int, default=32, help="Group size for quantization"
    )
    args = parser.parse_args()

    print("Starting conversion!")

    print(f"Loading {args.src_model} file...")

    pt_weights = load_pt_file(args.src_model)

    # Remove an unexpected key. TODO: figure out from where it cames.
    if "model.diffusion_model.norm_final.weight" in pt_weights.keys():
        del(pt_weights["model.diffusion_model.norm_final.weight"])

    print("Reverting ComfyUI format...")

    keys = list(pt_weights.keys())

    for k in tqdm(keys):
        if ".qkv." in k:
            remap_qkv(k, pt_weights)
        else:
            remap_keys(k, pt_weights)

    print("Converting to MLX...")

    mlx_weights = []

    for k, v in tqdm(pt_weights.items()):
        mlx_weights.append(map_key_and_convert(k, v))

    print("Loading converted weights...")

    model = ZImageTransformerMLX(config)
    model.load_weights(mlx_weights)

    print(f"Quantizing (bits=4, group_size={args.group_size})...")

    nn.quantize(model, bits=4, group_size=args.group_size)

    print(f"Saving {args.dst_model} file...")

    quant_weights = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(args.dst_model, quant_weights)

    print("Done!")


if __name__ == "__main__":
    main()
