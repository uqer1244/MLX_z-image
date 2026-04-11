import os
import json
import torch
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from safetensors.torch import load_file as load_pt_file
from mlx_z_image import ZImageTransformerMLX
from tqdm import tqdm

# Model configuration based on the Z-Image-Turbo transformer config
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

def remap_qkv(key, state_dict):
    """
    Splits combined QKV weights from ComfyUI format back into separate
    to_q, to_k, and to_v weights for Diffusers compatibility.
    """
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
    """
    Renames keys from ComfyUI specific naming to match the
    expected model architecture.
    """
    new_key = key
    for r, rr in replace_keys.items():
        new_key = new_key.replace(r, rr)
    state_dict[new_key] = state_dict.pop(key)

def map_key_and_convert(key, tensor):
    """
    Converts PyTorch tensors to MLX arrays and maps key names
    to align with the ZImageTransformerMLX structure.
    """
    if isinstance(tensor, torch.Tensor):
        val = tensor.detach().cpu().float().numpy()
    else:
        val = tensor

    new_key = key

    # Specific key mapping logic for transformer components
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

    # Remove the diffusion model prefix and cast to BF16
    return (
        new_key.replace("model.diffusion_model.", ""),
        mx.array(val).astype(mx.bfloat16),
    )

def main():
    # Configuration: Modify these variables directly
    src_model = "comfy.safetensors"
    dst_model = "mlx_model_4bit.safetensors"
    group_size = 32

    print("Starting conversion process")
    print(f"Loading {src_model}...")

    # Load original weights using safetensors
    pt_weights = load_pt_file(src_model)

    # Remove problematic keys if present
    if "model.diffusion_model.norm_final.weight" in pt_weights.keys():
        del(pt_weights["model.diffusion_model.norm_final.weight"])

    print("Reverting ComfyUI format to standard naming...")
    keys = list(pt_weights.keys())

    for k in tqdm(keys):
        if ".qkv." in k:
            remap_qkv(k, pt_weights)
        else:
            remap_keys(k, pt_weights)

    print("Converting weights to MLX format...")
    mlx_weights = []
    for k, v in tqdm(pt_weights.items()):
        mlx_weights.append(map_key_and_convert(k, v))

    print("Initializing MLX model and loading weights...")
    model = ZImageTransformerMLX(config)
    model.load_weights(mlx_weights)

    print(f"Applying 4-bit quantization (Group Size: {group_size})...")
    nn.quantize(model, bits=4, group_size=group_size)

    print(f"Saving quantized model to {dst_model}...")
    quant_weights = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(dst_model, quant_weights)

    print("Conversion and quantization completed successfully.")

if __name__ == "__main__":
    main()