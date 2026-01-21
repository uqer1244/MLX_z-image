import mlx.core as mx
import mlx.nn as nn
import os
import re
import math
from safetensors import safe_open


class LoRALinearWrapper(nn.Module):
    def __init__(self, base_layer, lora_a, lora_b, scale=1.0):
        super().__init__()
        self.base_layer = base_layer
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scale = scale

    def __call__(self, x):
        base_out = self.base_layer(x)

        # MLX LoRA calculation
        # x @ A.T @ B.T * scale

        dtype = x.dtype
        x = x.astype(self.lora_a.dtype)

        # Shape Auto-Correction (Auto-Transpose)
        a = self.lora_a
        b = self.lora_b

        # A (Down): Should be (In, Rank) (Must match the last dimension of input x)
        if a.shape[0] != x.shape[-1]:
            a = a.T

        # B (Up): Should be (Rank, Out) (Must match the last dimension of A)
        if b.shape[0] != a.shape[-1]:
            b = b.T

        lora_out = (x @ a @ b) * self.scale
        return base_out + lora_out.astype(dtype)


def get_module_by_name(model, module_name):
    parts = module_name.split('.')
    obj = model
    for part in parts:
        try:
            if part.isdigit():
                idx = int(part)
                if isinstance(obj, list):
                    obj = obj[idx]
                elif isinstance(obj, dict):
                    obj = obj[idx] if idx in obj else obj[part]
                else:
                    obj = getattr(obj, part) if hasattr(obj, part) else None
            else:
                obj = getattr(obj, part) if hasattr(obj, part) else None

            if obj is None: return None
        except:
            return None
    return obj


def set_module_by_name(model, module_name, new_module):
    parts = module_name.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            idx = int(part)
            if isinstance(parent, list):
                parent = parent[idx]
            elif isinstance(parent, dict):
                parent = parent[idx] if idx in parent else parent[part]
            else:
                parent = getattr(parent, part)
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        idx = int(last)
        if isinstance(parent, list):
            parent[idx] = new_module
        elif isinstance(parent, dict):
            parent[idx] = new_module
    else:
        setattr(parent, last, new_module)


def convert_unet_key_to_mlx(key):
    # 1. Remove prefixes
    new_key = key.replace("lora_unet_", "")
    new_key = new_key.replace("diffusion_model.", "")  # z-image-anime style

    # 2. Unify layer number format (layers_0_ -> layers.0.)
    new_key = re.sub(r'layers_(\d+)_', r'layers.\1.', new_key)

    # 3. Map main module names (Underscore -> Dot)
    # Attention QKV/Out
    new_key = new_key.replace("attention_to_q", "attention.to_q")
    new_key = new_key.replace("attention_to_k", "attention.to_k")
    new_key = new_key.replace("attention_to_v", "attention.to_v")

    # Output projection (AI Toolkit often uses to_out_0)
    new_key = new_key.replace("attention_to_out_0", "attention.to_out")
    new_key = new_key.replace("attention_to_out", "attention.to_out")

    # Feed Forward
    new_key = new_key.replace("feed_forward_w1", "feed_forward.w1")
    new_key = new_key.replace("feed_forward_w2", "feed_forward.w2")
    new_key = new_key.replace("feed_forward_w3", "feed_forward.w3")

    # Modulation (adaLN)
    # adaLN inside Z-Image blocks is usually a single Linear or Sequential
    # Depending on the file, it might be split into adaLN_modulation_0, 1, etc.
    if "adaLN_modulation" in new_key:
        # Attempt simple mapping first (May vary by structure)
        if "adaLN_modulation.1" in new_key:
            # If it's Sequential, it could be layers.1
            pass
        else:
            # Try removing _0, _1 and attaching to the main body
            new_key = new_key.replace("adaLN_modulation_0", "adaLN_modulation")
            new_key = new_key.replace("adaLN_modulation_1", "adaLN_modulation")

    return new_key


def apply_lora(model, lora_path, scale=1.0):
    if not os.path.exists(lora_path):
        print(f"LoRA file not found: {lora_path}")
        return model

    print(f"   [LoRA] Loading weights from {lora_path} (User Scale: {scale})")

    tensors = {}
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = mx.array(f.get_tensor(k).float().numpy()).astype(mx.bfloat16)
    except Exception as e:
        print(f"Failed to load LoRA: {e}")
        return model

    # Extract and group LoRA pairs (A, B, Alpha)
    lora_groups = {}

    for key in tensors.keys():
        # Process Alpha keys
        if "alpha" in key:
            base = key.replace(".alpha", "")
            if base not in lora_groups: lora_groups[base] = {}
            lora_groups[base]["alpha"] = tensors[key]
            continue

        if "lora" not in key: continue

        # Distinguish A/B (Up/Down)
        type_ = None
        base = None

        if "lora_down" in key:
            base = key.split(".lora_down")[0]
            type_ = "A"  # Down
        elif "lora_up" in key:
            base = key.split(".lora_up")[0]
            type_ = "B"  # Up
        elif "lora_A" in key:
            base = key.split(".lora_A")[0]
            type_ = "A"
        elif "lora_B" in key:
            base = key.split(".lora_B")[0]
            type_ = "B"

        if base and type_:
            if base not in lora_groups: lora_groups[base] = {}
            lora_groups[base][type_] = tensors[key]

    applied_count = 0
    print(f"   [LoRA] Applying adapters with Alpha scaling...")

    for lora_key, group in lora_groups.items():
        if "A" not in group or "B" not in group: continue

        # 1. Key Conversion (AI Toolkit -> MLX)
        final_key = convert_unet_key_to_mlx(lora_key)

        # 2. Find target layer in the model
        target = get_module_by_name(model, final_key)

        if target:
            # 3. Calculate Scaling Factor (Alpha / Rank)
            # If Alpha is missing, assume 1.0 (Same as Rank)

            lora_a = group["A"]
            lora_b = group["B"]

            # Auto-detect Rank (Usually the smaller dimension of A)
            rank = min(lora_a.shape)

            if "alpha" in group:
                alpha = group["alpha"].item()
                scale_factor = alpha / rank
            else:
                scale_factor = 1.0

            # Final Scale = User Input Scale * (Alpha / Rank)
            final_scale = scale * scale_factor

            # 4. Apply Wrapper
            if isinstance(target, LoRALinearWrapper):
                base = target.base_layer
            else:
                base = target

            # Wrap the layer
            wrapped = LoRALinearWrapper(base, lora_a, lora_b, final_scale)
            set_module_by_name(model, final_key, wrapped)
            applied_count += 1
        else:
            # For Debugging: Log failed matches (Uncomment if needed)
            # if "attention" in final_key:
            #    print(f"   ⚠ Unmatched: {lora_key} -> {final_key}")
            pass

    if applied_count == 0:
        print("    Failed to apply any layers. Naming mismatch suspected.")
    else:
        print(f"   [LoRA] Applied {applied_count} layers. (Logic: Auto-Alpha & Rename)")

    return model