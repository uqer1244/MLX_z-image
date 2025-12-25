import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import json
import os
import time
import gc
from PIL import Image
from transformers import AutoTokenizer
from diffusers import AutoencoderKL
from huggingface_hub import snapshot_download

from mlx_z_image import ZImageTransformerMLX
from mlx_text_encoder import TextEncoderMLX
from lora_utils import apply_lora


def create_coordinate_grid(size, start):
    d0, d1, d2 = size
    s0, s1, s2 = start
    i = mx.arange(s0, s0 + d0)
    j = mx.arange(s1, s1 + d1)
    k = mx.arange(s2, s2 + d2)
    grid_i = mx.broadcast_to(i[:, None, None], (d0, d1, d2))
    grid_j = mx.broadcast_to(j[None, :, None], (d0, d1, d2))
    grid_k = mx.broadcast_to(k[None, None, :], (d0, d1, d2))
    return mx.stack([grid_i, grid_j, grid_k], axis=-1).reshape(-1, 3)


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def load_sharded_weights(model_path):
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    weights = {}
    if os.path.exists(index_path):
        print(f"   [Loader] Detected sharded weights index: {index_path}")
        with open(index_path, "r") as f:
            index_data = json.load(f)
        shard_files = sorted(list(set(index_data["weight_map"].values())))
        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            print(f"   [Loader] Loading shard: {shard_file}...")
            weights.update(mx.load(shard_path))
            if hasattr(mx, "clear_cache"): mx.clear_cache()
    else:
        single_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(single_path):
            weights = mx.load(single_path)
        else:
            import glob
            files = glob.glob(os.path.join(model_path, "*.safetensors"))
            for f in files:
                weights.update(mx.load(f))
    return weights


# =========================================
# MLX Scheduler
# =========================================

class MLXFlowMatchEulerScheduler:
    def __init__(self, shift: float = 3.0, use_dynamic_shifting: bool = True):
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int, mu: float = None):
        ts = np.linspace(1.0, 0.0, num_inference_steps + 1)

        # Time Shifting
        if self.use_dynamic_shifting and mu is not None:
            ts = self._time_shift(mu, ts)

        self.timesteps = mx.array(ts).astype(mx.float32)

    def _time_shift(self, mu: float, t: np.ndarray):
        mask = t > 0
        res = np.zeros_like(t)

        res[mask] = np.exp(mu) / (np.exp(mu) + (1 / t[mask] - 1))

        return res

    def step(self, model_output, timestep_idx, sample):
        t_curr = self.timesteps[timestep_idx]
        t_prev = self.timesteps[timestep_idx + 1]

        dt = t_prev - t_curr

        prev_sample = sample + dt * model_output

        return prev_sample


# =========================================
# Main Pipeline Class
# =========================================
class ZImagePipeline:
    def __init__(self,
                 model_path="Z-Image-Turbo-MLX",
                 text_encoder_path="Z-Image-Turbo-MLX/text_encoder",
                 repo_id="uqer1244/MLX-z-image"):
        self.model_path = model_path
        self.text_encoder_path = text_encoder_path
        self.repo_id = repo_id

        if not os.path.exists(self.model_path):
            print(f"Downloading base model from {self.repo_id}...")
            snapshot_download(repo_id=self.repo_id, local_dir=self.model_path)

    def generate(self, prompt, width=1024, height=1024, steps=9, seed=42, lora_path=None, lora_scale=1.0):
        mx.set_cache_limit(0)

        print(f"Pipeline Started | Size: {width}x{height} | Steps: {steps}")
        global_start = time.time()

        # ----------------------------------------------------------------
        # [Phase 1] Text Encoding
        # ----------------------------------------------------------------
        t_start = time.time()
        print(f"[Phase 1] Text Encoding...", end=" ", flush=True)

        tokenizer_path = os.path.join(self.model_path, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        te_config_path = os.path.join(self.text_encoder_path, "config.json")
        with open(te_config_path, "r") as f:
            te_config = json.load(f)

        text_encoder = TextEncoderMLX(te_config)

        quantized_weights_path = os.path.join(self.text_encoder_path, "model.safetensors")

        if os.path.exists(quantized_weights_path):
            print(f"(Fast Load: Pre-Quantized)...", end=" ", flush=True)

            nn.quantize(text_encoder, bits=4, group_size=32)
            text_encoder.load_weights(quantized_weights_path)

        else: #Legacy support
            print(f"(Slow Load: On-the-fly Quantization)...", end=" ", flush=True)
            te_weights = load_sharded_weights(self.text_encoder_path)
            text_encoder.load_weights(list(te_weights.items()))
            del te_weights
            nn.quantize(text_encoder, bits=4, group_size=32)

        mx.eval(text_encoder)

        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            prompt_fmt = prompt

        inputs = tokenizer(prompt_fmt, padding="max_length", max_length=512, truncation=True, return_tensors="np")

        prompt_embeds = text_encoder(mx.array(inputs["input_ids"]))
        mx.eval(prompt_embeds)

        cap_feats_np = np.array(prompt_embeds)
        pad = (-cap_feats_np.shape[1]) % 32
        if pad > 0: cap_feats_np = np.concatenate([cap_feats_np, np.repeat(cap_feats_np[:, -1:, :], pad, axis=1)],
                                                  axis=1)
        cap_feats_mx = mx.array(cap_feats_np).astype(mx.bfloat16)

        del text_encoder, tokenizer
        mx.clear_cache()
        gc.collect()
        print(f"Done ({time.time() - t_start:.2f}s)")

        # ----------------------------------------------------------------
        # [Phase 2] Transformer Loading (4-bit)
        # ----------------------------------------------------------------
        t_start = time.time()
        trans_path = os.path.join(self.model_path, "transformer")
        print(f"[Phase 2] Loading Transformer (4-bit GS32)...", end=" ", flush=True)

        with open(os.path.join(trans_path, "config.json"), "r") as f:
            config = json.load(f)

        model = ZImageTransformerMLX(config)

        nn.quantize(model, bits=4, group_size=32)

        if os.path.exists(os.path.join(trans_path, "model.safetensors.index.json")):
            weights = load_sharded_weights(trans_path)
            model.load_weights(list(weights.items()))
            del weights
        else:
            model.load_weights(os.path.join(trans_path, "model.safetensors"))

        # ----------------------------------------------------------------
        # [Phase 2.5] Apply lora
        # ----------------------------------------------------------------
        if lora_path:
            print(f"[Phase 2.5] Applying LoRA...", end=" ", flush=True)
            model = apply_lora(model, lora_path, scale=lora_scale)
            print("Done")

        model.eval()
        print(f"Done ({time.time() - t_start:.2f}s)")

        # ----------------------------------------------------------------
        # [Phase 3] Denoising (Fully MLX & Compiled)
        # ----------------------------------------------------------------
        print(f"[Phase 3] Denoising (MLX Scheduler & Compiled)...", end="\n")

        scheduler = MLXFlowMatchEulerScheduler(shift=3.0, use_dynamic_shifting=True)

        latents_np = np.random.randn(1, 16, height // 8, width // 8).astype(np.float32)
        if seed is not None:
            np.random.seed(seed)
            latents_np = np.random.randn(1, 16, height // 8, width // 8).astype(np.float32)

        latents = mx.array(latents_np).astype(mx.bfloat16)

        mu = calculate_shift((latents.shape[2] // 2) * (latents.shape[3] // 2))
        scheduler.set_timesteps(steps, mu=mu)

        total_len = cap_feats_mx.shape[1]
        H_tok, W_tok = (height // 8) // 2, (width // 8) // 2
        img_pos = mx.array(
            create_coordinate_grid((1, H_tok, W_tok), (total_len + 1, 0, 0)).reshape(-1, 3)[None]).astype(mx.bfloat16)
        cap_pos = mx.array(create_coordinate_grid((total_len, 1, 1), (1, 0, 0)).reshape(-1, 3)[None]).astype(
            mx.bfloat16)

        unified_pos_all = mx.concatenate([img_pos, cap_pos], axis=1)
        cos_cached, sin_cached = model.prepare_rope(unified_pos_all)
        cos_cached = cos_cached.astype(mx.bfloat16)
        sin_cached = sin_cached.astype(mx.bfloat16)

        @mx.compile
        def step_fn(x, t, feats, i_pos, c_pos, cos, sin):
            B, C, H, W = x.shape
            x_reshaped = x.reshape(C, 1, 1, H_tok, 2, W_tok, 2).transpose(1, 2, 3, 5, 4, 6, 0).reshape(1, -1, C * 4)
            out = model(x_reshaped, t, feats, i_pos, c_pos, cos, sin, cap_mask=None)
            noise_pred = -out.reshape(1, 1, H_tok, W_tok, 2, 2, C).transpose(6, 0, 1, 2, 4, 3, 5).reshape(1, C, H, W)
            return noise_pred

        denoise_start = time.time()

        for i in range(steps):
            step_start = time.time()

            t_curr = scheduler.timesteps[i]
            t_input = (1.0 - t_curr)[None].astype(mx.bfloat16)

            # Compiled Function Call
            noise_pred = step_fn(latents, t_input, cap_feats_mx, img_pos, cap_pos, cos_cached, sin_cached)
            # Scheduler Update
            latents = scheduler.step(noise_pred, i, latents)
            # Sync
            mx.eval(latents)

            print(f"   Step {i + 1}/{steps}: {time.time() - step_start:.2f}s")

        print(f"   Avg Speed: {(time.time() - denoise_start) / steps:.2f} s/it")

        # ----------------------------------------------------------------
        # [Phase 4] Decoding (VAE with Tiling & Memory Cleanup)
        # ----------------------------------------------------------------
        print("[Phase 4] Decoding...", end=" ", flush=True)
        t_dec = time.time()

        del model, scheduler, step_fn, cos_cached, sin_cached
        for _ in range(3):
            mx.clear_cache()
            gc.collect()

        vae_path = os.path.join(self.model_path, "vae")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        vae = AutoencoderKL.from_pretrained(vae_path).to(device)

        vae.enable_tiling()

        latents_pt = torch.from_numpy(np.array(latents.astype(mx.float32))).to(device)
        latents_pt = (latents_pt / vae.config.scaling_factor) + getattr(vae.config, "shift_factor", 0.0)

        with torch.no_grad():
            image = vae.decode(latents_pt).sample

        image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        pil_image = Image.fromarray((image[0] * 255).round().astype("uint8"))

        print(f"Done ({time.time() - t_dec:.2f}s)")
        print(f"✨ Pipeline Finished in {time.time() - global_start:.2f}s")

        return pil_image