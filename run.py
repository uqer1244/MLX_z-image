import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import json
import os
import gc
import argparse
import time
from PIL import Image
from transformers import AutoTokenizer
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download

from mlx_z_image import ZImageTransformerMLX
from mlx_text_encoder import TextEncoderMLX


def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


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


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo MLX Local Runner")


    parser.add_argument("--model_path", type=str, default="Z-Image-Turbo-MLX",
                        help="Local folder containing the full MLX package")

    parser.add_argument("--repo_id", type=str, default="uqer1244/MLX-z-image",
                        help="HF Repo ID to download from if local path is empty (e.g., User/Model)")

    parser.add_argument("--prompt", type=str,
                        default="8k, super detailed semi-realistic anime style female warrior, detailed armor, backlighting, dynamic pose, illustration, highly detailed, dramatic lighting",
                        help="Prompt to generate")

    parser.add_argument("--output", type=str, default="res.png", help="Output filename")
    parser.add_argument("--steps", type=int, default=5, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"PyTorch Device: {device}")

    # ==========================================
    # 0. Model Path Check & Download
    # ==========================================

    if not os.path.exists(args.model_path):
        if args.repo_id:
            print(f"Local path '{args.model_path}' not found.")
            print(f"Downloading from Hugging Face Hub: '{args.repo_id}'...")
            try:
                snapshot_download(repo_id=args.repo_id, local_dir=args.model_path)
                print(f"Download complete.")
            except Exception as e:
                print(f"Download failed: {e}")
                return
        else:
            print(f"Error: Local path '{args.model_path}' not found and no --repo_id provided.")
            return
    else:
        print(f"Found local model at: '{args.model_path}'")

    global_start_time = time.time()
    print(f"🚀 Prompt: '{args.prompt}'")

    # ==========================================
    # 1. Text Encoding (MLX 4-bit)
    # ==========================================
    print("\n[Phase 1] Processing Text (MLX 4-bit)...")
    phase1_start = time.time()

    te_path = os.path.join(args.model_path, "text_encoder")
    tok_path = os.path.join(args.model_path, "tokenizer")

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    except:
        print("⚠️ Warning: Could not load local tokenizer. Trying to download config...")
        # fallback if critical files missing
        tokenizer = AutoTokenizer.from_pretrained("Tongyi-MAI/Z-Image-Turbo", subfolder="tokenizer",
                                                  trust_remote_code=True)

    # Text Encoder
    with open(os.path.join(te_path, "config.json"), "r") as f:
        te_config = json.load(f)

    text_encoder = TextEncoderMLX(te_config)
    nn.quantize(text_encoder, bits=4, group_size=32)
    text_encoder.load_weights(os.path.join(te_path, "model.safetensors"))
    text_encoder.eval()

    # Chat Template
    messages = [{"role": "user", "content": args.prompt}]
    try:
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        prompt_formatted = args.prompt

    inputs = tokenizer(prompt_formatted, padding="max_length", max_length=512, truncation=True, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])

    # Inference
    prompt_embeds = text_encoder(input_ids)
    mx.eval(prompt_embeds)
    cap_feats_np = np.array(prompt_embeds)

    del text_encoder, tokenizer
    cleanup()

    # Padding
    SEQ_MULTI_OF = 32
    cap_len = cap_feats_np.shape[1]
    cap_padding_len = (-cap_len) % SEQ_MULTI_OF
    total_cap_len = cap_len + cap_padding_len
    cap_padded_np = np.concatenate([cap_feats_np, np.repeat(cap_feats_np[:, -1:, :], cap_padding_len, axis=1)], axis=1)
    cap_feats_mx = mx.array(cap_padded_np, dtype=mx.float32)
    cap_mask_np = np.zeros((1, total_cap_len), dtype=bool)
    cap_mask_np[:, cap_len:] = True
    cap_mask_mx = mx.array(cap_mask_np)

    phase1_duration = time.time() - phase1_start
    print(f"⏱️ [Phase 1] Done in {phase1_duration:.2f}s")

    # ==========================================
    # 2. Transformer Loading (MLX 4-bit)
    # ==========================================
    print("\n[Phase 2] Loading Transformer (MLX 4-bit)...")
    phase2_start = time.time()

    trans_path = os.path.join(args.model_path, "transformer")
    with open(os.path.join(trans_path, "config.json"), "r") as f:
        config = json.load(f)

    model = ZImageTransformerMLX(config)
    nn.quantize(model, bits=4, group_size=32)
    model.load_weights(os.path.join(trans_path, "model.safetensors"))
    model.eval()
    mx.eval(model.parameters())

    phase2_duration = time.time() - phase2_start
    print(f"⏱️ [Phase 2] Done in {phase2_duration:.2f}s")

    # ==========================================
    # 3. Denoising Loop
    # ==========================================
    print(f"\n[Phase 3] Denoising ({args.steps} Steps)...")
    inference_start = time.time()

    sched_path = os.path.join(args.model_path, "scheduler")
    try:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sched_path)
    except:
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True)

    latents_shape = (1, 16, args.height // 8, args.width // 8)
    generator = torch.Generator().manual_seed(args.seed)
    latents_pt = torch.randn(latents_shape, generator=generator, dtype=torch.float32)

    image_seq_len = (latents_pt.shape[2] // 2) * (latents_pt.shape[3] // 2)
    mu = calculate_shift(image_seq_len)
    scheduler.set_timesteps(args.steps, mu=mu)

    p = 2
    H_tok, W_tok = (args.height // 8) // p, (args.width // 8) // p
    img_pos_mx = mx.array(create_coordinate_grid((1, H_tok, W_tok), (total_cap_len + 1, 0, 0)).reshape(-1, 3)[None])
    cap_pos_mx = mx.array(create_coordinate_grid((total_cap_len, 1, 1), (1, 0, 0)).reshape(-1, 3)[None])

    for i, t in enumerate(scheduler.timesteps):
        t_val = t.item()
        t_norm = (1000.0 - t_val) / 1000.0
        t_mx = mx.array([t_norm], dtype=mx.float32)

        x_in_mx = mx.array(latents_pt.numpy()).astype(mx.float32)
        B, C, H, W = x_in_mx.shape
        x_reshaped = x_in_mx.reshape(C, 1, 1, H_tok, p, W_tok, p)
        x_transposed = x_reshaped.transpose(1, 2, 3, 5, 4, 6, 0)
        x_model_in = x_transposed.reshape(1, -1, C * p * p)

        noise_pred_mx = model(x_model_in, t_mx, cap_feats_mx, img_pos_mx, cap_pos_mx, cap_mask=cap_mask_mx)
        mx.eval(noise_pred_mx)

        out_reshaped = noise_pred_mx.reshape(1, 1, H_tok, W_tok, p, p, C)
        noise_pred_tensor_mx = out_reshaped.transpose(6, 0, 1, 2, 4, 3, 5).reshape(1, C, H, W)
        noise_pred_tensor_mx = -noise_pred_tensor_mx

        noise_pred_pt = torch.from_numpy(np.array(noise_pred_tensor_mx)).float()
        latents_pt = scheduler.step(noise_pred_pt, t, latents_pt, return_dict=False)[0]
        print(f"  Step {i + 1}/{args.steps} | t={t_val:.1f}")

    del model
    cleanup()
    inference_duration = time.time() - inference_start
    print(f"⏱️ [Phase 3] Done in {inference_duration:.2f}s")

    # ==========================================
    # 4. Decoding (PyTorch VAE)
    # ==========================================
    print("\n[Phase 4] Decoding (PyTorch VAE)...")
    phase4_start = time.time()

    vae_path = os.path.join(args.model_path, "vae")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.enable_tiling()

    latents_pt = latents_pt.to(device)
    scaling_factor = vae.config.scaling_factor
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    latents_pt = (latents_pt / scaling_factor) + shift_factor

    with torch.no_grad():
        image = vae.decode(latents_pt).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")

    pil_image = Image.fromarray(image[0])
    pil_image.save(args.output)

    phase4_duration = time.time() - phase4_start
    print(f"⏱️ [Phase 4] Done in {phase4_duration:.2f}s")
    print(f"🎉 Success! Image Saved to {args.output}")


if __name__ == "__main__":
    main()