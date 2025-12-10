import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import json
import os
import gc
import argparse
from PIL import Image
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, DiffusionPipeline

from mlx_z_image import ZImageTransformerMLX


def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


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


def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def main():
    parser = argparse.ArgumentParser(description="Run Z-Image-Turbo MLX Inference")
    parser.add_argument("--prompt", type=str, default="A lone astronaut discovering an ancient alien artifact",
                        help="Prompt to generate")
    parser.add_argument("--mlx_model_path", type=str, default="Z-Image-Turbo-MLX-4bit", help="Path to MLX model folder")
    parser.add_argument("--pt_model_id", type=str, default="Z-Image-Turbo",
                        help="Original HF Model ID (for VAE/Tokenizer)")
    parser.add_argument("--output", type=str, default="res.png", help="Output filename")
    parser.add_argument("--steps", type=int, default=5, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⚡️ Using PyTorch Device for VAE/TextEnc: {device}")
    print(f"🚀 Generating: '{args.prompt}' (Steps: {args.steps})")

    # ==========================================
    # 1. Text Encoding
    # ==========================================
    print("\n[Phase 1] Extracting Text Encoder...")
    # 로컬 경로가 아닌 HF Hub ID를 사용하도록 변경
    temp_pipe = DiffusionPipeline.from_pretrained(
        args.pt_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    scheduler = temp_pipe.scheduler
    new_config = dict(scheduler.config)
    new_config['use_beta_sigmas'] = True
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(new_config)

    tokenizer = temp_pipe.tokenizer
    text_encoder = temp_pipe.text_encoder.to(device)

    text_inputs = tokenizer(
        args.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        prompt_embeds = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]

    cap_feats_np = prompt_embeds.cpu().float().numpy()

    del temp_pipe, text_encoder, tokenizer
    cleanup()

    # 텍스트 전처리
    SEQ_MULTI_OF = 32
    cap_len = cap_feats_np.shape[1]
    cap_padding_len = (-cap_len) % SEQ_MULTI_OF
    total_cap_len = cap_len + cap_padding_len

    cap_padded_np = np.concatenate([cap_feats_np, np.repeat(cap_feats_np[:, -1:, :], cap_padding_len, axis=1)], axis=1)
    cap_feats_mx = mx.array(cap_padded_np, dtype=mx.float32)

    cap_mask_np = np.zeros((1, total_cap_len), dtype=bool)
    cap_mask_np[:, cap_len:] = True
    cap_mask_mx = mx.array(cap_mask_np)

    # ==========================================
    # 2. MLX Model Load
    # ==========================================
    print("\n[Phase 2] Loading MLX Transformer...")
    if not os.path.exists(args.mlx_model_path):
        print(f"❌ MLX Model path not found: {args.mlx_model_path}")
        print("💡 Please run 'convert.py' or 'quantize.py' first.")
        return

    with open(os.path.join(args.mlx_model_path, "config.json"), "r") as f:
        config = json.load(f)

    model = ZImageTransformerMLX(config)

    # 4비트 양자화 모델인지 확인하여 로드 방식 결정 (간단한 체크)
    # 일반적으로 safetensors 로드 시 구조가 맞으면 알아서 들어갑니다.
    # 만약 4비트 로드시에는 nn.quantize를 먼저 호출해야 구조가 맞습니다.
    # 여기서는 파일 경로에 '4bit'가 있거나, config를 체크하는 등의 로직이 필요하지만
    # 사용자 편의를 위해 일단 무조건 quantize 시도 후 로드 (실패 시 FP16으로 가정 가능하나, 여기선 4비트가 기본이라 가정)
    try:
        nn.quantize(model, bits=4, group_size=32)  # 일단 4비트 구조로 변경
        model.load_weights(os.path.join(args.mlx_model_path, "model.safetensors"))
    except Exception:
        print("⚠️ 4-bit load failed, trying FP16 structure...")
        model = ZImageTransformerMLX(config)  # 다시 초기화
        model.load_weights(os.path.join(args.mlx_model_path, "model.safetensors"))

    model.eval()

    # Latent Init
    latents_shape = (1, 16, args.height // 8, args.width // 8)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents_pt = torch.randn(latents_shape, generator=generator, dtype=torch.float32)

    image_seq_len = (latents_pt.shape[2] // 2) * (latents_pt.shape[3] // 2)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )

    try:
        scheduler.set_timesteps(args.steps, mu=mu)
    except TypeError:
        scheduler.set_timesteps(args.steps)

    # Grid Setup
    p = 2
    H_tok, W_tok = (args.height // 8) // p, (args.width // 8) // p
    img_pos_mx = mx.array(create_coordinate_grid((1, H_tok, W_tok), (total_cap_len + 1, 0, 0)).reshape(-1, 3)[None])
    cap_pos_mx = mx.array(create_coordinate_grid((total_cap_len, 1, 1), (1, 0, 0)).reshape(-1, 3)[None])

    # ==========================================
    # 4. Denoising Loop
    # ==========================================
    print("Starting Denoising Loop...")

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

        out_reshaped = noise_pred_mx.reshape(1, 1, H_tok, W_tok, p, p, C)
        noise_pred_tensor_mx = out_reshaped.transpose(6, 0, 1, 2, 4, 3, 5).reshape(1, C, H, W)
        noise_pred_tensor_mx = -noise_pred_tensor_mx

        noise_pred_pt = torch.from_numpy(np.array(noise_pred_tensor_mx)).float()
        step_output = scheduler.step(noise_pred_pt, t, latents_pt, return_dict=False)[0]
        latents_pt = step_output

        print(f"Step {i + 1}/{args.steps} | t={t_val:.1f}")
        cleanup()

    del model
    cleanup()

    # ==========================================
    # 5. VAE Decode
    # ==========================================
    print("\n[Phase 3] Decoding Image...")
    # HF Hub에서 VAE 로드
    vae = AutoencoderKL.from_pretrained(args.pt_model_id, subfolder="vae").to(device)
    vae.enable_tiling()
    vae.enable_slicing()

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
    print(f"🎉 Success! Image Saved to {args.output}")


if __name__ == "__main__":
    main()