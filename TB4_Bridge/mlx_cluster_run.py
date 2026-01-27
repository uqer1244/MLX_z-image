import sys
import os
import argparse
import mlx.core as mx
import mlx.core.distributed as dist

current_dir = os.path.dirname(os.path.abspath(__file__))  # TB4_Bridge
parent_dir = os.path.dirname(current_dir)  # local_A
sys.path.append(parent_dir)

from mlx_cluster_pipeline import ClusterZImagePipeline


def main():
    group = dist.init()
    rank = group.rank()  # 0: Host(M3), 1: Node(M4)
    world_size = group.size()

    parser = argparse.ArgumentParser(description="Cluster Z-Image Generator (TB4)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="cluster_res.png")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    args = parser.parse_args()

    prompt_file = os.path.join(parent_dir, "prompt.txt")

    if not os.path.exists(prompt_file):
        if rank == 0:
            print(f"Error: {prompt_file} not found.")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().replace("\n", " ").strip()

    if rank == 0:
        print(f"🚀 [Cluster] Running on {world_size} devices (TB4 Bridge)")
        print(f"   - Master (Rank 0): {parent_dir}")
        print(f"   - Prompt File: {prompt_file}")
        print(f"   - Prompt Content: {prompt[:60]}...")  # 내용 일부 확인

    model_dir = os.path.join(parent_dir, "Z-Image-Turbo-MLX")

    if not os.path.exists(model_dir) and rank == 0:
        print(f"Warning: Model path not found at {model_dir}")

    pipeline = ClusterZImagePipeline(rank=rank, model_path=model_dir)

    local_seed = args.seed + rank

    filename, ext = os.path.splitext(args.output)
    save_filename = f"{filename}_rank{rank}{ext}"
    save_path = os.path.join(parent_dir, save_filename)

    image = pipeline.generate(
        prompt=prompt,
        steps=args.steps,
        seed=local_seed,
        width=args.width,
        height=args.height,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale
    )

    image.save(save_path)
    print(f"[Rank {rank}] Saved: {save_filename} (Seed: {local_seed})")


if __name__ == "__main__":
    main()