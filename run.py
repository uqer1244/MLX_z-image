import argparse
import os
from mlx_pipeline import ZImagePipeline


def main():
    parser = argparse.ArgumentParser(description="Run Z-Image MLX Pipeline")
    parser.add_argument("--output", type=str, default="res.png")
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    args = parser.parse_args()

    prompt_file = "prompt.txt"
    if not os.path.exists(prompt_file):
        print(f"Error: {prompt_file} not found.")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().replace("\n", " ").strip()

    pipeline = ZImagePipeline()

    image = pipeline.generate(
        prompt=prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        lora_path = args.lora_path,
        lora_scale = args.lora_scale
    )

    image.save(args.output)
    print(f"Image saved to {args.output}")


if __name__ == "__main__":
    main()
