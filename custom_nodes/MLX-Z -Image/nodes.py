import torch
import numpy as np
import os
import sys
import traceback

current_node_path = os.path.dirname(os.path.abspath(__file__))
if current_node_path not in sys.path:
    sys.path.append(current_node_path)

LORA_DIR = os.path.join(current_node_path, "LoRA")

ZImagePipeline = None
try:
    from mlx_pipeline import ZImagePipeline

    print("MLX Pipeline loaded successfully in nodes.py")
except Exception as e:
    print("\n" + "=" * 50)
    print("[CRITICAL ERROR] MLX Pipeline Import Failed")
    traceback.print_exc()
    print("=" * 50 + "\n")


class MLX_Z_Image_Gen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        if not os.path.exists(LORA_DIR):
            os.makedirs(LORA_DIR, exist_ok=True)

        lora_files = ["None"]
        if os.path.exists(LORA_DIR):
            lora_files += [f for f in os.listdir(LORA_DIR) if f.endswith(".safetensors")]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic shot of a futuristic city"}),
                "width": ("INT", {"default": 720, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 720, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "lora_name": (lora_files,),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "MLX_Z_Image"

    def generate_image(self, prompt, width, height, steps, seed, lora_name="None", lora_strength=1.0):
        if ZImagePipeline is None:
            raise ImportError("MLX Pipeline load failed.")

        model_path = os.path.join(current_node_path, "Z-Image-Turbo-MLX")
        text_encoder_path = os.path.join(model_path, "text_encoder")

        if not os.path.exists(model_path):
            print(f"Warning: Model folder not found at {model_path}")

        lora_path = None
        if lora_name != "None":
            lora_path = os.path.join(LORA_DIR, lora_name)

            if not os.path.exists(lora_path):
                print(f"Warning: LoRA not found at {lora_path}")
                lora_path = None

        print(f"Generating: {prompt} | Size: {width}x{height} | Steps: {steps}")
        if lora_path:
            print(f"Loading LoRA from: {lora_path}")

        pipeline = ZImagePipeline(
            model_path=model_path,
            text_encoder_path=text_encoder_path
        )

        pil_image = pipeline.generate(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            lora_path=lora_path,
            lora_scale=lora_strength
        )

        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
    "MLX_Z_Image_Gen": MLX_Z_Image_Gen
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLX_Z_Image_Gen": "MLX Z-Image Turbo (Native)"
}