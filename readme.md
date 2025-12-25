
# MLX z-image 🍎

An efficient **MLX implementation** of [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) optimized for Apple Silicon (M1/M2/M3/M4).

This repository allows you to run high-quality image generation locally on your Mac using **4-bit quantization**, significantly reducing memory usage while maintaining quality.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?style=flat-square)](https://huggingface.co/uqer1244/MLX-z-image)

## 📂 Project Structure

It is recommended to organize your folders as follows:

```text
MLX_z-image/
├── converting/                    # Scripts to convert PyTorch to MLX (just non functional files)
├── Z-Image-Turbo-MLX/             # MLX Weights (auto download on first run)
├── mlx_text_encoder.py            # MLX converted Text Encoder
├── mlx_z_image.py                 # MLX converted transformer
├── check_lora.py                  # Checking lora is suitable for mlx-z-image
├── lora_utils.py                  # Apply lora
├── run.py                         # Run Script
├── prompt.txt                     # prompt
└── mlx_pipeline.py                # mlx Pipeline
````
## 📊 Performance & Gallery

### Benchmarks
Inference tests were conducted on Apple Silicon devices using **native MLX** with **4-bit quantization**.

- **Resolution**: 1024x1024
- **Steps**: 9 (Turbo)

| Device     | RAM  | Total Time | Denoise Time | Time per Step |
|:-----------|:-----|:-----------|:-------------|:--------------|
| **M3 Pro** | 18GB | ~ 150 s    | 140 s        | 15 s/Step     |
| **M4**     | 16GB | ~ 240 s    | 230 s        | 25 s/Step     |


### Gallary
Uncurated samples generated directly on a Mac using the 4-bit quantized model.

*"anime digital painting  She sat poised on a ledge of polished peach quartz, the very image of a classical statue brought to wild, impish life within the sunlit cave dwellings. Her Korean features were framed by a stunning ginger hime cut, its straight, . A wild assembly of leather straps and sheer, iridescent fabric served as her lingerie, barely covering her slim figure while perfectly accentuating her narrow waist, tight ass, and breasts. The natural sunlight filtering through the cave's opening bathed her in a warm, rosy glow, making her pale skin seem to glow from within. One hand rested flat on the quartz beside her, supporting her lean, while the other was raised to her mouth, a single finger resting thoughtfully on her lower lip as she watched the dust motes dance in the light."*

|               **MLX**               |
|:-----------------------------------:|
| <img src="img/res.png" width="512"> | 


## Installation

### 1\. Clone the repository

```bash
git clone https://github.com/uqer1244/MLX_z-image.git
cd MLX_z-image
```

### 2\. Install dependencies

Ensure you have Python installed (Python 3.10+ recommended).

```bash
pip install -r requirements.txt
```

*(Note: `huggingface_hub` is required for downloading models)*

-----

## Quick Start

We have packaged everything (Transformer, Text Encoder, VAE, Tokenizer, Scheduler) into a single repository for easy usage.

### Option 1: Automatic Download & Run (Recommended)

Simply run the script. If the model is not found locally, it will automatically download the full 4-bit quantized package from Hugging Face.

```bash
python run.py 
```


> **Note:** The prompt is always loaded from `prompt.txt` to handle long/complex prompts easily.


### Options

| Argument       | Description     | Default  |
|:---------------|:----------------|:---------|
| `--width`      | Image Width     | `1024`   |
| `--height`     | Image Height    | `1024`   |
| `--steps`      | Inference Steps | `9`      |
| `--seed`       | Random Seed     | `42`     |
| `--output`     | Output filename | `res.png` |
| `--lora`       | Lora path       | `None`   |
| `--lora_scale` | Lora scale      | `1.0`    |


```bash
python run.py \
  --width 1024 \ 
  --height 1024 \
  --steps 9 \
  --seed 42 \
  --output "res.png" \
  --steps 5 \ 
  --lora "~~.safetensor" \
  --lora_scale 1.0 \
  ```

> **Note:** Width and Height resolutions are always able to devide by 8



### [ComfyUI Custom node setup](custom_nodes/readme.md)


-----

## Todo & Roadmap

We are actively working on making this implementation pure MLX and bug-free.

  - [x] **Fix Artifacts**: Investigate and resolve visual artifacts (tiling/color issues) currently visible in some generations.
  - [ ] **Full MLX Pipeline**: Port the remaining PyTorch components (VAE, Text Encoder, Tokenizer, Scheduler) to native MLX to remove the `torch` and `diffusers` dependencies completely.
    - [x] Text Encoder - 4bit
    - [ ] Tokenizer - tokenizer is faster enough
    - [x] Scheduler
    - [x] Transformer - 4bit
    - [ ] VAE - I tried MLX converting for VAE, but pytorch version is more stable
  - [x] **LoRA Support**: Add support for loading and applying LoRA adapters for style customization.
  - [x] **ComfyUI Node**: Add custom node for ComfyUI GUI.
  - [ ] **IP over ThunderBolt (or RDMA on TB5) support** : Add support for multiple mac cluster.
  - [ ] **Z-Image-Edit, Base model support** : now on turbo only.

-----


## Acknowledgements

  - Original Model: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
  - MLX Framework: [Apple Machine Learning Research](https://github.com/ml-explore/mlx)

## License

This project is a modification of [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image) and is licensed under the **Apache License 2.0**.

  - **Original Code**: Copyright (c) Tongyi-MAI
  - **Modifications**: Ported to Apple MLX by uqer1244

<!-- end list -->

