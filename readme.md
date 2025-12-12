
# MLX z-image 🍎

An efficient **MLX implementation** of [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) optimized for Apple Silicon (M1/M2/M3/M4).

This repository allows you to run high-quality image generation locally on your Mac using **4-bit quantization**, significantly reducing memory usage while maintaining quality.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?style=flat-square)](https://huggingface.co/uqer1244/MLX-z-image)

## 📂 Project Structure

It is recommended to organize your folders as follows:

```text
MLX_z-image/
├── converting/                    # Scripts to convert PyTorch to MLX
├── Z-Image-Turbo-MLX-4bit/        # MLX Quantized Weights 
├── mlx_text_encoder.py            # MLX converted Text Encoder
├── run.py                         # Inference Script
├── mlx_z_image.py                 # MLX converted transformer
└── quantize.py                    # Script to quantize FP16 model to 4-bit
````
## 📊 Performance & Gallery

### Benchmarks
Inference tests were conducted on Apple Silicon devices using **native MLX** with **4-bit quantization**.

- **Resolution**: 1024x1024
- **Steps**: 5 (Turbo)

| Device     | RAM  | Total Time | Denoise Time (MLX) | Time per Step |
|:-----------|:-----|:-----------|:-------------------|:--------------|
| **M3 Pro** | 18GB | ~ 125 s    | 95.5 s             | 19.1 s/Step   |
| **M4 Pro** | 24GB | ~ 90 s     | 75.1 s             | 15 s/Step     |


### Gallary
Uncurated samples generated directly on a Mac using the 4-bit quantized model.

*"semi-realistic anime style female warrior, detailed armor, backlighting, dynamic pose, illustration, highly detailed, dramatic lighting"*

|                      **MLX**                      |              **Pytorch**               |
|:-------------------------------------------------:|:--------------------------------------:|
| <img src="img/res_4bit.png" width="400">          | <img src="img/res_OG.png" width="400"> |



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
python run.py \
  --repo_id "uqer1244/MLX-z-image" \
  --prompt "semi-realistic anime style female warrior, detailed armor, backlighting"
```

### Option 2: Manual Download

If you prefer to download the model manually (e.g., for offline usage), use the `huggingface-cli`.

**1. Download the Full Package**

```bash
# Download all components to 'checkpoints' folder
huggingface-cli download uqer1244/MLX-z-image --local-dir Z-Image-Turbo-MLX
```

**2. Run Inference**

Point the `--model_path` to your downloaded folder.

```bash
python run.py \
  --model_path "Z-Image-Turbo-MLX" \
  --prompt "semi-realistic anime style female warrior, detailed armor, backlighting" \
  --output "result.png" \
  --steps 5
```

### Options

| Argument | Description | Default                  |
| :--- | :--- |:-------------------------|
| `--prompt` | Text prompt for image generation | *...anime style...*      |
| `--model_path` | Path to local model folder | `Z-Image-Turbo-MLX-4bit` |
| `--repo_id` | Hugging Face Repo ID (for auto-download) | `None`                   |
| `--output` | Output image filename | `result.png`             |
| `--steps` | Number of inference steps | `5`                      |
| `--height` / `--width` | Image resolution (must be divisible by 8) | `1024`                   |

-----

## Todo & Roadmap

We are actively working on making this implementation pure MLX and bug-free.

  - [x] **Fix Artifacts**: Investigate and resolve visual artifacts (tiling/color issues) currently visible in some generations.
  - [ ] **Full MLX Pipeline**: Port the remaining PyTorch components (VAE, Text Encoder, Tokenizer, Scheduler) to native MLX to remove the `torch` and `diffusers` dependencies completely.
    - [x] Text Encoder
    - [ ] Tokenizer
    - [ ] Scheduler
    - [x] Transformer
    - [ ] VAE
  - [ ] **LoRA Support**: Add support for loading and applying LoRA adapters for style customization.

-----

## Advanced: Manual Conversion

If you want to convert the original PyTorch weights yourself (instead of downloading the pre-converted ones), follow these steps.

**1. Convert PyTorch to MLX (FP16)**

```bash
python convert.py \
  --model_id "Z-Image-Turbo" \
  --dest_path "Z-Image-Turbo-MLX"
```

**2. Quantize to 4-bit**

```bash
python quantize.py \
  --model_path "Z-Image-Turbo-MLX" \
  --dest_path "Z-Image-Turbo-MLX-4bit" \
  --group_size 32
```

## Acknowledgements

  - Original Model: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
  - MLX Framework: [Apple Machine Learning Research](https://github.com/ml-explore/mlx)

## License

This project is a modification of [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image) and is licensed under the **Apache License 2.0**.

  - **Original Code**: Copyright (c) Tongyi-MAI
  - **Modifications**: Ported to Apple MLX by uqer1244

<!-- end list -->

