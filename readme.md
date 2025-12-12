
# MLX z-image 🍎

An efficient **MLX implementation** of [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) optimized for Apple Silicon (M1/M2/M3/M4).

This repository allows you to run high-quality image generation locally on your Mac using **4-bit quantization**, significantly reducing memory usage while maintaining quality.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?style=flat-square)](https://huggingface.co/uqer1244/MLX-z-image)

## 📂 Project Structure

It is recommended to organize your folders as follows:

```text
MLX_z-image/
├── Z-Image-Turbo-MLX-4bit/        # MLX Quantized Weights 
├── Z-Image-Turbo/                 # Original PyTorch Model 
├── run.py                         # Inference Script
├── mlx_z_image.py                 # MLX converted transformer
├── convert.py                     # Script to convert PyTorch weights to MLX
└── quantize.py                    # Script to quantize FP16 model to 4-bit
````
## 📊 Performance & Gallery

### Benchmarks
Inference tests were conducted on Apple Silicon devices using **native MLX** with **4-bit quantization**.

- **Resolution**: 1024x1024
- **Steps**: 5 (Turbo)
- **Batch Size**: 1

| Device     | RAM  | Total Time | Denoise Time (MLX) | Time per Step |
|:-----------|:-----|:-----------|:-------------------|:--------------|
| **M3 Pro** | 18GB | ~ 182.17 s | 95.48 s            | 19.097 s/Step |


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

Make sure you have Python installed.

```bash
pip install -r requirements.txt
```

-----

## Quick Start

To run the model, you need two things:

1.  **Z-Image-Turbo-MLX-4bit**: The converted transformer model (4bit_quantized).
2.  **Z-Image-Turbo**: The original VAE, Text Encoder, and Scheduler.

### 1\. Download MLX Weights (Quantized)

Download the 4-bit converted weights from [uqer1244/MLX-z-image](https://huggingface.co/uqer1244/MLX-z-image).

```bash
# Install CLI if needed
pip install huggingface_hub

# Download to 'Z-Image-Turbo-MLX' folder
huggingface-cli download uqer1244/MLX-z-image --local-dir Z-Image-Turbo-MLX
```

### 2\. Download Base Model (Original)

Download the original [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) to use its VAE and Text Encoder.

```bash
# Download to 'Z-Image-Turbo' folder
huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir Z-Image-Turbo
```

### 3\. Run Inference

Run the `run.py` script. Make sure to point to both model paths.

```bash
python run.py \
  --prompt "A futuristic city with flying cars, cinematic lighting, 8k" \
  --mlx_model_path "Z-Image-Turbo-MLX-4bit" \
  --pt_model_id "Z-Image-Turbo" \
  --output "res.png" \
  --steps 5
```

### Options

| Argument | Description | Default                  |
| :--- | :--- |:-------------------------|
| `--prompt` | Text prompt for generation | *Astronaut...* |
| `--mlx_model_path` | Path to the MLX weights folder | `Z-Image-Turbo-MLX-4bit` |
| `--pt_model_id` | Path to the Base Model (or HF ID) | `Z-Image-Turbo`          |
| `--output` | Output filename | `res.png`                |
| `--steps` | Number of inference steps | `5`                      |
| `--height` | Image height | `1024`                   |
| `--width` | Image width | `1024`                   |

-----

## Todo & Roadmap

We are actively working on making this implementation pure MLX and bug-free.

  - [ ] **Fix Artifacts**: Investigate and resolve visual artifacts (tiling/color issues) currently visible in some generations.
  - [ ] **Full MLX Pipeline**: Port the remaining PyTorch components (VAE, Text Encoder, Tokenizer, Scheduler) to native MLX to remove the `torch` and `diffusers` dependencies completely.
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

