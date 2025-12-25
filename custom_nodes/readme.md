## ComfyUI Custom Node Setup


-----
### Node + Performance

<img src="img/a1.png" width="1024">

-----

## Project Structure


```text
custom_nodes/
    MLX_z-image/
    ├── LoRA/                          # Add LoRa files to this dir
    ├── Z-Image-Turbo-MLX/             # MLX Weights 
    ├── mlx_text_encoder.py            # MLX converted Text Encoder
    ├── mlx_z_image.py                 # MLX converted transformer
    ├── mlx_pipeline.py                # mlx Pipeline
    ├── __init__.py                    
    ├── nodes.py                       # Add node
    └── lora_utils.py                  # LoRA Adaption
````

-----

<img src="img/a2.png" width="512">

-----




# If you use ComfyUI desktop app

1. Make Directory on your ComfyUI/custom_nodes 

<img src="img/t1.png" width="256">

2. In ComfyUI app, go to consol -> terminal

<img src="img/t.png" width="256">


then, 
```bash
cd custom_nodes/MLX_Z_Image/
```

```bash
pip install -r requirements.txt
```
<img src="img/t2.png" width="256">

Restart ComfyUI app

then, custom node appear

<img src="img/t3.png" width="256">


## TODO

  - [ ] **Multiple LoRA**: Able to apply mutiple LoRA to node.
