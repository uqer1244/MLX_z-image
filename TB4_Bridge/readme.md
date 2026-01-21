## (Working on) TB4 Mac Cluster 

I am currently working on building a cluster using an M3 Pro MacBook Pro and an M4 Mac Mini connected via Thunderbolt 4 to accelerate image generation.


## Cluster Run Script

This script simplifies running distributed MLX jobs over a Thunderbolt Bridge (or any specific network interface) using OpenMPI.

### Setup
1. Open `cluster_run.sh`.
2. Edit the **Environment Configuration** section:
   - update `HOST_PYTHON` / `HOST_DIR` (Your local machine paths).
   - update `NODE_USER`, `NODE_IP`, `NODE_PYTHON`, `NODE_DIR` (Your remote machine details).

### Usage
```bash
./cluster_run.sh your_script.py --arg1 value
```

