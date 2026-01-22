# GPU Cluster Acceptance Test

A distributed PyTorch DDP acceptance test suite for validating GPU cluster configurations. This tool verifies that your Kubernetes or Slurm cluster nodes are correctly configured, GPUs are accessible, and NCCL (inter-GPU communication) is working properly.

## Overview

This project provides a lightweight acceptance test that:
- Validates multi-node GPU cluster setup
- Tests distributed training with PyTorch DDP
- Verifies NCCL communication between GPUs
- Uses synthetic data to avoid external dependencies
- Supports Slurm, Kubernetes, and Docker environments

## Project Structure

```
gpu-cluster-test/
├── src/
│   ├── train.py              # Main training script with DDP
│   ├── nccl_test.py          # NCCL collective operations test
│   └── requirements.txt      # Python dependencies
├── scripts/
│   ├── run_acceptance.sh     # Interactive Slurm launcher
│   ├── validate_clsuter.sh   # Slurm job with custom image
│   ├── nccl_test.sh          # NCCL operations test job
│   └── import_image.sh       # Import custom image from GHCR
├── images/                   # Container images (.sqsh files)
├── logs/                     # Job output logs
├── Dockerfile                # Container definition
└── readme.md                 # This file
```

## Prerequisites

- NVIDIA GPU(s) with CUDA support
- PyTorch 2.0+ with CUDA
- For Slurm: Slurm workload manager with GPU support
- For Kubernetes: Kubeflow PyTorch Operator
- For Docker: Docker with NVIDIA Container Toolkit

## Quick Start


##  Setup GPU Cluster Test
```bash
cd /shared
git clone https://github.com/smilenaderi/gpu-cluster-test.git
cd gpu-cluster-test
chmod +x scripts/run_acceptance.sh
./scripts/run_acceptance.sh
```

### Option 1: Slurm (Interactive)

Run interactively with real-time output:

```bash
chmod +x scripts/run_acceptance.sh
./scripts/run_acceptance.sh
```

### Option 2: Slurm (Custom Image from GHCR)

First, import your custom Docker image from GitHub Container Registry:

```bash
cd /shared/gpu-cluster-test
./scripts/import_image.sh
```

This will download and convert the image to `images/smilenaderi+gpu-cluster-test+main.sqsh`. Then submit the job:

```bash
sbatch scripts/validate_clsuter.sh
```

Monitor the job:

```bash
squeue -u $USER
tail -f logs/acceptance_<job_id>.out
```

### Option 3: NCCL Collective Operations Test

Test NCCL all_reduce, all_gather, broadcast, and reduce_scatter:

```bash
sbatch scripts/nccl_test.sh
```

This verifies that inter-GPU communication is working correctly by:
- Testing all_reduce SUM across all GPUs
- Testing all_gather to collect data from all ranks
- Testing broadcast from rank 0
- Testing reduce_scatter operations

### Option 4: Docker (Single Node)

Test on a single node with multiple GPUs:

```bash
docker run --gpus all --rm --ipc=host \
  -v $(pwd):/workspace/project \
  pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \
  bash -c "pip install -q torchvision && \
  torchrun --nproc_per_node=8 /workspace/project/src/train.py --epochs 5"
```

### Option 5: Kubernetes (Kubeflow)

Deploy using PyTorch Operator:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: gpu-acceptance-test
spec:
  pytorchReplicaSpecs:
    Worker:
      replicas: 2
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
              command: 
                - "torchrun"
                - "--nnodes=2"
                - "--nproc_per_node=8"
                - "/workspace/src/train.py"
                - "--epochs=5"
              resources:
                limits:
                  nvidia.com/gpu: 8
          restartPolicy: OnFailure
```

Apply the manifest:

```bash
kubectl apply -f pytorch-job.yaml
kubectl logs -f <pod-name>
```

## Configuration

### Training Parameters

Edit `src/train.py` or pass command-line arguments:

```bash
python src/train.py --epochs 10 --batch-size 64
```

Available options:
- `--epochs`: Number of training epochs (default: 2)
- `--batch-size`: Batch size per GPU (default: 32)
- `--dry-run`: Run on CPU for testing (no GPU required)

### Slurm Configuration

Edit `scripts/run_acceptance.sh` or `scripts/validate_clsuter.sh` to adjust:
- `--nodes`: Number of nodes (default: 2)
- `--gpus-per-node`: GPUs per node (default: 8)
- `--time`: Job time limit (default: 00:20:00)
- `--partition`: Slurm partition name

### Custom Image Configuration

To use your own Docker image from GitHub Container Registry:

1. Build and push your image to GHCR:
```bash
docker build -t ghcr.io/username/gpu-cluster-test:main .
docker push ghcr.io/username/gpu-cluster-test:main
```

2. Update `scripts/import_image.sh` with your image URL:
```bash
IMAGE_URL="docker://ghcr.io#username/gpu-cluster-test:main"
```

3. Import and run:
```bash
./scripts/import_image.sh
sbatch scripts/validate_clsuter.sh
```

## Expected Output

Successful run should show:

```
🚀 Starting training on cuda:0...
✅ Epoch 1/5 complete.
✅ Epoch 2/5 complete.
...
✅ Epoch 5/5 complete.
🎉 GPU Cluster Acceptance Test Passed!
```

## Troubleshooting

### NCCL Errors

If you see NCCL timeout or communication errors:
- Verify network connectivity between nodes
- Check firewall rules allow NCCL ports
- Ensure InfiniBand/RoCE is properly configured

### GPU Not Found

```bash
# Verify GPU visibility
nvidia-smi

# Check CUDA availability in PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### Container Issues

If using Slurm with containers:
- Verify container image is accessible
- Check mount paths are correct
- Ensure Enroot/Pyxis is configured

## Development

### Running Tests Locally

```bash
# CPU dry-run (no GPU required)
python src/train.py --dry-run --epochs 2

# Single GPU
python src/train.py --epochs 2

# Multi-GPU (single node)
torchrun --nproc_per_node=2 src/train.py --epochs 2
```

### Building Custom Container

```bash
docker build -t gpu-cluster-test:latest .
docker run --gpus all --rm gpu-cluster-test:latest --help
```

## License

This project is provided as-is for cluster validation purposes.

