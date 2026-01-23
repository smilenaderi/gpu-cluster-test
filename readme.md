# GPU Cluster Acceptance Test

A unified command-line tool for validating GPU cluster configurations. Tests distributed training with PyTorch DDP and NCCL communication across multiple nodes and GPUs.


**If you're using Slurm with Enroot/Pyxis**, you must import the container image first:
https://github.com/NVIDIA/pyxis/issues/70

```bash
# Import the container image locally (required for Slurm)
./gpu-test import
```

This downloads the image from GitHub Container Registry and prepares it for local use with Enroot. Run this command **before** running any validation or NCCL tests on Slurm clusters.

## Quick Start

### Installation

```bash
cd /shared
git clone https://github.com/smilenaderi/gpu-cluster-test.git
cd gpu-cluster-test
chmod +x gpu-test
```

### Basic Usage

```bash
# Show help
./gpu-test help

# Import container image (REQUIRED FIRST for Slurm clusters)
./gpu-test import

# Run cluster validation test (2 nodes × 2 GPUs)
./gpu-test validate --nodes 2 --gpus-per-node 2

# Run NCCL communication test
./gpu-test nccl --nodes 2 --gpus-per-node 2

# Interactive mode (see output in real-time)
./gpu-test validate --nodes 2 --gpus-per-node 2 -i

# CPU dry-run (no GPU needed)
./gpu-test validate --dry-run
```

## Features

✅ **Unified CLI** - Single command-line tool for all operations  
✅ **Real-time Output** - See results immediately, no log file hunting  
✅ **Flexible Configuration** - Run with any number of nodes and GPUs  
✅ **Multiple Platforms** - Slurm, Kubernetes, Docker, standalone VMs  
✅ **Comprehensive Testing** - Training + NCCL collective operations  
✅ **No Dependencies** - Uses synthetic data, no downloads required  

## Getting Started

## Commands

### `import` - Import Container Image (Run This First!)

**⚠️ Required for Slurm clusters** - Import the container image locally before running tests. This is necessary due to a known issue where Slurm clusters cannot access external Docker repositories directly.

```bash
# Import the container image from GitHub Container Registry
./gpu-test import
```

This command:
- Downloads the image from `ghcr.io/smilenaderi/gpu-cluster-test:main`
- Converts it to Enroot format (`.sqsh` file)
- Stores it locally in the `images/` directory
- Makes it available for all subsequent test runs

**When to run:**
- Before your first validation or NCCL test
- After updating the container image
- If you see "image not found" errors

### `validate` - Cluster Validation Test

Tests distributed training with PyTorch DDP to verify your cluster is properly configured.

```bash
# Small cluster (4 GPUs)
./gpu-test validate --nodes 2 --gpus-per-node 2

# Medium cluster (16 GPUs)
./gpu-test validate --nodes 4 --gpus-per-node 4 --epochs 10

# Large cluster (64 GPUs)
./gpu-test validate --nodes 8 --gpus-per-node 8 --epochs 20
```

### `nccl` - NCCL Communication Test

Tests all NCCL collective operations (all_reduce, all_gather, broadcast, reduce_scatter) to verify inter-GPU communication.

**Note:** For multi-node tests, batch mode is recommended. Interactive mode (`-i`) is not supported for NCCL tests.

```bash
# Test 2 nodes × 2 GPUs (batch mode)
./gpu-test nccl --nodes 2 --gpus-per-node 2

# Test 4 nodes × 4 GPUs
./gpu-test nccl --nodes 4 --gpus-per-node 4

# Monitor the job
squeue -u $USER
tail -f logs/nccl_*.out
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--nodes N` | Number of nodes | 2 |
| `--gpus-per-node N` | GPUs per node | 8 |
| `--epochs N` | Training epochs | 5 |
| `--batch-size N` | Batch size per GPU | 64 |
| `-i, --interactive` | Run interactively with real-time output (validate only) | false |
| `--dry-run` | Test on CPU without GPU | false |

## Examples

### Development Testing

```bash
# Quick test with minimal resources
./gpu-test validate --nodes 1 --gpus-per-node 1 --epochs 2

# CPU test (no GPU needed)
./gpu-test validate --dry-run
```

### Production Validation

```bash
# Standard acceptance test
./gpu-test validate --nodes 2 --gpus-per-node 4 --epochs 10

# NCCL communication test
./gpu-test nccl --nodes 2 --gpus-per-node 4
```

### Large Scale Testing

```bash
# High-performance cluster validation
./gpu-test validate --nodes 8 --gpus-per-node 8 --epochs 20 --batch-size 256
```

## Environment Detection

The tool automatically detects your environment:

- **Slurm**: Uses `srun` (interactive) or `sbatch` (batch)
- **Kubernetes**: Uses `kubectl` and PyTorchJob operator
- **Standalone**: Uses `torchrun` directly

## Output

All output is displayed in real-time. No need to check log files manually.

For batch jobs on Slurm, logs are also saved to:
- `logs/acceptance_<job_id>.out` - Validation test output
- `logs/nccl_<job_id>.out` - NCCL test output

Monitor batch jobs:
```bash
squeue -u $USER
tail -f logs/acceptance_*.out
```

---

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

## Advanced Usage

### Using Custom Container Images

If you need to use a custom Docker image with additional dependencies:

```bash
# Import the container image
./gpu-test import

# Or manually import with enroot
./scripts/import_image.sh
```

### Direct Script Access (Advanced Users)

If you prefer to use the underlying scripts directly:

**Slurm Batch Jobs:**
```bash
# Validation test
sbatch --nodes=2 --gpus-per-node=2 scripts/validate_clsuter.sh

# NCCL test
sbatch --nodes=2 --gpus-per-node=2 scripts/nccl_test.sh

# With environment variables
NODES=4 GPUS_PER_NODE=4 EPOCHS=10 sbatch scripts/validate_clsuter.sh
```

**Slurm Interactive:**
```bash
./scripts/run_acceptance.sh --nodes 2 --gpus-per-node 2 --epochs 5
```

**Kubernetes:**
```bash
kubectl apply -f kubernetes-example.yaml
kubectl get pytorchjobs
kubectl logs -f <pod-name>
```

**Docker (Single Node):**
```bash
docker run --gpus all --rm --ipc=host \
  -v $(pwd):/workspace/project \
  pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \
  bash -c "pip install -q torchvision && \
  torchrun --nproc_per_node=2 /workspace/project/src/train.py --epochs 5"
```

### Python Scripts

Run tests directly with Python:

```bash
# CPU dry-run (no GPU required)
python src/train.py --dry-run --epochs 2

# Single GPU
python src/train.py --epochs 2

# Multi-GPU (single node)
torchrun --nproc_per_node=2 src/train.py --epochs 5

# NCCL test
torchrun --nproc_per_node=2 src/nccl_test.py
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MASTER_ADDR` | Master node IP address | auto-detect | No |
| `MASTER_PORT` | Communication port | 29500 | No |
| `WORLD_SIZE` | Total number of processes | auto | No |
| `RANK` | Process rank (0 to WORLD_SIZE-1) | auto | No |
| `LOCAL_RANK` | Local process rank on node | auto | No |
| `NODES` | Number of nodes | 2 | No |
| `GPUS_PER_NODE` | GPUs per node | 8 | No |
| `EPOCHS` | Training epochs | 5 | No |
| `BATCH_SIZE` | Batch size per GPU | 64 | No |
| `CONTAINER_IMAGE` | Container image path | (varies) | No |
| `PROJECT_PATH` | Project mount path | /shared/gpu-cluster-test | No |

**Note:** `MASTER_ADDR`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` are automatically set by:
- Slurm (via `torchrun`)
- Kubernetes PyTorchJob operator
- Manual setup for standalone VMs

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

Or use directly:
```bash
./scripts/run_acceptance.sh --container ghcr.io/username/gpu-cluster-test:main
```

## Configuration

### Command-Line Options

All scripts support flexible configuration via command-line arguments:

**Training Parameters:**
```bash
python src/train.py --epochs 10 --batch-size 64 --dry-run
```
- `--epochs`: Number of training epochs (default: 2)
- `--batch-size`: Batch size per GPU (default: 32)
- `--dry-run`: Run on CPU for testing (no GPU required)

**Launcher Options:**
```bash
./scripts/run_acceptance.sh --nodes 4 --gpus-per-node 4 --epochs 10 --batch-size 128
```
- `--nodes`: Number of nodes (default: 2)
- `--gpus-per-node`: GPUs per node (default: 8)
- `--epochs`: Training epochs (default: 5)
- `--batch-size`: Batch size per GPU (default: 64)
- `--partition`: Slurm partition name (default: main)
- `--container`: Container image to use
- `--project-path`: Project mount path (default: /shared/gpu-cluster-test)
- `--master-port`: Master communication port (default: 29500)

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MASTER_ADDR` | Master node IP address | auto-detect | No |
| `MASTER_PORT` | Communication port | 29500 | No |
| `WORLD_SIZE` | Total number of processes | auto | No |
| `RANK` | Process rank (0 to WORLD_SIZE-1) | auto | No |
| `LOCAL_RANK` | Local process rank on node | auto | No |
| `NODES` | Number of nodes | 2 | No |
| `GPUS_PER_NODE` | GPUs per node | 8 | No |
| `EPOCHS` | Training epochs | 5 | No |
| `BATCH_SIZE` | Batch size per GPU | 64 | No |
| `CONTAINER_IMAGE` | Container image path | (varies) | No |
| `PROJECT_PATH` | Project mount path | /shared/gpu-cluster-test | No |

**Note:** `MASTER_ADDR`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` are automatically set by:
- Slurm (via `torchrun`)
- Kubernetes PyTorchJob operator
- Manual setup for standalone VMs

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

Or use directly:
```bash
./scripts/run_acceptance.sh --container ghcr.io/username/gpu-cluster-test:main
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

---

## Summary

The GPU cluster test provides a unified CLI tool (`./gpu-test`) for validating GPU clusters:

- **Simple**: Single command for all operations
- **Complete**: Handles validation, NCCL testing, and image management
- **Reliable**: Real-time output, no log file hunting
- **Flexible**: Works with any cluster size (1×1 to 8×8+ GPUs)
- **Portable**: Runs on Slurm, Kubernetes, Docker, standalone VMs

Quick start:
```bash
./gpu-test import                                    # Run this FIRST on Slurm clusters
./gpu-test validate --nodes 2 --gpus-per-node 2
./gpu-test nccl --nodes 2 --gpus-per-node 2
```

The `./gpu-test` tool is the recommended way to use this project. For advanced use cases, you can access the underlying scripts in the `scripts/` directory.

