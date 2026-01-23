# GPU Cluster Test - Usage Guide

## ⚠️ Important: Slurm Cluster Setup

**If you're using Slurm with Enroot/Pyxis**, you must import the container image first:
https://github.com/NVIDIA/pyxis/issues/70

```bash
# Import the container image locally (REQUIRED for Slurm)
./gpu-test import
```

Run this command **before** any validation or NCCL tests on Slurm clusters.

## Quick Reference

### Installation
```bash
git clone https://github.com/smilenaderi/gpu-cluster-test.git
cd gpu-cluster-test
chmod +x gpu-test
```

### Basic Commands

```bash
# Show help
./gpu-test help

# Import container image (REQUIRED FIRST for Slurm clusters)
./gpu-test import

# Validate cluster (2 nodes × 2 GPUs)
./gpu-test validate --nodes 2 --gpus-per-node 2

# Test NCCL communication
./gpu-test nccl --nodes 2 --gpus-per-node 2

# Interactive mode (real-time output)
./gpu-test validate --nodes 2 --gpus-per-node 2 -i

# CPU test (no GPU needed)
./gpu-test validate --dry-run
```

## Commands

### `import` - Import Container Image (Run This First!)

**⚠️ Required for Slurm clusters** - Import the container image locally before running tests.

**Example:**
```bash
# Import the container image from GitHub Container Registry
./gpu-test import
```

This is necessary due to a known issue where Slurm clusters cannot access external Docker repositories directly. The command downloads and converts the image to Enroot format for local use.

### `validate` - Cluster Validation
Tests distributed training with PyTorch DDP.

**Examples:**
```bash
# Small cluster (4 GPUs)
./gpu-test validate --nodes 2 --gpus-per-node 2

# Medium cluster (16 GPUs)
./gpu-test validate --nodes 4 --gpus-per-node 4 --epochs 10

# Large cluster (64 GPUs)
./gpu-test validate --nodes 8 --gpus-per-node 8 --epochs 20
```

### `nccl` - NCCL Communication Test
Tests all NCCL collective operations.

**Examples:**
```bash
# Test 2 nodes × 2 GPUs
./gpu-test nccl --nodes 2 --gpus-per-node 2

# Test 4 nodes × 4 GPUs
./gpu-test nccl --nodes 4 --gpus-per-node 4
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--nodes N` | Number of nodes | 2 |
| `--gpus-per-node N` | GPUs per node | 8 |
| `--epochs N` | Training epochs | 5 |
| `--batch-size N` | Batch size per GPU | 64 |
| `-i, --interactive` | Run interactively with real-time output | false |
| `--dry-run` | Test on CPU | false |

## Output

All output is displayed in real-time. No need to check log files.

**Success looks like:**
```
✅ Epoch 1/5 complete - Avg Loss: 2.3456
✅ Epoch 2/5 complete - Avg Loss: 2.1234
...
🎉 GPU Cluster Acceptance Test Passed!
```

**For batch jobs on Slurm:**
- Logs saved to: `logs/acceptance_<job_id>.out`
- Monitor: `tail -f logs/acceptance_*.out`

## Advanced Usage

### Using Custom Container Images

```bash
# Import the container image
./gpu-test import
```

### Direct Script Access (Advanced)

**Slurm Batch:**
```bash
sbatch --nodes=2 --gpus-per-node=2 scripts/validate_clsuter.sh
sbatch --nodes=2 --gpus-per-node=2 scripts/nccl_test.sh

# Import image manually
./scripts/import_image.sh
```

**Slurm Interactive:**
```bash
./scripts/run_acceptance.sh --nodes 2 --gpus-per-node 2
```

**Kubernetes:**
```bash
kubectl apply -f kubernetes-example.yaml
```

**Docker:**
```bash
docker run --gpus all --rm --ipc=host \
  -v $(pwd):/workspace/project \
  pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \
  bash -c "pip install -q torchvision && \
  torchrun --nproc_per_node=2 /workspace/project/src/train.py --epochs 5"
```

### Python Scripts

```bash
# CPU dry-run
python src/train.py --dry-run --epochs 2

# Single GPU
python src/train.py --epochs 2

# Multi-GPU
torchrun --nproc_per_node=2 src/train.py --epochs 5

# NCCL test
torchrun --nproc_per_node=2 src/nccl_test.py
```

## Troubleshooting

### Not enough GPUs
```bash
# Start with minimal resources
./gpu-test validate --nodes 1 --gpus-per-node 1
```

### Out of memory
```bash
# Reduce batch size
./gpu-test validate --nodes 2 --gpus-per-node 2 --batch-size 16
```

### NCCL timeout
```bash
# Test communication first
./gpu-test nccl --nodes 2 --gpus-per-node 2
```

### Test without GPU
```bash
# CPU dry-run
./gpu-test validate --dry-run
```

## Environment Variables

| Variable | Description | Auto-set by |
|----------|-------------|-------------|
| `MASTER_ADDR` | Master node IP | Slurm, Kubernetes |
| `MASTER_PORT` | Communication port | - |
| `WORLD_SIZE` | Total processes | Slurm, Kubernetes |
| `RANK` | Process rank | Slurm, Kubernetes |
| `LOCAL_RANK` | Local rank | Slurm, Kubernetes |

## File Structure

```
gpu-cluster-test/
├── gpu-test                    # Main CLI tool (use this!)
├── src/
│   ├── train.py               # Training script
│   ├── nccl_test.py           # NCCL test script
│   └── requirements.txt       # Python dependencies
├── scripts/
│   ├── validate_clsuter.sh    # Slurm validation job
│   ├── nccl_test.sh           # Slurm NCCL job
│   ├── run_acceptance.sh      # Interactive launcher
│   └── import_image.sh        # Import custom image
├── logs/                      # Job output logs
├── Dockerfile                 # Container definition
├── kubernetes-example.yaml    # Kubernetes deployment
└── readme.md                  # Full documentation
```

## Best Practices

1. **Import image first**: Run `./gpu-test import` before any tests on Slurm clusters
2. **Start small**: Test with 1-2 nodes first
3. **Run NCCL test**: Verify communication before training
4. **Use interactive mode**: For debugging and development
5. **Monitor resources**: Check GPU memory and utilization
6. **Scale gradually**: Increase nodes/GPUs incrementally

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review logs in `logs/` directory
3. Test with `--dry-run` to isolate GPU issues
4. Start with minimal resources and scale up

---

**Note:** The `./gpu-test` tool is the complete, recommended interface for all operations. It handles validation, NCCL testing, and image management in a unified way.
