# GPU Cluster Troubleshooting Guide

This guide helps diagnose and fix common GPU cluster issues.

## Quick Diagnostics

Run the diagnostics tool to identify issues:

```bash
# Single node diagnostics
python src/gpu_diagnostics.py

# Multi-GPU diagnostics
torchrun --nproc_per_node=8 src/gpu_diagnostics.py

# Multi-node diagnostics
torchrun --nnodes=2 --nproc_per_node=8 src/gpu_diagnostics.py
```

---

## Common Issues and Solutions

### 1. NCCL Timeout Errors

**Symptoms:**
```
NCCL error: unhandled system error
NCCL timeout
```

**Causes:**
- Network connectivity issues between nodes
- Firewall blocking NCCL ports
- Slow network (InfiniBand/RoCE misconfigured)

**Solutions:**

```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Test network connectivity
ping <other-node-ip>

# Check if ports are open (default: 29500)
nc -zv <master-node-ip> 29500

# Disable IB if causing issues
export NCCL_IB_DISABLE=1

# Force TCP instead of IB
export NCCL_SOCKET_IFNAME=eth0  # or your network interface

# Increase timeout
export NCCL_TIMEOUT=3600  # seconds
```

**Test NCCL specifically:**
```bash
./gpu-test nccl --nodes 2 --gpus-per-node 2
```

---

### 2. GPU Not Found / CUDA Not Available

**Symptoms:**
```
RuntimeError: No CUDA GPUs are available
CUDA not available
```

**Diagnosis:**
```bash
# Check GPU visibility
nvidia-smi

# Check CUDA in PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Run diagnostics
python src/gpu_diagnostics.py
```

**Solutions:**

**In Slurm:**
```bash
# Verify GPU allocation
srun --gpus=1 nvidia-smi

# Check if container has GPU access
srun --gpus=1 --container-image=<image> nvidia-smi
```

**In Docker:**
```bash
# Ensure --gpus flag is used
docker run --gpus all <image> nvidia-smi
```

**In Kubernetes:**
```yaml
# Ensure GPU resource request
resources:
  limits:
    nvidia.com/gpu: 1
```

---

### 3. Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Diagnosis:**
```bash
# Check GPU memory
nvidia-smi

# Run with smaller batch size
./gpu-test validate --nodes 1 --gpus-per-node 1 --batch-size 16
```

**Solutions:**

```bash
# Reduce batch size
./gpu-test validate --batch-size 16

# Use fewer GPUs initially
./gpu-test validate --nodes 1 --gpus-per-node 1

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**In code:**
```python
# Enable gradient checkpointing (if using custom models)
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
```

---

### 4. Distributed Training Hangs

**Symptoms:**
- Process starts but never completes
- No error messages
- Stuck at initialization

**Diagnosis:**
```bash
# Enable verbose logging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# Check if all ranks are starting
./gpu-test validate --nodes 2 --gpus-per-node 2
```

**Solutions:**

```bash
# Check master node is reachable
ping $MASTER_ADDR

# Verify all nodes can reach master
telnet $MASTER_ADDR $MASTER_PORT

# Try different port
export MASTER_PORT=29501

# Ensure all nodes have same code version
git pull  # on all nodes
```

**Common causes:**
- One node can't reach master
- Firewall blocking communication
- Different PyTorch versions on nodes
- Clock skew between nodes

---

### 5. Container Image Issues

**Symptoms:**
```
Error: container image not found
Failed to pull image
```

**Solutions:**

```bash
# Import image locally (Slurm)
./gpu-test import

# Or manually
./scripts/import_image.sh

# Verify image exists
ls -lh images/*.sqsh

# Use remote image directly
CONTAINER_IMAGE="ghcr.io#smilenaderi/gpu-cluster-test:main" \
  sbatch scripts/distributed_training_test.sh
```

---

### 6. Permission Denied Errors

**Symptoms:**
```
Permission denied
Cannot write to /workspace
```

**Solutions:**

```bash
# Check mount permissions
ls -la /shared/gpu-cluster-test

# Ensure scripts are executable
chmod +x scripts/*.sh
chmod +x gpu-test

# Check container user
docker run --rm <image> whoami
```

---

### 7. Slow Training / Poor Performance

**Diagnosis:**
```bash
# Run bandwidth test
python src/gpu_diagnostics.py

# Check GPU utilization
nvidia-smi dmon -s u

# Profile with nsys (if available)
nsys profile python src/train.py --epochs 1
```

**Solutions:**

```bash
# Enable P2P if available
export NCCL_P2P_LEVEL=NVL

# Use faster network interface
export NCCL_SOCKET_IFNAME=ib0  # for InfiniBand

# Increase batch size (if memory allows)
./gpu-test validate --batch-size 128

# Check data loading isn't bottleneck
# Increase num_workers in DataLoader
```

---

### 8. Version Mismatch Errors

**Symptoms:**
```
RuntimeError: NCCL version mismatch
CUDA version mismatch
```

**Diagnosis:**
```bash
# Check versions
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"
```

**Solutions:**

```bash
# Use consistent container image on all nodes
./gpu-test import  # on all nodes

# Or specify exact image
CONTAINER_IMAGE="ghcr.io#smilenaderi/gpu-cluster-test:main" \
  ./gpu-test validate
```

---

### 9. InfiniBand / RDMA Issues

**Symptoms:**
```
NCCL WARN NET/IB : No device found
NCCL WARN Failed to open IB device
```

**Diagnosis:**
```bash
# Check IB devices
ibstat
ibv_devices

# Check IB status
ibstatus
```

**Solutions:**

```bash
# Disable IB and use TCP
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# Or fix IB configuration (if available)
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3

# Test without IB first
NCCL_IB_DISABLE=1 ./gpu-test nccl --nodes 2 --gpus-per-node 2
```

---

### 10. Slurm Job Failures

**Symptoms:**
```
slurmstepd: error: Detected 1 oom-kill event(s)
Job failed with exit code 1
```

**Diagnosis:**
```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job <job_id>

# Check logs
cat logs/acceptance_*.err
cat logs/nccl_*.err
```

**Solutions:**

```bash
# Request more memory
sbatch --mem=64G scripts/distributed_training_test.sh

# Request more time
sbatch --time=01:00:00 scripts/distributed_training_test.sh

# Check node health
sinfo -N -l

# Try different partition
sbatch --partition=gpu scripts/distributed_training_test.sh
```

---

## Debug Mode

### Enable Maximum Verbosity

```bash
# PyTorch distributed
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO

# NCCL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# CUDA
export CUDA_LAUNCH_BLOCKING=1

# Run test
./gpu-test validate --nodes 2 --gpus-per-node 2
```

### Test Incrementally

```bash
# 1. Single GPU
./gpu-test validate --nodes 1 --gpus-per-node 1 --epochs 2

# 2. Multi-GPU single node
./gpu-test validate --nodes 1 --gpus-per-node 2 --epochs 2

# 3. Multi-node
./gpu-test validate --nodes 2 --gpus-per-node 2 --epochs 2

# 4. Full scale
./gpu-test validate --nodes 4 --gpus-per-node 8 --epochs 5
```

### Test NCCL Separately

```bash
# Test NCCL communication first
./gpu-test nccl --nodes 2 --gpus-per-node 2

# If NCCL works, training should work
./gpu-test validate --nodes 2 --gpus-per-node 2
```

---

## Getting Help

### Collect Diagnostic Information

```bash
# System info
uname -a
nvidia-smi
python --version

# PyTorch info
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Network info
ifconfig
hostname -I

# Run diagnostics
python src/gpu_diagnostics.py > diagnostics.txt 2>&1

# Slurm info (if applicable)
sinfo
squeue -u $USER
```

### Useful Log Files

```
logs/acceptance_*.out  # Training output
logs/acceptance_*.err  # Training errors
logs/nccl_*.out        # NCCL test output
logs/nccl_*.err        # NCCL test errors
```

### Environment Variables to Check

```bash
echo $MASTER_ADDR
echo $MASTER_PORT
echo $RANK
echo $LOCAL_RANK
echo $WORLD_SIZE
echo $CUDA_VISIBLE_DEVICES
echo $NCCL_DEBUG
```

---

## Best Practices

1. **Start Small**: Test with 1 GPU, then 2, then scale up
2. **Test NCCL First**: Run NCCL test before training
3. **Enable Debug Logs**: Use NCCL_DEBUG=INFO for troubleshooting
4. **Check Network**: Ensure all nodes can communicate
5. **Use Consistent Images**: Same container on all nodes
6. **Monitor Resources**: Watch GPU memory and utilization
7. **Incremental Scaling**: Add nodes/GPUs gradually
8. **Document Issues**: Keep notes on what works/doesn't

---

## Quick Reference

```bash
# Diagnostics
python src/gpu_diagnostics.py

# Test single GPU
./gpu-test validate --nodes 1 --gpus-per-node 1

# Test NCCL
./gpu-test nccl --nodes 2 --gpus-per-node 2

# Debug mode
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL \
  ./gpu-test validate --nodes 2 --gpus-per-node 2

# Check GPU
nvidia-smi

# Check logs
tail -f logs/acceptance_*.out
```

---

For more information, see:
- [README.md](readme.md) - Full documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
