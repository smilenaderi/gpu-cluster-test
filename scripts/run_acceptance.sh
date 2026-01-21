#!/bin/bash
#
# GPU Cluster Acceptance Test - Interactive Launcher
#
# This script launches a multi-node distributed training job interactively,
# allowing you to see real-time output and quickly validate cluster setup.
#
# Usage: ./scripts/run_acceptance.sh
#

set -e  # Exit on error

echo "=========================================="
echo "GPU Cluster Acceptance Test (Interactive)"
echo "=========================================="

# Resolve the master node IP
MASTER_IP=$(scontrol show hostnames worker-[0-1] | head -n 1)

if [ -z "$MASTER_IP" ]; then
    echo "❌ Error: Could not resolve master node IP"
    echo "   Make sure worker-[0-1] nodes are available"
    exit 1
fi

echo "Configuration:"
echo "  - Nodes: 2"
echo "  - GPUs per node: 8"
echo "  - Total GPUs: 16"
echo "  - Master node: $MASTER_IP:29500"
echo "  - Container: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"
echo "=========================================="
echo ""
echo "🚀 Launching distributed training job..."
echo ""

# Launch the job with srun (interactive mode)
srun --nodes=2 \
     --gpus-per-node=8 \
     --ntasks-per-node=1 \
     --partition=main \
     --job-name=gpu-cluster-test \
     --container-image=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \
     --container-mounts=/shared/gpu-cluster-test:/workspace/project \
     bash -c "pip install -q --no-cache-dir torchvision --no-deps && \
     torchrun \
         --nnodes=2 \
         --nproc_per_node=8 \
         --rdzv_id=$RANDOM \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_IP:29500 \
         /workspace/project/src/train.py --epochs 5 --batch-size 64"

# Capture the exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST STATUS: SUCCESS"
    echo ""
    echo "The 16-GPU cluster acceptance test passed successfully."
    echo "Your cluster is properly configured for distributed training."
else
    echo "❌ TEST STATUS: FAILED"
    echo ""
    echo "The job exited with error code $EXIT_CODE."
    echo "Common issues:"
    echo "  - NCCL communication problems between nodes"
    echo "  - GPU not accessible in container"
    echo "  - Network configuration issues"
    echo ""
    echo "Check the output above for specific error messages."
fi
echo "=========================================="

exit $EXIT_CODE