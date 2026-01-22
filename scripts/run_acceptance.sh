#!/bin/bash
#
# GPU Cluster Acceptance Test - Interactive Launcher
#
# This script launches a multi-node distributed training job interactively,
# allowing you to see real-time output and quickly validate cluster setup.
#
# Usage: 
#   ./scripts/run_acceptance.sh [OPTIONS]
#
# Options:
#   --nodes N              Number of nodes (default: 2)
#   --gpus-per-node N      GPUs per node (default: 8)
#   --epochs N             Training epochs (default: 5)
#   --batch-size N         Batch size per GPU (default: 64)
#   --partition NAME       Slurm partition (default: main)
#   --container IMAGE      Container image (default: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel)
#   --project-path PATH    Project mount path (default: /shared/gpu-cluster-test)
#   --master-port PORT     Master communication port (default: 29500)
#
# Examples:
#   ./scripts/run_acceptance.sh --nodes 2 --gpus-per-node 2
#   ./scripts/run_acceptance.sh --nodes 4 --gpus-per-node 4 --epochs 10
#

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Default configuration
NODES=2
GPUS_PER_NODE=8
EPOCHS=5
BATCH_SIZE=64
PARTITION="main"
CONTAINER="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"
PROJECT_PATH="/shared/gpu-cluster-test"
MASTER_PORT=29500

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --container)
            CONTAINER="$2"
            shift 2
            ;;
        --project-path)
            PROJECT_PATH="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --nodes N              Number of nodes (default: 2)"
            echo "  --gpus-per-node N      GPUs per node (default: 8)"
            echo "  --epochs N             Training epochs (default: 5)"
            echo "  --batch-size N         Batch size per GPU (default: 64)"
            echo "  --partition NAME       Slurm partition (default: main)"
            echo "  --container IMAGE      Container image"
            echo "  --project-path PATH    Project mount path (default: /shared/gpu-cluster-test)"
            echo "  --master-port PORT     Master communication port (default: 29500)"
            echo ""
            echo "Examples:"
            echo "  $0 --nodes 2 --gpus-per-node 2"
            echo "  $0 --nodes 4 --gpus-per-node 4 --epochs 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Calculate total GPUs
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

# Validation
if [ "$NODES" -lt 1 ]; then
    echo "❌ Error: NODES must be >= 1 (got: $NODES)"
    exit 1
fi

if [ "$GPUS_PER_NODE" -lt 1 ]; then
    echo "❌ Error: GPUS_PER_NODE must be >= 1 (got: $GPUS_PER_NODE)"
    exit 1
fi

if [ "$EPOCHS" -lt 1 ]; then
    echo "❌ Error: EPOCHS must be >= 1 (got: $EPOCHS)"
    exit 1
fi

if [ ! -d "$PROJECT_PATH" ]; then
    echo "❌ Error: PROJECT_PATH does not exist: $PROJECT_PATH"
    exit 1
fi

echo "=========================================="
echo "GPU Cluster Acceptance Test (Interactive)"
echo "=========================================="

# Resolve the master node IP - Make it environment-agnostic
if command -v scontrol &> /dev/null; then
    # Slurm environment
    if [ -n "$SLURM_JOB_NODELIST" ]; then
        MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    else
        # Get first available node from partition
        MASTER_IP=$(sinfo -N -h -p $PARTITION | awk '{print $1}' | head -n 1)
    fi
elif [ -n "$MASTER_ADDR" ]; then
    # Kubernetes/PyTorchJob sets this automatically
    MASTER_IP=$MASTER_ADDR
else
    # Fallback for VMs or manual setup
    MASTER_IP=${MASTER_IP:-localhost}
fi

if [ -z "$MASTER_IP" ]; then
    echo "❌ Error: Could not resolve master node IP"
    echo "   Make sure Slurm is configured or set MASTER_ADDR environment variable"
    exit 1
fi

echo "Configuration:"
echo "  - Nodes: $NODES"
echo "  - GPUs per node: $GPUS_PER_NODE"
echo "  - Total GPUs: $TOTAL_GPUS"
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Partition: $PARTITION"
echo "  - Master node: $MASTER_IP:$MASTER_PORT"
echo "  - Container: $CONTAINER"
echo "  - Project path: $PROJECT_PATH"
echo "=========================================="
echo ""
echo "🚀 Launching distributed training job..."
echo ""

# Launch the job with srun (interactive mode)
srun --nodes=$NODES \
     --gpus-per-node=$GPUS_PER_NODE \
     --ntasks-per-node=1 \
     --partition=$PARTITION \
     --job-name=gpu-cluster-test \
     --container-image=$CONTAINER \
     --container-mounts=$PROJECT_PATH:/workspace/project \
     bash -c "pip install -q --no-cache-dir torchvision --no-deps && \
     torchrun \
         --nnodes=$NODES \
         --nproc_per_node=$GPUS_PER_NODE \
         --rdzv_id=$RANDOM \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
         /workspace/project/src/train.py --epochs $EPOCHS --batch-size $BATCH_SIZE"

# Capture the exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST STATUS: SUCCESS"
    echo ""
    echo "The $TOTAL_GPUS-GPU cluster acceptance test passed successfully."
    echo "Your cluster is properly configured for distributed training."
else
    echo "❌ TEST STATUS: FAILED"
    echo ""
    echo "The job exited with error code $EXIT_CODE."
    echo ""
    echo "Common issues:"
    echo "  - NCCL communication problems between nodes"
    echo "  - GPU not accessible in container"
    echo "  - Network configuration issues"
    echo "  - Insufficient GPU memory (try reducing --batch-size)"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. Test with fewer resources: --nodes 1 --gpus-per-node 1"
    echo "  2. Run NCCL test: sbatch --nodes=2 --gpus-per-node=2 scripts/nccl_test.sh"
    echo "  3. Check logs for specific error messages"
    echo "  4. Verify GPU availability: srun nvidia-smi"
fi
echo "=========================================="

exit $EXIT_CODE