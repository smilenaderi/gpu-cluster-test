#!/bin/bash
#
# GPU Cluster Test - Custom Container (SBATCH)
#
# Uses custom-built Docker image from GitHub Container Registry
#
# Container Image Behavior:
#   - If local squashfs exists: uses /shared/gpu-cluster-test/images/smilenaderi+gpu-cluster-test+main.sqsh
#   - If not found: automatically uses ghcr.io#smilenaderi/gpu-cluster-test:main
#   - Override with: CONTAINER_IMAGE="your-image" sbatch scripts/distributed_training_test.sh
#
# Usage: 
#   sbatch scripts/distributed_training_test.sh
#   sbatch --nodes=4 --gpus-per-node=4 scripts/distributed_training_test.sh
#
# Override defaults with sbatch options or environment variables:
#   NODES=4 GPUS_PER_NODE=4 EPOCHS=10 sbatch scripts/distributed_training_test.sh
#

#SBATCH --job-name=gpu-test-custom
#SBATCH --partition=main
#SBATCH --output=logs/acceptance_%j.out
#SBATCH --error=logs/acceptance_%j.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration (can be overridden by environment variables)
NODES=${NODES:-$SLURM_JOB_NUM_NODES}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-64}
MASTER_PORT=${MASTER_PORT:-29500}
PROJECT_PATH=${PROJECT_PATH:-/shared/gpu-cluster-test}

# Container image: use local squashfs if exists, otherwise use GitHub Container Registry
LOCAL_SQUASHFS="/shared/gpu-cluster-test/images/smilenaderi+gpu-cluster-test+main.sqsh"
if [ -f "$LOCAL_SQUASHFS" ]; then
    CONTAINER_IMAGE=${CONTAINER_IMAGE:-$LOCAL_SQUASHFS}
else
    CONTAINER_IMAGE=${CONTAINER_IMAGE:-"ghcr.io#smilenaderi/gpu-cluster-test:main"}
fi

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

# Get master node IP
MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

if [ -z "$MASTER_IP" ]; then
    echo "❌ Error: Could not determine master node IP"
    exit 1
fi

# Calculate total GPUs
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

echo "=========================================="
echo "GPU Cluster Test - Custom Container"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Master Node: $MASTER_IP:$MASTER_PORT"
echo "Container: $CONTAINER_IMAGE"
echo "Project Path: $PROJECT_PATH"
echo "=========================================="
echo ""

# Launch distributed training on all nodes using srun
srun --container-image=$CONTAINER_IMAGE \
     --container-mounts=$PROJECT_PATH:/workspace/project \
     bash -c "torchrun \
         --nnodes=$NODES \
         --nproc_per_node=$GPUS_PER_NODE \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
         /workspace/project/src/train.py \
         --epochs $EPOCHS \
         --batch-size $BATCH_SIZE"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST PASSED"
    echo "Cluster ($TOTAL_GPUS GPUs) is properly configured for distributed training"
else
    echo "❌ TEST FAILED (Exit code: $EXIT_CODE)"
    echo "Check error logs for details"
    echo ""
    echo "Common issues:"
    echo "  - NCCL communication problems between nodes"
    echo "  - GPU not accessible in container"
    echo "  - Network configuration issues"
    echo "  - Insufficient GPU memory (try reducing --batch-size)"
fi
echo "=========================================="

exit $EXIT_CODE
