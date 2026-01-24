#!/bin/bash
#
# GPU Diagnostics Test (SBATCH)
#
# Runs comprehensive GPU diagnostics across all nodes and GPUs.
#
# Container Image Behavior:
#   - If local squashfs exists: uses /shared/gpu-cluster-test/images/smilenaderi+gpu-cluster-test+main.sqsh
#   - If not found: automatically uses ghcr.io#smilenaderi/gpu-cluster-test:main
#   - Override with: CONTAINER_IMAGE="your-image" sbatch scripts/diagnose.sh
#
# Usage: 
#   sbatch scripts/diagnose.sh
#   sbatch --nodes=2 --gpus-per-node=2 scripts/diagnose.sh
#
# Override defaults with environment variables:
#   NODES=2 GPUS_PER_NODE=2 sbatch scripts/diagnose.sh
#

#SBATCH --job-name=gpu-diagnose
#SBATCH --partition=main
#SBATCH --output=logs/diagnose_%j.out
#SBATCH --error=logs/diagnose_%j.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration (can be overridden by environment variables)
NODES=${NODES:-$SLURM_JOB_NUM_NODES}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
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
echo "GPU Cluster Diagnostics"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Master Node: $MASTER_IP:$MASTER_PORT"
echo "Container: $CONTAINER_IMAGE"
echo "Project Path: $PROJECT_PATH"
echo "=========================================="
echo ""

# Launch diagnostics on all nodes
srun --container-image=$CONTAINER_IMAGE \
     --container-mounts=$PROJECT_PATH:/workspace/project \
     bash -c "torchrun \
         --nnodes=$NODES \
         --nproc_per_node=$GPUS_PER_NODE \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
         /workspace/project/src/gpu_diagnostics.py"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ DIAGNOSTICS COMPLETE ($TOTAL_GPUS GPUs)"
    echo "All GPU diagnostics passed"
else
    echo "❌ DIAGNOSTICS FAILED (Exit code: $EXIT_CODE)"
    echo ""
    echo "Common issues:"
    echo "  - PyTorch not installed in container"
    echo "  - CUDA driver mismatch"
    echo "  - GPU not accessible"
    echo "  - Network connectivity problems"
    echo ""
    echo "Troubleshooting:"
    echo "  - Check container image has PyTorch"
    echo "  - Verify nvidia-smi works"
    echo "  - Check CUDA_VISIBLE_DEVICES"
fi
echo "=========================================="

exit $EXIT_CODE
