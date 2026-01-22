#!/bin/bash
#
# NCCL Collective Operations Test (SBATCH)
#
# Tests all_reduce, all_gather, broadcast, reduce_scatter
# to verify NCCL communication across all GPUs.
#
# Usage: 
#   sbatch scripts/nccl_test.sh
#   sbatch --nodes=4 --gpus-per-node=4 scripts/nccl_test.sh
#
# Override defaults with environment variables:
#   NODES=4 GPUS_PER_NODE=4 sbatch scripts/nccl_test.sh
#

#SBATCH --job-name=nccl-test
#SBATCH --partition=main
#SBATCH --output=logs/nccl_%j.out
#SBATCH --error=logs/nccl_%j.err
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
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/shared/gpu-cluster-test/images/smilenaderi+gpu-cluster-test+main.sqsh}
PROJECT_PATH=${PROJECT_PATH:-/shared/gpu-cluster-test}

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
echo "NCCL Collective Operations Test"
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

# Launch NCCL test on all nodes
srun --container-image=$CONTAINER_IMAGE \
     --container-mounts=$PROJECT_PATH:/workspace/project \
     bash -c "torchrun \
         --nnodes=$NODES \
         --nproc_per_node=$GPUS_PER_NODE \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
         /workspace/project/src/nccl_test.py"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ NCCL TEST PASSED ($TOTAL_GPUS GPUs)"
    echo "All collective operations working correctly"
else
    echo "❌ NCCL TEST FAILED (Exit code: $EXIT_CODE)"
    echo ""
    echo "Common issues:"
    echo "  - Network connectivity problems between nodes"
    echo "  - Firewall blocking NCCL ports"
    echo "  - InfiniBand/RoCE configuration issues"
    echo "  - NCCL version mismatch"
    echo ""
    echo "Troubleshooting:"
    echo "  - Check network: ping between nodes"
    echo "  - Verify NCCL_DEBUG=INFO in logs"
    echo "  - Test with fewer GPUs first"
fi
echo "=========================================="

exit $EXIT_CODE
