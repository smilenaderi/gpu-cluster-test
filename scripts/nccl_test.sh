#!/bin/bash
#
# NCCL Collective Operations Test
#
# Tests all_reduce, all_gather, broadcast, reduce_scatter
# to verify NCCL communication across all GPUs.
#
# Usage: sbatch scripts/nccl_test.sh
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

# Get master node IP
MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "=========================================="
echo "NCCL Collective Operations Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 8))"
echo "Master Node: $MASTER_IP:$MASTER_PORT"
echo "=========================================="
echo ""

# Launch NCCL test on all nodes
srun --container-image=/shared/gpu-cluster-test/images/smilenaderi+gpu-cluster-test+main.sqsh \
     --container-mounts=/shared/gpu-cluster-test:/workspace/project \
     bash -c "torchrun \
         --nnodes=$SLURM_JOB_NUM_NODES \
         --nproc_per_node=8 \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
         /workspace/project/src/nccl_test.py"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ NCCL TEST PASSED"
else
    echo "❌ NCCL TEST FAILED (Exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
