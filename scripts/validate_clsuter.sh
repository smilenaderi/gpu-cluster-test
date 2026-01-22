#!/bin/bash
#
# GPU Cluster Test - Custom Container
#
# Uses custom-built Docker image from GitHub Container Registry
#
# Usage: sbatch scripts/validate_clsuter.sh
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

# Get master node IP
MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "=========================================="
echo "GPU Cluster Test - Custom Container"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Master Node: $MASTER_IP:$MASTER_PORT"
echo "Container: /shared/gpu-cluster-test/images/smilenaderi+gpu-cluster-test+main.sqsh"
echo "=========================================="
echo ""

# Launch distributed training on all nodes using srun
srun --container-image=/shared/gpu-cluster-test/images/smilenaderi+gpu-cluster-test+main.sqsh \
     --container-mounts=/shared/gpu-cluster-test:/workspace/project \
     bash -c "torchrun \
         --nnodes=2 \
         --nproc_per_node=8 \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
         /workspace/project/src/train.py \
         --epochs 5 \
         --batch-size 64"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST PASSED"
    echo "Cluster is properly configured for distributed training"
else
    echo "❌ TEST FAILED (Exit code: $EXIT_CODE)"
    echo "Check error logs for details"
fi
echo "=========================================="

exit $EXIT_CODE
