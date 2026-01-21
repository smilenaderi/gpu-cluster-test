#!/bin/bash
#
# GPU Cluster Acceptance Test - Slurm Batch Job
#
# This script submits a multi-node distributed training job to validate
# GPU cluster configuration, NCCL communication, and container setup.
#
# Usage: sbatch scripts/cluster_acceptance.sh
#

#SBATCH --job-name=gpu-acceptance-test
#SBATCH --partition=main
#SBATCH --output=/shared/smoke-test/gpu-cluster-test/logs/acceptance_%j.out
#SBATCH --error=/shared/smoke-test/gpu-cluster-test/logs/acceptance_%j.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00

# Container configuration
#SBATCH --container-image=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
#SBATCH --container-mounts=/shared/smoke-test/gpu-cluster-test:/workspace/project

# Environment variables for distributed training
export MASTER_ADDR=worker-0
export MASTER_PORT=29500
export PYTHONPATH=/workspace/project/src

echo "=========================================="
echo "GPU Cluster Acceptance Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Master Node: $MASTER_ADDR:$MASTER_PORT"
echo "=========================================="
echo ""

# Launch distributed training
/usr/bin/srun --container-env=MASTER_ADDR,MASTER_PORT,PYTHONPATH \
    bash -c "pip install -q --no-cache-dir torchvision --no-deps && \
    torchrun \
        --nnodes=2 \
        --nproc_per_node=8 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
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
