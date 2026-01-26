#!/bin/bash
#
# NCCL Performance Tests (all_reduce_perf, all_gather_perf, etc.)
#
# Runs official NCCL performance benchmarks to measure bandwidth and latency
# across all GPUs. These are the industry-standard tests for validating
# multi-GPU communication performance.
#
# Usage: 
#   sbatch scripts/nccl_perf_test.sh
#   sbatch --nodes=4 --gpus-per-node=4 scripts/nccl_perf_test.sh
#
# Override defaults with environment variables:
#   NODES=4 GPUS_PER_NODE=4 TEST_TYPE=all_reduce sbatch scripts/nccl_perf_test.sh
#

#SBATCH --job-name=nccl-perf
#SBATCH --partition=main
#SBATCH --output=logs/nccl_perf_%j.out
#SBATCH --error=logs/nccl_perf_%j.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00

set -euo pipefail

# Configuration
NODES=${NODES:-$SLURM_JOB_NUM_NODES}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
TEST_TYPE=${TEST_TYPE:-all_reduce}  # all_reduce, all_gather, broadcast, reduce_scatter, alltoall, all
MIN_BYTES=${MIN_BYTES:-8}
MAX_BYTES=${MAX_BYTES:-8G}
STEP_FACTOR=${STEP_FACTOR:-2}
ITERS=${ITERS:-20}
WARMUP_ITERS=${WARMUP_ITERS:-5}

# Validation
if [ "$NODES" -lt 1 ]; then
    echo "❌ Error: NODES must be >= 1 (got: $NODES)"
    exit 1
fi

if [ "$GPUS_PER_NODE" -lt 1 ]; then
    echo "❌ Error: GPUS_PER_NODE must be >= 1 (got: $GPUS_PER_NODE)"
    exit 1
fi

# Calculate total GPUs
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

# Get master node for multi-node coordination
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

echo "=========================================="
echo "NCCL Performance Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Test Type: $TEST_TYPE"
echo "Size Range: $MIN_BYTES - $MAX_BYTES"
echo "Iterations: $ITERS (warmup: $WARMUP_ITERS)"
echo "Master Node: $MASTER_NODE"
echo "=========================================="
echo ""

# Function to run a specific NCCL test
run_nccl_test() {
    local test_name=$1
    local test_cmd="${test_name}_perf"
    
    echo ""
    echo "=========================================="
    echo "Running: $test_name"
    echo "=========================================="
    echo ""
    
    # Check if test binary exists
    if ! command -v $test_cmd &> /dev/null; then
        echo "❌ Error: $test_cmd not found in PATH"
        echo "   Make sure NCCL tests are installed"
        return 1
    fi
    
    echo "Using test binary: $(which $test_cmd)"
    echo ""
    
    # Run the test across all nodes and GPUs
    # Note: NCCL tests handle multi-GPU internally with -g flag
    # We need one MPI rank per node, and each rank will use -g GPUs
    srun --ntasks=$NODES \
         --ntasks-per-node=1 \
         $test_cmd \
         -b $MIN_BYTES \
         -e $MAX_BYTES \
         -f $STEP_FACTOR \
         -g $GPUS_PER_NODE \
         -n $ITERS \
         -w $WARMUP_ITERS \
         -c 1 \
         -a 1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ $test_name completed successfully"
    else
        echo ""
        echo "❌ $test_name failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Run tests based on TEST_TYPE
EXIT_CODE=0

case $TEST_TYPE in
    all_reduce)
        run_nccl_test "all_reduce" || EXIT_CODE=$?
        ;;
    all_gather)
        run_nccl_test "all_gather" || EXIT_CODE=$?
        ;;
    broadcast)
        run_nccl_test "broadcast" || EXIT_CODE=$?
        ;;
    reduce_scatter)
        run_nccl_test "reduce_scatter" || EXIT_CODE=$?
        ;;
    alltoall)
        run_nccl_test "alltoall" || EXIT_CODE=$?
        ;;
    all)
        echo "Running all NCCL performance tests..."
        run_nccl_test "all_reduce" || EXIT_CODE=$?
        run_nccl_test "all_gather" || EXIT_CODE=$?
        run_nccl_test "broadcast" || EXIT_CODE=$?
        run_nccl_test "reduce_scatter" || EXIT_CODE=$?
        run_nccl_test "alltoall" || EXIT_CODE=$?
        ;;
    *)
        echo "❌ Error: Unknown test type: $TEST_TYPE"
        echo "   Valid options: all_reduce, all_gather, broadcast, reduce_scatter, alltoall, all"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ NCCL PERFORMANCE TEST PASSED"
    echo "   Test: $TEST_TYPE"
    echo "   GPUs: $TOTAL_GPUS ($NODES nodes × $GPUS_PER_NODE GPUs)"
    echo ""
    echo "Results show bandwidth (GB/s) and latency (us) for different message sizes."
    echo "Higher bandwidth and lower latency indicate better performance."
else
    echo "❌ NCCL PERFORMANCE TEST FAILED"
    echo ""
    echo "Common issues:"
    echo "  - NCCL tests not installed (install nccl-tests package)"
    echo "  - Network connectivity problems"
    echo "  - InfiniBand/RoCE configuration issues"
    echo "  - Insufficient GPU memory for large message sizes"
    echo ""
    echo "Troubleshooting:"
    echo "  - Check if tests exist: which all_reduce_perf"
    echo "  - Test with smaller size: MAX_BYTES=128M"
    echo "  - Test single node first: --nodes=1"
fi
echo "=========================================="

exit $EXIT_CODE
