"""
NCCL Collective Operations Test

Tests all_reduce, all_gather, and sum operations across all GPUs
to verify NCCL communication is working correctly.

Usage:
    Multi-GPU:   torchrun --nproc_per_node=8 nccl_test.py
    Multi-Node:  torchrun --nnodes=2 --nproc_per_node=8 nccl_test.py
"""

import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed process group."""
    if "RANK" not in os.environ:
        print("[Error] Must run with torchrun for distributed testing")
        return None, None, None
    
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def test_all_reduce(rank, world_size, device):
    """
    Test all_reduce SUM operation.
    Each rank contributes its rank value, result should be sum of all ranks.
    """
    tensor = torch.tensor([float(rank)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    expected = sum(range(world_size))  # 0 + 1 + 2 + ... + (world_size-1)
    actual = tensor.item()
    passed = abs(actual - expected) < 1e-6
    
    return passed, expected, actual


def test_all_reduce_tensor(rank, world_size, device):
    """
    Test all_reduce SUM with larger tensor.
    Each rank contributes tensor filled with rank value.
    """
    size = 1024
    tensor = torch.full((size,), float(rank), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    expected = sum(range(world_size))
    actual = tensor[0].item()
    all_same = torch.all(tensor == expected).item()
    passed = all_same and abs(actual - expected) < 1e-6
    
    return passed, expected, actual


def test_all_gather(rank, world_size, device):
    """
    Test all_gather operation.
    Each rank contributes its rank value, result should be [0, 1, 2, ..., world_size-1].
    """
    tensor = torch.tensor([float(rank)], device=device)
    gathered = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    result = [t.item() for t in gathered]
    expected = list(range(world_size))
    passed = result == expected
    
    return passed, expected, result


def test_all_gather_tensor(rank, world_size, device):
    """
    Test all_gather with larger tensors.
    """
    size = 256
    tensor = torch.full((size,), float(rank), device=device)
    gathered = [torch.zeros(size, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    passed = True
    for i, t in enumerate(gathered):
        if not torch.all(t == float(i)).item():
            passed = False
            break
    
    return passed, "each tensor filled with rank", "verified" if passed else "mismatch"


def test_broadcast(rank, world_size, device):
    """
    Test broadcast from rank 0.
    """
    if rank == 0:
        tensor = torch.tensor([42.0, 43.0, 44.0], device=device)
    else:
        tensor = torch.zeros(3, device=device)
    
    dist.broadcast(tensor, src=0)
    
    expected = [42.0, 43.0, 44.0]
    actual = tensor.tolist()
    passed = actual == expected
    
    return passed, expected, actual


def test_reduce_scatter(rank, world_size, device):
    """
    Test reduce_scatter operation.
    """
    # Each rank has world_size chunks, reduce_scatter sums and scatters
    input_tensor = torch.full((world_size,), float(rank), device=device)
    output_tensor = torch.zeros(1, device=device)
    
    input_list = list(input_tensor.chunk(world_size))
    dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
    
    # Each output[i] should be sum of all ranks = 0+1+2+...+(world_size-1)
    expected = sum(range(world_size))
    actual = output_tensor.item()
    passed = abs(actual - expected) < 1e-6
    
    return passed, expected, actual


def main():
    rank, local_rank, world_size = setup_distributed()
    if rank is None:
        return 1
    
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("=" * 60)
        print("NCCL Collective Operations Test")
        print("=" * 60)
        print(f"World Size: {world_size} GPUs")
        print(f"Backend: NCCL")
        print("=" * 60)
        print()
    
    dist.barrier()
    
    tests = [
        ("all_reduce (scalar)", test_all_reduce),
        ("all_reduce (tensor)", test_all_reduce_tensor),
        ("all_gather (scalar)", test_all_gather),
        ("all_gather (tensor)", test_all_gather_tensor),
        ("broadcast", test_broadcast),
        ("reduce_scatter", test_reduce_scatter),
    ]
    
    all_passed = True
    results = []
    
    for name, test_fn in tests:
        try:
            passed, expected, actual = test_fn(rank, world_size, device)
            results.append((name, passed, expected, actual))
            if not passed:
                all_passed = False
        except Exception as e:
            results.append((name, False, "N/A", str(e)))
            all_passed = False
        
        dist.barrier()
    
    # Only rank 0 prints results
    if rank == 0:
        for name, passed, expected, actual in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} | {name}")
            print(f"       Expected: {expected}")
            print(f"       Actual:   {actual}")
            print()
        
        print("=" * 60)
        if all_passed:
            print("🎉 All NCCL collective operations passed!")
        else:
            print("❌ Some tests failed. Check NCCL configuration.")
        print("=" * 60)
    
    dist.destroy_process_group()
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
