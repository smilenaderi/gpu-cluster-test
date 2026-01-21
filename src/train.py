"""
GPU Cluster Acceptance Test - Distributed Training Script

This script validates GPU cluster configuration by running a distributed
PyTorch DDP training job. It tests:
- Multi-GPU communication via NCCL
- Distributed data loading
- Model synchronization across nodes
- GPU memory and compute capabilities

Usage:
    Single GPU:     python train.py --epochs 5
    Multi-GPU:      torchrun --nproc_per_node=8 train.py --epochs 5
    Multi-Node:     torchrun --nnodes=2 --nproc_per_node=8 train.py --epochs 5
    CPU Test:       python train.py --dry-run
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """
    Initialize the distributed process group for multi-GPU training.
    
    Returns:
        int or str: Local rank (GPU ID) if distributed, "cpu" otherwise
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        print(f"[Init] Rank {rank}/{world_size} (Local {local_rank}) initialized.")
        return local_rank
    else:
        print("[Init] Non-distributed mode (CPU/Single GPU dry-run).")
        return "cpu"


class SyntheticDataset(Dataset):
    """
    Generates synthetic image data for testing GPU throughput.
    
    This avoids network dependencies and ensures consistent testing
    across different cluster environments.
    
    Args:
        size (int): Number of samples to generate
    """
    def __init__(self, size=1000):
        self.data = torch.randn(size, 3, 224, 224)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def main():
    """Main training loop for cluster acceptance testing."""
    parser = argparse.ArgumentParser(
        description="GPU Cluster Acceptance Test - Distributed Training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=2,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size per GPU (default: 32)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Run on CPU for CI/testing without GPU"
    )
    args = parser.parse_args()


    # 1. Setup distributed environment
    if args.dry_run:
        device = torch.device("cpu")
        is_distributed = False
        print("[Mode] Running in CPU dry-run mode")
    else:
        local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = torch.distributed.is_initialized()
        
        if is_distributed:
            print(f"[Mode] Distributed training on {torch.distributed.get_world_size()} GPUs")
        else:
            print("[Mode] Single GPU training")

    # 2. Initialize model (ResNet18 for lightweight but realistic testing)
    model = resnet18(num_classes=10).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    print(f"[Model] ResNet18 initialized on {device}")

    # 3. Setup data pipeline with synthetic data
    dataset = SyntheticDataset(size=2048)
    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        num_workers=0,  # Avoid multiprocessing issues in containers
        pin_memory=True if not args.dry_run else False
    )
    
    print(f"[Data] Loaded {len(dataset)} samples with batch size {args.batch_size}")

    # 4. Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 5. Training loop
    model.train()
    print(f"\n🚀 Starting training for {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)  # Shuffle data differently each epoch
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"✅ Epoch {epoch+1}/{args.epochs} complete - Avg Loss: {avg_loss:.4f}")

    # 6. Final validation
    if not is_distributed or torch.distributed.get_rank() == 0:
        print("\n🎉 GPU Cluster Acceptance Test Passed!")
        print(f"   - Device: {device}")
        print(f"   - Epochs: {args.epochs}")
        print(f"   - Final Loss: {avg_loss:.4f}")
        
        # Write success marker for automated testing
        try:
            with open("/tmp/acceptance_test_success", "w") as f:
                f.write(f"Training completed successfully\nDevice: {device}\nEpochs: {args.epochs}\n")
        except Exception as e:
            print(f"   - Note: Could not write success marker: {e}")
    
    # Cleanup distributed resources
    if is_distributed:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()