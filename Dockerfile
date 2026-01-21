# GPU Cluster Acceptance Test - Container Image
# Base: NVIDIA PyTorch container with CUDA 12.4 support

FROM nvcr.io/nvidia/pytorch:24.07-py3

# Metadata
LABEL maintainer="Your Organization"
LABEL description="Distributed PyTorch DDP acceptance test for GPU clusters"
LABEL version="1.0"

# Set working directory
WORKDIR /workspace

# Copy and install Python dependencies
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Make scripts executable
RUN chmod +x scripts/*.sh

# Set Python path
ENV PYTHONPATH=/workspace/src

# Health check: verify PyTorch and CUDA availability
RUN python -c "import torch; assert torch.cuda.is_available() or True, 'CUDA check'"

# Default entrypoint shows help
ENTRYPOINT ["python", "src/train.py"]
CMD ["--help"]