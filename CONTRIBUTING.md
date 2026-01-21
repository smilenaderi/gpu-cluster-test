# Contributing Guide

Thank you for your interest in contributing to the GPU Cluster Acceptance Test project!

## Development Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- Docker (for container testing)
- Access to GPU resources (for full testing)

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```

3. Run tests locally:
   ```bash
   # CPU dry-run (no GPU required)
   python src/train.py --dry-run --epochs 2
   
   # Single GPU test
   python src/train.py --epochs 2
   
   # Multi-GPU test (if available)
   torchrun --nproc_per_node=2 src/train.py --epochs 2
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Add comments for complex logic

## Testing

Before submitting changes:

1. Test on CPU with `--dry-run` flag
2. Test on single GPU if available
3. Test multi-GPU setup if available
4. Verify container builds successfully:
   ```bash
   docker build -t gpu-cluster-test:dev .
   ```

## Submitting Changes

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request with:
   - Clear description of changes
   - Test results
   - Any relevant documentation updates

## Common Tasks

### Adding New Features

- Keep backward compatibility
- Update documentation
- Add appropriate error handling
- Test in multiple environments

### Fixing Bugs

- Include reproduction steps
- Add test case if applicable
- Document the fix in commit message

### Improving Documentation

- Keep examples up-to-date
- Use clear, concise language
- Include code examples where helpful

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions
