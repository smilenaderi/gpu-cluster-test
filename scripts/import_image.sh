#!/bin/bash
#
# Import custom Docker image from GHCR using enroot
#
# Usage: ./scripts/import_image.sh
#

set -e

IMAGE_URL="docker://ghcr.io#smilenaderi/gpu-cluster-test:main"
IMAGE_DIR="images"
IMAGE_FILE="$IMAGE_DIR/smilenaderi+gpu-cluster-test+main.sqsh"

# Create images directory if it doesn't exist
mkdir -p "$IMAGE_DIR"

echo "=========================================="
echo "Importing Docker Image with Enroot"
echo "=========================================="
echo "Source: ghcr.io/smilenaderi/gpu-cluster-test:main"
echo "Target: $IMAGE_FILE"
echo "=========================================="
echo ""

# Import the image with custom output path
enroot import -o "$IMAGE_FILE" "$IMAGE_URL"

echo ""
echo "✅ Image imported successfully!"
echo ""
echo "Image location: /shared/gpu-cluster-test/$IMAGE_FILE"
echo ""
echo "To use in Slurm job:"
echo "  sbatch gpu-cluster-test/scripts/validate_clsuter.sh"
