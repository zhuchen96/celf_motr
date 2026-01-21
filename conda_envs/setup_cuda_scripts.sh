#!/bin/bash
# Script to copy CUDA path activation scripts to the conda environment
# Run this after activating your conda environment (cuda128 or cuda130)

set -e

# Check if conda environment is activated
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No conda environment is activated."
    echo "Please activate your conda environment first:"
    echo "  conda activate cuda128"
    echo "  or"
    echo "  conda activate cuda130"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create activate.d and deactivate.d directories if they don't exist
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"

mkdir -p "$ACTIVATE_DIR"
mkdir -p "$DEACTIVATE_DIR"

# Copy the scripts
echo "Copying CUDA path scripts to $CONDA_PREFIX..."
cp "$SCRIPT_DIR/set_cuda_path.sh" "$ACTIVATE_DIR/"
cp "$SCRIPT_DIR/unset_cuda_path.sh" "$DEACTIVATE_DIR/"

# Make them executable
chmod +x "$ACTIVATE_DIR/set_cuda_path.sh"
chmod +x "$DEACTIVATE_DIR/unset_cuda_path.sh"

echo "✓ CUDA path scripts installed successfully!"
echo "  - Activation script: $ACTIVATE_DIR/set_cuda_path.sh"
echo "  - Deactivation script: $DEACTIVATE_DIR/unset_cuda_path.sh"
echo ""
echo "These scripts will automatically run when you activate/deactivate this conda environment."
