#!/bin/bash
# Automatically set CUDA_PATH when conda environment is activated
# This file is sourced by conda when the environment is activated

if [ -n "$CONDA_PREFIX" ]; then
    # Check if CUDA headers exist in the conda environment
    if [ -f "$CONDA_PREFIX/targets/x86_64-linux/include/cuda_fp16.h" ]; then
        # Use the targets directory as CUDA_PATH (standard conda CUDA layout)
        export CUDA_PATH="$CONDA_PREFIX/targets/x86_64-linux"
    elif [ -f "$CONDA_PREFIX/include/cuda_fp16.h" ]; then
        # Fallback: use conda prefix directly
        export CUDA_PATH="$CONDA_PREFIX"
    fi
    
    # Add nvvm/bin to PATH for CUDA compiler tools
    export PATH="$CONDA_PREFIX/nvvm/bin:$PATH"
    
    # Add conda include directories to compiler search paths
    export CPATH="$CONDA_PREFIX/include:${CPATH}"
    export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH}"
    export C_INCLUDE_PATH="$CONDA_PREFIX/include:${C_INCLUDE_PATH}"
    
    # Optionally print confirmation (comment out if too verbose)
    # echo "✓ CUDA_PATH set to: $CUDA_PATH"
fi
