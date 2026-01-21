#!/bin/bash
# Unset CUDA_PATH and related environment variables when conda environment is deactivated
# This prevents using CUDA_PATH from a deactivated environment

unset CUDA_PATH

# Note: PATH is automatically restored by conda, but we unset the include paths
# if they were only set by this environment
unset CPATH
unset CPLUS_INCLUDE_PATH
unset C_INCLUDE_PATH
