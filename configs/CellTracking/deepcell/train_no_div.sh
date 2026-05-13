#!/bin/bash
# ============================================================
# SelfMOTR — DeepCell — No-Division Training Launcher
#
# Usage (single GPU):
#   bash configs/CellTracking/deepcell/train_no_div.sh
#
# Usage (multi-GPU, e.g. 2):
#   NUM_GPUS=2 bash configs/CellTracking/deepcell/train_no_div.sh
#
# Override anything on the command line:
#   bash configs/CellTracking/deepcell/train_no_div.sh --epochs 50
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/train_no_div.yaml"
NUM_GPUS=${NUM_GPUS:-1}

OUTPUT_DIR=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('output_dir', 'outputs/cell_deepcell_no_div'))
")
mkdir -p "${OUTPUT_DIR}"

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

cd "${REPO_DIR}"

# CUDA_LAUNCH_BLOCKING makes CUDA kernels run synchronously so any
# device-side assertion is surfaced immediately with a Python traceback
# rather than silently deadlocking one rank in a DDP all-reduce.
export CUDA_LAUNCH_BLOCKING=1

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "[train_no_div.sh] Launching on ${NUM_GPUS} GPUs (port ${MASTER_PORT})"
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        train_cell_no_div.py --config "${CONFIG}" "$@"
else
    echo "[train_no_div.sh] Launching on 1 GPU"
    python3 train_cell_no_div.py --config "${CONFIG}" "$@"
fi
