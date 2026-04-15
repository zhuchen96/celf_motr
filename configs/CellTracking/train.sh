#!/bin/bash
# ============================================================
# SelfMOTR — Cell Tracking — Training launcher
#
# Usage (single GPU):
#   bash configs/CellTracking/train.sh
#
# Usage (multi-GPU, e.g. 2):
#   NUM_GPUS=2 bash configs/CellTracking/train.sh
#
# Any extra arguments are forwarded and override the YAML:
#   bash configs/CellTracking/train.sh --epochs 50
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/train.yaml"
NUM_GPUS=${NUM_GPUS:-1}

# Read output_dir from YAML so we can mkdir before training starts
OUTPUT_DIR=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('output_dir', 'outputs/cell'))
")
mkdir -p "${OUTPUT_DIR}"

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

cd "${REPO_DIR}"

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "[train.sh] Launching on ${NUM_GPUS} GPUs (port ${MASTER_PORT})"
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        train_cell.py --config "${CONFIG}" "$@"
else
    echo "[train.sh] Launching on 1 GPU"
    python3 train_cell.py --config "${CONFIG}" "$@"
fi
