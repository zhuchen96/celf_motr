#!/bin/bash
# ============================================================
# SelfMOTR — DeepCell — Inference launcher
#
# Writes CTC-format results (mask tifs + res_track.txt) to
# the output_dir defined in infer.yaml.
#
# Usage:
#   bash configs/CellTracking/deepcell/infer.sh
#
# Override anything on the command line:
#   bash configs/CellTracking/deepcell/infer.sh --resume outputs/cell_deepcell/checkpoint0029.pth
#   bash configs/CellTracking/deepcell/infer.sh --div_threshold 0.3
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/infer.yaml"

OUTPUT_DIR=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('output_dir', 'outputs/cell_deepcell_eval'))
")
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

echo "[infer.sh] Config:     ${CONFIG}"
echo "[infer.sh] Output dir: ${OUTPUT_DIR}"

python3 eval_cell.py --config "${CONFIG}" "$@"
