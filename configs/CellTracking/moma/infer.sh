#!/bin/bash
# ============================================================
# SelfMOTR — Cell Tracking — Inference launcher
#
# Usage:
#   bash configs/CellTracking/moma/infer.sh
#
# Override any YAML value on the command line:
#   bash configs/CellTracking/moma/infer.sh --split train
#   bash configs/CellTracking/moma/infer.sh --resume outputs/cell_moma/checkpoint0009.pth
#   bash configs/CellTracking/moma/infer.sh --score_threshold 0.4 --miss_tolerance 15
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/infer.yaml"

# Read output_dir from YAML so we can mkdir before inference starts
OUTPUT_DIR=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('output_dir', 'outputs/cell_eval'))
")
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

echo "[infer.sh] Config:     ${CONFIG}"
echo "[infer.sh] Output dir: ${OUTPUT_DIR}"

python3 eval_cell.py --config "${CONFIG}" "$@"
