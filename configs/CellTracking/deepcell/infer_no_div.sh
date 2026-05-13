#!/bin/bash
# ============================================================
# SelfMOTR — DeepCell — No-Division Inference Launcher
#
# Writes CTC results (mask tifs + res_track.txt, all parent=0)
# to the output_dir defined in infer_no_div.yaml.
#
# Usage:
#   bash configs/CellTracking/deepcell/infer_no_div.sh
#
# Override anything:
#   bash configs/CellTracking/deepcell/infer_no_div.sh \
#       --resume outputs/cell_deepcell_no_div/checkpoint0029.pth
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/infer_no_div.yaml"

OUTPUT_DIR=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('output_dir', 'outputs/cell_deepcell_no_div_eval'))
")
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

echo "[infer_no_div.sh] Config:     ${CONFIG}"
echo "[infer_no_div.sh] Output dir: ${OUTPUT_DIR}"

python3 eval_cell_no_div.py --config "${CONFIG}" "$@"
