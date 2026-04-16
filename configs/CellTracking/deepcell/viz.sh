#!/bin/bash
# ============================================================
# SelfMOTR — DeepCell — Visualisation launcher
#
# Generates one AVI per sequence showing tracked cells and
# detected mitotic events (yellow border + "M" label).
#
# Usage:
#   bash configs/CellTracking/deepcell/viz.sh
#
# Override anything on the command line:
#   bash configs/CellTracking/deepcell/viz.sh --div_threshold 0.4
#   bash configs/CellTracking/deepcell/viz.sh --fps 15 --scale 2
#   bash configs/CellTracking/deepcell/viz.sh --resume outputs/cell_deepcell/checkpoint0029.pth
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/infer.yaml"

cd "${REPO_DIR}"

echo "[viz.sh] Config: ${CONFIG}"
echo "[viz.sh] Videos will be written to: outputs/cell_deepcell_videos/"

python3 viz_cell.py --config "${CONFIG}" --output_dir outputs/cell_deepcell_videos "$@"
