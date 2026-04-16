#!/bin/bash
# ============================================================
# SelfMOTR — Cell Tracking — Visualisation launcher
#
# Generates one MP4 per sequence showing tracked cells and
# detected mitotic events (yellow border + "M" label).
#
# Usage:
#   bash configs/CellTracking/moma/viz.sh
#
# Override YAML values on the command line:
#   bash configs/CellTracking/moma/viz.sh --div_threshold 0.4
#   bash configs/CellTracking/moma/viz.sh --fps 15 --scale 10
#   bash configs/CellTracking/moma/viz.sh --resume outputs/cell_moma/checkpoint0009.pth
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/infer.yaml"

cd "${REPO_DIR}"

echo "[viz.sh] Config: ${CONFIG}"
echo "[viz.sh] Videos will be written to: outputs/cell_moma_videos/"

python3 viz_cell.py --config "${CONFIG}" --output_dir outputs/cell_moma_videos  "$@"
