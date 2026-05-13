#!/bin/bash
# ============================================================
# SelfMOTR — DeepCell — Division Score Diagnostic Launcher
#
# Runs the tracker on the val set and reports how the confidence
# score evolves for cells around GT division events.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash configs/CellTracking/deepcell/diag_division_scores.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/diag_division_scores.yaml"

cd "${REPO_DIR}"

echo "[diag_division_scores.sh] Config: ${CONFIG}"

python3 diag_division_scores.py --config "${CONFIG}" "$@"
