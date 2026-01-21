#!/bin/bash

# Activate conda environment
source /work/scratch/guelhan/miniconda3/etc/profile.d/conda.sh
conda activate pilot3

cd /work/scratch/guelhan/Publikation/SelfMOTR2/tools

#Parent directory containing detection output folders for several checkpoints
BASE_DIR=

for CHECKPOINT_DIR in ${BASE_DIR}/*/; do
    TRACKER_DIR=${CHECKPOINT_DIR}tracker
    LOG_FILE=${CHECKPOINT_DIR}mAP.log

    echo "Evaluating: ${TRACKER_DIR}"

    python3 coco_eval.py --det_root ${TRACKER_DIR} > ${LOG_FILE} 2>&1

done