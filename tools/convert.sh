#!/bin/bash
source /work/scratch/guelhan/miniconda3/etc/profile.d/conda.sh
conda activate motr_eval

cd /work/scratch/guelhan/MOTR/util_fabian

python3 convert_txt_to_json.py