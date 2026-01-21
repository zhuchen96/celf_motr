#!/bin/bash

# Activate conda environment
source /work/scratch/guelhan/miniconda3/etc/profile.d/conda.sh
conda activate pilot3

# Navigate to project directory
cd /work/scratch/guelhan/Publikation/SelfMOTR2

#layer=$1
layer=5

# Create directories for evaluation outputs and logs
CHECKPOINT=/work/scratch/guelhan/Publikation/checkpoints/selfmotr_dance.pth
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
EXP_DIR=/work/scratch/guelhan/publ_test/submit_layer_ablation/${TIMESTAMP}
mkdir -p ${EXP_DIR}
LOG_DIR=${EXP_DIR}/logs
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/submit.log

python3 submit_dance_layer_ablation.py \
    --meta_arch motrv2_self \
    --dataset_file e2e_dance \
    --mot_path /images/SegmentationDistillation/data \
    --epoch 40 \
    --with_box_refine \
    --lr_drop 20 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 10 18 30 \
    --sampler_lengths 2 3 4 5 \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIMv2' \
    --num_queries 10 \
    --num_queries_detect 300 \
    --resume ${CHECKPOINT} \
    --exp_name tracker \
    --proposal_threshold 0.05 \
    --score_threshold 0.5 \
    --miss_tolerance 20 \
    --layer $layer \
    > ${LOG_FILE} 2>&1