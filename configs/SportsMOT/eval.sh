#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate selfmotr

# Navigate to project directory
cd /work/scratch/mededovic/selfmotr/SelfMOTR_Pilot

# Create directories for evaluation outputs and logs
CHECKPOINT=/work/scratch/mededovic/selfmotr/SelfMOTR_Pilot/logs/20260118_003243/checkpoint0018.pth

BASE_EXP_DIR=/work/scratch/mededovic/selfmotr/SelfMOTR_Pilot/logs
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
EXP_DIR=${BASE_EXP_DIR}/${TIMESTAMP}

mkdir -p ${EXP_DIR}
LOG_DIR=${EXP_DIR}/logs

mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/eval.log

python3 eval_sportsmot.py \
    --meta_arch motrv2_self \
    --dataset_file e2e_sportsmot \
    --mot_path /images/TransformerTracking \
    --epoch 40 \
    --with_box_refine \
    --shared_decoder \
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
    > ${LOG_FILE} 2>&1
