#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate selfmotr

# Navigate to project directory
cd /work/scratch/mededovic/selfmotr/SelfMOTR_Pilot

# Set paths
PRETRAIN=/images/SegmentationDistillation/checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth

BASE_EXP_DIR=/work/scratch/mededovic/selfmotr/SelfMOTR_Pilot/logs
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
EXP_DIR=${BASE_EXP_DIR}/${TIMESTAMP}
mkdir -p ${EXP_DIR}

BASE_LOG_DIR=/work/scratch/mededovic/selfmotr/SelfMOTR_Pilot/exps
LOG_DIR=${BASE_LOG_DIR}/${TIMESTAMP}/logs
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/train_${TIMESTAMP}.log


# Number of GPUs to use
NUM_GPUS=4  # Adjust based on request_gpus in Condor TBI file

# Dynamically find a free port to avoid conflicts
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# Run training
python3 -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    --use_env train_sportsMOT.py \
    --meta_arch motrv2_self \
    --dataset_file e2e_sportsmot_v2 \
    --epoch 20 \
    --save_period 1 \
    --with_box_refine \
    --lr_drop 10 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${PRETRAIN} \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_lengths 4 \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIMv2' \
    --num_queries 10 \
    --num_queries_detect 300 \
    --mot_path /images/TransformerTracking \
    --accum_iter 2 \
    --score_threshold 0.05 \
    --lambda_detect 0.5 \
    --append_crowd \
    --query_denoise 0.05 \
    --use_checkpoint \
    > ${LOG_FILE} 2>&1
