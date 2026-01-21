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
NUM_GPUS=8  # Adjust based on request_gpus in Condor TBI file

# Dynamically find a free port to avoid conflicts
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# Run training
python3 -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    --use_env train_mot17.py \
    --meta_arch motrv2_self \
    --dataset_file e2e_joint \
    --epoch 240 \
    --save_period 10 \
    --with_box_refine \
    --shared_decoder \
    --lr_drop_ 80 \
    --lr_drop_epochs 80 160 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${PRETRAIN} \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 20 40 \
    --sampler_lengths 2 3 4 \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIMv2' \
    --num_queries 10 \
    --num_queries_detect 300 \
    --mot_path /images/SegmentationDistillation/data \
    --accum_iter 1 \
    --score_threshold 0.05 \
    --lambda_detect 0.5 \
    --data_txt_path_train ./datasets/data_path/joint.train \
    --data_txt_path_val ./datasets/data_path/mot17.train \
    --use_checkpoint \
    > ${LOG_FILE} 2>&1
