#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate chatflash

export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export TOKENIZERS_PARALLELISM=false

# Prevent accelerate from misreading the outer srun shell's single-task SLURM env
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_STEP_NUM_TASKS \
      SLURM_TASKS_PER_NODE SLURM_STEP_TASKS_PER_NODE SLURM_PROCID SLURM_LOCALID \
      SLURM_GTIDS SLURM_JOB_NUM_NODES SLURM_NNODES SLURM_STEP_NODELIST \
      SLURM_NODELIST SLURM_JOB_NODELIST

DATA_VERSION="data/stage3_surgvqa_all.yaml"
DATA_VERSION_CLEAN=$(basename "$DATA_VERSION")
VISION_MODEL_VERSION="vjepa2-large"
VISION_MODEL_VERSION_CLEAN="vjepa2-large"

# Path to your downloaded stage2 checkpoint
LLM_VERSION="/iopsstor/scratch/cscs/lpanta32/checkpoints/stage2-UMT-Qwen2_5_7B_1m-tome16_mlp_jepa"
LLM_VERSION_CLEAN="Qwen2_7B"

mm_projector_type=tome16_mlp_hd64
PROMPT_VERSION="qwen_2"
MID_RUN_NAME=stage3-surgvqa_overlay_jepa2
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

mkdir -p ./output_logs/stage3-video_sft
mkdir -p /iopsstor/scratch/cscs/lpanta32/checkpoints/stage3-video_sft/${MID_RUN_NAME}

# ---- GPU / distributed config ----
NUM_GPU=4
MASTER_PORT=29518
MASTER_ADDR=127.0.0.1

accelerate launch \
    --num_processes=${NUM_GPU} \
    --num_machines=1 \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    --mixed_precision=bf16 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero1.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter" \
    --mm_vision_tower_lr=7e-7 \
    --mm_vision_select_layer -2 \
    --mm_projector_type ${mm_projector_type} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_nopad \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_nopad \
    --mm_newline_position nothing \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir /iopsstor/scratch/cscs/lpanta32/checkpoints/stage3-video_sft/${MID_RUN_NAME} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 24000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --torch_compile False \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last False \
    --frames_upbound 800 \
    --frames_lowbound 32 \
    --time_msg short \
    --local_num_frames 4 \
    --vision_encode_type video_image \
    --sample_type dynamic_fps1 \
    --mm_local_num_frames 4 \
    --attn_implementation sdpa \
    --verbose_logging True 2>&1 | tee ./output_logs/stage3-video_sft/${MID_RUN_NAME}.log