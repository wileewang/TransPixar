#!/bin/bash
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="3"

DATA_ROOT="/hpc2hdd/home/lwang592/projects/finetrainers/training/data/video-matte-240k-rgb-prepared-f37"
MODEL="genmo/mochi-1-preview"
OUTPUT_PATH="mochi-rgba-lora-f37"

cmd="CUDA_VISIBLE_DEVICES=$GPU_IDS python train.py \
  --pretrained_model_name_or_path $MODEL \
  --cast_dit \
  --data_root $DATA_ROOT \
  --seed 42 \
  --output_dir $OUTPUT_PATH \
  --train_batch_size 2 \
  --dataloader_num_workers 4 \
  --pin_memory \
  --caption_dropout 0.0 \
  --max_train_steps 5000 \
  --gradient_checkpointing \
  --enable_slicing \
  --enable_tiling \
  --enable_model_cpu_offload \
  --optimizer adamw \
  --allow_tf32"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"