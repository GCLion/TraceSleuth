#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
#25678 $(shuf -i 20001-29999 -n 1)
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# Set visible devices ,2,3,4,5
export CUDA_VISIBLE_DEVICES="0,1"
NPROC_PER_NODE=2

# DeepSpeed configuration???
deepspeed=./scripts/zero3.json

# Model configuration
# llm=./ckpt/Qwen2.5-VL-7B-Instruct  # Replace with your actual model path
llm=./720-Qwen2.5-VL-7B-Instruct  # Replace with your actual model path

# Training hyperparameters
lr=2e-5
batch_size=4
grad_accum_steps=8

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
datasets="/train/myModel/VLM-R3-main/qwen_vl_finetune/dataset/data_train.json"
# "/train/myModel/VLM-R3-main/qwen_vl_finetune/dataset/train_CASIAv2_autosplice_test.json"
# "/train/myModel/VLM-R3-main/qwen_vl_finetune/dataset/ai_sft_data.json"
# "/train/myModel/VLM-R3-main/qwen_vl_finetune/dataset/vlir_sft_12k.json"  # Replace with your actual dataset path path/to/your/dataset

# Output configuration
run_name="qwen2.5vl-sft"
output_dir=./output_sft 

# Training arguments
# --num_train_epochs 4 \
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 15 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 1605632 \
    --min_pixels 3136 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to none"

# Launch training wandb
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
