#!/bin/bash
# ======== path ========
MODEL_PATH="path/to/model"
DATA_PATH="/mnt/pfs/zitao_team/lixueyi/Projects/TtT/data_config/data_config_TtT.json"
OUTPUT_DIR="./output"
DEEPSPEED_CONFIG="./ds_zero_3.json"

# ======== WandB ========
export WANDB_API_KEY="api-key"
export WANDB_PROJECT="audio_model_train"
export WANDB_RUN_NAME="Train_TtT"

# ======== distributed training ========
GPUS_PER_NODE=8

is_master=${MASTER-"0"}
if [[ $is_master -eq 1 ]]; then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $HOSTNAME --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
fi

# ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# ======== start training ========
wandb login $WANDB_API_KEY

torchrun $DISTRIBUTED_ARGS train_TtT.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 10 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --model_max_length 2048 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.01 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --lazy_preprocess True \
    --logging_steps 1 \
    --use_lora False \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --report_to wandb \
    --unmasked_audio_prob 0.3 \
    --prefix_preservation_ratio 0.3 \
    --quad_span_truncation_prob 0.5

echo "===== Finished! ====="
echo "Model saved to: ${OUTPUT_DIR}"
