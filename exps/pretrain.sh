#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=1

BATCH_SIZE=32
EPOCHS=20
WARMUP_EPOCHS=1
BASE_LEARNING_RATE=1e-4
WEIGHT_DECAY=0.0001
OPTIMIZER="adamw"

MODULE="CLIPBaseline"
VISUAL_ENCODER="clip_vit"
TEXT_ENCODER="clip_text"
JSON_LIST="2level_multi_disease_list"
# JSON_LIST="MIDRC_list"

DATA_DIR="/MGLL/Dataset/MLLM-Fundus"
OUTPUT_DIR="/MGLL/Snapshots/test"
LLAMA_PATH="/MGLL/ModelWeights/"
# RESUME=""

python -u -m torch.distributed.launch --master_port=1111 --nproc_per_node=1 --use_env \
    main_pretrain.py \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --warmup_epochs $WARMUP_EPOCHS \
    --blr $BASE_LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --module $MODULE \
    --visual_encoder $VISUAL_ENCODER \
    --text_encoder $TEXT_ENCODER \
    --json_list $JSON_LIST \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --llama_path $LLAMA_PATH \
#     --resume $RESUME \
