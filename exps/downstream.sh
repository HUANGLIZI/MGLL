#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=1
BATCH_SIZE=32
EPOCHS=50
WARMUP_EPOCHS=1
BASE_LEARNING_RATE=1e-3
WEIGHT_DECAY=0.001
OPTIMIZER="adamw"

VISUAL_ENCODER="clip_vit"
TEXT_ENCODER="clip_text"
IS_PROBE="False"
DEFAULT_PRETRAIN='openai'
TEST_ONLY="False"
CSV_NAME="GPT_REFUGE1200"
TASK="multiclass"
NUM_CLASSES=2

DATA_DIR="/MGLL/Dataset/MLLM-Fundus"
OUTPUT_DIR="/MGLL/Snapshots/test"

PRETRAINED="/MGLL/Snapshots/pretrain.pth"
# RESUME=""

python -u -m torch.distributed.launch --master_port=1117 --nproc_per_node=1 --use_env \
    main_finetune.py \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --warmup_epochs $WARMUP_EPOCHS \
    --blr $BASE_LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --visual_encoder $VISUAL_ENCODER \
    --text_encoder $TEXT_ENCODER \
    --is_probe $IS_PROBE \
    --test_only $TEST_ONLY \
    --csv_name $CSV_NAME \
    --task $TASK \
    --num_classes $NUM_CLASSES \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --default_pretrain $DEFAULT_PRETRAIN \
    --pretrained $PRETRAINED \
    # --resume $RESUME \

