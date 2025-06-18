#!/bin/bash

export WANDB_PROJECT=SeLeRoSa

for i in {1..5}; do
    export WANDB_NAME="ro_bert_cased_frozen_$i"
    MODEL_DIR="./results/ro_bert_cased_frozen_ft/$i"
    RESULTS_DIR="./results/inference/ro_bert_cased_frozen_ft/$i"
    mkdir -p "$RESULTS_DIR"

    python satire/experiments/baselines/bert_finetune.py \
        --data_file data/csv/selerosa.csv \
        --model_checkpoint "dumitrescustefan/bert-base-romanian-cased-v1" \
        --freeze_bert \
        --learning_rate 2e-3 \
        --weight_decay 0.01 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --num_epochs 40 \
        --seed "$i" \
        --save_model_dir "$MODEL_DIR"
    
    python satire/experiments/baselines/bert.py \
        --model "$MODEL_DIR" \
        --data-file data/csv/selerosa.csv \
        --batch-size 64 \
        --seed "$i" \
        --output-dir "$MODEL_DIR"
done
