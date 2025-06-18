#!/bin/bash

export WANDB_PROJECT=SeLeRoSa

for i in {1..5}; do
    export WANDB_NAME="ro_bert_cased_unfrozen_$i"
    MODEL_DIR="./results/ro_bert_cased_unfrozen_ft/$i"
    RESULTS_DIR="./results/inference/ro_bert_cased_unfrozen_ft/$i"
    mkdir -p "$RESULTS_DIR"

    python satire/experiments/baselines/bert_finetune.py \
        --data_file data/csv/selerosa.csv \
        --model_checkpoint "dumitrescustefan/bert-base-romanian-cased-v1" \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --num_epochs 5 \
        --seed "$i" \
        --save_model_dir "$MODEL_DIR"
    
    python satire/experiments/baselines/bert.py \
        --model "$MODEL_DIR" \
        --data-file data/csv/selerosa.csv \
        --batch-size 32 \
        --seed "$i" \
        --output-dir "$MODEL_DIR"
done
