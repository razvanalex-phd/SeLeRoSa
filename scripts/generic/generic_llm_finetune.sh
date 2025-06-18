#!/bin/bash

# Usage: source this script after exporting only model-specific variables:
#   MAX_TOKENS, TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
#
# All other common variables are set here.

BASE_MODEL_NAME="${BASE_MODEL##*/}"
BASE_MODEL_NAME="${BASE_MODEL_NAME%%:*}"

export WANDB_PROJECT="${WANDB_PROJECT:-SeLeRoSa}"
export WANDB_NAME="${WANDB_NAME:-${NEW_MODEL}_ft}"
export WANDB_GROUP="${BASE_MODEL_NAME}"
export WANDB_TAGS="$BASE_MODEL_NAME,finetune,llm"
export SAVE_PATH="${SAVE_PATH:-results}"
export DATASET="${DATASET:-data/csv/selerosa.csv}"
export CLASSES="${CLASSES:-factual,satiră}"
export LABEL_FIELD="${LABEL_FIELD:-label}"
export DATASET_TEXT_FIELD="${DATASET_TEXT_FIELD:-text}"
export WARMUP_STEPS="${WARMUP_STEPS:-0.1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export TRAIN_LR_SCHEDULER_TYPE="${TRAIN_LR_SCHEDULER_TYPE:-linear}"
export TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER:-paged_adamw_8bit}"
export ENABLE_LORA="${ENABLE_LORA:-1}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export EVALUATION_STRATEGY="${EVALUATION_STRATEGY:-steps}"
export EVALUATION_STEPS="${EVALUATION_STEPS:-100}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export LORA_BIAS="${LORA_BIAS:-none}"
export LORA_MODULES_TO_SAVE="${LORA_MODULES_TO_SAVE:-embed_tokens lm_head}"
export LORA_TASK_TYPE="${LORA_TASK_TYPE:-CAUSAL_LM}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-up_proj down_proj gate_proj k_proj q_proj v_proj o_proj}"
export LORA_FAN_IN_FAN_OUT="${LORA_FAN_IN_FAN_OUT:-false}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export QUANTIZATION_TYPE="${QUANTIZATION_TYPE:-4bit}"
export BNB_4BIT_QUANT_TYPE="${BNB_4BIT_QUANT_TYPE:-nf4}"
export BNB_4BIT_COMPUTE_DTYPE="${BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
export ENABLE_CHECKPOINTING="${ENABLE_CHECKPOINTING:-false}"
export CHECKPOINT_SAVE_STEPS="${CHECKPOINT_SAVE_STEPS:-100}"
export RESTORE_FROM_CHECKPOINT="${RESTORE_FROM_CHECKPOINT:-}"

QUANTIZATION_ARGS=""
if [ "$QUANTIZATION_TYPE" = "4bit" ]; then
  QUANTIZATION_ARGS="--load_in_4bit --bnb_4bit_quant_type $BNB_4BIT_QUANT_TYPE --bnb_4bit_use_double_quant --bnb_4bit_compute_dtype $BNB_4BIT_COMPUTE_DTYPE"
elif [ "$QUANTIZATION_TYPE" = "8bit" ]; then
  BNB_8BIT_QUANT_TYPE="${BNB_8BIT_QUANT_TYPE:-nf8}"
  BNB_8BIT_COMPUTE_DTYPE="${BNB_8BIT_COMPUTE_DTYPE:-bfloat16}"
  QUANTIZATION_ARGS="--load_in_8bit --bnb_8bit_quant_type $BNB_8BIT_QUANT_TYPE --bnb_8bit_use_double_quant --bnb_8bit_compute_dtype $BNB_8BIT_COMPUTE_DTYPE"
fi

LORA_CONFIG=""
if [ "$ENABLE_LORA" = "1" ]; then
  LORA_CONFIG="--lora --r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT --bias $LORA_BIAS --modules_to_save $LORA_MODULES_TO_SAVE --task_type $LORA_TASK_TYPE --target_modules $LORA_TARGET_MODULES"
  
  # Only add the fan_in_fan_out flag if it's set to true
  if [ "$LORA_FAN_IN_FAN_OUT" = "true" ]; then
    LORA_CONFIG="$LORA_CONFIG --lora_fan_in_fan_out"
  fi
fi

PROMPT_TUNING_CONFIG=""
if [ "$ENABLE_PROMPT_TUNING" = "1" ]; then
  PROMPT_TUNING_CONFIG="--prompt_tuning --prompt_tuning_system_path $PROMPT_TUNING_SYSTEM_PATH"
fi

P_TUNING_CONFIG=""
if [ "$ENABLE_P_TUNING" = "1" ]; then
  P_TUNING_NUM_VIRTUAL_TOKENS="${P_TUNING_NUM_VIRTUAL_TOKENS:-20}"
  P_TUNING_REPARAMETRIZATION_TYPE="${P_TUNING_REPARAMETRIZATION_TYPE:-MLP}"
  P_TUNING_HIDDEN_SIZE="${P_TUNING_HIDDEN_SIZE:-512}"
  P_TUNING_NUM_LAYERS="${P_TUNING_NUM_LAYERS:-2}"
  P_TUNING_DROPOUT="${P_TUNING_DROPOUT:-0.0}"
  P_TUNING_CONFIG="--p_tuning --p_tuning_num_virtual_tokens $P_TUNING_NUM_VIRTUAL_TOKENS --p_tuning_reparameterization_type $P_TUNING_REPARAMETRIZATION_TYPE --p_tuning_hidden_size $P_TUNING_HIDDEN_SIZE --p_tuning_num_layers $P_TUNING_NUM_LAYERS --p_tuning_dropout $P_TUNING_DROPOUT"
fi

DEVICE_MAP_ARG=""
if [ -n "$DEVICE_MAP" ]; then
  DEVICE_MAP_ARG="--device_map $DEVICE_MAP"
fi

MERGE_SYSTEM_INTO_USER_ARG=""
if [ -n "$MERGE_SYSTEM_INTO_USER" ]; then
  MERGE_SYSTEM_INTO_USER_ARG="--merge_system_into_user"
fi

USE_CHAT_TEMPLATE_ARG=""
if [ -n "$USE_CHAT_TEMPLATE" ]; then
  USE_CHAT_TEMPLATE_ARG="--use_chat_template"
fi

TO_TOKENS_ARG=""
if [ -n "$TO_TOKENS" ]; then
  TO_TOKENS_ARG="--to_tokens"
fi

NO_MERGE_PEFT_ARG=""
if [ -n "$NO_MERGE_PEFT" ]; then
  NO_MERGE_PEFT_ARG="--no_merge_peft"
fi

CHECKPOINTING_ARGS=""
if [ "$ENABLE_CHECKPOINTING" = "true" ]; then
  CHECKPOINTING_ARGS="--enable_checkpointing --checkpoint_save_steps $CHECKPOINT_SAVE_STEPS"
  if [ -n "$RESTORE_FROM_CHECKPOINT" ]; then
    CHECKPOINTING_ARGS="$CHECKPOINTING_ARGS --restore_from_checkpoint $RESTORE_FROM_CHECKPOINT"
  fi
fi

python3 satire/experiments/baselines/llm_finetune.py \
  --base_model "$BASE_MODEL" \
  --new_model "$NEW_MODEL" \
  --save_path "$SAVE_PATH" \
  --output_dir "$SAVE_PATH/$NEW_MODEL" \
  --dataset "$DATASET" \
  --classes "$CLASSES" \
  --label_field "$LABEL_FIELD" \
  --dataset_text_field "$DATASET_TEXT_FIELD" \
  --max_seq_length "$MAX_TOKENS" \
  --num_train_epochs "$TRAIN_EPOCHS" \
  --warmup_steps "$WARMUP_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "$TRAIN_LR_SCHEDULER_TYPE" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --optim "$TRAIN_OPTIMIZER" \
  $LORA_CONFIG \
  $PROMPT_TUNING_CONFIG \
  $P_TUNING_CONFIG \
  $DEVICE_MAP_ARG \
  $MERGE_SYSTEM_INTO_USER_ARG \
  $USE_CHAT_TEMPLATE_ARG \
  --evaluation_strategy "$EVALUATION_STRATEGY" \
  --eval_steps "$EVALUATION_STEPS" \
  --logging_steps "$LOGGING_STEPS" \
  $QUANTIZATION_ARGS \
  $TO_TOKENS_ARG \
  $NO_MERGE_PEFT_ARG \
  $CHECKPOINTING_ARGS
