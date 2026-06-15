#!/usr/bin/env bash
# Fine-tune the ASR model on sample_corpus.
# 自動偵測並啟動訓練 venv（train_env），無需手動 activate。
# 對應 Windows 版本：train.bat
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----- 自動偵測虛擬環境（按優先序；同時支援 POSIX bin/ 與 Windows Scripts/） -----
VENV_ACTIVATE=""
for venv_name in train_env asr_api .venv venv; do
    for sub in bin/activate Scripts/activate; do
        candidate="$SCRIPT_DIR/$venv_name/$sub"
        if [ -f "$candidate" ]; then
            VENV_ACTIVATE="$candidate"
            echo "Found venv: $venv_name ($sub)"
            break 2
        fi
    done
done

if [ -z "$VENV_ACTIVATE" ]; then
    echo "WARN: No venv found. Run 'bash setup_train_env.sh' first."
    echo "      Falling back to system Python..."
else
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi

# ----- 訓練設定 -----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output}"

mkdir -p "${OUTPUT_DIR}"

# dataset_config_name 支援 `ds:lang` 指定每份資料集的語系（Whisper 語言代碼）：
#   範例：混訓中/英/印尼 → "train_ds_01:zh+train_ds_02:en+train_ds_id:id"
# 未帶 `:lang` 時會 fallback 到下方的 --language（向後相容舊腳本）。
python train_asr.py \
    --model_name_or_path="model_for_finetune" \
    --dataset_name="csv" \
    --corpus_data_dir="sample_corpus" \
    --dataset_config_name="${DATASET_CONFIG_NAME:-train_ds_01:zh+train_ds_02:en}" \
    --language="zh" \
    --train_split_name="train+validated" \
    --eval_split_name="test" \
    --max_steps="2000" \
    --output_dir="${OUTPUT_DIR}" \
    --per_device_train_batch_size="4" \
    --gradient_accumulation_steps="1" \
    --per_device_eval_batch_size="16" \
    --logging_steps="25" \
    --learning_rate="1e-5" \
    --warmup_steps="500" \
    --eval_strategy="steps" \
    --eval_steps="1000" \
    --save_strategy="steps" \
    --save_steps="1000" \
    --generation_max_length="225" \
    --preprocessing_num_workers="16" \
    --length_column_name="input_length" \
    --max_duration_in_seconds="30" \
    --text_column_name="sentence" \
    --freeze_feature_encoder="False" \
    --gradient_checkpointing \
    --sortish_sampler=True \
    --fp16 \
    --streaming=False \
    --do_train \
    --do_eval \
    --predict_with_generate
