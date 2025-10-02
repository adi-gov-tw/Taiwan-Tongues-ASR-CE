export CUDA_VISIBLE_DEVICES="0"
export OUTPUT_DIR=./output

mkdir -p ${OUTPUT_DIR}

python train_asr.py \
	--model_name_or_path="$path/model_multilang" \
	--dataset_name="csv" \
	--corpus_data_dir="sample_corpus" \
	--dataset_config_name="train_ds_01+train_ds_02" \
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
	--evaluation_strategy="steps" \
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
	--group_by_length \
	--fp16 \
	--overwrite_output_dir \
	--streaming=False \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--use_auth_token=False
