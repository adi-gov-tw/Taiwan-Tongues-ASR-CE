@echo off
REM Fine-tune the ASR model on sample_corpus (Windows version of train.sh).
REM Auto-detects training venv (train_env). No need to manually activate.

setlocal

REM ========================================
REM Auto-detect virtual environment (priority order)
REM ========================================
set VENV_ACTIVATE=
if exist "%~dp0train_env\Scripts\activate.bat" (
    set VENV_ACTIVATE=%~dp0train_env\Scripts\activate.bat
    echo Found venv: train_env
) else if exist "%~dp0asr_api\Scripts\activate.bat" (
    set VENV_ACTIVATE=%~dp0asr_api\Scripts\activate.bat
    echo Found venv: asr_api
) else if exist "%~dp0.venv\Scripts\activate.bat" (
    set VENV_ACTIVATE=%~dp0.venv\Scripts\activate.bat
    echo Found venv: .venv
) else if exist "%~dp0venv\Scripts\activate.bat" (
    set VENV_ACTIVATE=%~dp0venv\Scripts\activate.bat
    echo Found venv: venv
)

if "%VENV_ACTIVATE%"=="" (
    echo WARN: No venv found. Run setup_train_env.bat first.
    echo       Falling back to system Python...
) else (
    call "%VENV_ACTIVATE%"
)

REM ========================================
REM Training settings
REM ========================================
if "%CUDA_VISIBLE_DEVICES%"=="" set CUDA_VISIBLE_DEVICES=0
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=.\output
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

cd /d "%~dp0"

REM dataset_config_name 支援 `ds:lang` 指定每份資料集的語系（Whisper 語言代碼）：
REM   範例：混訓中/英/印尼 → "train_ds_01:zh+train_ds_02:en+train_ds_id:id"
REM 未帶 `:lang` 時會 fallback 到下方的 --language（向後相容舊腳本）。
if "%DATASET_CONFIG_NAME%"=="" set DATASET_CONFIG_NAME=train_ds_01:zh+train_ds_02:en

python train_asr.py ^
    --model_name_or_path=model_for_finetune ^
    --dataset_name=csv ^
    --corpus_data_dir=sample_corpus ^
    --dataset_config_name=%DATASET_CONFIG_NAME% ^
    --language=zh ^
    --train_split_name=train+validated ^
    --eval_split_name=test ^
    --max_steps=2000 ^
    --output_dir=%OUTPUT_DIR% ^
    --per_device_train_batch_size=4 ^
    --gradient_accumulation_steps=1 ^
    --per_device_eval_batch_size=16 ^
    --logging_steps=25 ^
    --learning_rate=1e-5 ^
    --warmup_steps=500 ^
    --eval_strategy=steps ^
    --eval_steps=1000 ^
    --save_strategy=steps ^
    --save_steps=1000 ^
    --generation_max_length=225 ^
    --preprocessing_num_workers=16 ^
    --length_column_name=input_length ^
    --max_duration_in_seconds=30 ^
    --text_column_name=sentence ^
    --freeze_feature_encoder=False ^
    --gradient_checkpointing ^
    --sortish_sampler=True ^
    --fp16 ^
    --streaming=False ^
    --do_train ^
    --do_eval ^
    --predict_with_generate

endlocal
pause
