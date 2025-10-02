@echo off
chcp 65001 >nul
echo === 建立 Whisper 模型測試虛擬環境 ===
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo 錯誤: 找不到 Python，請先安裝 Python 3.8+
    pause
    exit /b 1
)

REM 檢查虛擬環境是否已存在
if exist "whisper_env" (
    echo 虛擬環境已存在，是否要重新建立？(Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        echo 正在刪除舊的虛擬環境...
        rmdir /s /q whisper_env
    ) else (
        echo 使用現有的虛擬環境...
        goto activate_env
    )
)

echo 正在建立虛擬環境...
python -m venv whisper_env

:activate_env
echo.
echo 正在啟動虛擬環境...
call whisper_env\Scripts\activate.bat

echo.
echo 正在升級 pip...
python -m pip install --upgrade pip

echo.
echo 正在安裝依賴套件...
echo 安裝 torch...
pip install torch>=1.12.0
if errorlevel 1 (
    echo 警告: torch 安裝失敗，嘗試安裝 CPU 版本...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
)

echo 安裝 transformers...
pip install transformers>=4.31.0

echo 安裝 librosa...
pip install librosa>=0.9.0

echo 安裝其他依賴套件...
pip install numpy>=1.21.0
pip install datasets>=1.18.0
pip install evaluate>=0.4.0
pip install accelerate>=0.20.0
pip install soundfile>=0.12.0
pip install protobuf>=3.20.0

echo.
echo 驗證安裝...
python -c "import torch; print('torch 安裝成功:', torch.__version__)" || echo "torch 安裝失敗"
python -c "import transformers; print('transformers 安裝成功:', transformers.__version__)" || echo "transformers 安裝失敗"
python -c "import librosa; print('librosa 安裝成功:', librosa.__version__)" || echo "librosa 安裝失敗"
python -c "import google.protobuf; print('protobuf 安裝成功:', google.protobuf.__version__)" || echo "protobuf 安裝失敗"

echo.
echo === 虛擬環境設定完成 ===
echo.
echo 使用方式：
echo 1. 啟動虛擬環境: whisper_env\Scripts\activate.bat
echo 2. 執行測試: python run_test.py
echo 3. 退出虛擬環境: deactivate
echo.
echo 或者直接執行 run_test_venv.bat
echo.
pause 