#!/usr/bin/env bash
# 建立 API 服務專用虛擬環境（asr_api）並安裝 api/requirements.txt 內依賴。
# 對應 Windows 版本：setup_api_env.bat
set -e

# 從專案根目錄執行（venv 建立在專案根，與其他環境一致）
cd "$(dirname "$0")/.."

echo "=== 建立 ASR API 服務專用虛擬環境 (asr_api) ==="

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "錯誤: 找不到 python，請先安裝 Python 3.9+"
    exit 1
fi

PY_VERSION=$($PYTHON -c 'import sys;print("%d.%d"%sys.version_info[:2])')
echo "使用 Python $PY_VERSION ($($PYTHON -c 'import sys;print(sys.executable)'))"

# 處理已存在的 venv
if [ -d "asr_api" ]; then
    read -r -p "虛擬環境 asr_api 已存在，是否要重新建立？(y/N) " choice
    case "$choice" in
        y|Y)
            echo "正在刪除舊的虛擬環境..."
            rm -rf asr_api
            ;;
        *)
            echo "使用現有的虛擬環境，僅更新依賴..."
            ;;
    esac
fi

if [ ! -d "asr_api" ]; then
    echo "正在建立虛擬環境..."
    "$PYTHON" -m venv asr_api
fi

# shellcheck disable=SC1091
source asr_api/bin/activate

echo
echo "正在升級 pip..."
python -m pip install --upgrade pip

echo
echo "偵測 NVIDIA GPU..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "偵測到 NVIDIA GPU；安裝 CUDA 12.4 版 PyTorch。"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    TORCH_LABEL="CUDA 12.4"
else
    echo "未偵測到 NVIDIA GPU；安裝 CPU 版 PyTorch。"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_LABEL="CPU"
fi

echo
echo "正在安裝 PyTorch ($TORCH_LABEL)..."
pip install torch --index-url "$TORCH_INDEX"

if [ "$TORCH_LABEL" != "CPU" ]; then
    echo
    echo "正在安裝 cuBLAS / cuDNN 9（ctranslate2 / faster-whisper GPU 推論需要）..."
    pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10"
fi

echo
echo "正在安裝 api/requirements.txt..."
pip install -r api/requirements.txt

echo
echo "=== 完成 ==="
echo "啟動服務：bash api/start_app.sh"
echo "（首次啟動前請先複製 api/.env.example 為 api/.env 並填入安全值）"
