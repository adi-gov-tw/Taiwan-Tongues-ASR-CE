#!/usr/bin/env bash
# 建立訓練/推論用虛擬環境（train_env）並安裝 requirements.txt 內依賴。
# 對應 Windows 版本：setup_train_env.bat
set -e

cd "$(dirname "$0")"

echo "=== 建立訓練/推論虛擬環境 (train_env) ==="

# 偵測 python 指令
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
if [ -d "train_env" ]; then
    read -r -p "虛擬環境 train_env 已存在，是否要重新建立？(y/N) " choice
    case "$choice" in
        y|Y)
            echo "正在刪除舊的虛擬環境..."
            rm -rf train_env
            ;;
        *)
            echo "使用現有的虛擬環境，僅更新依賴..."
            ;;
    esac
fi

if [ ! -d "train_env" ]; then
    echo "正在建立虛擬環境..."
    "$PYTHON" -m venv train_env
fi

# shellcheck disable=SC1091
source train_env/bin/activate

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
echo "正在安裝 requirements.txt 依賴..."
pip install -r requirements.txt

echo
echo "=== 驗證安裝 ==="
python - <<'PY'
import importlib
mods = ["torch", "transformers", "librosa", "datasets", "evaluate", "accelerate", "soundfile"]
for m in mods:
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
        print(f"  OK  {m:<14} {v}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
PY

echo
echo "=== 完成 ==="
echo "啟動虛擬環境：source train_env/bin/activate"
