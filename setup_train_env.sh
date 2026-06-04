#!/usr/bin/env bash
# 建立訓練/推論用虛擬環境（train_env）並安裝 requirements.txt 內依賴。
# 對應 Windows 版本：setup_train_env.bat
# 本專案要求 Python 3.10 以上。
set -e

cd "$(dirname "$0")"

echo "=== 建立訓練/推論虛擬環境 (train_env) — 需要 Python 3.10+ ==="

# 偵測 python 指令：優先 python3.10，再依序往上找符合 >=3.10 的直譯器
PYTHON=""
candidates="python3.10 python3.11 python3.12 python3.13 python3 python"
for cand in $candidates; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c 'import sys;exit(0 if sys.version_info[:2]>=(3,10) else 1)' >/dev/null 2>&1; then
            PYTHON="$cand"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "錯誤: 找不到 Python 3.10 以上版本，請先安裝後再試。"
    echo "      Linux: 透過 apt / pyenv 等安裝 python3.10 (或更新版)"
    echo "      macOS: brew install python@3.10 (或更新版)"
    exit 1
fi

PY_VERSION=$("$PYTHON" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')
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
echo "確認 venv Python 版本..."
python -c "import sys; assert sys.version_info[:2]>=(3,10), 'venv python is %s, need >=3.10' % sys.version.split()[0]; print('venv python', sys.version.split()[0])"

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
import sys, importlib
print(f"  OK  python         {sys.version.split()[0]}")
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
