#!/usr/bin/env bash
# 啟動 ASR API 整合服務（單一埠 5000）。
# 對應 Windows 版本：start_app.bat
#
# 機敏設定請放在 api/.env（請複製 api/.env.example 並填入安全值）。
# 程式啟動時會由 app.py 自動載入 .env；本腳本僅提供非機敏的運行預設值。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "ASR API 整合服務啟動腳本 (port 5000)"
echo "========================================"

# ----- 非機敏運行預設（已有環境變數時不覆蓋） -----
: "${ASR_API_AUTH_DB:=$SCRIPT_DIR/auth.db}"
: "${FASTAPI_SKIP_INIT:=0}"
: "${FASTAPI_WARMUP:=1}"
: "${FASTAPI_ASR_MODEL_SIZE:=models}"
: "${FASTAPI_HOST:=0.0.0.0}"
: "${FASTAPI_PORT:=5000}"
: "${BUFFERING_CHUNK_LENGTH_SECONDS:=1.5}"
: "${BUFFERING_CHUNK_OFFSET_SECONDS:=0.1}"
export ASR_API_AUTH_DB FASTAPI_SKIP_INIT FASTAPI_WARMUP FASTAPI_ASR_MODEL_SIZE \
    FASTAPI_HOST FASTAPI_PORT BUFFERING_CHUNK_LENGTH_SECONDS BUFFERING_CHUNK_OFFSET_SECONDS

echo "Listening on http://$FASTAPI_HOST:$FASTAPI_PORT"

# ----- 提醒 .env -----
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "⚠ 找不到 api/.env，將使用程式內的不安全預設值（JWT_SECRET=CHANGE_ME_SECRET 等）"
    echo "  建議：cp $SCRIPT_DIR/.env.example $SCRIPT_DIR/.env 並填入安全的 JWT 密鑰與 admin 密碼後再啟動"
    echo
fi

# ----- 自動偵測虛擬環境（按優先序；同時支援 POSIX bin/ 與 Windows Scripts/） -----
VENV_ACTIVATE=""
for venv_name in asr_api train_env .venv venv; do
    for sub in bin/activate Scripts/activate; do
        candidate="$PROJECT_ROOT/$venv_name/$sub"
        if [ -f "$candidate" ]; then
            VENV_ACTIVATE="$candidate"
            echo "✅ 找到 $venv_name 虛擬環境（$sub）"
            break 2
        fi
    done
done

cd "$SCRIPT_DIR"

if [ -z "$VENV_ACTIVATE" ]; then
    echo "⚠ 未找到虛擬環境，將直接使用系統 Python"
    echo "  建議先執行 bash $PROJECT_ROOT/api/setup_api_env.sh 建立虛擬環境"
    echo
    if command -v python3 >/dev/null 2>&1; then
        exec python3 app.py
    else
        exec python app.py
    fi
else
    echo "正在啟動整合服務..."
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
    exec python app.py
fi
