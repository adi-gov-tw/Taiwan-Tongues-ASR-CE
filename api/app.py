import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Windows: 把 pip 裝的 nvidia cuDNN/cuBLAS bin 目錄加入 DLL 搜尋路徑，
# 讓 ctranslate2 (faster-whisper 後端) 能載入。必須在 import faster_whisper 之前。
# 注意：nvidia.cudnn / nvidia.cublas 是 PEP 420 namespace package，
#       沒有 __init__.py 因此 __file__ 是 None；要改用 __path__ 取目錄。
if sys.platform == "win32":
    import importlib

    for _pkg_name in ("nvidia.cudnn", "nvidia.cublas"):
        try:
            _pkg = importlib.import_module(_pkg_name)
            _pkg_file = getattr(_pkg, "__file__", None)
            if _pkg_file:
                _pkg_dir = os.path.dirname(_pkg_file)
            else:
                _paths = list(getattr(_pkg, "__path__", []) or [])
                _pkg_dir = _paths[0] if _paths else None
            if _pkg_dir:
                _bin = os.path.join(_pkg_dir, "bin")
                if os.path.isdir(_bin):
                    os.add_dll_directory(_bin)
        except ImportError:
            pass

# 載入 .env（必須在所有讀取環境變數的模組 import 前完成；
# 如 auth_shared.JWT_SECRET 在 module load 時就會凍結）。
# 不存在 .env 時靜默略過，仍由系統環境變數或 start_app.bat 提供。
try:
    from dotenv import load_dotenv

    _ENV_PATH = Path(__file__).parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH, override=False)
except ImportError:
    pass

from fastapi import FastAPI
import uvicorn

# 匯入既有應用與啟動/關閉事件
from file_asr import app as file_app
import file_asr as file_module
from auth_api import auth_startup
import streaming_asr as streaming_module

# 建立聚合應用：
# - 保留檔案 ASR 原有路徑（/api/...）
# - 串流 ASR 以 /stream 為前綴（/stream/ws/stt 等）
app = FastAPI(
    title="Combined ASR API",
    version="1.0.0",
    swagger_ui_parameters={"persistAuthorization": True},
)

# 直接包含 file_asr 的所有路由到主應用
app.include_router(file_app.router)

# 掛載串流 ASR 應用到 /stream 前綴
app.mount("/stream", streaming_module.app)

# 註冊 WebSocket 路由
app.add_api_websocket_route(
    "/ws/v1/transcript", streaming_module.streaming_stt_recognization
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 手動執行子應用的啟動邏輯
    try:
        # file_asr 的啟動：建立/初始化授權資料
        auth_startup()
    except Exception:
        pass

    # 初始化字幕任務資料表（因為子應用掛載時其 lifespan 不一定會被觸發）
    try:
        if hasattr(file_module, "_ensure_tasks_schema"):
            file_module._ensure_tasks_schema()
    except Exception:
        pass

    try:
        # streaming_asr 的啟動事件
        await streaming_module.startup_event()
    except Exception:
        pass

    yield

    try:
        # streaming_asr 的關閉事件
        await streaming_module.shutdown_event()
    except Exception:
        pass


app.router.lifespan_context = lifespan

# 移除重複的 WebSocket 路由註冊（已在上面註冊）


def main():
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", "5000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
