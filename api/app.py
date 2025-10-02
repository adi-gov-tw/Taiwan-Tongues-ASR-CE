import os
from contextlib import asynccontextmanager
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
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")


if __name__ == "__main__":
    main()
