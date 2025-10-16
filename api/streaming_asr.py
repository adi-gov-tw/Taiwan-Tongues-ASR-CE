#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI STT Streaming 服務器
"""

import asyncio
import uuid
import random
import json
import logging
import time
import urllib.parse
import base64
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import os

# 導入現有的模組
import sys

sys.path.append(str(Path(__file__).parent / "stt_streaming" / "src"))

from stt_streaming.src.asr.asr_factory import ASRFactory
from stt_streaming.src.vad.vad_factory import VADFactory
from stt_streaming.src.client import Client

# 設定日誌（固定寫入到 api/logs）
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "stt_streaming_fastapi.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# 語言固定為 zh，不再依賴 modelCode 參數


# 響應模型
class Response(BaseModel):
    code: int
    description: str
    data: Optional[Dict] = None

    def model_dump_json(self):
        return json.dumps(self.model_dump(), ensure_ascii=False)


class ResponseCode:
    SUCCESS = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    TIMEOUT = 408


class Settings:
    def __init__(self):
        self.max_streaming_count = 10


settings = Settings()

# 準備就緒事件：ASR 初始化與（可選）WARM_UP 完成後設為 ready
asr_ready_event: asyncio.Event = asyncio.Event()

# FastAPI 應用
app = FastAPI(title="STT Streaming API", version="1.0.0")

# 全局變數
connected_clients: List[Client] = []
vad_pipeline = None
asr_pipeline = None


@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化（非阻塞）"""
    global vad_pipeline, asr_pipeline

    logging.info("=" * 50)
    logging.info("FastAPI STT Streaming 服務器啟動")
    logging.info("=" * 50)

    # 若設定跳過初始化，直接返回，WS 仍可用
    skip_init = os.getenv("FASTAPI_SKIP_INIT", "0") in ("1", "true", "True")
    if skip_init:
        logging.warning("跳過 VAD/ASR 初始化（FASTAPI_SKIP_INIT=1）")
        return

    async def _initialize_pipelines_background():
        global vad_pipeline, asr_pipeline
        # 初始化 VAD
        try:
            logging.info("正在初始化 VAD 管道...")
            vad_pipeline = VADFactory.create_vad_pipeline("simple", min_duration=0.1)
            logging.info("✅ VAD 管道初始化成功")
        except Exception as e:
            vad_pipeline = None
            logging.error(f"VAD 初始化失敗：{e}")

        # 初始化 ASR（可透過環境變數調整模型大小）
        model_size = os.getenv("FASTAPI_ASR_MODEL_SIZE", "models")
        try:
            logging.info("正在初始化 ASR 管道...")
            asr_pipeline = ASRFactory.create_asr_pipeline(
                "faster_whisper", model_size=model_size
            )
            logging.info("✅ ASR 管道初始化成功")
            # 若不進行預熱，ASR 初始化完成即視為就緒
            if os.getenv("FASTAPI_WARMUP", "0") not in ("1", "true", "True"):
                try:
                    asr_ready_event.set()
                except Exception:
                    pass
        except Exception as e:
            asr_pipeline = None
            logging.error(f"ASR 初始化失敗：{e}")

        # 預熱（可選）
        if asr_pipeline is not None and os.getenv("FASTAPI_WARMUP", "0") in (
            "1",
            "true",
            "True",
        ):
            try:
                logging.info("正在預熱 ASR 管道...")
                asr_pipeline.warm_up()
                logging.info("✅ ASR 管道預熱完成")
                try:
                    asr_ready_event.set()
                except Exception:
                    pass
            except Exception as e:
                logging.error(f"ASR 預熱失敗：{e}")

    # 背景初始化，不阻塞 WS 就緒
    try:
        asyncio.create_task(_initialize_pipelines_background())
        logging.info("VAD/ASR 初始化已在背景進行中...")
    except Exception as e:
        logging.error(f"無法啟動背景初始化任務：{e}")


@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉時清理"""
    logging.info("正在關閉 FastAPI STT Streaming 服務器...")
    # 清理所有連接的客戶端
    for client in connected_clients:
        logging.info(f"清理客戶端: {client.client_id}")
    connected_clients.clear()


def _send_error_and_close(websocket: WebSocket, error_message: str, job_record=None):
    """發送錯誤訊息並關閉連接"""
    try:
        error_response = Response(
            code=ResponseCode.BAD_REQUEST, description=error_message
        )
        asyncio.create_task(websocket.send_text(error_response.model_dump_json()))
    except Exception as e:
        logging.error(f"發送錯誤訊息失敗: {e}")


async def _validate_token(websocket: WebSocket, token: str) -> str:
    """驗證 token 並返回 user_id"""
    # 簡化的 token 驗證邏輯
    if not token:
        _send_error_and_close(websocket, "token is required")
        return None

    # 這裡可以添加更複雜的 token 驗證邏輯
    user_id = token if token else "default_user"
    logging.info(f"Token 驗證成功，user_id: {user_id}")
    return user_id


def generate_job_id() -> str:
    """生成作業 ID"""
    return f"job_{int(time.time())}"


async def handle_audio(client: Client, websocket: WebSocket):
    """處理音頻數據"""
    last_message_time = asyncio.get_running_loop().time()

    try:
        while True:
            # 檢查連接是否仍然活躍
            if (
                websocket.client_state == WebSocketState.DISCONNECTED
                or websocket.application_state == WebSocketState.DISCONNECTED
            ):
                break

            # 單一 receive，依內容分流處理（避免 text/bytes 交錯導致 KeyError）
            try:
                incoming = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                logging.warning(f"WebSocket 接收時斷開: {client.client_id}")
                break
            except Exception as e:
                logging.error(f"WebSocket 接收例外: {e}")
                continue

            msg_type = incoming.get("type")
            if msg_type == "websocket.receive":
                # 二進位音訊
                if incoming.get("bytes") is not None:
                    audio_chunk = incoming.get("bytes")
                    if audio_chunk:
                        client.append_audio_data(audio_chunk)
                # 文字訊息
                elif incoming.get("text") is not None:
                    message = incoming.get("text")
                    # 解析 JSON 訊息
                    try:
                        message_data = json.loads(message)
                    except json.JSONDecodeError:
                        logging.error(f"無效的 JSON 訊息: {message}")
                        continue

                    msg_payload_type = message_data.get("type")
                    if msg_payload_type == "config" and isinstance(
                        message_data.get("data"), dict
                    ):
                        cfg = message_data["data"]
                        config_update = {}
                        if "language" in cfg and cfg["language"]:
                            config_update["language"] = cfg["language"]
                        if "processing_strategy" in cfg and cfg["processing_strategy"]:
                            config_update["processing_strategy"] = cfg[
                                "processing_strategy"
                            ]
                        if "processing_args" in cfg and isinstance(
                            cfg["processing_args"], dict
                        ):
                            config_update["processing_args"] = cfg["processing_args"]
                        if config_update:
                            client.update_config(config_update)
                        try:
                            if "sampleRate" in cfg and isinstance(
                                cfg["sampleRate"], (int, float)
                            ):
                                client.sampling_rate = int(cfg["sampleRate"])
                            if "channels" in cfg and isinstance(cfg["channels"], int):
                                pass
                        except Exception:
                            pass
                        await websocket.send_text(
                            Response(
                                code=ResponseCode.SUCCESS, description="config 已更新"
                            ).model_dump_json()
                        )
                        continue
                    elif message_data.get("audio"):
                        try:
                            audio_bytes = base64.b64decode(message_data["audio"])
                            client.append_audio_data(audio_bytes)
                        except Exception:
                            logging.error("base64 音訊解析失敗")
                            continue
                    else:
                        logging.warning(f"未知訊息: {message_data}")
                        continue
            elif msg_type in ("websocket.disconnect", "websocket.close"):
                logging.info(f"WebSocket 收到關閉: {client.client_id}")
                break
            else:
                # 其他訊息型別，忽略
                continue

            # 處理音頻
            if vad_pipeline is None or asr_pipeline is None:
                # 在跳過初始化或初始化失敗時，避免處理導致連線中斷
                await websocket.send_text(
                    Response(
                        code=ResponseCode.SUCCESS,
                        description="audio received (ASR/VAD not initialized)",
                        data={"buffer_bytes": len(client.buffer)},
                    ).model_dump_json()
                )
            else:
                client.process_audio(websocket, vad_pipeline, asr_pipeline)
            last_message_time = asyncio.get_running_loop().time()

    except Exception as e:
        logging.error(f"處理音頻時發生錯誤: {e}", exc_info=True)
    finally:
        # 清理客戶端
        job_id = client.job_id
        transcript = client.transcript
        end_time = time.time()
        duration = end_time - client.start_time if client.start_time > 0 else 0
        logging.info(
            f"連接結束 - user_id: {client.client_id}, job_id: {job_id}, duration: {duration}"
        )
        # 在連線結束時一次性將整段音訊存檔
        """
        try:
            if client.session_audio_buffer and len(client.session_audio_buffer) > 0:
                from stt_streaming.src.audio_utils import save_audio_to_file
                session_file_name = client.get_session_file_name()
                await save_audio_to_file(client.session_audio_buffer, session_file_name)
                logging.info(f"已儲存本次連線音訊: {session_file_name}, bytes: {len(client.session_audio_buffer)}")
        except Exception as e:
            logging.error(f"儲存本次連線音訊失敗: {e}")
        """
        # 從連接列表中移除
        for c in connected_clients:
            if c.client_id == client.client_id:
                logging.info(f"移除客戶端: {client.client_id}")
                connected_clients.remove(c)
                break


@app.websocket("/ws/stt")
async def streaming_stt_recognization(
    websocket: WebSocket,
    token: str = None,
):
    """STT Streaming 識別端點"""
    streaming_record = None

    try:
        # 接受 WebSocket 連接
        await websocket.accept()
        logging.info("WebSocket 連接已接受")

        # 不再使用 modelCode/jobId 查詢參數
        # 驗證 token
        user_id = await _validate_token(websocket, token)
        if not user_id:
            return

        # 檢查連接數限制
        if len(connected_clients) >= settings.max_streaming_count:
            _send_error_and_close(websocket, "exceeded number of connections")
            return

        # 生成作業 ID、連線 ID 與此次連線的 taskId（六位數）
        job_id = generate_job_id()
        connection_id = str(uuid.uuid4())
        task_id = random.randint(100000, 999999)
        logging.info(
            f"user_id: {user_id}, job_id: {job_id}, task_id: {task_id}, connection_id: {connection_id}"
        )

        # 創建客戶端
        transcript = []
        last_start_time = 0
        client = Client(user_id, 16000, 2, job_id, last_start_time, transcript)
        # 固定每次連線的回覆 id
        try:
            setattr(client, "connection_id", connection_id)
        except Exception:
            pass

        # 語言固定為 zh，不再從參數設定

        # 添加到連接列表
        connected_clients.append(client)

        # 連線建立後回覆：服務準備中（code=100），id 為本次連線固定值
        try:
            await websocket.send_text(
                json.dumps(
                    {"id": connection_id, "code": 100, "message": "服務準備中"},
                    ensure_ascii=False,
                )
            )
        except Exception as e:
            logging.error(f"送出 '服務準備中' 訊息失敗: {e}")

        # 若 ASR 就緒，立即告知；否則等待預熱完成後通知（code=180）
        async def _notify_ready_when_warmup_done():
            try:
                if asr_ready_event.is_set():
                    await websocket.send_text(
                        json.dumps(
                            {
                                "id": connection_id,
                                "taskId": task_id,
                                "code": 180,
                                "message": "服務已就緒",
                            },
                            ensure_ascii=False,
                        )
                    )
                else:
                    await asr_ready_event.wait()
                    await websocket.send_text(
                        json.dumps(
                            {
                                "id": connection_id,
                                "taskId": task_id,
                                "code": 180,
                                "message": "服務已就緒",
                            },
                            ensure_ascii=False,
                        )
                    )
            except Exception as e:
                logging.error(f"通知 '服務已就緒' 失敗: {e}")

        try:
            asyncio.create_task(_notify_ready_when_warmup_done())
        except Exception as e:
            logging.error(f"無法建立就緒通知任務: {e}")

        logging.info(f"Client {user_id} 已連接: {websocket.client.host}")

        # 處理音頻數據
        await handle_audio(client, websocket)

    except WebSocketDisconnect:
        logging.info("WebSocket 連接已斷開")
    except Exception as e:
        logging.error(f"WebSocket 處理錯誤: {e}", exc_info=True)
        _send_error_and_close(
            websocket, f"Internal server error: {str(e)}", streaming_record
        )


@app.get("/")
async def get_root():
    """根路徑"""
    return {"message": "STT Streaming API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """健康檢查"""
    details = {
        "status": "healthy",
        "connected_clients": len(connected_clients),
        "vad_pipeline": "ready" if vad_pipeline else "not_ready",
        "asr_pipeline": "ready" if asr_pipeline else "not_ready",
    }
    try:
        if asr_pipeline is not None and hasattr(asr_pipeline, "asr_pipeline"):
            # faster-whisper 內部資訊
            details.update(
                {
                    "asr_device": getattr(asr_pipeline, "device", None),
                    "asr_compute_type": getattr(asr_pipeline, "compute_type", None),
                    "asr_model_size": getattr(asr_pipeline, "model_size", None),
                }
            )
    except Exception:
        pass
    return details


@app.get("/test")
async def get_test_page():
    """測試頁面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>STT Streaming 測試</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <h1>STT Streaming 測試頁面</h1>
        <p>WebSocket 端點: <code>ws://localhost:8000/ws/stt?modelCode=chinese&token=test_token&jobId=test_job</code></p>
        <p>健康檢查: <a href="/health">/health</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def main():
    """主函數"""
    logging.info("啟動 FastAPI STT Streaming 服務器...")

    # 確保日誌目錄存在（api/logs）
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 啟動服務器
    host = os.getenv("FASTAPI_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("FASTAPI_PORT", "8000"))
    except ValueError:
        port = 8000
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
