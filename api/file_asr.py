import os
import sqlite3
import asyncio
import tempfile
import time
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, List

# Windows: 把 pip 裝的 nvidia cuDNN/cuBLAS bin 目錄加入 DLL 搜尋路徑，
# 讓 ctranslate2 能載入。必須在 import faster_whisper 之前。
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

from faster_whisper import WhisperModel
import numpy as np
import librosa
import soundfile as sf  # 若未使用可移除
import re
import cn2an
import opencc
import unicodedata

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Header,
    HTTPException,
    status,
    Depends,
    Security,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

sys.path.append(os.path.dirname(__file__))
from auth_shared import (
    verify_jwt_token,
)
from auth_api import router as auth_router, auth_startup
import punctuation as punctuation_module

BASE_DIR = Path(__file__).parent

# 任務/狀態常數
TASK_DB_PATH = os.getenv(
    "ASR_API_AUTH_DB", os.path.join(os.path.dirname(__file__), "auth.db")
)

STATUS_WAIT_CONFIRM = 0
STATUS_SUCCESS = 3
STATUS_FAILED = 4
STATUS_CANCELLED = 5
STATUS_UPLOAD_IN_PROGRESS = 10
STATUS_WAIT_TRANSCRIPT = 11
STATUS_FILE_DOWNLOADING = 12
STATUS_TRANSCRIPT_PROCESSING = 13
STATUS_AUDIO_WAITING = 20
STATUS_AUDIO_PROCESSING = 21
STATUS_AUDIO_DONE = 22
STATUS_STREAMING_RUNNING = 30
STATUS_STREAMING_SUCCESS = 31
STATUS_STREAMING_FAILED = 32
STATUS_STREAMING_EMPTY = 33


def _ensure_tasks_schema() -> None:
    os.makedirs(os.path.dirname(TASK_DB_PATH), exist_ok=True)
    with sqlite3.connect(TASK_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subtitle_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status INTEGER NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                input_filename TEXT,
                temp_path TEXT,
                result_txt_path TEXT,
                result_srt_path TEXT,
                error TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()


def _tasks_conn():
    return sqlite3.connect(TASK_DB_PATH, check_same_thread=False)


def _now_iso() -> str:
    return datetime.now().isoformat()


# 日誌設定
def setup_logging() -> logging.Logger:
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("asr_api")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "asr_api.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(file_handler)

    error_handler = RotatingFileHandler(
        os.path.join(logs_dir, "asr_api_error.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(error_handler)

    return logger


logger = setup_logging()

# FastAPI 應用
app = FastAPI(
    title="ASR File API",
    version="1.0.0",
    swagger_ui_parameters={"persistAuthorization": True},
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 整合授權路由
app.include_router(auth_router)

# Swagger/OpenAPI：宣告 Bearer 安全方案（實際驗證仍由 _require_auth 執行）
bearer_scheme = HTTPBearer(auto_error=False)


def _require_auth(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    token = credentials.credentials if credentials else None
    if not token:
        raise HTTPException(status_code=401, detail="authorization header required")
    return verify_jwt_token(token)


# OpenCC 轉換器
s2tw = opencc.OpenCC("s2tw")

# 全域模型
whisper_model: Optional[WhisperModel] = None


def _resolve_device_compute() -> "tuple[str, str]":
    """依 api/config.py 與 torch.cuda.is_available() 決定 device/compute_type。

    優先序：config.py 內非 'auto' 的明確值 > 自動偵測。
    """
    cfg_device = None
    cfg_compute = None
    try:
        import config as app_config  # api/config.py（同目錄已加入 sys.path）

        cfg_device = getattr(app_config, "MODEL_DEVICE", None)
        cfg_compute = getattr(app_config, "MODEL_COMPUTE_TYPE", None)
    except Exception:
        pass

    if cfg_device in ("cpu", "cuda"):
        device = cfg_device
    else:
        try:
            import torch  # noqa: WPS433

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if cfg_compute and cfg_compute != "auto":
        compute_type = cfg_compute
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def load_model() -> bool:
    """載入 Whisper 模型；依 config.py 自動選 cuda/float16 或 cpu/int8，CUDA 失敗時 fallback CPU。

    優先複用 streaming_asr 已載入的同一份 WhisperModel，避免在 GPU 上重複佔用 ~3GB 記憶體。
    """
    global whisper_model
    if whisper_model is not None:
        return True

    # 嘗試複用 streaming pipeline 已載入的模型（同一份 models/，共享 GPU 記憶體）
    try:
        import streaming_asr  # type: ignore

        streaming_pipeline = getattr(streaming_asr, "asr_pipeline", None)
        shared = getattr(streaming_pipeline, "asr_pipeline", None) if streaming_pipeline else None
        if shared is not None:
            whisper_model = shared
            logger.info("複用 streaming 已載入的 Whisper 模型（共享 GPU 記憶體）")
            return True
    except Exception as e:
        logger.warning(f"無法共享 streaming 模型，將獨立載入：{e}")

    models_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    device, compute_type = _resolve_device_compute()
    logger.info(f"正在載入模型 (device={device}, compute_type={compute_type})...")
    logger.info(f"模型路徑: {models_path}")
    try:
        whisper_model = WhisperModel(models_path, device=device, compute_type=compute_type)
        logger.info(f"模型載入成功 ({device}/{compute_type})")
        return True
    except Exception as e:
        logger.error(f"模型載入失敗 ({device}/{compute_type}): {e}")
        # 若是 CUDA 失敗，自動退回 CPU int8
        if device == "cuda":
            try:
                logger.warning("嘗試回退到 CPU int8 ...")
                whisper_model = WhisperModel(models_path, device="cpu", compute_type="int8")
                logger.info("模型載入成功 (CPU/int8 fallback)")
                return True
            except Exception as e2:
                logger.error(f"CPU fallback 亦失敗: {e2}")
        return False


def split_sentence_to_words(text: str, is_split: bool):
    if is_split is False:
        return text
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF%]|\d+\.\d+|\d+)"
    )
    chars = pattern.split(text.strip().lower())
    return " ".join([w.strip() for w in chars if w is not None and w.strip()])


def replace_words(article: str) -> str:
    mappings = {
        "百分之十五": "15%",
        "百分之五": "5%",
        "百分之十二點五": "12.5%",
        "百分之七": "7%",
        "零八零零零九五九八": "080009598",
    }
    replaced_article = article
    for old, new in mappings.items():
        replaced_article = replaced_article.replace(old, new)
    return replaced_article


def convert_time(time_value: float) -> str:
    time_str = f"{time_value:.3f}"
    if "." in time_str:
        seconds, millisecond = time_str.split(".")
    else:
        seconds = time_str
        millisecond = "000"

    delta = timedelta(seconds=int(seconds))
    time_fmt = (datetime.min + delta).strftime("%H:%M:%S")
    t = str(time_fmt).split(":")
    return f"{':'.join([x.zfill(2) for x in t])}.{millisecond}"


def full_to_half(text: str) -> str:
    """全形 → 半形（NFKC；限 Latin/數字/標點，中文字不會被動到）。"""
    return unicodedata.normalize("NFKC", text)


def remove_special_characters_by_dataset_name(text: str) -> str:
    # 使用 raw 三引號字串避免引號轉義錯亂
    chars_to_ignore_regex_base = r"""[,"'。，^¿¡；「」《》:：＄$\[\]〜～·・‧―─–－⋯、＼【】=<>{}_〈〉　）（—『』«»→„…(),`&＆﹁﹂#＃\\!?！;]"""
    sentence = re.sub(chars_to_ignore_regex_base, "", text)
    sentence = full_to_half(sentence)
    return sentence


def chinese_number_to_arabic(text: str) -> str:
    """中文數字 → 阿拉伯數字（cn2an.transform）。

    例：'九百三十一' → '931'。解析失敗時統一吃下例外回傳原文，並抑制 cn2an 內部 warn。
    """
    if not text:
        return text
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return cn2an.transform(text, "cn2an")
    except Exception:
        return text


# 路由
@app.get("/api/health")
def health_check():
    logger.info("收到健康檢查請求")
    return {
        "status": "healthy",
        "model_loaded": whisper_model is not None,
        "timestamp": datetime.now().isoformat(),
    }


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        auth_startup()
    except Exception:
        pass
    try:
        _ensure_tasks_schema()
    except Exception:
        logger.exception("初始化任務資料表失敗")
    yield
    # shutdown (目前無需處理)


app.router.lifespan_context = lifespan


SUPPORTED_LANGUAGES = ("zh", "id")


def _resolve_language(lang: Optional[str]) -> str:
    if not lang:
        return "zh"
    code = str(lang).strip().lower()
    return code if code in SUPPORTED_LANGUAGES else "zh"


@app.post("/api/v1/subtitle/tasks")
async def create_subtitle_task(
    audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default="zh"),
    _: dict = Depends(_require_auth),
):
    """建立任務，背景處理音檔並產出字幕。"""
    logger.info("收到建立字幕任務請求")
    try:
        allowed_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac"}
        _, ext = os.path.splitext(audio.filename or "")
        if ext.lower() not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"不支援的檔案格式。支援的格式: {', '.join(sorted(allowed_extensions))}"
                },
            )

        # 準備儲存位置
        tasks_root = os.path.join(BASE_DIR, "audio_files", "tasks")
        os.makedirs(tasks_root, exist_ok=True)
        task_uuid = str(uuid.uuid4())
        task_dir = os.path.join(tasks_root, task_uuid)
        os.makedirs(task_dir, exist_ok=True)
        temp_file_path = os.path.join(task_dir, f"input{ext}")

        # 記錄任務（上傳中）
        with _tasks_conn() as conn:
            cur = conn.execute(
                "INSERT INTO subtitle_tasks (status, progress, input_filename, temp_path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    STATUS_UPLOAD_IN_PROGRESS,
                    0,
                    audio.filename or "",
                    temp_file_path,
                    _now_iso(),
                    _now_iso(),
                ),
            )
            task_id = cur.lastrowid
            conn.commit()

        # 儲存檔案
        try:
            with open(temp_file_path, "wb") as out:
                out.write(await audio.read())
        except Exception as e:
            with _tasks_conn() as conn:
                conn.execute(
                    "UPDATE subtitle_tasks SET status=?, error=?, updated_at=? WHERE id=?",
                    (STATUS_FAILED, f"upload failed: {e}", _now_iso(), task_id),
                )
                conn.commit()
            return JSONResponse(
                status_code=500, content={"error": f"檔案儲存失敗: {e}"}
            )

        # 更新為等待處理
        with _tasks_conn() as conn:
            conn.execute(
                "UPDATE subtitle_tasks SET status=?, progress=?, updated_at=? WHERE id=?",
                (STATUS_AUDIO_WAITING, 0, _now_iso(), task_id),
            )
            conn.commit()

        # 背景處理（sync function，透過 asyncio.to_thread 跑在 thread pool 避免阻塞 event loop）
        def _worker(_task_id: int, _file_path: str, _ref_text: Optional[str], _lang: str):
            try:
                with _tasks_conn() as conn:
                    conn.execute(
                        "UPDATE subtitle_tasks SET status=?, progress=?, updated_at=? WHERE id=?",
                        (STATUS_AUDIO_PROCESSING, 5, _now_iso(), _task_id),
                    )
                    conn.commit()

                # 確保模型載入
                if not load_model():
                    raise RuntimeError("模型載入失敗")

                # 執行轉錄（保留 segments 以產生 SRT，並逐段更新進度）
                try:
                    audio_data, _sr = librosa.load(_file_path, sr=16000, mono=True)
                    audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)

                    # 載入完音檔後拉到 10%，讓使用者看到「進入轉錄階段」
                    with _tasks_conn() as conn:
                        conn.execute(
                            "UPDATE subtitle_tasks SET progress=?, updated_at=? WHERE id=?",
                            (10, _now_iso(), _task_id),
                        )
                        conn.commit()

                    segs, info = whisper_model.transcribe(
                        audio_data,
                        language=_lang,
                        word_timestamps=False,
                        vad_filter=True,
                        beam_size=5,
                        condition_on_previous_text=True,
                        initial_prompt="",
                    )
                    total_duration = float(getattr(info, "duration", 0) or 0)

                    # faster-whisper 回傳 generator；逐段 materialize 以即時更新進度
                    segments_list = []
                    last_written = 10
                    last_write_ts = time.monotonic()
                    for seg in segs:
                        segments_list.append(seg)
                        if total_duration > 0:
                            seg_end = float(getattr(seg, "end", 0.0) or 0.0)
                            ratio = max(0.0, min(1.0, seg_end / total_duration))
                            # 將 10%~95% 區間映射到實際處理進度，留 5% 給後處理/落檔
                            progress_now = 10 + int(ratio * 85)
                        else:
                            progress_now = min(95, last_written + 1)
                        # 節流：進度增 ≥ 2 或距離上次寫入 ≥ 0.5s 才寫 DB
                        if progress_now - last_written >= 2 or (time.monotonic() - last_write_ts) >= 0.5:
                            try:
                                with _tasks_conn() as conn:
                                    conn.execute(
                                        "UPDATE subtitle_tasks SET progress=?, updated_at=? WHERE id=?",
                                        (progress_now, _now_iso(), _task_id),
                                    )
                                    conn.commit()
                            except Exception:
                                pass
                            last_written = progress_now
                            last_write_ts = time.monotonic()

                    # 轉錄完成，進入後處理（標點）階段，標 95%
                    with _tasks_conn() as conn:
                        conn.execute(
                            "UPDATE subtitle_tasks SET progress=?, updated_at=? WHERE id=?",
                            (95, _now_iso(), _task_id),
                        )
                        conn.commit()
                except Exception as e:
                    raise RuntimeError(f"轉錄失敗: {e}")

                # 取出每個 Whisper segment 的純文字（去 CR/LF）。
                segment_texts: List[str] = [
                    (getattr(seg, "text", "") or "").replace("\r", " ").replace("\n", " ").strip()
                    for seg in segments_list
                ]

                # 標點符號處理：使用 zhpr（p208p2002/zh-wiki-punctuation-restore），
                # 模型約 100MB；CPU/GPU 皆能跑。任何例外回退原文，不會讓任務 fail。
                punctuated_texts: List[str] = list(segment_texts)
                if punctuation_module.is_enabled() and segment_texts:
                    try:
                        processor = punctuation_module.get_processor()

                        # 進度條從 95% 推到 99%（保留最後 1% 給落檔）
                        def _on_progress(done: int, total: int) -> None:
                            if total <= 0:
                                return
                            progress_now = 95 + int(done / total * 4)
                            try:
                                with _tasks_conn() as conn:
                                    conn.execute(
                                        "UPDATE subtitle_tasks SET progress=?, updated_at=? WHERE id=?",
                                        (progress_now, _now_iso(), _task_id),
                                    )
                                    conn.commit()
                            except Exception:
                                pass

                        punctuated_texts = processor.punctuate_segments(
                            segment_texts, progress_callback=_on_progress
                        )
                    except Exception as e:
                        logger.warning(f"標點階段失敗，輸出未加標點之原文：{e}")
                        punctuated_texts = list(segment_texts)

                # 組裝文字（使用標點後的逐段文字串接）
                full_text = "".join(punctuated_texts)

                result_txt_path = os.path.join(task_dir, f"{_task_id}.txt")
                with open(result_txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

                # 產出 SRT（嚴格符合 hh:mm:ss,mmm 並處理毫秒進位、CRLF 換行）
                result_srt_path = os.path.join(task_dir, f"{_task_id}.srt")
                try:

                    def fmt_ts(t: float) -> str:
                        if t is None:
                            t = 0.0
                        if t < 0:
                            t = 0.0
                        total_ms = int(round(float(t) * 1000))
                        hours = total_ms // 3600000
                        total_ms %= 3600000
                        minutes = total_ms // 60000
                        total_ms %= 60000
                        seconds = total_ms // 1000
                        ms = total_ms % 1000
                        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

                    with open(
                        result_srt_path, "w", encoding="utf-8", newline="\r\n"
                    ) as srt:
                        for idx, (seg, text_line) in enumerate(
                            zip(segments_list, punctuated_texts), start=1
                        ):
                            start_ts = fmt_ts(getattr(seg, "start", 0.0))
                            end_ts = fmt_ts(getattr(seg, "end", 0.0))
                            text_line = (text_line or "").replace("\r", " ").replace("\n", " ").strip()
                            srt.write(f"{idx}\r\n")
                            srt.write(f"{start_ts} --> {end_ts}\r\n")
                            srt.write(f"{text_line}\r\n\r\n")
                except Exception as e:
                    # 若 SRT 失敗，記錄錯誤但不中斷 TXT 產出
                    logger.warning(f"SRT 產生失敗: {e}")

                # 更新資料庫完成（status=3 表示任務最終成功，符合 README 規格）
                with _tasks_conn() as conn:
                    conn.execute(
                        "UPDATE subtitle_tasks SET status=?, progress=?, result_txt_path=?, result_srt_path=?, updated_at=? WHERE id=?",
                        (
                            STATUS_SUCCESS,
                            100,
                            result_txt_path,
                            result_srt_path,
                            _now_iso(),
                            _task_id,
                        ),
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"任務 {_task_id} 處理失敗: {e}")
                with _tasks_conn() as conn:
                    conn.execute(
                        "UPDATE subtitle_tasks SET status=?, error=?, updated_at=? WHERE id=?",
                        (STATUS_FAILED, str(e), _now_iso(), _task_id),
                    )
                    conn.commit()

        resolved_lang = _resolve_language(language)
        try:
            asyncio.create_task(
                asyncio.to_thread(
                    _worker, task_id, temp_file_path, reference_text, resolved_lang
                )
            )
        except Exception as e:
            logger.error(f"背景任務建立失敗: {e}")
            with _tasks_conn() as conn:
                conn.execute(
                    "UPDATE subtitle_tasks SET status=?, error=?, updated_at=? WHERE id=?",
                    (
                        STATUS_FAILED,
                        f"background start failed: {e}",
                        _now_iso(),
                        task_id,
                    ),
                )
                conn.commit()
            return JSONResponse(
                status_code=500, content={"error": f"背景任務建立失敗: {e}"}
            )

        return {"code": 200, "message": "created", "id": task_id}

    except Exception as e:
        logger.error(f"建立字幕任務時發生錯誤: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"伺服器錯誤: {str(e)}"})


@app.post("/api/v1/subtitle/tasks/{task_id}")
async def get_task_status(task_id: int, _: dict = Depends(_require_auth)):
    try:
        with _tasks_conn() as conn:
            cur = conn.execute(
                "SELECT status, progress FROM subtitle_tasks WHERE id=?", (task_id,)
            )
            row = cur.fetchone()
            if not row:
                return JSONResponse(
                    status_code=404, content={"error": "task not found"}
                )
            status_val, progress_val = row
        return {
            "code": 200,
            "data": [{"status": int(status_val), "progress": int(progress_val)}],
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"查詢任務狀態失敗: {e}"}
        )


@app.get("/api/v1/subtitle/tasks/{task_id}/subtitle-types")
async def get_subtitle_types(task_id: int, _: dict = Depends(_require_auth)):
    """查詢指定任務可用之字幕格式（TXT/SRT/DIA）。"""
    try:
        with _tasks_conn() as conn:
            cur = conn.execute(
                "SELECT result_txt_path, result_srt_path FROM subtitle_tasks WHERE id=?",
                (task_id,),
            )
            row = cur.fetchone()
            if not row:
                return JSONResponse(
                    status_code=404, content={"error": "task not found"}
                )
            txt_path, srt_path = row

        types: List[str] = []
        if txt_path and os.path.exists(txt_path):
            types.append("TXT")
        if srt_path and os.path.exists(srt_path):
            types.append("SRT")
            # 目前 DIA 與語者標示服務尚未整合，暫以 SRT 檔存在作為可提供 DIA 文本之指標
            types.append("DIA")

        return {"code": 200, "data": [{"id": task_id, "types": types}]}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"查詢字幕格式失敗: {e}"}
        )


def _resolve_type_param(type_param: Optional[str]) -> str:
    if type_param is None:
        return "TXT"
    t = str(type_param).strip().upper()
    if t in ("1", "TXT", "TEXT"):
        return "TXT"
    if t in ("2", "SRT"):
        return "SRT"
    if t in ("3", "DIA"):
        return "DIA"
    return "TXT"


@app.get("/api/v1/subtitle/tasks/{task_id}/subtitle-link")
async def get_subtitle_link(
    task_id: int, type: Optional[str] = None, _: dict = Depends(_require_auth)
):
    try:
        subtype = _resolve_type_param(type)
        # 提供下載端點 URL
        url = f"/api/v1/subtitle/tasks/{task_id}/subtitle?type={subtype}"
        return {"code": 200, "data": [{"id": task_id, "type": subtype, "url": url}]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"產生連結失敗: {e}"})


@app.get("/api/v1/subtitle/tasks/{task_id}/subtitle")
async def download_subtitle(
    task_id: int, type: Optional[str] = None, _: dict = Depends(_require_auth)
):
    try:
        subtype = _resolve_type_param(type)
        with _tasks_conn() as conn:
            cur = conn.execute(
                "SELECT result_txt_path, result_srt_path FROM subtitle_tasks WHERE id=?",
                (task_id,),
            )
            row = cur.fetchone()
            if not row:
                return JSONResponse(
                    status_code=404, content={"error": "task not found"}
                )
            txt_path, srt_path = row
        if subtype == "TXT":
            target = txt_path
            media_type = "text/plain"
        elif subtype == "SRT":
            target = srt_path
            media_type = "application/x-subrip"
        elif subtype == "DIA":
            target = srt_path
            media_type = "text/plain"
        else:
            target = txt_path
            media_type = "text/plain"
        if not target or not os.path.exists(target):
            return JSONResponse(
                status_code=404, content={"error": f"{subtype} not available"}
            )
        return FileResponse(
            path=target, media_type=media_type, filename=os.path.basename(target)
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"下載失敗: {e}"})


STATIC_DIR = BASE_DIR / "static"


@app.get("/")
def get_index_html():
    """根路徑導引頁：列出兩種辨識服務、健康檢查與 Swagger 文件入口。"""
    index_file = STATIC_DIR / "index.html"
    try:
        if index_file.exists():
            return FileResponse(str(index_file), media_type="text/html")
        return JSONResponse(
            status_code=404, content={"error": "index.html 不存在"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"讀取 index.html 發生錯誤: {e}"}
        )


@app.get("/test_files.html")
def get_test_files_html():
    """回傳 static/test_files.html (健康檢查/模型資訊/單一音檔)"""
    test_file = STATIC_DIR / "test_files.html"
    try:
        if test_file.exists():
            return FileResponse(str(test_file), media_type="text/html")
        return JSONResponse(
            status_code=404, content={"error": "test_files.html 不存在"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"讀取 test_files.html 發生錯誤: {e}"}
        )


@app.get("/test_realtime.html")
async def get_test_realtime_html():
    """回傳 static/test_realtime.html (即時辨識頁)"""
    test_file = STATIC_DIR / "test_realtime.html"
    try:
        if test_file.exists():
            return FileResponse(str(test_file), media_type="text/html")
        return JSONResponse(
            status_code=404, content={"error": "test_realtime.html 不存在"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"讀取 test_realtime.html 發生錯誤: {e}"}
        )


def main():
    logger.info("啟動 FastAPI File ASR 服務器...")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")


if __name__ == "__main__":
    logger.info("正在載入模型...")
    load_model()
    main()
