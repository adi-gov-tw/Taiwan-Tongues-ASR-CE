import os
import sys
from faster_whisper import WhisperModel

from .asr_interface import ASRInterface
from audio_utils import save_audio_to_file
import logging
import time
import torch
from utils import language_codes, filter_text

logger = logging.getLogger(__name__)


class FasterWhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        logger.info("=" * 40)
        logger.info("初始化 Faster Whisper ASR...")
        logger.info("=" * 40)

        model_size = kwargs.get("model_size", "large-v3-turbo")
        logger.info(f"模型類型: {model_size}")

        # 修正模型路徑計算
        # 從 stt_streaming/src/asr/faster_whisper_asr.py 到 models 目錄的正確路徑
        current_file = os.path.abspath(
            __file__
        )  # stt_streaming/src/asr/faster_whisper_asr.py
        asr_dir = os.path.dirname(current_file)  # stt_streaming/src/asr/
        src_dir = os.path.dirname(asr_dir)  # stt_streaming/src/
        stt_dir = os.path.dirname(src_dir)  # stt_streaming/
        api_dir = os.path.dirname(stt_dir)  # api/
        project_root = os.path.dirname(api_dir)  # 訓練和推論腳本/
        model_path = os.path.join(project_root, model_size)
        if os.path.exists(model_path):
            logger.info(f"模型路徑: {model_path}")
            # 檢查模型文件
            required_files = ["model.bin", "config.json", "tokenizer.json"]
            missing_files = []
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.info(f"✅ {file} 存在 ({file_size:,} bytes)")
                else:
                    logger.error(f"❌ {file} 不存在")
                    missing_files.append(file)
            if missing_files:
                error_msg = f"缺少模型文件: {', '.join(missing_files)}"
                logger.error(f"❌ {error_msg}")
                raise FileNotFoundError(error_msg)
        else:
            model_path = model_size

        logger.info(f"專案根目錄: {project_root}")

        # 讀取 config 設定，若無則自動偵測
        try:
            # 將 api 目錄加入路徑以導入 config
            if api_dir not in sys.path:
                sys.path.append(api_dir)
            try:
                import config as app_config

                cfg_device = getattr(app_config, "MODEL_DEVICE", None)
                cfg_compute = getattr(app_config, "MODEL_COMPUTE_TYPE", None)
            except Exception:
                cfg_device = None
                cfg_compute = None

            # 選擇 device
            if cfg_device in ("cpu", "cuda"):
                device = cfg_device
                logger.info(f"使用 config 指定的設備: {device}")
            else:
                try:
                    if torch.cuda.is_available():
                        logger.info(
                            f"✅ CUDA 可用，GPU 數量: {torch.cuda.device_count()}"
                        )
                        for i in range(torch.cuda.device_count()):
                            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                        device = "cuda"
                    else:
                        logger.warning("⚠️ CUDA 不可用，將使用 CPU")
                        device = "cpu"
                except Exception as e:
                    logger.warning(f"⚠️ 無法檢查 CUDA 狀態: {e}，將使用 CPU")
                    device = "cpu"

            # 選擇 compute type（若未指定）
            if cfg_compute is not None:
                compute_type = cfg_compute
            else:
                compute_type = "float16" if device == "cuda" else "int8"
        except Exception as e:
            logger.warning(f"⚠️ 讀取模型設定時發生問題，改用預設：{e}")
            device = "cpu"
            compute_type = "int8"

        # 嘗試載入模型，若 GPU 失敗則回退到 CPU int8
        try:
            logger.info(
                f"正在載入 Whisper 模型 (設備: {device}, 計算類型: {compute_type})..."
            )
            # 以雲端權重或本地目錄載入，僅傳入支援的參數（語言於 transcribe() 指定）
            self.asr_pipeline = WhisperModel(
                model_path, device=device, compute_type=compute_type
            )
            # 暴露關鍵屬性以供健康檢查
            self.device = device
            self.compute_type = compute_type
            self.model_size = model_size
            self.model_path = model_path
            logger.info("✅ Whisper 模型載入成功")
        except Exception as e:
            logger.error(f"❌ Whisper 模型載入失敗: {e}")
            import traceback

            for line in traceback.format_exc().split("\n"):
                logger.error(f"  {line}")
            # 若是使用 CUDA 失敗，嘗試回退到 CPU int8
            if device == "cuda":
                try:
                    logger.warning("嘗試回退到 CPU int8 ...")
                    self.asr_pipeline = WhisperModel(
                        model_path, device="cpu", compute_type="int8"
                    )
                    logger.info("✅ Whisper 模型在 CPU int8 模式載入成功")
                except Exception as e2:
                    logger.error(f"❌ 回退 CPU 載入亦失敗: {e2}")
                    for line in traceback.format_exc().split("\n"):
                        logger.error(f"  {line}")
                    raise
            else:
                raise

        # 依 asr_core 的穩定參數調整預設轉錄配置
        self.default_transcribe_kwargs = {
            # 保留 word_timestamps 以便回傳逐字詞與時間
            "word_timestamps": False,
            # 串流情境下改為關閉內建 VAD，避免將整段視為靜音被移除
            # 由外部 VAD/緩衝策略決定觸發時機
            "vad_filter": True,
            "beam_size": 5,
            "condition_on_previous_text": True,
            # 與 asr_core 一致，避免模型 bias 特定提示
            "initial_prompt": "繁體中文",
        }

    async def transcribe(self, client):
        logger.debug(f"開始轉錄音頻，客戶端 ID: {client.client_id}")

        try:
            file_path = await save_audio_to_file(
                client.scratch_buffer, client.get_file_name()
            )
            logger.debug(f"音頻文件已保存: {file_path}")

            # 預設語言為 zh，若客戶端有指定則映射
            # 語言固定 zh
            language = "zh"
            logger.debug(f"語言設定: {language}")

            logger.debug("開始轉錄...")
            # 合併預設參數與語言設定
            transcribe_kwargs = dict(self.default_transcribe_kwargs)
            # 明確指定語言（預設 zh）
            transcribe_kwargs["language"] = language
            segments, info = self.asr_pipeline.transcribe(
                file_path, **transcribe_kwargs
            )

            segments = list(segments)
            logger.debug(f"轉錄完成，段落數量: {len(segments)}")

            # 清理臨時文件
            try:
                os.remove(file_path)
                logger.debug("臨時文件已清理")
            except Exception as e:
                logger.warning(f"清理臨時文件失敗: {e}")

            if len(segments) == 0:
                logger.debug(
                    "沒有檢測到語音內容（首次轉錄）。嘗試關閉 VAD 後重試以避免被過度過濾。"
                )
                try:
                    retry_kwargs = dict(transcribe_kwargs)
                    retry_kwargs["vad_filter"] = False
                    segments_retry, info_retry = self.asr_pipeline.transcribe(
                        file_path, **retry_kwargs
                    )
                    segments = list(segments_retry)
                    info = info_retry
                    logger.debug(f"重試後段落數量: {len(segments)}")
                except Exception as _:
                    pass
                if len(segments) == 0:
                    return None

            # 組合文字
            text = " ".join([getattr(s, "text", "").strip() for s in segments])
            logger.debug(f"轉錄文本: {text}")

            # 不再因語言機率低而直接放棄，僅記錄警告
            try:
                if getattr(info, "language_probability", 1.0) < 0.5:
                    logger.warning(f"語言概率偏低: {info.language_probability}")
            except Exception:
                pass

            text = filter_text(text)
            if text is None:
                # 若過濾後為 None，退回原始文本，避免整段丟失
                logger.debug("文本被過濾器過濾，退回原始文本")
                text = " ".join([getattr(s, "text", "").strip() for s in segments])

            logger.debug(f"最終文本: {text}")

            # 構造 words 陣列（若 word_timestamps 不可用，提供空陣列以維持回傳結構）
            flattened_words = []
            try:
                for segment in segments:
                    if hasattr(segment, "words") and segment.words:
                        flattened_words.extend(segment.words)
            except Exception:
                pass

            # 計算持續時間
            duration = None
            try:
                if flattened_words:
                    duration = flattened_words[-1].end
                elif segments:
                    duration = getattr(segments[-1], "end", None)
            except Exception:
                duration = None

            to_return = {
                "language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
                "final": True,
                "text": text,
                "duration": duration,
                "words": [
                    {
                        "word": getattr(w, "word", ""),
                        "start": (getattr(w, "start", 0) or 0) + client.last_start_time,
                        "end": (getattr(w, "end", 0) or 0) + client.last_start_time,
                        "probability": getattr(w, "probability", None),
                    }
                    for w in flattened_words
                ],
            }

            logger.debug(f"轉錄結果: {to_return}")
            return to_return

        except Exception as e:
            logger.error(f"❌ 轉錄過程中發生錯誤: {e}")
            import traceback

            logger.error("詳細錯誤信息:")
            for line in traceback.format_exc().split("\n"):
                logger.error(f"  {line}")
            return None

    def warm_up(self):
        logger.info("正在預熱 ASR 管道...")

        # 修正預熱文件路徑
        current_file = os.path.abspath(
            __file__
        )  # stt_streaming/src/asr/faster_whisper_asr.py
        asr_dir = os.path.dirname(current_file)  # stt_streaming/src/asr/
        src_dir = os.path.dirname(asr_dir)  # stt_streaming/src/
        stt_dir = os.path.dirname(src_dir)  # stt_streaming/
        warm_up_file = os.path.join(stt_dir, "warm_up.wav")

        logger.info(f"預熱文件路徑: {warm_up_file}")

        if not os.path.exists(warm_up_file):
            logger.warning(f"⚠️ 預熱文件不存在: {warm_up_file}")
            return

        try:
            logger.info("執行預熱轉錄...")
            segments, info = self.asr_pipeline.transcribe(
                warm_up_file,
                word_timestamps=True,
                language="zh",
                initial_prompt="繁體中文",
            )
            text = " ".join([s.text.strip() for s in list(segments)])
            logger.info(f"✅ ASR 管道預熱完成: {text}")
        except Exception as e:
            logger.error(f"❌ ASR 管道預熱失敗: {e}")
            import traceback

            logger.error("詳細錯誤信息:")
            for line in traceback.format_exc().split("\n"):
                logger.error(f"  {line}")
