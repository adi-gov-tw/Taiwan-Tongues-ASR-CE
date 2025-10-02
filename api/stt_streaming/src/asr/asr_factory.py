import logging
from .faster_whisper_asr import FasterWhisperASR

logger = logging.getLogger(__name__)


class ASRFactory:
    @staticmethod
    def create_asr_pipeline(type, **kwargs):
        logger.info(f"正在創建 ASR 管道，類型: {type}")
        logger.info(f"ASR 參數: {kwargs}")

        if type == "faster_whisper":
            logger.info("使用 Faster Whisper ASR 管道")
            try:
                asr_pipeline = FasterWhisperASR(**kwargs)
                logger.info("✅ Faster Whisper ASR 管道創建成功")
                return asr_pipeline
            except Exception as e:
                logger.error(f"❌ Faster Whisper ASR 管道創建失敗: {e}")
                import traceback

                logger.error("詳細錯誤信息:")
                for line in traceback.format_exc().split("\n"):
                    logger.error(f"  {line}")
                raise
        else:
            error_msg = f"不支援的 ASR 管道類型: {type}。目前只支援 'faster_whisper'"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
