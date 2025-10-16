import logging
from .simple_vad import SimpleVAD

logger = logging.getLogger(__name__)


class VADFactory:
    """
    Factory for creating instances of VAD systems.
    """

    @staticmethod
    def create_vad_pipeline(type, **kwargs):
        """
        Creates a VAD pipeline based on the specified type.

        Args:
            type (str): The type of VAD pipeline to create (e.g., 'simple').
            kwargs: Additional arguments for the VAD pipeline creation.

        Returns:
            VADInterface: An instance of a class that implements VADInterface.
        """
        logger.info(f"正在創建 VAD 管道，類型: {type}")
        logger.info(f"VAD 參數: {kwargs}")

        if type == "simple":
            logger.info("使用 Simple VAD 管道")
            try:
                vad_pipeline = SimpleVAD(**kwargs)
                logger.info("✅ Simple VAD 管道創建成功")
                return vad_pipeline
            except Exception as e:
                logger.error(f"❌ Simple VAD 管道創建失敗: {e}")
                import traceback

                logger.error("詳細錯誤信息:")
                for line in traceback.format_exc().split("\n"):
                    logger.error(f"  {line}")
                raise
        else:
            error_msg = f"不支援的 VAD 管道類型: {type}。目前只支援 'simple'"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
