import argparse, os
import asyncio
import json
import logging
import sys
import traceback

# 添加當前目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from server import Server
from asr.asr_factory import ASRFactory
from vad.vad_factory import VADFactory
from logging.handlers import TimedRotatingFileHandler


# 創建一個簡單的配置
class Settings:
    def __init__(self):
        self.base_dir = (
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
        )


settings = Settings()

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# 確保日誌目錄存在 - 統一寫到 api/logs
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs"
)
os.makedirs(log_dir, exist_ok=True)

root_logger = logging.getLogger()
root_logger.handlers = []
log_formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s"
)
log_file = os.path.join(log_dir, "stt_streaming_main.log")
log_handler = TimedRotatingFileHandler(
    log_file, when="H", interval=1, backupCount=24, encoding="utf-8"
)
logger = logging.getLogger("my_logger")
log_handler.setFormatter(log_formatter)
root_logger.addHandler(log_handler)
root_logger.setLevel(logging.INFO)


def check_dependencies():
    """檢查必要的依賴套件"""
    logger.info("=" * 40)
    logger.info("檢查 STT Streaming 依賴套件...")
    logger.info("=" * 40)

    required_modules = [
        "websockets",
        "faster_whisper",
        "transformers",
        "torch",
        "numpy",
        "librosa",
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module} 已安裝")
        except ImportError as e:
            logger.error(f"❌ {module} 未安裝: {e}")
            missing_modules.append(module)

    if missing_modules:
        logger.error(f"缺少以下模組: {', '.join(missing_modules)}")
        logger.error("請安裝缺少的依賴套件")
        return False

    logger.info("✅ 所有依賴套件檢查完成")
    return True


def check_models():
    """檢查模型文件"""
    logger.info("=" * 40)
    logger.info("檢查模型文件...")
    logger.info("=" * 40)

    # 修正模型目錄路徑計算
    # 從 stt_streaming/src/main.py 到 models 目錄的正確路徑
    current_file = os.path.abspath(__file__)  # stt_streaming/src/main.py
    src_dir = os.path.dirname(current_file)  # stt_streaming/src/
    stt_dir = os.path.dirname(src_dir)  # stt_streaming/
    api_dir = os.path.dirname(stt_dir)  # api/
    project_root = os.path.dirname(api_dir)  # 訓練和推論腳本/
    models_dir = os.path.join(project_root, "models")

    logger.info(f"專案根目錄: {project_root}")
    logger.info(f"模型目錄: {models_dir}")

    if not os.path.exists(models_dir):
        logger.error(f"❌ 模型目錄不存在: {models_dir}")
        return False

    # 檢查必要的模型文件
    required_files = ["model.bin", "config.json", "tokenizer.json"]
    missing_files = []

    for file in required_files:
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.info(f"✅ {file} 存在 ({file_size:,} bytes)")
        else:
            logger.error(f"❌ {file} 不存在")
            missing_files.append(file)

    if missing_files:
        logger.error(f"缺少以下模型文件: {', '.join(missing_files)}")
        return False

    logger.info("✅ 模型文件檢查完成")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="VoiceStreamAI Server: Real-time audio transcription using self-hosted Whisper and WebSocket"
    )
    parser.add_argument(
        "--vad-type",
        type=str,
        default="simple",
        help="Type of VAD pipeline to use (e.g., 'simple')",
    )
    parser.add_argument(
        "--vad-args",
        type=str,
        default='{"min_duration": 0.1}',
        help="JSON string of additional arguments for VAD pipeline",
    )
    parser.add_argument(
        "--asr-type",
        type=str,
        default="faster_whisper",
        help="Type of ASR pipeline to use (e.g., 'faster_whisper')",
    )
    parser.add_argument(
        "--asr-args",
        type=str,
        default='{"model_size": "large-v3-turbo"}',
        help="JSON string of additional arguments for ASR pipeline",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host for the WebSocket server"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port for the WebSocket server"
    )
    return parser.parse_args()


def main():
    logger.info("=" * 50)
    logger.info("STT Streaming 服務器啟動")
    logger.info("=" * 50)

    try:
        # 檢查依賴套件
        if not check_dependencies():
            logger.error("依賴套件檢查失敗，服務器無法啟動")
            return

        # 檢查模型文件
        if not check_models():
            logger.error("模型文件檢查失敗，服務器無法啟動")
            return

        args = parse_args()
        logger.info(f"啟動參數: host={args.host}, port={args.port}")
        logger.info(f"VAD 類型: {args.vad_type}")
        logger.info(f"ASR 類型: {args.asr_type}")

        try:
            vad_args = json.loads(args.vad_args)
            asr_args = json.loads(args.asr_args)
            logger.info(f"VAD 參數: {vad_args}")
            logger.info(f"ASR 參數: {asr_args}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 參數解析錯誤: {e}")
            return

        logger.info("正在初始化 VAD 管道...")
        try:
            vad_pipeline = VADFactory.create_vad_pipeline(args.vad_type, **vad_args)
            logger.info("✅ VAD 管道初始化成功")
        except Exception as e:
            logger.error(f"❌ VAD 管道初始化失敗: {e}")
            logger.error("詳細錯誤信息:")
            for line in traceback.format_exc().split("\n"):
                logger.error(f"  {line}")
            return

        logger.info("正在初始化 ASR 管道...")
        try:
            asr_pipeline = ASRFactory.create_asr_pipeline(args.asr_type, **asr_args)
            logger.info("✅ ASR 管道初始化成功")
        except Exception as e:
            logger.error(f"❌ ASR 管道初始化失敗: {e}")
            logger.error("詳細錯誤信息:")
            for line in traceback.format_exc().split("\n"):
                logger.error(f"  {line}")
            return

        logger.info("正在創建 WebSocket 服務器...")
        try:
            server = Server(
                vad_pipeline,
                asr_pipeline,
                host=args.host,
                port=args.port,
                sampling_rate=16000,
                samples_width=2,
            )
            logger.info("✅ WebSocket 服務器創建成功")
        except Exception as e:
            logger.error(f"❌ WebSocket 服務器創建失敗: {e}")
            logger.error("詳細錯誤信息:")
            for line in traceback.format_exc().split("\n"):
                logger.error(f"  {line}")
            return

        logger.info("正在啟動服務器...")
        try:
            # 使用新的 asyncio API
            async def run_server():
                await server.start()
                logger.info("✅ 服務器啟動成功")
                logger.info(f"WebSocket 服務器運行在: ws://{args.host}:{args.port}")
                # 保持服務器運行
                await asyncio.Future()  # 永不完成的 Future

            asyncio.run(run_server())
        except Exception as e:
            logger.error(f"❌ 服務器運行失敗: {e}")
            logger.error("詳細錯誤信息:")
            for line in traceback.format_exc().split("\n"):
                logger.error(f"  {line}")
            return

    except KeyboardInterrupt:
        logger.info("收到中斷信號，正在關閉服務器...")
    except Exception as e:
        logger.error(f"❌ 服務器啟動過程中發生未預期的錯誤: {e}")
        logger.error("詳細錯誤信息:")
        for line in traceback.format_exc().split("\n"):
            logger.error(f"  {line}")


if __name__ == "__main__":
    main()
