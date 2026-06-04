#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
環境建立與檢查腳本（不啟動服務）
自動檢測並安裝缺少的依賴套件，檢查模型/依賴完整性與基本可用性
"""

import os
import sys
import time
import logging
import urllib.request
import urllib.error
import subprocess
import webbrowser
import importlib.util
from pathlib import Path
import threading
import atexit
from typing import List, Tuple, Dict

# 設定日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 全域子進程註冊表，確保在程式結束或控制台關閉時能統一清理
PROCESS_LIST: List[Tuple[str, subprocess.Popen]] = []


def register_process(name: str, process: subprocess.Popen) -> None:
    PROCESS_LIST.append((name, process))


def terminate_all_processes() -> None:
    for name, process in list(PROCESS_LIST):
        try:
            if process.poll() is None:
                logger.info(f"正在停止 {name} 服務 (PID: {process.pid})...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                    logger.info(f"✅ {name} 服務已停止")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ {name} 服務無法正常停止，強制終止...")
                    process.kill()
                    logger.warning(f"✅ {name} 服務強制停止")
        except Exception as e:
            logger.error(f"❌ 停止 {name} 服務時發生錯誤: {e}")
    PROCESS_LIST.clear()


# 無論如何離開程式都嘗試清理子進程
atexit.register(terminate_all_processes)

# Windows: 捕捉控制台關閉事件（如關閉視窗、登出、關機），優雅終止子進程
if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    # Windows 控制台事件常數
    CTRL_C_EVENT = 0
    CTRL_BREAK_EVENT = 1
    CTRL_CLOSE_EVENT = 2
    CTRL_LOGOFF_EVENT = 5
    CTRL_SHUTDOWN_EVENT = 6

    def _console_ctrl_handler(ctrl_type: int) -> bool:
        # 對 Ctrl+C / Ctrl+Break：不要攔截，讓預設處理（KeyboardInterrupt）觸發，
        # 以便 main() 的 except KeyboardInterrupt 跑 terminate_all_processes()
        if ctrl_type in (CTRL_C_EVENT, CTRL_BREAK_EVENT):
            return False  # 交由預設處理（會導致 KeyboardInterrupt）

        # 對關閉視窗/登出/關機事件：主動清理並吞掉事件，避免殘留子進程
        logger.info(f"收到控制台關閉事件: {ctrl_type}，準備關閉服務...")
        try:
            terminate_all_processes()
        finally:
            time.sleep(0.5)
        return True

    HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    _HANDLER_INSTANCE = HandlerRoutine(_console_ctrl_handler)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_HANDLER_INSTANCE, True)


def check_python_and_pip():
    """檢查Python版本和pip可用性"""
    logger.info("檢查Python環境...")

    # 檢查Python版本（本專案要求 Python 3.10 以上）
    if sys.version_info[:2] < (3, 10):
        logger.error("❌ 需要 Python 3.10 或更高版本")
        logger.error(f"當前版本: {sys.version}")
        return False

    logger.info(f"✅ Python版本: {sys.version.split()[0]}")

    # 檢查pip是否可用
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("✅ pip可用")
        return True
    except subprocess.CalledProcessError:
        logger.error("❌ pip不可用，請確保pip已正確安裝")
        return False


def activate_virtual_environment():
    """檢測並啟動虛擬環境"""
    logger.info("檢測虛擬環境...")

    # 檢查當前目錄是否有虛擬環境
    venv_paths = [
        "asr_api",  # 當前目錄下的虛擬環境
        "../asr_api",  # 父目錄下的虛擬環境
        "venv",
        "env",
        ".venv",
    ]

    for venv_path in venv_paths:
        if os.path.exists(venv_path):
            # 檢查是否是有效的虛擬環境
            if os.path.exists(
                os.path.join(venv_path, "Scripts", "activate.bat")
            ) or os.path.exists(os.path.join(venv_path, "bin", "activate")):
                logger.info(f"找到虛擬環境: {venv_path}")

                # 設定環境變數
                if os.name == "nt":  # Windows
                    python_path = os.path.join(venv_path, "Scripts", "python.exe")
                    if os.path.exists(python_path):
                        os.environ["VIRTUAL_ENV"] = os.path.abspath(venv_path)
                        sys.executable = python_path
                        logger.info(f"✅ 已啟動虛擬環境: {venv_path}")
                        return True
                else:  # Unix/Linux
                    python_path = os.path.join(venv_path, "bin", "python")
                    if os.path.exists(python_path):
                        os.environ["VIRTUAL_ENV"] = os.path.abspath(venv_path)
                        sys.executable = python_path
                        logger.info(f"✅ 已啟動虛擬環境: {venv_path}")
                        return True

    logger.warning("未找到虛擬環境，將使用系統Python")
    return False


def install_package(package_name):
    """安裝單個套件"""
    try:
        logger.info(f"正在安裝 {package_name}...")

        # 對於某些套件，使用更長的超時時間
        timeout = 600 if package_name in ["ml_dtypes", "pyannote-audio"] else 300

        # 對於 torch，使用特定版本
        if package_name == "torch":
            install_cmd = [sys.executable, "-m", "pip", "install", "torch>=2.1.0"]
        else:
            install_cmd = [sys.executable, "-m", "pip", "install", package_name]

        result = subprocess.run(
            install_cmd, capture_output=True, text=True, timeout=timeout
        )

        if result.returncode == 0:
            logger.info(f"✅ {package_name} 安裝成功")

            # 如果是 torch，檢查版本
            if package_name == "torch":
                try:
                    import torch

                    torch_version = torch.__version__
                    version_parts = torch_version.split(".")
                    if len(version_parts) >= 2:
                        major = int(version_parts[0])
                        minor = int(version_parts[1])
                        if major < 2 or (major == 2 and minor < 1):
                            logger.warning(f"⚠️ 安裝的 torch 版本過舊 ({torch_version})")
                            logger.info("建議手動升級: pip install torch>=2.1.0")
                        else:
                            logger.info(f"✅ torch 版本符合要求 ({torch_version})")
                except Exception as e:
                    logger.warning(f"⚠️ 無法檢查安裝後的 torch 版本: {e}")

            return True
        else:
            logger.error(f"❌ {package_name} 安裝失敗:")
            logger.error(f"錯誤信息: {result.stderr}")

            # 對於某些套件，提供跳過選項
            if package_name in ["ml_dtypes", "pyannote-audio"]:
                logger.warning(f"⚠️ {package_name} 安裝失敗，這可能影響某些功能")
                logger.warning("您可以稍後手動安裝: pip install " + package_name)
                return True  # 允許繼續執行

            return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {package_name} 安裝超時")

        # 對於某些套件，提供跳過選項
        if package_name in ["ml_dtypes", "pyannote-audio"]:
            logger.warning(f"⚠️ {package_name} 安裝超時，這可能影響某些功能")
            logger.warning("您可以稍後手動安裝: pip install " + package_name)
            return True  # 允許繼續執行

        return False
    except Exception as e:
        logger.error(f"❌ {package_name} 安裝失敗: {e}")

        # 對於某些套件，提供跳過選項
        if package_name in ["ml_dtypes", "pyannote-audio"]:
            logger.warning(f"⚠️ {package_name} 安裝失敗，這可能影響某些功能")
            logger.warning("您可以稍後手動安裝: pip install " + package_name)
            return True  # 允許繼續執行

        return False


def install_requirements_file():
    """從 requirements.txt 安裝依賴"""
    requirements_files = ["requirements.txt"]

    for requirements_file in requirements_files:
        if os.path.exists(requirements_file):
            try:
                logger.info(f"從 {requirements_file} 安裝依賴套件...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", requirements_file],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )  # 10分鐘超時

                if result.returncode == 0:
                    logger.info(f"✅ 從 {requirements_file} 安裝依賴套件成功")

                    # 如果是 requirements.txt，檢查 torch 版本
                    if requirements_file == "requirements.txt":
                        try:
                            import torch

                            torch_version = torch.__version__
                            version_parts = torch_version.split(".")
                            if len(version_parts) >= 2:
                                major = int(version_parts[0])
                                minor = int(version_parts[1])
                                if major < 2 or (major == 2 and minor < 1):
                                    logger.warning(
                                        f"⚠️ 安裝的 torch 版本過舊 ({torch_version})"
                                    )
                                    logger.info(
                                        "建議手動升級: pip install torch>=2.1.0"
                                    )
                                else:
                                    logger.info(
                                        f"✅ torch 版本符合要求 ({torch_version})"
                                    )
                        except Exception as e:
                            logger.warning(f"⚠️ 無法檢查安裝後的 torch 版本: {e}")
                else:
                    logger.error(f"❌ 從 {requirements_file} 安裝失敗:")
                    logger.error(f"錯誤信息: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                logger.error(f"❌ 從 {requirements_file} 安裝超時")
                return False
            except Exception as e:
                logger.error(f"❌ 從 {requirements_file} 安裝失敗: {e}")
                return False
        else:
            logger.warning(f"找不到 {requirements_file} 文件")

    return True


def _parse_requirement_name(req_line: str) -> str:
    """解析 requirement 行並提取套件名稱（忽略版本、extras與環境標記）。"""
    line = req_line.strip()
    if not line or line.startswith("#"):
        return ""
    # 去除環境標記（; 後面）
    if ";" in line:
        line = line.split(";", 1)[0].strip()
    # 去除版本限制（==, >=, <=, ~=, !=, >, < 等）
    for sep in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
        if sep in line:
            line = line.split(sep, 1)[0].strip()
            break
    # 移除 extras 方括號
    if "[" in line:
        line = line.split("[", 1)[0].strip()
    return line


def _load_required_packages_from_requirements(
    requirements_files: List[str],
) -> Dict[str, str]:
    """從多個 requirements 檔蒐集需要檢查的套件，回傳 {package_name: import_name}。"""
    # 常見 package → import 名稱對應
    import_name_overrides = {
        "pyjwt": "jwt",
        "faster-whisper": "faster_whisper",
        "opencc-python-reimplemented": "opencc",
        "python-multipart": "multipart",
        "sentence-transformers": "sentence_transformers",
        "scikit-learn": "sklearn",
    }
    required: Dict[str, str] = {}
    for req_file in requirements_files:
        if not os.path.exists(req_file):
            continue
        try:
            with open(req_file, "r", encoding="utf-8") as f:
                for line in f:
                    name = _parse_requirement_name(line)
                    if not name:
                        continue
                    pkg = name.strip()
                    key = pkg.lower()
                    import_name = import_name_overrides.get(key, pkg.replace("-", "_"))
                    required[pkg] = import_name
        except Exception as e:
            logger.warning(f"讀取 {req_file} 失敗: {e}")
    return required


def check_and_install_dependencies():
    """檢查並安裝依賴套件"""
    logger.info("檢查依賴套件...")

    # 由 requirements 檔案動態取得需要檢查的套件
    requirements_files = ["requirements.txt"]
    required_packages = _load_required_packages_from_requirements(requirements_files)

    # 定義可選的套件（安裝失敗不會阻止程序繼續）
    optional_packages = ["ml_dtypes", "pyannote-audio"]

    missing_packages = []

    # 檢查每個套件
    for package_name, import_name in required_packages.items():
        try:
            module = __import__(import_name)

            # 特殊檢查 torch 版本
            if package_name.lower() == "torch":
                try:
                    import torch

                    torch_version = torch.__version__
                    logger.info(f"✅ {package_name} (版本: {torch_version})")

                    # 檢查 torch 版本是否 >= 2.1.0
                    version_parts = torch_version.split(".")
                    if len(version_parts) >= 2:
                        major = int(version_parts[0])
                        minor = int(version_parts[1])
                        if major < 2 or (major == 2 and minor < 1):
                            logger.warning(
                                f"⚠️ torch 版本過舊 ({torch_version})，建議升級到 2.1.0 或更高版本"
                            )
                            logger.info("建議執行: pip install torch>=2.1.0")
                        else:
                            logger.info(f"✅ torch 版本符合要求 ({torch_version})")
                    else:
                        logger.warning(f"⚠️ 無法解析 torch 版本: {torch_version}")
                except Exception as e:
                    logger.warning(f"⚠️ 無法檢查 torch 版本: {e}")
            else:
                logger.info(f"✅ {package_name}")
        except ImportError:
            if package_name in optional_packages:
                logger.warning(f"⚠️ {package_name} - 未安裝 (可選)")
            else:
                missing_packages.append(package_name)
                logger.error(f"❌ {package_name} - 未安裝")

    # 如果有缺少的套件，詢問用戶是否自動安裝
    if missing_packages:
        logger.info(f"發現缺少的套件: {', '.join(missing_packages)}")

        # 默認採用 requirements 安裝（非互動）
        logger.info("開始自動安裝缺少的套件（透過 requirements 檔）...")

        # 首先嘗試從 requirements 文件安裝
        if install_requirements_file():
            # 重新檢查是否還有缺少的套件
            still_missing = []
            for package_name, import_name in required_packages.items():
                if package_name in missing_packages:
                    try:
                        __import__(import_name)
                        logger.info(f"✅ {package_name} 安裝成功")
                    except ImportError:
                        still_missing.append(package_name)

            # 如果還有缺少的，逐個安裝
            if still_missing:
                logger.info("還有缺少的套件，逐個安裝...")
                for package in still_missing:
                    if not install_package(package):
                        logger.error(f"無法安裝 {package}，請手動安裝")
                        return False
        else:
            # 如果從 requirements 文件安裝失敗，逐個安裝
            logger.info("從 requirements 文件安裝失敗，嘗試逐個安裝...")
            for package in missing_packages:
                if not install_package(package):
                    logger.error(f"無法安裝 {package}，請手動安裝")
                    return False

    logger.info("所有依賴套件檢查完成")
    return True


def check_models_directory():
    """檢查模型目錄是否存在"""
    logger.info("檢查模型目錄...")

    # 檢查父目錄中的 models 目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    models_path = os.path.join(parent_dir, "models")

    if not os.path.exists(models_path):
        logger.error(f"❌ 找不到模型目錄: {models_path}")
        logger.error("請確保 models 目錄存在並包含 Whisper 模型檔案")
        return False

    # 檢查必要的模型文件
    required_files = ["model.bin", "config.json", "preprocessor_config.json"]
    missing_files = []

    for file_name in required_files:
        file_path = os.path.join(models_path, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)

    if missing_files:
        logger.error(f"❌ 模型目錄中缺少必要文件: {', '.join(missing_files)}")
        logger.error("請確保模型文件完整")
        return False

    logger.info(f"✅ 模型目錄檢查通過: {models_path}")
    logger.info("注意: STT Streaming 將使用 Faster Whisper 模型")
    return True


def test_vad_basic():
    """基本依賴套件測試"""
    logger.info("執行基本依賴套件測試...")

    try:
        # 測試 STT Streaming 相關套件
        import websockets

        logger.info("✅ websockets 可用")

        import faster_whisper

        logger.info("✅ faster_whisper 可用")

        # 測試 transformers (用於 VAD)
        try:
            import transformers

            logger.info("✅ transformers 可用")
        except ImportError:
            logger.warning("⚠️ transformers 不可用，某些 VAD 功能可能受影響")

        # 測試可選套件
        try:
            import ml_dtypes

            logger.info("✅ ml_dtypes 可用")
        except ImportError:
            logger.warning("⚠️ ml_dtypes 不可用，某些功能可能受影響")

        try:
            import pyannote

            logger.info("✅ pyannote-audio 可用")
        except ImportError:
            logger.warning("⚠️ pyannote-audio 不可用，將使用替代 VAD 方案")

        logger.info("✅ STT Streaming 依賴套件檢查通過")
        return True

    except Exception as e:
        logger.error(f"❌ STT Streaming 依賴套件檢查失敗: {e}")
        return False


def start_stt_streaming_server():
    """啟動 FastAPI STT Streaming 服務器"""
    logger.info("=" * 40)
    logger.info("啟動 FastAPI STT Streaming 服務器...")
    logger.info("=" * 40)

    try:
        # 檢查 FastAPI 服務器文件是否存在
        fastapi_server_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "stt_streaming_fastapi.py"
        )
        logger.info(f"檢查 FastAPI 服務器文件: {fastapi_server_path}")
        if not os.path.exists(fastapi_server_path):
            logger.error(f"❌ 找不到 FastAPI 服務器文件: {fastapi_server_path}")
            return False
        logger.info(f"✅ FastAPI 服務器文件存在: {fastapi_server_path}")

        # 讀取 FastAPI 目標埠號（預設 8000）
        try:
            fastapi_port = int(os.environ.get("FASTAPI_PORT", "8000"))
        except ValueError:
            fastapi_port = 8000

        # 檢查端口是否被佔用
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", fastapi_port))
            sock.close()
            if result == 0:
                logger.warning(f"⚠️ 端口 {fastapi_port} 已被佔用，可能導致服務啟動失敗")
            else:
                logger.info(f"✅ 端口 {fastapi_port} 可用")
        except Exception as e:
            logger.warning(f"⚠️ 無法檢查端口 {fastapi_port}: {e}")

        # 設定工作目錄為當前目錄
        working_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"設定工作目錄: {working_dir}")

        # 檢查 Python 環境
        logger.info(f"使用 Python 解釋器: {sys.executable}")
        logger.info(f"Python 版本: {sys.version}")

        # 檢查環境變數
        env_vars = dict(os.environ)
        # 強制子進程標準輸出使用 UTF-8，避免中文亂碼
        env_vars.setdefault("PYTHONIOENCODING", "utf-8")
        env_vars.setdefault("PYTHONUTF8", "1")
        logger.info(f"使用當前環境變數")

        # 構建啟動命令
        cmd = [sys.executable, "stt_streaming_fastapi.py"]
        logger.info(f"啟動命令: {' '.join(cmd)}")

        # 啟動服務器
        logger.info("正在啟動 FastAPI STT Streaming 服務器...")
        creation_flags = 0
        if os.name == "nt":
            # 使用新進程群組，方便傳送信號與在控制台關閉時獨立處理
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        # 使用可讀取的管道，以便在關閉時非阻塞讀取輸出避免子進程卡住
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_dir,
            creationflags=creation_flags,
            env=env_vars,
        )

        logger.info(f"進程 PID: {process.pid}")

        # 等待服務器啟動，改用 WebSocket 就緒檢查
        logger.info("等待服務器啟動 (WebSocket 就緒檢查)...")
        ready = False
        try:
            import websockets
        except ImportError:
            websockets = None
            logger.warning("未安裝 websockets，無法執行 WS 就緒檢查，將略過")

        for i in range(3):  # 最多等待20秒（FastAPI 啟動較慢）
            time.sleep(1)
            if process.poll() is not None:
                logger.error(f"❌ 進程在第 {i+1} 秒時退出")
                break
            if websockets:
                try:
                    import asyncio

                    async def _probe():
                        url = f"ws://127.0.0.1:{fastapi_port}/ws/stt?modelCode=chinese&token=probe&jobId=probe"
                        try:
                            async with websockets.connect(
                                url, open_timeout=3, close_timeout=1
                            ) as ws:
                                return True
                        except Exception:
                            return False

                    if asyncio.run(_probe()):
                        ready = True
                        logger.info(f"WebSocket 就緒 (第 {i+1} 秒)")
                        break
                except Exception:
                    pass
            logger.info(f"進程狀態檢查 {i+1}/3: 運行中，等待 WebSocket 就緒...")

        # 檢查進程是否還在運行
        if process.poll() is None and ready:
            logger.info(f"✅ FastAPI STT Streaming 服務器已啟動 (端口 {fastapi_port})")
            logger.info(f"進程 PID: {process.pid}")
            logger.info(f"🌐 WebSocket 端點: ws://localhost:{fastapi_port}/ws/stt")
            logger.info(f"🌐 HTTP 端點: http://localhost:{fastapi_port}")
            # 註冊到全域清單，確保程式結束時能清理
            register_process("FastAPI STT Streaming", process)
            return process
        elif process.poll() is None and not ready:
            # 進程仍在，但 WS 尚未就緒（可能初始化較慢）。啟動背景 WS 監測執行緒，不阻塞主流程。
            logger.warning(
                "⚠️ FastAPI 進程仍在運行，但 WebSocket 未在時限內就緒。將在背景持續監測 WS 狀態。"
            )

            def _monitor_ws_background(proc, port):
                try:
                    import asyncio, time as _t

                    try:
                        import websockets as _ws
                    except Exception:
                        return
                    max_secs = int(os.environ.get("FASTAPI_WS_MONITOR_SECS", "120"))
                    interval = float(os.environ.get("FASTAPI_WS_MONITOR_INTERVAL", "2"))
                    start_ts = _t.time()
                    while (proc.poll() is None) and ((_t.time() - start_ts) < max_secs):

                        async def _probe_bg():
                            url = f"ws://127.0.0.1:{port}/ws/stt?modelCode=chinese&token=monitor&jobId=monitor"
                            try:
                                async with _ws.connect(
                                    url, open_timeout=3, close_timeout=1
                                ) as ws:
                                    return True
                            except Exception:
                                return False

                        try:
                            if asyncio.run(_probe_bg()):
                                logger.info("✅ WebSocket 就緒（背景監測）")
                                return
                        except Exception:
                            pass
                        _t.sleep(interval)
                    logger.warning("⚠️ 背景監測在時限內未等到 WebSocket 就緒")
                except Exception:
                    pass

            try:
                threading.Thread(
                    target=_monitor_ws_background,
                    args=(process, fastapi_port),
                    daemon=True,
                ).start()
            except Exception:
                pass
            register_process("FastAPI STT Streaming", process)
            return process
        else:
            logger.error(f"❌ FastAPI STT Streaming 服務器啟動失敗（進程已退出）")
            logger.error(f"進程退出碼: {process.returncode}")

            # 僅在進程已退出時讀取輸出，避免阻塞
            try:
                stdout, stderr = process.communicate(timeout=2)
            except Exception:
                stdout, stderr = b"", b""

            # 嘗試解碼輸出，處理編碼問題
            try:
                stdout_text = stdout.decode("utf-8", errors="ignore")
                stderr_text = stderr.decode("utf-8", errors="ignore")
            except UnicodeDecodeError:
                try:
                    stdout_text = stdout.decode("gbk", errors="ignore")
                    stderr_text = stderr.decode("gbk", errors="ignore")
                except UnicodeDecodeError:
                    stdout_text = stdout.decode("latin-1", errors="ignore")
                    stderr_text = stderr.decode("latin-1", errors="ignore")

            logger.error("=== 詳細錯誤信息 ===")
            if stdout_text.strip():
                logger.error(f"STDOUT ({len(stdout_text)} 字元):")
                for line in stdout_text.strip().split("\n"):
                    logger.error(f"  {line}")
            else:
                logger.error("STDOUT: (空)")

            if stderr_text.strip():
                logger.error(f"STDERR ({len(stderr_text)} 字元):")
                for line in stderr_text.strip().split("\n"):
                    logger.error(f"  {line}")
            else:
                logger.error("STDERR: (空)")
            logger.error("=== 錯誤信息結束 ===")

            return False

    except Exception as e:
        logger.error(f"❌ 啟動 STT Streaming 服務器失敗: {e}")
        import traceback

        logger.error("詳細錯誤堆疊:")
        for line in traceback.format_exc().split("\n"):
            logger.error(f"  {line}")
        return False


def start_services():
    """啟動所有ASR服務"""
    logger.info("啟動ASR服務...")

    processes = []

    # 檢查服務文件是否存在
    service_files = ["asr_api.py"]
    for service_file in service_files:
        if not os.path.exists(service_file):
            logger.error(f"❌ 找不到 {service_file}")
            return False

    try:
        # 啟動 HTTP API 服務 (端口 5000)
        logger.info("啟動 HTTP API 服務 (端口 5000)...")
        http_port = 5000
        # 先檢查端口是否被佔用，若已佔用則視為已有執行個體，避免重複啟動導致退出
        _port_in_use = False
        try:
            import socket as _sock

            _s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
            _s.settimeout(1)
            _port_in_use = _s.connect_ex(("127.0.0.1", http_port)) == 0
            _s.close()
        except Exception:
            pass

        if _port_in_use:
            logger.warning(
                f"⚠️ 端口 {http_port} 已被佔用，推測 HTTP API 已在執行，將不重複啟動。"
            )
        else:
            api_creation_flags = 0
            if os.name == "nt":
                api_creation_flags = getattr(
                    subprocess, "CREATE_NO_WINDOW", 0
                ) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            api_env = dict(os.environ, PYTHONPATH=os.getcwd())
            api_env.setdefault("PYTHONIOENCODING", "utf-8")
            api_env.setdefault("PYTHONUTF8", "1")
            api_process = subprocess.Popen(
                [sys.executable, "asr_api.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=api_creation_flags,
                env=api_env,
            )
            processes.append(("HTTP API", api_process))
            register_process("HTTP API", api_process)

            # 等待 API /api/health 就緒（最長 60 秒）
            logger.info("等待 HTTP API 就緒...")
            import urllib.request as _u

            _ready = False
            for _i in range(60):
                time.sleep(1)
                if api_process.poll() is not None:
                    logger.error("❌ HTTP API 進程意外退出")
                    break
                try:
                    with _u.urlopen(
                        "http://127.0.0.1:5000/api/health", timeout=1
                    ) as _resp:
                        if _resp.status == 200:
                            _ready = True
                            break
                except Exception:
                    pass
            if _ready:
                logger.info("✅ HTTP API 就緒")
            else:
                logger.warning("⚠️ HTTP API 未在時限內就緒，後台將持續監測")

        # 啟動 FastAPI STT Streaming 服務 (端口 8000)
        logger.info("啟動 FastAPI STT Streaming 服務 (端口 8000)...")
        stt_process = start_stt_streaming_server()
        if stt_process:
            processes.append(("FastAPI STT Streaming", stt_process))
        else:
            logger.error("❌ FastAPI STT Streaming 服務啟動失敗")
            return False

        # 等待服務啟動
        time.sleep(3)

        # 檢查所有進程是否還在運行
        all_running = True
        for name, process in processes:
            if process.poll() is None:
                logger.info(f"✅ {name} 服務已啟動")
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ {name} 服務啟動失敗:")

                # 嘗試解碼輸出，處理編碼問題
                try:
                    stdout_text = stdout.decode("utf-8", errors="ignore")
                    stderr_text = stderr.decode("utf-8", errors="ignore")
                except UnicodeDecodeError:
                    try:
                        stdout_text = stdout.decode("gbk", errors="ignore")
                        stderr_text = stderr.decode("gbk", errors="ignore")
                    except UnicodeDecodeError:
                        stdout_text = stdout.decode("latin-1", errors="ignore")
                        stderr_text = stderr.decode("latin-1", errors="ignore")

                if stdout_text.strip():
                    logger.error(f"stdout: {stdout_text}")
                if stderr_text.strip():
                    logger.error(f"stderr: {stderr_text}")
                all_running = False

        if all_running:
            logger.info("✅ 所有ASR服務已啟動")
            return processes
        else:
            logger.error("❌ 部分服務啟動失敗")
            return False

    except Exception as e:
        logger.error(f"❌ 啟動服務失敗: {e}")
        return False


def open_test_pages():
    """打開測試頁面"""
    logger.info("打開測試頁面...")

    test_files = [
        ("test_fastapi.html", "FastAPI STT Streaming 測試"),
        ("test_api.html", "原始 STT Streaming 測試"),
        ("stt_streaming/client/VoiceStreamAI_Client.html", "STT Streaming 客戶端"),
        ("test_microphone.html", "麥克風測試"),
        ("asr_api.py", "HTTP API 測試"),
    ]

    for test_file, description in test_files:
        if os.path.exists(test_file):
            if test_file.endswith(".html"):
                file_path = f"file://{os.path.abspath(test_file)}"
                logger.info(f"打開 {description}: {test_file}")
                try:
                    webbrowser.open(file_path)
                except Exception as e:
                    logger.warning(f"無法自動打開瀏覽器: {e}")
                    logger.info(f"請手動打開: {file_path}")
            elif test_file == "asr_api.py":
                # 打開HTTP API測試頁面
                api_url = "http://localhost:5000"
                logger.info(f"打開 {description}: {api_url}")
                try:
                    webbrowser.open(api_url)
                except Exception as e:
                    logger.warning(f"無法自動打開瀏覽器: {e}")
                    logger.info(f"請手動訪問: {api_url}")
            break
    else:
        logger.warning("找不到測試頁面文件")

        # 顯示 FastAPI STT Streaming 服務信息
        logger.info("FastAPI STT Streaming 服務信息:")
        logger.info("WebSocket 端點: ws://localhost:8000/ws/stt")
        logger.info("HTTP 端點: http://localhost:8000")
        logger.info("健康檢查: http://localhost:8000/health")
        logger.info("測試頁面: http://localhost:8000/test")
        logger.info("請使用 FastAPI STT Streaming 客戶端進行測試")


def main():
    """主函數"""
    logger.info("=" * 50)
    logger.info("ASR 環境建立與檢查（不啟動服務）")
    logger.info("=" * 50)

    # 1. 啟動虛擬環境
    logger.info("正在啟動虛擬環境...")
    if not activate_virtual_environment():
        logger.warning("⚠️ 虛擬環境啟動失敗，將使用系統 Python")
        logger.warning("建議：確保 asr_api 虛擬環境存在且正確配置")

    # 顯示當前 Python 環境信息
    logger.info(f"當前 Python 路徑: {sys.executable}")
    logger.info(f"當前 Python 版本: {sys.version}")

    # 檢查是否在虛擬環境中
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        logger.info("✅ 運行在虛擬環境中")
        logger.info(f"虛擬環境路徑: {sys.prefix}")
    else:
        logger.warning("⚠️ 未檢測到虛擬環境")

    # 2. 檢查Python環境
    if not check_python_and_pip():
        return

    # 3. 檢查並安裝依賴
    if not check_and_install_dependencies():
        return

    # 4. 檢查模型目錄
    if not check_models_directory():
        return

    # 5. 基本依賴套件測試
    if not test_vad_basic():
        return

    # 6. 僅環境與依賴檢查；不啟動任何服務
    logger.info("=" * 50)
    logger.info("環境建立與檢查已完成")
    logger.info("下一步：如需啟動服務，請手動執行相關指令或批次檔。")
    logger.info("- HTTP API：python file_asr.py 或執行 start_file_asr.bat")
    logger.info(
        "- Streaming ASRI：python streaming_asr.py 或執行 start_streaming_asr.bat"
    )
    logger.info("=" * 50)
    return


if __name__ == "__main__":
    main()
