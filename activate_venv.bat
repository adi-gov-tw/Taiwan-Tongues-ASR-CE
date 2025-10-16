@echo off
chcp 65001 >nul
echo === 啟動 Whisper 虛擬環境 ===
echo.

REM 檢查虛擬環境是否存在
if not exist "whisper_env" (
    echo 錯誤: 找不到虛擬環境，請先執行 setup_venv.bat
    pause
    exit /b 1
)

echo 正在啟動虛擬環境...
call whisper_env\Scripts\activate.bat

echo.
echo 虛擬環境已啟動！

REM 啟動命令提示字元
cmd /k 