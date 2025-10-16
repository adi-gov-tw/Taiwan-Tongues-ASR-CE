@echo off
chcp 65001 >nul
echo ========================================
echo ASR API 整合服務啟動腳本 (port 5000)
echo ========================================

REM ========================================
REM 基本設定（可依需求調整）
REM ========================================
REM JWT 秘鑰：發行與驗證 Token 用，請務必修改成安全值
set ASR_API_JWT_SECRET=CHANGE_ME_SECRET
REM JWT 演算法：預設 HS256
set ASR_API_JWT_ALGORITHM=HS256

REM 授權資料庫檔案路徑（放在 api 目錄下）
set ASR_API_AUTH_DB=%~dp0auth.db

REM 預設管理員帳號（服務啟動時會檢查/建立）
set ASR_API_BOOTSTRAP_ADMIN_USERNAME=admin
REM 預設管理員暱稱
set ASR_API_BOOTSTRAP_ADMIN_NICKNAME=ADMIN
REM 預設管理員密碼（如啟用重設，將在啟動時套用）
set ASR_API_BOOTSTRAP_ADMIN_PASSWORD=your_custom_password_here

REM 啟動時是否強制重設 admin 密碼：1=是（預設）、0=否
REM 設為 0 可避免覆蓋手動修改後的密碼
set ASR_API_RESET_ADMIN_ON_STARTUP=1

REM 串流 ASR 初始化與預熱設定（在同一埠下運作）
set FASTAPI_SKIP_INIT=0
set FASTAPI_WARMUP=1
set FASTAPI_ASR_MODEL_SIZE=models
REM 若需指定模型大小，請取消註解並填入實際的模型名稱
REM set FASTAPI_ASR_MODEL_SIZE=your_model_size_here
set FASTAPI_PORT=5000
set BUFFERING_CHUNK_LENGTH_SECONDS=1.5
set BUFFERING_CHUNK_OFFSET_SECONDS=0.1

REM 檢查虛擬環境是否存在（在父目錄中）
if not exist "..\asr_api\Scripts\activate.bat" (
    echo ❌ 錯誤：找不到 asr_api 虛擬環境
    pause
    exit /b 1
)

echo ✅ 找到 asr_api 虛擬環境
echo 正在啟動整合服務...

REM 啟動虛擬環境並運行整合服務
call ..\asr_api\Scripts\activate.bat && python app.py

echo.
echo 服務已停止
pause
