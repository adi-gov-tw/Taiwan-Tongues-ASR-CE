@echo off
echo ========================================
echo ASR API server (port 5000)
echo ========================================

REM ========================================
REM Sensitive settings live in api\.env (copy from api\.env.example).
REM app.py auto-loads .env on startup; this script only sets non-secret defaults.
REM ========================================

REM Auth DB path (default: api\auth.db)
if "%ASR_API_AUTH_DB%"=="" set ASR_API_AUTH_DB=%~dp0auth.db

REM Streaming ASR runtime defaults
if "%FASTAPI_SKIP_INIT%"=="" set FASTAPI_SKIP_INIT=0
if "%FASTAPI_WARMUP%"=="" set FASTAPI_WARMUP=1
if "%FASTAPI_ASR_MODEL_SIZE%"=="" set FASTAPI_ASR_MODEL_SIZE=models
if "%FASTAPI_HOST%"=="" set FASTAPI_HOST=0.0.0.0
if "%FASTAPI_PORT%"=="" set FASTAPI_PORT=5000
if "%BUFFERING_CHUNK_LENGTH_SECONDS%"=="" set BUFFERING_CHUNK_LENGTH_SECONDS=1.5
if "%BUFFERING_CHUNK_OFFSET_SECONDS%"=="" set BUFFERING_CHUNK_OFFSET_SECONDS=0.1

echo Listening on http://%FASTAPI_HOST%:%FASTAPI_PORT%

REM Warn if .env missing
if not exist "%~dp0.env" (
    echo WARN: api\.env not found. Falling back to insecure defaults
    echo       ^(JWT_SECRET=CHANGE_ME_SECRET, etc^).
    echo       Recommended: copy %~dp0.env.example %~dp0.env
    echo                    then edit to set JWT secret and admin password.
    echo.
)

REM ========================================
REM Auto-detect virtual environment (priority order)
REM ========================================
set VENV_ACTIVATE=
if exist "%~dp0..\asr_api\Scripts\activate.bat" (
    set VENV_ACTIVATE=%~dp0..\asr_api\Scripts\activate.bat
    echo Found venv: asr_api
) else if exist "%~dp0..\train_env\Scripts\activate.bat" (
    set VENV_ACTIVATE=%~dp0..\train_env\Scripts\activate.bat
    echo Found venv: train_env
) else if exist "%~dp0..\.venv\Scripts\activate.bat" (
    set VENV_ACTIVATE=%~dp0..\.venv\Scripts\activate.bat
    echo Found venv: .venv
) else if exist "%~dp0..\venv\Scripts\activate.bat" (
    set VENV_ACTIVATE=%~dp0..\venv\Scripts\activate.bat
    echo Found venv: venv
)

if "%VENV_ACTIVATE%"=="" (
    echo WARN: No venv found, using system Python.
    echo       Recommended: run %~dp0setup_api_env.bat first.
    echo.
    pushd "%~dp0"
    python app.py
    popd
) else (
    echo Starting service...
    pushd "%~dp0"
    call "%VENV_ACTIVATE%" && python app.py
    popd
)

echo.
echo Service stopped
pause
