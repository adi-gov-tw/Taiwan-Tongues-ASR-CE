@echo off
echo === Setup ASR API venv (asr_api) — Python 3.10+ ===
echo.

REM Run from project root regardless of where this script lives
cd /d "%~dp0.."

REM ========================================
REM Locate Python 3.10+ (required)
REM Priority: py -3.10  -^>  python  -^>  py -3 (must be ^>=3.10)  -^>  python (must be ^>=3.10)
REM ========================================
set PY_CMD=
py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3.10"
    goto :py_found
)
python --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :py_found
)
py -3 -c "import sys;exit(0 if sys.version_info[:2]>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3"
    goto :py_found
)
python --version >nul 2>&1
if errorlevel 1 goto :no_python
python -c "import sys;exit(0 if sys.version_info[:2]>=(3,10) else 1)" >nul 2>&1
if errorlevel 1 goto :wrong_python
set "PY_CMD=python"

:py_found
echo Using Python: %PY_CMD%
%PY_CMD% --version

if exist "asr_api" goto :ask_recreate
goto :create_venv

:ask_recreate
set choice=
set /p choice=venv asr_api exists. Recreate? [Y/N]:
if /i "%choice%"=="Y" goto :recreate
echo Using existing venv, only updating dependencies...
goto :install_deps

:recreate
echo Removing old venv...
rmdir /s /q asr_api

:create_venv
echo Creating venv asr_api (Python 3.10+) ...
%PY_CMD% -m venv asr_api

:install_deps
echo.
echo Activating venv...
call asr_api\Scripts\activate.bat

echo.
echo Verifying venv Python version...
python -c "import sys; assert sys.version_info[:2]>=(3,10), 'venv python is %%s, need >=3.10' %% sys.version.split()[0]; print('venv python', sys.version.split()[0])"
if errorlevel 1 goto :install_failed

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Detecting NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected; installing CPU PyTorch.
    set TORCH_INDEX=https://download.pytorch.org/whl/cpu
    set TORCH_LABEL=CPU
) else (
    echo NVIDIA GPU detected; installing CUDA 12.4 PyTorch.
    set TORCH_INDEX=https://download.pytorch.org/whl/cu124
    set TORCH_LABEL=CUDA 12.4
)

echo.
echo Installing PyTorch (%TORCH_LABEL%)...
pip install torch --index-url %TORCH_INDEX%
if errorlevel 1 goto :install_failed

if not "%TORCH_LABEL%"=="CPU" (
    echo.
    echo Installing cuBLAS / cuDNN 9 ^(needed by ctranslate2 / faster-whisper for GPU^)...
    pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10"
    if errorlevel 1 goto :install_failed
)

echo.
echo Installing api/requirements.txt ...
pip install -r api\requirements.txt
if errorlevel 1 goto :install_failed

echo.
echo === Setup complete ===
echo To start the service: api\start_app.bat
echo Before first run, copy api\.env.example to api\.env and fill secrets.
echo.
pause
exit /b 0

:no_python
echo ERROR: Python not found. Please install Python 3.10 or newer.
echo        Recommended: run "py install 3.10" or download from
echo        https://www.python.org/downloads/
pause
exit /b 1

:wrong_python
echo ERROR: Default python is too old; this project requires Python ^>=3.10.
echo        Install Python 3.10+ (e.g. "py install 3.10") and re-run.
pause
exit /b 1

:install_failed
echo ERROR: dependency install failed.
pause
exit /b 1
