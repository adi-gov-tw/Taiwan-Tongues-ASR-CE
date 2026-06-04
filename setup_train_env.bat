@echo off
echo === Setup training/inference venv (train_env) ===
echo.

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

if exist "train_env" goto :ask_recreate
goto :create_venv

:ask_recreate
set choice=
set /p choice=venv train_env exists. Recreate? [Y/N]:
if /i "%choice%"=="Y" goto :recreate
echo Using existing venv, only updating dependencies...
goto :activate_env

:recreate
echo Removing old venv...
rmdir /s /q train_env

:create_venv
echo Creating venv train_env (Python 3.10+) ...
%PY_CMD% -m venv train_env

:activate_env
echo.
echo Activating venv...
call train_env\Scripts\activate.bat

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
echo Installing requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 goto :install_failed

:verify
echo.
echo === Verifying install ===
python -c "import sys; print('python', sys.version.split()[0])"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import librosa; print('librosa', librosa.__version__)"
python -c "import datasets; print('datasets', datasets.__version__)"

echo.
echo === Setup complete ===
echo To train: train.bat (venv will be auto-activated)
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
