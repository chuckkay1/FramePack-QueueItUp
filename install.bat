@echo off
setlocal
setlocal EnableDelayedExpansion

REM ================================================
REM install.bat — setup for FramePack - QueueItUp Version


conda create --name Framepack_QueueItUp python=3.10.6 pip=25.0 -y
echo.
echo === Activating environment ===
call conda activate Framepack_QueueItUp
if errorlevel 1 (
  echo ERROR: could not activate environment
  pause
  exit /b 1
)

REM  Install CUDA + cuDNN
echo === Installing CUDA + cuDNN ===
call conda install conda-forge::cuda-runtime=12.8.1 conda-forge::cudnn=9.8.0.87 -y > install_cuda.log 2>&1
echo log written to install_cuda.log
pause

REM  Install PyTorch & friends
echo.
echo === Installing PyTorch, Torchvision, Torchaudio ===
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  echo ERROR: PyTorch install failed.
  pause
  exit /b 1
)

REM  Install other requirements
echo.
if exist requirements.txt (
  echo === Installing other Python dependencies ===
  pip install -r requirements.txt
) else (
  echo WARNING: requirements.txt not found, skipping
)

REM  Finish
echo.
echo ===== Setup Complete! =====
echo To in the future use: python Framepack_QueueItUp.py --server 127.0.0.1 --inbrowser
echo or use Framepack_QueueItUp.bat
echo ...
echo Ready to start Framepack_QueueItUp?
pause

call Framepack_QueueItUp.bat
pause
cmd /k
