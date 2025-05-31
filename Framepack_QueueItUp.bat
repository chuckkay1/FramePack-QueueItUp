@echo off
setlocal

REM Activate the conda environment
echo Activating conda environment...
call conda activate FramePack_QueueItUp

REM Disable the automatic Y/N prompt
SET IGNORE_CTRL_C=1

REM Restart loop
:RESTART
echo.
echo Starting Framepack QueueItUp...

REM Run the app
python Framepack_QueueItUp.py --server 127.0.0.1 --inbrowser

REM Check for exit or crash
IF %ERRORLEVEL% NEQ 0 (
    echo.
    choice /m "Would you like to restart the app?"
    IF ERRORLEVEL 2 EXIT
)

GOTO RESTART
