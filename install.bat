@echo off
setlocal

set VENV_DIR=.venv

if "%~1"=="-r" (
    echo Removing existing virtual environment...
    rmdir /s /q "%VENV_DIR%" 2>nul
)

if not exist "%VENV_DIR%\" (
    echo Creating virtual environment in %VENV_DIR%...
    python -m venv "%VENV_DIR%"
) else (
    if not "%~1"=="-u" (
        echo Virtual environment already exists. Use -u to update or -r to recreate.
        exit /b 0
    )
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo Installing CPU-only PyTorch...
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Done! Activate the venv with: %VENV_DIR%\Scripts\activate.bat
echo.
echo To run the looper:
echo   python comfyui_looper\main.py -w sdxl -i ^<img^> -o output\test -j data\workflow.json
echo.
echo For interactive mode:
echo   run_interactive.bat -i ^<img^> -o output\test -j data\workflow.json

endlocal
