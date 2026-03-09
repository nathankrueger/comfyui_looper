@echo off
setlocal enabledelayedexpansion

:: --- Defaults ---
set COMFYUI_PYTHON=
set COMFYUI_MAIN=
set COMFYUI_URL=http://localhost:8188
set COMFYUI_WAIT=30
set LOOPER_PORT=5000
set WORKFLOW_TYPE=sdxl
set INPUT_IMG=
set OUTPUT_FOLDER=
set JSON_FILE=
set VENV_DIR=.venv

:: --- Parse arguments ---
:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="-i" (set INPUT_IMG=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-o" (set OUTPUT_FOLDER=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-j" (set JSON_FILE=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-w" (set WORKFLOW_TYPE=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-u" (set COMFYUI_URL=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-s" (set COMFYUI_WAIT=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-P" (set COMFYUI_PYTHON=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-M" (set COMFYUI_MAIN=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-L" (set LOOPER_PORT=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-v" (set VENV_DIR=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-h" goto usage
echo Unknown option: %~1
goto usage
:done_args

:: --- Validate required args ---
if "%JSON_FILE%"=="" goto missing_args
goto args_ok

:missing_args
echo Error: -j is required.
echo.
goto usage

:args_ok

:: --- Activate venv ---
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    echo Error: Virtual environment not found at %VENV_DIR%
    echo Run install.bat first to create it.
    exit /b 1
)

:: --- Check if ComfyUI is reachable ---
echo Checking ComfyUI at %COMFYUI_URL%...
powershell -Command "try { $r = Invoke-WebRequest -Uri '%COMFYUI_URL%' -TimeoutSec 3 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
    echo ComfyUI is already running at %COMFYUI_URL%
    goto start_looper
)

echo ComfyUI not detected at %COMFYUI_URL%.

:: --- Try to launch ComfyUI if paths are configured ---
if "%COMFYUI_PYTHON%"=="" goto no_comfyui_paths
if "%COMFYUI_MAIN%"=="" goto no_comfyui_paths

if not exist "%COMFYUI_PYTHON%" (
    echo Error: ComfyUI python not found at %COMFYUI_PYTHON%
    exit /b 1
)
if not exist "%COMFYUI_MAIN%" (
    echo Error: ComfyUI main.py not found at %COMFYUI_MAIN%
    exit /b 1
)

echo Launching ComfyUI...
start "" "%COMFYUI_PYTHON%" "%COMFYUI_MAIN%" --listen 0.0.0.0

echo Waiting up to %COMFYUI_WAIT%s for ComfyUI to start...
set /a elapsed=0
:wait_loop
if %elapsed% geq %COMFYUI_WAIT% goto wait_timeout
powershell -Command "try { $r = Invoke-WebRequest -Uri '%COMFYUI_URL%' -TimeoutSec 2 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
    echo ComfyUI is ready after %elapsed%s.
    goto start_looper
)
timeout /t 1 /nobreak >nul
set /a elapsed+=1
goto wait_loop

:wait_timeout
echo Warning: ComfyUI not responding after %COMFYUI_WAIT%s. Proceeding anyway...
goto start_looper

:no_comfyui_paths
echo To auto-launch ComfyUI, use -P and -M flags to specify paths.
echo Proceeding without launching ComfyUI...

:: --- Launch looper ---
:start_looper
echo.
:: --- Build optional flags ---
set INPUT_FLAG=
set INPUT_DISPLAY=^<none (txt2img)^>
if not "%INPUT_IMG%"=="" (
    set INPUT_FLAG=-i "%INPUT_IMG%"
    set INPUT_DISPLAY=%INPUT_IMG%
)

set OUTPUT_FLAG=
set OUTPUT_DISPLAY=^<auto^>
if not "%OUTPUT_FOLDER%"=="" (
    set OUTPUT_FLAG=-o "%OUTPUT_FOLDER%"
    set OUTPUT_DISPLAY=%OUTPUT_FOLDER%
)

echo Starting comfyui_looper in interactive mode...
echo   Workflow: %WORKFLOW_TYPE%
echo   Input:    %INPUT_DISPLAY%
echo   Output:   %OUTPUT_DISPLAY%
echo   JSON:     %JSON_FILE%
echo   ComfyUI:  %COMFYUI_URL%
echo   Port:     %LOOPER_PORT%
echo.

python comfyui_looper\main.py ^
    --interactive ^
    --port %LOOPER_PORT% ^
    --comfyui-url "%COMFYUI_URL%" ^
    -w %WORKFLOW_TYPE% ^
    %INPUT_FLAG% ^
    %OUTPUT_FLAG% ^
    -j "%JSON_FILE%"

goto :eof

:usage
echo Usage: %~nx0 -j ^<json_file^> [options]
echo.
echo Required:
echo   -j ^<path^>     Workflow JSON file
echo.
echo Options:
echo   -o ^<path^>     Output folder (default: data\^<workflow_name^>_^<timestamp^>)
echo   -i ^<path^>     Input image (if omitted, first frame uses txt2img)
echo   -w ^<type^>     Workflow type (default: sdxl)
echo   -u ^<url^>      ComfyUI server URL (default: %COMFYUI_URL%)
echo   -s ^<seconds^>  Seconds to wait for ComfyUI to start (default: %COMFYUI_WAIT%)
echo   -P ^<path^>     ComfyUI python.exe path
echo   -M ^<path^>     ComfyUI main.py path
echo   -L ^<port^>     Looper web UI port (default: %LOOPER_PORT%)
echo   -v ^<path^>     Python venv directory (default: %VENV_DIR%)
echo   -h            Show this help
echo.
echo Examples:
echo   %~nx0 -i photo.png -o output\run1 -j data\evolution.json -w flux1d
echo   %~nx0 -j data\evolution.json                   (auto-named output folder)
echo   %~nx0 -o output\run1 -j data\evolution.json   (no input image, txt2img)
exit /b 1
