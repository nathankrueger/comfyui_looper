@echo off
setlocal enabledelayedexpansion

:: --- Defaults for script-only flags ---
set COMFYUI_PYTHON=
set COMFYUI_MAIN=
set COMFYUI_URL=http://localhost:8188
set COMFYUI_WAIT=30
set LOOPER_PORT=5000
set VENV_DIR=.venv
set PASSTHROUGH=

:: --- Parse arguments ---
:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="-P" (set COMFYUI_PYTHON=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-M" (set COMFYUI_MAIN=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-u" (set COMFYUI_URL=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-s" (set COMFYUI_WAIT=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-L" (set LOOPER_PORT=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-v" (set VENV_DIR=%~2& shift & shift & goto parse_args)
if /i "%~1"=="-h" goto usage
if /i "%~1"=="--" (shift & goto collect_rest)
:: Unknown flag — collect into passthrough
set PASSTHROUGH=!PASSTHROUGH! %1
shift
goto parse_args

:collect_rest
if "%~1"=="" goto done_args
set PASSTHROUGH=!PASSTHROUGH! %1
shift
goto collect_rest

:done_args

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
echo Starting comfyui_looper in interactive mode...
echo   ComfyUI:  %COMFYUI_URL%
echo   Port:     %LOOPER_PORT%
echo   Args:     %PASSTHROUGH%
echo.

python comfyui_looper\main.py ^
    --interactive ^
    --port %LOOPER_PORT% ^
    --comfyui-url "%COMFYUI_URL%" ^
    %PASSTHROUGH%

goto :eof

:usage
echo Usage: %~nx0 [script-options] [-- main.py options]
echo.
echo Script options (parsed by this script):
echo   -P ^<path^>     ComfyUI python.exe path
echo   -M ^<path^>     ComfyUI main.py path
echo   -u ^<url^>      ComfyUI server URL (default: %COMFYUI_URL%)
echo   -s ^<seconds^>  Seconds to wait for ComfyUI to start (default: %COMFYUI_WAIT%)
echo   -L ^<port^>     Looper web UI port (default: %LOOPER_PORT%)
echo   -v ^<path^>     Python venv directory (default: %VENV_DIR%)
echo   -h            Show this help
echo.
echo All other flags are forwarded to main.py (run with --help to see them):
echo   -j ^<path^>     Workflow JSON file
echo   -o ^<path^>     Output folder
echo   -i ^<path^>     Input image
echo   -w ^<type^>     Workflow type (default: sdxl)
echo   -z            Use zip storage
echo   ...and more. Run: python comfyui_looper\main.py --help
echo.
echo Examples:
echo   %~nx0                                          (opens workflow picker)
echo   %~nx0 -j data\evolution.json                   (auto-named output folder)
echo   %~nx0 -i photo.png -o output\run1 -j data\evolution.json -w flux1d
echo   %~nx0 -L 8080 -s 60 -- -j data\test.json -z   (script flags, then main.py flags)
exit /b 1
