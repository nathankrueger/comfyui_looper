#!/bin/bash
set -e

# --- Defaults for script-only flags ---
WINDOWS_HOST_IP=$(ip route show default | awk '{print $3}')
COMFYUI_PYTHON="/mnt/e/ComfyUI_windows_portable/python_embeded/python.exe"
COMFYUI_MAIN="/mnt/e/ComfyUI_windows_portable/ComfyUI/main.py"
COMFYUI_URL="http://${WINDOWS_HOST_IP}:8188"
COMFYUI_WAIT=10
LOOPER_PORT=8080
VENV_DIR=".venv"

usage() {
    cat <<EOF
Usage: $0 [script-options] [-- main.py options]

Script options (parsed by this script):
  -P <path>     ComfyUI python.exe path (default: $COMFYUI_PYTHON)
  -M <path>     ComfyUI main.py path (default: $COMFYUI_MAIN)
  -u <url>      ComfyUI server URL (default: $COMFYUI_URL)
  -s <seconds>  Seconds to wait for ComfyUI to start (default: $COMFYUI_WAIT)
  -L <port>     Looper web UI port (default: $LOOPER_PORT)
  -v <path>     Python venv directory (default: $VENV_DIR)
  -h            Show this help

All other flags are forwarded to main.py (run with --help to see them):
  -j <path>     Workflow JSON file
  -o <path>     Output folder
  -i <path>     Input image
  -w <type>     Workflow type (default: sdxl)
  -z            Use zip storage
  ...and more. Run: python comfyui_looper/main.py --help

Examples:
  $0                                         # opens workflow picker
  $0 -j data/evolution.json                  # auto-named output folder
  $0 -i photo.png -o output/run1 -j data/evolution.json -w flux1d
  $0 -L 8080 -s 60 -- -j data/test.json -z  # script flags, then main.py flags
EOF
    exit 1
}

# --- Two-tier arg parsing ---
# Script-only flags: -P, -M, -u, -s, -L, -v, -h
# Everything else is collected into PASSTHROUGH for main.py
PASSTHROUGH=()

while [ $# -gt 0 ]; do
    case "$1" in
        -P) COMFYUI_PYTHON="$2"; shift 2 ;;
        -M) COMFYUI_MAIN="$2"; shift 2 ;;
        -u) COMFYUI_URL="$2"; shift 2 ;;
        -s) COMFYUI_WAIT="$2"; shift 2 ;;
        -L) LOOPER_PORT="$2"; shift 2 ;;
        -v) VENV_DIR="$2"; shift 2 ;;
        -h) usage ;;
        --) shift; PASSTHROUGH+=("$@"); break ;;
        *)  PASSTHROUGH+=("$1"); shift ;;
    esac
done

# --- Activate venv ---
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Run install.sh first to create it."
    exit 1
fi

# --- Check if ComfyUI is reachable, launch if not ---
check_comfyui() {
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$COMFYUI_URL" 2>/dev/null)
    echo "  [check] curl returned HTTP $http_code" >&2
    [ "$http_code" = "200" ]
}

if check_comfyui; then
    echo "ComfyUI is already running at $COMFYUI_URL"
else
    echo "ComfyUI not detected at $COMFYUI_URL, launching..."

    if [ ! -f "$COMFYUI_PYTHON" ]; then
        echo "Error: ComfyUI python not found at $COMFYUI_PYTHON"
        exit 1
    fi
    if [ ! -f "$COMFYUI_MAIN" ]; then
        echo "Error: ComfyUI main.py not found at $COMFYUI_MAIN"
        exit 1
    fi

    # Convert WSL path to Windows path for python.exe
    COMFYUI_MAIN_WIN=$(wslpath -w "$COMFYUI_MAIN")
    "$COMFYUI_PYTHON" "$COMFYUI_MAIN_WIN" --listen 0.0.0.0 &
    COMFYUI_PID=$!
    echo "Started ComfyUI (PID $COMFYUI_PID), waiting up to ${COMFYUI_WAIT}s..."

    elapsed=0
    while [ $elapsed -lt "$COMFYUI_WAIT" ]; do
        echo "  [wait] ${elapsed}s / ${COMFYUI_WAIT}s ..."
        if check_comfyui; then
            echo "ComfyUI is ready after ${elapsed}s."
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    if [ $elapsed -ge "$COMFYUI_WAIT" ]; then
        echo "  [wait] final check at ${COMFYUI_WAIT}s..."
        if ! check_comfyui; then
            echo "Warning: ComfyUI not responding after ${COMFYUI_WAIT}s. Proceeding anyway..."
        fi
    fi
fi

# --- Launch looper in interactive mode ---
echo "Starting comfyui_looper in interactive mode..."
echo "  ComfyUI:  $COMFYUI_URL"
echo "  Port:     $LOOPER_PORT"
echo "  Args:     ${PASSTHROUGH[*]:-(none)}"
echo ""

python comfyui_looper/main.py \
    --interactive \
    --port "$LOOPER_PORT" \
    --comfyui-url "$COMFYUI_URL" \
    "${PASSTHROUGH[@]}"
