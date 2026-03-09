#!/bin/bash
set -e

# --- Defaults (tunable via flags) ---
WINDOWS_HOST_IP=$(ip route show default | awk '{print $3}')
COMFYUI_PYTHON="/mnt/e/ComfyUI_windows_portable/python_embeded/python.exe"
COMFYUI_MAIN="/mnt/e/ComfyUI_windows_portable/ComfyUI/main.py"
COMFYUI_URL="http://${WINDOWS_HOST_IP}:8188"
COMFYUI_WAIT=10
LOOPER_PORT=5000
WORKFLOW_TYPE="sdxl"
INPUT_IMG=""
OUTPUT_FOLDER=""
JSON_FILE=""
VENV_DIR=".venv"

usage() {
    cat <<EOF
Usage: $0 -o <output_folder> -j <json_file> [options]

Required:
  -o <path>     Output folder
  -j <path>     Workflow JSON file

Options:
  -i <path>     Input image (if omitted, first frame uses txt2img)
  -w <type>     Workflow type (default: sdxl)
  -u <url>      ComfyUI server URL (default: $COMFYUI_URL)
  -s <seconds>  Seconds to wait for ComfyUI to start (default: $COMFYUI_WAIT)
  -P <path>     ComfyUI python.exe path (default: $COMFYUI_PYTHON)
  -M <path>     ComfyUI main.py path (default: $COMFYUI_MAIN)
  -L <port>    Looper web UI port (default: $LOOPER_PORT)
  -v <path>     Python venv directory (default: $VENV_DIR)
  -h            Show this help

Examples:
  $0 -i photo.png -o output/run1 -j data/evolution.json -w flux1d
  $0 -o output/run1 -j data/evolution.json   # no input image (txt2img)
EOF
    exit 1
}

while getopts "i:o:j:w:u:s:P:M:L:v:h" opt; do
    case $opt in
        i) INPUT_IMG="$OPTARG" ;;
        o) OUTPUT_FOLDER="$OPTARG" ;;
        j) JSON_FILE="$OPTARG" ;;
        w) WORKFLOW_TYPE="$OPTARG" ;;
        u) COMFYUI_URL="$OPTARG" ;;
        s) COMFYUI_WAIT="$OPTARG" ;;
        P) COMFYUI_PYTHON="$OPTARG" ;;
        M) COMFYUI_MAIN="$OPTARG" ;;
        L) LOOPER_PORT="$OPTARG" ;;
        v) VENV_DIR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# --- Validate required args ---
if [ -z "$OUTPUT_FOLDER" ] || [ -z "$JSON_FILE" ]; then
    echo "Error: -o and -j are required."
    echo ""
    usage
fi

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

# --- Build optional flags ---
INPUT_FLAG=""
if [ -n "$INPUT_IMG" ]; then
    INPUT_FLAG="-i $INPUT_IMG"
fi

# --- Launch looper in interactive mode ---
echo "Starting comfyui_looper in interactive mode..."
echo "  Workflow: $WORKFLOW_TYPE"
echo "  Input:    ${INPUT_IMG:-<none (txt2img)>}"
echo "  Output:   $OUTPUT_FOLDER"
echo "  JSON:     $JSON_FILE"
echo "  ComfyUI:  $COMFYUI_URL"
echo ""

python comfyui_looper/main.py \
    --interactive \
    --port "$LOOPER_PORT" \
    --comfyui-url "$COMFYUI_URL" \
    -w "$WORKFLOW_TYPE" \
    $INPUT_FLAG \
    -o "$OUTPUT_FOLDER" \
    -j "$JSON_FILE"
