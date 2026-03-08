#!/bin/bash
set -e

VENV_DIR=".venv"
MODE=""

while getopts "ur" opt; do
    case $opt in
        u) MODE="update" ;;
        r) MODE="reinstall" ;;
        *) echo "Usage: $0 [-u | -r]"; echo "  -u  Update existing venv"; echo "  -r  Delete and recreate venv"; exit 1 ;;
    esac
done

if [ "$MODE" = "reinstall" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
elif [ "$MODE" != "update" ]; then
    echo "Virtual environment already exists. Use -u to update or -r to recreate."
    exit 0
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Installing CPU-only PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Done! Activate the venv with: source $VENV_DIR/bin/activate"
echo ""
echo "To run the looper against a Windows ComfyUI server:"
echo "  python comfyui_looper/main.py --comfyui-url http://<windows-ip>:8188 -w sdxl -i <img> -o output/test -j data/workflow.json"
echo ""
echo "Make sure ComfyUI is running on Windows with: python main.py --listen 0.0.0.0"
