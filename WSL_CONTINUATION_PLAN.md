# WSL Continuation Plan

## What was done (Windows session)

The looper has been refactored from direct ComfyUI Python module imports to using ComfyUI's HTTP API. This decouples the looper from needing ComfyUI in the same Python environment.

### New files created:
- `comfyui_looper/utils/comfyui_client.py` — HTTP/WebSocket client for ComfyUI server (upload images, submit workflows, download results)
- `comfyui_looper/workflow/api_workflow_builder.py` — Converts `LoopSettings` into ComfyUI API-format workflow JSON dicts for SDXL, Flux.1D, and SD3.5
- `comfyui_looper/workflow/api_engine.py` — `APIWorkflowEngine` that replaces the old direct-import engines, using the client and builders
- `tests/test_comfyui_client.py` — Unit tests for the client (mocked HTTP/WebSocket)
- `tests/test_api_workflow_builder.py` — Unit tests for all three workflow builders
- `scripts/install_wsl.sh` — WSL setup script (venv + CPU torch + requirements)
- `scripts/install_comfyui.ps1` — PowerShell script to download latest ComfyUI portable from GitHub

### Files modified:
- `comfyui_looper/utils/json_spec.py` — `folder_paths` import is now conditional (`HAS_FOLDER_PATHS`), validation skips filesystem checks when ComfyUI isn't local
- `comfyui_looper/workflow/looper_workflow.py` — Removed module-level `add_comfyui_directory_to_sys_path()` and `import_custom_nodes()` calls
- `comfyui_looper/workflow/engine_factory.py` — Rewritten: `create_workflow(name, client)` returns `APIWorkflowEngine`, no more direct-import engines
- `comfyui_looper/main.py` — Added `--comfyui-url` arg, creates `ComfyUIClient`, removed old `sys.argv` hack
- `comfyui_looper/utils/util.py` — Removed dead functions: `find_path`, `add_comfyui_directory_to_sys_path`, `import_custom_nodes`, `get_torch_device_vram_used_gb`
- `requirements.txt` — Added `requests`, `websocket-client`; removed `gunicorn`

### Files deleted:
- `comfyui_looper/workflow/sdxl_engine.py`
- `comfyui_looper/workflow/flux1d_engine.py`
- `comfyui_looper/workflow/sd3p5_engine.py`
- `comfyui_looper/utils/node_wrappers.py`

## What needs to be done (WSL session)

### 1. Setup WSL environment
```bash
cd /path/to/looper
bash scripts/install_wsl.sh
source .venv/bin/activate
```

### 2. Run tests and fix any issues
```bash
pytest tests/ -v
```

Expected: All tests should pass — client tests use mocks, builder tests validate workflow graph structure. If any existing tests (`test_json_spec.py`, `test_transforms.py`, `test_util.py`) fail due to import issues, they may need torch or other adjustments.

### 3. Remaining work items

**Fix `install.sh`**: The root `install.sh` uses `bin/activate` which is correct for Linux/WSL. Consider whether it should also install CPU torch or just use `scripts/install_wsl.sh` as the canonical WSL setup.

**Review torch imports**: `transforms.py` and `util.py` import torch at module level. These work fine with CPU torch, but if you ever want to run tests in a lighter environment, torch could be made a lazy import in these files.

**End-to-end integration test**: With ComfyUI running on Windows (`python main.py --listen 0.0.0.0`), test from WSL:
```bash
python comfyui_looper/main.py \
  --comfyui-url http://<windows-ip>:8188 \
  -w sdxl \
  -i <input_image> \
  -o output/test \
  -j data/star_wars.json \
  -p 1
```

**Interactive mode test**:
```bash
python comfyui_looper/main.py \
  --interactive \
  --comfyui-url http://<windows-ip>:8188 \
  -w sdxl \
  -i <input_image> \
  -o output/test \
  -j data/star_wars.json
# Open http://localhost:5000 in browser
```

**Potential issues to watch for**:
- The `LOOP_IMG='output/looper.png'` path in main.py is relative — make sure it resolves correctly from the WSL working directory
- The `torch.inference_mode()` context manager in `looper_workflow.py` and `interactive_loop.py` — this works with CPU torch but confirm it doesn't cause issues
- WebSocket connection from WSL to Windows — ensure firewall allows port 8188

### 4. Architecture summary

```
WSL (looper)                         Windows (ComfyUI)
─────────────                        ──────────────────
main.py                              ComfyUI server :8188
  → ComfyUIClient ──HTTP/WS──────→  POST /prompt
  → APIWorkflowEngine                POST /upload/image
    → build_*_workflow()              GET /view, /history
    → client.upload_image()           WebSocket /ws
    → client.execute_workflow()
    → client.download_image()
  → transforms (local PIL/OpenCV)
  → animation (local moviepy)
  → Flask interactive UI :5000
```
