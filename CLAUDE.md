# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI Looper generates psychedelic looping animations by repeatedly feeding img2img output back into a diffusion model via a remote ComfyUI server. It supports SDXL, SD3.5, and Flux.1D models, with features like audio-driven FFT parameter modulation, expression-based dynamic controls, 24+ image transforms, and an interactive web UI for live loop control.

## Commands

### Run tests
```bash
pytest
```

### Run a single test
```bash
pytest tests/test_transforms.py::test_name
```

### Run headless (batch mode)
```bash
python comfyui_looper/main.py -w sdxl -i <input_img> -o <output_folder> -j <workflow_json> -p <passes> -t mp4 -a output.mp4 -x frame_delay:110 -x max_dim:768 -x v_bitrate:4000k
```

### Run interactive web UI
```bash
python comfyui_looper/main.py -I -P 5000 -w sdxl [-j workflow.json]
```
Without `-j`, opens workflow picker page. With `-j`, auto-starts the loop.

### Dump elaborated settings (no diffusion)
```bash
python comfyui_looper/main.py -j workflow.json -l output_log.txt
```

### Run transform tester
```bash
python scripts/transform_tester.py -i <input_img> -o <output_folder> -n <num_iterations>
```

### Generate transform example GIFs
```bash
python scripts/generate_transform_examples.py
```

### CLI flags
- `-c, --comfyui-url` — ComfyUI server address (default: `http://localhost:8188`)
- `-z, --zip-storage` — Store output images in a ZIP file instead of loose PNGs
- `-I, --interactive` — Launch Flask web UI
- `-P, --port` — Web UI port (default: 5000)

## Architecture

### Core Loop (`comfyui_looper/workflow/looper_workflow.py`)
`looper_main()` drives the iteration loop: reads settings from JSON, evaluates expressions, applies image transforms, calls `engine.compute_iteration()`, and saves output. After all iterations, it assembles the animation.

`WorkflowEngine` is the abstract base class. Each model engine overrides `setup()`, `compute_iteration()`, and `resize_images_for_model()`.

### Unified API Engine (`comfyui_looper/workflow/api_engine.py`)
All models use the single `APIWorkflowEngine` class, which submits jobs to a remote ComfyUI server via HTTP. The `engine_factory.py` maps model names (`sdxl`, `flux1d`, `sd3.5`) to this engine with the appropriate model type.

### Dynamic Workflow Builder (`comfyui_looper/workflow/api_workflow_builder.py`)
Generates ComfyUI node graph JSON on-the-fly based on `LoopSettings`. Each model type has a builder function that handles checkpoint loading, CLIP encoding, LoRA chaining, ControlNet (Canny), and KSampler configuration. `BUILDER_MAP` maps model type strings to builder functions.

### Interactive Web UI (`comfyui_looper/interactive/`)
Flask-based web interface for live loop control:

- **`flask_app.py`** — REST API: workflow management (list/clone/edit/delete from `data/` and `data/user/`), image management (upload/list/serve from `data/img/`), loop control (pause/resume/restart from specific frame), frame-by-frame overrides, export (GIF/MP4), and settings introspection. Serves `picker.html` (workflow selection) or `index.html` (active loop control).
- **`app_state.py`** — Thread-safe session orchestration: manages loop lifecycle, tracks current JSON file and output folder, creates image store instances.
- **`loop_state.py`** — Thread-safe shared state between GPU loop thread and Flask: `LoopStatus` (running/paused/stopped), pause/resume synchronization, restart requests (jump to any frame), frame override accumulation, iteration timing/ETA, export status tracking, elaborated settings cache.
- **`interactive_loop.py`** — Modified loop runner that respects pause/resume events, handles restart requests (deletes images and regenerates from specified frame), applies one-shot frame overrides from the web UI, and tracks iteration timing.

### Settings System (`comfyui_looper/utils/json_spec.py`)
`LoopSettings` is a dataclass representing per-iteration config (prompts, seeds, denoise params, transforms, loras, canny, clip, etc.). Unset fields use sentinel values (`EMPTY_OBJECT`, `EMPTY_LIST`, `EMPTY_DICT`) that trigger inheritance from the previous `LoopSettings`.

`SettingsManager` handles JSON parsing (with `//` comment stripping), backward lookup for inherited settings, and expression evaluation. It pre-evaluates all iterations during `validate()` to catch errors early.

Workflow JSON files live in `data/` (built-in) and `data/user/` (user-created via web UI).

### Expression Evaluation (`comfyui_looper/utils/simple_expr_eval.py`)
`SimpleExprEval` safely evaluates math expressions in string-typed settings. Automatic variables: `n` (iteration), `offset` (section start), `total_n` (total iterations). Supports `sin`, `cos`, `sqrt`, `floor`, `ceil`, etc. Custom functions like `get_power_at_freq_range(low_f, high_f)` enable FFT-driven effects.

Expressions work in: `denoise_amt`, `denoise_steps`, `cfg`, `con_deltas` strength, `loras` strength, `canny` fields, and transform parameters.

### Image Transforms (`comfyui_looper/image_processing/transforms.py`)
24 transform classes inheriting from `Transform` base class, discovered via `all_subclasses()`. Categories: zoom variants (5), fold/squeeze (4), distortions (wave, spiral, ripple, elastic, fisheye), geometric (rotate, perspective, pan, mirror, kaleidoscope), color (hue_shift, color_channel_offset, contrast_brightness), composite (paste_img), and retro (pixelate). Each transform has expression-evaluable parameters via `EVAL_PARAMS`.

### Image Store (`comfyui_looper/utils/image_store.py`)
`ImageStore` ABC with two implementations:
- `FilesystemImageStore` — Traditional loose PNG files in output folder
- `ZipImageStore` — Thread-safe storage in a single `images.zip` file

### Audio/FFT (`comfyui_looper/utils/fft.py`)
`WaveFile` handles MP3→WAV conversion and Welch power spectral density analysis. Used via `get_power_at_freq_range()` in expressions to modulate parameters by audio frequency content.

### Animation (`comfyui_looper/image_processing/animator.py`)
`make_animation()` creates GIF or MP4 from output image sequences. Supports frame delay, max dimension scaling, video bitrate, audio sync, and bounce (ping-pong) effects. Also has a standalone CLI entry point.

### ComfyUI Client (`comfyui_looper/utils/comfyui_client.py`)
HTTP/WebSocket client for communicating with remote ComfyUI server: image upload/download, workflow execution with async completion waiting, output image extraction from history.

## Key Patterns

- **Sentinel-based inheritance**: `LoopSettings` fields default to sentinel values (`EMPTY_OBJECT`, `EMPTY_LIST`, `EMPTY_DICT`). `SettingsManager.get_setting_for_iter()` walks backward through previous settings to find the last explicitly-set value.
- **Expression strings**: Numeric fields can be strings containing expressions (e.g., `"0.2 + (0.5/35)*n"`). The `eval_expressions()` method in `SettingsManager` resolves these per-iteration.
- **pytest pythonpath**: `pytest.ini` sets `pythonpath=./comfyui_looper`, so internal imports use bare module names (e.g., `from utils.json_spec import ...`), not `comfyui_looper.utils.json_spec`.
- **ComfyUI dependency**: The project communicates with ComfyUI via HTTP API (`ComfyUIClient`). Tests that don't need ComfyUI can run without it.
- **Transform discovery**: Transform subclasses are auto-discovered via `all_subclasses(Transform)`, so adding a new transform just requires subclassing `Transform` with a `NAME` class attribute.
- **Thread-safe interactive loop**: The interactive mode runs the diffusion loop in a background thread, with `LoopState` providing lock-protected shared state for the Flask API to read/write.
