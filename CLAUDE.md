# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI Looper generates psychedelic looping animations by repeatedly feeding img2img output back into a diffusion model. It supports SDXL, SD3.5, and Flux.1D models, with features like audio-driven FFT parameter modulation, expression-based dynamic controls, and image transforms.

## Commands

### Run tests
```bash
pytest
```

### Run a single test
```bash
pytest tests/test_transforms.py::test_name
```

### Run the main program
```bash
python comfyui_looper/main.py -w sdxl -i <input_img> -o <output_folder> -j <workflow_json> -p <passes> -t mp4 -a output.mp4 -x frame_delay:110 -x max_dim:768 -x v_bitrate:4000k
```

### Run transform tester
```bash
python scripts/transform_tester.py -i <input_img> -o <output_folder> -n <num_iterations>
```

## Architecture

### Core Loop (`comfyui_looper/workflow/looper_workflow.py`)
`looper_main()` drives the iteration loop: for each iteration it reads settings from JSON, evaluates expressions, applies image transforms, calls `engine.compute_iteration()`, and saves output. After all iterations, it assembles the animation.

`WorkflowEngine` is the abstract base class. Each model engine overrides `setup()`, `compute_iteration()`, and `resize_images_for_model()`.

### Workflow Engines (`comfyui_looper/workflow/`)
- `sdxl_engine.py` — SDXL (1024x1024, 8x latent reduction)
- `flux1d_engine.py` — Flux.1D (1024x1024, beta scheduler)
- `sd3p5_engine.py` — SD3.5 (1024x1024)
- `engine_factory.py` — Registry mapping engine names to classes via `WORKFLOW_LIBRARY` dict

### Settings System (`comfyui_looper/utils/json_spec.py`)
`LoopSettings` is a dataclass representing per-iteration config (prompts, seeds, denoise params, transforms, loras, canny, etc.). Unset fields use sentinel values (`EMPTY_OBJECT`, `EMPTY_LIST`) that trigger inheritance from the previous `LoopSettings` in the workflow.

`SettingsManager` handles JSON parsing (with `//` comment stripping), backward lookup for inherited settings, and expression evaluation. It pre-evaluates all iterations during `validate()` to catch errors early.

Workflow JSON files live in `data/` and are collections of `LoopSettings` wrapped in a `Workflow` dataclass (with schema versioning via `dataclasses_json`).

### Expression Evaluation (`comfyui_looper/utils/simple_expr_eval.py`)
`SimpleExprEval` safely evaluates math expressions in string-typed settings. Automatic variables: `n` (iteration), `offset` (section start), `total_n` (total iterations). Supports `sin`, `cos`, `sqrt`, `floor`, `ceil`, etc. Custom functions like `get_power_at_freq_range(low_f, high_f)` enable FFT-driven effects.

Expressions work in: `denoise_amt`, `denoise_steps`, `cfg`, `con_deltas` strength, `loras` strength, `canny` fields, and transform parameters.

### Image Transforms (`comfyui_looper/image_processing/transforms.py`)
`Transform` enum with 12+ types (zoom variants, fold, squeeze, fisheye, rotate, paste_img, wave, perspective). Each transform has expression-evaluable parameters. `load_image_with_transforms()` loads an image and applies a chain of transforms.

### Audio/FFT (`comfyui_looper/utils/fft.py`)
`WaveFile` handles MP3→WAV conversion and Welch power spectral density analysis. Used via `get_power_at_freq_range()` in expressions to modulate parameters by audio frequency content.

### Animation (`comfyui_looper/image_processing/animator.py`)
`make_animation()` creates GIF or MP4 from output image sequences. Supports frame delay, max dimension scaling, video bitrate, audio sync, and bounce (ping-pong) effects. Also has a standalone CLI entry point.

### ComfyUI Integration (`comfyui_looper/utils/node_wrappers.py`)
Abstractions over ComfyUI nodes: `LoraManager`, `CheckpointManager`, `ClipEncodeWrapper`, `Flux1DModelManager`. These handle model/LoRA caching and loading.

## Key Patterns

- **Sentinel-based inheritance**: `LoopSettings` fields default to sentinel values (`EMPTY_OBJECT`, `EMPTY_LIST`, `EMPTY_DICT`). `SettingsManager.get_setting_for_iter()` walks backward through previous settings to find the last explicitly-set value.
- **Expression strings**: Numeric fields can be strings containing expressions (e.g., `"0.2 + (0.5/35)*n"`). The `eval_expressions()` method in `SettingsManager` resolves these per-iteration.
- **pytest pythonpath**: `pytest.ini` sets `pythonpath=./comfyui_looper`, so internal imports use bare module names (e.g., `from utils.json_spec import ...`), not `comfyui_looper.utils.json_spec`.
- **ComfyUI dependency**: The project imports ComfyUI at runtime via `add_comfyui_directory_to_sys_path()`. Tests that don't need ComfyUI can run without it.
