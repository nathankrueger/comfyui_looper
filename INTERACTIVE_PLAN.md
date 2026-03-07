# Plan: Interactive Web Control for ComfyUI Looper

## Implementation Status

All code is written and committed. 28 unit tests pass (LoopState + Flask routes).

| Step | Status | Notes |
|------|--------|-------|
| 1. `interactive/loop_state.py` | DONE | Thread-safe shared state, 13 unit tests |
| 2. `interactive/flask_app.py` | DONE | 7 REST endpoints, 15 unit tests |
| 3. `interactive/static/index.html` | DONE | Dark theme, mobile-responsive, polling |
| 4. `interactive/interactive_loop.py` | DONE | Modified loop with pause/restart |
| 5. `main.py` modifications | DONE | `--interactive` flag wired up |
| 6. `requirements.txt` | DONE | flask, gunicorn added |
| 7. `install.sh` | DONE | Creates/reuses .venv |
| **Manual smoke test on GPU** | TODO | Needs ComfyUI + GPU workstation |

### To test on GPU workstation:
```bash
bash install.sh
source .venv/bin/activate
python comfyui_looper/main.py --interactive -w sdxl -i <img> -o output/test -j data/star_wars.json
# Open http://<workstation-ip>:5000 on phone
```

## Context

The user wants to control the looping image generation from a phone/browser while the GPU loop runs on their workstation. The web UI lets them browse generated images, and critically, "fork" the generation when the AI produces a bad seed — going back and regenerating with a new seed to steer the scene.

## New Files

```
comfyui_looper/interactive/
  __init__.py                    # empty
  loop_state.py                  # Thread-safe shared state
  interactive_loop.py            # Modified loop with pause/resume/restart
  flask_app.py                   # Flask routes + API
  static/
    index.html                   # Single-page UI (vanilla HTML/CSS/JS)
```

```
tests/
  test_loop_state.py             # LoopState unit tests
  test_flask_app.py              # Flask route tests
```

## Modified Files

- `comfyui_looper/main.py` — Add `--interactive` flag, launch Flask + GPU loop thread
- `requirements.txt` — Add `flask`, `gunicorn`

## Architecture

```
main.py --interactive
  ├─ Creates LoopState (thread-safe shared state)
  ├─ Starts GPU loop in background Thread (daemon)
  │   └─ interactive_looper_main() — same as looper_main() but checks
  │      LoopState between iterations for pause/restart commands
  └─ Starts Flask dev server on 0.0.0.0:5000
      └─ Serves index.html + REST API, reads/writes LoopState
```

The GPU loop and Flask share a single `LoopState` object protected by `threading.Lock`. Communication is through atomic state checks — the loop polls for commands at the top of each iteration (cheap, since iterations take seconds of GPU time).

## Step 1: `interactive/loop_state.py`

Thread-safe shared state class `LoopState` with:
- **Status**: `RUNNING` / `PAUSED` / `STOPPED` (enum)
- **Iteration tracking**: `current_iteration`, `total_iterations`, `latest_image_index` (1-based file index)
- **Settings cache**: `dict[int, str]` mapping iteration → JSON string, with `clear_settings_from(n)` for restart cleanup
- **Pause/resume**: `threading.Event` — `wait_if_paused()` blocks the loop thread, `pause()` clears the event, `resume()` sets it
- **Restart**: `request_restart(from_image_index)` stores an int, `get_and_clear_restart_request()` atomically reads and clears it
- **Error**: `set_error(str)` sets status to STOPPED

**Bug fix from plan agent**: `resume()` must NOT clear `_restart_request`. Only `get_and_clear_restart_request()` clears it. Otherwise restarting from a paused state loses the request (race condition: Flask sets restart, then calls resume which clears it).

## Step 2: `interactive/flask_app.py`

Flask app factory `create_app(state: LoopState) -> Flask`:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve index.html |
| GET | `/api/status` | Returns `{status, current_iteration, total_iterations, latest_image_index, error}` |
| GET | `/api/image/<index>` | Serve PNG image by file index (0=start, 1+=generated) |
| GET | `/api/settings/<index>` | Returns elaborated LoopSettings JSON for image |
| POST | `/api/restart` | Body: `{from_image_index: N}`. If loop is paused, calls `resume()` to unblock it so it can process the restart |
| POST | `/api/resume` | Resume from pause |
| POST | `/api/pause` | Pause the loop |

Images served via `send_file()` from the output folder. Settings served as raw JSON strings from the LoopState cache.

## Step 3: `interactive/static/index.html`

Single-page, vanilla HTML/CSS/JS. Dark theme, mobile-responsive (flexbox, viewport meta, large touch targets).

**Layout:**
```
[STATUS BAR: RUNNING | Iteration 5/50]
[IMAGE (max-width: 100%)]
[Image 5 / 30 label]
[GO BACK] [RESTART] [RESUME*] [PAUSE] [SYNC] [GO FORWARD]
[Settings JSON panel (pre, scrollable)]
```
*RESUME shown only when paused; PAUSE hidden when paused/stopped.

**JS behavior:**
- `viewIndex` (client-side): which image is displayed
- `isLiveMode`: when true, auto-advances to latest on each poll
- Polls `/api/status` every 1.5s. If live mode and new image available, updates view
- GO BACK/FORWARD: adjust `viewIndex`, set `isLiveMode = false`
- SYNC: `viewIndex = latest`, `isLiveMode = true`
- RESTART: confirmation dialog, then `POST /api/restart` with current `viewIndex`
- Cache-bust image URLs (`?t=Date.now()`) so restarted images refresh correctly

## Step 4: `interactive/interactive_loop.py`

`interactive_looper_main()` — modified `looper_main()` from [looper_workflow.py](comfyui_looper/workflow/looper_workflow.py).

Key differences from non-interactive loop:
- Uses `while iter < total_iter` instead of `for iter in range(total_iter)` (restart can rewind `iter`)
- Top of each iteration: `state.wait_if_paused()`, then `state.get_and_clear_restart_request()`
- Updates `LoopState` after each iteration: `set_latest_image_index()`, `store_settings()`
- Wraps everything in try/except to call `state.set_error()` on failure

**`_handle_restart(restart_from)` helper:**
1. `redo_iter = restart_from - 1` (convert file index to 0-based iteration)
2. Delete files `loop_img_{restart_from}` through `loop_img_{total_iter}` from output folder
3. Copy `loop_img_{restart_from - 1}` to `looper.png`
4. Get settings via `sm.get_elaborated_loopsettings_for_iter(redo_iter)`, override seed with `default_seed()`
5. Call `sm.update_seed(redo_iter, new_seed)`
6. Load image, apply transforms, run `engine.compute_iteration()`
7. Save to both `looper.png` and `loop_img_{restart_from}`
8. Update LoopState: `set_latest_image_index(restart_from)`, `store_settings()`, `clear_settings_from(redo_iter)`
9. Call `state.pause()` — loop will block at top of next iteration
10. Return `(redo_iter + 1, new_seed)` so main while-loop continues from there when resumed

**Restart walkthrough** (user views image 25 of 50, clicks RESTART):
- Loop finishes current GPU work, sees restart request
- Deletes files 25-50, copies file 24 to looper.png
- Generates new image 25 from image 24 + settings for iter 24 + new seed
- Pauses. User sees new image 25, can RESTART again or RESUME
- On RESUME: loop continues iter 25→49, generating images 26→50 (original total preserved)

Reuse from existing code:
- `load_image_with_transforms()` from [transforms.py](comfyui_looper/image_processing/transforms.py)
- `save_tensor_to_images()`, `get_loop_img_filename()` from [util.py](comfyui_looper/utils/util.py)
- `SettingsManager`, `default_seed()` from [json_spec.py](comfyui_looper/utils/json_spec.py)
- `AutomaticTransformParams` from [transforms.py](comfyui_looper/image_processing/transforms.py)
- `make_animation()` from [animator.py](comfyui_looper/image_processing/animator.py)

## Step 5: Modify `main.py`

Add `--interactive` flag (`store_true`). Branch at the bottom:

```python
if args.interactive:
    # Conditional imports (avoid pulling Flask when non-interactive)
    # Only single pass in interactive mode (warn if passes > 1)
    # Create SettingsManager temporarily to get total_iterations
    # Create LoopState
    # Setup output folder, resize input image (same as existing code)
    # Open log file
    # Start GPU loop in daemon Thread
    # Create Flask app, run on 0.0.0.0:5000
else:
    # Existing non-interactive code (unchanged)
```

## Step 6: Update `requirements.txt`

Add `flask` and `gunicorn`.

## Verification

1. **Unit tests**: `pytest tests/test_loop_state.py tests/test_flask_app.py` — tests LoopState thread safety (including actual thread blocking), Flask routes, edge cases
2. **Manual smoke test** (requires ComfyUI + GPU):
   - Run `python main.py --interactive -w sdxl -i <img> -o output/test -j data/star_wars.json -p 1`
   - Open `http://<workstation-ip>:5000` on phone
   - Verify: images auto-refresh, GO BACK/FORWARD navigate, RESTART regenerates + pauses, RESUME continues, stats show full settings
3. **Non-interactive regression**: Run existing non-interactive mode, verify no behavior change
