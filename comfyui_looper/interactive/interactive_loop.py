import os
import shutil
import time
import torch
from typing import IO
from dataclasses import fields
from PIL.PngImagePlugin import PngInfo

from image_processing.animator import make_animation
from utils.json_spec import SettingsManager, default_seed, LoopSettings
from image_processing.transforms import load_image_with_transforms, AutomaticTransformParams
from utils.image_store import ImageStore
from utils.util import save_tensor_to_images, tensor_to_pil, get_loop_img_filename
from workflow.looper_workflow import WorkflowEngine
from interactive.loop_state import LoopState, LoopStatus


def _apply_engine_defaults(engine: WorkflowEngine, loopsettings: LoopSettings):
    for setting in fields(loopsettings):
        setting_name = setting.name
        setting_val = getattr(loopsettings, setting_name)
        if setting_val is None or (isinstance(setting_val, list) and len(setting_val) == 0):
            if (default := engine.get_default_for_setting(setting_name)) is not None:
                setattr(loopsettings, setting_name, default)


def _run_iteration(
    engine: WorkflowEngine,
    sm: SettingsManager,
    iter: int,
    total_iter: int,
    loop_img_path: str,
    output_folder: str,
    log_file: IO[str],
    state: LoopState,
    image_store: ImageStore,
    prev_seed: int | None,
    force_new_seed: bool = False,
    no_input_image: bool = False,
) -> int:
    """Run a single iteration. Returns the seed used."""
    loopsettings = sm.get_elaborated_loopsettings_for_iter(iter)

    # First iteration with no input image: force full denoise (txt2img)
    if iter == 0 and no_input_image:
        loopsettings.denoise_amt = 1.0

    # Apply one-shot frame overrides from the UI
    frame_overrides = state.get_and_clear_frame_overrides()
    for field_name, value in frame_overrides.items():
        setattr(loopsettings, field_name, value)

    transforms = loopsettings.transforms
    seed = loopsettings.seed

    if 'seed' not in frame_overrides:
        if force_new_seed or seed == prev_seed:
            seed = default_seed()
            loopsettings.seed = seed
    sm.update_seed(iter, seed)

    _apply_engine_defaults(engine, loopsettings)

    auto_params = AutomaticTransformParams(
        n=iter, offset=loopsettings.offset,
        total_n=total_iter, wavefile=sm.get_wavefile()
    )
    image_tensor, loopsettings.transforms = load_image_with_transforms(
        image_path=loop_img_path,
        transforms=transforms,
        auto_params=auto_params
    )

    vae_decode_result = engine.compute_iteration(image_tensor, loopsettings)

    loopsettings_json = loopsettings.to_json(indent=4)
    image_index = iter + 1
    pnginfo = PngInfo()
    pnginfo.add_text(key='looper_settings', value=loopsettings_json, zip=False)

    # save the working loop image to disk
    save_tensor_to_images(
        output_filenames=[loop_img_path],
        image=vae_decode_result[0],
        png_info=pnginfo
    )

    # save the numbered archive image via the store
    output_image_filename = get_loop_img_filename(image_index)
    pil_img = tensor_to_pil(vae_decode_result[0])
    image_store.write_image(output_image_filename, pil_img, png_info=pnginfo)

    state.set_latest_image_index(image_index)
    state.store_settings(iter, loopsettings_json)

    log_file.write(f"{output_image_filename}:\n{loopsettings_json}\n\n")
    log_file.flush()

    return seed


def _handle_restart(
    restart_from: int,
    engine: WorkflowEngine,
    sm: SettingsManager,
    total_iter: int,
    loop_img_path: str,
    output_folder: str,
    log_file: IO[str],
    state: LoopState,
    image_store: ImageStore,
    no_input_image: bool = False,
) -> tuple[int, int]:
    """
    Restart from a viewed image index (1-based file index).

    1. Delete image files from restart_from onward
    2. Copy image (restart_from - 1) to looper.png
    3. Regenerate iteration (restart_from - 1) with a new seed
    4. Return (next_iter, prev_seed)
    """
    redo_iter = restart_from - 1
    state.clear_frame_overrides()
    state.clear_timestamps_from(redo_iter)

    # Delete images from restart_from onward
    filenames_to_delete = [get_loop_img_filename(idx) for idx in range(restart_from, total_iter + 1)]
    image_store.delete_images(filenames_to_delete)

    # Clear cached settings from redo_iter onward
    state.clear_settings_from(redo_iter)

    # Update latest_image_index BEFORE regeneration so the UI knows the valid range
    state.set_latest_image_index(restart_from - 1)
    state.set_current_iteration(redo_iter)

    # Copy the previous image as the new input
    prev_filename = get_loop_img_filename(restart_from - 1)
    if image_store.has_image(prev_filename):
        image_store.copy_image_to_path(prev_filename, loop_img_path)
    elif no_input_image and restart_from == 1:
        # txt2img mode: frame 0 was never saved, create a blank image
        engine.create_blank_image_for_model([loop_img_path])
    else:
        raise FileNotFoundError(f"Cannot restart: {prev_filename} not found in image store")

    # Regenerate with a new seed
    log_file.write(f"[RESTART from image {restart_from}]\n")
    new_seed = _run_iteration(
        engine=engine,
        sm=sm,
        iter=redo_iter,
        total_iter=total_iter,
        loop_img_path=loop_img_path,
        output_folder=output_folder,
        log_file=log_file,
        state=state,
        image_store=image_store,
        prev_seed=None,
        force_new_seed=True,
        no_input_image=no_input_image,
    )

    # Next iteration to run
    return (redo_iter + 1, new_seed)


def interactive_looper_main(
    engine: WorkflowEngine,
    loop_img_path: str,
    output_folder: str,
    json_file: str,
    animation_file: str | None,
    animation_type: str,
    animation_params: dict[str, str],
    log_file: IO[str],
    state: LoopState,
    image_store: ImageStore = None,
    no_input_image: bool = False,
):
    sm = SettingsManager(json_file, animation_params)
    sm.validate()
    state.set_settings_manager(sm)

    try:
        with torch.inference_mode():
            engine.setup()
            prev_seed = None
            total_iter = sm.get_total_iterations()
            iter = 0

            while True:
                # Run iterations until complete
                while iter < total_iter:
                    # Check for stop (reset/picker return)
                    if state.is_stop_requested():
                        state.set_status(LoopStatus.STOPPED)
                        return

                    # Check for restart request before anything else so
                    # restarts are processed immediately — including after
                    # being woken from pause (request_restart unblocks the
                    # pause event)
                    restart_from = state.get_and_clear_restart_request()
                    if restart_from is not None:
                        iter, prev_seed = _handle_restart(
                            restart_from=restart_from,
                            engine=engine,
                            sm=sm,
                            total_iter=total_iter,
                            loop_img_path=loop_img_path,
                            output_folder=output_folder,
                            log_file=log_file,
                            state=state,
                            image_store=image_store,
                            no_input_image=no_input_image,
                        )
                        continue

                    # Block while paused. Also wakes on restart request
                    # (which sets the event); we just continue back to the
                    # top where the restart check will pick it up.
                    if state.get_status() == LoopStatus.PAUSED:
                        state.wait_if_paused()
                        continue

                    # Normal iteration
                    state.set_current_iteration(iter)
                    state.set_status(LoopStatus.RUNNING)
                    state.mark_iteration_start()

                    prev_seed = _run_iteration(
                        engine=engine,
                        sm=sm,
                        iter=iter,
                        total_iter=total_iter,
                        loop_img_path=loop_img_path,
                        output_folder=output_folder,
                        log_file=log_file,
                        state=state,
                        image_store=image_store,
                        prev_seed=prev_seed,
                        no_input_image=no_input_image,
                    )

                    state.mark_iteration_complete()
                    iter += 1

                # All iterations complete
                state.set_status(LoopStatus.COMPLETED)
                if animation_file is not None:
                    anim_folder, needs_cleanup = image_store.get_paths_for_animation()
                    try:
                        make_animation(
                            type=animation_type,
                            input_folder=anim_folder,
                            output_animation=os.path.join(output_folder, animation_file),
                            params=animation_params
                        )
                    finally:
                        if needs_cleanup:
                            shutil.rmtree(anim_folder, ignore_errors=True)

                # Wait for restart requests after completion
                while not state.is_stop_requested():
                    restart_from = state.get_and_clear_restart_request()
                    if restart_from is not None:
                        iter, prev_seed = _handle_restart(
                            restart_from=restart_from,
                            engine=engine,
                            sm=sm,
                            total_iter=total_iter,
                            loop_img_path=loop_img_path,
                            output_folder=output_folder,
                            log_file=log_file,
                            state=state,
                            image_store=image_store,
                            no_input_image=no_input_image,
                        )
                        break  # Back to the main iteration loop
                    time.sleep(0.5)
                else:
                    # Stop was requested during the wait
                    state.set_status(LoopStatus.STOPPED)
                    return

    except Exception as e:
        state.set_error(str(e))
        raise
