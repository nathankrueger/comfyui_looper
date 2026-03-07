import os
import shutil
import torch
from typing import IO
from dataclasses import fields
from PIL.PngImagePlugin import PngInfo

from image_processing.animator import make_animation
from utils.json_spec import SettingsManager, default_seed, LoopSettings
from image_processing.transforms import load_image_with_transforms, AutomaticTransformParams
from utils.util import save_tensor_to_images, get_loop_img_filename
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
    prev_seed: int | None,
    force_new_seed: bool = False,
) -> int:
    """Run a single iteration. Returns the seed used."""
    loopsettings = sm.get_elaborated_loopsettings_for_iter(iter)
    transforms = loopsettings.transforms
    seed = loopsettings.seed

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
    output_image_filename = os.path.join(output_folder, get_loop_img_filename(image_index))
    pnginfo = PngInfo()
    pnginfo.add_text(key='looper_settings', value=loopsettings_json, zip=False)
    save_tensor_to_images(
        output_filenames=[loop_img_path, output_image_filename],
        image=vae_decode_result[0],
        png_info=pnginfo
    )

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
) -> tuple[int, int]:
    """
    Restart from a viewed image index (1-based file index).

    1. Delete image files from restart_from onward
    2. Copy image (restart_from - 1) to looper.png
    3. Regenerate iteration (restart_from - 1) with a new seed
    4. Pause the loop
    5. Return (next_iter, prev_seed)
    """
    redo_iter = restart_from - 1

    # Delete images from restart_from onward
    for idx in range(restart_from, total_iter + 1):
        img_path = os.path.join(output_folder, get_loop_img_filename(idx))
        if os.path.exists(img_path):
            os.remove(img_path)

    # Clear cached settings from redo_iter onward
    state.clear_settings_from(redo_iter)

    # Copy the previous image as the new input
    input_image_path = os.path.join(output_folder, get_loop_img_filename(restart_from - 1))
    shutil.copy(input_image_path, loop_img_path)

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
        prev_seed=None,
        force_new_seed=True,
    )

    # Pause — user must approve or restart again
    state.pause()

    # Next iteration to run when resumed
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
):
    sm = SettingsManager(json_file, animation_params)
    sm.validate()

    try:
        with torch.inference_mode():
            engine.setup()
            prev_seed = None
            total_iter = sm.get_total_iterations()
            iter = 0

            while iter < total_iter:
                # Check for pause
                state.wait_if_paused()

                # Check for restart request
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
                    )
                    continue

                # Normal iteration
                state.set_current_iteration(iter)
                state.set_status(LoopStatus.RUNNING)

                prev_seed = _run_iteration(
                    engine=engine,
                    sm=sm,
                    iter=iter,
                    total_iter=total_iter,
                    loop_img_path=loop_img_path,
                    output_folder=output_folder,
                    log_file=log_file,
                    state=state,
                    prev_seed=prev_seed,
                )

                iter += 1

        state.set_status(LoopStatus.STOPPED)
        if animation_file is not None:
            make_animation(
                type=animation_type,
                input_folder=output_folder,
                output_animation=os.path.join(output_folder, animation_file),
                params=animation_params
            )

    except Exception as e:
        state.set_error(str(e))
        raise
