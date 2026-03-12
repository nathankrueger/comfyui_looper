import glob
import os
import logging
import re
import shutil
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from interactive.loop_state import LoopState

logger = logging.getLogger(__name__)
from interactive.interactive_loop import interactive_looper_main
from workflow.engine_factory import create_workflow
from utils.json_spec import SettingsManager
from utils.comfyui_client import ComfyUIClient
from utils.image_store import ImageStore, FilesystemImageStore, ZipImageStore
from utils.util import get_log_filename, get_loop_img_filename

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_BASENAME = 'looper_log.log'
LOOP_IMG = str(PROJECT_ROOT / 'output' / 'looper.png')


def get_default_output_folder(json_file: str) -> str:
    """Generate a default output folder name from the workflow JSON filename and a timestamp."""
    workflow_name = Path(json_file).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return str(PROJECT_ROOT / 'output' / f'{workflow_name}_{timestamp}')


class AppState:
    """Manages the lifecycle of loop sessions. Flask gets a reference to this."""

    def __init__(
        self,
        workflow_type: str,
        comfyui_url: str,
        input_img: Optional[str],
        animation_type: str,
        animation_filename: Optional[str],
        animation_params: dict,
        use_zip: bool = False,
    ):
        self._workflow_type = workflow_type
        self._input_img = input_img
        self._animation_type = animation_type
        self._animation_filename = animation_filename
        self._animation_params = animation_params
        self._client = ComfyUIClient(comfyui_url)
        self._use_zip = use_zip

        self._lock = threading.Lock()
        self._loop_state: Optional[LoopState] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._current_json_file: Optional[str] = None
        self._current_output_folder: Optional[str] = None
        self._log_file = None
        self._image_store: Optional[ImageStore] = None

    def start_loop(self, json_file: str, output_folder: Optional[str] = None, input_img: Optional[str] = None) -> dict:
        """Start a new loop session. Raises if a loop is already running.

        input_img overrides the CLI-provided input image:
          - None  → use CLI default (self._input_img)
          - ""    → force txt2img (no starting image)
          - path  → use that image
        """
        with self._lock:
            if self._loop_state is not None and self._loop_thread is not None and self._loop_thread.is_alive():
                raise RuntimeError("A loop is already running. Stop it first.")

            if output_folder is None:
                output_folder = get_default_output_folder(json_file)

            output_folder = os.path.abspath(output_folder)
            os.makedirs(output_folder, exist_ok=True)

            # Copy workflow JSON to output folder (exact byte-for-byte copy)
            shutil.copy2(json_file, os.path.join(output_folder, Path(json_file).name))

            starting_point_filename = str(Path(output_folder) / get_loop_img_filename(0))

            # create the image store
            if self._use_zip:
                image_store = ZipImageStore(output_folder)
            else:
                image_store = FilesystemImageStore(output_folder)
            self._image_store = image_store

            # Determine effective input image
            if input_img is not None:
                effective_input_img = input_img if input_img != "" else None
            else:
                effective_input_img = self._input_img

            engine = create_workflow(self._workflow_type, self._client)
            no_input_image = effective_input_img is None
            if no_input_image:
                engine.create_blank_image_for_model([LOOP_IMG])
            else:
                engine.resize_images_for_model(effective_input_img, [LOOP_IMG])
                image_store.import_from_path(LOOP_IMG, get_loop_img_filename(0))

            sm = SettingsManager(json_file, self._animation_params)
            sm.validate()
            total_iterations = sm.get_total_iterations()

            loop_state = LoopState(total_iterations=total_iterations, output_folder=output_folder, no_input_image=no_input_image)

            log_filename = get_log_filename(LOG_BASENAME)
            log_file = open(os.path.join(output_folder, log_filename), 'w', encoding='utf-8')

            self._loop_state = loop_state
            self._current_json_file = json_file
            self._current_output_folder = output_folder
            self._log_file = log_file

            def run_loop():
                try:
                    interactive_looper_main(
                        engine=engine,
                        loop_img_path=LOOP_IMG,
                        output_folder=output_folder,
                        json_file=json_file,
                        animation_file=self._animation_filename,
                        animation_type=self._animation_type,
                        animation_params=self._animation_params,
                        log_file=log_file,
                        state=loop_state,
                        image_store=image_store,
                        no_input_image=no_input_image,
                    )
                except Exception as e:
                    logger.error("Loop thread error: %s", e, exc_info=True)
                finally:
                    try:
                        log_file.close()
                    except Exception:
                        pass

            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
            self._loop_thread = thread

            return {
                'json_file': json_file,
                'output_folder': output_folder,
                'total_iterations': total_iterations,
            }

    def resume_loop(self, folder_name: str) -> dict:
        """Resume a previous loop session from an output folder. Starts paused."""
        with self._lock:
            if self._loop_state is not None and self._loop_thread is not None and self._loop_thread.is_alive():
                raise RuntimeError("A loop is already running. Stop it first.")

            output_folder = str(PROJECT_ROOT / 'output' / folder_name)
            if not os.path.isdir(output_folder):
                raise FileNotFoundError(f"Output folder not found: {folder_name}")

            # Find the workflow JSON in the output folder
            json_files = [f for f in os.listdir(output_folder)
                          if f.endswith('.json') and not f.startswith('.')]
            if not json_files:
                raise FileNotFoundError(f"No workflow JSON found in {folder_name}")
            json_file = os.path.join(output_folder, json_files[0])

            # Detect image store type
            zip_path = os.path.join(output_folder, 'images.zip')
            if os.path.exists(zip_path):
                image_store = ZipImageStore(output_folder)
            else:
                image_store = FilesystemImageStore(output_folder)
            self._image_store = image_store

            # Determine latest_image_index from stored images
            images = image_store.list_images()
            loop_images = sorted([img for img in images if re.match(r'loop_img_\d+\.png$', img)])
            if not loop_images:
                raise FileNotFoundError(f"No images found in {folder_name}")

            # Parse highest image index
            last_img = loop_images[-1]
            latest_image_index = int(re.search(r'(\d+)', last_img).group(1))

            # Determine no_input_image: if loop_img_000000.png exists, we had an input image
            no_input_image = not image_store.has_image(get_loop_img_filename(0))

            # Copy latest image to LOOP_IMG (working image for next iteration)
            engine = create_workflow(self._workflow_type, self._client)
            if latest_image_index > 0:
                image_store.copy_image_to_path(get_loop_img_filename(latest_image_index), LOOP_IMG)
            elif not no_input_image:
                image_store.copy_image_to_path(get_loop_img_filename(0), LOOP_IMG)
            else:
                engine.create_blank_image_for_model([LOOP_IMG])

            # Validate settings from the output folder JSON
            sm = SettingsManager(json_file, self._animation_params)
            sm.validate()
            total_iterations = sm.get_total_iterations()

            # start_iter = latest_image_index (image N is produced by iteration N-1,
            # so the next iteration to run is latest_image_index)
            start_iter = latest_image_index

            loop_state = LoopState(
                total_iterations=total_iterations,
                output_folder=output_folder,
                no_input_image=no_input_image,
            )
            loop_state.pause()  # Start paused so user can inspect/adjust
            loop_state.set_latest_image_index(latest_image_index)

            # Pre-populate elaborated settings and settings manager so the
            # View tab works immediately (before the background thread starts).
            # interactive_looper_main will redo this, which is harmless.
            loop_state.set_settings_manager(sm)
            loop_state.init_pre_elaborated(sm, json_file, self._animation_params)

            log_filename = get_log_filename(LOG_BASENAME)
            log_file = open(os.path.join(output_folder, log_filename), 'a', encoding='utf-8')
            log_file.write(f"\n[RESUMED from image {latest_image_index}]\n")
            log_file.flush()

            self._loop_state = loop_state
            self._current_json_file = json_file
            self._current_output_folder = output_folder
            self._log_file = log_file

            def run_loop():
                try:
                    interactive_looper_main(
                        engine=engine,
                        loop_img_path=LOOP_IMG,
                        output_folder=output_folder,
                        json_file=json_file,
                        animation_file=self._animation_filename,
                        animation_type=self._animation_type,
                        animation_params=self._animation_params,
                        log_file=log_file,
                        state=loop_state,
                        image_store=image_store,
                        no_input_image=no_input_image,
                        start_iter=start_iter,
                    )
                except Exception as e:
                    logger.error("Loop thread error: %s", e, exc_info=True)
                finally:
                    try:
                        log_file.close()
                    except Exception:
                        pass

            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
            self._loop_thread = thread

            return {
                'json_file': json_file,
                'output_folder': output_folder,
                'total_iterations': total_iterations,
                'latest_image_index': latest_image_index,
            }

    def stop_loop(self):
        """Stop the current loop session. No-op if no loop is running."""
        with self._lock:
            if self._loop_state is None:
                return

            loop_state = self._loop_state
            loop_thread = self._loop_thread

        # Signal stop and unblock if paused (outside lock to avoid deadlock)
        loop_state.request_stop()
        loop_state.resume()

        if loop_thread is not None:
            loop_thread.join(timeout=15)

        with self._lock:
            if self._log_file is not None:
                try:
                    self._log_file.close()
                except Exception:
                    pass
            if self._image_store is not None:
                try:
                    self._image_store.close()
                except Exception:
                    pass
            self._loop_state = None
            self._loop_thread = None
            self._current_json_file = None
            self._current_output_folder = None
            self._log_file = None
            self._image_store = None

    def is_loop_running(self) -> bool:
        with self._lock:
            return (
                self._loop_state is not None
                and self._loop_thread is not None
                and self._loop_thread.is_alive()
            )

    def get_loop_state(self) -> Optional[LoopState]:
        with self._lock:
            return self._loop_state

    def get_current_json_file(self) -> Optional[str]:
        with self._lock:
            return self._current_json_file

    def get_current_json_name(self) -> Optional[str]:
        with self._lock:
            if self._current_json_file is None:
                return None
            return os.path.basename(self._current_json_file)

    def get_current_output_folder(self) -> Optional[str]:
        with self._lock:
            return self._current_output_folder

    def get_image_store(self) -> Optional[ImageStore]:
        with self._lock:
            return self._image_store
