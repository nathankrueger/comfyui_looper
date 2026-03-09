import time
import threading
from enum import Enum
from typing import Any, Optional


class LoopStatus(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class LoopState:
    """Thread-safe shared state between the GPU loop thread and Flask server."""

    def __init__(self, total_iterations: int, output_folder: str):
        self._lock = threading.Lock()
        self._status = LoopStatus.RUNNING
        self._current_iteration = 0
        self._total_iterations = total_iterations
        self._output_folder = output_folder
        self._latest_image_index = 0
        self._elaborated_settings: dict[int, str] = {}
        self._pause_event = threading.Event()
        self._pause_event.set()  # starts unpaused
        self._restart_request: Optional[int] = None
        self._error: Optional[str] = None
        self._export_status: Optional[str] = None
        self._export_error: Optional[str] = None
        self._export_file: Optional[str] = None
        self._frame_overrides: dict[str, Any] = {}
        self._settings_manager = None
        self._iteration_timestamps: list[float] = []
        self._iter_start_time: Optional[float] = None
        self._stop_event = threading.Event()  # starts unset (not stopped)

    # --- Status ---

    def get_status(self) -> LoopStatus:
        with self._lock:
            return self._status

    def set_status(self, status: LoopStatus):
        with self._lock:
            self._status = status

    # --- Iteration tracking ---

    def get_current_iteration(self) -> int:
        with self._lock:
            return self._current_iteration

    def set_current_iteration(self, iter_num: int):
        with self._lock:
            self._current_iteration = iter_num

    def get_total_iterations(self) -> int:
        with self._lock:
            return self._total_iterations

    def get_latest_image_index(self) -> int:
        with self._lock:
            return self._latest_image_index

    def set_latest_image_index(self, idx: int):
        with self._lock:
            self._latest_image_index = idx

    def get_output_folder(self) -> str:
        return self._output_folder

    # --- Settings cache ---

    def store_settings(self, iter_num: int, settings_json: str):
        with self._lock:
            self._elaborated_settings[iter_num] = settings_json

    def get_settings(self, iter_num: int) -> Optional[str]:
        with self._lock:
            return self._elaborated_settings.get(iter_num)

    def clear_settings_from(self, iter_num: int):
        with self._lock:
            keys_to_remove = [k for k in self._elaborated_settings if k >= iter_num]
            for k in keys_to_remove:
                del self._elaborated_settings[k]

    # --- Pause/resume ---

    def pause(self):
        with self._lock:
            self._status = LoopStatus.PAUSED
        self._pause_event.clear()

    def resume(self):
        with self._lock:
            self._status = LoopStatus.RUNNING
        self._pause_event.set()

    def wait_if_paused(self):
        self._pause_event.wait()

    # --- Restart ---

    def request_restart(self, from_image_index: int):
        with self._lock:
            self._restart_request = from_image_index

    def get_and_clear_restart_request(self) -> Optional[int]:
        with self._lock:
            req = self._restart_request
            self._restart_request = None
            return req

    # --- Error ---

    def set_error(self, error: str):
        with self._lock:
            self._error = error
            self._status = LoopStatus.STOPPED

    def get_error(self) -> Optional[str]:
        with self._lock:
            return self._error

    # --- Export ---

    def get_export_status(self) -> Optional[str]:
        with self._lock:
            return self._export_status

    def set_export_status(self, status: Optional[str]):
        with self._lock:
            self._export_status = status

    def get_export_error(self) -> Optional[str]:
        with self._lock:
            return self._export_error

    def set_export_error(self, error: Optional[str]):
        with self._lock:
            self._export_error = error

    def get_export_file(self) -> Optional[str]:
        with self._lock:
            return self._export_file

    def set_export_file(self, filepath: Optional[str]):
        with self._lock:
            self._export_file = filepath

    def clear_export(self):
        with self._lock:
            self._export_status = None
            self._export_error = None
            self._export_file = None

    # --- Frame overrides ---

    def set_frame_overrides(self, overrides: dict[str, Any]):
        with self._lock:
            self._frame_overrides = dict(overrides)

    def get_and_clear_frame_overrides(self) -> dict[str, Any]:
        with self._lock:
            overrides = self._frame_overrides
            self._frame_overrides = {}
            return overrides

    def get_frame_overrides(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._frame_overrides)

    def clear_frame_overrides(self):
        with self._lock:
            self._frame_overrides = {}

    # --- Iteration timing ---

    def mark_iteration_start(self):
        with self._lock:
            self._iter_start_time = time.monotonic()

    def mark_iteration_complete(self):
        with self._lock:
            if self._iter_start_time is not None:
                self._iteration_timestamps.append(time.monotonic() - self._iter_start_time)
                self._iter_start_time = None

    def clear_timestamps_from(self, iter_num: int):
        with self._lock:
            if iter_num < len(self._iteration_timestamps):
                self._iteration_timestamps = self._iteration_timestamps[:iter_num]

    def get_progress_info(self) -> dict:
        with self._lock:
            completed = self._latest_image_index
            total = self._total_iterations
            remaining = total - completed
            timestamps = self._iteration_timestamps

            if len(timestamps) == 0:
                return {
                    'completed': completed,
                    'total': total,
                    'remaining': remaining,
                    'avg_secs': None,
                    'eta_secs': None,
                    'elapsed_secs': None,
                }

            # Use recent window (last 10) for avg to account for changing iteration cost
            window = timestamps[-10:]
            avg_secs = sum(window) / len(window)
            eta_secs = avg_secs * remaining
            elapsed_secs = sum(timestamps)

            return {
                'completed': completed,
                'total': total,
                'remaining': remaining,
                'avg_secs': round(avg_secs, 1),
                'eta_secs': round(eta_secs, 1),
                'elapsed_secs': round(elapsed_secs, 1),
            }

    # --- Stop ---

    def request_stop(self):
        self._stop_event.set()

    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    # --- Settings manager reference ---

    def set_settings_manager(self, sm):
        self._settings_manager = sm

    def get_settings_manager(self):
        return self._settings_manager
