import threading
from enum import Enum
from typing import Optional


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
