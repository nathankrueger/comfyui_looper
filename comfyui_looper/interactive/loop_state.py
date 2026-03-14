from __future__ import annotations

import json
import os
import shutil
import time
import threading
from copy import deepcopy
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.json_spec import LoopSettings, SettingsManager


def _remove_json_field(text: str, field_name: str) -> str:
    """Remove a top-level JSON field and its value from text, preserving other content."""
    lines = text.split('\n')

    # Find the line containing the field key (skip // comment lines)
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('//'):
            continue
        if f'"{field_name}"' in stripped and ':' in stripped:
            start = i
            break

    if start is None:
        return text

    # Count braces to find the matching closing } of the field value.
    # Uses a state machine to ignore braces inside JSON string literals.
    depth = 0
    end = None
    found_open = False
    in_string = False
    escaped = False
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith('//'):
            continue
        for ch in stripped:
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    depth += 1
                    found_open = True
                elif ch == '}':
                    depth -= 1
                    if found_open and depth == 0:
                        end = i
                        break
        if end is not None:
            break

    if end is None:
        return text  # Could not parse, return unchanged

    # Check if the field's closing line has a trailing comma (field is not last)
    has_trailing_comma = lines[end].rstrip().endswith('},')

    # Remove lines [start, end] inclusive
    lines = lines[:start] + lines[end + 1:]

    if not has_trailing_comma:
        # Field was the last one — remove trailing comma from the previous content line
        for i in range(start - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped and not stripped.startswith('//'):
                if stripped.endswith(','):
                    lines[i] = lines[i].rstrip()[:-1]
                break

    return '\n'.join(lines)


def _inject_json_field(text: str, field_name: str, value: dict) -> str:
    """Inject a top-level JSON field before the closing brace, preserving comments."""
    lines = text.split('\n')

    # Find the last line that is just '}'
    closing_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == '}':
            closing_idx = i
            break

    if closing_idx is None:
        raise ValueError("Could not find closing brace in workflow JSON")

    # Find last non-empty, non-comment line before closing brace and add comma
    for i in range(closing_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith('//'):
            if not stripped.endswith(','):
                lines[i] = lines[i].rstrip() + ','
            break

    # Format value with proper indentation
    value_json = json.dumps(value, indent=4)
    value_lines = value_json.split('\n')
    formatted = ['    "' + field_name + '": ' + value_lines[0]]
    for vl in value_lines[1:]:
        formatted.append('    ' + vl)

    # Insert before closing brace
    result = lines[:closing_idx] + formatted + lines[closing_idx:]
    return '\n'.join(result)


def _write_workflow_with_overrides(
    input_path: str,
    output_path: str,
    frame_overrides: dict[int, dict[str, Any]],
    formula_overrides: dict[int, dict[str, Any]],
):
    """Write a copy of the input workflow JSON with override fields added/updated."""
    from utils.json_spec import serialize_override_value

    # Serialize values and convert int keys to sorted string keys
    ser_frame = {
        str(k): {fn: serialize_override_value(v) for fn, v in fields.items()}
        for k, fields in sorted(frame_overrides.items())
    } if frame_overrides else {}
    ser_formula = {
        str(k): {fn: serialize_override_value(v) for fn, v in fields.items()}
        for k, fields in sorted(formula_overrides.items())
    } if formula_overrides else {}

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove existing override fields (no-op if absent)
    text = _remove_json_field(text, 'frame_overrides')
    text = _remove_json_field(text, 'formula_overrides')

    # Inject non-empty override fields
    if ser_formula:
        text = _inject_json_field(text, 'formula_overrides', ser_formula)
    if ser_frame:
        text = _inject_json_field(text, 'frame_overrides', ser_frame)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


class LoopStatus(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"


class LoopState:
    """Thread-safe shared state between the GPU loop thread and Flask server."""

    def __init__(self, total_iterations: int, output_folder: str, no_input_image: bool = False):
        self._lock = threading.Lock()
        self._status = LoopStatus.RUNNING
        self._current_iteration = 0
        self._total_iterations = total_iterations
        self._output_folder = output_folder
        self._latest_image_index = 0
        self._no_input_image = no_input_image
        self._elaborated_settings: dict[int, str] = {}
        self._pause_event = threading.Event()
        self._pause_event.set()  # starts unpaused
        self._restart_request: Optional[int] = None
        self._error: Optional[str] = None
        self._export_status: Optional[str] = None
        self._export_error: Optional[str] = None
        self._export_file: Optional[str] = None
        self._export_generation: int = 0
        self._warning: Optional[str] = None
        self._settings_manager: Optional[SettingsManager] = None
        self._iteration_timestamps: list[float] = []
        self._iter_start_time: Optional[float] = None
        self._stop_event = threading.Event()  # starts unset (not stopped)

        # Pre-elaborated settings and persistent override tracking
        self._pre_elaborated: dict[int, LoopSettings] = {}
        self._overridden_fields: dict[int, dict[str, Any]] = {}
        self._formula_overrides: dict[int, dict[str, Any]] = {}
        self._json_file: Optional[str] = None
        self._animation_params: dict[str, str] = {}

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

    def has_input_image(self) -> bool:
        return not self._no_input_image

    # --- Settings cache (View tab JSON strings) ---

    def store_settings(self, iter_num: int, settings_json: str):
        with self._lock:
            self._elaborated_settings[iter_num] = settings_json

    def get_settings(self, iter_num: int) -> Optional[str]:
        with self._lock:
            cached = self._elaborated_settings.get(iter_num)
            if cached is not None:
                return cached
            # Fall back to pre-elaborated settings (e.g. after override cleared the cache)
            ls = self._pre_elaborated.get(iter_num)
            if ls is not None:
                return ls.to_json(indent=4)
            return None

    def clear_settings_from(self, iter_num: int):
        with self._lock:
            keys_to_remove = [k for k in self._elaborated_settings if k >= iter_num]
            for k in keys_to_remove:
                del self._elaborated_settings[k]

    # --- Pause/resume ---

    def pause(self):
        with self._lock:
            self._pause_event.clear()
            self._status = LoopStatus.PAUSED

    def resume(self):
        with self._lock:
            if self._status == LoopStatus.PAUSED:
                self._status = LoopStatus.RUNNING
        self._pause_event.set()

    def wait_if_paused(self):
        self._pause_event.wait()

    # --- Restart ---

    def request_restart(self, from_image_index: int):
        """Set a restart request and ensure the loop thread is unblocked to process it."""
        with self._lock:
            self._restart_request = from_image_index
            # Move out of PAUSED so the loop thread proceeds after waking.
            if self._status == LoopStatus.PAUSED:
                self._status = LoopStatus.RUNNING
        # Unblock the pause event so the loop thread can pick up the request.
        self._pause_event.set()

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

    # --- Warning (transient, non-fatal) ---

    def set_warning(self, warning: Optional[str]):
        with self._lock:
            self._warning = warning

    def get_warning(self) -> Optional[str]:
        with self._lock:
            return self._warning

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

    def start_export(self, filepath: str) -> int:
        """Prepare export state and return the generation token."""
        with self._lock:
            self._export_generation += 1
            self._export_file = filepath
            self._export_error = None
            self._export_status = 'running'
            return self._export_generation

    def finish_export(self, generation: int, status: str, error: str = None):
        """Set export result only if generation still matches (not cancelled)."""
        with self._lock:
            if self._export_generation != generation:
                return
            self._export_status = status
            self._export_error = error

    def cancel_export(self) -> Optional[str]:
        """Cancel in-progress export. Returns the file path for cleanup."""
        with self._lock:
            self._export_generation += 1
            filepath = self._export_file
            self._export_status = 'cancelled'
            self._export_error = None
            self._export_file = None
            return filepath

    # --- Pre-elaborated settings & persistent overrides ---

    def init_pre_elaborated(self, sm: SettingsManager, json_file: str, animation_params: dict[str, str]):
        """Pre-elaborate all frame settings from the SettingsManager at loop start."""
        with self._lock:
            self._json_file = json_file
            self._animation_params = dict(animation_params)
            self._pre_elaborated = {}
            self._overridden_fields = {}
            self._formula_overrides = {}

            # Apply formula overrides from workflow before elaboration
            # (Workflow.__post_init__ already normalized keys to int and values to typed)
            if sm.workflow.formula_overrides:
                for section_idx, field_overrides in sm.workflow.formula_overrides.items():
                    if section_idx < len(sm.workflow.all_settings):
                        ls = sm.workflow.all_settings[section_idx]
                        for field_name, value in field_overrides.items():
                            setattr(ls, field_name, value)
                        self._formula_overrides[section_idx] = dict(field_overrides)

            # Elaborate all frames
            total = sm.get_total_iterations()
            for i in range(total):
                self._pre_elaborated[i] = sm.get_elaborated_loopsettings_for_iter(i)

            # Apply frame overrides from workflow after elaboration
            if sm.workflow.frame_overrides:
                for iter_num, field_overrides in sm.workflow.frame_overrides.items():
                    if iter_num in self._pre_elaborated:
                        ls = self._pre_elaborated[iter_num]
                        for field_name, value in field_overrides.items():
                            setattr(ls, field_name, value)
                        self._overridden_fields[iter_num] = dict(field_overrides)

    def get_pre_elaborated(self, iter: int) -> LoopSettings:
        """Return a deep copy of the pre-elaborated settings for a given iteration."""
        with self._lock:
            if iter not in self._pre_elaborated:
                raise KeyError(f"No pre-elaborated settings for iteration {iter}")
            return deepcopy(self._pre_elaborated[iter])

    def apply_frame_override(self, iter: int, overrides: dict[str, Any]):
        """Apply persistent frame overrides to a specific iteration's pre-elaborated settings."""
        with self._lock:
            if iter not in self._pre_elaborated:
                raise KeyError(f"No pre-elaborated settings for iteration {iter}")
            ls = self._pre_elaborated[iter]
            if iter not in self._overridden_fields:
                self._overridden_fields[iter] = {}
            for field_name, value in overrides.items():
                setattr(ls, field_name, value)
                self._overridden_fields[iter][field_name] = value
            # Clear stale cache so get_settings() falls back to updated _pre_elaborated
            self._elaborated_settings.pop(iter, None)

    def remove_frame_override(self, iter: int, sm: SettingsManager):
        """Remove all persistent frame overrides for a specific iteration and re-elaborate."""
        with self._lock:
            if iter not in self._overridden_fields:
                return
            del self._overridden_fields[iter]
            self._pre_elaborated[iter] = sm.get_elaborated_loopsettings_for_iter(iter)
            self._elaborated_settings.pop(iter, None)

    def remove_formula_override(self, section_idx: int, sm_fresh: SettingsManager):
        """Remove a formula override for a section and re-elaborate affected frames."""
        with self._lock:
            if section_idx not in self._formula_overrides:
                return
            del self._formula_overrides[section_idx]
            # Re-apply remaining formula overrides to the fresh SM's all_settings
            for idx, field_overrides in self._formula_overrides.items():
                if idx < len(sm_fresh.workflow.all_settings):
                    ls = sm_fresh.workflow.all_settings[idx]
                    for field_name, value in field_overrides.items():
                        setattr(ls, field_name, value)
            # Compute the first iteration of the removed section
            first_iter = 0
            for i in range(section_idx):
                if i < len(sm_fresh.workflow.all_settings):
                    first_iter += sm_fresh.workflow.all_settings[i].loop_iterations
            # Re-elaborate from that point onward
            total = sm_fresh.get_total_iterations()
            for i in range(first_iter, total):
                self._pre_elaborated[i] = sm_fresh.get_elaborated_loopsettings_for_iter(i)
                # Re-apply any per-frame overrides
                if i in self._overridden_fields:
                    ls = self._pre_elaborated[i]
                    for field_name, value in self._overridden_fields[i].items():
                        setattr(ls, field_name, value)
            # Clear View tab cache for affected frames
            keys_to_remove = [k for k in self._elaborated_settings if k >= first_iter]
            for k in keys_to_remove:
                del self._elaborated_settings[k]

    def re_elaborate_from(self, iter: int, sm: SettingsManager):
        """Re-elaborate frames from `iter` onward after a formula override,
        preserving any per-frame overrides."""
        with self._lock:
            total = sm.get_total_iterations()
            for i in range(iter, total):
                self._pre_elaborated[i] = sm.get_elaborated_loopsettings_for_iter(i)
                # Re-apply any frame-level overrides
                if i in self._overridden_fields:
                    ls = self._pre_elaborated[i]
                    for field_name, value in self._overridden_fields[i].items():
                        setattr(ls, field_name, value)
            # Clear the View tab cache for affected frames
            keys_to_remove = [k for k in self._elaborated_settings if k >= iter]
            for k in keys_to_remove:
                del self._elaborated_settings[k]

    def reset_all_overrides(self, sm: SettingsManager):
        """Reset all overrides by re-elaborating everything from a clean SettingsManager."""
        with self._lock:
            self._overridden_fields = {}
            self._formula_overrides = {}
            total = sm.get_total_iterations()
            for i in range(total):
                self._pre_elaborated[i] = sm.get_elaborated_loopsettings_for_iter(i)
            self._elaborated_settings = {}

    def get_all_overridden_frames(self) -> dict[int, list[str]]:
        """Return {iteration: [field_names]} for all frames with persistent overrides."""
        with self._lock:
            return {k: list(v.keys()) for k, v in self._overridden_fields.items() if v}

    def record_formula_override(self, section_idx: int, overrides: dict[str, Any]):
        """Record a formula override on a section for persistence tracking."""
        with self._lock:
            if section_idx not in self._formula_overrides:
                self._formula_overrides[section_idx] = {}
            self._formula_overrides[section_idx].update(overrides)

    def get_all_overridden_sections(self) -> dict[int, list[str]]:
        """Return {section_idx: [field_names]} for all sections with formula overrides."""
        with self._lock:
            return {k: list(v.keys()) for k, v in self._formula_overrides.items() if v}

    def persist_overrides(self):
        """Write the current overrides to a JSON file in the output folder."""
        with self._lock:
            frame_ov = {k: dict(v) for k, v in self._overridden_fields.items() if v}
            formula_ov = {k: dict(v) for k, v in self._formula_overrides.items() if v}
            json_file = self._json_file
            output_folder = self._output_folder

        if not json_file:
            return

        if not frame_ov and not formula_ov:
            self.restore_clean_json()
            return

        output_path = os.path.join(output_folder, os.path.basename(json_file))
        _write_workflow_with_overrides(json_file, output_path, frame_ov, formula_ov)

    def restore_clean_json(self):
        """Restore the output folder JSON to a clean state (no override fields).

        For a fresh start (json_file points to data/ folder), re-copies from the original.
        For a resume (json_file IS the output folder copy), strips override fields in-place.
        """
        with self._lock:
            json_file = self._json_file
            output_folder = self._output_folder
        if not json_file:
            return
        output_path = os.path.join(output_folder, os.path.basename(json_file))

        if os.path.abspath(json_file) != os.path.abspath(output_path):
            # Fresh start: re-copy from original source
            shutil.copy2(json_file, output_path)
        else:
            # Resume: strip override fields from the file in-place
            with open(output_path, 'r', encoding='utf-8') as f:
                text = f.read()
            text = _remove_json_field(text, 'frame_overrides')
            text = _remove_json_field(text, 'formula_overrides')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)

    def is_field_overridden(self, iter: int, field: str) -> bool:
        """Check if a specific field was overridden by the user for a given iteration."""
        with self._lock:
            return iter in self._overridden_fields and field in self._overridden_fields[iter]

    def get_json_file(self) -> Optional[str]:
        with self._lock:
            return self._json_file

    def get_animation_params(self) -> dict[str, str]:
        with self._lock:
            return dict(self._animation_params)

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
