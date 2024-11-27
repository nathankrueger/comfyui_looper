import os
from typing import Any, Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class LoopSettings:
    loop_iterations: int

    # all of these have empty defaults -- they will grab the value from the previous LoopSettings, if available
    checkpoint: Optional[str] = None
    prompt: Optional[str] = None
    denoise_steps: Optional[int] = None
    denoise_amt: Optional[float] = None
    canny: Optional[tuple[float, float, float]] = None
    loras: list[tuple[str, float]] = field(default_factory=list)
    transforms: list[dict[str, Any]] = field(default_factory=list)

@dataclass_json
@dataclass
class Workflow:
    all_settings: list[LoopSettings]
    version: int

    def get_total_iterations(self) -> int:
        cnt = 0
        for setting in self.all_settings:
            cnt += setting.loop_iterations
        return cnt

class SettingsManager:
    SUPPORTED_VERSION = 1

    def __init__(self, workflow_json_path: str):
        with open(workflow_json_path, "r", encoding="utf-8") as json_file:
            json_no_comments = os.linesep.join([line for line in json_file.readlines() if not line.strip().startswith("//")])
            self.workflow: Workflow = Workflow.schema().loads(json_no_comments)

        assert self.workflow.version == SettingsManager.SUPPORTED_VERSION
        self.iter_to_setting_map: dict[int, tuple[int, LoopSettings]] = {}
        self.prev_setting_map: dict[str, Any] = {}

    def get_loopsettings_for_iter(self, iter: int) -> tuple[int, LoopSettings]:
        if iter in self.iter_to_setting_map:
            return self.iter_to_setting_map[iter]

        offset = 0
        for idx, loopsetting in enumerate(self.workflow.all_settings):
            if iter < (offset + loopsetting.loop_iterations):
                self.iter_to_setting_map[iter] = (idx, loopsetting)
                break
            else:
                offset += loopsetting.loop_iterations

        if iter in self.iter_to_setting_map:
            return self.iter_to_setting_map[iter]
        else:
            raise Exception(f"Invalid iteration: {iter}")

    def get_total_iterations(self) -> int:
        return self.workflow.get_total_iterations()

    def get_setting_for_iter(self, setting_name: str, iter: int) -> Any:
        setting_idx, loopsetting = self.get_loopsettings_for_iter(iter)
        setting_val = loopsetting.__getattribute__(setting_name)

        if setting_val is None:
            setting_idx -= 1
            while setting_idx >= 0:
                prev_loopsetting = self.workflow.all_settings[setting_idx]
                prev_setting_val = prev_loopsetting.__getattribute__(setting_name)
                if prev_setting_val is not None:
                    return prev_setting_val
                setting_idx -= 1
            return None
        else:
            return setting_val
