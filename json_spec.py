import json
from typing import Any
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class LoopSettings:
    loop_iterations: int

    # all of these have empty defaults -- they will grab the value from the previous LoopSettings, if available
    checkpoint: str = None
    prompt: str = None
    denoise_steps: int = None
    denoise_amt: float = None
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
            self.workflow: Workflow = Workflow.schema().loads(json_file.read())

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

test = LoopSettings(
    loop_iterations=20,
    checkpoint='foo.safetensors',
    denoise_steps=30,
    prompt='a beautfiul photo, one with commas',
    loras=[('lora_a.safetensors', 1.0), ('lora_b.safetensors', .212354)],
    transforms=[{'name':'zoom_in', 'zoom_amt':0.985}],
    denoise_amt=0.5
)

json_text = test.to_json(indent=4)
print(json_text)
test2: LoopSettings = LoopSettings.schema().loads(json_text)
assert test == test2
test2.loop_iterations = 5
test2.denoise_steps = 123

test = Workflow(
    all_settings=[test, test2],
    version=1
)
json_text = test.to_json(indent=4)
print(json_text)
test2 = Workflow.schema().loads(json_text)
assert test == test2

with open('test.json', 'w', encoding='utf-8') as f:
    f.write(json_text)

sm = SettingsManager('test.json')
print(sm.get_total_iterations())
print(sm.get_setting_for_iter('denoise_steps', 5))
print(sm.get_setting_for_iter('denoise_steps', 24))