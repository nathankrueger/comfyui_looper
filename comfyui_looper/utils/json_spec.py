import os
import random
from typing import Any, Optional
from dataclasses import dataclass, field, fields
from dataclasses_json import dataclass_json

import folder_paths
from image_processing.transforms import Transform
from utils.util import MathParser

CURRENT_WORKFLOW_VERSION = 1

EMPTY_LIST = [None]
def empty_list_factory() -> list:
    return list(EMPTY_LIST)

def default_seed() -> int:
    return random.randint(1, 2**64)

@dataclass_json
@dataclass
class LoopSettings:
    loop_iterations: int = 1

    # all of these have empty defaults -- they will grab the value from the previous LoopSettings, if available
    checkpoint: Optional[str] = None
    prompt: Optional[str] = None
    denoise_steps: Optional[int | str] = None
    denoise_amt: Optional[float | str] = None
    clip: list[str | None] = field(default_factory=empty_list_factory)
    cfg: Optional[float | str] = None
    seed: int = field(default_factory=default_seed)
    canny: list[float | None] = field(default_factory=empty_list_factory)
    loras: list[tuple[str, float] | None] = field(default_factory=empty_list_factory)
    transforms: list[dict[str, Any] | None] = field(default_factory=empty_list_factory)

    def __post_init__(self):
        self.offset = None

@dataclass_json
@dataclass
class Workflow:
    all_settings: list[LoopSettings]
    version: int = CURRENT_WORKFLOW_VERSION

    def get_total_iterations(self) -> int:
        cnt = 0
        for setting in self.all_settings:
            cnt += setting.loop_iterations
        return cnt

class SettingsManager:
    EXPR_VARIABLES = {
        'denoise_amt',
        'denoise_steps',
        'cfg'
    }

    def __init__(self, workflow_json_path: str):
        with open(workflow_json_path, "r", encoding="utf-8") as json_file:
            json_no_comments = os.linesep.join([line for line in json_file.readlines() if not line.strip().startswith("//")])
            self.workflow: Workflow = Workflow.schema().loads(json_no_comments)

        assert self.workflow.version == CURRENT_WORKFLOW_VERSION
        self.iter_to_setting_map: dict[int, tuple[int, LoopSettings]] = {}
        self.prev_setting_map: dict[str, Any] = {}
        
        # set offsets
        running_offset = 0
        for ls in self.workflow.all_settings:
            ls.offset = running_offset
            running_offset += ls.loop_iterations

        self.total_iterations: int = self.workflow.get_total_iterations()

    def validate(self):
        # transforms
        transform_params = [tdict for loopsettings in self.workflow.all_settings for tdict in loopsettings.transforms if tdict is not None]
        Transform.validate_transformation_params(transform_params)

        # canny
        for ls in self.workflow.all_settings:
            assert ls.canny == EMPTY_LIST or len(ls.canny) == 3 or len(ls.canny) == 0

        # lora & model files
        for ls in self.workflow.all_settings:
            if ls.loras != EMPTY_LIST:
                for lorafile in ls.loras:
                    assert os.path.exists(os.path.join(folder_paths.get_folder_paths("loras")[0], lorafile[0]))
            if ls.checkpoint is not None:
                ckpt_found = False
                for folder_query in {"checkpoints", "diffusion_models"}:
                    for specific_folder in folder_paths.get_folder_paths(folder_query):
                        path = os.path.join(specific_folder, ls.checkpoint)
                        ckpt_found |= os.path.exists(path)
                assert ckpt_found

    def update_seed(self, iter: int, seed: int):
        self.get_loopsettings_for_iter(iter)[1].seed = seed

    @staticmethod
    def should_infer_setting(setting_val: Any) -> bool:
        return setting_val is None or (isinstance(setting_val, list) and setting_val == EMPTY_LIST)

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
        return self.total_iterations

    def eval_expr(self, iter: int, setting_name: str, expr: str, loopsetting: LoopSettings) -> float:
        """
        n --> total iteration sequence number
        offset --> current LoopSettings sequence number
        """

        if setting_name in SettingsManager.EXPR_VARIABLES and isinstance(expr, str):
            return MathParser({'n':iter, 'offset':loopsetting.offset, 'total_n': self.get_total_iterations()})(expr)
        else:
            return expr

    def get_setting_for_iter(self, setting_name: str, iter: int) -> Any:
        setting_idx, loopsetting = self.get_loopsettings_for_iter(iter)
        setting_val = loopsetting.__getattribute__(setting_name)

        if SettingsManager.should_infer_setting(setting_val):
            setting_idx -= 1
            while setting_idx >= 0:
                prev_loopsetting = self.workflow.all_settings[setting_idx]
                prev_setting_val = prev_loopsetting.__getattribute__(setting_name)
                if prev_setting_val is not None and prev_setting_val != EMPTY_LIST:
                    return self.eval_expr(iter, setting_name, prev_setting_val, prev_loopsetting)
                setting_idx -= 1
            return [] if setting_val == EMPTY_LIST else None
        else:
            return self.eval_expr(iter, setting_name, setting_val, loopsetting)

    def get_elaborated_loopsettings_for_iter(self, iter:int) -> LoopSettings:
        _, loopsetting = self.get_loopsettings_for_iter(iter)
        result = LoopSettings()
        for dc_field in fields(loopsetting):
            result.__setattr__(dc_field.name, self.get_setting_for_iter(dc_field.name, iter))
        result.loop_iterations = 1

        return result
