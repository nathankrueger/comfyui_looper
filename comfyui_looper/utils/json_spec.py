import os
import random
from typing import Any, Optional
from dataclasses import dataclass, field, fields
from dataclasses_json import dataclass_json
from copy import deepcopy

import folder_paths
from image_processing.transforms import Transform
from image_processing.animator import get_animation_param_value
from utils.simple_expr_eval import SimpleExprEval
from utils.fft import WaveFile

CURRENT_WORKFLOW_VERSION = 1

class JSONFormatException(Exception):
    pass

EMPTY_DICT = {'null':'null'}
def empty_dict_factory() -> dict:
    return dict(EMPTY_DICT)

EMPTY_LIST = [None]
def empty_list_factory() -> list:
    return list(EMPTY_LIST)

def default_seed() -> int:
    return random.randint(1, 2**64)

@dataclass
class EmptyObject:
    pass

EMPTY_OBJECT = EmptyObject()
def empty_object_factory():
    return EMPTY_OBJECT

@dataclass_json
@dataclass
class ConDelta:
    pos: str
    neg: str
    strength: float | str

    def validate(self):
        assert len(self.neg) > 0
        assert len(self.pos) > 0

@dataclass_json
@dataclass(unsafe_hash=True)
class LoraFilter:
    lora_path: str
    lora_strength: float

    def validate(self):
        assert os.path.exists(os.path.join(folder_paths.get_folder_paths("loras")[0], self.lora_path))
        assert self.lora_strength >= 0

@dataclass_json
@dataclass
class Canny:
    low_thresh: float
    high_thresh: float
    strength: float

    def validate(self):
        assert self.low_thresh >= 0
        assert self.high_thresh >= 0
        assert self.strength >= 0

@dataclass_json
@dataclass
class LoopSettings:
    loop_iterations: int = 1

    # all of these have empty defaults -- they will grab the value from the previous LoopSettings, if available
    seed: int = field(default_factory=default_seed)
    checkpoint: Optional[str | EmptyObject] = field(default_factory=empty_object_factory)
    prompt: Optional[str | EmptyObject] = field(default_factory=empty_object_factory)
    neg_prompt: Optional[str | EmptyObject] = field(default_factory=empty_object_factory)
    denoise_steps: Optional[int | str | EmptyObject] = field(default_factory=empty_object_factory)
    denoise_amt: Optional[float | str | EmptyObject] = field(default_factory=empty_object_factory)
    cfg: Optional[float | str | EmptyObject] = field(default_factory=empty_object_factory)
    canny: Optional[Canny | EmptyObject] = field(default_factory=empty_object_factory)
    clip: list[str | None] = field(default_factory=empty_list_factory)
    con_deltas: list[ConDelta | None]  = field(default_factory=empty_list_factory)
    loras: list[LoraFilter | None] = field(default_factory=empty_list_factory)
    transforms: list[dict[str, Any] | None] = field(default_factory=empty_list_factory)

    def __post_init__(self):
        self.offset = None

    def validate(self):
        # transforms
        transform_params = [tdict for tdict in self.transforms if tdict is not None]
        Transform.validate_transformation_params(transform_params)

        # canny
        if isinstance(self.canny, Canny):
            self.canny.validate()

        # lora
        if self.loras != EMPTY_LIST:
            for lora in self.loras:
                assert isinstance(lora, LoraFilter)
                lora.validate()

        # model files
        if self.checkpoint is not None and self.checkpoint is not EMPTY_OBJECT:
            ckpt_found = False
            for folder_query in {"checkpoints", "diffusion_models"}:
                for specific_folder in folder_paths.get_folder_paths(folder_query):
                    path = os.path.join(specific_folder, self.checkpoint)
                    ckpt_found |= os.path.exists(path)
            assert ckpt_found

        # con delta
        if self.con_deltas != EMPTY_LIST:
            for cd in self.con_deltas:
                cd.validate()

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
    EXPR_LS_VARIABLES = {
        'denoise_amt',
        'denoise_steps',
        'con_deltas',
        'loras',
        'canny',
        'cfg',
    }

    def __init__(self, workflow_json_path: str, animation_params: dict[str, str]):
        with open(workflow_json_path, "r", encoding="utf-8") as json_file:
            json_no_comments = os.linesep.join([line for line in json_file.readlines() if not line.strip().startswith("//")])
            self.workflow: Workflow = Workflow.schema().loads(json_no_comments)

        assert self.workflow.version == CURRENT_WORKFLOW_VERSION
        self.iter_to_setting_map: dict[int, tuple[int, LoopSettings]] = {}
        
        # set offsets
        running_offset = 0
        for ls in self.workflow.all_settings:
            ls.offset = running_offset
            running_offset += ls.loop_iterations

        self.total_iterations: int = self.workflow.get_total_iterations()
        self.animation_params = animation_params

        # audio handling
        if 'mp3_file' in self.animation_params:
            animation_length = self.get_total_iterations() * (float(get_animation_param_value('frame_delay', self.animation_params)) / 1000.0)
            self.wavefile = WaveFile.get_wavefile(self.animation_params['mp3_file'], animation_length)
        else:
            self.wavefile = None

    def get_wavefile(self):
        return self.wavefile

    def validate(self):
        for ls in self.workflow.all_settings:
            ls.validate()

        # by prepopulating all elaborated loopsettings, we are evaluating
        # all expressions up-front, avoiding costly issues deep into a run.
        for i in range(self.get_total_iterations()):
            self.get_elaborated_loopsettings_for_iter(i)

    def log_elaborated_settings(self, log_path: str):
        ls_texts = []
        for i in range(self.get_total_iterations()):
            ls = self.get_elaborated_loopsettings_for_iter(i)
            ls_texts.append(f'Iteration: {i:04}\n' + ls.to_json(indent=4))
        
        with open(log_path, 'w') as lf:
            lf.write('\n\n'.join(ls_texts))

    def update_seed(self, iter: int, seed: int):
        self.get_loopsettings_for_iter(iter)[1].seed = seed

    @staticmethod
    def should_infer_setting(setting_val: Any) -> bool:
        if setting_val == EMPTY_OBJECT:
            return True
        elif (isinstance(setting_val, list) and setting_val == EMPTY_LIST):
            return True
        elif (isinstance(setting_val, dict) and setting_val == EMPTY_DICT):
            return True
        else:
            return False

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

    def eval_expressions(self, iter: int, setting_name: str, setting_val: Any, loopsetting: LoopSettings) -> float:
        """
        n       --> total iteration sequence number
        offset  --> current LoopSettings sequence number
        total_n --> length of entire workflow
        """

        def get_power_at_freq_range(low_f: int, high_f: int):
            if self.wavefile is not None:
                start_percent = (iter / self.total_iterations) * 100.0
                end_percent = ((iter + 1) / self.total_iterations) * 100.0
                pow_at_frange = self.wavefile.get_power_in_freq_ranges(start_percent, end_percent, [(low_f, high_f)])[0]
                return pow_at_frange
            else:
                print("Warning no audio file specified, nothing to do for 'get_power_at_freq_range()'")
                return 0

        if setting_name in SettingsManager.EXPR_LS_VARIABLES:
            iter_vars = {'n':iter, 'offset':loopsetting.offset, 'total_n':self.get_total_iterations()}
            expr_eval = SimpleExprEval(local_vars=iter_vars, permitted_fns={"get_power_at_freq_range":get_power_at_freq_range})
            if isinstance(setting_val, str):
                return expr_eval(setting_val)
            elif isinstance(setting_val, list):
                result = []
                for item in setting_val:
                    if isinstance(item, ConDelta) and isinstance(item.strength, str):
                        strength_eval = expr_eval(item.strength)
                        cd_eval = ConDelta(pos=item.pos, neg=item.neg, strength=strength_eval)
                        result.append(cd_eval)
                    elif isinstance(item, Canny):
                        for canny_field in fields(item):
                            attr_val = item.__getattribute__(canny_field.name)
                            if isinstance(attr_val, str):
                                eval_result = expr_eval(attr_val)
                                item.__setattr__(canny_field.name, eval_result)
                        result.append(item)
                    elif isinstance(item, LoraFilter) and isinstance(item.lora_strength, str):
                        strength_eval = expr_eval(item.lora_strength)
                        lora_eval = LoraFilter(lora_path=item.lora_path, lora_strength=strength_eval)
                        result.append(lora_eval)
                    else:
                        result.append(item)
                return result

        # pass through
        return setting_val

    def get_setting_for_iter(self, setting_name: str, iter: int) -> Any:
        """
        Search backwards for the last valid setting from a previous LoopSettings
        """

        setting_idx, loopsetting = self.get_loopsettings_for_iter(iter)
        setting_val = loopsetting.__getattribute__(setting_name)

        if SettingsManager.should_infer_setting(setting_val):
            setting_idx -= 1
            while setting_idx >= 0:
                prev_loopsetting = self.workflow.all_settings[setting_idx]
                prev_setting_val = prev_loopsetting.__getattribute__(setting_name)
                if prev_setting_val is None:
                    # setting explicity disabled
                    return None
                elif prev_setting_val != EMPTY_LIST and prev_setting_val != EMPTY_DICT and prev_setting_val != EMPTY_OBJECT:
                    # infer & evaluate the setting value for the current iteration
                    # use the correct offset for the current loopsetting
                    prev_loopsetting = deepcopy(prev_loopsetting)
                    prev_loopsetting.offset = loopsetting.offset
                    return self.eval_expressions(iter, setting_name, prev_setting_val, prev_loopsetting)
                setting_idx -= 1
            if setting_val == EMPTY_LIST:
                return []
            elif setting_val == EMPTY_DICT:
                return {}
            elif setting_val == EMPTY_OBJECT:
                return None
            else:
                return None
        else:
            return self.eval_expressions(iter, setting_name, setting_val, loopsetting)

    def get_elaborated_loopsettings_for_iter(self, iter:int) -> LoopSettings:
        """
        Used by workflows which use a fully populated LoopSettings as their input
        """

        _, loopsetting = self.get_loopsettings_for_iter(iter)
        result = LoopSettings()
        
        # assign the non-dataclass fields
        result.offset = loopsetting.offset

        # grab dataclass fields
        for dc_field in fields(loopsetting):
            result.__setattr__(dc_field.name, self.get_setting_for_iter(dc_field.name, iter))

        # minor detail, but from the perspective of a given image or compute iteration this is 1
        result.loop_iterations = 1

        return result
