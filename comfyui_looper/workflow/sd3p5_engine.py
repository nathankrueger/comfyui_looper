import torch

from workflow.looper_workflow import WorkflowEngine
from utils.json_spec import LoopSettings
from utils.util import resize_image_match_area

# ComfyUI imports
from utils.node_wrappers import (
    LoraManager,
    SD3p5ModelManager,
    ClipEncodeWrapper,
)
from nodes import (
    VAEDecode,
    VAEEncode,
    KSampler,
)

# constants
SD3p5_AREA=1024**2
SD3p5_LATENT_REDUCTION_FACTOR=8
NEGATIVE_TEXT='text, watermark, logo'

class SD3p5WorkflowEngine(WorkflowEngine):
    NAME = "sd3.5"
    DEFAULT_SETTING_DICT = {
        "checkpoint": "sd3.5_large.safetensors",
        "clip": ["t5xxl_fp8_e4m3fn.safetensors", "clip_g.safetensors"],
        "cfg": 3.5
    }

    def __init__(self):
        self.vaeencode = None
        self.ksampler = None
        self.vaedecode = None
        self.lora_mgr = None
        self.ckpt_mgr = None
        self.text_cond = None

    def resize_images_for_model(self, input_path: str, output_paths: list[str]):
        for output_path in output_paths:
            resize_image_match_area(input_path, output_path, SD3p5_AREA, SD3p5_LATENT_REDUCTION_FACTOR)

    def setup(self):
        # comfy nodes
        self.vaeencode = VAEEncode()
        self.ksampler = KSampler()
        self.vaedecode = VAEDecode()

        # wrappers
        self.lora_mgr = LoraManager()
        self.ckpt_mgr = SD3p5ModelManager()
        self.text_cond = ClipEncodeWrapper()

    def compute_iteration(self, image_tensor: torch.Tensor, loopsettings: LoopSettings):
        positive_text = loopsettings.prompt
        steps = loopsettings.denoise_steps
        denoise = loopsettings.denoise_amt
        cfg = loopsettings.cfg
        lora_list = loopsettings.loras
        checkpoint = loopsettings.checkpoint
        clip = loopsettings.clip
        seed = loopsettings.seed

        # load in new checkpoint if changed
        assert len(clip) == 2
        ckpt_model, ckpt_clip, ckpt_vae = self.ckpt_mgr.reload_if_needed(checkpoint, clip[0], clip[1])

        # only load in new loras as needd
        lora_model, lora_clip = self.lora_mgr.reload_if_needed(lora_list, ckpt_model, ckpt_clip)

        # conditioning
        pos_cond, neg_cond = self.text_cond.encode(positive_text, NEGATIVE_TEXT, lora_clip)

        # VAE encode
        vaeencode_result, = self.vaeencode.encode(
            pixels=image_tensor,
            vae=ckpt_vae
        )

        # latent sampler
        ksampler_result, = self.ksampler.sample(
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name="euler",
            scheduler="beta",
            denoise=denoise,
            model=lora_model,
            positive=pos_cond,
            negative=neg_cond,
            latent_image=vaeencode_result
        )

        # VAE decode
        vaedecode_result, = self.vaedecode.decode(
            samples=ksampler_result,
            vae=ckpt_vae
        )

        return vaedecode_result
