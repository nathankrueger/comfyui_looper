import torch

from workflow.looper_workflow import WorkflowEngine
from utils.json_spec import LoopSettings
from utils.util import resize_image_match_area

# ComfyUI imports
from utils.node_wrappers import (
    LoraManager,
    SDXLCheckpointManager,
    ControlNetManager,
    ConDeltaManager,
)
from nodes import (
    VAEDecode,
    VAEEncode,
    KSampler,
    ControlNetApply,
    NODE_CLASS_MAPPINGS,
)

# constants
SDXL_AREA=1024**2
SDXL_LATENT_REDUCTION_FACTOR=8
NEGATIVE_TEXT='text, watermark, logo'
CANNY_CONTROLNET='control-lora-canny-rank256.safetensors'

class SDXLWorkflowEngine(WorkflowEngine):
    NAME = "sdxl"
    DEFAULT_SETTING_DICT = {
        "checkpoint": "sdXL_v10VAEFix.safetensors",
        "cfg": 8.0,
        "denoise_steps": 20,
        "neg_prompt": NEGATIVE_TEXT
    }

    def __init__(self):
        self.canny_node = None
        self.controlnetapply = None
        self.vaeencode = None
        self.ksampler = None
        self.vaedecode = None
        self.lora_mgr = None
        self.ckpt_mgr = None
        self.text_cond = None
        self.control_net_mgr = None

    def resize_images_for_model(self, input_path: str, output_paths: list[str]):
        for output_path in output_paths:
            resize_image_match_area(input_path, output_path, SDXL_AREA, SDXL_LATENT_REDUCTION_FACTOR)

    def setup(self):
        # comfy nodes
        self.canny_node = NODE_CLASS_MAPPINGS["Canny"]()
        self.controlnetapply = ControlNetApply()
        self.vaeencode = VAEEncode()
        self.ksampler = KSampler()
        self.vaedecode = VAEDecode()

        # wrappers
        self.lora_mgr = LoraManager()
        self.ckpt_mgr = SDXLCheckpointManager()
        self.control_net_mgr = ControlNetManager(CANNY_CONTROLNET)
        self.con_delta_mgr = ConDeltaManager()

    def compute_iteration(self, image_tensor: torch.Tensor, loopsettings: LoopSettings):
        positive_text = loopsettings.prompt
        negative_text = loopsettings.neg_prompt
        steps = loopsettings.denoise_steps
        denoise = loopsettings.denoise_amt
        cfg = loopsettings.cfg
        con_deltas = loopsettings.con_deltas
        lora_list = loopsettings.loras
        checkpoint = loopsettings.checkpoint
        canny = loopsettings.canny
        seed = loopsettings.seed

        # load in new checkpoint if changed
        ckpt_model, ckpt_clip, ckpt_vae = self.ckpt_mgr.reload_if_needed(checkpoint)

        # only load in new loras as needd
        lora_model, lora_clip = self.lora_mgr.reload_if_needed(lora_list, ckpt_model, ckpt_clip)

        # conditioning w/ con_delta
        pos_cond, neg_cond = self.con_delta_mgr.encode(
            clip=lora_clip,
            pos_text=positive_text,
            neg_text=negative_text,
            con_deltas=con_deltas
        )

        # VAE encode
        vaeencode_result, = self.vaeencode.encode(
            pixels=image_tensor,
            vae=ckpt_vae
        )

        # load in the canny if needed
        if (controlnetloader_result := self.control_net_mgr.reload_if_needed(canny)) is not None:
            canny_result, = self.canny_node.detect_edge(
                low_threshold=canny.low_thresh,
                high_threshold=canny.high_thresh,
                image=image_tensor,
            )
            pos_cond, = self.controlnetapply.apply_controlnet(
                strength=canny.strength,
                conditioning=pos_cond,
                control_net=controlnetloader_result,
                image=canny_result,
            )

        # latent sampler
        ksampler_result, = self.ksampler.sample(
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name="euler",
            scheduler="normal",
            denoise=round(denoise, 2),
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
