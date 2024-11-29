import torch

from workflow.looper_workflow import WorkflowEngine
from utils.json_spec import LoopSettings
from utils.util import resize_image_match_area

# ComfyUI imports
from utils.node_wrappers import (
    Flux1DModelManager,
    LoraManager,
)
from nodes import (
    CLIPTextEncode,
    VAEDecode,
    VAEEncode,
    NODE_CLASS_MAPPINGS,
)

# constants
FLUX1D_WIDTH=1024
FLUX1D_LATENT_REDUCTION_FACTOR=8
FLUX_SCHEDULER="beta"
FLUX_SAMPLER="euler"
FLUX_VAE="ae.safetensors"
FLUX_CLIP1="t5xxl_fp8_e4m3fn.safetensors"  # TODO: these should be part of json spec
FLUX_CLIP2="clip_l.safetensors"
FLUX_GUIDANCE=3.5

class Flux1DWorkflowEngine(WorkflowEngine):
    NAME = "flux1d"

    def __init__(self):
        self.vaeencode = None
        self.vaedecode = None
        self.cliptextencode = None
        self.fluxguidance = None
        self.basicguider = None
        self.basicscheduler = None
        self.samplercustomadvanced = None
        self.randomnoise = None
        self.sampler_select_val = None
        self.lora_mgr = None
        self.model_mgr = None

    def resize_images_for_model(self, input_path: str, output_paths: list[str]):
        for output_path in output_paths:
            resize_image_match_area(input_path, output_path, FLUX1D_WIDTH**2, FLUX1D_LATENT_REDUCTION_FACTOR)

    def setup(self):
        # comfy nodes
        self.vaeencode = VAEEncode()
        self.vaedecode = VAEDecode()
        self.cliptextencode = CLIPTextEncode()
        self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        self.basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        self.basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        self.samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        self.randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        self.sampler_select_val = NODE_CLASS_MAPPINGS["KSamplerSelect"]().get_sampler(sampler_name=FLUX_SAMPLER)[0]

        # wrappers
        self.lora_mgr = LoraManager()
        self.model_mgr = Flux1DModelManager()

    def compute_iteration(self, image_tensor: torch.Tensor, loopsettings: LoopSettings) -> torch.Tensor:
        positive_text = loopsettings.prompt
        steps = loopsettings.denoise_steps
        denoise = loopsettings.denoise_amt
        lora_list = loopsettings.loras
        checkpoint = loopsettings.checkpoint
        seed = loopsettings.seed

        # load in new model if changed
        unet_model, clip_model, vae_model = self.model_mgr.reload_if_needed(checkpoint, FLUX_VAE, FLUX_CLIP1, FLUX_CLIP2)

        # only load in new loras as needd
        lora_model, lora_clip = self.lora_mgr.reload_if_needed(lora_list, unet_model, clip_model)

        # VAE encode
        vaeencode_result, = self.vaeencode.encode(
            pixels=image_tensor,
            vae=vae_model
        )

        # text conditioning
        conditioning, = self.cliptextencode.encode(
            text=positive_text,
            clip=lora_clip,
        )
        fluxguidance_result, = self.fluxguidance.append(
            guidance=FLUX_GUIDANCE, conditioning=conditioning
        )
        guider_result, = self.basicguider.get_guider(
            model=lora_model,
            conditioning=fluxguidance_result
        )

        # latent sampler
        noise_result, = self.randomnoise.get_noise(noise_seed=seed)
        sigma_result, = self.basicscheduler.get_sigmas(
            scheduler=FLUX_SCHEDULER,
            steps=steps,
            denoise=denoise,
            model=lora_model,
        )
        sampler_output, denoised_output = self.samplercustomadvanced.sample(
            noise=noise_result,
            guider=guider_result,
            sampler=self.sampler_select_val,
            sigmas=sigma_result,
            latent_image=vaeencode_result
        )

        # VAE decode
        vaedecode_result, = self.vaedecode.decode(
            samples=sampler_output,
            vae=vae_model
        )

        return vaedecode_result
    