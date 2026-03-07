"""
Build ComfyUI API-format workflow dicts from LoopSettings.

The API format is: {node_id: {"class_type": "NodeName", "inputs": {...}}}
Node references use ["node_id", output_index] tuples.
"""

from utils.json_spec import LoopSettings

# Save prefix used to identify looper outputs on the ComfyUI server
SAVE_PREFIX = "looper_output"


class _NodeGraph:
    """Helper to build a ComfyUI workflow graph with auto-incrementing node IDs."""

    def __init__(self):
        self._next_id = 1
        self.nodes = {}

    def add(self, class_type: str, inputs: dict) -> str:
        node_id = str(self._next_id)
        self._next_id += 1
        self.nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    def ref(self, node_id: str, output_index: int = 0) -> list:
        return [node_id, output_index]


def build_sdxl_workflow(input_image_name: str, settings: LoopSettings, defaults: dict) -> dict:
    """Build an SDXL img2img workflow."""
    g = _NodeGraph()

    checkpoint = settings.checkpoint or defaults.get("checkpoint")
    prompt = settings.prompt or ""
    neg_prompt = settings.neg_prompt or defaults.get("neg_prompt", "")
    steps = settings.denoise_steps or defaults.get("denoise_steps", 20)
    denoise = round(settings.denoise_amt, 2) if settings.denoise_amt is not None else 0.5
    cfg = settings.cfg or defaults.get("cfg", 8.0)
    seed = settings.seed
    loras = settings.loras or []
    canny = settings.canny
    con_deltas = settings.con_deltas or []

    # Load checkpoint
    ckpt_id = g.add("CheckpointLoaderSimple", {"ckpt_name": checkpoint})
    model_ref = g.ref(ckpt_id, 0)
    clip_ref = g.ref(ckpt_id, 1)
    vae_ref = g.ref(ckpt_id, 2)

    # Chain LoRAs
    for lora in loras:
        lora_id = g.add("LoraLoader", {
            "lora_name": lora.lora_path,
            "strength_model": lora.lora_strength,
            "strength_clip": lora.lora_strength,
            "model": model_ref,
            "clip": clip_ref,
        })
        model_ref = g.ref(lora_id, 0)
        clip_ref = g.ref(lora_id, 1)

    # SDXL CLIP encode (positive)
    pos_id = g.add("CLIPTextEncodeSDXL", {
        "width": 1024,
        "height": 1024,
        "crop_w": 0,
        "crop_h": 0,
        "target_width": 1024,
        "target_height": 1024,
        "text_g": prompt,
        "text_l": prompt,
        "clip": clip_ref,
    })
    pos_ref = g.ref(pos_id)

    # SDXL CLIP encode (negative)
    neg_id = g.add("CLIPTextEncodeSDXL", {
        "width": 1024,
        "height": 1024,
        "crop_w": 0,
        "crop_h": 0,
        "target_width": 1024,
        "target_height": 1024,
        "text_g": neg_prompt,
        "text_l": neg_prompt,
        "clip": clip_ref,
    })
    neg_ref = g.ref(neg_id)

    # ConDeltas
    for cd in con_deltas:
        cd_pos_id = g.add("CLIPTextEncode", {"text": cd.pos, "clip": clip_ref})
        cd_neg_id = g.add("CLIPTextEncode", {"text": cd.neg, "clip": clip_ref})
        delta_id = g.add("ConditioningSubtract", {
            "conditioning_a": g.ref(cd_pos_id),
            "conditioning_b": g.ref(cd_neg_id),
        })
        apply_id = g.add("ConditioningAddConDelta", {
            "conditioning_delta_strength": cd.strength,
            "conditioning_base": pos_ref,
            "conditioning_delta": g.ref(delta_id),
        })
        pos_ref = g.ref(apply_id)

    # Load input image
    load_id = g.add("LoadImage", {"image": input_image_name})
    image_ref = g.ref(load_id, 0)

    # VAE Encode
    encode_id = g.add("VAEEncode", {"pixels": image_ref, "vae": vae_ref})
    latent_ref = g.ref(encode_id)

    # Optional Canny ControlNet
    if canny is not None:
        cn_load_id = g.add("ControlNetLoader", {
            "control_net_name": "control-lora-canny-rank256.safetensors",
        })
        canny_id = g.add("Canny", {
            "low_threshold": canny.low_thresh,
            "high_threshold": canny.high_thresh,
            "image": image_ref,
        })
        cn_apply_id = g.add("ControlNetApply", {
            "strength": canny.strength,
            "conditioning": pos_ref,
            "control_net": g.ref(cn_load_id),
            "image": g.ref(canny_id),
        })
        pos_ref = g.ref(cn_apply_id)

    # KSampler
    sampler_id = g.add("KSampler", {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": denoise,
        "model": model_ref,
        "positive": pos_ref,
        "negative": neg_ref,
        "latent_image": latent_ref,
    })

    # VAE Decode
    decode_id = g.add("VAEDecode", {"samples": g.ref(sampler_id), "vae": vae_ref})

    # Save
    g.add("SaveImage", {"images": g.ref(decode_id), "filename_prefix": SAVE_PREFIX})

    return g.nodes


def build_flux1d_workflow(input_image_name: str, settings: LoopSettings, defaults: dict) -> dict:
    """Build a Flux.1D img2img workflow."""
    g = _NodeGraph()

    checkpoint = settings.checkpoint
    clip = settings.clip or defaults.get("clip", ["t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors"])
    prompt = settings.prompt or ""
    steps = settings.denoise_steps or defaults.get("denoise_steps", 20)
    denoise = round(settings.denoise_amt, 2) if settings.denoise_amt is not None else 0.5
    seed = settings.seed
    loras = settings.loras or []

    # Load UNET
    unet_id = g.add("UNETLoader", {"unet_name": checkpoint, "weight_dtype": "default"})
    model_ref = g.ref(unet_id)

    # Load Dual CLIP
    clip_id = g.add("DualCLIPLoader", {
        "clip_name1": clip[0],
        "clip_name2": clip[1],
        "type": "flux",
    })
    clip_ref = g.ref(clip_id)

    # Load VAE
    vae_id = g.add("VAELoader", {"vae_name": "ae.safetensors"})
    vae_ref = g.ref(vae_id)

    # Chain LoRAs
    for lora in loras:
        lora_id = g.add("LoraLoader", {
            "lora_name": lora.lora_path,
            "strength_model": lora.lora_strength,
            "strength_clip": lora.lora_strength,
            "model": model_ref,
            "clip": clip_ref,
        })
        model_ref = g.ref(lora_id, 0)
        clip_ref = g.ref(lora_id, 1)

    # Text conditioning
    text_id = g.add("CLIPTextEncode", {"text": prompt, "clip": clip_ref})
    guidance_id = g.add("FluxGuidance", {"guidance": 3.5, "conditioning": g.ref(text_id)})
    guider_id = g.add("BasicGuider", {"model": model_ref, "conditioning": g.ref(guidance_id)})

    # Load input image + VAE encode
    load_id = g.add("LoadImage", {"image": input_image_name})
    encode_id = g.add("VAEEncode", {"pixels": g.ref(load_id), "vae": vae_ref})

    # Sampling
    noise_id = g.add("RandomNoise", {"noise_seed": seed})
    sampler_select_id = g.add("KSamplerSelect", {"sampler_name": "euler"})
    scheduler_id = g.add("BasicScheduler", {
        "scheduler": "beta",
        "steps": steps,
        "denoise": denoise,
        "model": model_ref,
    })
    sample_id = g.add("SamplerCustomAdvanced", {
        "noise": g.ref(noise_id),
        "guider": g.ref(guider_id),
        "sampler": g.ref(sampler_select_id),
        "sigmas": g.ref(scheduler_id),
        "latent_image": g.ref(encode_id),
    })

    # VAE Decode (output index 0 from SamplerCustomAdvanced)
    decode_id = g.add("VAEDecode", {"samples": g.ref(sample_id, 0), "vae": vae_ref})

    # Save
    g.add("SaveImage", {"images": g.ref(decode_id), "filename_prefix": SAVE_PREFIX})

    return g.nodes


def build_sd3p5_workflow(input_image_name: str, settings: LoopSettings, defaults: dict) -> dict:
    """Build an SD3.5 img2img workflow."""
    g = _NodeGraph()

    checkpoint = settings.checkpoint or defaults.get("checkpoint")
    clip = settings.clip or defaults.get("clip", ["t5xxl_fp8_e4m3fn.safetensors", "clip_g.safetensors"])
    prompt = settings.prompt or ""
    neg_prompt = settings.neg_prompt or defaults.get("neg_prompt", "")
    steps = settings.denoise_steps or defaults.get("denoise_steps", 20)
    denoise = round(settings.denoise_amt, 2) if settings.denoise_amt is not None else 0.5
    cfg = settings.cfg or defaults.get("cfg", 3.5)
    seed = settings.seed
    loras = settings.loras or []

    # Load checkpoint (model + VAE, ignore checkpoint's CLIP)
    ckpt_id = g.add("CheckpointLoaderSimple", {"ckpt_name": checkpoint})
    model_ref = g.ref(ckpt_id, 0)
    vae_ref = g.ref(ckpt_id, 2)

    # Load Dual CLIP (SD3 uses separate CLIP loader)
    clip_id = g.add("DualCLIPLoader", {
        "clip_name1": clip[0],
        "clip_name2": clip[1],
        "type": "sd3",
    })
    clip_ref = g.ref(clip_id)

    # Chain LoRAs
    for lora in loras:
        lora_id = g.add("LoraLoader", {
            "lora_name": lora.lora_path,
            "strength_model": lora.lora_strength,
            "strength_clip": lora.lora_strength,
            "model": model_ref,
            "clip": clip_ref,
        })
        model_ref = g.ref(lora_id, 0)
        clip_ref = g.ref(lora_id, 1)

    # Text conditioning
    pos_id = g.add("CLIPTextEncode", {"text": prompt, "clip": clip_ref})
    neg_id = g.add("CLIPTextEncode", {"text": neg_prompt, "clip": clip_ref})

    # Load input image + VAE encode
    load_id = g.add("LoadImage", {"image": input_image_name})
    encode_id = g.add("VAEEncode", {"pixels": g.ref(load_id), "vae": vae_ref})

    # KSampler
    sampler_id = g.add("KSampler", {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "euler",
        "scheduler": "beta",
        "denoise": denoise,
        "model": model_ref,
        "positive": g.ref(pos_id),
        "negative": g.ref(neg_id),
        "latent_image": g.ref(encode_id),
    })

    # VAE Decode
    decode_id = g.add("VAEDecode", {"samples": g.ref(sampler_id), "vae": vae_ref})

    # Save
    g.add("SaveImage", {"images": g.ref(decode_id), "filename_prefix": SAVE_PREFIX})

    return g.nodes


BUILDER_MAP = {
    "sdxl": build_sdxl_workflow,
    "flux1d": build_flux1d_workflow,
    "sd3.5": build_sd3p5_workflow,
}
