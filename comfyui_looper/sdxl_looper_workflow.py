import os
import torch
import tqdm
from typing import IO
from PIL.PngImagePlugin import PngInfo

from animator import make_animation
from json_spec import SettingsManager, default_seed
from transforms import load_image_with_transforms
from util import (
    import_custom_nodes,
    add_comfyui_directory_to_sys_path,
    add_extra_model_paths,
    save_tensor_to_images,
    get_loop_img_filename,
)

# ComfyUI imports
add_comfyui_directory_to_sys_path()
add_extra_model_paths()
import_custom_nodes()
from node_wrappers import (
    LoraManager,
    CheckpointManager,
    ClipEncodeWrapper,
    ControlNetManager,
)
from nodes import (
    VAEDecode,
    VAEEncode,
    KSampler,
    ControlNetApply,
    NODE_CLASS_MAPPINGS,
)

# constants
SDXL_WIDTH=1024
SDXL_LATENT_REDUCTION_FACTOR=8
NEGATIVE_TEXT='text, watermark, logo'
CANNY_CONTROLNET='control-lora-canny-rank256.safetensors'

def sdxl_looper_main(
    loop_img_path: str,
    output_folder: str,
    json_file: str,
    animation_file: str | None,
    animation_type: str,
    animation_params: dict[str, str],
    log_file: IO[str]
):
    sm = SettingsManager(json_file)
    sm.validate()

    with torch.inference_mode():
        # comfy nodes
        canny_node = NODE_CLASS_MAPPINGS["Canny"]()
        controlnetapply = ControlNetApply()
        vaeencode = VAEEncode()
        ksampler = KSampler()
        vaedecode = VAEDecode()

        # wrappers
        lora_mgr = LoraManager()
        ckpt_mgr = CheckpointManager()
        text_cond = ClipEncodeWrapper()
        control_net_mgr = ControlNetManager(CANNY_CONTROLNET)

        prev_seed = None
        controlnetloader_result = None
        total_iter = sm.get_total_iterations()

        for iter in tqdm.tqdm(range(total_iter)):
            print()
            
            # load settings from JSON
            loopsettings = sm.get_elaborated_loopsettings_for_iter(iter)
            positive_text = loopsettings.prompt
            steps = loopsettings.denoise_steps
            denoise = loopsettings.denoise_amt
            lora_list = loopsettings.loras
            checkpoint = loopsettings.checkpoint
            canny = loopsettings.canny
            transforms = loopsettings.transforms if iter > 0 else None
            seed = loopsettings.seed

            # if a new seed is explicitly set, use it, otherwise always get a new one
            if seed == prev_seed:
                seed = default_seed()
                loopsettings.seed = seed
            sm.update_seed(iter, seed)
            prev_seed = seed

            # load in image & resize it
            image_tensor = load_image_with_transforms(
                image_path=loop_img_path,
                transforms=transforms,
                iter=iter,
                offset=loopsettings.offset,
                total_iter=total_iter
            )

            # load in new checkpoint if changed
            ckpt_model, ckpt_clip, ckpt_vae = ckpt_mgr.reload_if_needed(checkpoint)

            # VAE encode
            vaeencode_result, = vaeencode.encode(
                pixels=image_tensor,
                vae=ckpt_vae
            )

            # only load in new loras as needd
            lora_model, lora_clip = lora_mgr.reload_if_needed(lora_list, ckpt_model, ckpt_clip)

            # conditioning
            pos_cond, neg_cond = text_cond.encode(positive_text, NEGATIVE_TEXT, lora_clip)

            # load in the canny if needed
            if (controlnetloader_result := control_net_mgr.reload_if_needed(canny)) is not None:
                canny_strength, canny_low_thresh, canny_high_thresh = canny  
                canny_result, = canny_node.detect_edge(
                    low_threshold=canny_low_thresh,
                    high_threshold=canny_high_thresh,
                    image=image_tensor,
                )
                pos_cond, = controlnetapply.apply_controlnet(
                    strength=canny_strength,
                    conditioning=pos_cond,
                    control_net=controlnetloader_result,
                    image=canny_result,
                )

            # latent sampler
            ksampler_result, = ksampler.sample(
                seed=seed,
                steps=steps,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=denoise,
                model=lora_model,
                positive=pos_cond,
                negative=neg_cond,
                latent_image=vaeencode_result
            )

            # VAE decode
            vaedecode_result, = vaedecode.decode(
                samples=ksampler_result,
                vae=ckpt_vae
            )

            # save the images -- loop filename, and requested output
            loopsettings_json = loopsettings.to_json(indent=4)
            output_image_filename = os.path.join(output_folder, get_loop_img_filename(iter+1))
            pnginfo = PngInfo()
            pnginfo.add_text(key='looper_settings', value=loopsettings_json, zip=False)
            save_tensor_to_images(
                output_filenames=[loop_img_path, output_image_filename],
                image=vaedecode_result[0],
                png_info=pnginfo
            )

            # add entry to the logfile
            log_file.write(f"{output_image_filename}:" + loopsettings_json + os.linesep)

    # save animation
    if animation_file is not None:
        make_animation(
            type=animation_type,
            input_folder=output_folder,
            output_animation=os.path.join(output_folder, animation_file),
            params=animation_params
        )
