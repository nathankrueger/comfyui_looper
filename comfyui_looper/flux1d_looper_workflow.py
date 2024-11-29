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
FLUX_VAE="ae.safetensors"
FLUX_CLIP1="t5xxl_fp8_e4m3fn.safetensors"  #TODO: these should be part of json spec
FLUX_CLIP2="clip_l.safetensors"
FLUX_GUIDANCE=3.5

def flux1d_looper_main(
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
        vaeencode = VAEEncode()
        vaedecode = VAEDecode()
        cliptextencode = CLIPTextEncode()
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        sampler_select_val = NODE_CLASS_MAPPINGS["KSamplerSelect"]().get_sampler(sampler_name="ddim")[0]

        # wrappers
        lora_mgr = LoraManager()
        model_mgr = Flux1DModelManager()

        prev_seed = None
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

            # load in new model if changed
            unet_model, clip_model, vae_model = model_mgr.reload_if_needed(checkpoint, FLUX_VAE, FLUX_CLIP1, FLUX_CLIP2)

            # only load in new loras as needd
            lora_model, lora_clip = lora_mgr.reload_if_needed(lora_list, unet_model, clip_model)

            # VAE encode
            vaeencode_result, = vaeencode.encode(
                pixels=image_tensor,
                vae=vae_model
            )

            # text conditioning
            conditioning, = cliptextencode.encode(
                text=positive_text,
                clip=lora_clip,
            )
            fluxguidance_result, = fluxguidance.append(
                guidance=FLUX_GUIDANCE, conditioning=conditioning
            )
            guider_result, = basicguider.get_guider(
                model=lora_model,
                conditioning=fluxguidance_result
            )

            # latent sampler
            noise_result, = randomnoise.get_noise(noise_seed=seed)
            sigma_result, = basicscheduler.get_sigmas(
                scheduler="beta",
                steps=steps,
                denoise=denoise,
                model=lora_model,
            )
            sampler_output, denoised_output = samplercustomadvanced.sample(
                noise=noise_result,
                guider=guider_result,
                sampler=sampler_select_val,
                sigmas=sigma_result,
                latent_image=vaeencode_result
            )

            # VAE decode
            vaedecode_result, = vaedecode.decode(
                samples=denoised_output,
                vae=vae_model
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
