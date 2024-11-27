import os
import random
import sys
import torch
import argparse
from pathlib import Path

import gif_maker as gif_maker
from json_spec import SettingsManager
from transforms import load_image_with_transforms
from util import (
    import_custom_nodes,
    add_comfyui_directory_to_sys_path,
    add_extra_model_paths,
    save_tensor_to_images,
    get_loop_img_filename,
    resize_image,
)

# handle args which are otherwise consumed by comfyui
LOOPER_ARGS = sys.argv[1:]
print(f'ARGS: {LOOPER_ARGS}')
sys.argv = [sys.argv[0]]

# ComfyUI imports
add_comfyui_directory_to_sys_path()
add_extra_model_paths()
from node_helpers import (
    LoraManager,
    CheckpointManager,
    SDXLClipEncodeWrapper,
)
import folder_paths
from nodes import (
    VAEDecode,
    VAEEncode,
    KSampler,
    ControlNetLoader,
    ControlNetApply,
    NODE_CLASS_MAPPINGS,
)

# constants
LOOP_IMG='looper.png'
SDXL_WIDTH=1024
CANNY_CONTROLNET='control-lora-canny-rank256.safetensors'
NEGATIVE_TEXT='text, watermark, logo'

def looper_main(loop_img_path: str, output_folder: str, json_file: str, gif_file: str | None, gif_max_dim: int, gif_frame_delay: int):
    import_custom_nodes()
    sm = SettingsManager(json_file)
    with torch.inference_mode():
        # comfy nodes
        canny_node = NODE_CLASS_MAPPINGS["Canny"]()
        controlnetloader = ControlNetLoader()
        controlnetapply = ControlNetApply()
        vaeencode = VAEEncode()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        lora_mgr = LoraManager()
        ckpt_mgr = CheckpointManager()
        text_cond = SDXLClipEncodeWrapper()

        controlnetloader_result = None
    
        for iter in range(sm.get_total_iterations()):
            positive_text = sm.get_setting_for_iter('prompt', iter)
            steps = sm.get_setting_for_iter('denoise_steps', iter)
            denoise = sm.get_setting_for_iter('denoise_amt', iter)
            lora_list = sm.get_setting_for_iter('loras', iter)
            checkpoint = sm.get_setting_for_iter('checkpoint', iter)
            transforms = sm.get_setting_for_iter('transforms', iter) if iter > 0 else None
            canny = sm.get_setting_for_iter('canny', iter)

            # load in the canny if needed
            if canny is not None:
                canny_strength, canny_low_thresh, canny_high_thresh = canny
                if controlnetloader_result is None:
                    controlnetloader_result, = controlnetloader.load_controlnet(CANNY_CONTROLNET)

            # load in image & resize it
            image_tensor = load_image_with_transforms(image_path=loop_img_path, transforms=transforms)

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
            h, w = image_tensor.shape[1:3]
            pos_cond, neg_cond = text_cond.encode(w, h, positive_text, NEGATIVE_TEXT, lora_clip)

            # canny
            if canny is not None:
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
                seed=random.randint(1, 2**64),
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
            save_tensor_to_images(output_filenames=[loop_img_path, os.path.join(output_folder, get_loop_img_filename(iter+1))], image=vaedecode_result[0])
            iter += 1

    # save gif animation
    if gif_file is not None:
        gif_maker.make_gif(
            input_folder=output_folder,
            frame_delay=gif_frame_delay,
            max_dimension=gif_max_dim,
            gif_output=gif_file
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loop hallucinate')
    parser.add_argument('-i', '--input_img', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-j', '--json_file', type=str, required=True)
    parser.add_argument('-g', '--gif_file', type=str, required=False)
    parser.add_argument('-d', '--gif_frame_delay', type=int, default=250)
    parser.add_argument('-s', '--gif_max_dimension', type=int, default=0)
    args=parser.parse_args(LOOPER_ARGS)

    loopback_filename = str(Path(folder_paths.get_input_directory()) / LOOP_IMG)
    starting_point_filename = str(Path(args.output_folder) / get_loop_img_filename(0))
    
    # ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # make a copy of the starting image to the loopback image, and to the output folder
    resize_image(args.input_img, loopback_filename, SDXL_WIDTH)
    resize_image(args.input_img, starting_point_filename, SDXL_WIDTH)
    
    # run the diffusion
    looper_main(
        loop_img_path=loopback_filename,
        output_folder=os.path.abspath(args.output_folder),
        json_file=args.json_file,
        gif_file=args.gif_file,
        gif_frame_delay=args.gif_frame_delay,
        gif_max_dim=args.gif_max_dimension
    )
