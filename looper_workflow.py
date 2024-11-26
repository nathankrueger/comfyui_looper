import os
import random
import sys
import shutil
import torch
import csv
import argparse
from pathlib import Path

import gif_maker
from util import (
    import_custom_nodes,
    add_comfyui_directory_to_sys_path,
    add_extra_model_paths,
    load_image,
    save_images,
    get_loop_img_filename,
)

LOOP_IMG='looper.png'

# handle args which are otherwise consumed by comfyui
LOOPER_ARGS = sys.argv[1:]
print(f'ARGS: {LOOPER_ARGS}')
sys.argv = [sys.argv[0]]

add_comfyui_directory_to_sys_path()
add_extra_model_paths()

from lora_manager import LoraManager
import folder_paths
from nodes import (
    VAEDecode,
    VAEEncode,
    CheckpointLoaderSimple,
    KSampler,
    NODE_CLASS_MAPPINGS,
)

def looper_main(loop_img_path: str, output_folder: str, csv_file: str, gif_file: str | None, gif_max_dim: int, gif_frame_delay: int):
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        ckpt_model, ckpt_clip, ckpt_vae = checkpointloadersimple.load_checkpoint(
            ckpt_name="sdXL_v10VAEFix.safetensors"
        )

        # define the things that must be reprocessed each loop
        cliptextencodesdxl = NODE_CLASS_MAPPINGS["CLIPTextEncodeSDXL"]()
        image_scale_to_side = NODE_CLASS_MAPPINGS["Image scale to side"]()
        vaeencode = VAEEncode()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        lora_mgr = LoraManager()

        # csv format is as follows: iterations % comma separted keywords % steps % denoise amount % lora strength
        count = 0

        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='%', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for idx, row in enumerate(reader):
                # decode csv
                num_iter = int(row[0].strip())
                positive_keywords = row[1]
                steps = int(row[2].strip())
                denoise = float(row[3].strip())
                lora_strength = float(row[4].strip())
            
                for _ in range(num_iter):
                    image_load_result = load_image(image_path=loop_img_path)

                    image_scale_to_side_result, = image_scale_to_side.upscale(
                        side_length=1024,
                        side="Longest",
                        upscale_method="nearest-exact",
                        crop="disabled",
                        image=image_load_result[0]
                    )
                    vaeencode_result, = vaeencode.encode(
                        pixels=image_scale_to_side_result,
                        vae=ckpt_vae
                    )

                    # only load in new loras as needd
                    lora_model, lora_clip = lora_mgr.reload_if_needed([('SDXL-PsyAI-v4.safetensors', 1.0), ('nathan.safetensors', 1.0)], ckpt_model, ckpt_clip)

                    positive_encoding, = cliptextencodesdxl.encode(
                        width=1024,
                        height=1024,
                        crop_w=0,
                        crop_h=0,
                        target_width=1024,
                        target_height=1024,
                        text_g=positive_keywords,
                        text_l=positive_keywords,
                        clip=lora_clip
                    )
                    negative_encoding, = cliptextencodesdxl.encode(
                        width=1024,
                        height=1024,
                        crop_w=0,
                        crop_h=0,
                        target_width=1024,
                        target_height=1024,
                        text_g="text, watermark, logo",
                        text_l="text, watermark, logo",
                        clip=lora_clip
                    )
                    ksampler_result, = ksampler.sample(
                        seed=random.randint(1, 2**64),
                        steps=steps,
                        cfg=8,
                        sampler_name="euler",
                        scheduler="normal",
                        denoise=denoise,
                        model=lora_model,
                        positive=positive_encoding,
                        negative=negative_encoding,
                        latent_image=vaeencode_result
                    )
                    vaedecode_result, = vaedecode.decode(
                        samples=ksampler_result,
                        vae=ckpt_vae
                    )

                    # save the images -- loop filename, and requested output
                    save_images(output_filenames=[loop_img_path, os.path.join(output_folder, get_loop_img_filename(count+1))], image=vaedecode_result[0])
                    count += 1

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
    parser.add_argument('-c', '--csv_file', type=str, required=True)
    parser.add_argument('-g', '--gif_file', type=str, required=False)
    parser.add_argument('-d', '--gif_frame_delay', type=int, default=250)
    parser.add_argument('-s', '--gif_max_dimension', type=int, default=0)
    args=parser.parse_args(LOOPER_ARGS)

    loopback_filename = str(Path(folder_paths.get_input_directory()) / LOOP_IMG)
    starting_point_filename = str(Path(args.output_folder) / get_loop_img_filename(0))
    
    # ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # make a copy of the starting image to the loopback image, and to the output folder
    shutil.copy(args.input_img, loopback_filename)
    shutil.copy(args.input_img, starting_point_filename)
    
    # run the diffusion
    looper_main(
        loop_img_path=loopback_filename,
        output_folder=os.path.abspath(args.output_folder),
        csv_file=args.csv_file,
        gif_file=args.gif_file,
        gif_frame_delay=args.gif_frame_delay,
        gif_max_dim=args.gif_max_dimension
    )
