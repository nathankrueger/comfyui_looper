import os
import random
import sys
import shutil
from typing import Sequence, Mapping, Any, Union
import torch
import csv
import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageFile
import numpy as np
import gif_maker

LOOP_IMG='looper.png'

# handle args which are otherwise consumed by comfyui
LOOPER_ARGS = sys.argv[1:]
print(f'ARGS: {LOOPER_ARGS}')
sys.argv = [sys.argv[0]]

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)

def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")

def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()

add_comfyui_directory_to_sys_path()
add_extra_model_paths()

import folder_paths

from nodes import (
    LoraLoader,
    VAEDecode,
    NODE_CLASS_MAPPINGS,
    VAEEncode,
    CheckpointLoaderSimple,
    KSampler,
)

def zoom_in(img: ImageFile, amt: float) -> ImageFile:
    init_width, init_height = img.size

    mod_width = int(amt * float(init_width))
    mod_height = int(amt * float(init_height))

    left = init_width - mod_width
    right = init_width - left
    top = init_height - mod_height
    bottom = init_height - top

    cropped = img.crop((left, top, right, bottom))
    result = cropped.resize((init_width, init_height))
    return result
    
def load_image(image):
    image_path = folder_paths.get_annotated_filepath(image)
    i = Image.open(image_path)

    if True:
        i = zoom_in(i, 0.985)

    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    return (image, mask.unsqueeze(0))

def save_images(image, output_filenames: list[str]):
    first_image_path = None
    for output_filename in output_filenames:
        output_folder = os.path.dirname(output_filename)
        os.makedirs(output_folder, exist_ok=True)
        if first_image_path is None:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(output_filename, pnginfo=None, compress_level=0)
            first_image_path = output_filename
        else:
            shutil.copy(first_image_path, output_filename)

def get_loop_img_filename(idx: int) -> str:
    return f"loop_img_{idx:05}.png"

def main(loop_img_path: str, output_folder: str, csv_file: str, gif_file: str | None, gif_max_dim: int, gif_frame_delay: int):
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        loaded_checkpoint = checkpointloadersimple.load_checkpoint(
            ckpt_name="sdXL_v10VAEFix.safetensors"
        )

        # define the things that must be reprocessed each loop
        loraloader = LoraLoader()
        cliptextencodesdxl = NODE_CLASS_MAPPINGS["CLIPTextEncodeSDXL"]()
        image_scale_to_side = NODE_CLASS_MAPPINGS["Image scale to side"]()
        vaeencode = VAEEncode()
        ksampler = KSampler()
        vaedecode = VAEDecode()

        # csv format is as follows: iterations % comma separted keywords % steps % denoise amount % lora strength
        loraloader_result = None
        count = 0

        with open(csv_file, 'r', newline='') as csvfile:
            prev_lora_strength = None
            reader = csv.reader(csvfile, delimiter='%', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for idx, row in enumerate(reader):
                # decode csv
                num_iter = int(row[0].strip())
                positive_keywords = row[1]
                steps = int(row[2].strip())
                denoise = float(row[3].strip())
                lora_strength = float(row[4].strip())
            
                for _ in range(num_iter):
                    image_load_result = load_image(image=loop_img_path)

                    image_scale_to_side_result = image_scale_to_side.upscale(
                        side_length=1024,
                        side="Longest",
                        upscale_method="nearest-exact",
                        crop="disabled",
                        image=get_value_at_index(image_load_result, 0),
                    )
                    vaeencode_result = vaeencode.encode(
                        pixels=get_value_at_index(image_scale_to_side_result, 0),
                        vae=get_value_at_index(loaded_checkpoint, 2),
                    )
                
                    # only load in new lora as needd
                    if prev_lora_strength is not lora_strength:
                        loraloader_result = loraloader.load_lora(
                            lora_name="SDXL-PsyAI-v4.safetensors",
                            strength_model=lora_strength,
                            strength_clip=lora_strength,
                            model=get_value_at_index(loaded_checkpoint, 0),
                            clip=get_value_at_index(loaded_checkpoint, 1),
                        )
                        prev_lora_strength = lora_strength

                    positive_encoding = cliptextencodesdxl.encode(
                        width=1024,
                        height=1024,
                        crop_w=0,
                        crop_h=0,
                        target_width=1024,
                        target_height=1024,
                        text_g=positive_keywords,
                        text_l=positive_keywords,
                        clip=get_value_at_index(loraloader_result, 1),
                    )
                    negative_encoding = cliptextencodesdxl.encode(
                        width=1024,
                        height=1024,
                        crop_w=0,
                        crop_h=0,
                        target_width=1024,
                        target_height=1024,
                        text_g="text, watermark, logo",
                        text_l="text, watermark, logo",
                        clip=get_value_at_index(loraloader_result, 1),
                    )
                    ksampler_result = ksampler.sample(
                        seed=random.randint(1, 2**64),
                        steps=steps,
                        cfg=8,
                        sampler_name="euler",
                        scheduler="normal",
                        denoise=denoise,
                        model=get_value_at_index(loraloader_result, 0),
                        positive=get_value_at_index(positive_encoding, 0),
                        negative=get_value_at_index(negative_encoding, 0),
                        latent_image=get_value_at_index(vaeencode_result, 0),
                    )
                    vaedecode_result = vaedecode.decode(
                        samples=get_value_at_index(ksampler_result, 0),
                        vae=get_value_at_index(loaded_checkpoint, 2),
                    )

                    # save the images -- loop filename, and requested output
                    save_images(
                        output_filenames=[loop_img_path, os.path.join(output_folder, get_loop_img_filename(count+1))],
                        image=get_value_at_index(vaedecode_result, 0)[0]
                    )
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

    path_parts = os.path.splitext(args.input_img)
    loopback_filename = str(Path(folder_paths.get_input_directory()) / LOOP_IMG)
    starting_point_filename = str(Path(args.output_folder) / get_loop_img_filename(0))
    
    # ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # make a copy of the starting image to the loopback image, and to the output folder
    shutil.copy(args.input_img, loopback_filename)
    shutil.copy(args.input_img, starting_point_filename)
    
    # run the diffusion
    main(
        loop_img_path=loopback_filename,
        output_folder=os.path.abspath(args.output_folder),
        csv_file=args.csv_file,
        gif_file=args.gif_file,
        gif_frame_delay=args.gif_frame_delay,
        gif_max_dim=args.gif_max_dimension
    )
