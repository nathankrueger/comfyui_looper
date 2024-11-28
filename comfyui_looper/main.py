import os
import sys
import argparse
from pathlib import Path
from sdxl_looper_workflow import sdxl_looper_main, SDXL_WIDTH, SDXL_LATENT_REDUCTION_FACTOR
from util import (
    resize_image_match_area,
    get_log_filename,
    get_loop_img_filename,
    parse_params
)
from folder_paths import get_input_directory

# handle args which are otherwise consumed by comfyui
LOOPER_ARGS = sys.argv[1:]
print(f'ARGS: {LOOPER_ARGS}')
sys.argv = [sys.argv[0]]

# constants
LOG_BASENAME='looper_log.log'
LOOP_IMG='looper.png'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loop hallucinate')
    parser.add_argument('-i', '--input_img', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-j', '--json_file', type=str, required=True)
    parser.add_argument('-t', '--animation_type', type=str, default='gif', choices=['gif', 'mp4'])
    parser.add_argument('-a', '--animation_filename', type=str, required=False)
    parser.add_argument('-x', '--animation_param', action='append', dest='animation_params')
    parser.add_argument('-w', '--workflow_type', default='sdxl', choices=['sdxl'])
    args=parser.parse_args(LOOPER_ARGS)
    animation_params = parse_params(args.animation_params)

    loopback_filename = str(Path(get_input_directory()) / LOOP_IMG)
    starting_point_filename = str(Path(args.output_folder) / get_loop_img_filename(0))
    
    # ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # make a copy of the starting image to the loopback image, and to the output folder
    resize_image_match_area(args.input_img, loopback_filename, SDXL_WIDTH**2, SDXL_LATENT_REDUCTION_FACTOR)
    resize_image_match_area(args.input_img, starting_point_filename, SDXL_WIDTH**2, SDXL_LATENT_REDUCTION_FACTOR)
    
    # run the diffusion
    log_filename = get_log_filename(LOG_BASENAME)
    with open(os.path.join(args.output_folder, log_filename), 'w', encoding='utf-8') as log_file:
        if args.workflow_type == 'sdxl':
            sdxl_looper_main(
                loop_img_path=loopback_filename,
                output_folder=os.path.abspath(args.output_folder),
                json_file=args.json_file,
                animation_file=args.animation_filename,
                animation_type=args.animation_type,
                animation_params=args.animation_params,
                log_file=log_file
            )
