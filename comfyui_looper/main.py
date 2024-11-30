import os
import sys
import argparse
from pathlib import Path

from workflow.engine_factory import get_all_workflows, create_workflow
from workflow.looper_workflow import looper_main
from image_processing.transforms import get_transform_help_string
from utils.util import (
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

def get_output_folder(output_folder_arg: str, total_reps: int, current_rep: int) -> str:
    if total_reps > 1:
        return os.path.join(output_folder_arg, f"pass_{current_rep}")
    else:
        return output_folder_arg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loop hallucinate', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=get_transform_help_string())
    parser.add_argument('-i', '--input_img', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-j', '--json_file', type=str, required=True)
    parser.add_argument('-w', '--workflow_type', default='sdxl', choices=list(get_all_workflows()))
    parser.add_argument('-p', '--passes', type=int, default=1)
    parser.add_argument('-t', '--animation_type', type=str, default='gif', choices=['gif', 'mp4'])
    parser.add_argument('-a', '--animation_filename', type=str, required=False)
    parser.add_argument('-x', '--animation_param', action='append', dest='animation_params')
    args = parser.parse_args(LOOPER_ARGS)
    animation_params = parse_params(args.animation_params)

    for rep in range(args.passes):
        output_folder = os.path.abspath(get_output_folder(args.output_folder, args.passes, rep))
        loopback_filename = str(Path(get_input_directory()) / LOOP_IMG)
        starting_point_filename = str(Path(output_folder) / get_loop_img_filename(0))
        
        # ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # run the diffusion
        log_filename = get_log_filename(LOG_BASENAME)
        with open(os.path.join(output_folder, log_filename), 'w', encoding='utf-8') as log_file:
            workflow_engine = create_workflow(args.workflow_type)
            workflow_engine.resize_images_for_model(args.input_img, [loopback_filename, starting_point_filename])

            looper_main(
                engine=workflow_engine,
                loop_img_path=loopback_filename,
                output_folder=output_folder,
                json_file=args.json_file,
                animation_file=args.animation_filename,
                animation_type=args.animation_type,
                animation_params=animation_params,
                log_file=log_file
            )