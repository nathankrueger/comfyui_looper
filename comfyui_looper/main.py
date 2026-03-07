import os
import argparse
from pathlib import Path

from workflow.engine_factory import get_all_workflows, create_workflow
from workflow.looper_workflow import looper_main
from image_processing.transforms import get_transform_help_string
from utils.json_spec import SettingsManager
from utils.comfyui_client import ComfyUIClient
from utils.util import (
    get_log_filename,
    get_loop_img_filename,
    parse_params
)

# constants
LOG_BASENAME='looper_log.log'
LOOP_IMG='output/looper.png'

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
    parser.add_argument('-l', '--log_elaborated_settings', required=False, type=str, help='Will log the elaborated settings values only')
    parser.add_argument('-t', '--animation_type', type=str, default='gif', choices=['gif', 'mp4'])
    parser.add_argument('-a', '--animation_filename', type=str, required=False)
    parser.add_argument('-x', '--animation_param', action='append', dest='animation_params')
    parser.add_argument('--interactive', action='store_true', default=False,
                        help='Launch interactive web control interface on port 5000')
    parser.add_argument('--comfyui-url', type=str, default='http://localhost:8188',
                        help='URL of the ComfyUI server (default: http://localhost:8188)')
    args = parser.parse_args()
    animation_params = parse_params(args.animation_params)

    client = ComfyUIClient(args.comfyui_url)

    if args.log_elaborated_settings:
        sm = SettingsManager(args.json_file, animation_params)
        sm.validate()
        sm.log_elaborated_settings(args.log_elaborated_settings)
        exit(0)

    if args.interactive:
        import threading
        from interactive.loop_state import LoopState
        from interactive.interactive_loop import interactive_looper_main
        from interactive.flask_app import create_app

        if args.passes > 1:
            print("Warning: --interactive only supports a single pass. Using passes=1.")
            args.passes = 1

        output_folder = os.path.abspath(get_output_folder(args.output_folder, args.passes, 0))
        os.makedirs(output_folder, exist_ok=True)
        starting_point_filename = str(Path(output_folder) / get_loop_img_filename(0))

        workflow_engine = create_workflow(args.workflow_type, client)
        workflow_engine.resize_images_for_model(args.input_img, [LOOP_IMG, starting_point_filename])

        # Get total iterations for state initialization
        sm_temp = SettingsManager(args.json_file, animation_params)
        sm_temp.validate()
        total_iterations = sm_temp.get_total_iterations()

        loop_state = LoopState(total_iterations=total_iterations, output_folder=output_folder)

        log_filename = get_log_filename(LOG_BASENAME)
        log_file = open(os.path.join(output_folder, log_filename), 'w', encoding='utf-8')

        def run_loop():
            try:
                interactive_looper_main(
                    engine=workflow_engine,
                    loop_img_path=LOOP_IMG,
                    output_folder=output_folder,
                    json_file=args.json_file,
                    animation_file=args.animation_filename,
                    animation_type=args.animation_type,
                    animation_params=animation_params,
                    log_file=log_file,
                    state=loop_state,
                )
            except Exception as e:
                print(f"Loop thread error: {e}")
            finally:
                log_file.close()

        loop_thread = threading.Thread(target=run_loop, daemon=True)
        loop_thread.start()

        print(f"Interactive mode: open http://0.0.0.0:5000 in your browser")
        app = create_app(loop_state)
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        for rep in range(args.passes):
            output_folder = os.path.abspath(get_output_folder(args.output_folder, args.passes, rep))
            starting_point_filename = str(Path(output_folder) / get_loop_img_filename(0))

            # ensure the output folder exists
            os.makedirs(output_folder, exist_ok=True)

            # run the diffusion
            log_filename = get_log_filename(LOG_BASENAME)
            with open(os.path.join(output_folder, log_filename), 'w', encoding='utf-8') as log_file:
                workflow_engine = create_workflow(args.workflow_type, client)
                workflow_engine.resize_images_for_model(args.input_img, [LOOP_IMG, starting_point_filename])

                looper_main(
                    engine=workflow_engine,
                    loop_img_path=LOOP_IMG,
                    output_folder=output_folder,
                    json_file=args.json_file,
                    animation_file=args.animation_filename,
                    animation_type=args.animation_type,
                    animation_params=animation_params,
                    log_file=log_file
                )
