import os
import sys
import signal
import logging
import argparse
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

from workflow.engine_factory import get_all_workflows, create_workflow
from workflow.looper_workflow import looper_main
from image_processing.transforms import get_transform_help_string
from utils.json_spec import SettingsManager
from utils.comfyui_client import ComfyUIClient
from utils.image_store import FilesystemImageStore, ZipImageStore
from utils.util import (
    get_log_filename,
    get_loop_img_filename,
    parse_params
)

# constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_BASENAME='looper_log.log'
LOOP_IMG=str(PROJECT_ROOT / 'output' / 'looper.png')
WSGI_THREADS=8

def get_default_output_folder(json_file: str) -> str:
    """Generate a default output folder name from the workflow JSON filename and a timestamp."""
    workflow_name = Path(json_file).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return str(PROJECT_ROOT / 'output' / f'{workflow_name}_{timestamp}')

def get_output_folder(output_folder_arg: str, total_reps: int, current_rep: int) -> str:
    if total_reps > 1:
        return os.path.join(output_folder_arg, f"pass_{current_rep}")
    else:
        return output_folder_arg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loop hallucinate', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=get_transform_help_string())
    parser.add_argument('-i', '--input_img', type=str, required=False, default=None)
    parser.add_argument('-o', '--output_folder', type=str, required=False, default=None, help='Output folder (default: data/<workflow_name>_<timestamp>)')
    parser.add_argument('-j', '--json_file', type=str, required=False, default=None)
    parser.add_argument('-w', '--workflow_type', default='sdxl', choices=list(get_all_workflows()))
    parser.add_argument('-p', '--passes', type=int, default=1)
    parser.add_argument('-l', '--log_elaborated_settings', required=False, type=str, help='Will log the elaborated settings values only')
    parser.add_argument('-t', '--animation_type', type=str, default='gif', choices=['gif', 'mp4'])
    parser.add_argument('-a', '--animation_filename', type=str, required=False)
    parser.add_argument('-x', '--animation_param', action='append', dest='animation_params')
    parser.add_argument('-I', '--interactive', action='store_true', default=False, help='Launch interactive web control interface')
    parser.add_argument('-P', '--port', type=int, default=5000, help='Port for interactive web UI (default: 5000)')
    parser.add_argument('-c', '--comfyui-url', type=str, default='http://localhost:8188', help='URL of the ComfyUI server (default: http://localhost:8188)')
    parser.add_argument('-z', '--zip-storage', action='store_true', default=False, help='Store output images in a zip file instead of loose PNGs')
    args = parser.parse_args()

    # In non-interactive mode, -j is required
    if not args.interactive and not args.json_file:
        parser.error('-j/--json_file is required in non-interactive mode')

    if args.output_folder is None and args.json_file is not None:
        args.output_folder = get_default_output_folder(args.json_file)
        print(f"No output folder specified, using: {args.output_folder}")

    animation_params = parse_params(args.animation_params)

    client = ComfyUIClient(args.comfyui_url)

    if args.log_elaborated_settings:
        sm = SettingsManager(args.json_file, animation_params)
        sm.validate()
        sm.log_elaborated_settings(args.log_elaborated_settings)
        exit(0)

    if args.interactive:
        from interactive.app_state import AppState
        from interactive.flask_app import create_app

        app_state = AppState(
            workflow_type=args.workflow_type,
            comfyui_url=args.comfyui_url,
            input_img=args.input_img,
            animation_type=args.animation_type,
            animation_filename=args.animation_filename,
            animation_params=animation_params,
            use_zip=args.zip_storage,
        )

        # If json_file provided on CLI, auto-start the loop
        if args.json_file:
            output_folder = args.output_folder or get_default_output_folder(args.json_file)
            app_state.start_loop(args.json_file, output_folder)

        if args.json_file:
            print(f"Interactive mode: open http://0.0.0.0:{args.port} in your browser")
        else:
            print(f"Interactive mode (picker): open http://0.0.0.0:{args.port} in your browser")

        app = create_app(app_state)

        from waitress import create_server
        server = create_server(app, host='0.0.0.0', port=args.port, threads=WSGI_THREADS)

        def shutdown_handler(signum, frame):
            print("\nShutting down...")
            try:
                app_state.stop_loop()
            except Exception:
                pass
            server.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        server.run()
    else:
        for rep in range(args.passes):
            output_folder = os.path.abspath(get_output_folder(args.output_folder, args.passes, rep))

            # ensure the output folder exists
            os.makedirs(output_folder, exist_ok=True)

            # create the image store
            if args.zip_storage:
                image_store = ZipImageStore(output_folder)
            else:
                image_store = FilesystemImageStore(output_folder)

            # run the diffusion
            log_filename = get_log_filename(LOG_BASENAME)
            with open(os.path.join(output_folder, log_filename), 'w', encoding='utf-8') as log_file:
                workflow_engine = create_workflow(args.workflow_type, client)
                no_input_image = args.input_img is None
                if no_input_image:
                    workflow_engine.create_blank_image_for_model([LOOP_IMG])
                else:
                    workflow_engine.resize_images_for_model(args.input_img, [LOOP_IMG])
                    image_store.import_from_path(LOOP_IMG, get_loop_img_filename(0))

                looper_main(
                    engine=workflow_engine,
                    loop_img_path=LOOP_IMG,
                    output_folder=output_folder,
                    json_file=args.json_file,
                    animation_file=args.animation_filename,
                    animation_type=args.animation_type,
                    animation_params=animation_params,
                    log_file=log_file,
                    no_input_image=no_input_image,
                    image_store=image_store,
                )

            image_store.close()
