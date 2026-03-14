"""Demo: blend_ref ordering relative to zoom.

Scenario A: blend_ref THEN zoom_in  (current futurama approach)
Scenario B: zoom_in THEN blend_ref  (reference stays undistorted each frame)

Run from repo root:
    python scripts/blend_order_demo.py
"""

from os.path import abspath, dirname, join
from os import makedirs
from shutil import copy
import sys

import tqdm

SCRIPT_DIR = dirname(abspath(__file__))
sys.path.append(dirname(SCRIPT_DIR))

import comfyui_looper.image_processing.transforms as transforms
import comfyui_looper.image_processing.animator as animator

INPUT_IMG = "data/img/planetexpress_futurama.png"
BLEND_REF = "data/img/fry.png"
OUTPUT_BASE = "output/blend_order_demo"
NUM_FRAMES = 30

BLEND_FULL = {
    "name": "blend_ref",
    "img_path": BLEND_REF,
    "opacity": 0.25,
    "blend_mode": "normal",
    "mask": "uniform",
}

BLEND_HALF = {
    "name": "blend_ref",
    "img_path": BLEND_REF,
    "opacity": 0.25,
    "blend_mode": "normal",
    "mask": "uniform",
    "scale": 0.5,
}

ZOOM_PARAMS = {
    "name": "zoom_in",
    "zoom_amt": 0.06,
}

SCENARIOS = {
    "A_blend_then_zoom": [BLEND_FULL, ZOOM_PARAMS],
    "B_zoom_then_blend": [ZOOM_PARAMS, BLEND_FULL],
    "C_zoom_then_blend_half": [ZOOM_PARAMS, BLEND_HALF],
}


def get_filename(idx: int, out_dir: str) -> str:
    return join(out_dir, f"frame_{idx:04}.png")


def run_scenario(name: str, transform_list: list, input_img: str, num_frames: int):
    out_dir = join(OUTPUT_BASE, name)
    makedirs(out_dir, exist_ok=True)
    copy(input_img, get_filename(0, out_dir))

    transforms.Transform.validate_transformation_params(transform_list)

    print(f"\n--- {name} ---")
    print(f"  Transform order: {[t['name'] for t in transform_list]}")
    for idx in tqdm.tqdm(range(num_frames)):
        curr_path = get_filename(idx, out_dir)
        next_path = get_filename(idx + 1, out_dir)
        auto_params = transforms.AutomaticTransformParams(
            n=idx, offset=0, total_n=num_frames, wavefile=None
        )
        img, _ = transforms.load_image_with_transforms(
            curr_path, transform_list, auto_params
        )
        img.save(next_path, compress_level=5)

    gif_path = join(out_dir, f"{name}.gif")
    animator.make_gif(
        input_folder=out_dir,
        gif_output=gif_path,
        params={"frame_delay": 150, "max_dim": 512},
    )
    print(f"  GIF saved: {gif_path}")


if __name__ == "__main__":
    for scenario_name, transform_list in SCENARIOS.items():
        run_scenario(scenario_name, transform_list, INPUT_IMG, NUM_FRAMES)

    print(f"\nDone! Compare the GIFs in {OUTPUT_BASE}/")
