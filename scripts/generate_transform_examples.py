"""
Generate example GIF animations for each transform.
Each GIF shows the transform applied iteratively (output fed back as input),
which is how the looper uses them.
"""
import sys
from os.path import abspath, dirname
SCRIPT_DIR = dirname(abspath(__file__))
sys.path.append(dirname(SCRIPT_DIR))

from PIL import Image, ImageOps
import tempfile
from pathlib import Path
from comfyui_looper.image_processing.transforms import Transform, TRANSFORM_LIBRARY, AutomaticTransformParams
from comfyui_looper.image_processing.animator import make_gif

INPUT_IMAGE = Path(SCRIPT_DIR).parent / 'data' / 'img' / 'simpsons_couch.png'
REF_IMAGE = Path(SCRIPT_DIR).parent / 'data' / 'img' / 'tatooine.png'
OUTPUT_DIR = Path(SCRIPT_DIR).parent / 'output' / 'transform_examples'
NUM_FRAMES = 30

# Define example configs for every transform
EXAMPLES = [
    # ---- Zoom variants ----
    {
        'name': 'zoom_in',
        'params': {'zoom_amt': 0.03},
        'description': 'Zoom into center of image',
    },
    {
        'name': 'zoom_in_left',
        'params': {'zoom_amt': 0.03},
        'description': 'Zoom into left side of image',
    },
    {
        'name': 'zoom_in_right',
        'params': {'zoom_amt': 0.03},
        'description': 'Zoom into right side of image',
    },
    {
        'name': 'zoom_in_up',
        'params': {'zoom_amt': 0.03},
        'description': 'Zoom into top of image',
    },
    {
        'name': 'zoom_in_down',
        'params': {'zoom_amt': 0.03},
        'description': 'Zoom into bottom of image',
    },
    {
        'name': 'zoom_out',
        'params': {'zoom_amt': 0.03},
        'description': 'Zoom out with reflect fill',
        'single_frame_variants': {
            'zoom_out_reflect': {'zoom_amt': 0.15, 'fill_mode': 'reflect'},
            'zoom_out_blur': {'zoom_amt': 0.15, 'fill_mode': 'blur'},
        }
    },
    # ---- Fold / Squeeze ----
    {
        'name': 'fold_vertical',
        'params': {'fold_amt': 10},
        'description': 'Fold image inward vertically',
    },
    {
        'name': 'fold_horizontal',
        'params': {'fold_amt': 10},
        'description': 'Fold image inward horizontally',
    },
    {
        'name': 'squeeze_wide',
        'params': {'squeeze_amt': 0.03},
        'description': 'Squeeze image wider',
    },
    {
        'name': 'squeeze_tall',
        'params': {'squeeze_amt': 0.03},
        'description': 'Squeeze image taller',
    },
    # ---- Distortions ----
    {
        'name': 'wave',
        'params': {'period': 60, 'strength': 8, 'rate': 1},
        'description': 'Sinusoidal wave distortion',
    },
    {
        'name': 'spiral',
        'params': {'strength': 0.3},
        'description': 'Twirl vortex distortion',
    },
    {
        'name': 'ripple',
        'params': {'amplitude': 8, 'wavelength': 80},
        'description': 'Concentric circular ripple distortion',
        'use_phase': True,
    },
    {
        'name': 'elastic',
        'params': {'strength': 30, 'smoothness': 10, 'seed': 42},
        'description': 'Smooth elastic deformation',
    },
    {
        'name': 'fisheye',
        'params': {'strength': 0.3},
        'description': 'Barrel / fisheye lens distortion',
    },
    # ---- Geometric ----
    {
        'name': 'rotate',
        'params': {'angle': 5},
        'description': 'Rotate and crop to fill',
    },
    {
        'name': 'perspective',
        'params': {'strength': 3, 'shrink_edge': 'top'},
        'description': 'Perspective warp (shrink top edge)',
        'single_frame_variants': {
            'perspective_top': {'strength': 15, 'shrink_edge': 'top'},
            'perspective_bottom': {'strength': 15, 'shrink_edge': 'bottom'},
            'perspective_left': {'strength': 15, 'shrink_edge': 'left'},
            'perspective_right': {'strength': 15, 'shrink_edge': 'right'},
        }
    },
    {
        'name': 'pan',
        'params': {'dx': 15, 'dy': 8},
        'description': 'Toroidal pan/scroll',
    },
    {
        'name': 'mirror',
        'params': {'mode': 'left_to_right'},
        'description': 'Mirror left half to right',
        # combine with a small zoom+pan so each frame has new content to mirror
        'pre_transforms': [
            {'name': 'zoom_in', 'params': {'zoom_amt': 0.02}},
            {'name': 'pan', 'params': {'dx': 3, 'dy': 0}},
        ],
        'single_frame_variants': {
            'mirror_left_to_right': {'mode': 'left_to_right'},
            'mirror_right_to_left': {'mode': 'right_to_left'},
            'mirror_top_to_bottom': {'mode': 'top_to_bottom'},
            'mirror_bottom_to_top': {'mode': 'bottom_to_top'},
        }
    },
    {
        'name': 'kaleidoscope',
        'params': {'segments': 6},
        'description': 'Kaleidoscope with 6 segments',
        # rotate the source slightly each frame so the kaleidoscope turns
        'pre_transforms': [
            {'name': 'rotate', 'params': {'angle': 3}},
        ],
        'single_frame_variants': {
            'kaleidoscope_4': {'segments': 4},
            'kaleidoscope_6': {'segments': 6},
            'kaleidoscope_8': {'segments': 8},
        }
    },
    # ---- Color ----
    {
        'name': 'hue_shift',
        'params': {'shift_deg': 12},
        'description': 'Hue rotation cycling through rainbow',
    },
    {
        'name': 'color_channel_offset',
        'params': {'r_offset': 8, 'g_offset': 0, 'b_offset': -8},
        'description': 'RGB channel separation / chromatic aberration',
    },
    {
        'name': 'contrast_brightness',
        'params': {'contrast': 1.08, 'brightness': 0.02},
        'description': 'Gradual contrast and brightness increase',
    },
    # ---- Composite ----
    {
        'name': 'paste_img',
        'params': {'img_path': str(REF_IMAGE), 'opacity': 0.0},
        'description': 'Linear blend with reference image',
        'animate_opacity': True,
    },
    {
        'name': 'blend_ref',
        'params': {'img_path': str(REF_IMAGE), 'opacity': 0.0, 'blend_mode': 'overlay', 'easing': 'ease_in_out'},
        'description': 'Overlay blend with reference image (ease_in_out)',
        'animate_opacity': True,
        'single_frame_variants': {
            'blend_ref_normal': {'img_path': str(REF_IMAGE), 'opacity': 0.5, 'blend_mode': 'normal'},
            'blend_ref_overlay': {'img_path': str(REF_IMAGE), 'opacity': 0.5, 'blend_mode': 'overlay'},
            'blend_ref_soft_light': {'img_path': str(REF_IMAGE), 'opacity': 0.5, 'blend_mode': 'soft_light'},
            'blend_ref_screen': {'img_path': str(REF_IMAGE), 'opacity': 0.5, 'blend_mode': 'screen'},
            'blend_ref_multiply': {'img_path': str(REF_IMAGE), 'opacity': 0.5, 'blend_mode': 'multiply'},
        }
    },
    # ---- Retro / Stylization ----
    {
        'name': 'pixelate',
        'params': {'block_size': 4},
        'description': 'Mosaic pixelation effect',
    },
]

def load_input():
    img = Image.open(INPUT_IMAGE)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    img.thumbnail((512, 512))
    return img

def apply_transform(img, name, params):
    t_cls = TRANSFORM_LIBRARY[name]
    # add dummy auto params
    full_params = dict(params)
    full_params.setdefault('n', 0)
    full_params.setdefault('offset', 0)
    full_params.setdefault('total_n', NUM_FRAMES)
    t = t_cls(full_params)
    return t.transform(img)

def generate_gif(example):
    name = example['name']
    params = dict(example['params'])

    print(f"Generating: {name} - {example['description']}")

    with tempfile.TemporaryDirectory() as tmpdir:
        img = load_input()

        for i in range(NUM_FRAMES):
            # save current frame
            frame_path = Path(tmpdir) / f'frame_{i:04d}.png'
            img.save(str(frame_path))

            # apply transform for next frame
            frame_params = dict(params)
            frame_params['n'] = i
            frame_params['offset'] = i
            frame_params['total_n'] = NUM_FRAMES

            # for ripple, animate the phase
            if example.get('use_phase'):
                frame_params['phase'] = i * 0.5

            # for paste_img / blend_ref, ramp opacity from 0 to 1 over the GIF
            if example.get('animate_opacity'):
                frame_params['opacity'] = i / (NUM_FRAMES - 1)

            # apply pre-transforms (e.g. slight rotation before kaleidoscope)
            for pre in example.get('pre_transforms', []):
                pre_params = dict(pre['params'])
                pre_params['n'] = i
                pre_params['offset'] = i
                pre_params['total_n'] = NUM_FRAMES
                pre_t = TRANSFORM_LIBRARY[pre['name']](pre_params)
                img = pre_t.transform(img)

            t_cls = TRANSFORM_LIBRARY[name]
            t = t_cls(frame_params)
            img = t.transform(img)

        # create gif
        output_path = OUTPUT_DIR / f'{name}.gif'
        make_gif(
            input_folder=tmpdir,
            gif_output=str(output_path),
            params={'frame_delay': '100', 'max_dim': '512', 'bounce': '1'}
        )
        print(f"  Saved: {output_path}")

def generate_single_frames(example):
    """For transforms like mirror/kaleidoscope, save variant images instead."""
    variants = example['single_frame_variants']
    for variant_name, variant_params in variants.items():
        img = load_input()
        full_params = dict(variant_params)
        full_params['n'] = 0
        full_params['offset'] = 0
        full_params['total_n'] = 1

        t_cls = TRANSFORM_LIBRARY[example['name']]
        t = t_cls(full_params)
        result = t.transform(img)

        output_path = OUTPUT_DIR / f'{variant_name}.png'
        result.save(str(output_path))
        print(f"  Saved: {output_path}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # verify we cover every transform
    covered = {e['name'] for e in EXAMPLES}
    all_transforms = set(TRANSFORM_LIBRARY.keys())
    missing = all_transforms - covered
    if missing:
        print(f"WARNING: No examples defined for: {', '.join(sorted(missing))}")

    for example in EXAMPLES:
        if example.get('single_frame_variants'):
            print(f"Generating variants: {example['name']} - {example['description']}")
            generate_single_frames(example)

        # always generate the GIF too
        generate_gif(example)

    print(f"\nDone! All examples saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
