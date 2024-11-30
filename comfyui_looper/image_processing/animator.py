from os.path import abspath, dirname
import sys
from typing import Any
import argparse
from moviepy import ImageSequenceClip
import glob
from PIL import Image, ImageOps

try:
    SCRIPT_DIR = dirname(abspath(__file__))
    sys.path.append(dirname(dirname(SCRIPT_DIR)))
    from comfyui_looper.utils.util import parse_params
except ModuleNotFoundError:
    from utils.util import parse_params

IMG_TYPE = '.png'
DEFAULT_GIF_FRAME_DELAY_MS = 250
DEFAULT_MAX_DIM = 768
DEFAULT_VIDEO_BITRATE = '3000k'

def get_image_paths(input_folder: str, params: dict[str, Any] = None) -> list[str]:
    # find all images
    frame_paths = [img_path for img_path in glob.glob(f'{input_folder}/*{IMG_TYPE}')]

    if params is not None:
        if 'bounce' in params:
            bounce_frame_skip = 0
            if 'bounce_frame_skip' in params:
                bounce_frame_skip = int(params['bounce_frame_skip'])
            frame_paths.extend([frame_paths[i] for i in range(len(frame_paths)-1, -1, -(bounce_frame_skip+1))])
            if frame_paths[0] != frame_paths[-1]:
                frame_paths.append(frame_paths[0])
    
    return frame_paths

def get_frames(input_folder: str, max_dim: int, params: dict[str, Any] = None) -> list[Image]:
    frame_paths = get_image_paths(input_folder=input_folder, params=params)
    frames = [ImageOps.exif_transpose(Image.open(img_path)) for img_path in frame_paths]

    # resize as needed
    if max_dim > 0:
        for frame in frames: frame.thumbnail((max_dim, max_dim))
    
    return frames

def make_gif(input_folder: str, gif_output: str, params: dict[str, Any] = None):
    max_dim = DEFAULT_MAX_DIM
    if 'max_dim' in params:
        max_dim = int(params['max_dim'])

    # find all images
    frames = get_frames(input_folder=input_folder, max_dim=max_dim, params=params)

    frame_delay = DEFAULT_GIF_FRAME_DELAY_MS
    if 'frame_delay' in params:
        frame_delay = int(params['frame_delay'])

    # convert colors see: https://github.com/python-pillow/Pillow/issues/6832
    frames = [frame.convert('RGBA') for frame in frames]
    #frames = [frame.convert('P', palette=Image.Palette.ADAPTIVE) for frame in frames]
    
    # create the GIF animation
    frame_one = frames[0]
    frame_one.save(
        gif_output,
        format="gif",
        append_images=frames,
        save_all=True,
        duration=frame_delay,
        loop=0,
        lossless=True,
        optimize=False
    )

def make_mp4(input_folder: str, mp4_output: str, params: dict[str, Any] = None):
    # find all images
    frames = get_image_paths(input_folder=input_folder, params=params)

    frame_delay = DEFAULT_GIF_FRAME_DELAY_MS
    if 'frame_delay' in params:
        frame_delay = int(params['frame_delay'])
    fps = int(float(1) / float(frame_delay / 1000.0))

    v_bitrate = DEFAULT_VIDEO_BITRATE
    if 'v_bitrate' in params:
        v_bitrate = params['v_bitrate']
    
    # Create the video clip
    clip = ImageSequenceClip(frames, fps=fps)
    
    # Write the video file
    clip.write_videofile(mp4_output, codec='libx264', bitrate=v_bitrate)

def make_animation(type: str, input_folder: str, output_animation: str, params: dict[str, Any] = None):
    if type == 'gif':
        make_gif(
            input_folder=input_folder,
            gif_output=output_animation,
            params=params
        )
    elif type == 'mp4':
        make_mp4(
            input_folder=input_folder,
            mp4_output=output_animation,
            params=params
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images')
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    parser.add_argument('-t', '--type', type=str, default='gif', choices=['gif', 'mp4'])
    parser.add_argument('-x', '--param', action='append', dest='params')
    args = parser.parse_args()
    animation_params = parse_params(args.params)

    make_animation(
        type=args.type,
        input_folder=args.input_dir,
        output_animation=args.output_file,
        params=animation_params
    )