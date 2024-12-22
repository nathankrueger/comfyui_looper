from os.path import abspath, dirname
import sys
from typing import Any
import argparse
from moviepy import ImageSequenceClip, AudioFileClip
import glob
from PIL import Image, ImageOps
from PIL.ImageFile import ImageFile

try:
    from utils.util import parse_params
except ModuleNotFoundError:
    SCRIPT_DIR = dirname(abspath(__file__))
    sys.path.append(dirname(dirname(SCRIPT_DIR)))
    from comfyui_looper.utils.util import parse_params

IMG_TYPE = '.png'
DEFAULT_FRAME_DELAY_MS = 250
DEFAULT_MAX_DIM = 768
DEFAULT_VIDEO_BITRATE = '4000k'

def get_animation_param_value(param_name: str, animation_params: dict[str, str]) -> Any | None:
    if param_name in animation_params:
        return animation_params[param_name]

    match param_name:
        case 'max_dim':
            return DEFAULT_MAX_DIM
        case 'frame_delay':
            return DEFAULT_FRAME_DELAY_MS
        case 'v_bitrate':
            return DEFAULT_VIDEO_BITRATE
        case _:
            return None

def get_image_paths(input_folder: str, params: dict[str, str] = None) -> list[str]:
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

def get_frames(input_folder: str, max_dim: int, params: dict[str, str] = None) -> list[ImageFile]:
    frame_paths = get_image_paths(input_folder=input_folder, params=params)
    frames = [ImageOps.exif_transpose(Image.open(img_path)) for img_path in frame_paths]

    # resize as needed
    if max_dim > 0:
        for frame in frames: frame.thumbnail((max_dim, max_dim))
    
    return frames

def make_gif(input_folder: str, gif_output: str, params: dict[str, str] = None):
    # parse params
    max_dim = int(get_animation_param_value('max_dim', params))
    frame_delay = int(get_animation_param_value('frame_delay', params))

    # find all images
    frames = get_frames(input_folder=input_folder, max_dim=max_dim, params=params)

    # convert colors see: https://github.com/python-pillow/Pillow/issues/6832
    frames = [frame.convert('RGBA') for frame in frames]
    #frames = [frame.convert('P', palette=Image.Palette.ADAPTIVE) for frame in frames]
    
    # create the GIF animation
    frame_one = frames[0]
    frame_one.save(
        gif_output,
        format='gif',
        append_images=frames,
        save_all=True,
        duration=frame_delay,
        loop=0,
        lossless=True,
        optimize=False
    )

def make_mp4(input_folder: str, mp4_output: str, params: dict[str, str] = None):
    # parse params
    v_bitrate = get_animation_param_value('v_bitrate', params)
    frame_delay = int(get_animation_param_value('frame_delay', params))
    fps = int(float(1) / float(frame_delay / 1000.0))

    # find all images
    frames = get_image_paths(input_folder=input_folder, params=params)

    # create the video clip
    video_clip = ImageSequenceClip(frames, fps=fps)

    # add sound to it if requested
    if 'mp3_file' in params:
        audio_clip = AudioFileClip(params['mp3_file'])
        audio_clip = audio_clip.subclipped(0, video_clip.duration)
        video_clip = video_clip.with_audio(audio_clip)

    # write the video file
    video_clip.write_videofile(mp4_output, codec='libx264', bitrate=v_bitrate)

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