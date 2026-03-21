from os.path import abspath, dirname
import sys
import re
import tempfile
from typing import Any, Callable, Optional
import argparse
from moviepy import ImageSequenceClip, AudioFileClip
from proglog import ProgressBarLogger
import glob
from PIL import Image, ImageOps
from PIL.ImageFile import ImageFile
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

ProgressCallback = Optional[Callable[[float, str], None]]


class _ExportProgressLogger(ProgressBarLogger):
    """Captures MoviePy frame encoding progress and forwards to a callback."""

    def __init__(self, callback: Callable[[float, str], None], loading_pct: float = 0.2):
        super().__init__()
        self._callback = callback
        self._loading_pct = loading_pct  # fraction reserved for loading phase
        self._total = 0

    def bars_callback(self, bar, attr, value, old_value=None):
        if attr == 'total' and value:
            self._total = value
        elif attr == 'index' and self._total > 0:
            frac = self._loading_pct + (value / self._total) * (1.0 - self._loading_pct)
            self._callback(min(frac, 1.0), 'encoding')

    def callback(self, **kw):
        pass

try:
    from utils.util import parse_params
    from utils.fft import WaveFile
except ModuleNotFoundError:
    SCRIPT_DIR = dirname(abspath(__file__))
    sys.path.append(dirname(dirname(SCRIPT_DIR)))
    from comfyui_looper.utils.util import parse_params
    from comfyui_looper.utils.fft import WaveFile

IMG_TYPE = '.png'
DEFAULT_FRAME_DELAY_MS = 250
DEFAULT_MAX_DIM = 768
DEFAULT_VIDEO_BITRATE = '4000k'

def parse_tuples(s: str) -> list[tuple[int, int]]:
    result = []
    if s.startswith('[') and s.endswith(']'):
        pattern = r"\(([^,]+),([^,]+)\)"
        matches = re.findall(pattern, s)
        result = [tuple([int(match[0].strip()), int(match[1].strip())]) for match in matches]
    return result

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
    frame_paths = sorted([img_path for img_path in glob.glob(f'{input_folder}/*{IMG_TYPE}')])

    # apply frame range if specified
    if params is not None:
        start_frame = int(params.get('start_frame', 0))
        end_frame = int(params.get('end_frame', -1))
        if start_frame > 0 or end_frame >= 0:
            end_idx = (end_frame + 1) if end_frame >= 0 else len(frame_paths)
            frame_paths = frame_paths[start_frame:end_idx]

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

def make_gif(input_folder: str, gif_output: str, params: dict[str, str] = None,
             progress_callback: ProgressCallback = None):
    max_dim = int(get_animation_param_value('max_dim', params))
    frame_delay = int(get_animation_param_value('frame_delay', params))

    frame_paths = get_image_paths(input_folder=input_folder, params=params)
    total = len(frame_paths)
    if progress_callback:
        progress_callback(0.0, 'loading')

    # load and resize frames
    frames = []
    for i, img_path in enumerate(frame_paths):
        frame = ImageOps.exif_transpose(Image.open(img_path))
        if max_dim > 0:
            frame.thumbnail((max_dim, max_dim))
        frames.append(frame)
        if progress_callback:
            progress_callback((i + 1) / total * 0.5, 'loading')

    # convert colors see: https://github.com/python-pillow/Pillow/issues/6832
    # convert in-place to avoid holding two full frame lists simultaneously
    for i, frame in enumerate(frames):
        frames[i] = frame.convert('RGBA')
        frame.close()
        if progress_callback:
            progress_callback(0.5 + (i + 1) / total * 0.3, 'converting')

    if progress_callback:
        progress_callback(0.8, 'writing')

    # create the GIF animation
    try:
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
    finally:
        for frame in frames:
            frame.close()
        frames.clear()

    if progress_callback:
        progress_callback(1.0, 'done')

def make_mp4(input_folder: str, mp4_output: str, params: dict[str, str] = None,
             progress_callback: ProgressCallback = None):
    v_bitrate = get_animation_param_value('v_bitrate', params)
    frame_delay = int(get_animation_param_value('frame_delay', params))
    fps = int(float(1) / float(frame_delay / 1000.0))

    if progress_callback:
        progress_callback(0.0, 'loading')

    frames = get_image_paths(input_folder=input_folder, params=params)
    video_clip = ImageSequenceClip(frames, fps=fps)
    audio_clip = None

    if progress_callback:
        progress_callback(0.2, 'encoding')

    logger = _ExportProgressLogger(progress_callback) if progress_callback else 'bar'

    try:
        if 'mp3_file' in params:
            audio_clip = AudioFileClip(params['mp3_file'])
            audio_clip = audio_clip.subclipped(0, video_clip.duration)
            video_clip = video_clip.with_audio(audio_clip)

        video_clip.write_videofile(mp4_output, codec='libx264', bitrate=v_bitrate, logger=logger)
    finally:
        video_clip.close()
        if audio_clip is not None:
            audio_clip.close()

    if progress_callback:
        progress_callback(1.0, 'done')

def make_fft_animation(mp4_output: str, params: dict[str, str] = None):
    assert 'mp3_file' in params
    assert 'len_seconds' in params
    assert 'freq_ranges' in params

    total_seconds = float(get_animation_param_value('len_seconds', params))
    frame_delay = int(get_animation_param_value('frame_delay', params))
    freq_range_tuples = parse_tuples(params['freq_ranges'])
    total_frames = int(total_seconds / (frame_delay / 1000.0))
    wf: WaveFile = WaveFile.get_wavefile(params['mp3_file'], total_seconds)

    try:
        temp_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        for i in tqdm(range(total_frames)):
            start_percent = (i / total_frames) * 100.0
            end_percent = ((i + 1) / total_frames) * 100.0
            pow_at_frange = wf.get_power_in_freq_ranges(start_percent, end_percent, freq_range_tuples)
            plt.cla()
            plt.stem(list(pow_at_frange))
            plt.ylim(0, 500.0)
            plt.savefig(str(Path(temp_directory.name) / f'img{i}.png'))
        make_mp4(input_folder=temp_directory.name, mp4_output=mp4_output, params=params)
            
    finally:
        temp_directory.cleanup()

def make_animation(type: str, input_folder: str, output_animation: str, params: dict[str, Any] = None,
                   progress_callback: ProgressCallback = None):
    if type == 'gif':
        make_gif(
            input_folder=input_folder,
            gif_output=output_animation,
            params=params,
            progress_callback=progress_callback
        )
    elif type == 'mp4':
        make_mp4(
            input_folder=input_folder,
            mp4_output=output_animation,
            params=params,
            progress_callback=progress_callback
        )
    elif type == 'fft_test':
        make_fft_animation(
            mp4_output=output_animation,
            params=params
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images')
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    parser.add_argument('-t', '--type', type=str, default='gif', choices=['gif', 'mp4', 'fft_test'])
    parser.add_argument('-x', '--param', action='append', dest='params')
    args = parser.parse_args()
    animation_params = parse_params(args.params)

    make_animation(
        type=args.type,
        input_folder=args.input_dir,
        output_animation=args.output_file,
        params=animation_params
    )