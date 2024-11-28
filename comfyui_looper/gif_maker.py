from typing import Any
import argparse
import glob
from PIL import Image, ImageOps

IMG_TYPE = '.png'

def make_gif(input_folder: str, gif_output: str, frame_delay: int, max_dim: int, params: dict[str, Any] = None):
    # find all images
    frame_paths = [img_path for img_path in glob.glob(f'{input_folder}/*{IMG_TYPE}')]

    if params is not None:
        if 'bounce' in params:
            bounce_frame_skip = 0
            if 'bounce_frame_skip' in params:
                bounce_frame_skip = int(params['bounce_frame_skip'])
            frame_paths.extend([frame_paths[i] for i in range(len(frame_paths)-1, -1, -(bounce_frame_skip+1))])

    frames = [ImageOps.exif_transpose(Image.open(img_path)) for img_path in frame_paths]

    # resize as needed
    if max_dim > 0:
        for frame in frames: frame.thumbnail((max_dim, max_dim))

    # convert colors see: https://github.com/python-pillow/Pillow/issues/6832
    frames = [frame.convert('RGBA') for frame in frames]
    #frames = [frame.convert('P', palette=Image.Palette.ADAPTIVE) for frame in frames]
    
    # create the GIF animation
    frame_one = frames[0]
    frame_one.save(gif_output,
                   format="gif",
                   append_images=frames,
                   save_all=True,
                   duration=frame_delay,
                   loop=0,
                   lossless=True,
                   optimize=False
                   )

def main():
    parser = argparse.ArgumentParser(description='Resize images')
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_gif', type=str, required=True)
    parser.add_argument('-d', '--frame_delay', type=int, default=250)
    parser.add_argument('-s', '--max_dimension', type=int, default=0)
    parser.add_argument('-x', '--param', action='append', dest='params')
    args = parser.parse_args()

    params = {}
    if args.params is not None:
        for param in args.params:
            param_key = param.split(':')[0]
            param_val = param.split(':')[1] if ':' in param else None
            params[param_key] = param_val

    make_gif(
        input_folder=args.input_dir,
        gif_output=args.output_gif,
        frame_delay=args.frame_delay,
        max_dim=args.max_dimension,
        params=params
    )

if __name__ == "__main__":
    main()