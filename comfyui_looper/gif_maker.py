import argparse
import glob
import os
from PIL import Image

IMG_TYPE = '.png'

def make_gif(input_folder: str, gif_output: str, frame_delay: int, max_dimension: int):
    # find all images
    frames = [Image.open(image_path) for image_path in glob.glob(f'{input_folder}/*{IMG_TYPE}')]

    # resize as needed
    if max_dimension > 0:
        for frame in frames: frame.thumbnail((max_dimension, max_dimension))

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
    args=parser.parse_args()

    make_gif(
        input_folder=args.input_dir,
        gif_output=args.output_gif,
        frame_delay=args.frame_delay,
        max_dimension=args.max_dimension
    )

if __name__ == "__main__":
    main()