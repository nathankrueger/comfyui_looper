import sys
import os
import shutil
from pathlib import Path
import argparse
import tqdm
import torchvision.transforms as T

ppath = str(os.path.realpath(Path(os.path.dirname(__file__)) / ".." / "comfyui_looper"))
sys.path.append(ppath)

import transforms
import gif_maker

TRANSFORMS_TO_TEST = [
    # {
    #     'name': 'fisheye',
    #     'strength': 0.175
    # },
    # {
    #     'name': 'zoom_in',
    #     'zoom_amt': 0.075
    # }
    # {
    #     'name': 'rotate',
    #     'angle': 1
    # }
    {
        'name': 'squeeze_tall',
        'squeeze_amt': 0.05
    }
]

def get_filename_for_idx(idx: int, output_dir: str) -> str:
    return os.path.join(output_dir, f"test_file_{idx:06}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform tester')
    parser.add_argument('-i', '--input_img', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-n', '--loops', type=int, default=15)
    args=parser.parse_args()

    transforms.Transform.validate_transformation_params(TRANSFORMS_TO_TEST)

    os.makedirs(args.output_folder, exist_ok=True)
    shutil.copy(args.input_img, get_filename_for_idx(0, args.output_folder))

    for idx in tqdm.tqdm(range(args.loops)):
        curr_img_path = get_filename_for_idx(idx, args.output_folder)
        next_img_path = get_filename_for_idx(idx+1, args.output_folder)
        torch_tensor = transforms.load_image_with_transforms(curr_img_path, TRANSFORMS_TO_TEST)

        torch_tensor = torch_tensor.squeeze(0)
        torch_tensor = torch_tensor.permute(2, 0, 1)
        img = T.ToPILImage()(torch_tensor)
        img.save(next_img_path, pnginfo=None, compress_level=5)

    gif_maker.make_gif(
        input_folder=args.output_folder,
        frame_delay=200,
        max_dimension=768,
        gif_output=os.path.join(args.output_folder, "test.gif")
    )