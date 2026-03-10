import os
import math
from datetime import datetime
from PIL import Image, ImageOps

def get_loop_img_filename(idx: int) -> str:
    return f"loop_img_{idx:06}.png"

def get_log_filename(log_basename: str) -> str:
    dt = datetime.now()
    dt_str = dt.strftime("%Y_%m_%d__%H_%M_%S")

    log_ext = log_basename.split('.')[-1] if '.' in log_basename else ""
    log_basename_no_ext = '.'.join(log_basename.split('.')[:-1])

    return f"{log_basename_no_ext}_{dt_str}.{log_ext}"

def all_subclasses(cls) -> set:
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])

def resize_image(input_path: str, output_path: str, max_dim: int):
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    width, height = img.size
    
    if width > height:
        ratio = max_dim / width
        img = img.resize((max_dim, int(height * ratio)))
    else:
        ratio = max_dim / height
        img = img.resize((int(width * ratio), max_dim))

    img.save(output_path, pnginfo=None, compress_level=0)

def resize_image_match_area(input_path: str, output_path: str, area: int, modulo: int | None):
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    width, height = img.size

    w2h_ratio = float(width) / float(height)
    new_h = int(math.sqrt(float(area) / w2h_ratio))
    new_w = int(float(area) / float(new_h))

    if modulo is not None:
        new_h = new_h - (new_h % modulo)
        new_w = new_w - (new_w % modulo)
    
    img = img.resize((new_w, new_h))
    img.save(output_path, pnginfo=None, compress_level=0)

def parse_params(params_list: list[str]) -> dict[str, str]:
    params = {}
    if params_list is not None:
        for param in params_list:
            param_key = param.split(':')[0]
            param_val = param.split(':')[1] if ':' in param else None
            params[param_key] = param_val

    return params