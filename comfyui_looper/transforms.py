from typing import Any
from PIL import Image, ImageOps
import numpy as np
import torch

import util

class Transform:
    NAME = None

    def __init__(self, params: dict[str, Any]):
        self.params = params

    @classmethod
    def get_name(cls) -> str:
        return cls.NAME
    
    def transform(self, img: Image) -> Image:
        # override me
        return None
    
    @staticmethod
    def apply_transformations(img: Image, transforms: list[dict[str, Any]]) -> Image:
        curr_img = img
        for tdict in transforms:
            tdict = dict(tdict)
            t_name = tdict.pop('name')
            t_params = tdict
            t = TRANSFORM_LIBRARY[t_name](t_params)
            curr_img = t.transform(curr_img)

        return curr_img

class ZoomInTransform(Transform):
    NAME = 'zoom_in'

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']

        mod_width = int(zoom_amt * float(init_width))
        mod_height = int(zoom_amt * float(init_height))

        left = (init_width - mod_width) // 2
        right = init_width - left
        top = (init_height - mod_height) // 2
        bottom = init_height - top

        cropped = img.crop((left, top, right, bottom))
        result = cropped.resize((init_width, init_height))
        return result

class ZoomInLeftTransform(Transform):
    NAME = 'zoom_in_left'

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']

        mod_width = int(zoom_amt * float(init_width))

        left = init_width - mod_width
        right = init_width
        top = 0
        bottom = init_height

        cropped = img.crop((left, top, right, bottom))
        result = cropped.resize((init_width, init_height))
        return result
    
class ZoomInRightTransform(Transform):
    NAME = 'zoom_in_right'

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']

        mod_width = int(zoom_amt * float(init_width))

        left = 0
        right = mod_width
        top = 0
        bottom = init_height

        cropped = img.crop((left, top, right, bottom))
        result = cropped.resize((init_width, init_height))
        return result
    
class ZoomInUpTransform(Transform):
    NAME = 'zoom_in_up'

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']

        mod_height = int(zoom_amt * float(init_height))

        left = 0
        right = init_width
        top = init_height - mod_height
        bottom = init_height

        cropped = img.crop((left, top, right, bottom))
        result = cropped.resize((init_width, init_height))
        return result
    
class ZoomInDownTransform(Transform):
    NAME = 'zoom_in_down'

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']

        mod_height = int(zoom_amt * float(init_height))

        left = 0
        right = init_width
        top = 0
        bottom = mod_height

        cropped = img.crop((left, top, right, bottom))
        result = cropped.resize((init_width, init_height))
        return result

TRANSFORM_LIBRARY: dict[str, Transform] = {t.get_name(): t for t in util.all_subclasses(Transform)}

def load_image_with_transforms(image_path: str, transforms: dict[str, Any]):
    i = Image.open(image_path)

    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")

    if transforms is not None:
        image = Transform.apply_transformations(image, transforms)

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image