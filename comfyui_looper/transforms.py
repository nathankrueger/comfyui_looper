from typing import Any
from PIL import Image, ImageOps
import numpy as np
import torch
import cv2

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

        mod_width = int((1.0 - zoom_amt) * float(init_width))
        mod_height = int((1.0 - zoom_amt) * float(init_height))

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

        mod_width = int((1.0 - zoom_amt) * float(init_width))

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

        mod_width = int((1.0 - zoom_amt) * float(init_width))

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

        mod_height = int((1.0 - zoom_amt) * float(init_height))

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

        mod_height = int((1.0 - zoom_amt) * float(init_height))

        left = 0
        right = init_width
        top = 0
        bottom = mod_height

        cropped = img.crop((left, top, right, bottom))
        result = cropped.resize((init_width, init_height))
        return result
    
class SqueezeWideTransform(Transform):
    NAME = 'squeeze_wide'

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        squeeze_amt: float = self.params['squeeze_amt']

        mod_width = int((1.0 - squeeze_amt) * float(init_width))

        left = (init_width - mod_width) // 2
        right = init_width - left
        top = 0
        bottom = init_height

        cropped = img.crop((left, top, right, bottom))
        result = cropped.resize((init_width, init_height))
        return result

class SqueezeTallTransform(Transform):
    NAME = 'squeeze_tall'

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        squeeze_amt: float = self.params['squeeze_amt']

        mod_height = int((1.0 - squeeze_amt) * float(init_height))

        left = 0
        right = init_width
        top = (init_height - mod_height) // 2
        bottom = init_height - top

        cropped = img.crop((left, top, right, bottom))
        result = cropped.resize((init_width, init_height))
        return result

class FisheyeTransform(Transform):
    NAME = 'fisheye'

    def transform(self, img: Image) -> Image:
        width, height = img.size
        strength: float = self.params['strength']

        center_x, center_y = width // 2, height // 2

        # Create a meshgrid for the image coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Calculate the distance from the center of the image
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Apply the fisheye distortion
        theta = np.arctan2(y - center_y, x - center_x)
        r = radius * (1 + strength * radius / np.sqrt(width**2 + height**2))

        # Calculate the new coordinates
        x_new = center_x + r * np.cos(theta)
        y_new = center_y + r * np.sin(theta)

        # Remap the image using the new coordinates
        fisheye_img = cv2.remap(np.array(img), x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)
        return Image.fromarray(fisheye_img)

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
