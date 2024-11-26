from typing import Any
from PIL import Image, ImageOps
import numpy as np
import math
import torch
import cv2

import util

class Transform:
    NAME = None
    REQUIRED_PARAMS = None

    def __init__(self, params: dict[str, Any]):
        self.params = params

    @classmethod
    def get_name(cls) -> str:
        return cls.NAME
    
    @classmethod
    def get_required_params(cls) -> set[str]:
        return cls.REQUIRED_PARAMS
    
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
    
    @staticmethod
    def validate_transformation_params(transforms: list[dict[str, Any]]):
        for tdict in transforms:
            tdict = dict(tdict)
            if len(tdict) == 0:
                continue

            assert 'name' in tdict
            t_name = tdict.pop('name')
            assert t_name in TRANSFORM_LIBRARY
            tdict_params = tdict
            assert set(tdict_params.keys()).issuperset(TRANSFORM_LIBRARY[t_name].get_required_params())
            for key in tdict_params:
                if key not in TRANSFORM_LIBRARY[t_name].get_required_params():
                    print(f"Warning, ignored transform param: {key} for transform {t_name}")

class ZoomInTransform(Transform):
    NAME = 'zoom_in'
    REQUIRED_PARAMS = {'zoom_amt'}

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
    REQUIRED_PARAMS = {'zoom_amt'}

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
    REQUIRED_PARAMS = {'zoom_amt'}

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
    REQUIRED_PARAMS = {'zoom_amt'}

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
    REQUIRED_PARAMS = {'zoom_amt'}

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
    REQUIRED_PARAMS = {'squeeze_amt'}

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
    REQUIRED_PARAMS = {'squeeze_amt'}

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
    REQUIRED_PARAMS = {'strength'}

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

# algo stolen from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
class RotateTransform(Transform):
    NAME = 'rotate'
    REQUIRED_PARAMS = {'angle'}

    def rotate(self, image: Image, angle: float):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """
        image = np.asarray(image)

        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result

    def largest_rotated_rect(self, w: int, h: int, angle: float):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

        Converted to Python by Aaron Snoswell
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    def crop_around_center(self, image, width: int, height: int):
        """
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around it's centre point
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if(width > image_size[0]):
            width = image_size[0]

        if(height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    def transform(self, img: Image) -> Image:
        image_width, image_height = img.size
        rotate_angle = self.params['angle']
        image_rotated = self.rotate(img, rotate_angle)
        image_rotated_cropped = self.crop_around_center(
            image_rotated,
            *self.largest_rotated_rect(
                image_width,
                image_height,
                math.radians(rotate_angle)
            )
        )

        # resize back to initial size
        resized_result = Image.fromarray(image_rotated_cropped).resize((image_width, image_height))
        return resized_result

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
