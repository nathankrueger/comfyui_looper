from os.path import abspath, dirname, isabs, isfile, join
from pathlib import Path
import os
import sys
from typing import Any
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from dataclasses import dataclass
import numpy as np
import math
import cv2

try:
    from utils.util import all_subclasses
    from utils.simple_expr_eval import SimpleExprEval
    from utils.fft import WaveFile
except ModuleNotFoundError:
    SCRIPT_DIR = dirname(abspath(__file__))
    sys.path.append(dirname(dirname(SCRIPT_DIR)))
    from comfyui_looper.utils.util import all_subclasses
    from comfyui_looper.utils.simple_expr_eval import SimpleExprEval
    from comfyui_looper.utils.fft import WaveFile

AUTOMATIC_SEQUENCE_PARAMS = {
    'n',
    'offset',
    'total_n'
}

@dataclass
class AutomaticTransformParams:
    n: int
    offset: int
    total_n: int
    wavefile: WaveFile

class Transform:
    NAME = None
    REQUIRED_PARAMS = None
    EVAL_PARAMS = set()
    OPTIONAL_PARAMS = set()

    def __init__(self, params: dict[str, Any]):
        self.params = params

    @classmethod
    def get_name(cls) -> str:
        return cls.NAME

    @classmethod
    def get_required_params(cls) -> set[str]:
        return cls.REQUIRED_PARAMS

    @classmethod
    def get_eval_params(cls) -> set[str]:
        return cls.EVAL_PARAMS

    @classmethod
    def get_optional_params(cls) -> set[str]:
        return cls.OPTIONAL_PARAMS
    
    def transform(self, img: Image) -> Image:
        # override me
        return img
    
    @staticmethod
    def get_transform(name: str):
        return TRANSFORM_LIBRARY[name]
    
    @staticmethod
    def apply_transformations(img: Image, transforms: list[dict[str, Any]], auto_params: AutomaticTransformParams) -> Image:
        curr_img = img
        for tdict in transforms:
            tdict = dict(tdict)
            t_name = tdict.pop('name')
            t_params = tdict

            # automatic params partaining to where we are in the sequence
            # these are intended to be used by the Python transform code itself,
            # not expressions passed in, e.g. *not* for zoom_amt = sin(n)
            t_params['n'] = auto_params.n
            t_params['offset'] = auto_params.offset
            t_params['total_n'] = auto_params.total_n

            t = TRANSFORM_LIBRARY[t_name](t_params)
            curr_img = t.transform(curr_img)

        return curr_img
    
    @staticmethod
    def validate_transformation_params(transforms: list[dict[str, Any]]):
        for tdict in transforms:
            tdict = dict(tdict)
            if len(tdict) == 0:
                continue

            assert 'name' in tdict, f"Transform missing 'name' field: {tdict}"
            t_name = tdict.pop('name')
            assert t_name in TRANSFORM_LIBRARY, f"Unknown transform '{t_name}'. Available: {sorted(TRANSFORM_LIBRARY.keys())}"
            tdict_params = tdict
            required = TRANSFORM_LIBRARY[t_name].get_required_params()
            missing = required - set(tdict_params.keys())
            assert not missing, f"Transform '{t_name}' missing required params: {missing}"
            for key in tdict_params:
                assert key not in AUTOMATIC_SEQUENCE_PARAMS, f"Transform '{t_name}' uses reserved param '{key}'"
                all_known_params = TRANSFORM_LIBRARY[t_name].get_eval_params() | TRANSFORM_LIBRARY[t_name].get_required_params() | TRANSFORM_LIBRARY[t_name].get_optional_params()
                if key not in all_known_params:
                    print(f"Warning, ignored transform param: {key} for transform {t_name}")

class ZoomInTransform(Transform):
    NAME = 'zoom_in'
    REQUIRED_PARAMS = {'zoom_amt'}
    EVAL_PARAMS = {'zoom_amt'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']
        if zoom_amt == 0.0:
            return img

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
    EVAL_PARAMS = {'zoom_amt'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']
        if zoom_amt == 0.0:
            return img

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
    EVAL_PARAMS = {'zoom_amt'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']
        if zoom_amt == 0.0:
            return img

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
    EVAL_PARAMS = {'zoom_amt'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']
        if zoom_amt == 0.0:
            return img

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
    EVAL_PARAMS = {'zoom_amt'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']
        if zoom_amt == 0.0:
            return img

        mod_height = int((1.0 - zoom_amt) * float(init_height))

        left = 0
        right = init_width
        top = 0
        bottom = mod_height

        cropped = img.crop((left, top, right, bottom))
        result = cropped.resize((init_width, init_height))
        return result

class ZoomOutTransform(Transform):
    NAME = 'zoom_out'
    REQUIRED_PARAMS = {'zoom_amt'}
    EVAL_PARAMS = {'zoom_amt'}
    OPTIONAL_PARAMS = {'fill_mode'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        zoom_amt: float = self.params['zoom_amt']
        fill_mode: str = self.params.get('fill_mode', 'reflect')
        if zoom_amt == 0.0:
            return img

        new_width = int((1.0 - zoom_amt) * init_width)
        new_height = int((1.0 - zoom_amt) * init_height)
        shrunk = img.resize((new_width, new_height), Image.LANCZOS)

        pad_left = (init_width - new_width) // 2
        pad_top = (init_height - new_height) // 2

        if fill_mode == 'blur':
            blur_radius = max(init_width, init_height) * zoom_amt * 0.5
            result = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            result.paste(shrunk, (pad_left, pad_top))
            return result
        else:
            # reflect mode (default)
            arr = np.array(shrunk)
            pad_bottom = init_height - new_height - pad_top
            pad_right = init_width - new_width - pad_left
            padded = np.pad(arr, (
                (pad_top, pad_bottom),
                (pad_left, pad_right),
                (0, 0)
            ), mode='reflect')
            return Image.fromarray(padded)

class FoldVerticalTransform(Transform):
    NAME = 'fold_vertical'
    REQUIRED_PARAMS = {'fold_amt'}
    EVAL_PARAMS = {'fold_amt'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        img_middle = init_width // 2
        fold_amt: int = int(self.params['fold_amt'] // 2)

        # crop into two pieces, removing some in the middle
        top = 0
        bottom = init_height

        left = 0
        right = img_middle - fold_amt
        l_image = img.crop((left, top, right, bottom))

        left = img_middle + fold_amt
        right = init_width
        r_image = img.crop((left, top, right, bottom))

        # attach images together
        combined_img = Image.new('RGB', (init_width - (2 * fold_amt), init_height))
        combined_img.paste(im=l_image, box=(0, 0))
        combined_img.paste(im=r_image, box=(img_middle - fold_amt, 0))

        # resize to normal size
        result = combined_img.resize((init_width, init_height))
        return result
    
class FoldHorizontalTransform(Transform):
    NAME = 'fold_horizontal'
    REQUIRED_PARAMS = {'fold_amt'}
    EVAL_PARAMS = {'fold_amt'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        img_middle = init_height // 2
        fold_amt: int = int(self.params['fold_amt'] // 2)

        # crop into two pieces, removing some in the middle
        left = 0
        right = init_width

        top = 0
        bottom = img_middle - fold_amt
        top_image = img.crop((left, top, right, bottom))

        top = img_middle + fold_amt
        bottom = init_height
        bot_image = img.crop((left, top, right, bottom))

        # attach images together
        combined_img = Image.new('RGB', (init_width, init_height - (2 * fold_amt)))
        combined_img.paste(im=top_image, box=(0, 0))
        combined_img.paste(im=bot_image, box=(0, img_middle - fold_amt))

        # resize to normal size
        result = combined_img.resize((init_width, init_height))
        return result

class SqueezeWideTransform(Transform):
    NAME = 'squeeze_wide'
    REQUIRED_PARAMS = {'squeeze_amt'}
    EVAL_PARAMS = {'squeeze_amt'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        squeeze_amt: float = self.params['squeeze_amt']
        if squeeze_amt == 0.0:
            return img

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
    EVAL_PARAMS = {'squeeze_amt'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        squeeze_amt: float = self.params['squeeze_amt']
        if squeeze_amt == 0.0:
            return img

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
    EVAL_PARAMS = {'strength'}

    def transform(self, img: Image) -> Image:
        width, height = img.size
        strength: float = self.params['strength']
        if strength == 0.0:
            return img

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

# algorithm taken from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
class RotateTransform(Transform):
    NAME = 'rotate'
    REQUIRED_PARAMS = {'angle'}
    EVAL_PARAMS = {'angle'}

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
        if rotate_angle == 0.0:
            return img

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

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

def _resolve_img_path(img_path: str) -> str:
    """Resolve img_path: absolute paths used as-is, relative paths resolved against project root."""
    if isabs(img_path):
        return img_path
    resolved = join(_PROJECT_ROOT, img_path)
    if isfile(resolved):
        return resolved
    return img_path

class PasteImageTransform(Transform):
    NAME = 'paste_img'
    REQUIRED_PARAMS = {'img_path', 'opacity'}
    EVAL_PARAMS = {'opacity'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        opacity: float = self.params['opacity']
        paste_img_path = _resolve_img_path(self.params['img_path'])

        paste_img = Image.open(paste_img_path).convert('RGB')
        paste_img = paste_img.resize((init_width, init_height))
        
        result = Image.blend(img, paste_img, opacity)
        return result

def _apply_easing(opacity: float, easing: str) -> float:
    t = max(0.0, min(1.0, opacity))
    match easing:
        case 'linear':
            return t
        case 'ease_in':
            return t * t
        case 'ease_out':
            return 1.0 - (1.0 - t) * (1.0 - t)
        case 'ease_in_out':
            return t * t * (3.0 - 2.0 * t)
        case _:
            raise ValueError(f"Unknown easing function: {easing}")

def _generate_mask(width: int, height: int, mask_type: str, invert: bool) -> np.ndarray:
    match mask_type:
        case 'uniform':
            mask = np.ones((height, width), dtype=np.float32)
        case 'radial':
            cx, cy = width / 2.0, height / 2.0
            max_radius = math.sqrt(cx**2 + cy**2)
            y, x = np.mgrid[0:height, 0:width]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask = (1.0 - dist / max_radius).astype(np.float32)
            mask = np.clip(mask, 0.0, 1.0)
        case 'horizontal':
            mask = np.linspace(0.0, 1.0, width, dtype=np.float32)
            mask = np.broadcast_to(mask[np.newaxis, :], (height, width)).copy()
        case 'vertical':
            mask = np.linspace(0.0, 1.0, height, dtype=np.float32)
            mask = np.broadcast_to(mask[:, np.newaxis], (height, width)).copy()
        case _:
            raise ValueError(f"Unknown mask type: {mask_type}")

    if invert:
        mask = 1.0 - mask

    return mask

def _blend_pixels(base: np.ndarray, ref: np.ndarray, blend_mode: str) -> np.ndarray:
    match blend_mode:
        case 'normal':
            return ref
        case 'overlay':
            low = 2.0 * base * ref
            high = 1.0 - 2.0 * (1.0 - base) * (1.0 - ref)
            return np.where(base < 0.5, low, high)
        case 'soft_light':
            return (1.0 - 2.0 * ref) * (base ** 2) + 2.0 * ref * base
        case 'screen':
            return 1.0 - (1.0 - base) * (1.0 - ref)
        case 'multiply':
            return base * ref
        case _:
            raise ValueError(f"Unknown blend mode: {blend_mode}")

class BlendRefTransform(Transform):
    NAME = 'blend_ref'
    REQUIRED_PARAMS = {'img_path', 'opacity'}
    EVAL_PARAMS = {'opacity', 'scale'}
    OPTIONAL_PARAMS = {'blend_mode', 'mask', 'mask_invert', 'easing', 'scale'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        opacity: float = self.params['opacity']
        img_path: str = _resolve_img_path(self.params['img_path'])
        blend_mode: str = self.params.get('blend_mode', 'normal')
        mask_type: str = self.params.get('mask', 'uniform')
        mask_invert: bool = self.params.get('mask_invert', False)
        easing: str = self.params.get('easing', 'linear')
        scale: float = self.params.get('scale', 1.0)

        ref_img = Image.open(img_path).convert('RGB')

        eased_opacity = _apply_easing(opacity, easing)

        if eased_opacity == 0.0:
            return img

        base = np.array(img, dtype=np.float32) / 255.0

        scale = float(np.clip(scale, 0.01, 1.0))
        if scale < 1.0:
            scaled_w = max(1, int(init_width * scale))
            scaled_h = max(1, int(init_height * scale))
            ref_img = ref_img.resize((scaled_w, scaled_h))
            ref_scaled = np.array(ref_img, dtype=np.float32) / 255.0

            # place scaled ref centered on a canvas of base pixels
            ref = base.copy()
            x_off = (init_width - scaled_w) // 2
            y_off = (init_height - scaled_h) // 2
            ref[y_off:y_off + scaled_h, x_off:x_off + scaled_w] = ref_scaled

            # placement mask: blend only where the ref was placed
            placement_mask = np.zeros((init_height, init_width), dtype=np.float32)
            placement_mask[y_off:y_off + scaled_h, x_off:x_off + scaled_w] = 1.0
        else:
            ref_img = ref_img.resize((init_width, init_height))
            ref = np.array(ref_img, dtype=np.float32) / 255.0
            placement_mask = np.ones((init_height, init_width), dtype=np.float32)

        mask = _generate_mask(init_width, init_height, mask_type, mask_invert)
        mask = mask * eased_opacity * placement_mask

        blended = _blend_pixels(base, ref, blend_mode)
        blended = np.clip(blended, 0.0, 1.0)

        mask_3d = mask[:, :, np.newaxis]
        result = base * (1.0 - mask_3d) + blended * mask_3d
        result = np.clip(result, 0.0, 1.0)

        return Image.fromarray((result * 255.0).astype(np.uint8))

class WaveDistortionTransformation(Transform):
    NAME = 'wave'
    REQUIRED_PARAMS = {'period', 'strength', 'rate'}
    EVAL_PARAMS = {'period', 'strength', 'rate'}

    # Adapted from: https://www.pythoninformer.com/python-libraries/pillow/imageops-deforming/
    class WaveDeformer:
        def __init__(self, positive: bool, period: int, strength: int):
            self.positive = positive
            self.strength = strength
            self.period = period

        def transform(self, x, y):
            if self.positive:
                y = y + self.strength*math.sin(x/self.period)
            else:
                y = y - self.strength*math.sin(x/self.period)

            return x, y

        def transform_rectangle(self, x0, y0, x1, y1):
            return (*self.transform(x0, y0),
                    *self.transform(x0, y1),
                    *self.transform(x1, y1),
                    *self.transform(x1, y0),
                    )

        def getmesh(self, img):
            self.w, self.h = img.size
            gridspace = 20

            target_grid = []
            for x in range(0, self.w, gridspace):
                for y in range(0, self.h, gridspace):
                    target_grid.append((x, y, x + gridspace, y + gridspace))

            source_grid = [self.transform_rectangle(*rect) for rect in target_grid]

            return [t for t in zip(target_grid, source_grid)]

    def transform(self, img: Image) -> Image:
        n = int(self.params['n'])
        strength = int(self.params['strength'])
        period = int(self.params['period'])
        rate = int(self.params['rate'])
        if strength == 0.0:
            return img

        # alternating pattern
        modval = (n / rate) % rate
        positive = modval > (rate / 4) and modval <= ((rate*3) / 4)

        result = ImageOps.deform(img, WaveDistortionTransformation.WaveDeformer(positive=positive, period=period, strength=strength))
        return result

class PerspectiveTransformation(Transform):
    NAME = 'perspective'
    REQUIRED_PARAMS = {'strength', 'shrink_edge'}
    EVAL_PARAMS = {'strength'}

    def transform(self, img: Image) -> Image:
        image_width, image_height = img.size
        strength = int(self.params['strength'])
        shrink_edge = self.params['shrink_edge']
        if strength == 0.0:
            return img

        image = np.asarray(img)
        src_pts = np.float32(
            [
                # top left
                [0,0], 

                # bottom left
                [0,image_height],

                # top right
                [image_width, 0],

                # bottom right
                [image_width, image_height]
            ]
        )

        match shrink_edge:
            case 'top':
                dest_pts = np.float32(
                    [
                        # top left
                        [strength,0],

                        # bottom left
                        [0,image_height],

                        # top right
                        [image_width-(strength+1), 0],

                        # bottom right
                        [image_width, image_height]
                    ]
                )
            case 'bottom':
                dest_pts = np.float32(
                    [
                        # top left
                        [0,0],

                        # bottom left
                        [strength,image_height],

                        # top right
                        [image_width, 0],

                        # bottom right
                        [image_width-(strength+1), image_height]
                    ]
                )
            case 'left':
                dest_pts = np.float32(
                    [
                        # top left
                        [0,strength],

                        # bottom left
                        [0,image_height-(strength+1)],

                        # top right
                        [image_width, 0],

                        # bottom right
                        [image_width, image_height]
                    ]
                )
            case 'right':
                dest_pts = np.float32(
                    [
                        # top left
                        [0,0],

                        # bottom left
                        [0,image_height],

                        # top right
                        [image_width, strength],

                        # bottom right
                        [image_width, image_height-(strength+1)]
                    ]
                )
            case _:
                raise Exception(f"Illegal value for shrink_edge: {shrink_edge}")

        matrix = cv2.getPerspectiveTransform(src_pts, dest_pts)
        result = cv2.warpPerspective(image, matrix, (image_width, image_height))

        # resize back to initial size
        resized_result = Image.fromarray(result).resize((image_width, image_height))
        return resized_result

class SpiralTransform(Transform):
    NAME = 'spiral'
    REQUIRED_PARAMS = {'strength'}
    EVAL_PARAMS = {'strength'}

    def transform(self, img: Image) -> Image:
        strength: float = self.params['strength']
        if strength == 0.0:
            return img

        width, height = img.size
        center_x, center_y = width / 2, height / 2
        max_radius = math.sqrt(center_x**2 + center_y**2)

        image = np.array(img)
        y, x = np.mgrid[0:height, 0:width]

        dx = x - center_x
        dy = y - center_y
        radius = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)

        # rotation amount proportional to distance from center
        twist = strength * radius / max_radius
        new_angle = angle + twist

        x_new = (center_x + radius * np.cos(new_angle)).astype(np.float32)
        y_new = (center_y + radius * np.sin(new_angle)).astype(np.float32)

        result = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(result)

class PanTransform(Transform):
    NAME = 'pan'
    REQUIRED_PARAMS = {'dx', 'dy'}
    EVAL_PARAMS = {'dx', 'dy'}

    def transform(self, img: Image) -> Image:
        dx: int = int(self.params['dx'])
        dy: int = int(self.params['dy'])
        if dx == 0 and dy == 0:
            return img

        image = np.array(img)
        result = np.roll(image, dy, axis=0)
        result = np.roll(result, dx, axis=1)
        return Image.fromarray(result)

class MirrorTransform(Transform):
    NAME = 'mirror'
    REQUIRED_PARAMS = {'mode'}
    EVAL_PARAMS = set()

    def transform(self, img: Image) -> Image:
        width, height = img.size
        mode = self.params['mode']
        image = np.array(img)

        match mode:
            case 'left_to_right':
                half = image[:, :width//2, :]
                image[:, width//2:, :] = half[:, ::-1, :]
            case 'right_to_left':
                half = image[:, width//2:, :]
                image[:, :width//2, :] = half[:, ::-1, :]
            case 'top_to_bottom':
                half = image[:height//2, :, :]
                image[height//2:, :, :] = half[::-1, :, :]
            case 'bottom_to_top':
                half = image[height//2:, :, :]
                image[:height//2, :, :] = half[::-1, :, :]
            case _:
                raise Exception(f"Illegal value for mirror mode: {mode}")

        return Image.fromarray(image)

class KaleidoscopeTransform(Transform):
    NAME = 'kaleidoscope'
    REQUIRED_PARAMS = {'segments'}
    EVAL_PARAMS = {'segments'}

    def transform(self, img: Image) -> Image:
        segments: int = int(self.params['segments'])
        if segments < 2:
            return img

        width, height = img.size
        center_x, center_y = width / 2, height / 2
        image = np.array(img)

        y, x = np.mgrid[0:height, 0:width]
        dx = x - center_x
        dy = y - center_y

        angle = np.arctan2(dy, dx)
        radius = np.sqrt(dx**2 + dy**2)

        # map all angles into one wedge, then mirror alternating wedges
        wedge_angle = 2 * math.pi / segments
        wedge_index = np.floor(angle / wedge_angle).astype(int)
        local_angle = angle - wedge_index * wedge_angle

        # mirror odd-numbered wedges for seamless tiling
        odd_mask = (wedge_index % 2) == 1
        local_angle[odd_mask] = wedge_angle - local_angle[odd_mask]

        x_new = (center_x + radius * np.cos(local_angle)).astype(np.float32)
        y_new = (center_y + radius * np.sin(local_angle)).astype(np.float32)

        result = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(result)

class HueShiftTransform(Transform):
    NAME = 'hue_shift'
    REQUIRED_PARAMS = {'shift_deg'}
    EVAL_PARAMS = {'shift_deg'}

    def transform(self, img: Image) -> Image:
        shift_deg: float = self.params['shift_deg']
        if shift_deg == 0.0:
            return img

        image = np.array(img)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # OpenCV hue range is 0-179
        shift_amount = int((shift_deg / 360.0) * 180) % 180
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + shift_amount) % 180
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(result)

class ColorChannelOffsetTransform(Transform):
    NAME = 'color_channel_offset'
    REQUIRED_PARAMS = {'r_offset', 'g_offset', 'b_offset'}
    EVAL_PARAMS = {'r_offset', 'g_offset', 'b_offset'}

    def transform(self, img: Image) -> Image:
        r_offset: int = int(self.params['r_offset'])
        g_offset: int = int(self.params['g_offset'])
        b_offset: int = int(self.params['b_offset'])
        if r_offset == 0 and g_offset == 0 and b_offset == 0:
            return img

        image = np.array(img)
        result = np.zeros_like(image)
        result[:, :, 0] = np.roll(image[:, :, 0], r_offset, axis=1)
        result[:, :, 1] = np.roll(image[:, :, 1], g_offset, axis=1)
        result[:, :, 2] = np.roll(image[:, :, 2], b_offset, axis=1)
        return Image.fromarray(result)

class ContrastBrightnessTransform(Transform):
    NAME = 'contrast_brightness'
    REQUIRED_PARAMS = {'contrast', 'brightness'}
    EVAL_PARAMS = {'contrast', 'brightness'}

    def transform(self, img: Image) -> Image:
        contrast: float = self.params['contrast']
        brightness: float = self.params['brightness']
        if contrast == 1.0 and brightness == 0.0:
            return img

        result = ImageEnhance.Contrast(img).enhance(contrast)
        result = ImageEnhance.Brightness(result).enhance(1.0 + brightness)
        return result

class RippleTransform(Transform):
    NAME = 'ripple'
    REQUIRED_PARAMS = {'amplitude', 'wavelength'}
    EVAL_PARAMS = {'amplitude', 'wavelength', 'phase'}

    def transform(self, img: Image) -> Image:
        amplitude: float = self.params['amplitude']
        wavelength: float = self.params['wavelength']
        phase: float = self.params.get('phase', 0.0)
        if amplitude == 0.0:
            return img

        width, height = img.size
        center_x, center_y = width / 2, height / 2
        image = np.array(img)

        y, x = np.mgrid[0:height, 0:width]
        dx = x - center_x
        dy = y - center_y
        radius = np.sqrt(dx**2 + dy**2)

        # radial displacement
        displacement = amplitude * np.sin(2 * math.pi * radius / wavelength + phase)
        angle = np.arctan2(dy, dx)

        x_new = (x + displacement * np.cos(angle)).astype(np.float32)
        y_new = (y + displacement * np.sin(angle)).astype(np.float32)

        result = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(result)

class PixelateTransform(Transform):
    NAME = 'pixelate'
    REQUIRED_PARAMS = {'block_size'}
    EVAL_PARAMS = {'block_size'}

    def transform(self, img: Image) -> Image:
        block_size: int = int(self.params['block_size'])
        if block_size <= 1:
            return img

        width, height = img.size
        small_w = max(1, width // block_size)
        small_h = max(1, height // block_size)

        result = img.resize((small_w, small_h), Image.NEAREST)
        result = result.resize((width, height), Image.NEAREST)
        return result

class ElasticTransform(Transform):
    NAME = 'elastic'
    REQUIRED_PARAMS = {'strength', 'smoothness'}
    EVAL_PARAMS = {'strength', 'smoothness', 'seed'}

    def transform(self, img: Image) -> Image:
        from scipy.ndimage import gaussian_filter

        strength: float = self.params['strength']
        smoothness: float = self.params['smoothness']
        seed: int = int(self.params.get('seed', 0))
        if strength == 0.0:
            return img

        width, height = img.size
        image = np.array(img)

        rng = np.random.RandomState(seed)
        dx = gaussian_filter((rng.rand(height, width) * 2 - 1), smoothness) * strength
        dy = gaussian_filter((rng.rand(height, width) * 2 - 1), smoothness) * strength

        y, x = np.mgrid[0:height, 0:width]
        x_new = (x + dx).astype(np.float32)
        y_new = (y + dy).astype(np.float32)

        result = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(result)

def elaborate_transform_expr(transform_expr: str | float, auto_params: AutomaticTransformParams):
        """
        n           --> total iteration sequence number
        offset      --> current LoopSettings sequence number
        total_n     --> length of entire workflow
        fft_samples --> optional tuple of amplitudes for requested frequency ranges
        """

        def get_power_at_freq_range(low_f: int, high_f: int):
            if auto_params.wavefile is not None:
                start_percent = (auto_params.n / auto_params.total_n) * 100.0
                end_percent = ((auto_params.n + 1) / auto_params.total_n) * 100.0
                pow_at_frange = auto_params.wavefile.get_power_in_freq_ranges(start_percent, end_percent, [(low_f, high_f)])[0]
                return pow_at_frange
            else:
                print("Warning no audio file specified, nothing to do for 'get_power_at_freq_range()'")
                return 0

        if isinstance(transform_expr, str):
            local_vars = {
                'n':auto_params.n,
                'offset':auto_params.offset,
                'total_n':auto_params.total_n
            }
            return SimpleExprEval(local_vars=local_vars, permitted_fns={'get_power_at_freq_range':get_power_at_freq_range})(transform_expr)
        else:
            return transform_expr

def get_elaborated_transform_values(transforms: list[dict[str, Any]], auto_params: AutomaticTransformParams) -> list[dict[str, Any]]:
    elaborated_transforms = []
    if transforms is not None:
        for tdict in transforms:
            elab_tdict = dict()
            for key, val in tdict.items():
                if key in TRANSFORM_LIBRARY[tdict['name']].get_eval_params():
                    elab_tdict[key] = elaborate_transform_expr(val, auto_params)
                else:
                    elab_tdict[key] = val
            elaborated_transforms.append(elab_tdict)
    return elaborated_transforms

def load_image_with_transforms(image_path: str, transforms: list[dict[str, Any]], auto_params: AutomaticTransformParams) -> tuple[Image.Image, list[dict[str, Any]]]:
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")

    elaborated_transforms = transforms
    if len(transforms) > 0:
        elaborated_transforms = get_elaborated_transform_values(transforms=transforms, auto_params=auto_params)
        image = Transform.apply_transformations(
            img=image,
            transforms=elaborated_transforms,
            auto_params=auto_params
        )

    return image, elaborated_transforms

TRANSFORM_LIBRARY: dict[str, Transform] = {t.get_name(): t for t in all_subclasses(Transform)}

def get_transform_help_string() -> str:
    result = 'Available transforms:\n'
    for tfname in TRANSFORM_LIBRARY:
        optional = TRANSFORM_LIBRARY[tfname].get_optional_params()
        optional_str = f' optional:{optional}' if optional else ''
        result += f'\tname:{tfname} params:{TRANSFORM_LIBRARY[tfname].get_required_params()}{optional_str}\n'
    return result + '\n'