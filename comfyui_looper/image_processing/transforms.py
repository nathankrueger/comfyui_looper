from os.path import abspath, dirname
import sys
from typing import Any
from PIL import Image, ImageOps
import numpy as np
import math
import torch
import cv2

try:
    from utils.util import all_subclasses
    from utils.simple_expr_eval import SimpleExprEval
except ModuleNotFoundError:
    SCRIPT_DIR = dirname(abspath(__file__))
    sys.path.append(dirname(dirname(SCRIPT_DIR)))
    from comfyui_looper.utils.util import all_subclasses
    from comfyui_looper.utils.simple_expr_eval import SimpleExprEval

MAGIC_SEQUENCE_PARAMS = {
    'n',
    'offset',
    'total_n'
}

class Transform:
    NAME = None
    REQUIRED_PARAMS = None
    EVAL_PARAMS = set()

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
    
    def transform(self, img: Image) -> Image:
        # override me
        return img
    
    @staticmethod
    def get_transform(name: str):
        return TRANSFORM_LIBRARY[name]
    
    @staticmethod
    def apply_transformations(img: Image, transforms: list[dict[str, Any]], iter: int, offset: int, total_iter: int) -> Image:
        curr_img = img
        for tdict in transforms:
            tdict = dict(tdict)
            t_name = tdict.pop('name')
            t_params = tdict

            # magic params partaining to where we are in the sequence
            t_params['n'] = iter
            t_params['offset'] = offset
            t_params['total_n'] = total_iter

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
                assert key not in MAGIC_SEQUENCE_PARAMS
                if key not in TRANSFORM_LIBRARY[t_name].get_required_params():
                    print(f"Warning, ignored transform param: {key} for transform {t_name}")

class ZoomInTransform(Transform):
    NAME = 'zoom_in'
    REQUIRED_PARAMS = {'zoom_amt'}
    EVAL_PARAMS = {'zoom_amt'}

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
    EVAL_PARAMS = {'zoom_amt'}

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
    EVAL_PARAMS = {'zoom_amt'}

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
    EVAL_PARAMS = {'zoom_amt'}

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
    EVAL_PARAMS = {'zoom_amt'}

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
    EVAL_PARAMS = {'squeeze_amt'}

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
    EVAL_PARAMS = {'squeeze_amt'}

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
    EVAL_PARAMS = {'strength'}

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

class PasteImageTransform(Transform):
    NAME = 'paste_img'
    REQUIRED_PARAMS = {'img_path', 'opacity'}
    EVAL_PARAMS = {'opacity'}

    def transform(self, img: Image) -> Image:
        init_width, init_height = img.size
        opacity: float = self.params['opacity']
        paste_img_path = self.params['img_path']

        paste_img = Image.open(paste_img_path)
        paste_img = paste_img.resize((init_width, init_height))
        
        result = Image.blend(img, paste_img, opacity)
        return result

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

        # alternating pattern
        modval = (n / rate) % rate
        positive = modval > (rate / 4) and modval <= ((rate*3) / 4)

        result = ImageOps.deform(img, WaveDistortionTransformation.WaveDeformer(positive=positive, period=period, strength=strength))
        return result

class PerspectiveTransformation(Transform):
    NAME = 'perspective'
    REQUIRED_PARAMS = {'strength', 'shrink_edge'}
    EVAL_PARAMS = {'strength', 'shrink_edge'}

    def transform(self, img: Image) -> Image:
        image_width, image_height = img.size
        strength = int(self.params['strength'])
        shrink_edge = self.params['shrink_edge']
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

def elaborate_transform_expr(transform_expr: str | float, iter: int, offset: int, total_iter: int):
        """
        n       --> total iteration sequence number
        offset  --> current LoopSettings sequence number
        total_n --> length of entire workflow
        """

        if isinstance(transform_expr, str):
            return SimpleExprEval(local_vars={'n':iter, 'offset':offset, 'total_n':total_iter})(transform_expr)
        else:
            return transform_expr

def get_elaborated_transform_values(transforms: list[dict[str, Any]], iter: int, offset: int, total_iter: int) -> list[dict[str, Any]]:
    elaborated_transforms = []
    if transforms is not None:
        for tdict in transforms:
            elab_tdict = dict()
            for key, val in tdict.items():
                if key in TRANSFORM_LIBRARY[tdict['name']].get_eval_params():
                    elab_tdict[key] = elaborate_transform_expr(val, iter, offset, total_iter)
                else:
                    elab_tdict[key] = val
            elaborated_transforms.append(elab_tdict)
    return elaborated_transforms

def load_image_with_transforms(image_path: str, transforms: list[dict[str, Any]], iter: int, offset: int, total_iter: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")

    elaborated_transforms = transforms
    if len(transforms) > 0:
        elaborated_transforms = get_elaborated_transform_values(transforms=transforms, iter=iter, offset=offset, total_iter=total_iter)
        image = Transform.apply_transformations(
            img=image,
            transforms=elaborated_transforms,
            iter=iter,
            offset=offset,
            total_iter=total_iter
        )

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image, elaborated_transforms

TRANSFORM_LIBRARY: dict[str, Transform] = {t.get_name(): t for t in all_subclasses(Transform)}

def get_transform_help_string() -> str:
    result = 'Available transforms:\n'
    for tfname in TRANSFORM_LIBRARY:
        result += f'\tname:{tfname} params:{TRANSFORM_LIBRARY[tfname].get_required_params()}\n'
    return result + '\n'