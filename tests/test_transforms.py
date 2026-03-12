import pytest
import numpy as np
from PIL import Image
from comfyui_looper.image_processing.transforms import *

# workaround for running in debugger -- it picks up pytest.ini this way
if __name__ == '__main__':
    pytest.main(['-s'])

def test_instantiate_all_transforms():
    for trans_name in TRANSFORM_LIBRARY:
        t = TRANSFORM_LIBRARY[trans_name]({})
        assert t.get_name() == trans_name

def _make_test_image(width=256, height=256):
    """Create a test image with a color gradient so edge content is non-trivial."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)  # red gradient L-R
    arr[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]  # green gradient T-B
    arr[:, :, 2] = 128
    return Image.fromarray(arr)

class TestZoomOutTransform:
    def test_registered_in_library(self):
        assert 'zoom_out' in TRANSFORM_LIBRARY

    def test_zero_zoom_returns_same_image(self):
        img = _make_test_image()
        t = ZoomOutTransform({'zoom_amt': 0.0})
        result = t.transform(img)
        assert result is img

    def test_output_size_matches_input_reflect(self):
        img = _make_test_image(300, 200)
        t = ZoomOutTransform({'zoom_amt': 0.2, 'fill_mode': 'reflect'})
        result = t.transform(img)
        assert result.size == (300, 200)

    def test_output_size_matches_input_blur(self):
        img = _make_test_image(300, 200)
        t = ZoomOutTransform({'zoom_amt': 0.2, 'fill_mode': 'blur'})
        result = t.transform(img)
        assert result.size == (300, 200)

    def test_default_fill_mode_is_reflect(self):
        img = _make_test_image()
        t_default = ZoomOutTransform({'zoom_amt': 0.2})
        t_reflect = ZoomOutTransform({'zoom_amt': 0.2, 'fill_mode': 'reflect'})
        r_default = np.array(t_default.transform(img))
        r_reflect = np.array(t_reflect.transform(img))
        assert np.array_equal(r_default, r_reflect)

    def test_center_preserved_reflect(self):
        """The center region of the output should match the shrunk image content."""
        img = _make_test_image()
        zoom_amt = 0.2
        t = ZoomOutTransform({'zoom_amt': zoom_amt, 'fill_mode': 'reflect'})
        result = np.array(t.transform(img))

        w, h = img.size
        new_w = int((1.0 - zoom_amt) * w)
        new_h = int((1.0 - zoom_amt) * h)
        shrunk = np.array(img.resize((new_w, new_h), Image.LANCZOS))

        pad_left = (w - new_w) // 2
        pad_top = (h - new_h) // 2
        center = result[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
        assert np.array_equal(center, shrunk)

    def test_center_preserved_blur(self):
        """The center region of blur output should match the shrunk image."""
        img = _make_test_image()
        zoom_amt = 0.2
        t = ZoomOutTransform({'zoom_amt': zoom_amt, 'fill_mode': 'blur'})
        result = np.array(t.transform(img))

        w, h = img.size
        new_w = int((1.0 - zoom_amt) * w)
        new_h = int((1.0 - zoom_amt) * h)
        shrunk = np.array(img.resize((new_w, new_h), Image.LANCZOS))

        pad_left = (w - new_w) // 2
        pad_top = (h - new_h) // 2
        center = result[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
        assert np.array_equal(center, shrunk)

    def test_reflect_edges_not_black(self):
        """Reflect fill should produce non-black border pixels."""
        img = _make_test_image()
        t = ZoomOutTransform({'zoom_amt': 0.3, 'fill_mode': 'reflect'})
        result = np.array(t.transform(img))
        # Check top-left corner region is not all black
        corner = result[:10, :10]
        assert corner.sum() > 0

    def test_blur_edges_not_black(self):
        """Blur fill should produce non-black border pixels."""
        img = _make_test_image()
        t = ZoomOutTransform({'zoom_amt': 0.3, 'fill_mode': 'blur'})
        result = np.array(t.transform(img))
        corner = result[:10, :10]
        assert corner.sum() > 0

    def test_reflect_and_blur_produce_different_results(self):
        img = _make_test_image()
        t_reflect = ZoomOutTransform({'zoom_amt': 0.2, 'fill_mode': 'reflect'})
        t_blur = ZoomOutTransform({'zoom_amt': 0.2, 'fill_mode': 'blur'})
        r_reflect = np.array(t_reflect.transform(img))
        r_blur = np.array(t_blur.transform(img))
        assert not np.array_equal(r_reflect, r_blur)

    def test_validate_params(self):
        transforms = [{'name': 'zoom_out', 'zoom_amt': 0.1}]
        Transform.validate_transformation_params(transforms)