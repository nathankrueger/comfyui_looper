import pytest
import numpy as np
from PIL import Image
from comfyui_looper.image_processing.transforms import *
from comfyui_looper.image_processing.transforms import _apply_easing, _generate_mask, _blend_pixels

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


class TestValidationErrorMessages:
    def test_unknown_transform_name(self):
        with pytest.raises(AssertionError, match=r"Unknown transform 'zoom_left'"):
            Transform.validate_transformation_params([{'name': 'zoom_left', 'zoom_amt': 0.1}])

    def test_unknown_transform_lists_available(self):
        with pytest.raises(AssertionError, match=r"Available:"):
            Transform.validate_transformation_params([{'name': 'bogus'}])

    def test_missing_name_field(self):
        with pytest.raises(AssertionError, match=r"missing 'name' field"):
            Transform.validate_transformation_params([{'zoom_amt': 0.1}])

    def test_missing_required_params(self):
        with pytest.raises(AssertionError, match=r"missing required params"):
            Transform.validate_transformation_params([{'name': 'blend_ref'}])

    def test_reserved_automatic_param(self):
        with pytest.raises(AssertionError, match=r"reserved param"):
            Transform.validate_transformation_params([{'name': 'zoom_in', 'zoom_amt': 0.1, 'n': 5}])

    def test_valid_transform_passes(self):
        Transform.validate_transformation_params([{'name': 'zoom_in', 'zoom_amt': 0.1}])

    def test_empty_dict_passes(self):
        Transform.validate_transformation_params([{}])


def _make_ref_image(tmp_path, width=256, height=256):
    """Create a reference image that differs from the gradient test image."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = 200
    arr[:, :, 1] = 50
    arr[:, :, 2] = 50
    ref = Image.fromarray(arr)
    path = str(tmp_path / 'ref.png')
    ref.save(path)
    return path

class TestBlendRefTransform:
    def test_registered_in_library(self):
        assert 'blend_ref' in TRANSFORM_LIBRARY

    def test_zero_opacity_returns_original(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        t = BlendRefTransform({'img_path': ref_path, 'opacity': 0.0})
        result = t.transform(img)
        assert result is img

    def test_full_opacity_normal_returns_ref(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        t = BlendRefTransform({'img_path': ref_path, 'opacity': 1.0})
        result = np.array(t.transform(img))
        ref = np.array(Image.open(ref_path).resize(img.size))
        assert np.allclose(result, ref, atol=1)

    def test_output_size_matches_input(self, tmp_path):
        img = _make_test_image(300, 200)
        ref_path = _make_ref_image(tmp_path)
        for mode in ['normal', 'overlay', 'soft_light', 'screen', 'multiply']:
            t = BlendRefTransform({'img_path': ref_path, 'opacity': 0.5, 'blend_mode': mode})
            result = t.transform(img)
            assert result.size == (300, 200), f"Size mismatch for blend_mode={mode}"

    def test_normal_matches_paste_img(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        opacity = 0.4
        t_blend = BlendRefTransform({'img_path': ref_path, 'opacity': opacity})
        t_paste = PasteImageTransform({'img_path': ref_path, 'opacity': opacity})
        r_blend = np.array(t_blend.transform(img))
        r_paste = np.array(t_paste.transform(img))
        assert np.allclose(r_blend, r_paste, atol=1)

    def test_different_blend_modes_produce_different_results(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        results = {}
        for mode in ['normal', 'overlay', 'soft_light', 'screen', 'multiply']:
            t = BlendRefTransform({'img_path': ref_path, 'opacity': 0.5, 'blend_mode': mode})
            results[mode] = np.array(t.transform(img))
        modes = list(results.keys())
        for i in range(len(modes)):
            for j in range(i+1, len(modes)):
                assert not np.array_equal(results[modes[i]], results[modes[j]]), \
                    f"{modes[i]} and {modes[j]} should differ"

    def test_blend_modes_valid_pixel_range(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        for mode in ['normal', 'overlay', 'soft_light', 'screen', 'multiply']:
            t = BlendRefTransform({'img_path': ref_path, 'opacity': 0.8, 'blend_mode': mode})
            result = np.array(t.transform(img))
            assert result.min() >= 0 and result.max() <= 255, f"Out of range for {mode}"

    def test_mask_types_differ_from_uniform(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        r_uniform = np.array(BlendRefTransform(
            {'img_path': ref_path, 'opacity': 0.5, 'mask': 'uniform'}).transform(img))
        for mask_type in ['radial', 'horizontal', 'vertical']:
            r_other = np.array(BlendRefTransform(
                {'img_path': ref_path, 'opacity': 0.5, 'mask': mask_type}).transform(img))
            assert not np.array_equal(r_uniform, r_other), \
                f"mask={mask_type} should differ from uniform"

    def test_mask_invert_differs(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        r_normal = np.array(BlendRefTransform(
            {'img_path': ref_path, 'opacity': 0.5, 'mask': 'radial', 'mask_invert': False}).transform(img))
        r_invert = np.array(BlendRefTransform(
            {'img_path': ref_path, 'opacity': 0.5, 'mask': 'radial', 'mask_invert': True}).transform(img))
        assert not np.array_equal(r_normal, r_invert)

    def test_easing_linear_is_default(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        r_default = np.array(BlendRefTransform(
            {'img_path': ref_path, 'opacity': 0.5}).transform(img))
        r_linear = np.array(BlendRefTransform(
            {'img_path': ref_path, 'opacity': 0.5, 'easing': 'linear'}).transform(img))
        assert np.array_equal(r_default, r_linear)

    def test_easing_affects_result(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        r_linear = np.array(BlendRefTransform(
            {'img_path': ref_path, 'opacity': 0.5, 'easing': 'linear'}).transform(img))
        r_ease_in = np.array(BlendRefTransform(
            {'img_path': ref_path, 'opacity': 0.5, 'easing': 'ease_in'}).transform(img))
        assert not np.array_equal(r_linear, r_ease_in)

    def test_easing_at_extremes(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        for opacity in [0.0, 1.0]:
            results = []
            for easing in ['linear', 'ease_in', 'ease_out', 'ease_in_out']:
                r = np.array(BlendRefTransform(
                    {'img_path': ref_path, 'opacity': opacity, 'easing': easing}).transform(img))
                results.append(r)
            for i in range(1, len(results)):
                assert np.allclose(results[0], results[i], atol=1), \
                    f"Easing should not matter at opacity={opacity}"

    def test_overlay_with_half_gray_is_identity(self, tmp_path):
        """Overlay blend with a 50% gray reference should approximate identity."""
        img = _make_test_image()
        arr = np.full((256, 256, 3), 128, dtype=np.uint8)
        ref = Image.fromarray(arr)
        ref_path = str(tmp_path / 'gray.png')
        ref.save(ref_path)
        t = BlendRefTransform({'img_path': ref_path, 'opacity': 1.0, 'blend_mode': 'overlay'})
        result = np.array(t.transform(img))
        original = np.array(img)
        # overlay(base, 0.5) ~= base for most values; allow small rounding error
        assert np.allclose(result, original, atol=2)

    def test_screen_lightens(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        t = BlendRefTransform({'img_path': ref_path, 'opacity': 1.0, 'blend_mode': 'screen'})
        result = np.array(t.transform(img), dtype=np.int16)
        original = np.array(img, dtype=np.int16)
        assert (result >= original - 1).all(), "Screen should never darken"

    def test_multiply_darkens(self, tmp_path):
        img = _make_test_image()
        ref_path = _make_ref_image(tmp_path)
        t = BlendRefTransform({'img_path': ref_path, 'opacity': 1.0, 'blend_mode': 'multiply'})
        result = np.array(t.transform(img), dtype=np.int16)
        original = np.array(img, dtype=np.int16)
        assert (result <= original + 1).all(), "Multiply should never brighten"

    def test_validate_params(self, tmp_path):
        ref_path = _make_ref_image(tmp_path)
        transforms = [{'name': 'blend_ref', 'img_path': ref_path, 'opacity': 0.5}]
        Transform.validate_transformation_params(transforms)

    def test_validate_params_with_optionals(self, tmp_path):
        ref_path = _make_ref_image(tmp_path)
        transforms = [{'name': 'blend_ref', 'img_path': ref_path, 'opacity': 0.5,
                        'blend_mode': 'overlay', 'mask': 'radial', 'easing': 'ease_in'}]
        Transform.validate_transformation_params(transforms)


class TestEasingFunctions:
    def test_linear(self):
        assert _apply_easing(0.0, 'linear') == 0.0
        assert _apply_easing(0.5, 'linear') == 0.5
        assert _apply_easing(1.0, 'linear') == 1.0

    def test_ease_in(self):
        assert _apply_easing(0.0, 'ease_in') == 0.0
        assert _apply_easing(0.5, 'ease_in') == 0.25
        assert _apply_easing(1.0, 'ease_in') == 1.0

    def test_ease_out(self):
        assert _apply_easing(0.0, 'ease_out') == 0.0
        assert _apply_easing(0.5, 'ease_out') == 0.75
        assert _apply_easing(1.0, 'ease_out') == 1.0

    def test_ease_in_out(self):
        assert _apply_easing(0.0, 'ease_in_out') == 0.0
        assert _apply_easing(0.5, 'ease_in_out') == 0.5
        assert _apply_easing(1.0, 'ease_in_out') == 1.0

    def test_clamps_input(self):
        assert _apply_easing(-0.5, 'linear') == 0.0
        assert _apply_easing(1.5, 'linear') == 1.0

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _apply_easing(0.5, 'bogus')


class TestGenerateMask:
    def test_uniform(self):
        mask = _generate_mask(100, 100, 'uniform', False)
        assert mask.shape == (100, 100)
        assert np.all(mask == 1.0)

    def test_radial_center_is_max(self):
        mask = _generate_mask(100, 100, 'radial', False)
        assert mask[50, 50] > mask[0, 0]

    def test_horizontal_gradient(self):
        mask = _generate_mask(100, 50, 'horizontal', False)
        assert mask[0, 0] < mask[0, -1]
        # rows should be identical
        assert np.array_equal(mask[0], mask[-1])

    def test_vertical_gradient(self):
        mask = _generate_mask(50, 100, 'vertical', False)
        assert mask[0, 0] < mask[-1, 0]
        # columns should be identical
        assert np.array_equal(mask[:, 0], mask[:, -1])

    def test_invert(self):
        mask = _generate_mask(100, 100, 'horizontal', False)
        mask_inv = _generate_mask(100, 100, 'horizontal', True)
        assert np.allclose(mask + mask_inv, 1.0)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _generate_mask(10, 10, 'bogus', False)