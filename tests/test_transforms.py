from comfyui_looper.transforms import *

def test_instantiate_all_transforms():
    for trans_name in TRANSFORM_LIBRARY:
        t = TRANSFORM_LIBRARY[trans_name]({})
        assert t.get_name() == trans_name