import pytest
from comfyui_looper.image_processing.transforms import *

# workaround for running in debugger -- it picks up pytest.ini this way
if __name__ == '__main__':
    pytest.main(['-s'])

def test_instantiate_all_transforms():
    for trans_name in TRANSFORM_LIBRARY:
        t = TRANSFORM_LIBRARY[trans_name]({})
        assert t.get_name() == trans_name