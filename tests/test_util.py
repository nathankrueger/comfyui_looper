import pytest

# workaround for running in debugger -- it picks up pytest.ini this way
if __name__ == '__main__':
    pytest.main(['-s'])

from comfyui_looper.util import *

def test_math_parser():
    test_val = 5
    assert MathParser({'n': test_val})("n * 5") == 25
    assert MathParser({'n': test_val})("n * 3 + (cos(pi) + cos(1.243))") == 14.321957475114269