import pytest

# workaround for running in debugger -- it picks up pytest.ini this way
if __name__ == '__main__':
    pytest.main(['-s'])

from comfyui_looper.utils.util import *
from comfyui_looper.utils.simple_expr_eval import SimpleExprEval

def test_math_parser():
    test_val = 5
    assert SimpleExprEval(local_vars={'n': test_val})("n * 5") == 25
    assert SimpleExprEval(local_vars={'n': test_val})("n * 3 + (cos(pi) + cos(1.243))") == 14.321957475114269