import pytest

# workaround for running in debugger -- it picks up pytest.ini this way
if __name__ == '__main__':
    pytest.main(['-s'])

from comfyui_looper.utils.util import *
from comfyui_looper.utils.simple_expr_eval import SimpleExprEval

def test_expr_evaluation():
    test_val = 5
    assert SimpleExprEval()("3") == 3
    assert SimpleExprEval()("3+7-1+1") == 10
    assert SimpleExprEval(local_vars={'n': test_val})("n") == 5
    assert SimpleExprEval(local_vars={'n': test_val})("n * 5") == 25
    assert SimpleExprEval(local_vars={'n': test_val})("n * 3 + (cos(pi) + cos(1.243))") == 14.321957475114269
    assert SimpleExprEval(local_vars={'n': test_val})("-n == -5")
    assert SimpleExprEval(local_vars={'n': test_val})("n**n") == 3125
    assert SimpleExprEval(local_vars={'x': math.pi})("floor(5.1)") == 5
    assert SimpleExprEval(local_vars={'x': math.pi})("list(str(ceil(5.1)))") == ["6"]
    assert SimpleExprEval(local_vars={'x': -1})("relu(x)") == 0
    assert SimpleExprEval(local_vars={'x': 6.5})("relu(x)") == 6.5
    assert SimpleExprEval(local_vars={'n': 1})("8 if n == 1 else 2") == 8