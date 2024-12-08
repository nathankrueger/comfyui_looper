import re
import ast
import math
import operator
import textwrap
from typing import Callable, Any

class UnsupportedFunctionException(Exception):
    pass

class NameNotFoundException(Exception):
    pass

# inspired by: https://stackoverflow.com/questions/43836866/safely-evaluate-simple-string-equation
class SimpleExprEval:
    NODE_TO_OP = {
        # math / binary
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,

        # boolean
        ast.Or: operator.or_,
        ast.And: operator.and_,

        # comparison
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.In: operator.contains,

        # unary
        ast.USub: operator.neg,
        ast.Not: operator.not_,
        ast.Invert: operator.inv,
    }

    DEFAULT_PERMITTED_FUNCTIONS = {
        # expr match
        "re.match": re.match,

        # common builtins
        "len": len,
        "range": range,
        "max": max,
        "min": min,
        "any": any,
        "all": all,

        # type conversion
        "str": str,
        "float": float,
        "int": int,
        "list": list,
        "dict": dict,

        # debug
        "print": print,

        # math
        "floor": math.floor,
        "ceil": math.ceil,
        "cos": math.cos,
        "sin": math.sin,
        "tan": math.tan,
        "tanh": math.tanh,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "relu": lambda x: x if x > 0 else 0,
        "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
    }

    DEFAULT_VARIABLES = {
        "pi": math.pi,
    }

    @staticmethod
    def get_function_name(ast_node: ast.expr) -> str:
        fn_name_list = []
        attr_walker = ast_node
        while isinstance(attr_walker, ast.Attribute):
            fn_name_list.append(attr_walker.attr)
            attr_walker = attr_walker.value
        fn_name_list.append (attr_walker.id)
        fn_name = ".".join(reversed(fn_name_list))
        return fn_name

    def __init__(self, mode: str = "eval", local_vars: dict = None, permitted_fns: dict = None):
        # parser mode -- best practice is to use the minimum needed ('eval' vs 'exec')
        self.mode: str = mode

        # local vars
        self.local_vars: dict[str, Any] = dict(SimpleExprEval.DEFAULT_VARIABLES)
        if local_vars is not None:
            self.local_vars.update(local_vars)

        # cache initial state of local_vars
        self.init_locals_state: dict[str, Any] = dict(local_vars) if local_vars is not None else {}

        # permitted functions
        self.permitted_functions: dict[str, Callable] = dict(SimpleExprEval.DEFAULT_PERMITTED_FUNCTIONS)
        if permitted_fns is not None:
            self.permitted_functions.update(permitted_fns)

        # stack var
        self.loop_vars: dict [str, Any] = {}

        # state vars
        self.returned = False
        self.break_stack: list[bool] = []

    def __call__(self, expr: str):
        return self._eval_expression(expr)
    
    def reset_locals(self):
        self.local_vars = dict(self.init_locals_state)
        
    def _eval_expression(self, expr: str) -> str:
        self.returned = False
        self.break_stack = []
        return self._eval_recursive(ast.parse(textwrap.dedent(expr).strip(), mode=self.mode))
    
    def _eval_recursive(self, ast_node: ast.expr):
        # bottom out recursion
        if ast_node is None:
            return None

        if self.returned:
            return ast_node
        
        # handle each node as appropriate
        match type(ast_node):
            case ast.Module:
                for expr in ast_node.body:
                    result = self._eval_recursive(expr)
                    if self.returned:
                        break
                return self._eval_recursive(result)
            case ast.Expression:
                return self._eval_recursive(ast_node.body)
            case ast.Expr:
                return self._eval_recursive(ast_node.value)
            case ast.If:
                if self._eval_recursive(ast_node.test):
                    for expr in ast_node.body:
                        result = self._eval_recursive(expr)
                        if self.returned:
                            break
                else:
                    for expr in ast_node.orelse:
                        result = self._eval_recursive(expr)
                        if self.returned:
                            break
                return self._eval_recursive(result)
            case ast.Match:
                subject_val = self._eval_recursive(ast_node.subject)
                for case in ast_node.cases:
                    if isinstance(case.pattern, ast.MatchAs) or (self._eval_recursive(case) == subject_val):
                        for expr in case.body:
                            result = self._eval_recursive(expr)
                            if self.returned:
                                break
                    else:
                            continue
                    break
                return self._eval_recursive(result)
            case ast.match_case:
                return self._eval_recursive(ast_node.pattern)
            case ast.MatchValue:
                return self._eval_recursive(ast_node.value)
            case ast.For:
                result = None
                self.break_stack.append(False)
                for val in self._eval_recursive(ast_node.iter):
                    self.loop_vars[ast_node.target.id] = val
                    for expr in ast_node.body:
                        result = self._eval_recursive(expr)
                        if self.returned or self.break_stack[-1]:
                            break
                    else:
                        del self.loop_vars[ast_node.target.id]
                        continue
                    break
                self.break_stack = self.break_stack[:-1]
                return self._eval_recursive(result)
            case ast.While:
                result = None
                self.break_stack.append(False)
                while self._eval_recursive(ast_node.test):
                    for expr in ast_node.body:
                        result = self._eval_recursive(expr)
                        if self.returned or self.break_stack[-1]:
                            break
                    else:
                        continue
                self.break_stack = self.break_stack[:-1]
                return self._eval_recursive(result)
            case ast.Return:
                result = self._eval_recur_(ast_node.value)
                self.returned = True
                return result
            case ast.Break:
                self.break_stack[-1] = True
                return None
            case ast.UnaryOp:
                op = SimpleExprEval.NODE_TO_OP[type(ast_node.op)]
                return op(self._eval_recursive(ast_node.operand))
            case ast.BinOp:
                op = SimpleExprEval.NODE_TO_OP[type(ast_node.op)]
                return op(self._eval_recursive(ast_node.left), self._eval_recursive(ast_node.right))
            case ast.BoolOp:
                op = SimpleExprEval.NODE_TO_OP[type(ast_node.op)]
                assert len(ast_node.values) >= 2
                result = op(self._eval_recursive(ast_node.values[0]), self._eval_recursive(ast_node.values[1]))
                for i in range (len(ast_node.values) - 2):
                    result = op(result, self._eval_recursive(ast_node.values[i + 2]))
                return result
            case ast.Compare:
                match (op := SimpleExprEval.NODE_TO_OP[type(ast_node.ops[0])]):
                    case operator.contains:
                        return op(self._eval_recursive(ast_node.comparators[0]), self._eval_recursive(ast_node.left))
                    case _:
                        return op(self._eval_recursive(ast_node.left), self._eval_recursive(ast_node.comparators[0]))
            case ast.Call:
                func_name = SimpleExprEval.get_function_name(ast_node.func)
                if func_name in self.permitted_functions:
                    args = tuple([self._eval_recursive(arg) for arg in ast_node.args] )
                    kwargs = {}
                    for keyword in ast_node.keywords:
                        name, val = self._eval_recursive(keyword)
                        kwargs[name] = val
                    result = self.permitted_functions[func_name](*args, **kwargs)
                    return result
                else:
                    raise UnsupportedFunctionException(f"Function call not permitted: {func_name}")
            case ast.Assign:
                # tuple not supported
                assert len(ast_node.targets) == 1
                value = self.eval_recur_(ast_node.value)
                target = ast_node.targets[0].id
                self.loop_vars[target] = value
                return None
            case ast.Assert:
                assert self._eval_recursive(ast_node.test)
                return None
            case ast.Constant:
                return ast_node.value
            case ast.Name:
                var_name = ast_node.id
                if var_name in self.local_vars:
                    return self.local_vars[var_name]
                elif var_name in self.loop_vars:
                    return self.loop_vars[var_name]
                else:
                    raise NameNotFoundException(f"Attempted to access undefined variable: {var_name}")
            case ast.keyword:
                return (ast_node.arg, self._eval_recursive(ast_node.value))
            case ast.Attribute:
                return getattr(self._eval_recursive(ast_node.value), ast_node.attr)
            case ast.List:
                return [self._eval_recursive(list_item) for list_item in ast_node.elts]
            case ast.Subscript:
                return operator.getitem(self._eval_recursive(ast_node.value), self._eval_recursive(ast_node.slice))
            case ast.Slice:
                lower = self._eval_recursive(ast_node.lower)
                upper = self._eval_recursive(ast_node.upper)
                step  = self._eval_recursive(ast_node.step)
                return slice(lower, upper, step)

        raise NotImplementedError(f"Unsupported operation attempted: {type(ast_node)}")