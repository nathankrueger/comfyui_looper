import re
import ast
import math
import operator
import textwrap
from typing import Callable, Any

# inspired by: https://stackoverflow.com/questions/43836866/safely-evaluate-simple-string-equation
class SimpleExprEval:
    AST_NODE_TO_OPERATOR = {
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
        "cos": math.cos,
        "sin": math.sin,
        "tan": math.tan,
        "tanh": math.tanh,
        "sinh": math.sinh,
        "cosh": math.cosh,
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
        self.init_locals_state: dict[str, Any] = dict(local_vars)

        # permitted functions
        self.permitted_functions: dict[str, Callable] = dict(SimpleExprEval.DEFAULT_PERMITTED_FUNCTIONS)
        if permitted_fns is not None:
            self.permitted_functions.update(permitted_fns)

        # stack var
        self.stack_vars: dict [str, Any] = {}

        # state vars
        self.has_returned = False
        self.break_stack: list[bool] = []

    def __call__(self, expr: str):
        return self._eval_expression(expr)
    
    def reset_locals(self):
        self.local_vars = dict(self.init_locals_state)
    
    def add_implict_mult_token(self, expr: str) -> str:
        """
        Renders expression suitable for 'eval_expr' by inserting implicit multiply
            e.g.  
                '9x'       --> '9*x'
                '3(7 + n)  --> '3*(7 + n)'
        """

        if len(self.local_vars) > 0:
            # match for all declared variables
            name_or = "|".join(self.local_vars.keys())

            # 35y --> 35*y
            result = re.sub(rf"(\d+)({name_or})", r"\1*\2", expr)

            # ab --> a*b
            result = re.sub(rf"({name_or})({name_or})", r"\1*\2", result)
            
            # 2(5 + 6) --> 2*(5 + 6), z(8 + 1) --> z*(8 + 1)
            result = re.sub(rf"({name_or}|[0-9]+)\(", r"\1*(", result)
        else:
            # 5(8 - 9) --> 5*(8 - 9)
            result = re.sub (r"([0-9]+)\(", r"\1*(", expr)
        
        # repeat as needed
        if result == expr:
            return result
        else:
            return self.add_implict_mult_token(result)
        
    def _eval_expression(self, expr: str) -> str:
        self.has_returned = False
        self.break_stack = []
        elaborated_expr = self.add_implict_mult_token(expr)
        elaborated_expr = textwrap.dedent(elaborated_expr).strip()
        return self._eval_recursive(ast.parse(elaborated_expr, mode=self.mode))
    
    def _eval_recursive(self, ast_node: ast.expr):
        # bottom out recursion
        if ast_node is None:
            return None

        if self.has_returned:
            return ast_node
        
        # handle each node as appropriate
        match type(ast_node):
            case ast.Module:
                for expr in ast_node.body:
                    result = self._eval_recursive(expr)
                    if self.has_returned:
                        break
                return self._eval_recursive(result)
            case ast.Expression:
                return self._eval_recursive(ast_node.body)
            case ast.Expr:
                return self._eval_recursive(ast_node.value)
            case ast.For:
                result = None
                self.break_stack.append(False)
                for val in self._eval_recursive(ast_node.iter):
                    self.stack_vars[ast_node.target.id] = val
                    for expr in ast_node.body:
                        result = self._eval_recursive(expr)
                        if self.has_returned or self.break_stack[-1]:
                            break
                    else:
                        del self.stack_vars[ast_node.target.id]
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
                        if self.has_returned or self.break_stack[-1]:
                            break
                    else:
                        continue
                self.break_stack = self.break_stack[:-1]
                return self._eval_recursive(result)
            case ast.Return:
                result = self._eval_recur_(ast_node.value)
                self.has_returned = True
                return result
            case ast.Break:
                self.break_stack[-1] = True
                return None
            case ast.If:
                if self._eval_recursive(ast_node.test):
                    for expr in ast_node.body:
                        result = self._eval_recursive(expr)
                        if self.has_returned:
                            break
                else:
                    for expr in ast_node.orelse:
                        result = self._eval_recursive(expr)
                        if self.has_returned:
                            break
                return self._eval_recursive(result)
            case ast.Match:
                subject_val = self._eval_recursive(ast_node.subject)
                for case in ast_node.cases:
                    if isinstance(case.pattern, ast.MatchAs) or (self._eval_recursive(case) == subject_val):
                        for expr in case.body:
                            result = self._eval_recursive(expr)
                            if self.has_returned:
                                break
                    else:
                            continue
                    break
                return self._eval_recursive(result)
            case ast.match_case:
                return self._eval_recursive(ast_node.pattern)
            case ast.MatchValue:
                return self._eval_recursive(ast_node.value)
            case ast.BinOp:
                op = SimpleExprEval.AST_NODE_TO_OPERATOR[type(ast_node.op)]
                return op(self._eval_recursive(ast_node.left), self._eval_recursive(ast_node.right))
            case ast.UnaryOp:
                op = SimpleExprEval.AST_NODE_TO_OPERATOR[type(ast_node.op)]
                return op(self._eval_recursive(ast_node.operand))
            case ast.Compare:
                match (op := SimpleExprEval.AST_NODE_TO_OPERATOR[type(ast_node.ops[0])]):
                    case operator.contains:
                        return op(self._eval_recursive(ast_node.comparators[0]), self._eval_recursive(ast_node.left))
                    case _:
                        return op(self._eval_recursive(ast_node.left), self._eval_recursive(ast_node.comparators[0]))
            case ast.BoolOp:
                op = SimpleExprEval.AST_NODE_TO_OPERATOR[type(ast_node.op)]
                assert len(ast_node.values) >= 2
                result = op(self._eval_recursive(ast_node.values[0]), self._eval_recursive(ast_node.values[1]))
                for i in range (len(ast_node.values) - 2):
                    result = op(result, self._eval_recursive(ast_node.values[i + 2]))
                return result
            case ast.Assign:
                assert len(ast_node.targets) == 1
                value = self.eval_recur_(ast_node.value)
                target = ast_node.targets[0].id
                self.stack_vars[target] = value
                return None
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
            case ast.Assert:
                assert self._eval_recursive(ast_node.test)
                return None
            case ast.Constant:
                return ast_node.value
            case ast.Name:
                if ast_node.id in self.local_vars:
                    return self.local_vars[ast_node.id]
                elif ast_node.id in self.stack_vars:
                    return self.stack_vars[ast_node.id]
            case ast.keyword:
                return (ast_node.arg, self._eval_recursive(ast_node.value))
            case ast.Attribute:
                return getattr(self._eval_recursive(ast_node.value), ast_node.attr)
            case ast.List:
                return [self._eval_recursive(list_item) for list_item in ast_node.elts]
            case ast.Slice:
                upper = self._eval_recursive(ast_node, upper)
                lower = self._eval_recursive(ast_node, lower)
                step  = self._eval_recursive(ast_node, step)
                return slice(lower, upper, step)
            case ast.Subscript:
                return operator.getitem(self._eval_recursive(ast_node, value), self._eval_recursive(ast_node.slice))

        raise NotImplementedError(f"Illegal operation attempted: {type(ast_node)}")