import inspect
from functools import wraps
import torch
from torch import nn

def wrap_kwargs(f):
    """
    Given a callable f that can consume some named arguments,
    wrap it with a kwargs that passes back any unused args
    EXAMPLES
    --------
    Basic usage:
    def foo(x, y=None):
        return x
    wrap_kwargs(foo)(0, y=1, z=2) == (0, {'z': 2})
    --------
    The wrapped function can return its own argument dictionary,
    which gets merged with the new kwargs.
    def foo(x, y=None):
        return x, {}
    wrap_kwargs(foo)(0, y=1, z=2) == (0, {'z': 2})
    def foo(x, y=None):
        return x, {"y": y, "z": None}
    wrap_kwargs(foo)(0, y=1, z=2) == (0, {'y': 1, 'z': 2})
    --------
    The wrapped function can have its own kwargs parameter:
    def foo(x, y=None, **kw_args):
        return x, {}
    wrap_kwargs(foo)(0, y=1, z=2) == (0, {})
    --------
    Partial functions and modules work automatically:
    class Module:
        def forward(self, x, y=0):
            return x, {"y": y+1}
    m = Module()
    wrap_kwargs(m.forward)(0, y=1, z=2) == (0, {'y': 2, 'z': 2})
    """
    sig = inspect.signature(f)
    # Check if f already has kwargs
    has_kwargs = any([
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    ])
    if has_kwargs:
        @wraps(f)
        def f_kwargs(*args, **kwargs):
            y = f(*args, **kwargs)
            if isinstance(y, tuple) and isinstance(y[-1], dict):
                return y
            else:
                return y, {}
    else:
        param_kwargs = inspect.Parameter("kwargs", kind=inspect.Parameter.VAR_KEYWORD)
        sig_kwargs = inspect.Signature(parameters=list(sig.parameters.values())+[param_kwargs])
        @wraps(f)
        def f_kwargs(*args, **kwargs):
            bound = sig_kwargs.bind(*args, **kwargs)
            if "kwargs" in bound.arguments:
                kwargs = bound.arguments.pop("kwargs")
            else:
                kwargs = {}
            y = f(**bound.arguments)
            if isinstance(y, tuple) and isinstance(y[-1], dict):
                return *y[:-1], {**y[-1], **kwargs}
            else:
                return y, kwargs
    return f_kwargs