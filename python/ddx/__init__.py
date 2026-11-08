"""Reverse-mode automatic differentiation over a graph built at run time.

    >>> import ddx
    >>> @ddx.equation
    ... def f():
    ...     x = ddx.var("x")
    ...     y = ddx.var("y")
    ...     return ddx.exp(x) * ddx.sin(y)
    >>> values, gradient = f.jacobian({"x": 2.0, "y": 3.0})

``equation`` is a decorator or a call, and takes whatever the model returns: one
expression, or a tuple of them for a system. A point is a sequence, a
``(symbols, points)`` array, or a dict keyed by symbol name.

A model can also be written down rather than built, one string per function:

    >>> f = ddx.equation("exp(x) * sin(y)")
    >>> system = ddx.equation(["x*x + y*y - 4", "x*y - 1"])

The grammar is Python's arithmetic: ``+ - * /``, ``**`` for powers, one
comparison ``< <= > >= == !=`` per expression, calls to functions such as
``sin(x)``, ``pow(x, y)`` and ``select(c, a, b)``, and every free identifier a
symbol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._ddx import (Backend, Call, Expression, OptLevel, Restricted, VecLib, Want,
                   abs,  # noqa: A004  -- the opcode is spelled `abs`, as it is in C++
                   acos, acosh, asin, asinh, atan, atan2, atanh,
                   cbrt, cos, cosh, erf, errc, exp,
                   has_jit, hypot, log, log10, max,  # noqa: A004
                   min,  # noqa: A004
                   pow,  # noqa: A004
                   implicit, param, select, sign, sin, sinh, sqrt, tan, tanh,
                   var,
                   derivative, gradient, mat_constant, outer, pack, release_tapes,
                   sum,  # noqa: A004
                   Transposed, transpose, vec, vec_constant, vec_param, )
from ._options import Options

if TYPE_CHECKING:
	from collections.abc import Callable
	import ddx._ddx as _extension
	
	# Declares what the runtime adds and the generated stub cannot see, such as
	# Error.code and the members installed below.
	class Error(_extension.Error):
		"""A refusal from ddx, carrying the code it refused with."""
		code: errc
	
	class Equation(_extension.Equation):
		"""A frozen model, with the conveniences this module installs."""
		__name__: str
		options: Options
		
		def compile(self, **kwargs: Any) -> Equation:  # noqa: ANN401, D102
			pass
	
	Model = Callable[[], Expression | tuple[Expression, ...]]
	
	def equation(model: Model | str | list[str]) -> Equation:
		"""Build an equation from a model, or from expressions written as text --
		one string per function."""

else:
	from ._ddx import Equation, Error, equation

# The version compiled into the extension from CMake's project() line, which
# also works for an in-tree build that was never installed.
from ._ddx import __version__

# Equation.options converts between the pydantic model and the native struct.
def _get_options(self: Equation) -> Options:
	return Options._from_native(self._options)

def _set_options(self: Equation, options: Options) -> None:
	self._options = options._native()

def _compile(self: Equation, **kwargs: Any) -> Equation:  # noqa: ANN401
	"""Compile this equation's graph, and answer once the kernel is built.
	Keywords are ``Options`` fields; ``backend`` defaults to ``COMPILE``.
	Returns self, so
	``eq.compile(points=n).jacobian(xs)`` reads in one line.
	"""
	kwargs.setdefault("backend", Backend.COMPILE)
	self.options = Options(**kwargs)
	self.uses_kernel()
	return self

Equation.options = property(_get_options, _set_options)  # type: ignore[assignment]
Equation.compile = _compile  # type: ignore[method-assign]

__all__ = ["Backend", "Call", "Equation", "Error", "Expression", "OptLevel", "Restricted",
           "Options", "VecLib", "Want", "__version__", "abs", "acos", "acosh",
           "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "cos",
           "cosh", "equation", "erf", "errc", "exp",
           "has_jit", "hypot", "log", "log10", "max",
           "min", "param", "pow", "select", "sign", "sin", "sinh", "sqrt",
           "implicit",
           "tan", "tanh", "var",
           "derivative", "gradient", "mat_constant", "outer", "pack", "release_tapes",
           "sum", "Transposed", "transpose", "vec", "vec_constant", "vec_param", ]
