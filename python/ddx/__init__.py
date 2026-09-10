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
comparison ``< <= > >= == !=`` per expression, calls by the name ddx gives the
operation, and every free identifier a symbol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._ddx import (Backend, Call, Expression, Level, VecLib, Want, abs,  # noqa: A004  -- the opcode is spelled `abs`, as it is in C++
                   acos, acosh, add, asin, asinh, atan, atan2, atanh,
                   cbrt, cos, cosh, div, erf, errc, exp,
                   has_jit, has_opencl, hypot,
                   log, log10, max,  # noqa: A004
                   min,  # noqa: A004
                   mul, neg,
                   pow,  # noqa: A004
                   select, sign, sin, sinh, sqrt, tan, tanh,
                   var, )
from ._options import Options

if TYPE_CHECKING:
	from collections.abc import Callable
	from os import PathLike
	import ddx._ddx as _extension
	
	StrPath = str | PathLike[str]
	
	# What the runtime adds and a generated stub cannot see: the exception
	# translator's code, the pydantic-facing members installed below, and the
	# decorator, which the bindings hand over as py::object.
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
	
	def equation(
		model: Model | str | list[str],
		*,
		cache: StrPath | None = None,
		remember: bool = False,
	) -> Equation:
		"""Build an equation from a model, or from expressions written as text --
		one string per function.  ``cache`` is a file to keep it in, and
		``remember`` makes a call at a point already asked come back off the
		last one."""

	def load(path: StrPath, *, remember: bool = False) -> Equation:
		"""Read an equation from a file, running no model and sweeping nothing."""

else:
	from ._ddx import Equation, Error, equation, load

# From the extension, which was built from CMake's project() line: the same
# number an installed package's metadata carries, and there for an in-tree
# build that was never installed.
from ._ddx import __version__

# The two places pydantic sits between a caller and the struct.
def _get_options(self: Equation) -> Options:
	return Options._from_native(self._options)

def _set_options(self: Equation, options: Options) -> None:
	self._options = options._native()

def _compile(self: Equation, **kwargs: Any) -> Equation:  # noqa: ANN401
	"""Compile this equation's graph, and answer once the kernel has landed.
	Keywords are ``Options`` fields; ``backend`` defaults to ``COMPILE``, and
	``DEVICE`` builds it for the OpenCL device instead. Returns self, so
	``eq.compile(points=n).jacobian(xs)`` reads in one line.
	"""
	kwargs.setdefault("backend", Backend.COMPILE)
	self.options = Options(**kwargs)
	self.wait_for_kernel()
	return self

Equation.options = property(_get_options, _set_options)  # type: ignore[assignment]
Equation.compile = _compile  # type: ignore[method-assign]

__all__ = ["Backend", "Call", "Equation", "Error", "Expression", "Level",
           "Options", "VecLib", "Want", "__version__",	"abs", "acos", "acosh",	"add",
           "asin", "asinh",	"atan", "atan2", "atanh", "cbrt", "cos",
           "cosh", "div", "equation", "erf", "errc", "exp",
           "has_jit", "has_opencl", "hypot", "load", "log", "log10", "max",
           "min", "mul", "neg", "pow", "select", "sign", "sin", "sinh", "sqrt",
           "tan", "tanh", "var",]
