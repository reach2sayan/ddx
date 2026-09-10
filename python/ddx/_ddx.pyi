"""
ddx's runtime expression graph, LLVM JIT and OpenCL device
"""
from __future__ import annotations
import enum
import os
import typing
__all__: list[str] = ['Backend', 'Call', 'Equation', 'Error', 'Expression',
                      'VecLib', 'Want', 'abs', 'acos', 'acosh', 'add', 'asin',
                      'asinh', 'atan', 'atan2', 'atanh', 'cbrt', 'cos',
                      'cosh', 'div', 'equation', 'erf', 'errc', 'exp',
                      'has_jit', 'has_opencl', 'hypot', 'Level', 'load', 'log', 'log10',
                      'max', 'min', 'mul', 'neg', 'pow', 'select', 'sign',
                      'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'var']
class Backend(enum.IntEnum):
    """
    Whether an equation compiles its graph.
    """
    ADAPT: typing.ClassVar[Backend]  # value = <Backend.ADAPT: 2>
    COMPILE: typing.ClassVar[Backend]  # value = <Backend.COMPILE: 1>
    DEVICE: typing.ClassVar[Backend]  # value = <Backend.DEVICE: 3>
    INTERPRET: typing.ClassVar[Backend]  # value = <Backend.INTERPRET: 0>
    @classmethod
    def __new__(cls, value): pass
    def __format__(self, format_spec): pass
    
class Call:
    """
    A call bound to its buffers: write the next point into `x`, call, read the
    blocks back.  Equation.buffer() is the only way to get one.
    """
    def __call__(self) -> None: pass
    def __repr__(self) -> str: pass
    @property
    def hessian(self) -> typing.Any: pass
    @property
    def jacobian(self) -> typing.Any: pass
    @property
    def value(self) -> typing.Any: pass
    @property
    def x(self) -> typing.Any: pass
    
class Equation:
    _options: _Options
    def __call__(self, x: typing.Any) -> typing.Any: pass
    def __repr__(self) -> str: pass
    def buffer(self, x: typing.Any, *, want: Want = ...) -> Call: pass
    def evaluate(self, x: typing.Any) -> typing.Any: pass
    def gradient(self, x: typing.Any) -> typing.Any: pass
    def hessian(self, x: typing.Any) -> tuple: pass
    def hvp(self, v: typing.Any, x: typing.Any) -> tuple: pass
    def jacobian(self, x: typing.Any) -> tuple: pass
    def jvp(self, v: typing.Any, x: typing.Any) -> tuple: pass
    def nodes(self, *, want: Want = ...) -> int: pass
    def save(self, path: str | os.PathLike) -> None: pass
    def to_dot(self, *, all: bool = False) -> str: pass
    def verify(self, path: str | os.PathLike) -> None: pass
    def vjp(self, w: typing.Any, x: typing.Any) -> tuple: pass
    def wait_for_kernel(self, *, want: Want = ...) -> bool: pass
    @property
    def arity(self) -> int: pass
    @property
    def device_status(self) -> str | None: pass
    @property
    def hessian_colors(self) -> int: pass
    @property
    def loaded(self) -> bool: pass
    @property
    def outputs(self) -> int: pass
    @property
    def symbols(self) -> list[str]: pass
    @property
    def uses_kernel(self) -> bool: pass
    
class Error(RuntimeError):
    pass
class Expression:
    def __abs__(self) -> Expression: pass
    @typing.overload
    def __add__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __add__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    def __bool__(self) -> bool: pass
    @typing.overload
    def __eq__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __eq__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __ge__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __ge__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __gt__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __gt__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    __hash__: typing.ClassVar[None] = None
    def __init__(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None: pass
    @typing.overload
    def __le__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __le__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __lt__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __lt__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __mul__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __mul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __ne__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __ne__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    def __neg__(self) -> Expression: pass
    @typing.overload
    def __pow__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __pow__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __radd__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __radd__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    def __repr__(self) -> str: pass
    @typing.overload
    def __rmul__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __rmul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __rpow__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __rpow__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __rsub__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __rsub__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __rtruediv__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __rtruediv__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __sub__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __sub__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __truediv__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __truediv__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    
class Level(enum.IntEnum):
    """
    LLVM's -O0 to -O3, for the IR pipeline and for codegen.
    """
    O0: typing.ClassVar[Level]  # value = <Level.O0: 0>
    O1: typing.ClassVar[Level]  # value = <Level.O1: 1>
    O2: typing.ClassVar[Level]  # value = <Level.O2: 2>
    O3: typing.ClassVar[Level]  # value = <Level.O3: 3>
    @classmethod
    def __new__(cls, value): pass
    def __format__(self, format_spec): pass
class VecLib(enum.IntEnum):
    """
    Which vector math library a lane may call.
    """
    AUTO: typing.ClassVar[VecLib]  # value = <VecLib.AUTO: 1>
    LIBMVEC: typing.ClassVar[VecLib]  # value = <VecLib.LIBMVEC: 2>
    NONE: typing.ClassVar[VecLib]  # value = <VecLib.NONE: 0>
    @classmethod
    def __new__(cls, value): pass
    def __format__(self, format_spec): pass
class Want(enum.IntEnum):
    """
    Which blocks a bound call fills.
    """
    GRADIENT: typing.ClassVar[Want]  # value = <Want.GRADIENT: 6>
    HESSIAN: typing.ClassVar[Want]  # value = <Want.HESSIAN: 2>
    HVP: typing.ClassVar[Want]  # value = <Want.HVP: 3>
    JACOBIAN: typing.ClassVar[Want]  # value = <Want.JACOBIAN: 1>
    JVP: typing.ClassVar[Want]  # value = <Want.JVP: 5>
    VALUE: typing.ClassVar[Want]  # value = <Want.VALUE: 0>
    VJP: typing.ClassVar[Want]  # value = <Want.VJP: 4>
    @classmethod
    def __new__(cls, value): pass
    def __format__(self, format_spec): pass
    
class _Options:
    __hash__: typing.ClassVar[None] = None
    backend: Backend
    cache_dir: str
    contract: bool
    device: str
    loop_vectorize: bool
    retain_object: bool
    slp: bool
    time_passes: bool
    veclib: VecLib
    def __eq__(self, arg0: _Options) -> bool: pass
    def __init__(self) -> None: pass
    @property
    def codegen_level(self) -> Level: pass
    @codegen_level.setter
    def codegen_level(self, arg0: Level) -> None: pass
    @property
    def lanes(self) -> int | None: pass
    @lanes.setter
    def lanes(self, arg0: typing.SupportsInt | typing.SupportsIndex | None) -> None: pass
    @property
    def hot_points(self) -> int: pass
    @hot_points.setter
    def hot_points(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: pass
    @property
    def opt_level(self) -> Level: pass
    @opt_level.setter
    def opt_level(self, arg0: Level) -> None: pass
    @property
    def points(self) -> int: pass
    @points.setter
    def points(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: pass
    @property
    def warm_points(self) -> int: pass
    @warm_points.setter
    def warm_points(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: pass
    
class errc(enum.IntEnum):
    """
    Why ddx refused; Error.code carries one.
    """
    archive_corrupt: typing.ClassVar[errc]  # value = <errc.archive_corrupt: 12>
    archive_io: typing.ClassVar[errc]  # value = <errc.archive_io: 10>
    archive_mismatch: typing.ClassVar[errc]  # value = <errc.archive_mismatch: 13>
    bad_archive: typing.ClassVar[errc]  # value = <errc.bad_archive: 11>
    index_out_of_range: typing.ClassVar[errc]  # value = <errc.index_out_of_range: 3>
    jit_lookup: typing.ClassVar[errc]  # value = <errc.jit_lookup: 18>
    jit_module: typing.ClassVar[errc]  # value = <errc.jit_module: 15>
    jit_object: typing.ClassVar[errc]  # value = <errc.jit_object: 16>
    jit_target: typing.ClassVar[errc]  # value = <errc.jit_target: 14>
    jit_verify: typing.ClassVar[errc]  # value = <errc.jit_verify: 17>
    no_arena: typing.ClassVar[errc]  # value = <errc.no_arena: 5>
    no_graph: typing.ClassVar[errc]  # value = <errc.no_graph: 6>
    not_univariate: typing.ClassVar[errc]  # value = <errc.not_univariate: 8>
    sealed_arena: typing.ClassVar[errc]  # value = <errc.sealed_arena: 7>
    short_point: typing.ClassVar[errc]  # value = <errc.short_point: 0>
    unknown_symbol: typing.ClassVar[errc]  # value = <errc.unknown_symbol: 2>
    unsupported_scalar: typing.ClassVar[errc]  # value = <errc.unsupported_scalar: 9>
    wrong_arity: typing.ClassVar[errc]  # value = <errc.wrong_arity: 1>
    wrong_column_count: typing.ClassVar[errc]  # value = <errc.wrong_column_count: 4>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
def abs(x: Expression) -> Expression:
    ...
def acos(x: Expression) -> Expression:
    ...
def acosh(x: Expression) -> Expression:
    ...
def add(x: Expression, y: Expression) -> Expression:
    ...
def asin(x: Expression) -> Expression:
    ...
def asinh(x: Expression) -> Expression:
    ...
def atan(x: Expression) -> Expression:
    ...
def atan2(x: Expression, y: Expression) -> Expression:
    ...
def atanh(x: Expression) -> Expression:
    ...
def cbrt(x: Expression) -> Expression:
    ...
def cos(x: Expression) -> Expression:
    ...
def cosh(x: Expression) -> Expression:
    ...
def div(x: Expression, y: Expression) -> Expression:
    ...
@typing.overload
def equation(source: str, *, cache: str | os.PathLike | None = None, remember: bool = False) -> Equation:
    ...
@typing.overload
def equation(source: list[str], *, cache: str | os.PathLike | None = None, remember: bool = False) -> Equation:
    ...
@typing.overload
def equation(model: typing.Any, *, cache: str | os.PathLike | None = None, remember: bool = False) -> typing.Any:
    ...
def erf(x: Expression) -> Expression:
    ...
def exp(x: Expression) -> Expression:
    ...
def hypot(x: Expression, y: Expression) -> Expression:
    ...
def log(x: Expression) -> Expression:
    ...
def log10(x: Expression) -> Expression:
    ...
def max(x: Expression, y: Expression) -> Expression:
    ...
def min(x: Expression, y: Expression) -> Expression:
    ...
def mul(x: Expression, y: Expression) -> Expression:
    ...
def neg(x: Expression) -> Expression:
    ...
def pow(x: Expression, y: Expression) -> Expression:
    ...
def select(cond: Expression, if_true: Expression, if_false: Expression) -> Expression:
    ...
def sign(x: Expression) -> Expression:
    ...
def sin(x: Expression) -> Expression:
    ...
def sinh(x: Expression) -> Expression:
    ...
def sqrt(x: Expression) -> Expression:
    ...
def tan(x: Expression) -> Expression:
    ...
def tanh(x: Expression) -> Expression:
    ...
def var(name: str) -> Expression:
    ...
__version__: str = ...
has_jit: bool = True
has_opencl: bool = True
