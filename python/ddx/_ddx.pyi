"""
ddx's runtime expression graph and LLVM JIT
"""
from __future__ import annotations
import enum
import numpy
import numpy.typing
import typing
__all__: list[str] = ['Backend', 'Call', 'Equation', 'Error', 'Expression',
                      'VecLib', 'Want', 'abs', 'acos', 'acosh', 'add', 'asin',
                      'asinh', 'atan', 'atan2', 'atanh', 'cbrt', 'cos',
                      'cosh', 'div', 'equation', 'erf', 'errc', 'exp',
                      'has_jit', 'hypot', 'le', 'OptLevel', 'log', 'log10',
                      'lt',
                      'max', 'min', 'mul', 'neg', 'pow', 'select', 'sign',
                      'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'var',
                      'implicit', 'param', 'Restricted',
                      'derivative', 'dot', 'gemv', 'gradient', 'vec_param',
                      'mat_constant', 'outer', 'pack', 'release_tapes', 'sum', 'Transposed', 'transpose', 'vec', 'vec_constant']
class Backend(enum.IntEnum):
    """
    Whether an equation compiles its graph.
    """
    COMPILE: typing.ClassVar[Backend]  # value = <Backend.COMPILE: 1>
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
    def dense_hessian(self, x: typing.Any) -> tuple: pass
    def hessian(self, x: typing.Any) -> tuple: pass
    def hvp(self, v: typing.Any, x: typing.Any) -> tuple: pass
    def jacobian(self, x: typing.Any) -> tuple: pass
    def jacobian_jvp(self, v: typing.Any, x: typing.Any) -> tuple: pass
    def residual(self, x: typing.Any) -> list[float]: pass
    def restrict(self, names: list[str]) -> Restricted: pass
    def jvp(self, v: typing.Any, x: typing.Any) -> tuple: pass
    def nodes(self, *, want: Want = ...) -> int: pass
    def vjp(self, w: typing.Any, x: typing.Any) -> tuple: pass
    def wait_for_kernel(self, *, want: Want = ...) -> bool: pass
    @property
    def arity(self) -> int: pass
    @property
    def hessian_colours(self) -> int: pass
    @property
    def inputs(self) -> list[str]: pass
    @property
    def outputs(self) -> int: pass
    @property
    def params(self) -> list[str]: pass
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
    @property
    def shape(self) -> tuple[int, ...]: pass
    def __init__(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None: pass
    def __len__(self) -> int: pass
    def __getitem__(self, i: typing.SupportsInt | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __le__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __le__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __lt__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __lt__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    def __matmul__(self, arg0: Expression) -> Expression: pass
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
    def __iadd__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __iadd__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __isub__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __isub__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __imul__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __imul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
    @typing.overload
    def __itruediv__(self, arg0: Expression) -> Expression: pass
    @typing.overload
    def __itruediv__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> Expression: pass
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
    
class OptLevel(enum.IntEnum):
    """
    LLVM's -O0 to -O3, for the IR pipeline and for codegen.
    """
    O0: typing.ClassVar[OptLevel]  # value = <OptLevel.O0: 0>
    O1: typing.ClassVar[OptLevel]  # value = <OptLevel.O1: 1>
    O2: typing.ClassVar[OptLevel]  # value = <OptLevel.O2: 2>
    O3: typing.ClassVar[OptLevel]  # value = <OptLevel.O3: 3>
    @classmethod
    def __new__(cls, value): pass
    def __format__(self, format_spec): pass
class Transposed:
    def __matmul__(self, arg0: Expression) -> Expression: pass
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
class Restricted:
    def jacobian(self, x: typing.Any) -> tuple: pass
    def jacobian_jvp(self, v: typing.Any, x: typing.Any) -> tuple: pass
    @property
    def jacobian_columns(self) -> int: pass
    @property
    def symbols(self) -> list[str]: pass
    
class Want(enum.IntEnum):
    """
    Which blocks a bound call fills.
    """
    DENSE_HESSIAN: typing.ClassVar[Want]  # value = <Want.DENSE_HESSIAN: 8>
    GRADIENT: typing.ClassVar[Want]  # value = <Want.GRADIENT: 6>
    HESSIAN: typing.ClassVar[Want]  # value = <Want.HESSIAN: 2>
    HVP: typing.ClassVar[Want]  # value = <Want.HVP: 3>
    JACOBIAN: typing.ClassVar[Want]  # value = <Want.JACOBIAN: 1>
    JACOBIAN_JVP: typing.ClassVar[Want]  # value = <Want.JACOBIAN_JVP: 7>
    JVP: typing.ClassVar[Want]  # value = <Want.JVP: 5>
    VALUE: typing.ClassVar[Want]  # value = <Want.VALUE: 0>
    VJP: typing.ClassVar[Want]  # value = <Want.VJP: 4>
    @classmethod
    def __new__(cls, value): pass
    def __format__(self, format_spec): pass
    
class _Options:
    __hash__: typing.ClassVar[None] = None
    backend: Backend
    contract: bool
    loop_vectorize: bool
    slp_vectorize: bool
    time_passes: bool
    veclib: VecLib
    def __eq__(self, arg0: _Options) -> bool: pass
    def __init__(self) -> None: pass
    @property
    def codegen_level(self) -> OptLevel: pass
    @codegen_level.setter
    def codegen_level(self, arg0: OptLevel) -> None: pass
    @property
    def lanes(self) -> int | None: pass
    @lanes.setter
    def lanes(self, arg0: typing.SupportsInt | typing.SupportsIndex | None) -> None: pass
    @property
    def opt_level(self) -> OptLevel: pass
    @opt_level.setter
    def opt_level(self, arg0: OptLevel) -> None: pass
    @property
    def points(self) -> int: pass
    @points.setter
    def points(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: pass
    
class errc(enum.IntEnum):
    """
    Why ddx refused; Error.code carries one.
    """
    short_point: typing.ClassVar[errc]  # value = <errc.short_point: 0>
    wrong_arity: typing.ClassVar[errc]  # value = <errc.wrong_arity: 1>
    wrong_direction: typing.ClassVar[errc]  # value = <errc.wrong_direction: 2>
    unknown_symbol: typing.ClassVar[errc]  # value = <errc.unknown_symbol: 3>
    index_out_of_range: typing.ClassVar[errc]  # value = <errc.index_out_of_range: 4>
    wrong_column_count: typing.ClassVar[errc]  # value = <errc.wrong_column_count: 5>
    outside_equation: typing.ClassVar[errc]  # value = <errc.outside_equation: 6>
    no_graph: typing.ClassVar[errc]  # value = <errc.no_graph: 7>
    name_clash: typing.ClassVar[errc]  # value = <errc.name_clash: 8>
    closed_twice: typing.ClassVar[errc]  # value = <errc.closed_twice: 9>
    nested_implicit: typing.ClassVar[errc]  # value = <errc.nested_implicit: 10>
    extern_order: typing.ClassVar[errc]  # value = <errc.extern_order: 11>
    extern_shape: typing.ClassVar[errc]  # value = <errc.extern_shape: 12>
    not_univariate: typing.ClassVar[errc]  # value = <errc.not_univariate: 13>
    not_single_output: typing.ClassVar[errc]  # value = <errc.not_single_output: 14>
    unsupported_scalar: typing.ClassVar[errc]  # value = <errc.unsupported_scalar: 15>
    tensor_shape: typing.ClassVar[errc]  # value = <errc.tensor_shape: 16>
    tensor_kernel: typing.ClassVar[errc]  # value = <errc.tensor_kernel: 17>
    jit_target: typing.ClassVar[errc]  # value = <errc.jit_target: 18>
    jit_module: typing.ClassVar[errc]  # value = <errc.jit_module: 19>
    jit_verify: typing.ClassVar[errc]  # value = <errc.jit_verify: 20>
    jit_lookup: typing.ClassVar[errc]  # value = <errc.jit_lookup: 21>
    bad_syntax: typing.ClassVar[errc]  # value = <errc.bad_syntax: 22>
    unknown_function: typing.ClassVar[errc]  # value = <errc.unknown_function: 23>
    wrong_argument_count: typing.ClassVar[errc]  # value = <errc.wrong_argument_count: 24>
    unclosed_implicit: typing.ClassVar[errc]  # value = <errc.unclosed_implicit: 25>
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
def equation(source: str) -> Equation:
    ...
@typing.overload
def equation(source: list[str]) -> Equation:
    ...
@typing.overload
def equation(model: typing.Any) -> typing.Any:
    ...
def erf(x: Expression) -> Expression:
    ...
def exp(x: Expression) -> Expression:
    ...
def hypot(x: Expression, y: Expression) -> Expression:
    ...
def le(x: Expression, y: Expression) -> Expression:
    ...
def log(x: Expression) -> Expression:
    ...
def log10(x: Expression) -> Expression:
    ...
def lt(x: Expression, y: Expression) -> Expression:
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
def implicit(name: str, residual: typing.Callable[[Expression], Expression | float]) -> Expression:
    """An input the caller solved for, a root of residual(z)."""
def param(name: str) -> Expression:
    ...
def var(name: str) -> Expression:
    ...
def derivative(f: Expression, symbol: str) -> Expression:
    ...
def dot(u: Expression, v: Expression) -> Expression:
    ...
def gemv(matrix: Expression, v: Expression, transposed: bool = False) -> Expression:
    """M v, or M^T v when transposed: a mat_constant table or a computed matrix."""
def gradient(f: Expression) -> list[Expression]:
    """df/dx for every symbol, in declaration order."""
@typing.overload
def mat_constant(rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex, data: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> Expression:
    """A constant rows x cols matrix from column-major data."""
@typing.overload
def mat_constant(matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> Expression:
    """A constant matrix from a 2-D array."""
def outer(u: Expression, w: Expression) -> Expression:
    """u w^T, the matrix of every product of an element of u with one of w."""
def pack(scalars: typing.Sequence[Expression]) -> Expression:
    """The vector of these scalars, in order."""
def release_tapes() -> None:
    """Free this thread's sweep tapes; the next sweep grows them again."""
def transpose(matrix: Expression) -> Transposed:
    """M^T, as the left side of @: transpose(M) @ v is gemv(M, v, True)."""
def sum(v: Expression) -> Expression:
    ...
def vec(name: str, n: typing.SupportsInt | typing.SupportsIndex) -> Expression:
    """n symbols named name[0] .. name[n-1], packed."""
def vec_param(name: str, n: typing.SupportsInt | typing.SupportsIndex) -> Expression:
    """n params named name[0] .. name[n-1], packed."""
def vec_constant(data: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> Expression:
    ...
__version__: str = ...
has_jit: bool = True
