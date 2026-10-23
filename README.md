[![GCC 14](https://github.com/reach2sayan/ddx/actions/workflows/gcc.yml/badge.svg?branch=main)](https://github.com/reach2sayan/ddx/actions/workflows/gcc.yml)
[![Clang 20](https://github.com/reach2sayan/ddx/actions/workflows/clang.yml/badge.svg?branch=main)](https://github.com/reach2sayan/ddx/actions/workflows/clang.yml)
[![MSVC 2022](https://github.com/reach2sayan/ddx/actions/workflows/msvc.yml/badge.svg?branch=main)](https://github.com/reach2sayan/ddx/actions/workflows/msvc.yml)
[![Conan](https://github.com/reach2sayan/ddx/actions/workflows/conan.yml/badge.svg?branch=main)](https://github.com/reach2sayan/ddx/actions/workflows/conan.yml)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white)](#install)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](#python)
[![License](https://img.shields.io/badge/license-BSL--1.0-4B8BBE)](LICENSE.txt)

# ddx

A C++23 library for differentiating expressions. Build a function over named
symbols while the program runs — terms looped over a data file, a model read
from configuration, an expression typed by a user, a loss that switches shape
on a comparison — and ask it for values, gradients, Jacobians and Hessians. One
point at a time or a batch of thousands in one call, interpreted or compiled
through LLVM. There are
[Python bindings](#python) over the same runtime, and a header-only
[compile-time API](#compile-time-expressions) for expressions whose shape is
known when you compile.

```cpp
#include "ddx.hpp"
#include "rt/equation.hpp"
using namespace ddx;

struct Sample { double t, y; };
const std::vector<Sample> data{{0.0, 2.00}, {1.0, 1.21}, {2.0, 0.74}, {3.0, 0.45}};

// Huber loss of the fit m(t) = a·exp(−b·t).
const double delta = 0.05;
const auto eq = rt::equation([&] {
  const auto a = rt::var("a");
  const auto b = rt::var("b");

  rt::Expression<double> loss = 0.0;
  for (const auto &s : data) {
    const auto r = s.y - a * exp(-b * s.t);
    loss += select(abs(r) < delta, r * r, delta * (2.0 * abs(r) - delta));
  }
  return loss;
});

*eq.evaluate(2.0, 0.5);     // 4.1343959887e-05
*eq.jacobian(2.0, 0.5);     // {-0.0010757424682, 0.0150678475595}
*eq.hessian(2.0, 0.5);      // 2 × 2, row-major
```

## Install

GCC 14+, Clang 20+ over libstdc++ 14+ (not libc++), or MSVC 2022 with
`/std:c++latest`; CMake 3.26+. Boost and GoogleTest are fetched at configure
time.

```cmake
add_subdirectory(ddx)
target_link_libraries(my_app PRIVATE ddx::rt)      # or ddx::jit
```

```cmake
find_package(ddx CONFIG REQUIRED COMPONENTS rt jit)
```

| Target | Is |
|---|---|
| `ddx::ddx` | header-only: compile-time expressions |
| `ddx::rt` | runtime equations |
| `ddx::jit` | `ddx::rt` with the LLVM backend (`-DDDX_BUILD_JIT=ON`, needs LLVM 20) |

Conan: `conan create .` packages it and `conan editable add .` uses the tree
in place; consumers `find_package(ddx)` as above.  Also packaged for vcpkg
(`--overlay-ports=contrib/vcpkg/ports`) and NuGet (`ddx`).  None of the three
carries the JIT. Presets: `debug`, `release`, `debug_jit`,
`release_jit`, `python`.

## Equations

`rt::equation(callback)` runs the callback and returns the equation. Symbols are
named inside it with `rt::var(name)` — any run-time string — and are ordered as
first named; that order is the order of points and results.

```cpp
const auto f = rt::equation([] { return exp(rt::var("x")) * sin(rt::var("y")); });

// A system: return an array (count in the type) or a vector (rt::System).
const auto sys = rt::equation([] {
  const auto x = rt::var("x");
  const auto y = rt::var("y");
  return std::array{x * y + sin(x), exp(x) - y * y};
});
```

Name symbols in separate statements when order matters — C++ leaves argument
evaluation order unspecified. Another scalar type: `rt::equation<T>` with
`rt::var<T>`.

| Inside the callback | Is |
|---|---|
| `rt::var(name)` | a symbol: differentiated |
| `rt::param(name)` | an input that is read, never differentiated |
| `rt::implicit(name, [&](auto z) { return F; })` | an input that is a root of `F`; derivatives carry ∂z/∂x = −F_x/F_z |
| `rt::extern_fn({.name, .arity, .order}, fn)` | your own function; `fn(in, out)` writes the value, then the partials (upper triangle of the second at `order = 2`) |
| `rt::vec(name, n)`, `rt::vec_constant(data)`, `rt::mat_constant(r, c, data)`, `rt::pack({…})`, `rt::outer(u, w)` | vectors and matrices (`rt/tensor.hpp`): `n` ordinary symbols as one node |

A point supplies symbols, then parameters, then implicits. `eq.residual(point)`
evaluates each implicit's `F`; ddx never solves it.

Vectors are elementwise under every operation, with `sum`, `dot` and
`gemv(M, v, transposed = false)`, and `%` spells both: `u % v` is `dot(u, v)`,
`M % v` is `gemv(M, v)` and `rt::transpose(M) % v` is `gemv(M, v, true)`,
bit for bit (`@` in Python). `v[i]` reads one element. `select` over vectors
and matrices takes the whole arm under a scalar condition and element by
element under a condition of the arms' shape, a scalar arm taking the other's
shape; the arm not taken never reaches a derivative, even where it is
infinite. They compute at
`double`, are never an output, and are computed by a kernel table —
`rt::use_tensor_kernels(table)` or `eq.tensor_kernels(table)`, built with
`rt::tensor_kernel_table<K>()` over a type satisfying `rt::TensorKernels`;
`rt/eigen_kernels.hpp` supplies `rt::eigen::table`.

Matrices are elementwise the same way: a `mat_constant` table taking part in
arithmetic is its matrix, so `exp(-(A0 + T * A1) / (R * T))` over tables is a
matrix of T computed in a few nodes, and `gemv(M, v, transposed)` and
`dot(M, N)` read it. `rt::outer(u, w)` is `u wᵀ`. Shapes must agree, rows and
columns both, and a scalar expression meets a matrix as a factor (`T * B`,
`B / T`), as it meets a vector. `dense_hessian` stays O(n²) over a matrix of tables,
parameters and scalar symbols: beside a vector `x`, each scalar `var` (T, P)
costs one more reverse sweep, and the block covers every symbol. A
matrix that moves with a symbol and is then multiplied (`gemv(exp(outer(x, x)),
x)`) has no O(n²) form, and `dense_hessian` refuses it with
`errc::tensor_shape`; `hessian`, `hvp` and `jacobian_jvp` answer it.

An equation can also be a string, in Python's arithmetic grammar
(`rt/text/equation.hpp`), answering `result<Equation>`:

```cpp
const auto eq  = *rt::equation("sin(x)*y + 3");
const auto sys = *rt::equation("x*x + y*y - 4", "x*y - 1");
```

## Expressions

| Kind | Available |
|---|---|
| Arithmetic | `+` `-` `*` `/` unary `-` `+=` `-=` `*=` `/=` |
| Unary | `sin` `cos` `tan` `exp` `log` `log10` `sqrt` `cbrt` `abs` `sign` `asin` `acos` `atan` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` `erf` |
| Binary | `pow` `atan2` `hypot` `max` `min` |
| Comparison | `<` `<=` `>` `>=` `==` `!=` — answer `1.0` or `0.0` |
| Conditional | `select(cond, if_true, if_false)` |

`select` evaluates both arms and differentiates the taken one; the condition is
never differentiated. `abs′(0) = 0`, `max`/`min` split a tie in half, and
`pow(0, b)` has zero partials.

## Points

```cpp
*eq.jacobian(1.3, 0.7);                             // positional
*eq.jacobian(std::array{1.3, 0.7});                 // any range
*eq.jacobian(named<"y">(0.7), named<"x">(1.3));     // by name
*eq.jacobian("y"_s = 0.7, "x"_s = 1.3);
*eq.jacobian(std::map<std::string, double>{{"y", 0.7}, {"x", 1.3}});   // varmap
```

A named point must name every input (`errc::short_point`). `eq.point(args…)`
builds the positional vector once for reuse.

## Calls

Per-point calls answer `result<T>` (`std::expected<T, ddx::error>`); for
f: ℝⁿ → ℝᵐ:

| Call | Answers |
|---|---|
| `evaluate(x)` | f(x): `T`, or `std::vector<T>` for a system |
| `jacobian(x)` | J, row-major m × n — the gradient when m = 1 |
| `gradient(x)` | J without computing f |
| `hessian(x)` | dense m × n × n |
| `dense_hessian(x)` | n × n from one O(n²) sweep, for densely coupled vector models; one output |
| `jvp(v, x)` | J v, length m |
| `vjp(w, x)` | wᵀ J, length n |
| `hvp(v, x)`, `hvp(name, x)` | H v, length n; one output |
| `jacobian_jvp(v, x)`, `jacobian_jvp(name, x)` | ∂J/∂v, m × n |
| `univariate_derivative<K>(x0)` | K-th derivative; one symbol, one output |

The batch calls take one input column per input and one output pointer per
output column, each `n` points long, write in place without allocating, and
answer `result<void>`:

```cpp
eq.evaluate(xs, f, n);
eq.jacobian(xs, f, g, n);
eq.gradient(xs, g, n);
eq.hessian(xs, f, g, h, n);
eq.jvp(xs, vs, f, p, n);
eq.vjp(xs, ws, f, p, n);
eq.hvp(xs, vs, f, g, p, n);
eq.dense_hessian(xs, f, g, h, n);
eq.jacobian_jvp(xs, vs, f, g, dg, n);
```

Size each block with `value_columns()`, `jacobian_columns()`,
`hessian_columns()`, `jvp_columns()`, `vjp_columns()`, `hvp_columns()`,
`dense_hessian_columns()`. The Jacobian block holds only structural nonzeros —
`jacobian_pattern()` maps columns to cells — and the Hessian block is
colour-compressed, read with `hessian_cell(i, j)` or `hessian_pattern()`.
`rt/eigen.hpp` adds `rt::eigen::columns_of(M)` and `rt::eigen::scatter(...)`.

`eq.restrict({"T", "P"})` is a view computing only those symbols' Jacobian
columns.

## Inspecting

| Member | Answers |
|---|---|
| `poisoned()`, `status()` | whether building failed, and the `ddx::error` |
| `arity()`, `outputs()` | symbol count n, function count m |
| `symbols()`, `params()`, `inputs()`, `input_count()` | names in declaration order |
| `prepare({rt::Want::Jacobian, …})` | build those lanes up front |

A poisoned equation answers every call with its error. `errc` and `error`
format and stream as text. Nothing throws.

All members are `const` and thread-safe except `options()` and `tensor_kernels(table)`.
The equation owns everything it needs and is move-only; symbols are valid only
inside the callback that named them. Each thread keeps the tapes it sweeps into
at the largest size it has needed; `rt::release_tapes()` (Python
`ddx.release_tapes()`) frees the calling thread's, for a pool worker after a
large sweep.

## Compiling

Link `ddx::jit` and set options; the same calls answer the same results.
Nothing compiles unless asked, and no call waits for a compile:

```cpp
eq.options({.backend = rt::Backend::Compile, .points = 4096});
eq.jacobian(xs, f, g, n);      // interpreted until the kernel lands
eq.wait_for_kernel();          // the only call that blocks
eq.uses_kernel();              // true
```

| `Backend` | Is |
|---|---|
| `Interpret` | default; LLVM is never loaded |
| `Compile` | compile now |
| `Adapt` | compile once batch traffic pays for it; `warming()` reports progress |

| `jit::Options` field | Default | Is |
|---|---|---|
| `backend` | `Interpret` | as above |
| `points` | `1` | the batch size one call will carry |
| `codegen.lanes` | `Lanes::derived()` | points per iteration |
| `codegen.opt_level`, `codegen.codegen_level` | build type, `O1` | LLVM levels; `codegen_level` moves compile time |
| `codegen.slp_vectorize`, `codegen.loop_vectorize` | `false` | extra vectorisation |
| `codegen.veclib` | `None` | `Libmvec` for vector transcendentals |
| `codegen.contract` | `DDX_FP_FLAGS` | fuse multiply-add |
| `warm_points`, `hot_points` | `65536`, `1048576` | `Adapt` thresholds |
| `cache_dir` | empty | reuse compiled objects across runs |
| `time_passes` | `false` | per-pass timing to stderr |

`kernel_level()` says which compiled tier is answering.

## Python

```sh
pip install ddx-ad
```

```python
import ddx

@ddx.equation
def f():
    x = ddx.var("x")
    y = ddx.var("y")
    return ddx.exp(x) * ddx.sin(y)

value, gradient = f.jacobian([2.0, 3.0])
```

Wheels for Linux x86_64 (JIT) and Windows x64 (interpreted), CPython 3.11–3.14.
See [python/README.md](python/README.md).

## Compile-time expressions

Header-only; symbols are template arguments, results are `std::array`s or
tensors, and all but `hessian` are `constexpr`:

```cpp
constexpr auto x = var<"x">;
constexpr auto y = var<"y">;

constexpr auto eq = Equation{x * y + 2.0 * x};
static_assert(eq.jacobian(2.0, 3.0)[0] == 5.0);

Equation{x * y}.hessian(2.0, 3.0);                // links ddx::rt
Equation{x * y}.derivative_tensor<3>(1.0, 2.0);   // rank 3
Equation{sin(x)}.univariate_derivative<4>(0.5);

std::format("{}", x * y + sin(x));                // "x * y + sin(x)"
std::format("{}", Equation{x * y});               // "f0: x * y\n  jac: y_c, x_c\n"
```

## License

[Boost Software License 1.0](LICENSE.txt). Bundled third-party licences are in
[THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt).
