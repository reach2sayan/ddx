[![GCC 14](https://github.com/reach2sayan/ddx/actions/workflows/gcc.yml/badge.svg?branch=main)](https://github.com/reach2sayan/ddx/actions/workflows/gcc.yml)
[![Clang 20](https://github.com/reach2sayan/ddx/actions/workflows/clang.yml/badge.svg?branch=main)](https://github.com/reach2sayan/ddx/actions/workflows/clang.yml)
[![MSVC 2022](https://github.com/reach2sayan/ddx/actions/workflows/msvc.yml/badge.svg?branch=main)](https://github.com/reach2sayan/ddx/actions/workflows/msvc.yml)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white)](#requirements)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](#python)
[![License](https://img.shields.io/badge/license-BSL--1.0-4B8BBE)](LICENSE.txt)

# ddx

A C++23 library for differentiating expressions. Build a function over named
symbols while the program runs — terms looped over a data file, a model read
from configuration, an expression typed by a user, a loss that switches shape
on a comparison — and ask it for values, gradients, Jacobians and Hessians. One
point at a time, or a batch of thousands in a single call, interpreted,
compiled to machine code through LLVM, or [run on a GPU](#running-on-a-gpu)
through OpenCL. There
are [Python bindings](#python) over the same runtime, and a header-only
[compile-time API](#compile-time-expressions) for expressions whose shape is
known when you compile.

```cpp
#include "ddx.hpp"
#include "rt/equation.hpp"
using namespace ddx;

struct Sample { double t, y; };
const std::vector<Sample> data{{0.0, 2.00}, {1.0, 1.21}, {2.0, 0.74}, {3.0, 0.45}};

// Huber loss of the fit m(t) = a·exp(−b·t): quadratic while a residual is
// inside δ, linear once it is past.  `select` is a value, not a branch, so the
// derivative follows whichever arm each residual is on.
const double delta = 0.05;
const auto eq = rt::equation([&] {
  const auto a = rt::var("a");
  const auto b = rt::var("b");

  rt::RTExpression<double> loss = 0.0;
  for (const auto &s : data) {
    const auto r = s.y - a * exp(-b * s.t);
    loss += select(abs(r) < delta, r * r, delta * (2.0 * abs(r) - delta));
  }
  return loss;
});

*eq.evaluate(2.0, 0.5);     // 4.1343959887e-05 — every residual inside δ: the RSS
*eq.jacobian(2.0, 0.5);     // {-0.0010757424682, 0.0150678475595}
*eq.hessian(2.0, 0.5);      // 2 × 2, row-major
*eq.evaluate(1.8, 0.45);    // 2.1570396028e-02 — two residuals past δ, on the linear arm
```

**Contents** — [Requirements](#requirements) · [Using it](#using-it) ·
[Building](#building) · [Equations](#equations) · [Expressions](#expressions) ·
[Points](#points) · [Values and derivatives](#values-and-derivatives) ·
[Batches](#batches) · [Errors](#errors) ·
[Ownership and threads](#ownership-and-threads) · [Compiling](#compiling) ·
[Running on a GPU](#running-on-a-gpu) ·
[Saving and loading](#saving-and-loading) · [Reference](#reference) ·
[Python](#python) · [Compile-time expressions](#compile-time-expressions) ·
[Printing](#printing)

---

## Requirements

A C++23 compiler and standard library:

| Compiler | Needs |
|---|---|
| **GCC** | 14 or newer |
| **Clang** | 20 or newer, over libstdc++ 14+ — libc++ is not supported |
| **MSVC** | Visual Studio 2022, `/std:c++latest` |

Clang compiles the accessor spelling ddx uses from 18 but does not advertise it
until 20, and libc++ has no `views::enumerate` — hence the two floors.

- **CMake 3.26+**.
- **Boost** is downloaded and unpacked at configure time; nothing needs to be
  installed, and no compiled Boost library is ever linked. Point
  `DDX_BOOST_INCLUDEDIR` at your own headers to use those instead.
- GoogleTest is fetched at configure time, so a first configure that builds
  the tests wants a network.
- `-DDDX_BUILD_JIT=ON` additionally needs an LLVM 20 installation, pointed at
  with `LLVM_DIR` — to build. The library it produces carries LLVM, and loads
  on a machine that has none.
- The OpenCL device backend (`DDX_BUILD_OPENCL`) fetches the Khronos headers
  and ICD loader at configure time and needs a C compiler for the loader;
  nothing is installed to build it. The library loads on a machine with no
  OpenCL at all; running on a device needs that device's driver.
- `-DDDX_BUILD_PYTHON=ON` additionally needs **Python 3.11+** and pybind11; the
  module imports NumPy 1.23+ and pydantic 2.7+.

## Using it

In a project that vendors ddx:

```cmake
add_subdirectory(ddx)
target_link_libraries(my_app PRIVATE ddx::rt)
```

Or from a package manager — both channels carry the interpreted library,
without the JIT. The vcpkg port builds the OpenCL device backend as `AUTO`
does, where the building machine has an OpenCL runtime; the NuGet package has
none:

- **vcpkg**: an overlay port lives in this repository —
  `vcpkg install ddx --overlay-ports=<ddx checkout>/contrib/vcpkg/ports` —
  then the `find_package` below. The port pins the latest release.
- **NuGet**, for MSVC projects: the [`ddx`](https://www.nuget.org/packages/ddx)
  package carries the headers and x64 Release and Debug binaries; referencing
  it wires up the include path, the import library and the DLL copy. The
  binaries require AVX2.

Or against an installed or built copy:

```cmake
find_package(ddx CONFIG REQUIRED COMPONENTS rt jit)
target_link_libraries(my_app PRIVATE ddx::jit)
```

```sh
cmake --install build/release_jit --prefix /opt/ddx
```

A build tree is consumable without installing:

```cmake
find_package(ddx CONFIG REQUIRED PATHS /path/to/ddx/build/release NO_DEFAULT_PATH)
```

Asking for a component the build does not have fails at configure time and says
which option turns it on.

### Targets

| Target | Is |
|---|---|
| `ddx::ddx` | the header-only library — expressions, `Equation`, forward mode |
| `ddx::rt` | that plus runtime expressions |
| `ddx::jit` | `ddx::rt` with the LLVM backend compiled in |
| `ddx::cl` | `ddx::rt` with the OpenCL device backend compiled in |

Components for `find_package`: `ddx`, `util`, `ops`, `md`, `symbolic`, `dual`,
`rt`, `jit`, `cl`.

## Building

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Or through the presets:

| Preset | Build type | JIT | OpenCL |
|---|---|---|---|
| `debug` | Debug | — | auto |
| `release` | Release | — | auto |
| `debug_jit` | Debug | yes | auto |
| `release_jit` | Release | yes | auto |
| `debug_jit_tsan` | Debug | yes — under ThreadSanitizer | auto |
| `python` | Release | yes — and the extension module | auto |
| `python_no_jit` | Release | — the extension module alone | auto |
| `release_cl` | Release | — | yes |
| `debug_jit_cl` | Debug | yes | yes |

"auto" builds the OpenCL backend where the configuring machine has an OpenCL
runtime installed. It looks only on Linux, for an ICD in `/etc/OpenCL/vendors`
or `$OCL_ICD_VENDORS`; on Windows and macOS it is off.

```sh
cmake --preset release_jit
cmake --build --preset release_jit
ctest --preset release_jit
```

The version is stated once, in `CMakeLists.txt`'s `project()` line. CMake
writes `util/version.hpp` from it — `DDX_VERSION_MAJOR`/`MINOR`/`PATCH`,
`DDX_VERSION` (`100200` for 1.2.0), `DDX_VERSION_STRING`, and the same as
`ddx::version_major`, `ddx::version_number`, `ddx::version` — the Python module
carries it as `ddx.__version__`, and a release tag `vX.Y.Z` must name it.
Every Monday that `main` has moved since the last tag, CI bumps the patch
number, tags, and publishes the wheels; minor and major bumps are made by hand.

The JIT presets point `LLVM_DIR` at the Debian/Ubuntu `llvm-20` layout;
override it on the command line, which wins over the preset:

```sh
cmake --preset release_jit -DLLVM_DIR=/opt/llvm-20/lib/cmake/llvm
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `DDX_BUILD_JIT` | `OFF` | compile the LLVM backend into the library |
| `DDX_BUILD_OPENCL` | `AUTO` | compile the OpenCL device backend into the library — `AUTO` does where this machine has an OpenCL runtime (Linux only), `ON` or `OFF` decides |
| `DDX_BUILD_PYTHON` | `OFF` | build the pybind11 extension module |
| `DDX_BUILD_TESTS` | `ON` | build the GoogleTest tests |
| `DDX_SANITIZE` | `off` | `thread`, `address` or `undefined` — instrument the build |
| `DDX_INSTALL` | on if top-level | generate the install and `find_package` rules |
| `ENABLE_NATIVE_ARCH` | `ON` | `-march=native`, else `x86-64-v3`; x86 only, arm64 takes the compiler's default |
| `DDX_FP_FLAGS` | `ON` | `-ffp-contract=fast -fno-math-errno` |
| `DDX_BOOST_INCLUDEDIR` | *(empty)* | use Boost headers from here instead of fetching |

`-ffast-math` is not used and not recommended: it changes derivative values.

ddx's own targets compile `-fno-exceptions`. Nothing in the library throws —
failures come back as values — so what you compile your own code with is your
business.

---

## Equations

```cpp
#include "ddx.hpp"
#include "rt/equation.hpp"
using namespace ddx;
```

Link `ddx::rt`, or `ddx::jit` for the compiled batch calls.

`rt::equation` takes a callback, runs it, and hands back the equation. Symbols
are named inside the callback with `rt::var(name)` — a string, not a template
argument — and the callback returns the expression it built:

```cpp
const auto eq = rt::equation([] {
  const auto x = rt::var("x");
  const auto y = rt::var("y");
  return exp(x) * sin(y);
});
```

Return a `std::array` of expressions for a system, f: ℝⁿ → ℝᵐ:

```cpp
const auto sys = rt::equation([] {
  const auto x = rt::var("x");
  const auto y = rt::var("y");
  return std::array{x * y + sin(x), exp(x) - y * y};
});

*sys.jacobian(1.3, 0.7);      // row-major, m × n
```

Another scalar type is a template argument on both: `rt::var<T>(name)` inside
`rt::equation<T>(...)`.

`rt::var` takes the name as a `std::string_view`, so it need not be a literal —
the symbols can be whatever the program has just read:

```cpp
std::vector<std::string> names = read_column_headers(file);

const auto eq = rt::equation([&] {
  rt::RTExpression<double> acc = 0.0;
  for (const std::string &n : names) {
    acc += rt::var(n) * rt::var(n);
  }
  return acc;
});

*eq.symbols();                // whatever the file named, alphabetically
```

Naming the same string twice inside one callback gives the same symbol, so a
name that recurs across the data is one variable and not several.

Symbols are ordered canonically — **alphabetically by name** — which is the
order positional points are read in and the order every result is laid out in.
`symbols()` lists them in that order, `arity()` counts them:

```cpp
*eq.arity();                  // 2
*eq.symbols();                // {"x", "y"}
```

### Written down instead

An equation can be a string rather than a callback. Include
`rt/text/equation.hpp` for it. One string per function, so a system is a string
per output:

```cpp
#include "rt/text/equation.hpp"

const auto eq  = *rt::equation("sin(x)*y + 3");
const auto sys = *rt::equation("x*x + y*y - 4", "x*y - 1");
```

These answer `result<Equation>` rather than an `Equation`, since a string can
fail to parse where a callback cannot — the `*` above is that check skipped.

Everything after that is the same — the same points, the same derivatives, the
same batch calls. Free identifiers become the symbols, still ordered
alphabetically, so `eq.symbols()` here is `{"x", "y"}`, and a
[varmap](#points-by-name-at-run-time) is the natural point for one.

The grammar is Python's arithmetic: `+ - * /`, `**` for exponentiation (`^` is
not an operator), parentheses, unary signs, decimals and exponent notation. The
callable functions are the ones in the [expression table](#expressions), spelled
the same. `-x**2` is `-(x**2)` and `2**-1` is legal, as in Python.

Comparisons are infix, as in C++ — `<` `<=` `>` `>=` `==` `!=` — and bind
looser than arithmetic, so `select(x*x < y + 1, x, y)` needs no parentheses.
**One comparison per expression**: `a < b < c` is refused rather than read, since Python chains it
into a conjunction and C folds it into `(a < b) < c`, and picking either
silently would be picking a language.

A string that does not parse comes back as an error — `bad_syntax`,
`unknown_function` or `wrong_argument_count` — rather than as an equation:

```cpp
const auto bad = rt::equation("x ^ 2");     // `^` is not exponentiation
if (!bad) {
  std::println("{}", bad.error());          // errc::bad_syntax
}
```

A string that parses but names no symbol is a different thing: it is an
`Equation`, and that equation is [poisoned](#errors) with `no_graph`.

## Expressions

`rt::RTExpression<T>` is the expression type. A bare number converts to one, so
an accumulator starts at `0.0` and a scalar mixes into arithmetic without
wrapping:

```cpp
rt::RTExpression<double> acc = 0.0;
for (const auto &s : data) {
  acc += (s.y - a * exp(-b * s.t)) * 2.0;
}
```

| Kind | Available |
|---|---|
| Arithmetic | `+` `-` `*` `/` unary `-` `+=` `-=` `*=` `/=` |
| Unary | `sin` `cos` `tan` `exp` `log` `log10` `sqrt` `cbrt` `abs` `sign` `asin` `acos` `atan` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` `erf` |
| Binary | `pow` `atan2` `hypot` `max` `min` |
| Comparison | `<` `<=` `>` `>=` `==` `!=` — each answers `1.0` or `0.0` |
| Conditional | `select(cond, if_true, if_false)` |

They are found by argument-dependent lookup, so they work whether or not `rt` is
in scope. The compound forms are members, and rebind the handle rather than
mutating what other expressions already refer to.

Where a rule has a choice to make, it is made the same way everywhere: `abs`
differentiates to `0` at zero and `sign` to `0` throughout, `max` and `min` give
each side half at a tie, and `pow(a, b)` at `a == 0` answers the zeros its
constant arms say — `a⁰` is `1` and `0ᵇ` is `0`, so neither partial is the `0·∞`
the textbook rule spells.

### Choosing between two expressions

`select` is the only conditional, and it is not a branch: **both arms are
evaluated** and the condition picks one, so a batch keeps every point on the
same instruction path and the kernel emits a blend. The loss at the top of this
page is the usual shape: a residual tested against a threshold, with a different
expression on each side.

```cpp
const auto capped = rt::equation([&] {
  const auto x = rt::var("x");
  return select(x < 1.0, x * x, 2.0 * x - 1.0);   // C¹ at x = 1
});
```

The derivative is the taken arm's — `d select(c, t, f) = select(c, dt, df)` —
and **the condition is never differentiated**, so the symbols it tests get no
partial through that node. At the switch the function is whatever the two arms
make it: `select` does not smooth anything, and a discontinuous pair gives a
discontinuous derivative.

A condition is any nonzero value, as in C, and comparisons are ordinary
expressions rather than a separate boolean type — which is why they answer
`1.0` and `0.0` and why `select(x, …)` is legal. Two consequences follow from
IEEE and are worth stating: a comparison against `NaN` is false (so
`NaN == NaN` is `0.0`), while a `NaN` used directly *as* a condition is
true and takes the first arm, exactly as `if (nan)` does in C.

A comparison answers an expression and never a `bool`, so `if (x < 1.0)` does
not compile — but `a < b < c` does, folded to `(a < b) < c` as C reads it. A
range test is `(a < b) * (b < c)`.

Only `<` and `<=` are operations; the other four are those two read the other
way round, so the graph carries two comparison opcodes and not six.

## Points

Every call that needs numbers takes them in five interchangeable spellings:

```cpp
*eq.jacobian(1.3, 0.7);                             // positional
*eq.jacobian(std::array{1.3, 0.7});                 // any range
*eq.jacobian(named<"y">(0.7), named<"x">(1.3));     // by name, any order
*eq.jacobian("y"_s = 0.7, "x"_s = 1.3);             // by name, assignment spelling
*eq.jacobian(std::map<std::string, double>{{"y", 0.7}, {"x", 1.3}});   // a varmap
```

**Positional order is alphabetical by symbol name**, not the order the symbols
appear in the expression.

### Points by name at run time

The `named<"y">` and `"y"_s` spellings key off a name you write down, which a
program that read its symbols out of a file does not have. The **varmap** is the
spelling for that case: any range of `(name, value)`, the name a run-time string,
matching `rt::var(name)` on the way in.

```cpp
std::map<std::string, double> at;
for (const auto &[name, column] : table) {
  at[name] = column.back();
}

*eq.evaluate(at);
*eq.jacobian(at);
```

A `std::map`, an `unordered_map`, a `vector` of pairs — anything whose elements
have a `.first` convertible to `std::string_view` and a numeric `.second`. Order
does not matter; each value lands in the slot its name names.

**A named point must be whole.** Every spelling of one — varmap, `named<>` and
`"x"_s` alike — refuses a point that leaves a symbol unreached, with
`errc::short_point`. Naming a point is not the same as giving one, and a symbol
nothing named would otherwise be read as zero, which is a value rather than a
refusal. A name matching no symbol is `errc::unknown_symbol`.

`point(args…)` builds the point vector on its own, in canonical order, so a
point assembled once can be reused:

```cpp
const auto at = *eq.point(named<"y">(0.7), named<"x">(1.3));
*eq.evaluate(at);
```

## Values and derivatives

Every per-point call answers `result<T>` — `std::expected<T, ddx::error>`. The
`*` in these examples is that check skipped.

| Call | Answers |
|---|---|
| `evaluate(point)` | `result<T>`, or `result<std::vector<T>>` for a system |
| `jacobian(point)` | `result<std::vector<T>>`, row-major m × n — so `n` long when m == 1, which is the gradient |
| `gradient(point)` | the same block from a graph with no value in it: what only the value needs is not computed |
| `hessian(point)` | `result<std::vector<T>>`, dense row-major m × n × n. Each row is its own sweep, so `H[i*n+j]` and `H[j*n+i]` can differ in the last ULP — symmetrise before a solver that checks |
| `univariate_derivative<K>(x0)` | `result<T>` — the K-th derivative, one symbol and one output only |

```cpp
const auto g = *eq.jacobian(2.0, 0.5);     // {∂f/∂a, ∂f/∂b}
const auto H = *eq.hessian(2.0, 0.5);      // H[i * n + j]
H[0];                                      //  3.1060035856
H[1];                                      // -3.1441109272
```

### What each call computes

For $f : \mathbb{R}^n \to \mathbb{R}^m$ at a point $x$, with $n$ = `arity()`
symbols in `symbols()` order and $m$ = `output_dim` functions:

| Call | Computes | Shape |
|---|---|---|
| `evaluate(x)` | $f(x)$ | $m$ |
| `jacobian(x)` | $J(x)$, where $J_{ij} = \dfrac{\partial f_i}{\partial x_j}$ | $m \times n$ |
| `gradient(x)` | $J(x)$ alone, from a graph that does not compute $f$ | $m \times n$ |
| `hessian(x)` | $H(x)$, where $H_{ij} = \dfrac{\partial^2 f}{\partial x_i \, \partial x_j}$ | $n \times n$ per function |
| `univariate_derivative<K>(x₀)` | $\dfrac{d^K f}{dx^K}$ at $x_0$ | scalar |
| `jvp(v, x)` | $J(x)\,v$, so $(Jv)_i = \sum_j \dfrac{\partial f_i}{\partial x_j} v_j$ — the directional derivative of $f$ along $v$ | $m$ |
| `vjp(w, x)` | $w^{\top} J(x)$, so $(w^{\top}J)_j = \sum_i w_i \dfrac{\partial f_i}{\partial x_j}$ — the gradient of $w \cdot f$ | $n$ |
| `hvp(v, x)` | $H(x)\,v$, so $(Hv)_i = \sum_j \dfrac{\partial^2 f}{\partial x_i \, \partial x_j} v_j$ — the directional derivative of $\nabla f$ along $v$ | $n$ |

The last three are the matrix-free products. Each is one number per output, and
none of them ever forms the matrix it is named after.

```cpp
const std::vector<double> v{1.0, 0.0};

*eq.jvp(v, 2.0, 0.5);      // J·v   — m long: how f moves if x moves along v
*eq.vjp(v, 2.0, 0.5);      // wᵀJ   — n long: ∇(w·f), one weight per function
*eq.hvp(v, 2.0, 0.5);      // H·v   — n long: how ∇f moves if x moves along v
```

`hvp` also takes a symbol name, for the common case where $v$ is a basis vector:
`eq.hvp("a", x)` is $H(x)\,e_a$, the Hessian column for `a`. It refuses with
`errc::unknown_symbol` if no such symbol exists.

```cpp
*eq.hvp("a", 2.0, 0.5);    // {∂²f/∂a², ∂²f/∂b∂a}
```

**Why they are worth asking for, and when they are not.** Each answers in one
number per output and needs no matrix: `hvp` is $n$ columns whatever the
coupling, where `hessian()` is $\text{colours} \times n$ and reaches $n^2$ when
every symbol touches every other; `vjp` is $n$ where `jacobian()` is one column
per structural nonzero; `jvp` is $m$.

What that buys is **storage and bandwidth, not time**. Measured, one `hvp`
costs about the same as one whole `hessian` — within 10% across $n = 8 \ldots 128$ on both dense and banded
coupling — because the two lanes share one interned graph and the model's own
cone dominates both. So:

- **Want the whole matrix?** Use `hessian()`. Asking for $n$ products to
  assemble it is far slower than one colouring — at $n = 64$, roughly 70×.
- **Want one product, or cannot afford $n^2$ storage?** Use `hvp`. At $n = 64$
  the Hessian block is 4096 columns per point against 64.

A direction is one value per symbol ($v \in \mathbb{R}^n$), a covector one per
function ($w \in \mathbb{R}^m$); either of the wrong length answers
`errc::wrong_direction` and computes nothing.

```cpp
const auto in_b = rt::equation([] { return sin(rt::var("b")); });
*in_b.univariate_derivative<4>(0.5);       //  sin⁗(0.5)
```

`univariate_derivative` sweeps the graph once in truncated-Taylor arithmetic,
and every operation above is defined there — `max`, `min`, `abs`, `select` and
the comparisons included, taking the side the point is on. Should a scalar type
ever lack one of them, the call answers `errc::unsupported_scalar` rather than a
zero.

Values, gradient and Hessian are prepared separately, each the first time it is
asked for. A caller who only ever evaluates never pays for a gradient.

### Remembering the last call

Hand the factory a cache and an equation keeps its last call per kind. Ask again
at the same point and the answer comes back off it; move one symbol and only the
part of the graph that symbol reaches is computed again. Nothing else changes:
the numbers are the ones the sweep gives, bit for bit, and there is no call to
make -- a point already answered is simply answered.

```cpp
const auto eq = ddx::rt::equation(model);                      // off
const auto eq = ddx::rt::equation(model, ddx::rt::LastValue{}); // on
```

It is off by default, and it earns its place where points repeat or arrive one
coordinate at a time -- a line search, a finite difference, a minimiser asking
for the value and then the gradient at the same `x`:

```cpp
const auto f = *eq.evaluate(x);   // swept
const auto g = *eq.gradient(x);   // its own lane, so its own first call
const auto h = *eq.evaluate(x);   // nothing swept at all
```

Every spelling that takes a point is covered; a batch is not, being amortised
already. A lane a kernel answers keeps the point but not the tape, so it serves
a repeat and computes a moved point whole -- compiled code has no way to run
part of itself.

`LastValue` holds one slot per kind behind a shared lock: any number of threads
can be served the same point at once, and one that arrives while another is
writing sweeps for itself rather than waiting. The object is copied in, so it
may be a temporary.

To supply your own, model `ddx::rt::CValueCache`: answer `active()`, and hand out
a read lease and a write lease per `Want`, each lending the point, the output
values and the tape of one remembered call. `active()` is asked before anything
else and is how a cache says no call of yours can hit -- a `false` there costs
the equation nothing at all, not even reading the point. `ddx::rt::Extent` says
what a slot has to hold and which frozen graph it holds it for; two calls belong
to the same entry only where their whole `Extent` matches, so a re-freeze or a
grown arena parts them without the cache having to know why.

## Batches

The batch calls take columns: one input column per symbol, one output pointer
per output column, every column `n` points long. They fill the outputs in place
and answer `result<void>`.

```cpp
result<void> evaluate(xs, values, n);
result<void> jacobian(xs, values, partials, n);
result<void> hessian (xs, values, partials, hessians, n);

// The seeded products take a second block of input columns: `vs` is the
// direction at each point, one column per symbol, and `ws` one per function.
result<void> jvp(xs, vs, values, products, n);   // J v,  m columns
result<void> vjp(xs, ws, values, products, n);   // wᵀJ,  n columns
result<void> hvp(xs, vs, values, partials, products, n);  // H v, n columns
```

`hvp` fills `partials` with $\nabla f$ as well: the gradient falls out of the
same sweep the product is built on, so asking for $Hv$ never costs it twice.

Ask the equation how many output columns each block wants, and size the buffers
by that:

```cpp
const std::vector<double> as{1.8, 1.9, 2.0, 2.1};        // candidate a values
const std::vector<double> bs{0.45, 0.48, 0.50, 0.52};    // candidate b values
const std::size_t n = as.size();

std::vector<double> f(n * *eq.value_columns());
std::vector<double> j(n * *eq.jacobian_columns());

const std::vector<const double *> xs{as.data(), bs.data()};
const std::vector<double *> values{f.data()};
const std::vector<double *> partials{j.data(), j.data() + n};

if (const auto ok = eq.jacobian(xs, values, partials, n); !ok) {
  std::println("{}", ok.error());
}

f[2];                    // 4.1343959887e-05  — the loss at (2.0, 0.50)
j[2];                    // ∂loss/∂a there
j[n + 2];                // ∂loss/∂b there
f[0];                    // 2.1570396028e-02  — (1.8, 0.45): two terms on the linear arm
j[0];                    // -0.1617863027     — and the gradient is theirs, ±2δ·∂r/∂a
```

Every point of a batch runs the same instructions: the `select` in the model
computes both arms and keeps one per point, so a batch that straddles δ costs no
more than one that does not, and compiles to a single blend.

A column count that does not match answers `errc::wrong_column_count`, and
nothing is written.

### The two compressed blocks

Both derivative blocks are **sparse**, so `jacobian_columns()` and
`hessian_columns()` are what size them — asking costs nothing — and a pattern
says which cell each column is. A caller who wants the dense matrix uses the
per-point calls above, which undo both compressions on the way out.

The Jacobian keeps only the cells that structurally exist. `∂fᵢ/∂xⱼ` gets a
column when the derivative is something other than the literal zero — a symbol
a function does not mention has no column, and neither has one whose partial
folded away. `jacobian_pattern()` places them:

```cpp
const rt::Sparsity &pattern = eq.jacobian_pattern()->get();
pattern.nonzeros();                    // == *eq.jacobian_columns()

for (const rt::Cell &c : pattern.entries()) {   // every cell that exists
  use(c.row, c.column, j[c.slot * n + point]);  // ∂f(row)/∂x(column)
}

const double dfi_dxj =                 // at() is nullopt off the pattern
    pattern.at(i, j)
        .transform([&](std::size_t cell) { return j[cell * n + point]; })
        .value_or(0.0);
```

For a system whose functions each touch a few symbols that is most of the
matrix: a tridiagonal residual over six variables has 16 columns rather than 36,
and the graph, the kernel and the compile shrink with it.

The Hessian takes a fourth block, compressed by colour rather than by pattern —
one output only. `hessian_cell(i, j)` places those:

```cpp
std::vector<double> h(n * *eq.hessian_columns());
const auto hessians =
    std::views::iota(0uz, *eq.hessian_columns()) |
    std::views::transform([&](std::size_t k) { return h.data() + k * n; }) |
    std::ranges::to<std::vector>();

const auto ok = eq.hessian(xs, values, partials, hessians, n);

const auto cell = eq.hessian_cell(i, j);         // std::optional
const double d2 = cell ? h[*cell * n + point] : 0.0;
```

`evaluate` is not the Jacobian call with the partials thrown away: columns
nobody asks for are work nobody does.

## Errors

`rt::equation` always hands back an `Equation`. When the callback or the string
could not produce one, that equation is *poisoned*: it carries the error rather
than a function, and every call on it answers with that error.

```cpp
const auto eq = rt::equation([] { return rt::var("x") * 2.0; });

if (const auto bad = eq.status()) {
  std::println("{}", *bad);      // errc::no_arena
  return;
}
```

`status()` is a `std::optional<ddx::error>`; `poisoned()` is the same question
without the reason.

Calls answer `result<T>` on top of that: the symbol list exists only at run
time, so an arity mismatch or an unknown name is a genuine runtime failure.

```cpp
const auto j = eq.jacobian(1.0, 2.0);      // two values, one symbol
if (!j) {
  std::println("{}", j.error());  // errc::wrong_arity
}
```

| `errc` | Means |
|---|---|
| `wrong_arity` | the point does not supply one value per symbol |
| `wrong_direction` | a direction does not supply one value per symbol, or a covector one per function |
| `short_point` | a range point is shorter than the symbol list, or a named point leaves a symbol unreached |
| `unknown_symbol` | a named point uses a name the equation does not have |
| `index_out_of_range` | the index does not name a symbol of this equation |
| `wrong_column_count` | a batch block has the wrong number of columns |
| `no_arena` | a symbol was named outside an `rt::equation` callback |
| `no_graph` | the expression is a bare literal, naming no function |
| `sealed_arena` | the symbols already back an equation, so they are final |
| `not_univariate` | `univariate_derivative` on more than one symbol |
| `unsupported_scalar` | `univariate_derivative` over an operation its Taylor arithmetic does not define |
| `bad_syntax` | a text equation the grammar does not accept |
| `unknown_function` | a text equation calls a function that does not exist |
| `wrong_argument_count` | a text equation calls one with the wrong arity |
| `archive_io` | the file could not be read or written |
| `bad_archive` | not a ddx file, or a format this build does not read |
| `archive_corrupt` | the file's checksum or structure does not hold |
| `archive_mismatch` | the file loads, but does not describe this equation |
| `jit_target`, `jit_module`, `jit_object`, `jit_verify`, `jit_lookup` | the compiler could not produce or link a kernel |

An `errc` and an `error` both format and stream as their text, so
`std::println("{}", j.error())` is the whole of reporting one. Nothing here
throws.

The accessors that answer no `result` answer `std::optional` instead —
`arity()`, `symbols()`, `value_columns()`, `jacobian_columns()`,
`hessian_columns()`, `hessian_colors()` and `jacobian_pattern()` are `nullopt`
on a poisoned equation.
A bare count could not say it: an equation over a literal-only expression
legitimately has no symbols and no output columns.

## Ownership and threads

**The equation owns everything it needs.** It is move-only: return it from a
function, store it in a class, keep it as long as you like.

```cpp
auto make_model(const std::vector<Sample> &data) {
  return rt::equation([&] {
    const auto a = rt::var("a");
    const auto b = rt::var("b");
    rt::RTExpression<double> rss = 0.0;
    for (const auto &s : data) {
      const auto r = s.y - a * exp(-b * s.t);
      rss += r * r;
    }
    return rss;
  });
}

const auto eq = make_model(load());     // `data` is gone; `eq` is complete
```

**Symbols belong to the callback.** `rt::var` is meaningful while the callback
that named it runs. Build the expression there and return it; do not store an
`RTExpression` and use it afterwards. A symbol named outside a callback gives a
poisoned equation carrying `errc::no_arena`.

**Batch columns are yours.** The equation writes through the pointers you give
it and keeps none of them; every buffer must be alive for the duration of the
call, and each column must hold at least `n` elements.

Building is independent per thread: two threads may run `rt::equation` callbacks
at the same time. An equation prepares itself on the first call of each kind —
make that first call before handing one to several threads, or give each thread
its own. A batch call splits across threads by slicing the columns, each thread
getting its own offset pointers and its own `n`:

```cpp
const std::size_t chunk = (n + threads - 1) / threads;
std::vector<std::jthread> pool;
for (std::size_t t = 0; t < n; t += chunk) {
  const std::size_t m = std::min(chunk, n - t);
  pool.emplace_back([&, t, m] {
    const std::vector<const double *> xs{as.data() + t, bs.data() + t};
    const std::vector<double *> values{f.data() + t};
    const std::vector<double *> partials{j.data() + t, j.data() + n + t};
    (void)eq.jacobian(xs, values, partials, m);
  });
}
```

## Compiling

Configure with `-DDDX_BUILD_JIT=ON` and link `ddx::jit`, and the batch calls can
run compiled code. The spellings do not change, the answers do not change, and
there is no compiler or kernel object for you to hold.

Nothing compiles unless you ask. An equation left alone interprets, which costs
no compiler and runs within ~1.4x of a kernel. Asking is what starts the build,
so it overlaps whatever you do next, and **no call ever waits for it**:

```cpp
auto eq = rt::equation([] {
  const auto x = rt::var("x");
  const auto y = rt::var("y");
  return x * log(x) + y * exp(x * y);
});

eq.options({.backend = rt::Backend::Compile, .points = 4096});
// Starts the compile here and now.  Returns immediately; nothing is ready yet.

eq.jacobian(xs, f, g, n);      // compile in flight -> interpreted
// ... the kernel lands about here ...
eq.jacobian(xs, f, g, n);      // compiled, and from here on
eq.uses_kernel();              // true
```

Values, gradient and Hessian compile separately, each launched the moment it is
first needed — so a caller who only evaluates never compiles a gradient.

Results either side of the switchover agree **to the bit**, with one exception
worth knowing. The compiled path sums a reduction spine of sixteen terms or more
in blocks, where the interpreted one sums it left to right — k dependent adds
are a latency no lane width hides. Addition is associative in exact arithmetic
and not in floating point, so a result reached through such a spine can move in
its last bits the moment the kernel lands: measured at 2 ULP over forty terms and
4 over eighty. Shorter spines are not rewritten and cannot move, and the Hessian
lane is built from one graph either way, so it never moves.

`wait_for_kernel()` is the only call that blocks, and only because it was asked
to:

```cpp
eq.wait_for_kernel();          // true once it lands
eq.jacobian(xs, f, g, n);      // compiled, guaranteed
```

Setting `Options` to the value it already holds does nothing. Changing it
discards the compiled code and abandons a compile still in flight rather than
waiting for it — including a change back to `Backend::Interpret`, after which no
compiler is asked at all.

### Backends

| `Backend` | Is |
|---|---|
| `Interpret` | the default. No compiler is asked, so a program that never says otherwise never loads LLVM |
| `Compile` | start compiling now |
| `Adapt` | start compiling once the batch traffic has paid for it — for a caller who cannot say up front which of their equations are hot |
| `Device` | build for an OpenCL device, a GPU or otherwise — see [Running on a GPU](#running-on-a-gpu) |

Either compiling backend answers first with a quick kernel and replaces it with
a better one; every level agrees to the bit, so the swap is invisible.
`kernel_level()` says which one is answering, and under `Adapt`, `warming()`
answers how far along the next step is:

```cpp
eq.options({.backend = rt::Backend::Adapt});

if (const auto w = eq.warming()) {
  std::println("{} of {} points", w->points, w->threshold);
}
*eq.kernel_level();            // 0 at first, then the level you asked for
```

### `jit::Options`

| Field | Default | Is |
|---|---|---|
| `backend` | `Interpret` | `Interpret`, `Compile`, `Adapt` or `Device` |
| `points` | `1` | the batch you intend to hand one call — stated, since the kernel is built before any call exists to infer it from |
| `codegen.lanes` | `Lanes::derived()` | points per loop iteration; derived is the host's register width, scalar for a batch too short to fill one. `Lanes::scalar()` or `*Lanes::exactly(w)` states one. Every width gives the same bits |
| `codegen.opt_level` | follows the build type | LLVM's IR pipeline, `Level::O0` to `Level::O3` — `O3` in a Release build, `O1` in a Debug one |
| `codegen.codegen_level` | `Level::O1` | LLVM's codegen, `Level::O0` to `Level::O3` — the knob that trades kernel speed for compile time |
| `codegen.slp` | `false` | pack independent subexpressions within one point; model-dependent, hence off |
| `codegen.loop_vectorize` | `false` | loop vectorisation, on a loop already emitted `lanes` wide |
| `codegen.veclib` | `None` | vector math library for transcendentals; `Libmvec` trades ~0.5 ULP for ~4, and a derived `lanes` is then the widest it serves (four) |
| `codegen.contract` | follows `DDX_FP_FLAGS` | fold a multiply feeding an add into one rounding |
| `warm_points` | `65536` | batch points that buy the first step, under `Adapt` |
| `hot_points` | `1048576` | further points that buy the top one, under `Adapt` |
| `retain_object` | `true` | keep the compiled object so [`save`](#saving-and-loading) can write it |
| `cache_dir` | *(empty)* | keep compiled objects here between runs; a second run links instead of compiling |
| `time_passes` | `false` | per-pass timing to stderr |
| `device` | *(empty)* | under `Device`, which OpenCL device — [Running on a GPU](#running-on-a-gpu). Not saved with the equation: which devices exist is the machine's |

`codegen` is everything the emitter reads, and so the identity a stored kernel
is matched against and the object cache is keyed on; the fields around it are
policy and change no machine code.

`points` decides the lane width and nothing else: a call carrying some other
number is answered correctly, just not by the kernel that number would have
built. At one point a scalar kernel is 1.2x to 4.1x quicker than the host's
vector width, and over a batch the reverse by 3x.

```cpp
eq.options({.backend = rt::Backend::Compile, .points = 1});     // a gradient per step
eq.options({.backend = rt::Backend::Compile, .points = 4096});  // a batch at a time
```

Naming a `cache_dir` is the cheapest thing you can do about compile time: a run
that finds its object there links it instead of compiling, which is roughly
three orders of magnitude quicker. Failing that, `codegen.codegen_level` — not
`codegen.opt_level` — is the knob that moves a compile time.

```cpp
eq.options({.backend = rt::Backend::Compile, .points = 4096, .cache_dir = "/var/cache/ddx"});
```

## Running on a GPU

Build with the OpenCL backend and link `ddx::cl`. `DDX_BUILD_OPENCL` is on by
default wherever the configuring machine has an OpenCL runtime. `Backend::Device`
then builds each lane's graph as an OpenCL kernel for a device with double
precision — an NVIDIA, AMD or Intel GPU, or a CPU runtime:

```cpp
eq.options({.backend = rt::Backend::Device});
eq.wait_for_kernel();          // true once the device's compiler has built it
eq.jacobian(xs, f, g, n);      // on the device
```

It behaves as the compiling backends do. Asking starts the build, calls are
swept until it lands, and each lane builds the first time it is needed.
`wait_for_kernel(want)` waits for one lane — the Jacobian's unless `want` names
another.

An empty `device` takes the first GPU with double precision, else the first
device of any kind that has it. Anything else is matched, ignoring case, against
the platform and device name:

```cpp
eq.options({.backend = rt::Backend::Device, .device = "NVIDIA"});
eq.options({.backend = rt::Backend::Device, .device = "gfx1035"});
eq.options({.backend = rt::Backend::Device, .device = "Intel"});   // its CPU runtime
```

`device_status()` names the device answering, or says why none is. It is empty
under any other backend:

```cpp
if (const auto status = eq.device_status()) {
  if (*status) {
    std::println("on {}", **status);   // "<platform> / <device> / <driver> / <version>"
  } else {
    std::println("swept: {}", status->error().detail);
  }
}
```

A selector nothing matches, a machine with no device and a kernel the driver
refuses all leave the equation answering from the sweep: `uses_kernel()` is
false and `device_status()` carries the reason — `errc::no_device` or
`errc::device_compile` with the driver's build log. A launch that fails is
answered by the sweep too, and so is every call in a library built without the
backend: `Backend::Device` is accepted there, and `device_status()` answers
`errc::no_device`.

Arithmetic agrees with the sweep **to the bit**: `+ - * /`, fused multiply-adds,
comparisons, `abs`, `sign`, `max`, `min` and `select`. The device computes the
graph the sweep walks, contraction included, and its compiler is given nothing
that lets it reorder or fuse. The transcendentals are the device's own math
library, which OpenCL bounds within a few ULP rather than rounding correctly, so
a model with `exp`, `log` or `sin` in it agrees closely and not exactly.
`kernel_level()` answers nothing for a device kernel.

Every call copies the point columns to the device and the output columns back,
so a short batch is quicker swept or compiled for the CPU; the device pays on
large batches, where its arithmetic outweighs the copy.
`scripts/compare.py --gpu` measures it against torch and JAX on the same GPU.

The device is usable without an equation as well: `cl::Device::create(selector)`
picks one, `compile(graph)` builds a `cl::Kernel` with the batch calls' column
layout, and `cl::source_of(graph)` is the OpenCL C a graph lowers to.

## Saving and loading

Preparing an equation for derivatives is work you can do once. `save` writes it
to a file, and `load` reads it back without a model:

```cpp
const auto eq = rt::equation([] { return exp(rt::var("x")) * rt::var("y"); });
eq.save("f.ddx");

const auto same = *rt::load("f.ddx");            // no model, nothing rebuilt
const auto sys  = *rt::load<double, 3>("s.ddx"); // three functions
```

The output count is a template parameter because it is part of the type, and a
file holding some other number is refused rather than adapted to.

Pair a model with a file and it becomes a cache — the lambda is the model as you
would write it anyway, and the path is only where the result is kept:

```cpp
const auto eq = rt::equation("f.ddx", [] {
  const auto x = rt::var("x");
  return exp(x) * x;
});
```

The first run builds and writes the file. A later one finds the file still
describes the model and takes the work off disk. Editing the model rebuilds and
overwrites rather than trusting a stale file, and it never refuses: an absent
file is the first run, a stale one is a rebuild. `loaded()` says which happened.

Two questions about a file, and they are different:

```cpp
rt::verify("f.ddx");     // result<void> — can this build read it at all?
eq.verify("f.ddx");      // result<void> — ...and is it *this* equation?
```

Files are binary and little-endian, checked before anything in them is used, so
one written by a different version of ddx is refused rather than misread.

Compiled code travels in the file too:

```cpp
eq.options({.backend = rt::Backend::Compile});
eq.wait_for_kernel();
eq.save("f.ddx");                 // the kernel goes with it

const auto warm = *rt::load("f.ddx");
warm.uses_kernel();               // true, with nothing having compiled
```

Measured at 64, 128 and 256 variables: 38/79/179 ms to build and compile against
0.40/0.63/1.12 ms to load. `retain_object` is on by default so the code is there
when a save comes; a caller that never saves turns it off, and the file then
carries the equation and no code.

Stored code runs only where the equation, the host and the options it was built
under all still agree. Anything else compiles instead — a mismatch is answered
by compiling, never by running the wrong code.

| Member | Answers |
|---|---|
| `save(path)` | `result<void>` — write this equation |
| `verify(path)` | `result<void>` — does `path` hold *this* equation? |
| `Equation::load(path)` | `result<Equation>` — read one |
| `loaded()` | whether this equation came off disk |
| `rt::verify<T>(path)` | `result<void>` — is `path` readable by this build? |

---

## Against other libraries

Five functions, a gradient with respect to every variable at one point, the
point changing every call. Every cell is the time for the gradient divided by
the time to evaluate the same function once at plain `double` — the overhead
factor, so lower is better and below `1.00x` means the derivatives came cheaper
than the function — as the median of 3 interleaved repetitions, with the
spread `(max − min) / median` beside it. The last column is that one evaluation.

| arm | what it is |
|---|---|
| `ddx-rt/r` | `Equation::gradient`, the graph interpreted |
| `ddx-jit/r` | the same graph compiled through LLVM, one point wide |
| `casadi/r` | CasADi's `SX` gradient `Function`, interpreted, through `Function.buffer()` |
| `casadi/j` | the same `Function` with `jit=True`, its generated C compiled by the system compiler |
| `adept/r` | Adept 2.1.3, `new_recording` and `compute_adjoint` per point |

**Regular solid solution**

| n | ddx-rt/r | ddx-jit/r | casadi/r | casadi/j | adept/r | one evaluation, ns |
|---|---|---|---|---|---|---|
| 16 | 3.61x ±5% | **1.03x** ±2% | 11.59x ±7% | 5.79x ±6% | 9.22x ±11% | 195 |
| 32 | 3.94x ±3% | **1.02x** ±1% | 10.70x ±11% | 2.48x ±6% | 10.19x ±3% | 650 |

**UNIQUAC**

| n | ddx-rt/r | ddx-jit/r | casadi/r | casadi/j | adept/r | one evaluation, ns |
|---|---|---|---|---|---|---|
| 16 | 4.57x ±2% | **0.92x** ±1% | 8.96x ±2% | 1.92x ±4% | 6.33x ±5% | 807 |
| 32 | 4.43x ±3% | **0.93x** ±1% | 7.85x ±3% | 1.55x ±1% | 6.90x ±1% | 2250 |

**Peng-Robinson** — n counts N−1 mole fractions plus Z.

| n | ddx-rt/r | ddx-jit/r | casadi/r | casadi/j | adept/r | one evaluation, ns |
|---|---|---|---|---|---|---|
| 17 | 3.10x ±5% | **0.48x** ±2% | 9.28x ±3% | 1.69x ±3% | 4.42x ±3% | 857 |
| 33 | 2.89x ±6% | **0.45x** ±2% | 7.84x ±17% | 1.03x ±3% | 4.18x ±1% | 3340 |

**Mixed Solvent Electrolyte**

| n | ddx-rt/r | ddx-jit/r | casadi/r | casadi/j | adept/r | one evaluation, ns |
|---|---|---|---|---|---|---|
| 16 | 5.16x ±3% | **0.95x** ±1% | 12.94x ±1% | 1.98x ±2% | 7.96x ±2% | 1164 |
| 32 | 5.20x ±3% | **1.03x** ±2% | 11.33x ±1% | 1.55x ±2% | 8.69x ±2% | 3589 |

**The op-coverage function**

| n | ddx-rt/r | ddx-jit/r | casadi/r | casadi/j | adept/r | one evaluation, ns |
|---|---|---|---|---|---|---|
| 16 | 2.62x ±3% | **0.71x** ±1% | 2.96x ±3% | 1.39x ±5% | 2.98x ±6% | 1616 |

`compare/` is the harness — the functions, the arms, the gate that checks every
arm against every other before anything is timed, and the synthetic families
behind `--trend` that the interpreter is tuned against.

## Reference

`ddx::rt::equation(callback)` → `Equation`. All of it is `const` and
thread-safe except `options()`.

| Member | Answers |
|---|---|
| `poisoned()`, `status()` | whether the build failed, and with what |
| `arity()` | `std::optional<std::size_t>` — symbol count, `n` |
| `symbols()` | `std::optional<std::span<const std::string>>` — canonical order |
| `point(args…)` | `result<std::vector<T>>` — a point in canonical order, from any of the five spellings |
| `evaluate(point)` | `result<T>`, or `result<std::vector<T>>` for a system |
| `jacobian(point)` | `result<std::vector<T>>`, dense row-major m × n |
| `gradient(point)` | the same, from a graph with no value block |
| `hessian(point)` | `result<std::vector<T>>`, dense row-major m × n × n |
| `univariate_derivative<K>(x0)` | `result<T>`, one symbol and one output only |
| `jvp(v, point)` | `result<std::vector<T>>`, $m$ long — $Jv$ |
| `vjp(w, point)` | `result<std::vector<T>>`, $n$ long — $w^{\top}J$ |
| `hvp(v, point)` | `result<std::vector<T>>`, $n$ long — $Hv$, one output only |
| `hvp(name, point)` | the same, along the basis vector for that symbol |
| `evaluate(xs, f, n)` | `result<void>` — a batch of `n` |
| `jacobian(xs, f, g, n)` | `result<void>` |
| `gradient(xs, g, n)` | `result<void>` |
| `hessian(xs, f, g, h, n)` | `result<void>`, one output only |
| `jvp(xs, vs, f, p, n)` | `result<void>` |
| `vjp(xs, ws, f, p, n)` | `result<void>` |
| `hvp(xs, vs, f, g, p, n)` | `result<void>`, one output only — `g` gets $\nabla f$ |
| `value_columns()`, `jacobian_columns()`, `hessian_columns()` | `std::optional<std::size_t>` — what sizes the batch buffers |
| `jvp_columns()`, `vjp_columns()`, `hvp_columns()` | the same, for the seeded products — `m`, `n`, `n` |
| `jacobian_pattern()` | `std::optional<std::reference_wrapper<const rt::Sparsity>>` — which (function, symbol) cell each Jacobian column is |
| `hessian_cell(i, j)` | `std::optional<std::size_t>` — which Hessian column holds H(i, j), or none |
| `hessian_colors()` | `std::optional<std::size_t>` — groups in the Hessian's compression |
| `options(opts)`, `options()` | set or read the compile options; setting returns `*this` |
| `uses_kernel()`, `kernel_level()` | whether a batch call runs compiled code, and at which level |
| `warming()` | under `Adapt`, points seen against the next threshold |
| `wait_for_kernel(want)` | block until the compile in flight for a lane has landed — the Jacobian's unless `want` names another |
| `device_status()` | under `Device`, the device answering or why none is; empty under any other backend |
| `save(path)`, `verify(path)` | write this equation; ask whether a file holds it |
| `load(path)`, `loaded()` | read one; whether this one was read |

---

## Python

The same runtime, as an extension module, published on PyPI as
[`ddx-ad`](https://pypi.org/project/ddx-ad/) and imported as `ddx` —
`pip install ddx` is an unrelated project. There is one wheel per platform and
CPython version (3.11–3.14), and a wheel needs nothing installed beside it —
LLVM is inside the library:

```sh
pip install ddx-ad
```

Each [release](https://github.com/reach2sayan/ddx/releases) carries the same
wheels and a source distribution.

| Wheel | JIT | OpenCL |
|---|---|---|
| Linux x86_64 (glibc 2.28+) | yes | no |
| Windows x64 | no — calls interpret | no |

There is no macOS wheel: the tree uses C++23 ranges that libc++ does not have
(`views::enumerate`, `cartesian_product`, `chunk`, `stride`, `ranges::fold`),
so on a Mac it builds from source with a libstdc++ toolchain and without the
JIT.

Building from source instead — `pip install .`, or a preset — needs a C++23
compiler and, for the JIT, LLVM 20's archives with a static zlib and zstd;
`scripts/build_llvm.py` builds that set from source into a prefix, and is what
the Linux wheel uses:

```sh
pip install .                       # wheel; JIT on, so building needs LLVM 20
cmake --preset python               # in-tree, JIT
cmake --preset python_no_jit        # in-tree, no LLVM
```

`ddx.has_jit` says whether the copy you have was built with the LLVM backend,
and `ddx.has_opencl` whether it was built with the OpenCL one.
Everything below works either way; without it, calls interpret.

`equation` takes a model — a callable of no arguments returning one expression,
or a tuple of them for a system — as a call or a decorator:

```python
import ddx

@ddx.equation
def f():
    x = ddx.var("x")
    y = ddx.var("y")
    return ddx.exp(x) * ddx.sin(y)
```

Or a string, or a list of strings for a system, in the same grammar the
[C++ side](#written-down-instead) reads:

```python
g   = ddx.equation("exp(x) * sin(y)")
sys = ddx.equation(["x*x + y*y - 4", "x*y - 1"])
```

Symbols are named inside the model with `ddx.var(name)`, and they order
alphabetically as they do in C++. `f.symbols` lists them, `f.arity` counts them,
`f.outputs` counts the model's outputs. Arithmetic operators work on
`Expression`, and a bare number mixes in; the free functions are the same set
the C++ side has.

### Points

A point is a sequence, or a dict keyed by symbol name. A 2-D array is a
**batch** — shape `(symbols, points)`, one row per symbol:

```python
f.jacobian([2.0, 3.0])                      # positional, alphabetical
f.jacobian({"x": 2.0, "y": 3.0})            # by name
f.jacobian(np.array([[2.0], [3.0]]))        # a batch of one
f.jacobian(np.array([[2.0, 2.5], [3.0, 3.5]]))   # a batch of two
```

### Values and derivatives

Each call returns everything up to and including what it names, so a gradient
never costs a second evaluation:

| Call | Returns |
|---|---|
| `evaluate(x)` | `f` |
| `jacobian(x)` | `(f, J)` |
| `gradient(x)` | `J` alone, from a graph that does not compute `f` |
| `hessian(x)` | `(f, J, H)` |
| `jvp(v, x)` | $(f,\; J v)$ — the directional derivative of $f$ along $v$ |
| `vjp(w, x)` | $(f,\; w^{\top} J)$ — the gradient of $w \cdot f$ |
| `hvp(v, x)` | $(f,\; \nabla f,\; H v)$ — the directional derivative of $\nabla f$ along $v$ |

```python
value, gradient = f.jacobian([2.0, 3.0])
value, gradient, hessian = f.hessian([2.0, 3.0])
hessian.shape                               # (2, 2)

value, gradient, hv = f.hvp([1.0, 0.0], [2.0, 3.0])
hv.shape                                    # (2,) — H·v, without forming H
```

Shapes follow the point: a single point gives a scalar `f`, an `(n,)` gradient
and an `(n, n)` Hessian; a batch of `p` appends that axis, giving `(p,)`,
`(n, p)` and `(n, n, p)`. A system prepends its output axis. Unlike the C++
batch calls, the Hessian arrives **dense** — the compression is undone on the
way out, so nothing here needs `hessian_columns`.

The three products never form the matrix they are named after, which is what
makes them worth having when $n^2$ storage is the problem — though not when
speed is, since one product costs about what the whole matrix does. $Jv$ and
$Hv$ take a direction over symbols, so it accepts the same spellings a point
does — a sequence, a dict, or a `(symbols, points)` batch alongside a batched
point. $w^{\top}J$ takes one weight per *function*, so it is positional only.
`hvp` needs a single-output model, as `hessian`'s compressed path does.

### Calling in a loop

The calls above allocate their answers. `buffer(x)` binds the point and the
answers once and hands back a `Call`: write the next point into `x`, call it,
read the blocks back. Same arrays every time, so nothing is allocated per call.

```python
call = f.buffer(np.array([2.0, 3.0]))
for _ in range(steps):
    call()                                  # fills call.value and call.jacobian
    call.x[:] = next_point(call.jacobian)
```

`want` chooses how far it goes, and a block nobody asked for is one nobody
computes:

| `want` | Fills |
|---|---|
| `Want.VALUE` | `value` |
| `Want.JACOBIAN` *(default)* | `value`, `jacobian` |
| `Want.GRADIENT` | `jacobian` |
| `Want.HESSIAN` | `value`, `jacobian`, `hessian` |

Reading a block the call did not ask for raises `errc.wrong_column_count`, and
so does binding `Want.HESSIAN` on a system — a frozen graph carries one
colouring, so m Hessians are read off the arena a point at a time.

Shapes are the ones the allocating calls answer with, `value` included: one
output at one point is a `float`, and everything else is an array. The point is
bound as an array whatever was passed, so `call.x` is writable even when the
argument was a list.

### Remembering the last call

`remember=True` gives the equation the same last-call cache the C++ side gets
from `LastValue`, on the same terms: a repeated point is answered off the last
one, a point one symbol away sweeps only what that symbol reaches, and the
numbers are unchanged.

```python
f = ddx.equation(model, remember=True)
value = f(x)              # swept
grad = f.gradient(x)      # its own lane
again = f(x)              # nothing swept
```

It applies to a point at a time and not to an array of them, and a lane a kernel
answers serves a repeat without amending a moved point.

### Errors

Python raises where C++ returns `result<T>`. `ddx.Error` is a `RuntimeError`
carrying the same code:

```python
try:
    f.jacobian({"z": 1.0})
except ddx.Error as e:
    print(e)                                # "no symbol of that name"
    e.code is ddx.errc.unknown_symbol       # True
```

`errc` is an `IntEnum`, so it compares and formats as its number — `e.code.name`
is the spelling. The codes are the ones in the [table above](#errors); the
exception's own message is the text.

### Compiling

`Options` is a frozen pydantic model of the same fields as `jit::Options`,
validated on the way in. `eq.options` reads and assigns it; `eq.compile()` sets
`backend=COMPILE`, waits for the kernel, and returns the equation, so a
configure-and-use reads in one line:

```python
f.compile(points=batch.shape[1]).jacobian(batch)
f.uses_kernel                               # True, once it has landed
f.options = ddx.Options(backend=ddx.Backend.INTERPRET)   # discards the kernel
```

`compile()` blocks by construction — it is `wait_for_kernel()` with the options
set first. Assigning `options` does not: calls interpret until the kernel lands
and switch over when it does, as in C++.

`f.compile(backend=ddx.Backend.DEVICE)` builds for an OpenCL device instead, and
`Options.device` picks one as in [Running on a GPU](#running-on-a-gpu).
`f.device_status` names the device, is `None` under any other backend, and
raises `ddx.Error` when no device answers.

No wheel carries the device backend, and neither does `pip install .` unless
asked with `-C cmake.define.DDX_BUILD_OPENCL=ON`; the `python` preset builds it
as `AUTO` does. Where it is missing, `ddx.has_opencl` is `False`, `DEVICE`
interprets, and `device_status` raises `errc.no_device`.

### Saving and loading

Equations save and load here too, over the same file format — a file written by
C++ loads in Python and the other way round:

```python
eq.save("f.ddx")
same = ddx.load("f.ddx")          # no model runs, nothing is rebuilt

@ddx.equation                      # or pair a model with a file, as a cache
def model() -> ddx.Expression:
    x, y = ddx.var("x"), ddx.var("y")
    return ddx.exp(x) * y

cached = ddx.equation(model, cache="f.ddx")
cached.loaded                      # False the first run, True after
```

A text equation caches the same way: `ddx.equation("exp(x) * y", cache="f.ddx")`.

`save`, `load` and `verify` raise `ddx.Error` rather than answering `False`:
unreadable, unloadable and "a different equation" are three different `errc`
values, and only the code says which.

### Reference

| Member | Is |
|---|---|
| `arity`, `outputs`, `symbols` | properties — symbol count, output count, canonical names |
| `evaluate(x)`, `__call__(x)` | `f` at the point or batch |
| `jacobian(x)` | `(f, J)` |
| `gradient(x)` | `J` alone |
| `jvp(v, x)`, `vjp(w, x)`, `hvp(v, x)` | $(f, Jv)$, $(f, w^{\top}J)$, $(f, \nabla f, Hv)$ |
| `hessian(x)` | `(f, J, H)`, dense |
| `options` | property — read or assign an `Options` |
| `compile(**fields)` | set `Options`, block for the kernel, return self |
| `uses_kernel`, `wait_for_kernel(*, want)` | whether a call runs compiled code, and blocking for it — for the Jacobian lane unless `want` names another |
| `device_status` | property — under `DEVICE`, the device answering; `None` otherwise; raises when none answers |
| `hessian_colors` | groups in the Hessian's compression |
| `buffer(x, *, want)` | a `Call` bound to its buffers, for a loop |
| `to_dot(*, all=False)` | the expression in Graphviz form; `all=True` draws the pruned nodes too |
| `nodes(*, want)` | how many nodes a call for `want` evaluates |
| `save(path)`, `verify(path)` | write this equation; raise unless `path` holds it |
| `loaded` | property — whether this equation was read rather than built |

`ddx.load(path)` reads one, and `ddx.equation(model, cache=path)` builds or reads
as the file allows.

---

## Compile-time expressions

The same derivatives are available on expressions whose shape is known when you
compile, header-only and `constexpr` throughout — `#include "ddx.hpp"`, link
`ddx::ddx`. Symbols are named by template argument, and `Equation` carries the
whole API:

```cpp
constexpr auto x = var<"x">;
constexpr auto y = var<"y">;

constexpr auto eq = Equation{x * y + 2.0 * x};
constexpr auto g = eq.jacobian(std::array{2.0, 3.0});
static_assert(g[0] == 5.0);        // ∂f/∂x
static_assert(g[1] == 2.0);        // ∂f/∂y
```

Points come in the same four spellings, and results are `std::array` or a tensor
rather than `result<std::vector<T>>` — the shapes are in the type, so nothing
has to be checked at run time.

| Member | Answers |
|---|---|
| `evaluate(point)` | f at the point |
| `jacobian(point)` | J, `J[i, j] = ∂fᵢ/∂xⱼ`; ∇f itself when m == 1 |
| `hessian(point)` | ∇²f — on symbols declared `var<"x", dual>` |
| `derivative_tensor<K>(point)` | all K-th order partials, any `K ≥ 1` |
| `univariate_derivative<K>(x0)` | the K-th derivative of a one-symbol function |

```cpp
constexpr auto H = Equation{var<"a", dual> * var<"b", dual>}.hessian(2.0, 3.0);
H[0, 1];                                          // 1.0

Equation{x * y}.derivative_tensor<3>(1.0, 2.0);   // rank 3
```

`record(named<"n">(3), named<"x">(1.5))` collects keyword arguments into a value
with the keys in its type, readable with `m.get<"n">()` or `m["x"_s]`.

## Printing

Compile-time expressions and `Equation`s are formattable and streamable:

```cpp
std::format("{}", x * y + sin(x));      // "x * y + sin(x)"
std::format("{::.3f}", 2.0 * x);        // "2.000 * x" — the spec applies to every number
std::cout << (x - y * x) << '\n';       // "x - y * x"
```

An `Equation` prints each output with its derivative row underneath. A symbol
held constant for one partial prints with a `_c` suffix:

```
f0: x * y
  grad: y_c, x_c
```

Slot 0 of an `Equation` is the function and slot `k > 0` is ∂f/∂xₖ in canonical
order, each an expression you can print or store:

```cpp
auto eq = Equation{x * y};

std::format("{}", eq[idx<0>()]);        // "x * y"
std::format("{}", eq[idx<1>()]);        // "y_c" — ∂f/∂x
```

## License

[Boost Software License 1.0](LICENSE.txt). Suggestions and pull requests are
welcome.

What ddx bundles is permissively licensed as well, and
[THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) holds each licence: the
OpenCL headers and loader (Apache 2.0) in an OpenCL build, LLVM (Apache 2.0 with
LLVM exception), zlib and zstd in a JIT build and so in the wheels, pybind11 in
the Python module, and the vendored mdspan header. It is installed beside
`LICENSE.txt` and ships in the wheels and the NuGet package; whoever
redistributes a JIT or OpenCL build of the library passes it on with it.
