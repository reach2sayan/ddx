[![PyPI](https://img.shields.io/pypi/v/ddx-ad?logo=pypi&logoColor=white)](https://pypi.org/project/ddx-ad/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://pypi.org/project/ddx-ad/)
[![License](https://img.shields.io/badge/license-BSL--1.0-4B8BBE)](https://github.com/reach2sayan/ddx/blob/main/LICENSE.txt)

# ddx

Differentiate expressions built while the program runs. Write a model over
named symbols — terms looped over a data file, a model read from
configuration, an expression a user typed, a loss that switches shape on a
comparison — and ask it for values, gradients, Jacobians and Hessians. One
point at a time or a NumPy batch of thousands in a single call, interpreted or
compiled to machine code through LLVM.

```sh
pip install ddx-ad
```

The distribution is `ddx-ad` and the import is `ddx`; `pip install ddx` is an
unrelated project. There is one wheel per platform and CPython version
(3.11–3.14), and a wheel needs nothing installed beside it — LLVM is inside
the library.

| Wheel | JIT | OpenCL |
|---|---|---|
| Linux x86_64 (glibc 2.28+) | yes | no |
| Windows x64 | no — calls interpret | no |

There is no macOS wheel; elsewhere, `pip install .` from a checkout of the
[repository](https://github.com/reach2sayan/ddx) builds from source.
`ddx.has_jit` says whether the copy you have was built with the LLVM backend,
and `ddx.has_opencl` whether it was built with the OpenCL one. Everything below
works either way; without them, calls interpret.

## Models

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

Symbols are named inside the model with `ddx.var(name)` and order
alphabetically. `f.symbols` lists them, `f.arity` counts them, `f.outputs`
counts the model's outputs. A bare number mixes into an expression without
wrapping, so an accumulator can start at `0.0`:

```python
@ddx.equation
def fit():
    a, b = ddx.var("a"), ddx.var("b")
    loss = 0.0
    for t, y in data:
        loss += (y - a * ddx.exp(-b * t)) ** 2
    return loss
```

| Kind | Available |
|---|---|
| Arithmetic | `+` `-` `*` `/` `**` unary `-` `abs()` |
| Unary | `sin` `cos` `tan` `exp` `log` `log10` `sqrt` `cbrt` `abs` `sign` `asin` `acos` `atan` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` `erf` |
| Binary | `pow` `atan2` `hypot` `max` `min` |
| Comparison | `<` `<=` `>` `>=` `==` `!=` — each answers `1.0` or `0.0` |
| Conditional | `select(cond, if_true, if_false)` |

`add`, `mul`, `div` and `neg` are the operators spelled as functions.

Where a rule has a choice to make, it is made the same way everywhere: `abs`
differentiates to `0` at zero and `sign` to `0` throughout, `max` and `min`
give each side half at a tie, and `pow(a, b)` at `a == 0` answers the zeros its
constant arms say — `a⁰` is `1` and `0ᵇ` is `0`.

### Choosing between two expressions

A comparison is an expression, not a `bool`, so a model cannot branch on one
with `if`. `select` is the conditional, and it is not a branch: **both arms are
evaluated** and the condition picks one, so a batch keeps every point on the
same path.

```python
@ddx.equation
def capped():
    x = ddx.var("x")
    return ddx.select(x < 1.0, x * x, 2.0 * x - 1.0)   # C¹ at x = 1
```

The derivative is the taken arm's, and **the condition is never
differentiated**, so the symbols it tests get no partial through it. A
condition is any nonzero value; a comparison against `NaN` is false, while a
`NaN` used directly as a condition is true. A range test is
`(a < b) * (b < c)`.

### Written down instead

A model can be a string, or a list of strings for a system:

```python
g   = ddx.equation("exp(x) * sin(y)")
sys = ddx.equation(["x*x + y*y - 4", "x*y - 1"])
```

Free identifiers become the symbols, still ordered alphabetically. The grammar
is Python's arithmetic: `+ - * /`, `**` for exponentiation (`^` is not an
operator), parentheses, unary signs, decimals and exponent notation, and the
functions in the table above. Comparisons bind looser than arithmetic, so
`select(x*x < y + 1, x, y)` needs no parentheses, and there is **one
comparison per expression**: `a < b < c` is refused rather than read. A string
that does not parse raises `ddx.Error` with `bad_syntax`, `unknown_function` or
`wrong_argument_count`.

## Points

A point is a sequence, or a dict keyed by symbol name. A 2-D array is a
**batch** — shape `(symbols, points)`, one row per symbol:

```python
f.jacobian([2.0, 3.0])                      # positional, alphabetical
f.jacobian({"x": 2.0, "y": 3.0})            # by name
f.jacobian(np.array([[2.0], [3.0]]))        # a batch of one
f.jacobian(np.array([[2.0, 2.5], [3.0, 3.5]]))   # a batch of two
```

## Values and derivatives

Each call returns everything up to and including what it names, so a gradient
never costs a second evaluation:

| Call | Returns |
|---|---|
| `evaluate(x)` | `f` |
| `jacobian(x)` | `(f, J)` |
| `gradient(x)` | `J` alone, from a graph that does not compute `f` |
| `hessian(x)` | `(f, J, H)` |
| `jvp(v, x)` | `(f, J·v)` — the directional derivative of `f` along `v` |
| `vjp(w, x)` | `(f, wᵀJ)` — the gradient of `w · f` |
| `hvp(v, x)` | `(f, ∇f, H·v)` — the directional derivative of `∇f` along `v` |

```python
value, gradient = f.jacobian([2.0, 3.0])
value, gradient, hessian = f.hessian([2.0, 3.0])
hessian.shape                               # (2, 2)

value, gradient, hv = f.hvp([1.0, 0.0], [2.0, 3.0])
hv.shape                                    # (2,) — H·v, without forming H
```

Shapes follow the point: a single point gives a scalar `f`, an `(n,)` gradient
and an `(n, n)` Hessian; a batch of `p` appends that axis, giving `(p,)`,
`(n, p)` and `(n, n, p)`. A system prepends its output axis. The Hessian
arrives dense.

The three products never form the matrix they are named after, which is what
makes them worth having when n² storage is the problem — though not when speed
is, since one product costs about what the whole matrix does. `J·v` and `H·v`
take a direction over symbols, so it accepts the same spellings a point does —
a sequence, a dict, or a `(symbols, points)` batch alongside a batched point.
`wᵀJ` takes one weight per *function*, so it is positional only. `hvp` needs a
single-output model.

## Calling in a loop

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
so does binding `Want.HESSIAN` on a system.

Shapes are the ones the allocating calls answer with, `value` included: one
output at one point is a `float`, and everything else is an array. The point is
bound as an array whatever was passed, so `call.x` is writable even when the
argument was a list.

## Remembering the last call

`remember=True` keeps the last call: a repeated point is answered off the last
one, a point one symbol away sweeps only what that symbol reaches, and the
numbers are unchanged.

```python
f = ddx.equation(model, remember=True)
value = f(x)              # swept
grad = f.gradient(x)      # its own lane
again = f(x)              # nothing swept
```

It applies to a point at a time and not to an array of them.

## Errors

`ddx.Error` is a `RuntimeError` carrying a code:

```python
try:
    f.jacobian({"z": 1.0})
except ddx.Error as e:
    print(e)                                # "no symbol of that name"
    e.code is ddx.errc.unknown_symbol       # True
```

`errc` is an `IntEnum`, so it compares and formats as its number — `e.code.name`
is the spelling.

| `errc` | Means |
|---|---|
| `wrong_arity` | the point does not supply one value per symbol |
| `wrong_direction` | a direction does not supply one value per symbol, or a covector one per function |
| `short_point` | a named point leaves a symbol unreached |
| `unknown_symbol` | a named point uses a name the equation does not have |
| `wrong_column_count` | a batch block has the wrong number of columns |
| `no_arena` | a symbol was named outside a model |
| `no_graph` | the model is a bare number, naming no function |
| `bad_syntax` | a string the grammar does not accept |
| `unknown_function` | a string calls a function that does not exist |
| `wrong_argument_count` | a string calls one with the wrong arity |
| `archive_io` | the file could not be read or written |
| `bad_archive` | not a ddx file, or a format this build does not read |
| `archive_corrupt` | the file's checksum or structure does not hold |
| `archive_mismatch` | the file loads, but does not describe this equation |
| `no_device`, `device_compile`, `device_launch` | no OpenCL device answered, its compiler refused the kernel, or a launch failed |
| `jit_target`, `jit_module`, `jit_object`, `jit_verify`, `jit_lookup` | the compiler could not produce or link a kernel |

## Compiling

`Options` is a frozen pydantic model, validated on the way in. `eq.options`
reads and assigns it; `eq.compile()` sets `backend=COMPILE`, waits for the
kernel, and returns the equation, so a configure-and-use reads in one line:

```python
f.compile(points=batch.shape[1]).jacobian(batch)
f.uses_kernel                               # True, once it has landed
f.options = ddx.Options(backend=ddx.Backend.INTERPRET)   # discards the kernel
```

`compile()` blocks by construction — it is `wait_for_kernel()` with the options
set first. Assigning `options` does not: calls interpret until the kernel lands
and switch over when it does.

| `Backend` | Calls |
|---|---|
| `INTERPRET` | walk the graph |
| `COMPILE` | start the compile at once, interpret until it lands |
| `ADAPT` | compile a lane once it has been asked for `warm_points` batch points |
| `DEVICE` | build the graph for the OpenCL device `device` names |

`points` is the batch you intend to hand one call, stated because the kernel is
built before any call exists to infer it from. It decides the lane width and
nothing else: a call carrying some other number is answered correctly, just not
by the kernel that number would have built. `cache_dir` keeps compiled objects
between runs, so a second run links instead of compiling — roughly three orders
of magnitude quicker; an empty string disables it.

`f.compile(backend=ddx.Backend.DEVICE)` builds for an OpenCL device instead. An
empty `Options.device` takes the first GPU with double precision; anything else
is matched, ignoring case, against the platform and device name — `"NVIDIA"`,
`"gfx1035"`, `"Intel"`. `f.device_status` names the device, is `None` under
any other backend, and raises `ddx.Error` when no device answers. No wheel
carries the device backend; a source build asks for it with
`-C cmake.define.DDX_BUILD_OPENCL=ON`. Where it is missing, `ddx.has_opencl` is
`False`, `DEVICE` interprets, and `device_status` raises `errc.no_device`.

## Saving and loading

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

A string model caches the same way: `ddx.equation("exp(x) * y", cache="f.ddx")`.

`save`, `load` and `verify` raise `ddx.Error` rather than answering `False`:
unreadable, unloadable and "a different equation" are three different `errc`
values, and only the code says which.

## Reference

| Member | Is |
|---|---|
| `arity`, `outputs`, `symbols` | properties — symbol count, output count, canonical names |
| `evaluate(x)`, `__call__(x)` | `f` at the point or batch |
| `jacobian(x)` | `(f, J)` |
| `gradient(x)` | `J` alone |
| `jvp(v, x)`, `vjp(w, x)`, `hvp(v, x)` | `(f, J·v)`, `(f, wᵀJ)`, `(f, ∇f, H·v)` |
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

`ddx.load(path)` reads one, and `ddx.equation(model, cache=path)` builds or
reads as the file allows.

## License

[Boost Software License 1.0](https://github.com/reach2sayan/ddx/blob/main/LICENSE.txt).
The wheels bundle LLVM (Apache 2.0 with LLVM exception), zlib, zstd and
pybind11; [THIRD-PARTY-NOTICES.txt](https://github.com/reach2sayan/ddx/blob/main/THIRD-PARTY-NOTICES.txt)
holds each licence and ships in the wheel.
