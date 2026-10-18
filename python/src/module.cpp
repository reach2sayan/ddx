#include "columns.hpp"
#include "equation.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/builder.hpp"
#include "rt/expressions.hpp"
#include "rt/tensor.hpp"
#include "rt/text/ast.hpp"
#include "rt/text/lower.hpp"
#include "util/ranges.hpp"
#include "util/scope_guard.hpp"
#include "util/version.hpp"

#include "jit/kernel.hpp"

#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ddx::py {
namespace {

// The arena a model is assembled in, for the duration of the call.  Ours rather
// than the one rt::var(name) reads, because what has to survive is the shared
// ownership: every symbol below goes through the two-argument rt::var.
thread_local std::shared_ptr<rt::Builder<double>> current_arena;

// The exception type Python sees.  Deliberately leaked: it is a type object
// that lives as long as the module, and a translator is a plain function
// pointer, so it cannot capture.
pyb::handle error_class;

// A symbol or a parameter of the model running now.  C++ carries a poison
// instead, unable to afford a branch per input; Python already pays more per
// call than the branch costs.
[[nodiscard]] PyExpression make_input(rt::OpCode kind, std::string_view name) {
  if (!current_arena) {
    fail_with(errc::outside_equation);
  }
  const auto e = rt::named(*current_arena, kind, name);
  if (e.poisoned()) {
    fail_with(errc::name_clash);
  }
  return {e, current_arena};
}

// An implicit input and the residual `residual(z)` it is a root of; a
// poisoned residual is kept on the arena for the equation.
[[nodiscard]] PyExpression make_implicit(std::string_view name,
                                         const pyb::function &residual) {
  if (!current_arena) {
    fail_with(errc::outside_equation);
  }
  const auto z =
      rt::implicit(*current_arena, name, [&](const rt::Expression<double> &e) {
        return residual(PyExpression{e, current_arena})
            .cast<PyExpression>()
            .base();
      });
  if (z.poisoned()) {
    fail_with(errc::name_clash);
  }
  return {z, current_arena};
}

// The model running now, for a vector that names no scalar of it.
[[nodiscard]] PyExpression::Arena arena_now() {
  if (!current_arena) {
    fail_with(errc::outside_equation);
  }
  return current_arena;
}

// One root, or m of them from a tuple or list.  The count is a runtime value
// throughout, which is why these bindings sit below impl::Equation.
[[nodiscard]] std::vector<rt::NodeId> roots_of(const pyb::object &built,
                                               rt::Builder<double> &arena) {
  const bool many =
      pyb::isinstance<pyb::tuple>(built) || pyb::isinstance<pyb::list>(built);
  const auto returned =
      many ? pyb::reinterpret_borrow<pyb::sequence>(built)
           : pyb::reinterpret_borrow<pyb::sequence>(pyb::make_tuple(built));
  if (returned.empty()) {
    fail_with(errc::no_graph); // a system needs at least one function
  }

  return returned | std::views::transform([&arena](const pyb::handle item) {
           // By value, not by reference: a bare number is a pending literal,
           // and implicitly_convertible is what lets `lambda: 2.0` name a
           // constant.
           const auto e = pyb::cast<PyExpression>(item);
           if (e.poisoned()) {
             fail_with(errc::outside_equation);
           }
           return e.root_in(arena);
         }) |
         impl::to<std::vector<rt::NodeId>>();
}

// ddx.equation over sources rather than a callback: the same three steps, with
// the model read instead of run.  The roots go through roots_of, so a model
// naming no symbol is refused exactly as `lambda: 2.0` is, and every source is
// read before any of them is an equation -- a symbol the second function
// introduces still wants a slot.
[[nodiscard]] pyb::object text_equation(std::span<const std::string> sources) {
  auto arena = std::make_shared<rt::Builder<double>>();
  pyb::list built;
  for (const std::string &source : sources) {
    built.append(pyb::cast(PyExpression{unwrap(rt::text::parse(source).and_then(
                                            [&arena](const rt::text::Ast &ast) {
                                              return rt::text::lower(*arena,
                                                                     ast);
                                            })),
                                        arena}));
  }
  auto roots = roots_of(built, *arena);
  return pyb::cast(PyEquation{std::move(arena), std::move(roots)});
}

// The whole of ddx.equation: install an arena, run the model in it, take what
// comes back -- rt::equation(assemble)'s three steps, with the output count a
// number rather than a template parameter.
[[nodiscard]] pyb::object make_equation(const pyb::object &model) {
  auto arena = std::make_shared<rt::Builder<double>>();
  pyb::object built;
  {
    // RAII, and load-bearing: if the model raises, pybind11 unwinds through
    // here as py::error_already_set and this is what puts the previous arena
    // back.  Without it a raising model would poison every one after it.
    const impl::scoped_value scope{current_arena, arena};
    built = model();
  }

  auto roots = roots_of(built, *arena);
  pyb::object eq = pyb::cast(PyEquation{std::move(arena), std::move(roots)});

  // What functools.wraps would do, without a Python wrapper to do it: the
  // equation answers help() and repr() with the model's own name and docstring.
  static constexpr std::array kWrapped{"__name__", "__qualname__", "__doc__",
                                       "__module__"};
  for (const char *attr : kWrapped) {
    // One getattr, not a hasattr probe and then the fetch; an attribute that
    // exists and holds None is still copied, as hasattr read it.
    if (const auto value = pyb::reinterpret_steal<pyb::object>(
            PyObject_GetAttrString(model.ptr(), attr))) {
      eq.attr(attr) = value;
    } else {
      PyErr_Clear();
    }
  }
  return eq;
}

} // namespace
} // namespace ddx::py

PYBIND11_MODULE(_ddx, m) {
  namespace pyb = pybind11;
  using namespace ddx;
  using ddx::py::PyCall;
  using ddx::py::PyEquation;
  using ddx::py::PyExpression;
  using ddx::py::PyRestricted;

  m.doc() = "ddx's runtime expression graph, LLVM JIT and OpenCL device";
  m.attr("__version__") = DDX_VERSION_STRING;

#ifdef DDX_HAS_JIT
  m.attr("has_jit") = true;
#else
  m.attr("has_jit") = false;
#endif
#ifdef DDX_HAS_OPENCL
  m.attr("has_opencl") = true;
#else
  m.attr("has_opencl") = false;
#endif

  // The class first, then a translator that puts the code on the instance --
  // register_exception would install one that carries only the message.
  ddx::py::error_class =
      pyb::exception<ddx::py::PyError>(m, "Error", PyExc_RuntimeError)
          .release();
  pyb::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const ddx::py::PyError &e) {
      pyb::object raised = ddx::py::error_class(e.what());
      raised.attr("code") = pyb::cast(e.code);
      PyErr_SetObject(ddx::py::error_class.ptr(), raised.ptr());
    }
  });

  // Off the library's own table, so an errc added there reaches Python without
  // this file changing, each member carrying that table's text as its
  // docstring.  The literal rather than detail::kMessages[...].data(): a
  // string_view need not be null-terminated.
  pyb::native_enum<errc> errors(m, "errc", "enum.IntEnum",
                                "Why ddx refused; Error.code carries one.");
#define DDX_PY_ERRC(name, text) errors.value(#name, errc::name, text);
  DDX_ERRC_TABLE(DDX_PY_ERRC)
#undef DDX_PY_ERRC
  errors.finalize();

  // Each operator twice: once against another expression, once against a plain
  // number.  implicitly_convertible already makes the second work at run time,
  // but only a real overload puts `float` in the signature -- and the signature
  // is what reaches the generated stubs, so without it a type checker rejects
  // `x * 2.0` while the interpreter accepts it.
  pyb::class_<PyExpression> expression(m, "Expression");
  const auto binop = [&expression](const char *name, auto fn) {
    expression
        .def(
            name,
            [fn](const PyExpression &l, const PyExpression &r) {
              return fn(l, r);
            },
            pyb::is_operator())
        .def(
            name,
            [fn](const PyExpression &l, double v) {
              return fn(l, PyExpression{v});
            },
            pyb::is_operator());
  };
  expression.def(pyb::init<double>(), pyb::arg("value"))
      .def("__repr__", &PyExpression::repr)
      // A vector: its length, one element as a scalar, and its shape.
      .def("__len__", &PyExpression::size)
      .def("__getitem__", &PyExpression::at, pyb::arg("i"))
      .def_property_readonly("shape", [](const PyExpression &e) {
        return pyb::tuple(pyb::cast(e.shape()));
      });
  binop("__add__", [](const auto &l, const auto &r) { return l + r; });
  binop("__radd__", [](const auto &l, const auto &r) { return r + l; });
  // On a copy, so `y = acc; acc += t` leaves y alone; a loop of += is one
  // node of all its terms.
  binop("__iadd__", [](const auto &l, const auto &r) {
    PyExpression acc = l;
    return acc += r;
  });
  binop("__isub__", [](const auto &l, const auto &r) {
    PyExpression acc = l;
    return acc += -r;
  });
  binop("__sub__", [](const auto &l, const auto &r) { return l - r; });
  binop("__rsub__", [](const auto &l, const auto &r) { return r - l; });
  binop("__mul__", [](const auto &l, const auto &r) { return l * r; });
  binop("__rmul__", [](const auto &l, const auto &r) { return r * l; });
  binop("__truediv__", [](const auto &l, const auto &r) { return l / r; });
  binop("__rtruediv__", [](const auto &l, const auto &r) { return r / l; });
  binop("__pow__", [](const auto &l, const auto &r) {
    return binary(rt::OpCode::Pow, l, r);
  });
  binop("__rpow__", [](const auto &l, const auto &r) {
    return binary(rt::OpCode::Pow, r, l);
  });
  // `1.0 < x` reaches __gt__ by Python's own reflection.  __ne__ is
  // spelled out: Python would otherwise derive it as `not (a == b)`.
  binop("__lt__", [](const auto &l, const auto &r) { return l < r; });
  binop("__le__", [](const auto &l, const auto &r) { return l <= r; });
  binop("__gt__", [](const auto &l, const auto &r) { return l > r; });
  binop("__ge__", [](const auto &l, const auto &r) { return l >= r; });
  binop("__eq__", [](const auto &l, const auto &r) { return l == r; });
  binop("__ne__", [](const auto &l, const auto &r) { return l != r; });
  expression
      .def(
          "__neg__", [](const PyExpression &u) { return -u; },
          pyb::is_operator())
      .def(
          "__abs__",
          [](const PyExpression &u) { return unary(rt::OpCode::Abs, u); },
          pyb::is_operator())
      // A comparison is a value with no truth until a point is given, so
      // `if x < 1:` and `a if x < 1 else b` refuse rather than take the first
      // arm every time.
      .def("__bool__", [](const PyExpression &) -> bool {
        throw pyb::type_error("the truth value of an Expression is ambiguous; "
                              "use select(cond, if_true, if_false)");
      });

  pyb::implicitly_convertible<double, PyExpression>();

  // After the class, not before: a signature naming a type pybind11 has not
  // been told about yet renders as the raw C++ spelling, which is what reaches
  // the generated stubs.
  // Text first, and typed: the model overload takes pyb::object, which a str
  // matches too, and pybind11 tries these in the order they are registered.
  m.def(
      "equation",
      [](const std::string &source) {
        return ddx::py::text_equation(std::span{&source, 1});
      },
      pyb::arg("source"));
  m.def(
      "equation",
      [](const std::vector<std::string> &sources) {
        return ddx::py::text_equation(sources);
      },
      pyb::arg("source"));
  m.def("equation", &ddx::py::make_equation, pyb::arg("model"));
#define DDX_PY_INPUT(fn, Op, ...)                                              \
  if constexpr (rt::OpCode::Op != rt::OpCode::Implicit) {                      \
    m.def(                                                                     \
        #fn,                                                                   \
        [](std::string_view name) {                                            \
          return ddx::py::make_input(rt::OpCode::Op, name);                    \
        },                                                                     \
        pyb::arg("name"));                                                     \
  }
  DDX_RT_NAMED_TABLE(DDX_PY_INPUT)
#undef DDX_PY_INPUT
  m.def("implicit", &ddx::py::make_implicit, pyb::arg("name"),
        pyb::arg("residual"));

  // Off the opcode table, by each row's factory spelling, so an opcode added
  // there reaches Python without this file changing; select, the one ternary,
  // is below.
  for (const auto i : std::views::iota(0uz, rt::op_count)) {
    const auto op = static_cast<rt::OpCode>(i);
    const std::string name{rt::name_of(op)};
    switch (rt::kind_of(op)) {
    case rt::Kind::Unary:
      m.def(
          name.c_str(), [op](const PyExpression &u) { return unary(op, u); },
          pyb::arg("x"));
      break;
    case rt::Kind::Binary:
      m.def(
          name.c_str(),
          [op](const PyExpression &l, const PyExpression &r) {
            return binary(op, l, r);
          },
          pyb::arg("x"), pyb::arg("y"));
      break;
    default:
      break;
    }
  }

  // Both arms are evaluated and the condition picks one; any nonzero condition
  // is true, as in C.
  m.def(
      "select",
      [](const PyExpression &c, const PyExpression &t, const PyExpression &f) {
        return ddx::py::select(c, t, f);
      },
      pyb::arg("cond"), pyb::arg("if_true"), pyb::arg("if_false"));

  // The vector factories and reductions: an Expression with a width is a
  // vector, and the arithmetic above is elementwise over one.
  m.def(
      "vec",
      [](std::string_view name, std::size_t n) {
        const auto arena = ddx::py::arena_now();
        return PyExpression::checked(rt::vec(*arena, name, n), arena);
      },
      pyb::arg("name"), pyb::arg("n"));
  m.def(
      "pack",
      [](const std::vector<PyExpression> &scalars) {
        return PyExpression::pack(scalars, ddx::py::current_arena);
      },
      pyb::arg("scalars"));
  m.def(
      "vec_constant",
      [](const std::vector<double> &data) {
        const auto arena = ddx::py::arena_now();
        return PyExpression::checked(
            rt::vec_constant(*arena, std::span<const double>{data}), arena);
      },
      pyb::arg("data"));
  m.def(
      "mat_constant",
      [](std::size_t rows, std::size_t cols, const std::vector<double> &data) {
        const auto arena = ddx::py::arena_now();
        return PyExpression::checked(
            rt::mat_constant(*arena, rows, cols, std::span<const double>{data}),
            arena);
      },
      pyb::arg("rows"), pyb::arg("cols"), pyb::arg("data"));
  m.def(
      "sum",
      [](const PyExpression &v) {
        return reduce(v, [](const auto &a) { return sum(a); });
      },
      pyb::arg("v"));
  m.def(
      "dot",
      [](const PyExpression &u, const PyExpression &v) {
        return reduce(u, v,
                      [](const auto &a, const auto &b) { return dot(a, b); });
      },
      pyb::arg("u"), pyb::arg("v"));
  m.def(
      "gemv",
      [](const PyExpression &M, const PyExpression &v, bool transposed) {
        return reduce(M, v, [transposed](const auto &a, const auto &b) {
          return gemv(a, b, transposed);
        });
      },
      pyb::arg("matrix"), pyb::arg("v"), pyb::arg("transposed") = false);
  m.def("release_tapes", &rt::release_tapes<double>,
        "Free this thread's sweep tapes; the next sweep grows them again.");
  m.def(
      "outer",
      [](const PyExpression &u, const PyExpression &w) {
        return reduce(u, w,
                      [](const auto &a, const auto &b) { return outer(a, b); });
      },
      pyb::arg("u"), pyb::arg("w"));

  pyb::native_enum<jit::Backend>(m, "Backend", "enum.IntEnum",
                                 "Whether an equation compiles its graph.")
      .value("INTERPRET", jit::Backend::Interpret)
      .value("COMPILE", jit::Backend::Compile)
      .value("ADAPT", jit::Backend::Adapt)
      .value("DEVICE", jit::Backend::Device)
      .finalize();

  pyb::native_enum<jit::VecLib>(
      m, "VecLib", "enum.IntEnum",
      "Which vector math library a compiled kernel may call.")
      .value("NONE", jit::VecLib::None)
      .value("AUTO", jit::VecLib::Auto)
      .value("LIBMVEC", jit::VecLib::Libmvec)
      .finalize();

  pyb::native_enum<jit::OptLevel>(m, "OptLevel", "enum.IntEnum",
                                  "LLVM's -O0 to -O3, for the IR pipeline and "
                                  "for codegen.")
      .value("O0", jit::OptLevel::O0)
      .value("O1", jit::OptLevel::O1)
      .value("O2", jit::OptLevel::O2)
      .value("O3", jit::OptLevel::O3)
      .finalize();

  pyb::native_enum<PyEquation::Want> want{m, "Want", "enum.IntEnum",
                                          "Which blocks a bound call fills."};
  // JacobianJvp as JACOBIAN_JVP: an underscore before every inner capital.
  for (const rt::Want w : rt::want_values) {
    std::string name;
    for (const auto [i, c] : rt::name_of(w) | std::views::enumerate) {
      if (i > 0 && std::isupper(static_cast<unsigned char>(c))) {
        name.push_back('_');
      }
      name.push_back(
          static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    want.value(name.c_str(), w);
  }
  want.finalize();

  // Bound once, called repeatedly: `x` is written into and the blocks are read
  // back.  Not constructible from Python -- Equation.buffer() is the only way
  // to get one, because it is the equation that owns the pipeline behind it.
  pyb::class_<PyCall>(m, "Call")
      .def("__repr__", &PyCall::repr)
      .def("__call__", &PyCall::operator())
      .def_property_readonly("x", &PyCall::x)
      .def_property_readonly("value", &PyCall::value)
      .def_property_readonly("jacobian", &PyCall::jacobian)
      .def_property_readonly("hessian", &PyCall::hessian);

  // Internal.  ddx.Options is the pydantic model over this, which is where the
  // ranges are checked; the defaults stay here, because they follow how the
  // library was built.
  pyb::class_<jit::Options> options{m, "_Options"};
  // Flat in Python: the identity/policy split is the C++ struct's, and the
  // pydantic model over this names every field at one level.
  const auto codegen = [&options](const char *name, auto jit::Codegen::*field) {
    options.def_property(
        name, [field](const jit::Options &o) { return o.codegen.*field; },
        [field](jit::Options &o, decltype(o.codegen.*field) v) {
          o.codegen.*field = v;
        });
  };
  options.def(pyb::init<>())
      .def_readwrite("backend", &jit::Options::backend)
      .def_readwrite("points", &jit::Options::points)
      // None derives the width; a stated one must hold a point.
      .def_property(
          "lanes",
          [](const jit::Options &o) { return o.codegen.lanes.stated(); },
          [](jit::Options &o, std::optional<unsigned> width) {
            const auto lanes = width ? jit::Lanes::exactly(*width)
                                     : std::optional{jit::Lanes::derived()};
            if (!lanes) {
              throw pyb::value_error("lanes: a stated width is at least 1");
            }
            o.codegen.lanes = *lanes;
          })
      .def_readwrite("warm_points", &jit::Options::warm_points)
      .def_readwrite("hot_points", &jit::Options::hot_points)
      .def_readwrite("time_passes", &jit::Options::time_passes)
      .def_readwrite("cache_dir", &jit::Options::cache_dir)
      .def_readwrite("device", &jit::Options::device)
      .def(pyb::self == pyb::self);
  codegen("opt_level", &jit::Codegen::opt_level);
  codegen("codegen_level", &jit::Codegen::codegen_level);
  codegen("slp_vectorize", &jit::Codegen::slp_vectorize);
  codegen("loop_vectorize", &jit::Codegen::loop_vectorize);
  codegen("veclib", &jit::Codegen::veclib);
  codegen("contract", &jit::Codegen::contract);

  pyb::class_<PyRestricted>(m, "Restricted")
      .def("jacobian", &PyRestricted::jacobian, pyb::arg("x"))
      .def("jacobian_jvp", &PyRestricted::jacobian_jvp, pyb::arg("v"),
           pyb::arg("x"))
      .def_property_readonly("symbols", &PyRestricted::symbols)
      .def_property_readonly("jacobian_columns",
                             &PyRestricted::jacobian_columns);

  pyb::class_<PyEquation>(m, "Equation", pyb::dynamic_attr())
      .def("__repr__", &PyEquation::repr)
      .def("__call__", &PyEquation::evaluate, pyb::arg("x"))
      .def("evaluate", &PyEquation::evaluate, pyb::arg("x"))
      .def("jacobian", &PyEquation::jacobian, pyb::arg("x"))
      .def("gradient", &PyEquation::gradient, pyb::arg("x"))
      .def("hessian", &PyEquation::hessian, pyb::arg("x"))
      // The direction leads, as it does in C++: a trailing one could not be
      // told from the point.
      .def("jvp", &PyEquation::jvp, pyb::arg("v"), pyb::arg("x"))
      .def("vjp", &PyEquation::vjp, pyb::arg("w"), pyb::arg("x"))
      .def("hvp", &PyEquation::hvp, pyb::arg("v"), pyb::arg("x"))
      .def("dense_hessian", &PyEquation::dense_hessian, pyb::arg("x"))
      .def("jacobian_jvp", &PyEquation::jacobian_jvp, pyb::arg("v"),
           pyb::arg("x"))
      .def("residual", &PyEquation::residual_at, pyb::arg("x"))
      .def(
          "restrict",
          [](const PyEquation &self, const std::vector<std::string> &names) {
            return PyRestricted{self, py::unwrap(self.restrict(names))};
          },
          pyb::arg("names"), pyb::keep_alive<0, 1>())
      .def(
          "buffer",
          [](PyEquation &self, const pyb::handle &x, PyEquation::Want want) {
            // By pointer: a Point pins its buffers, so a PyCall never moves.
            return std::make_unique<PyCall>(self, want, x);
          },
          pyb::arg("x"), pyb::kw_only(),
          pyb::arg("want") = PyEquation::Want::Jacobian,
          pyb::keep_alive<0, 1>())
      .def("nodes", &PyEquation::nodes, pyb::kw_only(),
           pyb::arg("want") = PyEquation::Want::Jacobian)
      .def_property_readonly("symbols", &PyEquation::symbols_list)
      .def_property_readonly("params", &PyEquation::params_list)
      .def_property_readonly("inputs", &PyEquation::inputs_list)
      .def_property_readonly("arity", &PyEquation::width)
      .def_property_readonly("outputs", &PyEquation::height)
      .def_property_readonly("uses_kernel", &PyEquation::uses_kernel)
      .def_property_readonly("device_status", &PyEquation::device)
      // Colours, not n: the Hessian is stored compressed and scattered on
      // read, so this is what the compiled kernel actually computes per point.
      .def_property_readonly("hessian_colours",
                             [](const PyEquation &self) {
                               const auto colours = self.hessian_colours();
                               if (!colours) {
                                 py::fail_with(errc::not_single_output);
                               }
                               return *colours;
                             })
      .def("wait_for_kernel", &PyEquation::wait_for_kernel, pyb::kw_only(),
           pyb::arg("want") = PyEquation::Want::Jacobian)
      .def_property(
          "_options", [](const PyEquation &self) { return self.options(); },
          [](PyEquation &self, const jit::Options &opt) {
            (void)self.options(opt);
          });
}
