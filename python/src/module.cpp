#include "columns.hpp"
#include "equation.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/arena.hpp"
#include "rt/expressions.hpp"
#include "rt/tensor.hpp"
#include "rt/text/ast.hpp"
#include "rt/text/lower.hpp"
#include "util/ranges.hpp"
#include "util/scope_guard.hpp"
#include "util/version.hpp"

#include "jit/kernel.hpp"

#include <boost/describe/enumerators.hpp>
#include <boost/mp11/algorithm.hpp>

#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ddx::py {
namespace {

// The arena the running model builds into. It is separate from rt's own
// because Python needs shared ownership of it.
thread_local std::shared_ptr<rt::Arena<double>> current_arena;

// The exception type Python sees. It is leaked on purpose, because the
// translator is a plain function pointer and cannot capture it.
pyb::handle error_class;

// The arena of the model running now, or a refusal outside equation().
[[nodiscard]] PyExpression::SharedArena arena_now() {
  if (!current_arena) {
    fail_with(errc::outside_equation);
  }
  return current_arena;
}

// A symbol or parameter of the running model. Unlike C++, which carries a
// poison, Python raises at once, since the extra branch costs nothing here.
[[nodiscard]] PyExpression make_input(rt::OpCode kind, std::string_view name) {
  const auto arena = arena_now();
  const auto e = rt::named(*arena, kind, name);
  if (e.poisoned()) {
    fail_with(errc::name_clash);
  }
  return {e, arena};
}

// An implicit input z defined as a root of `residual(z)`. A poisoned residual
// is left on the arena for the equation to report.
[[nodiscard]] PyExpression make_implicit(std::string_view name,
                                         const pyb::function &residual) {
  const auto arena = arena_now();
  const auto z =
      rt::implicit(*arena, name, [&](const rt::Expression<double> &e) {
        return residual(PyExpression{e, arena}).cast<PyExpression>().base();
      });
  if (z.poisoned()) {
    fail_with(errc::name_clash);
  }
  return {z, arena};
}

// The roots of a model's result, one expression or a tuple or list of them.
[[nodiscard]] std::vector<rt::NodeId> roots_of(const pyb::object &built,
                                               rt::Arena<double> &arena) {
  const bool many =
      pyb::isinstance<pyb::tuple>(built) || pyb::isinstance<pyb::list>(built);
  const auto returned =
      many ? pyb::reinterpret_borrow<pyb::sequence>(built)
           : pyb::reinterpret_borrow<pyb::sequence>(pyb::make_tuple(built));
  if (returned.empty()) {
    fail_with(errc::no_graph); // a system needs at least one function
  }

  return returned | std::views::transform([&arena](const pyb::handle item) {
           // Cast by value, so a bare number such as `lambda: 2.0` converts
           // to a constant.
           const auto e = pyb::cast<PyExpression>(item);
           if (e.poisoned()) {
             fail_with(*rt::detail::Router::why(e.base()));
           }
           return e.root_in(arena);
         }) |
         impl::to<std::vector<rt::NodeId>>();
}

// ddx.equation over source strings instead of a callback. Every source is
// parsed into one arena before the equation is formed.
[[nodiscard]] pyb::object text_equation(std::span<const std::string> sources) {
  auto arena = std::make_shared<rt::Arena<double>>();
  pyb::list built;
  for (const std::string &source : sources) {
    const rt::text::Ast ast = unwrap(rt::text::parse(source));
    built.append(
        pyb::cast(PyExpression{unwrap(rt::text::lower(*arena, ast)), arena}));
  }
  auto roots = roots_of(built, *arena);
  return pyb::cast(PyEquation{std::move(arena), std::move(roots)});
}

// ddx.equation: installs an arena, runs the model in it, and forms an
// equation from what the model returns.
[[nodiscard]] pyb::object make_equation(const pyb::object &model) {
  auto arena = std::make_shared<rt::Arena<double>>();
  pyb::object built;
  {
    // Restores the previous arena even when the model raises, or one failing
    // model would break every later one.
    const impl::scoped_value scope{current_arena, arena};
    built = model();
  }

  auto roots = roots_of(built, *arena);
  pyb::object eq = pyb::cast(PyEquation{std::move(arena), std::move(roots)});

  // Copies the model's name and docstring onto the equation, as
  // functools.wraps would.
  static constexpr std::array kWrapped{"__name__", "__qualname__", "__doc__",
                                       "__module__"};
  for (const char *attr : kWrapped) {
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
  using ddx::py::PyTransposed;

  m.doc() = "ddx's runtime expression graph and LLVM JIT";
  m.attr("__version__") = DDX_VERSION_STRING;

#ifdef DDX_HAS_JIT
  m.attr("has_jit") = true;
#else
  m.attr("has_jit") = false;
#endif

  // A custom translator, because register_exception's would carry only the
  // message and not the error code.
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

  // Generated from the errc table, so new codes reach Python with no edit here.
  pyb::native_enum<errc> errors(m, "errc", "enum.IntEnum",
                                "Why ddx refused; Error.code carries one.");
#define DDX_PY_ERRC(name, text) errors.value(#name, errc::name, text);
  DDX_ERRC_TABLE(DDX_PY_ERRC)
#undef DDX_PY_ERRC
  errors.finalize();

  // Each operator is bound twice, for an expression and for a float, so the
  // generated stubs let type checkers accept `x * 2.0`.
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
      // Vector access: length, elements and shape.
      .def("__len__", &PyExpression::size)
      .def("__getitem__", &PyExpression::at, pyb::arg("i"))
      .def("__iter__",
           [](const PyExpression &e) {
             if (e.size() == 0) {
               ddx::py::fail_with(ddx::errc::tensor_shape);
             }
             pyb::list elements;
             for (const std::size_t i : std::views::iota(0uz, e.size())) {
               elements.append(e.at(i));
             }
             return pyb::iter(elements);
           })
      .def_property_readonly("shape", [](const PyExpression &e) {
        return pyb::tuple(pyb::cast(e.shape()));
      });
  // Uses the C++ operators, so `x - y` builds the same node as in C++. __ne__
  // is bound explicitly, or Python would derive it as `not (a == b)`.
  using Base = rt::Expression<double>;
  const auto infix = [&binop](const char *name, const char *reflected,
                              auto fn) {
    binop(name, [fn](const PyExpression &l, const PyExpression &r) {
      return combine(l, r, fn);
    });
    if (reflected != nullptr) {
      binop(reflected, [fn](const PyExpression &l, const PyExpression &r) {
        return combine(r, l, fn);
      });
    }
  };
  infix("__add__", "__radd__", std::plus<>{});
  infix("__sub__", "__rsub__", std::minus<>{});
  infix("__mul__", "__rmul__", std::multiplies<>{});
  infix("__truediv__", "__rtruediv__", std::divides<>{});
  infix("__pow__", "__rpow__", [](const Base &l, const Base &r) {
    return Base::form(rt::OpCode::Pow, l, r);
  });
  infix("__lt__", nullptr, std::less<>{});
  infix("__le__", nullptr, std::less_equal<>{});
  infix("__gt__", nullptr, std::greater<>{});
  infix("__ge__", nullptr, std::greater_equal<>{});
  infix("__eq__", nullptr, std::equal_to<>{});
  infix("__ne__", nullptr, std::not_equal_to<>{});
  // Applied to a copy, so `y = acc; acc += t` leaves y unchanged. A loop of
  // +=, *= or /= builds one node over all its terms.
  binop("__iadd__", [](PyExpression acc, const PyExpression &r) {
    return acc += r;
  });
  binop("__isub__", [](PyExpression acc, const PyExpression &r) {
    return acc += unary(rt::OpCode::Neg, r);
  });
  binop("__imul__", [](PyExpression acc, const PyExpression &r) {
    return acc *= r;
  });
  binop("__itruediv__", [](PyExpression acc, const PyExpression &r) {
    return acc /= r;
  });
  expression
      .def(
          "__neg__",
          [](const PyExpression &u) { return unary(rt::OpCode::Neg, u); },
          pyb::is_operator())
      .def(
          "__abs__",
          [](const PyExpression &u) { return unary(rt::OpCode::Abs, u); },
          pyb::is_operator())
      // C++'s `%`: a dot product of vectors, or a matrix-vector product.
      .def(
          "__matmul__",
          [](const PyExpression &l, const PyExpression &r) {
            return reduce(l, r,
                          [](const auto &a, const auto &b) { return a % b; });
          },
          pyb::is_operator())
      // A comparison has no truth value until evaluated, so `if x < 1:` raises
      // instead of silently taking one branch.
      .def("__bool__", [](const PyExpression &) -> bool {
        throw pyb::type_error("the truth value of an Expression is ambiguous; "
                              "use select(cond, if_true, if_false)");
      });

  pyb::implicitly_convertible<double, PyExpression>();

  // Registered after the class so the stubs show its Python name. The str
  // overload comes first, because the callable overload would also match a str.
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

  // Generated from the opcode table, so new opcodes reach Python with no edit
  // here.
  for (const auto i : std::views::iota(0uz, rt::op_count)) {
    const auto op = static_cast<rt::OpCode>(i);
    // These are already bound as operators.
    if (std::ranges::contains(std::array{rt::OpCode::Neg, rt::OpCode::Add,
                                         rt::OpCode::Mul, rt::OpCode::Div,
                                         rt::OpCode::Lt, rt::OpCode::Le},
                              op)) {
      continue;
    }
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

  // Both arms are evaluated; any nonzero condition is true, as in C.
  m.def(
      "select",
      [](const PyExpression &c, const PyExpression &t, const PyExpression &f) {
        return ddx::py::select(c, t, f);
      },
      pyb::arg("cond"), pyb::arg("if_true"), pyb::arg("if_false"));

  // Vector factories and reductions. Factories build in the arena of the
  // running equation() callback.
  const auto in_arena = [](auto make) {
    const auto arena = ddx::py::arena_now();
    return PyExpression::checked(make(*arena), arena);
  };
  m.def(
      "vec",
      [in_arena](std::string_view name, std::size_t n) {
        return in_arena([&](auto &b) { return rt::vec(b, name, n); });
      },
      pyb::arg("name"), pyb::arg("n"));
  m.def(
      "vec_param",
      [in_arena](std::string_view name, std::size_t n) {
        return in_arena([&](auto &b) { return rt::vec_param(b, name, n); });
      },
      pyb::arg("name"), pyb::arg("n"));
  m.def(
      "pack",
      [](const std::vector<PyExpression> &scalars) {
        return PyExpression::pack(scalars, ddx::py::current_arena);
      },
      pyb::arg("scalars"));
  // Read in place when already column-major, otherwise copied once.
  using Columns = pyb::array_t<double, pyb::array::f_style | pyb::array::forcecast>;
  const auto data_of = [](const Columns &a) {
    return std::span<const double>{a.data(), static_cast<std::size_t>(a.size())};
  };
  m.def(
      "vec_constant",
      [in_arena, data_of](const Columns &data) {
        return in_arena([&](auto &b) { return rt::vec_constant(b, data_of(data)); });
      },
      pyb::arg("data"));
  m.def(
      "mat_constant",
      [in_arena, data_of](const Columns &m) {
        if (m.ndim() != 2) {
          ddx::py::fail_with(errc::tensor_shape);
        }
        return in_arena([&](auto &b) {
          return rt::mat_constant(b, static_cast<std::size_t>(m.shape(0)),
                                  static_cast<std::size_t>(m.shape(1)),
                                  data_of(m));
        });
      },
      pyb::arg("matrix"), "A constant matrix from a 2-D array.");
  m.def(
      "sum",
      [](const PyExpression &v) {
        return reduce(v, [](const auto &a) { return sum(a); });
      },
      pyb::arg("v"));
  pyb::class_<PyTransposed>(m, "Transposed")
      .def(
          "__matmul__",
          [](const PyTransposed &t, const PyExpression &v) {
            return reduce(t.m, v, [](const auto &a, const auto &b) {
              return gemv(a, b, true);
            });
          },
          pyb::is_operator());
  m.def(
      "transpose", [](const PyExpression &M) { return PyTransposed{M}; },
      pyb::arg("matrix"), "M^T, as the left side of @.");
  m.def("release_tapes", &rt::release_tapes<double>,
        "Free this thread's sweep tapes; the next sweep grows them again.");
  m.def(
      "outer",
      [](const PyExpression &u, const PyExpression &w) {
        return reduce(u, w,
                      [](const auto &a, const auto &b) { return outer(a, b); });
      },
      pyb::arg("u"), pyb::arg("w"));

  // Symbolic derivatives, built in the model running now.
  const auto rooted = [](const PyExpression &f) {
    if (f.poisoned()) {
      ddx::py::fail_with(*rt::detail::Router::why(f.base()));
    }
    const auto arena = f.arena() ? f.arena() : ddx::py::arena_now();
    return std::pair{PyExpression::Base{*arena, f.base().id(*arena)}, arena};
  };
  m.def(
      "gradient",
      [rooted](const PyExpression &f) {
        const auto [e, arena] = rooted(f);
        return rt::gradient(e) | std::views::transform([&](const auto &d) {
                 return PyExpression::checked(d, arena);
               }) |
               impl::to<std::vector<PyExpression>>();
      },
      pyb::arg("f"), "df/dx for every symbol, in declaration order.");
  m.def(
      "derivative",
      [rooted](const PyExpression &f, std::string_view symbol) {
        const auto [e, arena] = rooted(f);
        return PyExpression::checked(rt::derivative(e, symbol), arena);
      },
      pyb::arg("f"), pyb::arg("symbol"));

  // Converts CamelCase to UPPER_SNAKE, such as JacobianJvp to JACOBIAN_JVP.
  const auto upper = [](std::string_view name) {
    std::string out;
    for (const auto [i, c] : name | std::views::enumerate) {
      if (i > 0 && std::isupper(static_cast<unsigned char>(c))) {
        out.push_back('_');
      }
      out.push_back(
          static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    return out;
  };
  const auto described = [&]<typename E>(const char *name, const char *doc) {
    pyb::native_enum<E> e{m, name, "enum.IntEnum", doc};
    boost::mp11::mp_for_each<boost::describe::describe_enumerators<E>>(
        [&](auto d) { e.value(upper(d.name).c_str(), d.value); });
    e.finalize();
  };
  described.operator()<jit::Backend>("Backend",
                                     "Whether an equation compiles its graph.");
  described.operator()<jit::VecLib>(
      "VecLib", "Which vector math library a compiled kernel may call.");
  described.operator()<jit::OptLevel>(
      "OptLevel", "LLVM's -O0 to -O3, for the IR pipeline and for codegen.");
  pyb::native_enum<PyEquation::Want> wants{m, "Want", "enum.IntEnum",
                                           "Which blocks a bound call fills."};
  for (const rt::Want w : rt::want_values) {
    wants.value(upper(rt::name_of(w)).c_str(), w);
  }
  wants.finalize();

  // A call bound once and run repeatedly. Only Equation.buffer() creates one,
  // because the equation owns the pipeline behind it.
  pyb::class_<PyCall>(m, "Call")
      .def("__repr__", &PyCall::repr)
      .def("__call__", &PyCall::operator())
      .def_property_readonly("x", &PyCall::x)
      .def_property_readonly("value", &PyCall::value)
      .def_property_readonly("jacobian", &PyCall::jacobian)
      .def_property_readonly("hessian", &PyCall::hessian);

  // Internal; ddx.Options wraps it and checks ranges. The defaults live here
  // because they follow how the library was built.
  pyb::class_<jit::Options> options{m, "_Options"};
  // The nested Codegen fields are exposed flat, as ddx.Options expects.
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
      // None derives the width; a stated width must be at least 1.
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
      .def(pyb::self == pyb::self);
  codegen("opt_level", &jit::Codegen::opt_level);
  codegen("codegen_level", &jit::Codegen::codegen_level);
  codegen("slp_vectorize", &jit::Codegen::slp_vectorize);
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
      .def("evaluate", &PyEquation::evaluate, pyb::arg("x"))
      .def("jacobian", &PyEquation::jacobian, pyb::arg("x"))
      .def("gradient", &PyEquation::gradient, pyb::arg("x"))
      .def("hessian", &PyEquation::hessian, pyb::arg("x"))
      // The direction comes first, as in C++.
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
            // Held by pointer, since a PyCall's Point must never move.
            return std::make_unique<PyCall>(self, want, x);
          },
          pyb::arg("x"), pyb::kw_only(),
          pyb::arg("want") = PyEquation::Want::Jacobian,
          pyb::keep_alive<0, 1>())
      .def("nodes", &PyEquation::nodes, pyb::kw_only(),
           pyb::arg("want") = PyEquation::Want::Jacobian)
      .def("census", &PyEquation::census_dict, pyb::kw_only(),
           pyb::arg("want") = PyEquation::Want::Jacobian)
      .def_property_readonly("symbols", &PyEquation::symbols_list)
      .def_property_readonly("params", &PyEquation::params_list)
      .def_property_readonly("inputs", &PyEquation::inputs_list)
      .def_property_readonly("arity", &PyEquation::width)
      .def_property_readonly("outputs", &PyEquation::height)
      // The number of colours the compressed Hessian sweep computes per point.
      .def_property_readonly("hessian_colours",
                             [](const PyEquation &self) {
                               const auto colours = self.hessian_colours();
                               if (!colours) {
                                 py::fail_with(errc::not_single_output);
                               }
                               return *colours;
                             })
      .def("uses_kernel", &PyEquation::uses_kernel, pyb::kw_only(),
           pyb::arg("want") = PyEquation::Want::Jacobian)
      .def("compile_status", &PyEquation::compile_status_of, pyb::kw_only(),
           pyb::arg("want") = PyEquation::Want::Jacobian)
      .def_property(
          "_options", [](const PyEquation &self) { return self.options(); },
          [](PyEquation &self, const jit::Options &opt) {
            (void)self.options(opt);
          });
}
