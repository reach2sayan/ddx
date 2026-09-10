#include "columns.hpp"
#include "equation.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/builder.hpp"
#include "rt/expressions.hpp"
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
#include <pybind11/stl/filesystem.h>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/describe/enum.hpp>
#include <boost/mp11/algorithm.hpp>

#include <algorithm>
#include <array>
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

[[nodiscard]] PyExpression make_var(std::string_view name) {
  if (!current_arena) {
    // C++ carries a poison instead, unable to afford a branch per symbol.
    // Python already pays more per call than the branch costs.
    fail_with(errc::no_arena);
  }
  return {rt::var(*current_arena, name), current_arena};
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
             fail_with(errc::no_arena);
           }
           return e.root_in(arena);
         }) |
         impl::to<std::vector<rt::NodeId>>();
}

// The model as built, or what a cache file holds if it still describes it.
// Best effort throughout: an absent file is the first run and a stale one is a
// rebuild, so neither raises -- Equation.loaded is how a caller tells.
[[nodiscard]] pyb::object
built_or_loaded(const std::optional<std::filesystem::path> &cache,
                std::shared_ptr<rt::Builder<double>> arena,
                std::vector<rt::NodeId> roots, bool remember) {
  if (cache) {
    if (auto snap = rt::load_snapshot<double>(*cache);
        snap && std::ranges::equal((*snap)->roots, roots) &&
        rt::digest<double>((*snap)->symbols, (*snap)->nodes,
                           (*snap)->model_nodes) ==
            rt::digest<double>(arena->symbols(), arena->nodes(),
                               arena->size())) {
      return pyb::cast(PyEquation{std::move(*snap), remember});
    }
  }
  pyb::object eq =
      pyb::cast(PyEquation{std::move(arena), std::move(roots), remember});
  if (cache) {
    // A cache that cannot be written is still a working equation.
    (void)rt::save(eq.cast<PyEquation &>().snapshot(), *cache);
  }
  return eq;
}

// An equation from a file and nothing else: no model runs and nothing is
// swept.  The output count is a number here, so unlike the C++ side there is
// nothing to state and nothing to refuse.
[[nodiscard]] pyb::object load_equation(const std::filesystem::path &path,
                                        bool remember) {
  return pyb::cast(
      PyEquation{unwrap(rt::load_snapshot<double>(path)), remember});
}

// ddx.equation over sources rather than a callback: the same three steps, with
// the model read instead of run.  The roots go through roots_of, so a model
// naming no symbol is refused exactly as `lambda: 2.0` is, and every source is
// read before any of them is an equation -- a symbol the second function
// introduces still wants a slot.
[[nodiscard]] pyb::object
text_equation(std::span<const std::string> sources,
              const std::optional<std::filesystem::path> &cache,
              bool remember) {
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
  return built_or_loaded(cache, std::move(arena), std::move(roots), remember);
}

// The whole of ddx.equation: install an arena, run the model in it, take what
// comes back -- rt::equation(assemble)'s three steps, with the output count a
// number rather than a template parameter.
[[nodiscard]] pyb::object
make_equation(const pyb::object &model,
              const std::optional<std::filesystem::path> &cache,
              bool remember) {
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
  pyb::object eq =
      built_or_loaded(cache, std::move(arena), std::move(roots), remember);

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
      .def("__repr__", &PyExpression::repr);
  binop("__add__", [](const auto &l, const auto &r) { return l + r; });
  binop("__radd__", [](const auto &l, const auto &r) { return r + l; });
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
      [](const std::string &source,
         const std::optional<std::filesystem::path> &cache, bool remember) {
        return ddx::py::text_equation(std::span{&source, 1}, cache, remember);
      },
      pyb::arg("source"), pyb::kw_only(), pyb::arg("cache") = pyb::none(),
      pyb::arg("remember") = false);
  m.def(
      "equation",
      [](const std::vector<std::string> &sources,
         const std::optional<std::filesystem::path> &cache, bool remember) {
        return ddx::py::text_equation(sources, cache, remember);
      },
      pyb::arg("source"), pyb::kw_only(), pyb::arg("cache") = pyb::none(),
      pyb::arg("remember") = false);
  m.def("equation", &ddx::py::make_equation, pyb::arg("model"), pyb::kw_only(),
        pyb::arg("cache") = pyb::none(), pyb::arg("remember") = false);
  m.def("load", &ddx::py::load_equation, pyb::arg("path"), pyb::kw_only(),
        pyb::arg("remember") = false);
  m.def("var", &ddx::py::make_var, pyb::arg("name"));

  // Off the opcode table, by each row's factory spelling, so an opcode added
  // there reaches Python without this file changing.  The comparisons are
  // operators here, and select is the one ternary, below.
  for (const auto i : std::views::iota(0uz, rt::op_count)) {
    const auto op = static_cast<rt::OpCode>(i);
    const std::string name{rt::name_of(op)};
    if (rt::arity_of(op) == 1) {
      m.def(
          name.c_str(), [op](const PyExpression &u) { return unary(op, u); },
          pyb::arg("x"));
    } else if (rt::arity_of(op) == 2 && op != rt::OpCode::Lt &&
               op != rt::OpCode::Le) {
      m.def(
          name.c_str(),
          [op](const PyExpression &l, const PyExpression &r) {
            return binary(op, l, r);
          },
          pyb::arg("x"), pyb::arg("y"));
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

  pyb::native_enum<jit::Backend>(m, "Backend", "enum.IntEnum",
                                 "Whether an equation compiles its graph.")
      .value("INTERPRET", jit::Backend::Interpret)
      .value("COMPILE", jit::Backend::Compile)
      .value("ADAPT", jit::Backend::Adapt)
      .value("DEVICE", jit::Backend::Device)
      .finalize();

  pyb::native_enum<jit::VecLib>(m, "VecLib", "enum.IntEnum",
                                "Which vector math library a lane may call.")
      .value("NONE", jit::VecLib::None)
      .value("AUTO", jit::VecLib::Auto)
      .value("LIBMVEC", jit::VecLib::Libmvec)
      .finalize();

  pyb::native_enum<jit::Level>(m, "Level", "enum.IntEnum",
                               "LLVM's -O0 to -O3, for the IR pipeline and "
                               "for codegen.")
      .value("O0", jit::Level::O0)
      .value("O1", jit::Level::O1)
      .value("O2", jit::Level::O2)
      .value("O3", jit::Level::O3)
      .finalize();

  pyb::native_enum<PyEquation::Want> want{m, "Want", "enum.IntEnum",
                                          "Which blocks a bound call fills."};
  boost::mp11::mp_for_each<
      boost::describe::describe_enumerators<PyEquation::Want>>([&](auto D) {
    const std::string name = boost::algorithm::to_upper_copy(std::string{D.name});
    want.value(name.c_str(), D.value);
  });
  want.finalize();

  // Bound once, called repeatedly: `x` is written into and the blocks are read
  // back.  Not constructible from Python -- Equation.buffer() is the only way
  // to get one, because it is the equation that owns the lane behind it.
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
      // Policy rather than identity: neither changes what a compile emits, so
      // neither belongs in a key that decides whether stored code may be run.
      .def_readwrite("retain_object", &jit::Options::retain_object)
      .def_readwrite("cache_dir", &jit::Options::cache_dir)
      .def_readwrite("device", &jit::Options::device)
      .def(pyb::self == pyb::self);
  codegen("opt_level", &jit::Codegen::opt_level);
  codegen("codegen_level", &jit::Codegen::codegen_level);
  codegen("slp", &jit::Codegen::slp);
  codegen("loop_vectorize", &jit::Codegen::loop_vectorize);
  codegen("veclib", &jit::Codegen::veclib);
  codegen("contract", &jit::Codegen::contract);

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
      .def(
          "buffer",
          [](PyEquation &self, const pyb::handle &x, PyEquation::Want want) {
            // By pointer: a Point pins its buffers, so a PyCall never moves.
            return std::make_unique<PyCall>(self, want, x);
          },
          pyb::arg("x"), pyb::kw_only(),
          pyb::arg("want") = PyEquation::Want::Jacobian, pyb::keep_alive<0, 1>())
      .def("to_dot", &PyEquation::to_dot, pyb::kw_only(),
           pyb::arg("all") = false)
      .def("nodes", &PyEquation::nodes, pyb::kw_only(),
           pyb::arg("want") = PyEquation::Want::Jacobian)
      .def("save", &PyEquation::save, pyb::arg("path"))
      .def("verify", &PyEquation::verify, pyb::arg("path"))
      .def_property_readonly("loaded", &PyEquation::loaded)
      .def_property_readonly("symbols", &PyEquation::symbols_list)
      .def_property_readonly("arity", &PyEquation::arity)
      .def_property_readonly("outputs", &PyEquation::outputs)
      .def_property_readonly("uses_kernel", &PyEquation::uses_kernel)
      .def_property_readonly("device_status", &PyEquation::device_status)
      .def_property_readonly("hessian_colors", &PyEquation::hessian_colors)
      .def("wait_for_kernel", &PyEquation::wait_for_kernel, pyb::kw_only(),
           pyb::arg("want") = PyEquation::Want::Jacobian)
      .def_property("_options", &PyEquation::options, &PyEquation::set_options);
}
