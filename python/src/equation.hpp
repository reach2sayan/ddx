#pragma once

#include "columns.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "rt/pipeline.hpp"
#include "util/ranges.hpp"

// Included even without the JIT, so the Python surface is the same either way.
#include "jit/kernel.hpp"
#include "rt/equation.hpp"

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <format>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ddx::py {

class PyCall;

// Expands a compressed Jacobian or Hessian block into the dense array the
// caller reads, zero-filling the cells the layout leaves out.
inline void scatter(const auto &layout, Columns &cells, Dense &dense,
                    std::size_t n, std::size_t rows, std::size_t points) {
  std::ranges::fill_n(dense.at(0),
                      static_cast<std::ptrdiff_t>(rows * n * points), 0.0);
  rt::scatter(
      layout, [&cells](std::size_t k) { return cells.at(k); },
      [&dense](std::size_t c) { return dense.at(c); }, n, points);
}

// Each sweep releases the GIL while it runs. Pipelines and the arena lock are
// only touched while holding it.
using PyCore = rt::detail::Core<double, pyb::gil_scoped_release>;

// rt::detail::Core with a Python interface: points are dicts or arrays, and
// refusals are raised as exceptions.
class PyEquation : public PyCore {
public:
  using Want = rt::Want;

  // The arena is copied, so the expressions that built it stay usable.
  PyEquation(const std::shared_ptr<rt::Arena<double>> &arena,
             std::vector<rt::NodeId> roots)
      : PyCore{std::make_unique<rt::Arena<double>>(*arena), std::move(roots)},
        symbols_{*sound().symbols()}, inputs_{*inputs()} {}

private:
  // Raises a poisoned equation's error before anything reads the core.
  [[nodiscard]] const PyCore &sound() const {
    if (const auto why = status()) {
      throw PyError{*why};
    }
    return *this;
  }

public:
  [[nodiscard]] std::size_t width() const { return symbol_count(); }
  [[nodiscard]] std::size_t height() const { return roots().size(); }

  [[nodiscard]] pyb::list symbols_list() const { return symbols_.listed(); }
  [[nodiscard]] pyb::list params_list() const {
    return pyb::cast(
        std::vector<std::string>(params()->begin(), params()->end()));
  }
  // The inputs a point supplies: the symbols, then the parameters.
  [[nodiscard]] pyb::list inputs_list() const { return inputs_.listed(); }

  // Each call is one pipeline run through run<W>(), returning the blocks that
  // pipeline fills, in order.
  [[nodiscard]] pyb::object evaluate(const pyb::handle &x) {
    return run<Want::Value>(x)[0];
  }
  [[nodiscard]] pyb::tuple jacobian(const pyb::handle &x) {
    return run<Want::Jacobian>(x);
  }
  // The Jacobian alone, from a pipeline without a value block.
  [[nodiscard]] pyb::object gradient(const pyb::handle &x) {
    return run<Want::Gradient>(x)[0];
  }
  // A single function uses the coloured pipeline; a system walks the arena one
  // point at a time.
  [[nodiscard]] pyb::tuple hessian(const pyb::handle &x) {
    unwrap(orders(Want::Hessian));
    return height() == 1 ? run<Want::Hessian>(x)
                         : hessian_from_arena(Point{x, inputs_});
  }
  [[nodiscard]] pyb::tuple jvp(const pyb::handle &v, const pyb::handle &x) {
    return run<Want::Jvp>(x, &v);
  }
  [[nodiscard]] pyb::tuple vjp(const pyb::handle &w, const pyb::handle &x) {
    return run<Want::Vjp>(x, &w);
  }
  // Returns (value, gradient, H v), all from the same sweep.
  [[nodiscard]] pyb::tuple hvp(const pyb::handle &v, const pyb::handle &x) {
    unwrap(ready(Want::Hvp));
    return run<Want::Hvp>(x, &v);
  }
  // Returns (value, gradient, H) with H dense, from one O(n^2) sweep instead of
  // the colouring hessian() uses.
  [[nodiscard]] pyb::tuple dense_hessian(const pyb::handle &x) {
    unwrap(ready(Want::DenseHessian));
    return run<Want::DenseHessian>(x);
  }
  // Returns (value, J, dJ/dv), both dense, from one forward sweep.
  [[nodiscard]] pyb::tuple jacobian_jvp(const pyb::handle &v,
                                        const pyb::handle &x) {
    return run<Want::JacobianJvp>(x, &v);
  }

  // Each implicit equation's residual F(z, x) at one point.
  [[nodiscard]] std::vector<double> residual_at(const pyb::handle &x) const {
    const Point at{x, inputs_};
    if (at.batched()) {
      fail_with(errc::wrong_column_count);
    }
    return unwrap(residual(scalars_of(at)));
  }

  // The number of steps one call for `want` evaluates.
  [[nodiscard]] std::size_t nodes(Want want) const {
    return unwrap(census(want)).steps;
  }

  // census() as a dict, with each by_op row as (label, steps, values).
  [[nodiscard]] pyb::dict census_dict(Want want) const {
    const rt::Census c = unwrap(census(want));
    pyb::list rows;
    for (const auto &r : c.by_op) {
      rows.append(pyb::make_tuple(std::string{rt::label_of(r.op)}, r.steps,
                                  r.values));
    }
    pyb::dict out;
    out["steps"] = c.steps;
    out["values"] = c.values;
    out["by_op"] = rows;
    return out;
  }

  // None when calls for `want` run compiled code, else (errc, detail).
  [[nodiscard]] pyb::object compile_status_of(Want want) const {
    const auto s = compile_status(want);
    return s ? pyb::none()
             : pyb::object{pyb::make_tuple(s.error().code, s.error().detail)};
  }

  // Must not compile or freeze a pipeline. Uses Boost's join because
  // views::join_with is broken in libstdc++ 14.2 under clang.
  [[nodiscard]] std::string repr() const {
    return std::format("<ddx.Equation ({}) -> {} output{}>",
                       boost::algorithm::join(symbols_.text(), ", "), height(),
                       height() == 1 ? "" : "s");
  }

private:
  friend class PyCall;
  friend class PyRestricted;

  // --- running -------------------------------------------------------------

  // Storage for one derivative block. Compressed blocks go to scratch and are
  // scattered afterwards; the other shapes are written straight into the array.
  struct Out {
    std::optional<Scratch> cells{};
    std::optional<Block> block{};
    [[nodiscard]] std::span<double *const> rows() {
      return cells   ? cells->rows()
             : block ? block->rows()
                     : std::span<double *const>{};
    }
  };
  [[nodiscard]] Out out_for(rt::BlockShape shape, std::size_t columns,
                            const Point &at) const {
    switch (shape) {
    case rt::BlockShape::None:
      return {};
    case rt::BlockShape::Cells:
    case rt::BlockShape::Colours:
      return {.cells = Scratch{columns, at.size()}};
    case rt::BlockShape::Dense:
      return {.block = Block{shape_of({ssize(height()), ssize(width()),
                                       ssize(width())},
                                      at),
                             columns, at.size()}};
    case rt::BlockShape::Symbols:
      return {.block = Block{symbol_shape(at), columns, at.size()}};
    case rt::BlockShape::Functions:
      return {.block =
                  Block{shape_of({ssize(height())}, at), columns, at.size()}};
    }
    std::unreachable();
  }
  // `pattern` scatters the Jacobian cells (a restricted view supplies its own),
  // and `colouring` scatters the Hessian.
  [[nodiscard]] pyb::object finish(rt::BlockShape shape, Out &o,
                                   const Point &at, const rt::Sparsity &pattern,
                                   const rt::Colouring *colouring) const {
    const std::size_t m = height();
    const std::size_t n = width();
    switch (shape) {
    case rt::BlockShape::None:
      return pyb::none();
    case rt::BlockShape::Symbols:
    case rt::BlockShape::Functions:
    case rt::BlockShape::Dense:
      return std::move(*o.block).array();
    case rt::BlockShape::Cells: {
      Dense g{shape_of({ssize(m), ssize(pattern.columns)}, at), at.size()};
      py::scatter(pattern, *o.cells, g, pattern.columns, pattern.size(),
                  at.size());
      return std::move(g).array();
    }
    case rt::BlockShape::Colours: {
      Dense h{shape_of({ssize(m), ssize(n), ssize(n)}, at), at.size()};
      py::scatter(*colouring, *o.cells, h, n, n, at.size());
      return std::move(h).array();
    }
    }
    std::unreachable();
  }

  // One call of `W` at `x`, seeded along `*v` when the pipeline takes a
  // direction. The point's columns come first, then the direction's.
  template <Want W>
  [[nodiscard]] pyb::tuple run(const pyb::handle &x,
                               const pyb::handle *v = nullptr) {
    const auto l = pipeline(W);
    return run_on<W>(
        [this](auto &&...a) { return dispatch(DDX_FWD(a)...); },
        l->graph->layout(), derivative().pattern, &l->graph->colouring(), x, v);
  }
  // The same over any dispatcher, such as a restricted view's, with its own
  // block sizes, pattern and colouring.
  template <Want W>
  [[nodiscard]] pyb::tuple run_on(auto &&dispatcher, const rt::Layout &sizes,
                                  const rt::Sparsity &pattern,
                                  const rt::Colouring *colouring,
                                  const pyb::handle &x,
                                  const pyb::handle *v = nullptr) const {
    constexpr rt::WantInfo w = rt::want_info(W);
    const Point at{x, inputs_};
    Cell cell;
    // The values array, or nothing when the value is a single float.
    std::optional<Block> f;
    if (w.values && (height() != 1 || at.batched())) {
      f.emplace(shape_of({ssize(height())}, at), sizes.values, at.size());
    }
    Out g = out_for(w.jacobian, sizes.jacobian, at);
    Out h = out_for(w.hessian, sizes.hessian, at);
    const rt::Columns<double> columns{
        .values = w.values ? (f ? f->rows() : cell.rows())
                           : std::span<double *const>{},
        .jacobian = g.rows(),
        .hessian = h.rows()};
    if constexpr (w.direction == rt::Direction::None) {
      unwrap(dispatcher(W, at.columns(), columns, at.size()));
    } else {
      // Built by a lambda because Point is not movable and its positional
      // constructor is private, which rules out optional::emplace.
      const Point dir = [&] {
        if constexpr (w.direction == rt::Direction::Functions) {
          return Point(*v, height());
        } else {
          return Point(*v, symbols_);
        }
      }();
      if (dir.size() != at.size()) {
        fail_with(errc::wrong_direction);
      }
      const auto xs = widen(at.columns(), dir.columns());
      unwrap(dispatcher(W, as_columns(xs), columns, at.size()));
    }
    pyb::list parts;
    if (w.values) {
      parts.append(f ? std::move(*f).array()
                     : pyb::object{pyb::float_(cell.value)});
    }
    if (w.jacobian != rt::BlockShape::None) {
      parts.append(finish(w.jacobian, g, at, pattern, nullptr));
    }
    if (w.hessian != rt::BlockShape::None) {
      parts.append(finish(w.hessian, h, at, pattern, colouring));
    }
    return pyb::tuple(parts);
  }

  // One unbatched point's input values: the symbols, then the parameters.
  [[nodiscard]] static boost::container::small_vector<double, 16>
  scalars_of(const Point &at) {
    return at.columns() |
           std::views::transform([](const double *c) { return *c; }) |
           impl::to<boost::container::small_vector<double, 16>>();
  }

  // A system's Hessian, walked over the arena one point at a time.
  [[nodiscard]] pyb::tuple hessian_from_arena(const Point &at) const {
    if (at.batched() && at.size() != 1) {
      fail_with(errc::wrong_column_count);
    }
    const std::size_t m = height();
    const std::size_t n = width();
    const auto point = scalars_of(at);
    Dense f{shape_of({ssize(m)}, at), at.size()};
    Dense g{shape_of({ssize(m), ssize(n)}, at), at.size()};
    Dense h{shape_of({ssize(m), ssize(n), ssize(n)}, at), at.size()};
    walk(point, {f.at(0), m}, {g.at(0), m * n}, {h.at(0), m * n * n});
    return pyb::make_tuple(std::move(f).array(), std::move(g).array(),
                           std::move(h).array());
  }

  // --- shapes --------------------------------------------------------------

  // Drops the function axis for one output and the batch axis for one point,
  // so a scalar model gives a float, an (n,) gradient and an (n, n) Hessian.
  [[nodiscard]] Shape shape_of(std::initializer_list<pyb::ssize_t> base,
                               const Point &at) const {
    const std::span axes{base};
    // A count rather than a bool, which libstdc++ 14.2's drop() rejects under
    // clang.
    const auto lead =
        static_cast<std::ptrdiff_t>(height() == 1 && !axes.empty());
    Shape dims;
    std::ranges::copy(axes | std::views::drop(lead), std::back_inserter(dims));
    if (at.batched()) {
      dims.push_back(ssize(at.size()));
    }
    return dims;
  }

  // The shape of a block indexed by symbol, which has no function axis.
  [[nodiscard]] Shape symbol_shape(const Point &at) const {
    Shape dims{ssize(width())};
    if (at.batched()) {
      dims.push_back(ssize(at.size()));
    }
    return dims;
  }

  // Stack storage for that single value, with the row pointer the ABI wants.
  struct Cell {
    double value{};
    double *row{&value};
    [[nodiscard]] constexpr std::span<double *const> rows() {
      return {&row, 1};
    }
  };

  // A direction covers the symbols; a point covers every input.
  Symbols symbols_;
  Symbols inputs_;
};

// Equation.restrict(names), which narrows the Jacobian to those symbols'
// columns, with pipelines of its own.
class PyRestricted {
public:
  using Want = rt::Want;

  PyRestricted(const PyEquation &eq, rt::Restricted<PyCore> view)
      : eq_(&eq), view_(std::move(view)) {}

  [[nodiscard]] pyb::list symbols() const {
    return pyb::cast(std::vector<std::string>(view_.symbols().begin(),
                                              view_.symbols().end()));
  }
  [[nodiscard]] std::size_t jacobian_columns() const {
    return view_.jacobian_columns();
  }

  // Returns (value, J) or (value, J, dJ/dv), where J covers only the kept
  // symbols and `v` covers every symbol.
  [[nodiscard]] pyb::tuple jacobian(const pyb::handle &x) const {
    return run<Want::Jacobian>(x);
  }
  [[nodiscard]] pyb::tuple jacobian_jvp(const pyb::handle &v,
                                        const pyb::handle &x) const {
    return run<Want::JacobianJvp>(x, &v);
  }

private:
  template <Want W>
  [[nodiscard]] pyb::tuple run(const pyb::handle &x,
                               const pyb::handle *v = nullptr) const {
    const std::size_t kept = view_.jacobian_columns();
    return eq_->run_on<W>(
        [this](auto &&...a) { return view_.dispatch(DDX_FWD(a)...); },
        {.values = eq_->height(),
         .jacobian = kept,
         .hessian = rt::want_info(W).hessian == rt::BlockShape::None ? 0 : kept},
        view_.jacobian_pattern(), nullptr, x, v);
  }

  const PyEquation *eq_; // kept alive by keep_alive on the factory
  rt::Restricted<PyCore> view_;
};

// A call bound to its own buffers, set up once, so each repeat is a pipeline
// lookup and a sweep with no shape, dtype or stride checks.
class PyCall {
public:
  // The equation outlives the call through pyb::keep_alive on the factory.
  PyCall(PyEquation &eq, PyEquation::Want want, const pyb::handle &x)
      : eq_(&eq), want_(want), x_(pyb::cast<Array>(x)), at_(x_, eq.inputs_) {
    // Refuses wants with no pipeline to bind, such as a system's Hessian.
    unwrap(eq_->ready(want_));
    // Direction columns and dense Hessian blocks are not bound, so wants that
    // need them are refused here.
    if (const auto &w = rt::want_info(want_);
        w.direction != rt::Direction::None ||
        (w.hessian != rt::BlockShape::None &&
         w.hessian != rt::BlockShape::Colours)) {
      fail_with(errc::wrong_column_count);
    }
    const auto l = eq_->pipeline(want_);
    const auto outputs = ssize(eq_->height());
    const auto n = eq_->width();
    // The blocks the pipeline fills: values, Jacobian cells, and the Hessian
    // compressed by colour.
    const auto blocks = l->graph->output_blocks();
    if (!blocks.values.empty()) {
      f_.emplace(eq_->shape_of({outputs}, at_), blocks.values.size(),
                 at_.size());
    }
    // An empty pattern still yields a block, of zeros.
    if (!blocks.jacobian.empty() ||
        rt::want_info(want_).jacobian == rt::BlockShape::Cells) {
      partials_.emplace(blocks.jacobian.size(), at_.size());
      g_.emplace(eq_->shape_of({outputs, ssize(n)}, at_), at_.size());
    }
    if (want_ == PyEquation::Want::Hessian) {
      compressed_.emplace(blocks.hessian.size(), at_.size());
      h_.emplace(eq_->shape_of({outputs, ssize(n), ssize(n)}, at_), at_.size());
    }
  }

  // The pipeline is looked up on each call rather than cached, so a kernel
  // swapped in by adopt() is picked up.
  void operator()() {
    const auto l = eq_->pipeline(want_);
    unwrap(eq_->dispatch(
        want_, at_.columns(),
        {.values = f_ ? f_->rows() : std::span<double *const>{},
         .jacobian = partials_ ? partials_->rows() : std::span<double *const>{},
         .hessian =
             compressed_ ? compressed_->rows() : std::span<double *const>{}},
        at_.size()));
    if (g_) {
      const rt::Sparsity &pattern = eq_->derivative().pattern;
      py::scatter(pattern, *partials_, *g_, pattern.columns, pattern.size(),
                  at_.size());
    }
    if (h_) {
      py::scatter(l->graph->colouring(), *compressed_, *h_, eq_->width(),
                  eq_->width(), at_.size());
    }
  }

  [[nodiscard]] pyb::object x() const { return x_; }
  [[nodiscard]] pyb::object value() const { return read(f_); }
  [[nodiscard]] pyb::object jacobian() const { return read(g_); }
  [[nodiscard]] pyb::object hessian() const { return read(h_); }

  [[nodiscard]] std::string repr() const {
    return std::format(
        "<ddx.Call {} at {} point{}>",
        boost::algorithm::to_lower_copy(std::string{rt::name_of(want_)}),
        at_.size(), at_.size() == 1 ? "" : "s");
  }

private:
  // A rank-0 block is returned as a float, matching the allocating calls.
  template <typename B>
  [[nodiscard]] static pyb::object read(const std::optional<B> &b) {
    if (!b) {
      fail_with(errc::wrong_column_count);
    }
    const Array &a = b->bound();
    return a.ndim() == 0 ? pyb::object{pyb::float_(*a.data())} : pyb::object{a};
  }

  PyEquation *eq_; // kept alive by keep_alive on the factory
  PyEquation::Want want_;
  Array x_; // declared before at_, which points into it
  Point at_;
  std::optional<Block> f_;
  std::optional<Scratch> partials_; // Jacobian cells, scattered into g_
  std::optional<Dense> g_;
  std::optional<Scratch> compressed_; // coloured Hessian, scattered into h_
  std::optional<Dense> h_;
};

} // namespace ddx::py
