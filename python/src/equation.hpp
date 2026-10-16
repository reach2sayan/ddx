#pragma once

#include "columns.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/archive/archive.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/dot.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "rt/lanes.hpp"
#include "util/ranges.hpp"

// Unconditional: jit::Options and jit::Kernel are header types, so the Python
// surface has one shape whether or not the backend was compiled in.
#include "jit/kernel.hpp"
#include "rt/equation.hpp"

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <future>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ddx::py {

class PyCall;

// A compressed block -- the Jacobian's cells or the Hessian's owned cells --
// back into the dense array a caller reads, `points` wide: one fill for the
// zeros the layout left out, then rt::scatter over its entries.
inline void scatter(const auto &layout, Columns &cells, Dense &dense,
                    std::size_t n, std::size_t rows, std::size_t points) {
  std::ranges::fill_n(dense.at(0),
                      static_cast<std::ptrdiff_t>(rows * n * points), 0.0);
  rt::scatter(
      layout, [&cells](std::size_t k) { return cells.at(k); },
      [&dense](std::size_t c) { return dense.at(c); }, n, points);
}

// Every sweep gives up the GIL while it runs, and only the sweep: the lanes
// and the arena's lock are touched holding it.
using PyCore = rt::detail::Core<double, pyb::gil_scoped_release>;

// rt::detail::Core, spelled in arrays: the output count a number, a point a
// dict or an array, and a refusal an exception.
class PyEquation : public PyCore {
public:
  using Want = rt::Want;

  // The arena is copied: the expressions that built it stay usable.
  PyEquation(const std::shared_ptr<rt::Builder<double>> &arena,
             std::vector<rt::NodeId> roots)
      : PyCore{std::make_unique<rt::Builder<double>>(*arena), std::move(roots)},
        symbols_{*sound().symbols()}, inputs_{*inputs()} {}

  // From a file: the sweeps it carries arrive built.  Binding an extern
  // function by name has no Python spelling, so a file naming one is refused.
  explicit PyEquation(rt::Verified<double> snap)
      : PyCore{std::move(snap).rebuild()},
        symbols_{*sound().symbols()}, inputs_{*inputs()} {}

  // The same struct the C++ equations fill: one serialiser, and what the cache
  // path in module.cpp writes and keys on.
  using PyCore::describes;
  using PyCore::snapshot;

private:
  // What C++ poisons is raised here, before anything reads the core.
  [[nodiscard]] const PyCore &sound() const {
    if (const auto why = status()) {
      throw PyError{*why};
    }
    return *this;
  }

public:

  [[nodiscard]] std::size_t width() const { return symbol_count(); }
  [[nodiscard]] std::size_t height() const { return roots().size(); }

  // The Python spelling: a list over the keys Symbols already built, so an
  // access costs increfs and never a PyUnicode.
  [[nodiscard]] pyb::list symbols_list() const { return symbols_.listed(); }
  [[nodiscard]] pyb::list params_list() const {
    return pyb::cast(std::vector<std::string>(params()->begin(),
                                              params()->end()));
  }
  // What a point supplies: the symbols, then the parameters.
  [[nodiscard]] pyb::list inputs_list() const { return inputs_.listed(); }

  [[nodiscard]] pyb::object evaluate(const pyb::handle &x) {
    const Point at{x, inputs_};
    const auto l = lane(Want::Value);
    if (scalar(at)) {
      Cell f;
      run(Want::Value, at, {.values = f.rows()});
      return pyb::float_(f.value);
    }
    Block f{shape_of({ssize(height())}, at),
            count(*l, &rt::Layout::values), at.size()};
    run(Want::Value, at, {.values = f.rows()});
    return std::move(f).array();
  }

  [[nodiscard]] pyb::tuple jacobian(const pyb::handle &x) {
    const Point at{x, inputs_};
    const auto l = lane(Want::Jacobian);
    Scratch cells{count(*l, &rt::Layout::jacobian), at.size()};
    Dense g{shape_of({ssize(height()), ssize(width())}, at), at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run(Want::Jacobian, at,
        {.values = f ? f->rows() : cell.rows(), .jacobian = cells.rows()});
    scatter_jacobian(cells, g, at.size());
    return pyb::make_tuple(value_of(f, cell), std::move(g).array());
  }

  // The Jacobian alone, from a lane with no value block.
  [[nodiscard]] pyb::object gradient(const pyb::handle &x) {
    const Point at{x, inputs_};
    const auto l = lane(Want::Gradient);
    Scratch cells{count(*l, &rt::Layout::jacobian), at.size()};
    Dense g{shape_of({ssize(height()), ssize(width())}, at), at.size()};
    run(Want::Gradient, at, {.jacobian = cells.rows()});
    scatter_jacobian(cells, g, at.size());
    return std::move(g).array();
  }

  [[nodiscard]] pyb::tuple hessian(const pyb::handle &x) {
    const Point at{x, inputs_};
    unwrap(orders(Want::Hessian));
    return height() == 1 ? hessian_from_lane(at) : hessian_from_arena(at);
  }

  [[nodiscard]] pyb::tuple jvp(const pyb::handle &v, const pyb::handle &x) {
    const Point at{x, inputs_};
    const Point dir{v, symbols_};
    const auto l = lane(Want::Jvp);
    Block out{shape_of({ssize(height())}, at),
              count(*l, &rt::Layout::jacobian), at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run_seeded(Want::Jvp, at, dir,
               {.values = f ? f->rows() : cell.rows(), .jacobian = out.rows()});
    return pyb::make_tuple(value_of(f, cell), std::move(out).array());
  }

  [[nodiscard]] pyb::tuple vjp(const pyb::handle &w, const pyb::handle &x) {
    const Point at{x, inputs_};
    // By position: the covector is indexed by function, and a dict would have
    // nothing to key on.
    const Point dir{w, height()};
    const auto l = lane(Want::Vjp);
    Block out{symbol_shape(at), count(*l, &rt::Layout::jacobian),
              at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run_seeded(Want::Vjp, at, dir,
               {.values = f ? f->rows() : cell.rows(), .jacobian = out.rows()});
    return pyb::make_tuple(value_of(f, cell), std::move(out).array());
  }

  // (value, gradient, H v): the gradient comes off the same sweep.  One
  // output only, as hessian() is.
  [[nodiscard]] pyb::tuple hvp(const pyb::handle &v, const pyb::handle &x) {
    unwrap(ready(Want::Hvp));
    const Point at{x, inputs_};
    const Point dir{v, symbols_};
    const auto l = lane(Want::Hvp);
    const std::size_t n = width();

    Scratch partials{count(*l, &rt::Layout::jacobian), at.size()};
    Dense g{shape_of({ssize(height()), ssize(n)}, at), at.size()};
    Block out{symbol_shape(at), count(*l, &rt::Layout::hessian), at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run_seeded(Want::Hvp, at, dir,
               {.values = f ? f->rows() : cell.rows(),
                .jacobian = partials.rows(),
                .hessian = out.rows()});
    scatter_jacobian(partials, g, at.size());
    return pyb::make_tuple(value_of(f, cell), std::move(g).array(),
                           std::move(out).array());
  }

  // (value, gradient, H): the dense n x n from one O(n^2) sweep over matrix
  // rows, where hessian() colours.  One output only; the lane's block is
  // column-major and the array row-major.
  [[nodiscard]] pyb::tuple dense_hessian(const pyb::handle &x) {
    unwrap(ready(Want::DenseHessian));
    const Point at{x, inputs_};
    const auto l = lane(Want::DenseHessian);
    const std::size_t n = width();
    Dense g{shape_of({ssize(height()), ssize(n)}, at), at.size()};
    Scratch partials{count(*l, &rt::Layout::jacobian), at.size()};
    Scratch block{n * n, at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run(Want::DenseHessian, at,
        {.values = f ? f->rows() : cell.rows(),
         .jacobian = partials.rows(),
         .hessian = block.rows()});
    scatter_jacobian(partials, g, at.size());
    Dense dense{shape_of({ssize(height()), ssize(n), ssize(n)}, at),
                at.size()};
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        std::ranges::copy_n(block.rows()[i + j * n],
                            static_cast<std::ptrdiff_t>(at.size()),
                            dense.at(i * n + j));
      }
    }
    return pyb::make_tuple(value_of(f, cell), std::move(g).array(),
                           std::move(dense).array());
  }

  // F(z, x) for each implicit's residual at one point.
  [[nodiscard]] std::vector<double> residual_at(const pyb::handle &x) const {
    const Point at{x, inputs_};
    if (at.batched()) {
      fail_with(errc::wrong_column_count);
    }
    std::vector<double> point(at.columns().size());
    std::ranges::transform(at.columns(), point.begin(),
                           [](const double *c) { return *c; });
    return unwrap(residual(point));
  }

  // (value, J, dJ/dv): the Jacobian and its derivative along `v`, both dense,
  // from one forward sweep.
  [[nodiscard]] pyb::tuple jacobian_jvp(const pyb::handle &v,
                                        const pyb::handle &x) const {
    const Point at{x, inputs_};
    const Point dir{v, symbols_};
    const auto l = lane(Want::JacobianTangent);
    const auto shape = shape_of({ssize(height()), ssize(width())}, at);
    Scratch cells{count(*l, &rt::Layout::jacobian), at.size()};
    Scratch moved{count(*l, &rt::Layout::hessian), at.size()};
    Dense g{shape, at.size()};
    Dense dg{shape, at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run_seeded(Want::JacobianTangent, at, dir,
               {.values = f ? f->rows() : cell.rows(),
                .jacobian = cells.rows(),
                .hessian = moved.rows()});
    scatter_jacobian(cells, g, at.size());
    scatter_jacobian(moved, dg, at.size());
    return pyb::make_tuple(value_of(f, cell), std::move(g).array(),
                           std::move(dg).array());
  }

  // Under Backend.DEVICE the device answering, None under any other backend.
  // Raises with the reason where no device answers or its compiler refused.
  [[nodiscard]] std::optional<std::string> device() const {
    return device_status().transform([](const auto &status) {
      if (!status) {
        throw PyError{error{status.error().code}, status.error().detail};
      }
      return std::string{*status};
    });
  }

  // What one call for `want` evaluates: the contracted live order, which is
  // the sweep and what the kernel is lowered from.
  [[nodiscard]] std::size_t nodes(Want want) const {
    return lane(want)->graph->schedule().size();
  }

  // Only what is free to answer: uses_kernel() would freeze a lane, and a repr
  // that compiles something is one nobody can call in a debugger.  Boost's
  // join, not views::join_with: libstdc++ 14.2 gets that one wrong under
  // clang, and 14.2 is what Ubuntu 24.04 ships.
  [[nodiscard]] std::string repr() const {
    return std::format("<ddx.Equation ({}) -> {} output{}>",
                       boost::algorithm::join(symbols_.text(), ", "), height(),
                       height() == 1 ? "" : "s");
  }

  [[nodiscard]] std::string to_dot(bool all) const {
    return rt::Dot<double>{*lane(Want::Jacobian)->graph,
                           all ? rt::Scope::All : rt::Scope::Live}
        .str();
  }

private:
  friend class PyCall;
  friend class PyRestricted;

  using BlockMember = std::size_t rt::Layout::*;
  [[nodiscard]] static std::size_t count(const Compiled &c,
                                         BlockMember which) {
    return c.graph->layout().*which;
  }

  // --- running -------------------------------------------------------------

  void run(Want want, const Point &at, const rt::Columns<double> &out) const {
    unwrap(dispatch(want, at.columns(), out, at.size()));
  }

  // The point's columns and the direction's, in one array: symbols first, then
  // the seeds, which is the order input_column() states.
  void run_seeded(Want want, const Point &at, const Point &dir,
                  const rt::Columns<double> &out) const {
    if (dir.size() != at.size()) {
      fail_with(errc::wrong_direction);
    }
    const auto xs = widen(at.columns(), dir.columns());
    unwrap(dispatch(want, as_columns(xs), out, at.size()));
  }

  // This side always hands back the dense matrix, so the zeros go back in
  // here.
  void scatter_jacobian(Scratch &cells, Dense &dense,
                        std::size_t points) const {
    const rt::Sparsity &pattern = derivative().pattern;
    py::scatter(pattern, cells, dense, pattern.columns, pattern.size(),
                points);
  }

  // --- the Hessian ---------------------------------------------------------

  // One function: the lane holds it compressed by colour, and the colouring
  // says which (colour, row) column owns a given (i, j).
  [[nodiscard]] pyb::tuple hessian_from_lane(const Point &at) const {
    const auto l = lane(Want::Hessian);
    const std::size_t n = width();
    Dense g{shape_of({ssize(height()), ssize(n)}, at), at.size()};
    Scratch partials{count(*l, &rt::Layout::jacobian), at.size()};
    Scratch compressed{count(*l, &rt::Layout::hessian), at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run(Want::Hessian, at,
        {.values = f ? f->rows() : cell.rows(),
         .jacobian = partials.rows(),
         .hessian = compressed.rows()});
    scatter_jacobian(partials, g, at.size());

    Dense dense{shape_of({ssize(height()), ssize(n), ssize(n)}, at),
                at.size()};
    py::scatter(l->graph->coloring(), compressed, dense, n, n, at.size());
    return pyb::make_tuple(value_of(f, cell), std::move(g).array(),
                           std::move(dense).array());
  }

  // A system: Core's walk over the arena, which is a point at a time and says
  // so.
  [[nodiscard]] pyb::tuple hessian_from_arena(const Point &at) const {
    if (at.batched() && at.size() != 1) {
      fail_with(errc::wrong_column_count);
    }
    const std::size_t m = height();
    const std::size_t n = width();
    boost::container::small_vector<double, 16> point(n);
    std::ranges::transform(at.columns(), point.begin(),
                           [](const double *c) { return *c; });
    Dense f{shape_of({ssize(m)}, at), at.size()};
    Dense g{shape_of({ssize(m), ssize(n)}, at), at.size()};
    Dense h{shape_of({ssize(m), ssize(n), ssize(n)}, at), at.size()};
    walk(point, {f.at(0), m}, {g.at(0), m * n}, {h.at(0), m * n * n});
    return pyb::make_tuple(std::move(f).array(), std::move(g).array(),
                           std::move(h).array());
  }

  // --- shapes --------------------------------------------------------------

  // The leading axis goes at one function and the trailing one at one point, so
  // a scalar model at a point answers with a float, an (n,) gradient and an
  // (n, n) Hessian rather than three arrays with 1s in them.  Computed before
  // the array is built, so the array is built at this shape and never reshaped.
  [[nodiscard]] Shape shape_of(std::initializer_list<pyb::ssize_t> base,
                               const Point &at) const {
    const std::span axes{base};
    // A count and not a flag: drop() takes a difference_type, and libstdc++
    // 14.2 refuses the bool under clang where 14.3 accepts it.
    const auto lead =
        static_cast<std::ptrdiff_t>(height() == 1 && !axes.empty());
    Shape dims;
    std::ranges::copy(axes | std::views::drop(lead), std::back_inserter(dims));
    if (at.batched()) {
      dims.push_back(ssize(at.size()));
    }
    return dims;
  }

  // A block indexed by symbol rather than by function, so shape_of's leading
  // axis -- which it drops at one output -- is not one this has.
  [[nodiscard]] Shape symbol_shape(const Point &at) const {
    Shape dims{ssize(width())};
    if (at.batched()) {
      dims.push_back(ssize(at.size()));
    }
    return dims;
  }

  // Exactly the case where the values block is one double and the answer is a
  // Python float, so it needs no array at all.
  [[nodiscard]] bool scalar(const Point &at) const {
    return height() == 1 && !at.batched();
  }

  // One double on the stack for that case.  The ABI wants a row pointer, and
  // this is the whole of what it writes.
  struct Cell {
    double value{};
    double *row{&value};
    [[nodiscard]] constexpr std::span<double *const> rows() {
      return {&row, 1};
    }
  };

  // A float at one point, an array across a batch: only the first can skip the
  // allocation, and every derivative call serves both.
  [[nodiscard]] std::optional<Block> values_block(const Compiled &c,
                                                  const Point &at) const {
    return scalar(at) ? std::nullopt
                      : std::optional<Block>{
                            std::in_place, shape_of({ssize(height())}, at),
                            count(c, &rt::Layout::values), at.size()};
  }

  [[nodiscard]] static pyb::object value_of(std::optional<Block> &f,
                                            Cell &cell) {
    return f ? std::move(*f).array() : pyb::object{pyb::float_(cell.value)};
  }

  // A direction is over the symbols; a point over every input.
  Symbols symbols_;
  Symbols inputs_;
};

// Equation.restrict(names): the Jacobian narrowed to those symbols' columns,
// in symbol order, with lanes of its own.  Spelled as the equation is, and kept
// alive past it by pyb::keep_alive on the factory.
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

  // (value, J): J is outputs x the kept symbols.
  [[nodiscard]] pyb::tuple jacobian(const pyb::handle &x) const {
    const Point at{x, eq_->inputs_};
    Scratch cells{view_.jacobian_columns(), at.size()};
    Dense g{shape(at), at.size()};
    PyEquation::Cell cell;
    auto f = values(at);
    unwrap(view_.jacobian(at.columns(), f ? f->rows() : cell.rows(),
                          cells.rows(), at.size()));
    scatter_jacobian(cells, g, at.size());
    return pyb::make_tuple(PyEquation::value_of(f, cell),
                           std::move(g).array());
  }

  // (value, J, dJ/dv) over the kept columns; `v` is over every symbol.
  [[nodiscard]] pyb::tuple jacobian_jvp(const pyb::handle &v,
                                        const pyb::handle &x) const {
    const Point at{x, eq_->inputs_};
    const Point dir{v, eq_->symbols_};
    if (dir.size() != at.size()) {
      fail_with(errc::wrong_direction);
    }
    Scratch cells{view_.jacobian_columns(), at.size()};
    Scratch moved{view_.jacobian_columns(), at.size()};
    Dense g{shape(at), at.size()};
    Dense dg{shape(at), at.size()};
    PyEquation::Cell cell;
    auto f = values(at);
    unwrap(view_.jacobian_jvp(at.columns(), dir.columns(),
                              f ? f->rows() : cell.rows(), cells.rows(),
                              moved.rows(), at.size()));
    scatter_jacobian(cells, g, at.size());
    scatter_jacobian(moved, dg, at.size());
    return pyb::make_tuple(PyEquation::value_of(f, cell), std::move(g).array(),
                           std::move(dg).array());
  }

private:
  // PyEquation::scatter_jacobian, over this view's own pattern.  It cannot be
  // shared with the equation's: a restricted view keeps a subset of the
  // columns, so its Sparsity is a different one and is the only thing that
  // describes the block `view_.jacobian` just filled.
  void scatter_jacobian(Scratch &cells, Dense &dense,
                        std::size_t points) const {
    const rt::Sparsity &pattern = view_.jacobian_pattern();
    py::scatter(pattern, cells, dense, pattern.columns, pattern.size(),
                points);
  }

  [[nodiscard]] Shape shape(const Point &at) const {
    return eq_->shape_of(
        {ssize(eq_->height()), ssize(view_.symbols().size())}, at);
  }
  [[nodiscard]] std::optional<Block> values(const Point &at) const {
    return eq_->scalar(at) ? std::nullopt
                           : std::optional<Block>{
                                 std::in_place,
                                 eq_->shape_of({ssize(eq_->height())}, at),
                                 eq_->height(), at.size()};
  }

  const PyEquation *eq_; // outlived by the equation: keep_alive on the factory
  rt::Restricted<PyCore> view_;
};

// A call bound to its buffers: the point's columns, the output arrays and the
// row pointers are taken once, so a repeat is a lane lookup and the sweep.
// Owning them rather than binding the caller's is what leaves no shape, dtype
// or stride to check.
class PyCall {
public:
  // The equation outlives the call through pyb::keep_alive on the factory,
  // never through a member: the binding states the lifetime once.
  PyCall(PyEquation &eq, PyEquation::Want want, const pyb::handle &x)
      : eq_(&eq), want_(want), x_(pyb::cast<Array>(x)), at_(x_, eq.inputs_) {
    // A system's Hessian is read off the arena a point at a time, so there is
    // no lane to bind; the lane's own gate says so.
    unwrap(eq_->ready(want_));
    const auto l = eq_->lane(want_);
    const auto outputs = ssize(eq_->height());
    const auto n = eq_->width();
    // Each block the lane carries, as the graph says: the values, the
    // Jacobian's cells, and -- compressed by colour -- the Hessian's.
    const auto blocks = l->graph->output_blocks();
    if (!blocks.values.empty()) {
      f_.emplace(eq_->shape_of({outputs}, at_), blocks.values.size(),
                 at_.size());
    }
    if (!blocks.jacobian.empty()) {
      partials_.emplace(blocks.jacobian.size(), at_.size());
      g_.emplace(eq_->shape_of({outputs, ssize(n)}, at_), at_.size());
    }
    if (want_ == PyEquation::Want::Hessian) {
      compressed_.emplace(blocks.hessian.size(), at_.size());
      h_.emplace(eq_->shape_of({outputs, ssize(n), ssize(n)}, at_), at_.size());
    }
  }

  // The lane is looked up per call, never cached: adopt() swaps a kernel in,
  // and a call holding the old one interprets on with every answer still
  // right.
  void operator()() {
    const auto l = eq_->lane(want_);
    eq_->run(
        want_, at_,
        {.values = f_ ? f_->rows() : std::span<double *const>{},
         .jacobian = partials_ ? partials_->rows() : std::span<double *const>{},
         .hessian =
             compressed_ ? compressed_->rows() : std::span<double *const>{}});
    if (g_) {
      eq_->scatter_jacobian(*partials_, *g_, at_.size());
    }
    if (h_) {
      scatter(*l);
    }
  }

  [[nodiscard]] pyb::object x() const { return x_; }
  [[nodiscard]] pyb::object value() const { return read(f_); }
  [[nodiscard]] pyb::object jacobian() const { return read(g_); }
  [[nodiscard]] pyb::object hessian() const { return read(h_); }

  [[nodiscard]] std::string repr() const {
    return std::format("<ddx.Call {} at {} point{}>",
                       boost::algorithm::to_lower_copy(
                           std::string{rt::name_of(want_)}),
                       at_.size(), at_.size() == 1 ? "" : "s");
  }

private:
  void scatter(const PyEquation::Compiled &l) {
    py::scatter(l.graph->coloring(), *compressed_, *h_, eq_->width(),
                eq_->width(), at_.size());
  }

  // A rank-0 block is the scalar case, where the allocating calls answer with
  // a float; matching them costs a PyFloat only when the value is read.
  template <typename B>
  [[nodiscard]] static pyb::object read(const std::optional<B> &b) {
    if (!b) {
      fail_with(errc::wrong_column_count);
    }
    const Array &a = b->bound();
    return a.ndim() == 0 ? pyb::object{pyb::float_(*a.data())} : pyb::object{a};
  }

  PyEquation *eq_; // outlived by the equation: keep_alive on the factory
  PyEquation::Want want_;
  Array x_; // declared before at_, which points into it
  Point at_;
  std::optional<Block> f_;
  std::optional<Scratch> partials_; // the pattern's cells, on their way to g_
  std::optional<Dense> g_;
  std::optional<Scratch> compressed_; // the colouring's, on their way to h_
  std::optional<Dense> h_;
};

} // namespace ddx::py
