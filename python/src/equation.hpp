#pragma once

#include "columns.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
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

  // Each call is one lane run through run<W>(); the tuple's members are the
  // blocks the table says the lane fills, in order.
  [[nodiscard]] pyb::object evaluate(const pyb::handle &x) {
    return run<Want::Value>(x)[0];
  }
  [[nodiscard]] pyb::tuple jacobian(const pyb::handle &x) {
    return run<Want::Jacobian>(x);
  }
  // The Jacobian alone, from a lane with no value block.
  [[nodiscard]] pyb::object gradient(const pyb::handle &x) {
    return run<Want::Gradient>(x)[0];
  }
  // One function: the lane holds it compressed by colour.  A system: Core's
  // walk over the arena, which is a point at a time and says so.
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
  // (value, gradient, H v): the gradient comes off the same sweep.
  [[nodiscard]] pyb::tuple hvp(const pyb::handle &v, const pyb::handle &x) {
    unwrap(ready(Want::Hvp));
    return run<Want::Hvp>(x, &v);
  }
  // (value, gradient, H): the dense n x n from one O(n^2) sweep over matrix
  // rows, where hessian() colours.
  [[nodiscard]] pyb::tuple dense_hessian(const pyb::handle &x) {
    unwrap(ready(Want::DenseHessian));
    return run<Want::DenseHessian>(x);
  }
  // (value, J, dJ/dv): the Jacobian and its derivative along `v`, both dense,
  // from one forward sweep.
  [[nodiscard]] pyb::tuple jacobian_jvp(const pyb::handle &v,
                                        const pyb::handle &x) {
    return run<Want::JacobianTangent>(x, &v);
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

  // The storage one derivative block is swept into, and the array it becomes.
  // The pattern's cells, a coloured block and a dense n x n are scattered or
  // transposed afterwards; a block per symbol or per function is laid out as
  // the array already.
  struct Out {
    std::optional<Scratch> cells{};
    std::optional<Block> block{};
    [[nodiscard]] std::span<double *const> rows() {
      return cells ? cells->rows() : block ? block->rows()
                                           : std::span<double *const>{};
    }
  };
  [[nodiscard]] Out out_for(rt::Block shape, std::size_t columns,
                            const Point &at) const {
    switch (shape) {
    case rt::Block::None:
      return {};
    case rt::Block::Cells:
    case rt::Block::Colours:
    case rt::Block::Dense:
      return {.cells = Scratch{columns, at.size()}};
    case rt::Block::Symbols:
      return {.block = Block{symbol_shape(at), columns, at.size()}};
    case rt::Block::Functions:
      return {.block = Block{shape_of({ssize(height())}, at), columns,
                             at.size()}};
    }
    std::unreachable();
  }
  // `pattern` is what scatters the cells -- a restrict() view's own, since it
  // keeps a subset of the columns -- and `coloring` the Hessian lane's.
  [[nodiscard]] pyb::object finish(rt::Block shape, Out &o, const Point &at,
                                   const rt::Sparsity &pattern,
                                   const rt::Coloring *coloring) const {
    const std::size_t m = height();
    const std::size_t n = width();
    switch (shape) {
    case rt::Block::None:
      return pyb::none();
    case rt::Block::Symbols:
    case rt::Block::Functions:
      return std::move(*o.block).array();
    case rt::Block::Cells: {
      Dense g{shape_of({ssize(m), ssize(pattern.columns)}, at), at.size()};
      py::scatter(pattern, *o.cells, g, pattern.columns, pattern.size(),
                  at.size());
      return std::move(g).array();
    }
    case rt::Block::Colours: {
      Dense h{shape_of({ssize(m), ssize(n), ssize(n)}, at), at.size()};
      py::scatter(*coloring, *o.cells, h, n, n, at.size());
      return std::move(h).array();
    }
    case rt::Block::Dense: {
      // The lane's block is column-major and the array row-major.
      Dense h{shape_of({ssize(m), ssize(n), ssize(n)}, at), at.size()};
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          std::ranges::copy_n(o.cells->rows()[i + j * n],
                              static_cast<std::ptrdiff_t>(at.size()),
                              h.at(i * n + j));
        }
      }
      return std::move(h).array();
    }
    }
    std::unreachable();
  }

  // One call of `W` at `x`, seeded along `*v` where the lane takes a
  // direction -- one value per symbol, or per function for a covector, by
  // position: functions have no names for a dict to key on.  The point's
  // columns and the direction's go in one array, symbols first, which is the
  // order input_column() states.
  template <Want W>
  [[nodiscard]] pyb::tuple run(const pyb::handle &x,
                               const pyb::handle *v = nullptr) {
    constexpr rt::WantInfo w = rt::want_info(W);
    const Point at{x, inputs_};
    const auto l = lane(W);
    Cell cell;
    auto f = w.values ? values_block(*l, at) : std::nullopt;
    Out g = out_for(w.jacobian, count(*l, &rt::Layout::jacobian), at);
    Out h = out_for(w.hessian, count(*l, &rt::Layout::hessian), at);
    const rt::Columns<double> columns{
        .values = w.values ? (f ? f->rows() : cell.rows())
                           : std::span<double *const>{},
        .jacobian = g.rows(),
        .hessian = h.rows()};
    if constexpr (w.seeds == rt::Seeds::None) {
      unwrap(dispatch(W, at.columns(), columns, at.size()));
    } else {
      // Built in place: Point is pinned, and its positional constructor is
      // ours alone -- optional::emplace would construct it from outside.
      const Point dir = [&] {
        if constexpr (w.seeds == rt::Seeds::Functions) {
          return Point(*v, height());
        } else {
          return Point(*v, symbols_);
        }
      }();
      if (dir.size() != at.size()) {
        fail_with(errc::wrong_direction);
      }
      const auto xs = widen(at.columns(), dir.columns());
      unwrap(dispatch(W, as_columns(xs), columns, at.size()));
    }
    pyb::list parts;
    if (w.values) {
      parts.append(value_of(f, cell));
    }
    if (w.jacobian != rt::Block::None) {
      parts.append(finish(w.jacobian, g, at, derivative().pattern, nullptr));
    }
    if (w.hessian != rt::Block::None) {
      parts.append(
          finish(w.hessian, h, at, derivative().pattern, &l->graph->coloring()));
    }
    return pyb::tuple(parts);
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

  // (value, J): J is outputs x the kept symbols, scattered through the view's
  // own pattern -- it keeps a subset of the columns.
  [[nodiscard]] pyb::tuple jacobian(const pyb::handle &x) const {
    const Point at{x, eq_->inputs_};
    PyEquation::Cell cell;
    auto f = values(at);
    auto g = eq_->out_for(rt::Block::Cells, view_.jacobian_columns(), at);
    unwrap(view_.jacobian(at.columns(), f ? f->rows() : cell.rows(), g.rows(),
                          at.size()));
    return pyb::make_tuple(PyEquation::value_of(f, cell), kept(g, at));
  }

  // (value, J, dJ/dv) over the kept columns; `v` is over every symbol.
  [[nodiscard]] pyb::tuple jacobian_jvp(const pyb::handle &v,
                                        const pyb::handle &x) const {
    const Point at{x, eq_->inputs_};
    const Point dir{v, eq_->symbols_};
    if (dir.size() != at.size()) {
      fail_with(errc::wrong_direction);
    }
    PyEquation::Cell cell;
    auto f = values(at);
    auto g = eq_->out_for(rt::Block::Cells, view_.jacobian_columns(), at);
    auto dg = eq_->out_for(rt::Block::Cells, view_.jacobian_columns(), at);
    unwrap(view_.jacobian_jvp(at.columns(), dir.columns(),
                              f ? f->rows() : cell.rows(), g.rows(),
                              dg.rows(), at.size()));
    return pyb::make_tuple(PyEquation::value_of(f, cell), kept(g, at),
                           kept(dg, at));
  }

private:
  [[nodiscard]] pyb::object kept(PyEquation::Out &o, const Point &at) const {
    return eq_->finish(rt::Block::Cells, o, at, view_.jacobian_pattern(),
                       nullptr);
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
      const rt::Sparsity &pattern = eq_->derivative().pattern;
      py::scatter(pattern, *partials_, *g_, pattern.columns, pattern.size(),
                  at_.size());
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
