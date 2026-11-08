#pragma once

#include "error.hpp"
#include "rt/arena.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"
#include "rt/tensor.hpp" // pack
#include "util/ranges.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

// The Python expression handle: an rt::Expression plus shared ownership of its
// arena, since a Python closure can outlive any lexical arena scope.
//
// It holds an rt::Expression instead of deriving from it, because the base's
// hidden operator+ would cause constraint recursion inside pybind11's descr.
namespace ddx::py {

class PyExpression {
public:
  using Base = rt::Expression<double>;
  using SharedArena = std::shared_ptr<rt::Arena<double>>;

  constexpr PyExpression() = default;

  // Implicit, so that `2.0 * x` and `x * 2.0` both work through pybind11's
  // implicitly_convertible.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr PyExpression(double v) : expression_(v) {}
  PyExpression(Base e, SharedArena a) noexcept
      : expression_(e), arena_(std::move(a)) {}

  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return expression_.poisoned();
  }

  // Raises immediately if the expression is poisoned, instead of carrying the
  // poison on to equation().
  [[nodiscard]] static PyExpression checked(const Base &e, SharedArena arena) {
    if (const auto why = rt::detail::Router::why(e)) {
      fail_with(*why);
    }
    return {e, std::move(arena)};
  }
  // Packs scalars into a vector in their own arena, else in `fallback`, the
  // model running now.
  [[nodiscard]] static PyExpression pack(std::span<const PyExpression> scalars,
                                         const SharedArena &fallback) {
    const auto held = std::ranges::find_if(
        scalars, [](const PyExpression &e) { return e.arena_ != nullptr; });
    SharedArena arena = held != scalars.end() ? held->arena_ : fallback;
    if (!arena) {
      fail_with(errc::outside_equation);
    }
    const auto bases = scalars | std::views::transform(&PyExpression::base) |
                       impl::to<std::vector<Base>>();
    return checked(rt::pack(*arena, std::span<const Base>{bases}), arena);
  }

  // Accumulating operators, so a loop of `acc += term` builds one node. The
  // bindings call them on a copy, so other references to the handle never move.
  PyExpression &operator+=(const PyExpression &o) {
    return accumulate(o, [](Base &l, const Base &r) { l += r; });
  }
  PyExpression &operator*=(const PyExpression &o) {
    return accumulate(o, [](Base &l, const Base &r) { l *= r; });
  }
  PyExpression &operator/=(const PyExpression &o) {
    return accumulate(o, [](Base &l, const Base &r) { l /= r; });
  }

  // The number of elements per point, which is zero for a scalar.
  [[nodiscard]] std::size_t size() const noexcept { return expression_.size(); }
  [[nodiscard]] PyExpression at(std::size_t i) const {
    return checked(expression_[i], arena_);
  }
  // () for a scalar, (n,) for a vector, and (rows, cols) for a matrix.
  [[nodiscard]] std::vector<std::size_t> shape() const {
    if (!arena_ || expression_.pending()) {
      return {};
    }
    const rt::Node<double> &n = (*arena_)[expression_.id()];
    if (n.op == rt::OpCode::MatConst) {
      const auto &t =
          arena_->tensors()[rt::detail::get<rt::TensorRef>(n).value];
      return {t.rows, t.cols};
    }
    const rt::Shape s = expression_.shape();
    if (s.cols > 1) {
      return {s.rows, s.cols};
    }
    return size() == 0 ? std::vector<std::size_t>{}
                       : std::vector<std::size_t>{size()};
  }

  // The node id in `arena`, refusing a handle from another arena. A pending
  // literal is materialised into `arena`.
  [[nodiscard]] rt::NodeId root_in(rt::Arena<double> &arena) const {
    if (arena_ && arena_.get() != &arena) {
      fail_with(errc::no_graph);
    }
    return expression_.id(arena);
  }

  [[nodiscard]] std::string repr() const {
    if (!arena_) {
      return std::format("Expression({})", expression_.literal());
    }
    const rt::NodeId v = expression_.id(*arena_);
    const rt::Node<double> &n = (*arena_)[v];
    switch (n.op) {
    case rt::OpCode::Const:
      return std::format("Expression({})", n.value);
    case rt::OpCode::Var:
      return std::format("Expression({})",
                         arena_->symbols()[rt::detail::get<rt::Slot>(n).value]);
    default:
      // Shows the op and node id rather than the whole subtree.
      return std::format("Expression({} #{})", rt::label_of(n.op), v);
    }
  }

private:
  // The only functions that build a new handle with an arena. They are keyed by
  // opcode, so the module registers every operator without a friend each.
  friend PyExpression unary(rt::OpCode op, const PyExpression &u) {
    return {Base::form(op, u.expression_), u.arena_};
  }
  friend PyExpression binary(rt::OpCode op, const PyExpression &l,
                             const PyExpression &r) {
    return {Base::form(op, l.expression_, r.expression_),
            l.arena_ ? l.arena_ : r.arena_};
  }
  // Applies the base's own composition of two handles, keeping the arena.
  friend PyExpression combine(const PyExpression &l, const PyExpression &r,
                              auto &&fn) {
    return {fn(l.expression_, r.expression_), l.arena_ ? l.arena_ : r.arena_};
  }
  // A composition that may refuse, such as a reduction or a vector factory in
  // module.cpp, raises where it is formed.
  friend PyExpression reduce(const PyExpression &v, auto &&fn) {
    return checked(fn(v.expression_), v.arena_);
  }
  friend PyExpression reduce(const PyExpression &u, const PyExpression &v,
                             auto &&fn) {
    return checked(fn(u.expression_, v.expression_),
                   u.arena_ ? u.arena_ : v.arena_);
  }

public:
  [[nodiscard]] constexpr const Base &base() const noexcept {
    return expression_;
  }

  // Null only while this is a pending literal, which has no graph yet.
  [[nodiscard]] constexpr const SharedArena &arena() const noexcept {
    return arena_;
  }

private:
  PyExpression &accumulate(const PyExpression &o, auto &&fn) {
    fn(expression_, o.expression_);
    if (!arena_) {
      arena_ = o.arena_;
    }
    return *this;
  }

  Base expression_;
  SharedArena arena_;
};

// The result of transpose(M), kept as the matrix it reads.
struct PyTransposed {
  PyExpression m;
};

// The arena is taken from whichever operand has one.
[[nodiscard]] inline PyExpression
select(const PyExpression &c, const PyExpression &t, const PyExpression &f) {
  const auto arena = c.arena() ? c.arena() : t.arena() ? t.arena() : f.arena();
  return {select(c.base(), t.base(), f.base()), arena};
}

} // namespace ddx::py
