#pragma once

#include "error.hpp"
#include "rt/arena.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"
#include "rt/tensor.hpp" // pack
#include "util/ranges.hpp"

#include <cstddef>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

// The handle Python holds: an rt::Expression that owns its arena.
// rt::Expression stores a *non-owning* Arena* because in C++ an arena's
// lifetime is lexical; a Python closure can capture a symbol and outlive
// everything.  Shared ownership is the whole of what this adds.
//
// It holds one rather than deriving from one: pybind11 describes a bound type
// as descr<N, PyExpression>, and a base class is an associated class where a
// member's type is not.  Deriving would put rt::Expression's hidden operator+
// into the overload set for descr + descr, instantiating Numeric<descr> whose
// body is `{ a + b }` -- constraint recursion inside pybind11's machinery.
namespace ddx::py {

class PyExpression {
public:
  using Base = rt::Expression<double>;
  using SharedArena = std::shared_ptr<rt::Arena<double>>;

  constexpr PyExpression() = default;

  // Implicit, and the reason `2.0 * x` and `x * 2.0` both work: pybind11 is
  // told about it with implicitly_convertible, and what follows is the base's
  // own folding -- two pending literals never reach a graph.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr PyExpression(double v) : expression_(v) {}
  PyExpression(Base e, SharedArena a) noexcept
      : expression_(e), arena_(std::move(a)) {}

  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return expression_.poisoned();
  }

  // A factory or reduction raises where it refused -- a shape, a missing
  // arena -- rather than carrying the poison on to equation().
  [[nodiscard]] static PyExpression checked(const Base &e, SharedArena arena) {
    if (const auto why = rt::detail::Rooting::why(e)) {
      fail_with(*why);
    }
    return {e, std::move(arena)};
  }
  // The arena the scalars name, else `fallback`: the model running now.
  [[nodiscard]] static PyExpression pack(std::span<const PyExpression> scalars,
                                         const SharedArena &fallback) {
    SharedArena arena = fallback;
    for (const PyExpression &e : scalars) {
      if (e.arena_) {
        arena = e.arena_;
        break;
      }
    }
    if (!arena) {
      fail_with(errc::outside_equation);
    }
    const auto bases = scalars | std::views::transform(&PyExpression::base) |
                       impl::to<std::vector<Base>>();
    return checked(rt::pack(*arena, std::span<const Base>{bases}), arena);
  }

  // The base's accumulating `+=`, with the arena put back: what a loop of
  // `acc += term` builds as one node.  Python's `+=` rebinds the name to what
  // __iadd__ answers, so this is called on a copy and no other reference to
  // the old handle moves.
  PyExpression &operator+=(const PyExpression &o) {
    expression_ += o.expression_;
    if (!arena_) {
      arena_ = o.arena_;
    }
    return *this;
  }

  // Elements per point: zero for a scalar.
  [[nodiscard]] std::size_t size() const noexcept { return expression_.size(); }
  [[nodiscard]] PyExpression at(std::size_t i) const {
    return checked(expression_[i], arena_);
  }
  // (), (n,) for a vector, (rows, cols) for a matrix, constant or computed.
  [[nodiscard]] std::vector<std::size_t> shape() const {
    if (!arena_) {
      return {};
    }
    const rt::Node<double> &n = (*arena_)[expression_.id()];
    if (n.op == rt::OpCode::MatConst) {
      const auto &t = arena_->tensors()[rt::detail::index_of(n)];
      return {t.rows, t.cols};
    }
    const rt::Shape s = expression_.shape();
    if (s.cols > 1) {
      return {s.rows, s.cols};
    }
    return size() == 0 ? std::vector<std::size_t>{}
                       : std::vector<std::size_t>{size()};
  }

  // The id this names in `arena`, or a refusal: an id from another arena would
  // index this one at random.  A pending literal materialises into it.
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
                         arena_->symbols()[rt::detail::index_of(n)]);
    default:
      // The op and the identity, not the subtree: an id *is* the identity of a
      // subexpression here.
      return std::format("Expression({} #{})", rt::label_of(n.op), v);
    }
  }

private:
  // The arithmetic surface below is the only thing that puts an arena back on a
  // new handle, so the two halves of a handle are reachable there and nowhere
  // else.  Keyed by opcode rather than by name: the module registers every row
  // of the opcode table through these, so there is no friend per function.
  // Whichever operand names a graph, mirroring rt::Expression::form: a pending
  // literal folds against the other rather than materialising into it.
  friend PyExpression unary(rt::OpCode op, const PyExpression &u) {
    return {Base::form(op, u.expression_), u.arena_};
  }
  friend PyExpression binary(rt::OpCode op, const PyExpression &l,
                             const PyExpression &r) {
    return {Base::form(op, l.expression_, r.expression_),
            l.arena_ ? l.arena_ : r.arena_};
  }
  // The base's own composition of two handles, with the arena put back.
  friend PyExpression combine(const PyExpression &l, const PyExpression &r,
                              auto &&fn) {
    return {fn(l.expression_, r.expression_), l.arena_ ? l.arena_ : r.arena_};
  }
  // The one ternary.
  friend PyExpression select(const PyExpression &, const PyExpression &,
                             const PyExpression &);
  // The reductions and the vector factories in module.cpp.
  friend PyExpression reduce(const PyExpression &, auto &&);
  friend PyExpression reduce(const PyExpression &, const PyExpression &,
                             auto &&);

public:
  [[nodiscard]] constexpr const Base &base() const noexcept {
    return expression_;
  }

  // Null exactly while this is a pending literal, which names no graph yet.
  [[nodiscard]] constexpr const SharedArena &arena() const noexcept { return arena_; }

private:
  Base expression_;
  SharedArena arena_;
};

// A reduction to a scalar, or any other composition that may refuse, raises
// where it is formed; mixed operands take whichever arena is named.
[[nodiscard]] inline PyExpression reduce(const PyExpression &v, auto &&fn) {
  return PyExpression::checked(fn(v.expression_), v.arena_);
}
[[nodiscard]] inline PyExpression reduce(const PyExpression &u,
                                         const PyExpression &v, auto &&fn) {
  return PyExpression::checked(fn(u.expression_, v.expression_),
                               u.arena_ ? u.arena_ : v.arena_);
}

// transpose(M): rt::transpose's handle, kept as the matrix it reads.
struct PyTransposed {
  PyExpression m;
};

// The ternary: the arena is whichever operand names one, as above.
[[nodiscard]] inline PyExpression
select(const PyExpression &c, const PyExpression &t, const PyExpression &f) {
  const auto arena = c.arena() ? c.arena() : t.arena() ? t.arena() : f.arena();
  return {select(c.base(), t.base(), f.base()), arena};
}

} // namespace ddx::py
