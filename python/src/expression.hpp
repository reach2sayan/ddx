#pragma once

#include "error.hpp"
#include "rt/builder.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"

#include <format>
#include <memory>
#include <string>
#include <utility>

// The handle Python holds: an RTExpression that owns its arena.  RTExpression
// stores a *non-owning* Builder* because in C++ an arena's lifetime is lexical;
// a Python closure can capture a symbol and outlive everything.  Shared
// ownership is the whole of what this adds.
//
// It holds one rather than deriving from one: pybind11 describes a bound type
// as descr<N, PyExpression>, and a base class is an associated class where a
// member's type is not.  Deriving would put RTExpression's hidden operator+
// into the overload set for descr + descr, instantiating Numeric<descr> whose
// body is `{ a + b }` -- constraint recursion inside pybind11's machinery.
namespace ddx::py {

class PyExpression {
public:
  using Base = rt::RTExpression<double>;
  using Arena = std::shared_ptr<rt::Builder<double>>;

  constexpr PyExpression() = default;

  // Implicit, and the reason `2.0 * x` and `x * 2.0` both work: pybind11 is
  // told about it with implicitly_convertible, and what follows is the base's
  // own folding -- two pending literals never reach a graph.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr PyExpression(double v) : expression_(v) {}
  PyExpression(Base e, Arena a) noexcept
      : expression_(e), arena_(std::move(a)) {}

  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return expression_.poisoned();
  }

  // The id this names in `arena`, or a refusal: an id from another arena would
  // index this one at random.  A pending literal materialises into it.
  [[nodiscard]] rt::NodeId root_in(rt::Builder<double> &arena) const {
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
      return std::format("Expression({})", arena_->symbols()[n.slot]);
    default:
      // The op and the identity, not the subtree: an id *is* the identity of a
      // subexpression here, and a whole tree is what to_dot() is for.
      return std::format("Expression({} #{})", rt::label_of(n.op), v);
    }
  }

private:
  // The arithmetic surface below is the only thing that puts an arena back on a
  // new handle, so the two halves of a handle are reachable there and nowhere
  // else.  Keyed by opcode rather than by name: the module registers every row
  // of the opcode table through these, so there is no friend per function.
  // Whichever operand names a graph, mirroring RTExpression::form: a pending
  // literal folds against the other rather than materialising into it.
  friend PyExpression unary(rt::OpCode op, const PyExpression &u) {
    return {Base::form(op, u.expression_), u.arena_};
  }
  friend PyExpression binary(rt::OpCode op, const PyExpression &l,
                             const PyExpression &r) {
    return {Base::form(op, l.expression_, r.expression_),
            l.arena_ ? l.arena_ : r.arena_};
  }
  // `z` is a root of `f`; a misuse is kept on the arena for the equation.
  friend void closes(const PyExpression &z, const PyExpression &f) {
    rt::closes(z.expression_, f.expression_);
  }
  // The base's own composition of two handles, with the arena put back.
  friend PyExpression combine(const PyExpression &l, const PyExpression &r,
                              auto &&fn) {
    return {fn(l.expression_, r.expression_), l.arena_ ? l.arena_ : r.arena_};
  }
  // The one ternary.
  friend PyExpression select(const PyExpression &, const PyExpression &,
                             const PyExpression &);
  // The vector handle: one element of it is one of these, over the same
  // arena.
  friend class PyVector;

  [[nodiscard]] constexpr const Base &base() const noexcept {
    return expression_;
  }

  // Null exactly while this is a pending literal, which names no graph yet.
  [[nodiscard]] constexpr const Arena &arena() const noexcept { return arena_; }

  Base expression_;
  Arena arena_;
};

// The comparisons answer an expression, as the base's do.  Only `<` and `<=`
// are nodes; the rest are those two read the other way round, which the base
// already spells with NaN in mind, so each is one composition.
[[nodiscard]] inline PyExpression operator<(const PyExpression &l,
                                            const PyExpression &r) {
  return combine(l, r, [](const auto &a, const auto &b) { return a < b; });
}
[[nodiscard]] inline PyExpression operator<=(const PyExpression &l,
                                             const PyExpression &r) {
  return combine(l, r, [](const auto &a, const auto &b) { return a <= b; });
}
[[nodiscard]] inline PyExpression operator>(const PyExpression &l,
                                            const PyExpression &r) {
  return combine(l, r, [](const auto &a, const auto &b) { return a > b; });
}
[[nodiscard]] inline PyExpression operator>=(const PyExpression &l,
                                             const PyExpression &r) {
  return combine(l, r, [](const auto &a, const auto &b) { return a >= b; });
}
[[nodiscard]] inline PyExpression operator==(const PyExpression &l,
                                             const PyExpression &r) {
  return combine(l, r, [](const auto &a, const auto &b) { return a == b; });
}
[[nodiscard]] inline PyExpression operator!=(const PyExpression &l,
                                             const PyExpression &r) {
  return combine(l, r, [](const auto &a, const auto &b) { return a != b; });
}

// The ternary: the arena is whichever operand names one, as above.
[[nodiscard]] inline PyExpression
select(const PyExpression &c, const PyExpression &t, const PyExpression &f) {
  const auto arena = c.arena() ? c.arena() : t.arena() ? t.arena() : f.arena();
  return {select(c.base(), t.base(), f.base()), arena};
}

[[nodiscard]] inline PyExpression operator+(const PyExpression &l,
                                            const PyExpression &r) {
  return binary(rt::OpCode::Add, l, r);
}
[[nodiscard]] inline PyExpression operator*(const PyExpression &l,
                                            const PyExpression &r) {
  return binary(rt::OpCode::Mul, l, r);
}
[[nodiscard]] inline PyExpression operator/(const PyExpression &l,
                                            const PyExpression &r) {
  return binary(rt::OpCode::Div, l, r);
}
[[nodiscard]] inline PyExpression operator-(const PyExpression &u) {
  return unary(rt::OpCode::Neg, u);
}
[[nodiscard]] inline PyExpression operator-(const PyExpression &l,
                                            const PyExpression &r) {
  return l + (-r);
}

} // namespace ddx::py
