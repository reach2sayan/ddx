#pragma once

#include "rt/apply.hpp"
#include "rt/builder.hpp"
#include "rt/opcode.hpp"
#include "util/error.hpp"
#include "util/scope_guard.hpp"

#include <array>
#include <concepts> // std::same_as
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits> // std::remove_cvref_t
#include <utility>     // std::move

namespace ddx::rt {

template <impl::Numeric T = double> class RTExpression;

namespace detail {
// Befriended below: an equation is the one thing that takes an expression's
// arena over, and the only reader of the poison an absent arena leaves.
struct Rooting {
  template <impl::Numeric T>
  [[nodiscard]] static constexpr Builder<T> *
  builder(const RTExpression<T> &e) noexcept {
    return e.builder();
  }
  template <impl::Numeric T>
  [[nodiscard]] static constexpr std::optional<errc>
  why(const RTExpression<T> &e) noexcept {
    return e.why();
  }
  template <impl::Numeric T>
  [[nodiscard]] static constexpr RTExpression<T> poisoned(errc why) noexcept {
    return RTExpression<T>::poison(why);
  }
};

// The Builder's extern half, for the factory below and the sweeps.
struct Externs {
  template <impl::Numeric T>
  static constexpr std::uint32_t add(Builder<T> &b, ExternEntry<T> e) {
    return b.add_extern(std::move(e));
  }
  template <impl::Numeric T>
  static constexpr NodeId node(Builder<T> &b, OpCode op, ExternSlot slot,
                               NodeId args) {
    return b.extern_node(op, slot, args);
  }
  template <impl::Numeric T>
  static constexpr NodeId args(Builder<T> &b, NodeId first, NodeId rest) {
    return b.extern_args(first, rest);
  }
  template <impl::Numeric T>
  static constexpr void bind(Builder<T> &b, std::size_t i,
                             std::shared_ptr<const ExternFn<T>> fn) {
    b.bind(i, std::move(fn));
  }
};
} // namespace detail

// Declared ahead of RTExpression, which befriends it: `var` is where a missing
// arena turns into a poisoned expression.
template <impl::Numeric T = double>
[[nodiscard]] RTExpression<T> var(std::string_view name) noexcept;
template <impl::Numeric T = double>
[[nodiscard]] RTExpression<T> param(std::string_view name) noexcept;
template <impl::Numeric T = double>
[[nodiscard]] RTExpression<T> implicit(std::string_view name) noexcept;

// The primitive the one above reaches: it makes poison as well, for the arena
// that has stopped taking symbols.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> var(Builder<T> &b,
                                            std::string_view name);

// A handle onto one node, plus the arithmetic surface that builds more.  A null
// builder is a literal not yet given to a graph, which is what lets
// RTExpression{1} exist for CFieldLike's constructible_from<int>.
template <impl::Numeric T> class RTExpression {
public:
  using value_type = T;
  using builder_type = Builder<T>;

  constexpr RTExpression() = default;

  // By value: RTExpression is itself Numeric, and a Numeric&& would out-match
  // the copy constructor for a non-const lvalue.  Prevent circular check
  // clang diagnoses the cycle where GCC accepts it.
  template <typename V>
    requires(!std::same_as<std::remove_cvref_t<V>, RTExpression> &&
             impl::Numeric<V>)
  constexpr RTExpression(V v) : lit_{static_cast<T>(std::move(v))} {}

  constexpr RTExpression(Builder<T> &b, NodeId id) noexcept
      : builder_{&b}, id_{id} {}

  [[nodiscard]] constexpr bool pending() const noexcept {
    return builder_ == nullptr;
  }

  // Whether the input named no leaf -- no arena current, or a name another
  // kind of input has -- which surfaces as an errc out of equation() rather
  // than as a zero.
  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return why_.has_value();
  }
  [[nodiscard]] constexpr const T &literal() const noexcept { return lit_; }

  // Give the node an identity in `b`, materialising a pending literal.
  [[nodiscard]] constexpr NodeId id(Builder<T> &b) const {
    return builder_ ? id_ : b.constant(lit_);
  }

  // Two pending literals fold without reaching a graph, so the constants the
  // derivative rules produce never become nodes.
  [[nodiscard]] static constexpr RTExpression form(OpCode op,
                                                   const RTExpression &u) {
    if (u.why_) {
      return poison(*u.why_);
    }
    Builder<T> *const b = u.builder();
    return b ? RTExpression{*b, b->make(op, u.id(*b))}
             : RTExpression{apply<T>(op, u.literal())};
  }

  [[nodiscard]] static constexpr RTExpression
  form(OpCode op, const RTExpression &l, const RTExpression &r) {
    if (l.why_ || r.why_) {
      return poison(l.why_ ? *l.why_ : *r.why_);
    }
    Builder<T> *const b = l.builder() ? l.builder() : r.builder();
    return b ? RTExpression{*b, b->make(op, l.id(*b), r.id(*b))}
             : RTExpression{apply<T>(op, l.literal(), r.literal())};
  }

  [[nodiscard]] static constexpr RTExpression form(OpCode op,
                                                   const RTExpression &c,
                                                   const RTExpression &t,
                                                   const RTExpression &f) {
    if (c.why_ || t.why_ || f.why_) {
      return poison(c.why_ ? *c.why_ : (t.why_ ? *t.why_ : *f.why_));
    }
    Builder<T> *const b =
        c.builder() ? c.builder() : (t.builder() ? t.builder() : f.builder());
    return b ? RTExpression{*b, b->make(op, c.id(*b), t.id(*b), f.id(*b))}
             : RTExpression{
                   apply<T>(op, c.literal(), t.literal(), f.literal())};
  }

  // Hidden friends: ordinary functions for a given RTExpression<T>, so `x * 2`
  // and `2 * x` both convert.
  friend constexpr RTExpression operator+(const RTExpression &l,
                                          const RTExpression &r) {
    return form(OpCode::Add, l, r);
  }
  friend constexpr RTExpression operator*(const RTExpression &l,
                                          const RTExpression &r) {
    return form(OpCode::Mul, l, r);
  }
  friend constexpr RTExpression operator/(const RTExpression &l,
                                          const RTExpression &r) {
    return form(OpCode::Div, l, r);
  }
  friend constexpr RTExpression operator-(const RTExpression &u) {
    return form(OpCode::Neg, u);
  }
  friend constexpr RTExpression operator-(const RTExpression &l,
                                          const RTExpression &r) {
    return l + (-r);
  }

  // Through the operators above, never into builder_ and id_ directly: form()
  // propagates poison and folds pending literals, where an in-place version
  // would take a poisoned accumulator, adopt the other side's arena and answer
  // instead of refusing.  The parameter is the class type rather than a Numeric
  // auto, so += accepts exactly what + accepts.
  constexpr RTExpression &operator+=(const RTExpression &o) {
    return *this = *this + o;
  }
  constexpr RTExpression &operator-=(const RTExpression &o) {
    return *this = *this - o;
  }
  constexpr RTExpression &operator*=(const RTExpression &o) {
    return *this = *this * o;
  }
  constexpr RTExpression &operator/=(const RTExpression &o) {
    return *this = *this / o;
  }

#define DDX_RT_UNFN(fn, Op, label, ...)                                        \
  friend constexpr RTExpression fn(const RTExpression &u) {                    \
    return form(OpCode::Op, u);                                                \
  }
  DDX_UNARY_MATH_TABLE(DDX_RT_UNFN)
  DDX_RT_UNARY_TABLE(DDX_RT_UNFN)
#undef DDX_RT_UNFN

#define DDX_RT_BINFN(fn, Op, label, ...)                                       \
  friend constexpr RTExpression fn(const RTExpression &l,                      \
                                   const RTExpression &r) {                    \
    return form(OpCode::Op, l, r);                                             \
  }
  DDX_RT_BINARY_TABLE(DDX_RT_BINFN)
  DDX_RT_COMPARE_TABLE(DDX_RT_BINFN)
#undef DDX_RT_BINFN

  // Operators that answer an expression, not a bool.  That is what keeps
  // RTExpression outside std::totally_ordered -- every comparison there must be
  // boolean-testable -- so adjoints.hpp's numeric `sign` never outranks the
  // node-building one.  Only `<` and `<=` are nodes; the rest are those two
  // read the other way round.
  friend constexpr RTExpression operator<(const RTExpression &l,
                                          const RTExpression &r) {
    return lt(l, r);
  }
  friend constexpr RTExpression operator<=(const RTExpression &l,
                                           const RTExpression &r) {
    return le(l, r);
  }
  friend constexpr RTExpression operator>(const RTExpression &l,
                                          const RTExpression &r) {
    return lt(r, l);
  }
  friend constexpr RTExpression operator>=(const RTExpression &l,
                                           const RTExpression &r) {
    return le(r, l);
  }

  friend constexpr RTExpression operator==(const RTExpression &l,
                                           const RTExpression &r) {
    const RTExpression forward = le(l, r);
    return forward * le(r, l);
  }
  // rewritten `!(l == r)` is ill-formed when `==` is not bool.
  friend constexpr RTExpression operator!=(const RTExpression &l,
                                           const RTExpression &r) {
    return RTExpression{1} - (l == r);
  }

  // Both arms evaluate; `c` is a condition in the C sense, any nonzero value.
  friend constexpr RTExpression
  select(const RTExpression &c, const RTExpression &t, const RTExpression &f) {
    return form(OpCode::Select, c, t, f);
  }

private:
  // What an Equation needs of an expression: which arena it names, and --
  // through poison() -- that it names none.
  friend struct detail::Rooting;
  template <impl::Numeric U>
  friend RTExpression<U> var(std::string_view) noexcept;
  template <impl::Numeric U>
  friend constexpr RTExpression<U> var(Builder<U> &, std::string_view);
  template <impl::Numeric U>
  friend RTExpression<U> param(std::string_view) noexcept;
  template <impl::Numeric U>
  friend constexpr RTExpression<U> param(Builder<U> &, std::string_view);
  template <impl::Numeric U>
  friend RTExpression<U> implicit(std::string_view) noexcept;
  template <impl::Numeric U>
  friend constexpr RTExpression<U> implicit(Builder<U> &, std::string_view);
  template <impl::Numeric U>
  friend constexpr void closes(const RTExpression<U> &,
                               const RTExpression<U> &);

  [[nodiscard]] constexpr Builder<T> *builder() const noexcept {
    return builder_;
  }

  // Not a pending literal, which would materialise into the first graph it
  // meets and put a zero where a symbol should be.
  [[nodiscard]] static constexpr RTExpression poison(errc why) noexcept {
    RTExpression e;
    e.why_ = why;
    return e;
  }

  [[nodiscard]] constexpr std::optional<errc> why() const noexcept {
    return why_;
  }

  Builder<T> *builder_ = nullptr;
  NodeId id_ = no_node;
  T lit_{};
  std::optional<errc> why_;
};

template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> var(Builder<T> &b,
                                            std::string_view name) {
  const NodeId id = b.input(OpCode::Var, name);
  return id == no_node ? RTExpression<T>::poison(errc::name_clash)
                       : RTExpression<T>{b, id};
}
// An input that is read and never differentiated: no Jacobian column, and a
// value in every call.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> param(Builder<T> &b,
                                              std::string_view name) {
  const NodeId id = b.input(OpCode::Param, name);
  return id == no_node ? RTExpression<T>::poison(errc::name_clash)
                       : RTExpression<T>{b, id};
}
// An input the caller solved for: a value in every call, like a parameter,
// and a root of the residual closes() names -- so no derivative treats it as
// fixed, and every one carries dz/dx = -F_x / F_z.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> implicit(Builder<T> &b,
                                                 std::string_view name) {
  const NodeId id = b.input(OpCode::Implicit, name);
  return id == no_node ? RTExpression<T>::poison(errc::name_clash)
                       : RTExpression<T>{b, id};
}
// `z` is a root of `f`: F(z, x) = 0 at every point the caller supplies.  ddx
// never solves it; it differentiates as though it held.
template <impl::Numeric T>
constexpr void closes(const RTExpression<T> &z, const RTExpression<T> &f) {
  if (z.why_ || f.why_) {
    return; // the poison reaches the equation through its outputs
  }
  Builder<T> *const b = z.builder_ ? z.builder_ : f.builder_;
  if (b == nullptr) {
    return; // two literals: nothing names an arena to refuse in
  }
  b->close(z.builder_ ? z.id_ : no_node, f.id(*b));
}
// The direction a seeded product is taken along
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> seed(Builder<T> &b,
                                             std::uint32_t slot) {
  return RTExpression<T>{b, b.seed_node(slot)};
}
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> lit(Builder<T> &b, const T &v) {
  return RTExpression<T>{b, b.constant(v)};
}

// What extern_fn() registers: a name for files and errors, how many
// arguments, and how many orders of partials the function fills -- 1 or 2,
// since no call ddx answers asks for a third.
struct ExternSpec {
  std::string_view name;
  std::size_t arity = 1;
  std::size_t order = 2;
};

// extern_fn()'s answer: called on expressions, it puts the function into their
// graph, and every derivative through it reads the partials it supplies.
template <impl::Numeric T> class ExternFunction {
public:
  constexpr ExternFunction(Builder<T> *b, std::uint32_t index,
                           std::size_t arity) noexcept
      : b_{b}, index_{index}, arity_{arity} {}
  constexpr explicit ExternFunction(errc why) noexcept : why_{why} {}

  [[nodiscard]] constexpr RTExpression<T>
  operator()(std::span<const RTExpression<T>> args) const {
    if (why_) {
      return detail::Rooting::poisoned<T>(*why_);
    }
    if (args.size() != arity_) {
      return detail::Rooting::poisoned<T>(errc::wrong_argument_count);
    }
    for (const auto &a : args) {
      if (const auto why = detail::Rooting::why(a)) {
        return detail::Rooting::poisoned<T>(*why);
      }
    }
    // The list from the back: a list of one is the argument itself.
    NodeId list = args.back().id(*b_);
    for (const auto &a : args | std::views::reverse | std::views::drop(1)) {
      list = detail::Externs::args(*b_, a.id(*b_), list);
    }
    return RTExpression<T>{
        *b_, detail::Externs::node(*b_, OpCode::Extern,
                                   {.function = index_, .component = 0}, list)};
  }
  template <typename... A>
    requires(std::convertible_to<A, RTExpression<T>> && ...)
  [[nodiscard]] constexpr RTExpression<T> operator()(const A &...args) const {
    const std::array<RTExpression<T>, sizeof...(A)> all{
        RTExpression<T>{args}...};
    return (*this)(std::span<const RTExpression<T>>{all});
  }

private:
  Builder<T> *b_ = nullptr;
  std::uint32_t index_ = 0;
  std::size_t arity_ = 0;
  std::optional<errc> why_;
};

// A function of `spec.arity` arguments the caller computes, with its partials:
// `fn(in, out)` writes the value, then the first partials, then -- at order 2
// -- the upper triangle of the second, row-major.  ddx calls it from any
// thread that sweeps, as often as it needs, so it must be re-entrant and must
// not throw.
template <impl::Numeric T>
[[nodiscard]] ExternFunction<T>
extern_fn(Builder<T> &b, ExternSpec spec,
          std::type_identity_t<ExternFn<T>> fn) {
  if (spec.arity == 0) {
    return ExternFunction<T>{errc::wrong_argument_count};
  }
  if (spec.order < 1 || spec.order > 2 || !fn) {
    return ExternFunction<T>{errc::extern_order};
  }
  const auto index = detail::Externs::add(
      b, ExternEntry<T>{.name = std::string{spec.name},
                        .arity = static_cast<std::uint32_t>(spec.arity),
                        .order = static_cast<std::uint32_t>(spec.order),
                        .fn = std::make_shared<const ExternFn<T>>(
                            std::move(fn))});
  return ExternFunction<T>{&b, index, spec.arity};
}

// The arena var() registers into while an equation() callback runs.  Thread-
// local, and held by the same guard the drivers seed derivatives with.
namespace detail {

// Spell one type N times in a pack expansion.
template <std::size_t, impl::Numeric T> using Repeat = RTExpression<T>;

template <impl::Numeric T>
inline thread_local Builder<T> *current_arena = nullptr;

// The runtime counterpart of scoped_seed; the arena is not a constant, so it
// comes in as an argument.
template <impl::Numeric T>
[[nodiscard]] auto scoped_arena(Builder<T> &b) noexcept {
  return impl::scoped_value<Builder<T> *>{current_arena<T>, &b};
}

} // namespace detail

// The spellings a model uses; the two-argument forms above are the
// primitives.
template <impl::Numeric T>
[[nodiscard]] RTExpression<T> var(std::string_view name) noexcept {
  return (detail::current_arena<T> == nullptr)
             ? RTExpression<T>::poison(errc::no_arena)
             : var(*detail::current_arena<T>, name);
}
template <impl::Numeric T>
[[nodiscard]] RTExpression<T> param(std::string_view name) noexcept {
  return (detail::current_arena<T> == nullptr)
             ? RTExpression<T>::poison(errc::no_arena)
             : param(*detail::current_arena<T>, name);
}
// The spelling a model uses: registered in the arena its callback runs in.
template <impl::Numeric T = double>
[[nodiscard]] ExternFunction<T>
extern_fn(ExternSpec spec, std::type_identity_t<ExternFn<T>> fn) {
  return detail::current_arena<T> == nullptr
             ? ExternFunction<T>{errc::no_arena}
             : extern_fn(*detail::current_arena<T>, spec, std::move(fn));
}
template <impl::Numeric T>
[[nodiscard]] RTExpression<T> implicit(std::string_view name) noexcept {
  return (detail::current_arena<T> == nullptr)
             ? RTExpression<T>::poison(errc::no_arena)
             : implicit(*detail::current_arena<T>, name);
}

} // namespace ddx::rt

namespace ddx::impl {
template <Numeric S>
inline constexpr bool is_commutative_multiply_v<rt::RTExpression<S>> =
    is_commutative_multiply_v<S>;
} // namespace ddx::impl
