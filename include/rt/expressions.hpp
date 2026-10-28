#pragma once

#include "rt/apply.hpp"
#include "rt/arena.hpp"
#include "rt/opcode.hpp"
#include "util/config.hpp"
#include "util/error.hpp"
#include "util/scope_guard.hpp"

#include <algorithm>
#include <array>
#include <concepts> // std::same_as
#include <cstdint>
#include <functional> // std::invoke
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits> // std::remove_cvref_t
#include <utility>     // std::move

namespace ddx::rt {

template <impl::Numeric T = double> class Expression;

namespace detail {
// Befriended below: an equation is the one thing that takes an expression's
// arena over, and the only reader of the poison an absent arena leaves; the
// tensor factories in rt/tensor.hpp are the other way to a poisoned handle.
struct Router {
  template <impl::Numeric T>
  [[nodiscard]] static constexpr Arena<T> *
  arena(const Expression<T> &e) noexcept {
    return e.arena();
  }
  template <impl::Numeric T>
  [[nodiscard]] static constexpr std::optional<errc>
  why(const Expression<T> &e) noexcept {
    return e.why();
  }
  template <impl::Numeric T>
  [[nodiscard]] static constexpr Expression<T> poison(errc why) noexcept {
    return Expression<T>::poison(why);
  }
  // A table has no width, so size() alone takes it for a scalar.
  template <impl::Numeric T>
  [[nodiscard]] static constexpr bool scalar(const Expression<T> &e) noexcept {
    return e.size() == 0 && !e.is_matrix();
  }
  // The arena of the first expression that names one; a literal names none.
  template <impl::Numeric T>
  [[nodiscard]] static constexpr Arena<T> *
  first_arena(std::span<const Expression<T>> es) noexcept {
    const auto named = std::ranges::find_if(
        es, [](const auto &e) { return e.arena() != nullptr; });
    return named == es.end() ? nullptr : named->arena();
  }
};
} // namespace detail

// Declared ahead of Expression, which befriends it: `named` is where a
// missing arena, or a name another kind of input holds, turns into a poisoned
// expression.
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> named(OpCode op, std::string_view name) noexcept;
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> named(Arena<T> &b, OpCode op,
                                            std::string_view name);

// A handle onto one node, plus the arithmetic surface that builds more.  A null
// arena is a literal not yet given to a graph, which is what lets
// Expression{1} exist for CFieldLike's constructible_from<int>.
//
// The node may be a vector or a matrix: `size()` elements per point, and the
// same arithmetic then builds one elementwise node -- a Map or Zip carrying
// the scalar opcode, a Scale for a scalar times one.  A literal meeting a
// vector or a matrix takes its shape, so every derivative rule in
// ops/adjoints.hpp reads elementwise and no rule is written twice.  A
// mat_constant table taking part is its row.  rt/tensor.hpp holds the
// factories and reductions.
template <impl::Numeric T> class Expression {
public:
  using value_type = T;
  using arena_type = Arena<T>;

  constexpr Expression() = default;

  // By value: Expression is itself Numeric, and a Numeric&& would out-match
  // the copy constructor for a non-const lvalue.  Prevent circular check
  // clang diagnoses the cycle where GCC accepts it.
  template <typename V>
    requires(!std::same_as<std::remove_cvref_t<V>, Expression> &&
             impl::Numeric<V>)
  constexpr Expression(V v) : lit_{static_cast<T>(std::move(v))} {}

  constexpr Expression(Arena<T> &b, NodeId id) noexcept
      : arena_{&b}, id_{id},
        shape_{detail::is_tensor(b[id]) ? shape_of(b, id) : Shape{0, 0}} {}

  [[nodiscard]] constexpr bool pending() const noexcept {
    return arena_ == nullptr;
  }
  // Elements per point: zero for a scalar node or a pending literal.
  [[nodiscard]] constexpr std::uint32_t size() const noexcept {
    return shape_.size();
  }
  [[nodiscard]] constexpr Shape shape() const noexcept { return shape_; }

  // Whether the input named no leaf -- no arena current, or a name another
  // kind of input has -- which surfaces as an errc out of equation() rather
  // than as a zero.
  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return why_.has_value();
  }
  [[nodiscard]] constexpr const T &literal() const noexcept { return lit_; }

  // Give the node an identity in `b`, materialising a pending literal -- a
  // constant, or the row of `shape` it meets -- or a fold `op=` is still
  // accumulating.
  [[nodiscard]] constexpr NodeId id(Arena<T> &b, Shape shape = {0, 0}) const {
    if (arena_ == nullptr) {
      return shape.size() ? b.filled(shape, lit_) : b.constant(lit_);
    }
    // b may be a copy of arena_
    return fold_ == OpCode::Const ? id_ : b.fold_of(fold_, id_);
  }
  // The identity it has in its own arena; no_node for a pending literal.
  [[nodiscard]] constexpr NodeId id() const {
    return arena_ ? id(*arena_) : no_node;
  }

  // One element of a vector, as a scalar.
  [[nodiscard]] constexpr Expression operator[](std::size_t i) const {
    if (why_) {
      return poison(*why_);
    }
    if (pending()) {
      return Expression{lit_};
    }
    if (i >= size()) {
      return poison(errc::tensor_shape);
    }
    return {*arena_,
            arena_->make(opcode<OpCode::Gather>,
                         Element{static_cast<std::uint32_t>(i)}, id())};
  }

  // One node of `op` over its operands.  Pending literals fold without
  // reaching a graph, so the constants the derivative rules produce never
  // become nodes; a table is read as its MatRow.  A tensor operand makes the
  // node elementwise:
  //  - unary, a Map carrying `op`;
  //  - binary, a Zip of two rows at one shape, a literal taking the row's.  A
  //    scalar *node* meets a row only as a factor -- a Scale, one node rather
  //    than a broadcast and a Zip -- so anything else is a shape error;
  //  - a Select copies the arm taken, so the one not taken never meets a 0 --
  //    0 * inf is NaN where log or sqrt is infinite.  A tensor condition, of
  //    the arms' shape, chooses element by element.
  [[nodiscard]] static constexpr Expression
  form(OpCode op, const std::same_as<Expression> auto &...xs) {
    std::optional<errc> why;
    ((why = why ? why : xs.why_), ...);
    if (why) {
      return poison(*why);
    }
    Arena<T> *b = nullptr;
    ((b = b ? b : xs.arena()), ...);
    if (b == nullptr) {
      return Expression{apply<T>(op, xs.literal()...)};
    }
    if ((xs.is_matrix() || ...)) {
      return form(op, xs.row()...);
    }
    if (((xs.size() == 0) && ...)) {
      return {*b, b->make(op, xs.id(*b)...)};
    }
    if constexpr (sizeof...(xs) == 1) {
      return {*b, b->make(opcode<OpCode::Map>, op, xs.id(*b)...)};
    } else if constexpr (sizeof...(xs) == 2) {
      return zipped(op, *b, xs...);
    } else {
      return picked(*b, xs...);
    }
  }

private:
  [[nodiscard]] static constexpr Expression
  zipped(OpCode op, Arena<T> &b, const Expression &l, const Expression &r) {
    if (l.size() != 0 && r.size() != 0 && l.shape_ != r.shape_) {
      return poison(errc::tensor_shape);
    }
    const Expression &vec = l.size() ? l : r;
    const Expression &other = l.size() ? r : l;
    if (other.size() == 0) {
      if (op == OpCode::Mul) {
        return {b, b.make(OpCode::Scale, other.id(b), vec.id(b))};
      }
      if (op == OpCode::Div && &other == &r) {
        return {b, b.make(OpCode::Scale, (Expression{1} / r).id(b), vec.id(b))};
      }
      if (!other.pending()) {
        return poison(errc::tensor_shape);
      }
    }
    return {b, b.make(opcode<OpCode::Zip>, op, l.id(b, vec.shape_),
                      r.id(b, vec.shape_))};
  }
  [[nodiscard]] static constexpr Expression picked(Arena<T> &b,
                                                   const Expression &c,
                                                   const Expression &t,
                                                   const Expression &f) {
    const Shape shape = t.size() ? t.shape() : (f.size() ? f.shape() : c.shape());
    const auto fits = [&](const Expression &e) {
      return e.size() == 0 || e.shape() == shape;
    };
    if (!fits(c) || !fits(t) || !fits(f)) {
      return poison(errc::tensor_shape);
    }
    // A scalar arm spread to the shape: a literal as its row, a node as
    // s * 1, which is s exactly.
    const auto arm = [&](const Expression &e) {
      return e.size() || e.pending()
                 ? e.id(b, shape)
                 : b.make(OpCode::Scale, e.id(b), b.ones(shape));
    };
    return {b, b.make(OpCode::VSelect, c.id(b), arm(t), arm(f))};
  }

public:

  // Hidden friends: ordinary functions for a given Expression<T>, so `x * 2`
  // and `2 * x` both convert.
  friend constexpr Expression operator+(const Expression &l,
                                        const Expression &r) {
    return form(OpCode::Add, l, r);
  }
  friend constexpr Expression operator*(const Expression &l,
                                        const Expression &r) {
    return form(OpCode::Mul, l, r);
  }
  friend constexpr Expression operator/(const Expression &l,
                                        const Expression &r) {
    return form(OpCode::Div, l, r);
  }
  friend constexpr Expression operator-(const Expression &u) {
    return form(OpCode::Neg, u);
  }
  friend constexpr Expression operator-(const Expression &l,
                                        const Expression &r) {
    return l + (-r);
  }

  // `+=`, `*=` and `/=` on a scalar node accumulate: the terms are kept on
  // the arena and become one node -- a fold past two terms, the binary op at
  // two -- when the result is first read, so a loop of `op=` is one node of k
  // terms where the operator would be k.  A copy of an accumulator shares the
  // terms so far and keeps its value.  The op's identity on the right is
  // dropped, as `x + 0` and `x * 1` fold.  Everything else -- poison, a vector
  // or a matrix, a pending literal on the left -- goes through the operator,
  // which propagates poison and folds literals where an in-place version
  // would take a poisoned accumulator, adopt the other side's arena and
  // answer instead of refusing.  The parameter is the class type rather than
  // a Numeric auto, so `op=` accepts exactly what the operator accepts.
  constexpr Expression &operator+=(const Expression &o) {
    return accumulate(OpCode::SumN, o);
  }
  constexpr Expression &operator-=(const Expression &o) { return *this += -o; }
  constexpr Expression &operator*=(const Expression &o) {
    return accumulate(OpCode::ProdN, o);
  }
  constexpr Expression &operator/=(const Expression &o) {
    return accumulate(OpCode::QuotN, o);
  }

private:
  constexpr Expression &accumulate(OpCode fold, const Expression &o) {
    if (why_ || o.why_ || pending() || size() || o.size() || is_matrix() ||
        o.is_matrix()) {
      return *this = form(fold_op(fold), *this, o);
    }
    const T identity = fold == OpCode::SumN ? T{0} : T{1};
    if constexpr (std::equality_comparable<T>) {
      if (o.pending() && o.lit_ == identity) {
        return *this;
      }
    }
    const NodeId t = o.id(*arena_);
    if (arena_->is_constant(t, identity)) {
      return *this;
    }
    if (fold_ != fold) {
      id_ = arena_->push_term(no_node, id(*arena_));
      fold_ = fold;
    }
    id_ = arena_->push_term(id_, t);
    return *this;
  }

public:

#define DDX_RT_UNFN(fn, Op, label, ...)                                        \
  friend constexpr Expression fn(const Expression &u) {                        \
    return form(OpCode::Op, u);                                                \
  }
  DDX_UNARY_MATH_TABLE(DDX_RT_UNFN)
  DDX_RT_UNARY_TABLE(DDX_RT_UNFN)
#undef DDX_RT_UNFN

#define DDX_RT_BINFN(fn, Op, label, ...)                                       \
  friend constexpr Expression fn(const Expression &l, const Expression &r) {   \
    return form(OpCode::Op, l, r);                                             \
  }
  DDX_RT_BINARY_TABLE(DDX_RT_BINFN)
  DDX_RT_COMPARE_TABLE(DDX_RT_BINFN)
#undef DDX_RT_BINFN

  // Operators that answer an expression, not a bool.  That is what keeps
  // Expression outside std::totally_ordered -- every comparison there must be
  // boolean-testable -- so adjoints.hpp's numeric `sign` never outranks the
  // node-building one.  Only `<` and `<=` are nodes; the rest are those two
  // read the other way round.
  friend constexpr Expression operator<(const Expression &l,
                                        const Expression &r) {
    return lt(l, r);
  }
  friend constexpr Expression operator<=(const Expression &l,
                                         const Expression &r) {
    return le(l, r);
  }
  friend constexpr Expression operator>(const Expression &l,
                                        const Expression &r) {
    return lt(r, l);
  }
  friend constexpr Expression operator>=(const Expression &l,
                                         const Expression &r) {
    return le(r, l);
  }

  friend constexpr Expression operator==(const Expression &l,
                                         const Expression &r) {
    const Expression forward = le(l, r);
    return forward * le(r, l);
  }
  // rewritten `!(l == r)` is ill-formed when `==` is not bool.
  friend constexpr Expression operator!=(const Expression &l,
                                         const Expression &r) {
    return Expression{1} - (l == r);
  }

  // Both arms evaluate; `c` is a condition in the C sense, any nonzero value.
  friend constexpr Expression select(const Expression &c, const Expression &t,
                                     const Expression &f) {
    return form(OpCode::Select, c, t, f);
  }

private:
  // What an Equation needs of an expression: which arena it names, and --
  // through poison() -- that it names none.
  friend struct detail::Router;
  template <impl::Numeric U> friend class ExternFunction;
  template <impl::Numeric U>
  friend Expression<U> named(OpCode, std::string_view) noexcept;
  template <impl::Numeric U>
  friend constexpr Expression<U> named(Arena<U> &, OpCode, std::string_view);

  [[nodiscard]] constexpr Arena<T> *arena() const noexcept {
    return arena_;
  }

  // Not a pending literal, which would materialise into the first graph it
  // meets and put a zero where a symbol should be.
  [[nodiscard]] static constexpr Expression poison(errc why) noexcept {
    Expression e;
    e.why_ = why;
    return e;
  }

  [[nodiscard]] constexpr std::optional<errc> why() const noexcept {
    return why_;
  }
  // A mat_constant table, which is no row: arithmetic reads it as its
  // MatRow, gemv() in place.
  [[nodiscard]] constexpr bool is_matrix() const noexcept {
    return arena_ != nullptr && fold_ == OpCode::Const &&
           arena_->op_of(id_) == OpCode::MatConst;
  }
  [[nodiscard]] constexpr Expression row() const {
    return is_matrix()
               ? Expression{*arena_, arena_->make(opcode<OpCode::MatRow>,
                                                  Orientation::plain, id_)}
               : *this;
  }

  Arena<T> *arena_ = nullptr;
  NodeId id_ = no_node; // a node, or while folding the arena's pending entry
  Shape shape_{0, 0};
  OpCode fold_ = OpCode::Const; // the Nary op `op=` is accumulating, if any
  T lit_{};
  std::optional<errc> why_;
};

// The one way a named input enters a graph.  A Var is differentiated; a Param
// is read and never differentiated, no Jacobian column and a value in every
// call; an Implicit is one the caller solved for -- a value in every call,
// like a parameter, and a root of the residual implicit() names, so no
// derivative treats it as fixed and every one carries dz/dx = -F_x / F_z.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> named(Arena<T> &b, OpCode op,
                                            std::string_view name) {
  const NodeId id = b.input(op, name);
  return id == no_node ? Expression<T>::poison(errc::name_clash)
                       : Expression<T>{b, id};
}
// The implicit input `name` and the residual `residual(z)` it is a root of:
// F(z, x) = 0 at every point the caller supplies.  ddx never solves it; it
// differentiates as though it held.
template <impl::Numeric T, std::invocable<const Expression<T> &> F>
[[nodiscard]] constexpr Expression<T>
implicit(Arena<T> &b, std::string_view name, F &&residual) {
  const Expression<T> z = named(b, OpCode::Implicit, name);
  if (!detail::Router::why(z)) {
    const Expression<T> f = std::invoke(DDX_FWD(residual), z);
    const auto why = detail::Router::why(f);
    b.close(z.id(b), why ? no_node : f.id(b), why);
  }
  return z;
}
// The direction a seeded product is taken along
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> seed(Arena<T> &b, std::uint32_t slot) {
  return Expression<T>{b, b.seed_node(slot)};
}
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> lit(Arena<T> &b, const T &v) {
  return Expression<T>{b, b.constant(v)};
}

// extern_fn()'s answer: called on expressions, it puts the function into their
// graph, and every derivative through it reads the partials it supplies.
template <impl::Numeric T> class ExternFunction {
public:
  constexpr ExternFunction(Arena<T> *b, std::uint32_t index) noexcept
      : b_{b}, index_{index} {}
  constexpr explicit ExternFunction(errc why) noexcept : why_{why} {}

  [[nodiscard]] constexpr Expression<T>
  operator()(std::span<const Expression<T>> args) const {
    if (why_) {
      return Expression<T>::poison(*why_);
    }
    if (args.size() != b_->externs()[index_].arity) {
      return Expression<T>::poison(errc::wrong_argument_count);
    }
    for (const auto &a : args) {
      if (a.why_) {
        return Expression<T>::poison(*a.why_);
      }
      if (!detail::Router::scalar(a)) {
        return Expression<T>::poison(errc::tensor_shape);
      }
    }
    const auto ids =
        args |
        std::views::transform([this](const auto &a) { return a.id(*b_); }) |
        impl::to<boost::container::small_vector<NodeId, 8>>();
    return Expression<T>{*b_, b_->extern_call(index_, ids)};
  }
  template <typename... A>
    requires(std::convertible_to<A, Expression<T>> && ...)
  [[nodiscard]] constexpr Expression<T> operator()(const A &...args) const {
    const std::array<Expression<T>, sizeof...(A)> all{Expression<T>{args}...};
    return (*this)(std::span<const Expression<T>>{all});
  }

private:
  Arena<T> *b_ = nullptr;
  std::uint32_t index_ = 0;
  std::optional<errc> why_;
};

// A function of `record.arity` arguments the caller computes, with its
// partials: `fn(in, out)` writes the value, then the first partials, then --
// at order 2 -- the upper triangle of the second, row-major.  ddx calls it
// from any thread that sweeps, as often as it needs, so it must be re-entrant
// and must not throw.
template <impl::Numeric T>
[[nodiscard]] ExternFunction<T>
extern_fn(Arena<T> &b, ExternRecord record,
          std::type_identity_t<ExternFn<T>> fn) {
  if (!record.valid() || !fn) {
    return ExternFunction<T>{errc::extern_shape};
  }
  const auto index = b.add_extern(ExternEntry<T>{
      std::move(record), std::make_shared<const ExternFn<T>>(std::move(fn))});
  return index ? ExternFunction<T>{&b, *index}
               : ExternFunction<T>{errc::extern_shape};
}

// The arena var() registers into while an equation() callback runs.  Thread-
// local, and held by the same guard the drivers seed derivatives with.
namespace detail {

// Spell one type N times in a pack expansion.
template <std::size_t, impl::Numeric T> using Repeat = Expression<T>;

template <impl::Numeric T>
inline thread_local Arena<T> *current_arena = nullptr;

// The runtime counterpart of scoped_seed; the arena is not a constant, so it
// comes in as an argument.
template <impl::Numeric T>
[[nodiscard]] auto scoped_arena(Arena<T> &b) noexcept {
  return impl::scoped_value<Arena<T> *>{current_arena<T>, &b};
}

} // namespace detail

// The spelling a model uses: named in the arena its callback runs in.
template <impl::Numeric T>
[[nodiscard]] Expression<T> named(OpCode op, std::string_view name) noexcept {
  return (detail::current_arena<T> == nullptr)
             ? Expression<T>::poison(errc::outside_equation)
             : named(*detail::current_arena<T>, op, name);
}
template <impl::Numeric T = double>
[[nodiscard]] ExternFunction<T>
extern_fn(ExternRecord record, std::type_identity_t<ExternFn<T>> fn) {
  return detail::current_arena<T> == nullptr
             ? ExternFunction<T>{errc::outside_equation}
             : extern_fn(*detail::current_arena<T>, std::move(record),
                         std::move(fn));
}

// var() and param(), in both spellings, off the table of named kinds; an
// implicit is named with its residual, above.
#define DDX_RT_INPUT(fn, Op, ...)                                              \
  template <impl::Numeric T>                                                   \
    requires(OpCode::Op != OpCode::Implicit)                                   \
  [[nodiscard]] constexpr Expression<T> fn(Arena<T> &b,                      \
                                           std::string_view name) {            \
    return named(b, OpCode::Op, name);                                         \
  }                                                                            \
  template <impl::Numeric T = double>                                          \
    requires(OpCode::Op != OpCode::Implicit)                                   \
  [[nodiscard]] Expression<T> fn(std::string_view name) noexcept {             \
    return named<T>(OpCode::Op, name);                                         \
  }
DDX_RT_NAMED_TABLE(DDX_RT_INPUT)
#undef DDX_RT_INPUT
template <impl::Numeric T = double, std::invocable<const Expression<T> &> F>
[[nodiscard]] Expression<T> implicit(std::string_view name, F &&residual) {
  return detail::current_arena<T> == nullptr
             ? named<T>(OpCode::Implicit, name)
             : implicit(*detail::current_arena<T>, name, DDX_FWD(residual));
}

} // namespace ddx::rt

namespace ddx::impl {
template <Numeric S>
inline constexpr bool is_commutative_multiply_v<rt::Expression<S>> =
    is_commutative_multiply_v<S>;
} // namespace ddx::impl
