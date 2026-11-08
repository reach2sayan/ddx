#pragma once

#include "rt/apply.hpp"
#include "rt/arena.hpp"
#include "rt/opcode.hpp"
#include "util/config.hpp"
#include "util/error.hpp"
#include "util/meta.hpp"
#include "util/scope_guard.hpp"

#include <boost/hof/match.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace ddx::rt {

template <impl::Numeric T = double> class Expression;

namespace detail {
// A friend of Expression that gives equations and the tensor factories access
// to an expression's arena and poison.
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
  // A table has size() zero, so it must be excluded explicitly.
  template <impl::Numeric T>
  [[nodiscard]] static constexpr bool scalar(const Expression<T> &e) noexcept {
    return e.size() == 0 && !e.is_matrix();
  }
  // The arena of the first expression that has one; literals have none.
  template <impl::Numeric T>
  [[nodiscard]] static constexpr Arena<T> *
  first_arena(std::span<const Expression<T>> es) noexcept {
    const auto named = std::ranges::find_if(
        es, [](const auto &e) { return e.arena() != nullptr; });
    return named == es.end() ? nullptr : named->arena();
  }
};
} // namespace detail

// Declared before Expression, which befriends it. A missing arena or a
// clashing input kind makes named() return a poisoned expression.
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> named(OpCode op, std::string_view name) noexcept;
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> named(Arena<T> &b, OpCode op,
                                            std::string_view name);

// A handle to one graph node with arithmetic that builds more nodes. A null
// arena means a pending literal; tensor nodes get elementwise arithmetic.
template <impl::Numeric T> class Expression {
public:
  using value_type = T;
  using arena_type = Arena<T>;

  constexpr Expression() = default;

  // Taken by value, since a Numeric&& would beat the copy constructor for a
  // non-const lvalue. The exclusion stops a constraint cycle Clang rejects.
  template <typename V>
    requires(!std::same_as<std::remove_cvref_t<V>, Expression> &&
             impl::Numeric<V>)
  constexpr Expression(V v) : lit_{static_cast<T>(std::move(v))} {}

  constexpr Expression(Arena<T> &b, NodeId id) noexcept
      : arena_{&b}, at_{id},
        shape_{detail::is_tensor(b[id]) ? shape_of(b, id)
                                        : Shape{.rows = 0, .cols = 0}} {}

  [[nodiscard]] constexpr bool pending() const noexcept {
    return arena_ == nullptr;
  }
  // Elements per point, which is zero for a scalar or a pending literal.
  [[nodiscard]] constexpr std::uint32_t size() const noexcept {
    return shape_.size();
  }
  [[nodiscard]] constexpr Shape shape() const noexcept { return shape_; }

  // Whether building failed, for example outside an equation callback. The
  // error surfaces from equation() rather than as a silent zero.
  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return why_.has_value();
  }
  [[nodiscard]] constexpr const T &literal() const noexcept { return lit_; }

  // The node id in arena b, materialising a pending literal or an unfinished
  // op= fold first.
  [[nodiscard]] constexpr NodeId
  id(Arena<T> &b, Shape shape = {.rows = 0, .cols = 0}) const {
    assert(!why_ && "a refusal has no identity: it would read as a constant");
    if (arena_ == nullptr) {
      return shape.size() ? b.filled(shape, lit_) : b.constant(lit_);
    }
    // b may be a copy of arena_.
    return std::visit(boost::hof::match([](NodeId at) { return at; },
                                         [&b](const auto &at) { return b.fold_of(at); }),
                      at_);
  }
  // The node id in its own arena, or no_node for a pending literal.
  [[nodiscard]] constexpr NodeId id() const {
    return arena_ ? id(*arena_) : no_node;
  }

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

  // Builds one op node, folding pending literals without touching a graph.
  // Tensor operands give a Map, a Zip, a Scale or a VSelect instead.
  [[nodiscard]] static constexpr Expression
  form(OpCode op, const std::same_as<Expression> auto &...xs) {
    std::optional<errc> why;
    Arena<T> *b = nullptr;
    meta::for_each(std::tie(xs...), [&](const Expression &x) {
      why = why ? why : x.why_;
    });
    if (why) {
      return poison(*why);
    }
    meta::for_each(std::tie(xs...), [&](const Expression &x) {
      b = b ? b : x.arena();
    });
    if (b == nullptr) {
      return Expression{apply<T>(op, xs.literal()...)};
    }
    // Node ids are meaningless outside their own arena.
    if (((xs.arena() != nullptr && xs.arena() != b) || ...)) {
      return poison(errc::tensor_shape);
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
        return {b, b.make(opcode<OpCode::Scale>,
                          {.scalar = other.id(b), .vec = vec.id(b)})};
      }
      if (op == OpCode::Div && &other == &r) {
        return {
            b, b.make(opcode<OpCode::Scale>,
                      {.scalar = (Expression{1} / r).id(b), .vec = vec.id(b)})};
      }
      if (!other.pending()) {
        return poison(errc::tensor_shape);
      }
    }
    return {
        b, b.make(opcode<OpCode::Zip>, op,
                  {.left = l.id(b, vec.shape_), .right = r.id(b, vec.shape_)})};
  }
  [[nodiscard]] static constexpr Expression picked(Arena<T> &b,
                                                   const Expression &c,
                                                   const Expression &t,
                                                   const Expression &f) {
    const Shape shape =
        t.size() ? t.shape() : (f.size() ? f.shape() : c.shape());
    const auto fits = [&](const Expression &e) {
      return e.size() == 0 || e.shape() == shape;
    };
    if (!fits(c) || !fits(t) || !fits(f)) {
      return poison(errc::tensor_shape);
    }
    // Broadcasts a scalar arm to the shape; a node becomes s * 1, which is
    // exactly s.
    const auto arm = [&](const Expression &e) {
      return e.size() || e.pending()
                 ? e.id(b, shape)
                 : b.make(opcode<OpCode::Scale>,
                          {.scalar = e.id(b), .vec = b.ones(shape)});
    };
    return {b, b.make(opcode<OpCode::VSelect>,
                      {.cond = c.id(b), .then = arm(t), .otherwise = arm(f)})};
  }

public:
  // Hidden friends, so that both x * 2 and 2 * x convert the literal.
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

  // On a scalar node, op= collects terms into one n-ary node built on first
  // read, so a loop of k updates is one node. Other cases use the operator.
  constexpr Expression &operator+=(const Expression &o) {
    return accumulate<OpCode::SumN>(o);
  }
  constexpr Expression &operator-=(const Expression &o) { return *this += -o; }
  constexpr Expression &operator*=(const Expression &o) {
    return accumulate<OpCode::ProdN>(o);
  }
  constexpr Expression &operator/=(const Expression &o) {
    return accumulate<OpCode::QuotN>(o);
  }

private:
  template <OpCode Fold> constexpr Expression &accumulate(const Expression &o) {
    if (why_ || o.why_ || pending() || size() || o.size() || is_matrix() ||
        o.is_matrix() || (!o.pending() && o.arena_ != arena_)) {
      return *this = form(fold_op(Fold), *this, o);
    }
    const T identity = Fold == OpCode::SumN ? T{0} : T{1};
    if constexpr (std::equality_comparable<T>) {
      using impl::same_value;
      if (o.pending() && same_value(o.lit_, identity)) {
        return *this;
      }
    }
    const NodeId t = o.id(*arena_);
    if (arena_->is_constant(t, identity)) {
      return *this;
    }
    using Folding = detail::Folding<Fold>;
    const auto *folding = std::get_if<Folding>(&at_);
    const detail::PendingEntry last =
        folding ? folding->entry
                : arena_->push_term(detail::no_entry, id(*arena_));
    at_ = Folding{arena_->push_term(last, t)};
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

  // Comparisons return expressions, not bools, which keeps Expression out of
  // std::totally_ordered so the numeric sign() never outranks the node one.
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
  // Needed because the rewritten !(l == r) is ill-formed when == is not bool.
  friend constexpr Expression operator!=(const Expression &l,
                                         const Expression &r) {
    return Expression{1} - (l == r);
  }

  // Both arms are evaluated; any nonzero c selects t.
  friend constexpr Expression select(const Expression &c, const Expression &t,
                                     const Expression &f) {
    return form(OpCode::Select, c, t, f);
  }

private:
  // The accessors an Equation needs: the arena and the poison state.
  friend struct detail::Router;
  template <impl::Numeric U> friend class ExternFunction;
  template <impl::Numeric U>
  friend Expression<U> named(OpCode, std::string_view) noexcept;
  template <impl::Numeric U>
  friend constexpr Expression<U> named(Arena<U> &, OpCode, std::string_view);

  [[nodiscard]] constexpr Arena<T> *arena() const noexcept { return arena_; }

  // A poisoned expression is not a pending literal, which would otherwise turn
  // into a zero node in the first graph it meets.
  [[nodiscard]] static constexpr Expression poison(errc why) noexcept {
    Expression e;
    e.why_ = why;
    return e;
  }

  [[nodiscard]] constexpr std::optional<errc> why() const noexcept {
    return why_;
  }
  // Whether this is a mat_constant table, which arithmetic reads through
  // row().
  [[nodiscard]] constexpr bool is_matrix() const noexcept {
    const auto *node = std::get_if<NodeId>(&at_);
    return arena_ != nullptr && node &&
           arena_->op_of(*node) == OpCode::MatConst;
  }
  [[nodiscard]] constexpr Expression row() const {
    return is_matrix()
               ? Expression{*arena_, arena_->make(opcode<OpCode::MatRow>,
                                                  Orientation::plain,
                                                  std::get<NodeId>(at_))}
               : *this;
  }

  Arena<T> *arena_ = nullptr;
  // A node id, or the fold that op= is still accumulating.
#define DDX_RT_FOLDING(fn, Op, ...) , detail::Folding<OpCode::Op>
  std::variant<NodeId DDX_RT_NARY_TABLE(DDX_RT_FOLDING)> at_{no_node};
#undef DDX_RT_FOLDING
  Shape shape_{0, 0};
  T lit_{};
  std::optional<errc> why_;
};

// Adds a named input: a Var is differentiated, a Param never is, and an
// Implicit is differentiated through its residual as dz/dx = -F_x / F_z.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> named(Arena<T> &b, OpCode op,
                                            std::string_view name) {
  const auto id = b.input(op, name);
  return id ? Expression<T>{b, *id} : Expression<T>::poison(errc::name_clash);
}
// An implicit input z with residual F(z, x) = 0 at every supplied point. ddx
// never solves F; it differentiates as though F held.
template <impl::Numeric T, std::invocable<const Expression<T> &> F>
[[nodiscard]] constexpr Expression<T>
implicit(Arena<T> &b, std::string_view name, F &&residual) {
  const Expression<T> z = named(b, OpCode::Implicit, name);
  if (!detail::Router::why(z)) {
    const Expression<T> f = std::invoke(DDX_FWD(residual), z);
    const auto why = detail::Router::why(f);
    b.close(z.id(b), why ? std::expected<NodeId, errc>{std::unexpect, *why}
                         : std::expected<NodeId, errc>{f.id(b)});
  }
  return z;
}
// The direction a seeded product is taken along.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> seed(Arena<T> &b, std::uint32_t slot) {
  return Expression<T>{b, b.seed_node(Slot{slot})};
}
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> lit(Arena<T> &b, const T &v) {
  return Expression<T>{b, b.constant(v)};
}

// A user function returned by extern_fn(). Calling it adds a node whose
// derivatives use the partials the function supplies.
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

// fn(in, out) writes the value, the first partials, then at order 2 the
// row-major upper triangle of the second. It must be re-entrant, must not
// throw, and must not sweep the equation calling it, whose lock is already
// held.
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

namespace detail {

// The thread-local arena that var() registers into during an equation()
// callback.
template <impl::Numeric T>
inline thread_local Arena<T> *current_arena = nullptr;

// Sets the current arena for the lifetime of the returned guard.
template <impl::Numeric T>
[[nodiscard]] auto scoped_arena(Arena<T> &b) noexcept {
  return impl::scoped_value<Arena<T> *>{current_arena<T>, &b};
}

} // namespace detail

// Names an input in the current callback's arena.
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

// var() and param(), with and without an explicit arena.
#define DDX_RT_INPUT(fn, Op, ...)                                              \
  template <impl::Numeric T>                                                   \
    requires(OpCode::Op != OpCode::Implicit)                                   \
  [[nodiscard]] constexpr Expression<T> fn(Arena<T> &b,                        \
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
