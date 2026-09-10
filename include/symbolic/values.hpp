#pragma once
#include "ops/operations.hpp"
#include "ops/scalar.hpp"
#include "symbolic/expressions.hpp"
#include "symbolic/simplify.hpp"
#include "symbolic/symbol.hpp"
#include <functional> // the literal folds
#include <utility>

namespace ddx::impl {

template <typename LHS, typename RHS>
concept CSameValueType =
    std::same_as<typename LHS::value_type, typename RHS::value_type>;

template <typename LHS, typename RHS>
concept CompatibleValueTypes =
    CSameValueType<LHS, RHS> ||
    std::convertible_to<typename LHS::value_type, typename RHS::value_type> ||
    std::convertible_to<typename RHS::value_type, typename LHS::value_type>;

// Promote a bare scalar into value_type as a zero-derivative constant;
// ConstantEmbedder recurses through every Dual<> nesting level.
template <Numeric VT, CArithmetic S>
constexpr Constant<VT> promote_scalar(S s) noexcept {
  return Constant<VT>{
      ConstantEmbedder<VT>::embed(static_cast<scalar_base_t<VT>>(s))};
}

// The three spellings each arithmetic operator serves: two expressions of
// compatible value type, or an expression and a bare scalar on either side.
template <typename A, typename B>
concept CBinaryOperands =
    (CExpression<A> && CExpression<B> && CompatibleValueTypes<A, B>) ||
    (CExpression<A> && CArithmetic<B>) || (CArithmetic<A> && CExpression<B>);

namespace detail {

// What every arithmetic operator does before it builds a node: a bare scalar
// is promoted, two literals fold to a literal, two constants to a constant;
// anything else is what `build` makes.  A T-valued template argument exists
// only for structural T, so a dual scalar puts back only 0 and 1.  Fold is a
// type so that the literal fold is a constant expression.
template <typename Fold, typename A, typename B>
  requires CBinaryOperands<A, B>
constexpr auto arithmetic(const A &a, const B &b, auto build) noexcept {
  if constexpr (CArithmetic<A>) {
    return arithmetic<Fold>(promote_scalar<typename B::value_type>(a), b,
                            build);
  } else if constexpr (CArithmetic<B>) {
    return arithmetic<Fold>(a, promote_scalar<typename A::value_type>(b),
                            build);
  } else {
    using value_type = typename A::value_type;
    if constexpr (CLit<A> && CLit<B>) {
      constexpr auto folded = Fold{}(A::value, B::value);
      if constexpr (CArithmetic<value_type>) {
        return Lit<value_type, static_cast<value_type>(folded)>{};
      } else if constexpr (folded == value_type(0)) {
        return Lit<value_type, 0>{};
      } else if constexpr (folded == value_type(1)) {
        return Lit<value_type, 1>{};
      } else {
        return Constant<value_type>{folded};
      }
    } else if constexpr (CConstant<A> && CConstant<B>) {
      return Constant<value_type>{Fold{}(a.get(), b.get())};
    } else {
      return build(a, b);
    }
  }
}

// The node an operator builds once the ladder above has nothing to fold.
template <template <Numeric> class Op>
inline constexpr auto node_of =
    []<CExpression L>(const L &l, const CExpression auto &r) {
      return simplify_node<Op<typename L::value_type>>(l, r);
    };

} // namespace detail

template <typename A, typename B>
  requires CBinaryOperands<A, B>
constexpr auto operator+(const A &a, const B &b) noexcept {
  return detail::arithmetic<std::plus<>>(a, b, detail::node_of<SumOp>);
}
template <typename A, typename B>
  requires CBinaryOperands<A, B>
constexpr auto operator*(const A &a, const B &b) noexcept {
  return detail::arithmetic<std::multiplies<>>(a, b,
                                               detail::node_of<MultiplyOp>);
}
template <typename A, typename B>
  requires CBinaryOperands<A, B>
constexpr auto operator/(const A &a, const B &b) noexcept {
  return detail::arithmetic<std::divides<>>(a, b, detail::node_of<DivideOp>);
}
// a - b is a + (-b): one adjoint rule instead of two, and both operators get to
// apply their folding rules.
template <typename A, typename B>
  requires CBinaryOperands<A, B>
constexpr auto operator-(const A &a, const B &b) noexcept {
  return detail::arithmetic<std::minus<>>(
      a, b, []<CExpression L>(const L &l, const CExpression auto &r) {
        return detail::simplify_node<SumOp<typename L::value_type>>(l, -r);
      });
}

template <CExpression Expr> constexpr auto operator-(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  if constexpr (CLit<Expr> && CArithmetic<value_type>) {
    return Lit<value_type,
               static_cast<value_type>(-std::remove_cvref_t<Expr>::value)>{};
  } else {
    return detail::simplify_mono<NegateOp<value_type>>(a);
  }
}

// One factory per unary math function, generated from the registry.
#define DDX_EXPR_UNFN(FN, OP, LABEL)                                           \
  template <CExpression Expr> constexpr auto FN(const Expr &a) noexcept {      \
    return MonoExpression<OP<typename Expr::value_type>, Expr>{a};             \
  }
DDX_UNARY_MATH_TABLE(DDX_EXPR_UNFN)
// abs has no descriptor, so its factory is spelled out.
DDX_EXPR_UNFN(abs, AbsOp, "abs")
#undef DDX_EXPR_UNFN

// Function-style binary ops, plus scalar-promotion overloads.
#define DDX_EXPR_BINFN(NAME, OP)                                               \
  template <CExpression LHS, CExpression RHS>                                  \
    requires CompatibleValueTypes<LHS, RHS>                                    \
  constexpr auto NAME(const LHS &a, const RHS &b) noexcept {                   \
    using value_type = typename LHS::value_type;                               \
    return Expression<OP<value_type>, LHS, RHS>{a, b};                         \
  }                                                                            \
  template <CArithmetic S, CExpression RHS>                                    \
  constexpr auto NAME(S s, const RHS &b) noexcept {                            \
    return NAME(promote_scalar<typename RHS::value_type>(s), b);               \
  }                                                                            \
  template <CExpression LHS, CArithmetic S>                                    \
  constexpr auto NAME(const LHS &a, S s) noexcept {                            \
    return NAME(a, promote_scalar<typename LHS::value_type>(s));               \
  }
DDX_EXPR_BINFN(pow, PowOp)
DDX_EXPR_BINFN(atan2, Atan2Op)
DDX_EXPR_BINFN(hypot, HypotOp)
DDX_EXPR_BINFN(max, MaxOp)
DDX_EXPR_BINFN(min, MinOp)
#undef DDX_EXPR_BINFN

namespace detail {

// read() is the only member a specialisation supplies.  CRTP rather than a data
// member: the compile-time form has to stay std::is_empty_v.
template <typename Derived, Numeric T>
class ConstantOps : public EquationConvertible<Derived> {
public:
  using value_type = T;

  // A constant is the one thing that evaluates without a point, so its own
  // nullary eval() has to be named alongside the base's pack form.
  using EquationConvertible<Derived>::eval;

  [[nodiscard]] constexpr T get() const noexcept { return this->self().read(); }
  [[nodiscard]] constexpr T eval() const noexcept { return get(); }
  constexpr operator T() const noexcept { return get(); }

  // The int spelling of Lit works for every Numeric T.
  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Lit<T, 0>{};
  }

  template <std::size_t Base = 0>
  constexpr void backward(const auto &, T, auto &,
                          const auto &) const noexcept {}

  // At its own type the value passes through; deeper, ConstantEmbedder<U>
  // embeds it with zero dual parts.
  template <CSymbolList Syms, Numeric U, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded(const std::array<U, N> &) const noexcept {
    if constexpr (std::same_as<U, T>) {
      return get();
    } else {
      using S = scalar_base_t<U>;
      return ConstantEmbedder<U>::embed(
          static_cast<S>(get_real_part<dual_depth_v<T>>(get())));
    }
  }

  template <std::size_t I> [[nodiscard]] constexpr auto get() const noexcept {
    static_assert(I < 2);
    if constexpr (CTupleLike<T>) {
      return eval().template get<I>();
    } else if constexpr (I == 0) {
      return eval();
    } else {
      return static_cast<T>(derivative());
    }
  }
};

} // namespace detail

// Carried in the type, so the object is empty and derivative()'s 0s and 1s are
// free.  V is a T, or an int for a T that could never be a template argument
// itself.
template <Numeric T, auto V>
  requires std::same_as<std::remove_cv_t<decltype(V)>, T> ||
           std::same_as<std::remove_cv_t<decltype(V)>, int>
class Lit<T, V> : public detail::ConstantOps<Lit<T, V>, T> {
  friend detail::ConstantOps<Lit<T, V>, T>;
  [[nodiscard]] constexpr T read() const noexcept { return value; }

public:
  static constexpr T value = T(V);
};

// The storing form: one type per value_type, so it can hold a runtime number.
template <Numeric T> class Lit<T> : public detail::ConstantOps<Lit<T>, T> {
  friend detail::ConstantOps<Lit<T>, T>;
  T value;
  [[nodiscard]] constexpr T read() const noexcept { return value; }

public:
  constexpr explicit Lit(T value) noexcept : value(value) {}
};

template <Numeric T> Lit(T) -> Lit<T>;

// Frozen: still reads its slot from the seed array, but differentiates to
// zero.  Which of the two reasons it is frozen for matters only to the
// rewrites in traits.hpp; every engine here asks `frozen`.
template <Numeric T, CFixedString auto symbol, Freeze Kind>
class Variable : public EquationConvertible<Variable<T, symbol, Kind>> {
public:
  static constexpr auto label = symbol;
  static constexpr Freeze freeze = Kind;
  static constexpr bool frozen = Kind != Freeze::none;
  using value_type = T;

  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Lit<T, frozen ? 0 : 1>{};
  }

  template <std::size_t Base = 0>
  constexpr void backward(const auto &syms, T adj, auto &grads,
                          const auto &) const noexcept {
    if constexpr (!frozen) {
      using Syms = std::decay_t<decltype(syms)>;
      constexpr auto idx = symbol_index<symbol, Syms>();
      static_assert(idx < mp::mp_size<Syms>::value,
                    "backward: this symbol is not in the list swept");
      // `+` and assignment, not `+=`: CFieldLike promises only a + b.  Do not
      // "simplify" this back to `+=`.
      grads[idx] = std::move(grads[idx]) + adj;
    }
  }

  // A frozen symbol takes the seed's value and drops its derivative slots.
  // All three engines guard on Frozen or they disagree.
  template <CSymbolList Syms, Numeric U, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded(const std::array<U, N> &vals) const noexcept {
    constexpr auto idx = symbol_index<symbol, Syms>();
    static_assert(idx < N, "eval: no value supplied for this symbol");
    if constexpr (frozen) {
      return ConstantEmbedder<U>::embed(
          get_real_part<dual_depth_v<U>>(vals[idx]));
    } else {
      return vals[idx];
    }
  }
};

#define DEFINE_CONST_UDL(type, suffix)                                         \
  consteval ddx::impl::Constant<type> operator""_##suffix(                     \
      unsigned long long val) {                                                \
    return ddx::impl::Constant<type>{static_cast<type>(val)};                  \
  }                                                                            \
  consteval ddx::impl::Constant<type> operator""_##suffix(long double val) {   \
    return ddx::impl::Constant<type>{static_cast<type>(val)};                  \
  }

// The literal's value is unused; only its type selects value_type.
#define DEFINE_VAR_UDL(type, suffix, label)                                    \
  consteval auto operator""_##suffix(unsigned long long) {                     \
    return ddx::impl::Variable<type, ddx::impl::FixedString{label}>{};         \
  }                                                                            \
  consteval auto operator""_##suffix(long double) {                            \
    return ddx::impl::Variable<type, ddx::impl::FixedString{label}>{};         \
  }

// var<"x"> — name a symbol.
template <FixedString S, Numeric T = double>
inline constexpr Variable<T, S> var{};

// var_of<"x">(v) — the exemplar supplies the type and is never read, so a
// run-time value names a symbol as well as a literal does.
template <FixedString S, Numeric T>
[[nodiscard]] constexpr auto var_of(const T &) noexcept {
  return Variable<T, S>{};
}

// The same two, keyed by a symbol in hand rather than a template argument:
//
//   variable("x"_s)              // Variable<double, "x">
//   variable<dual>("x"_s)        // and over another scalar
//   var_of("x"_s, v)             // exemplar supplies the scalar
template <Numeric T = double, FixedString S>
[[nodiscard]] constexpr auto variable(symbol_type<S>) noexcept {
  return Variable<T, S>{};
}

template <FixedString S, Numeric T>
[[nodiscard]] constexpr auto var_of(symbol_type<S>, const T &) noexcept {
  return Variable<T, S>{};
}

// constant(3.0) — a value stored in the tree; a bare scalar promotes on its
// own.
template <Numeric T> [[nodiscard]] constexpr auto constant(T v) noexcept {
  return Lit<T>{v};
}

} // namespace ddx::impl

// In namespace literals, next to "x"_s: `using namespace ddx::literals;`.
namespace ddx::impl::literals {

DEFINE_CONST_UDL(int, ci)
DEFINE_CONST_UDL(double, cd)
DEFINE_VAR_UDL(int, vi, "c")
DEFINE_VAR_UDL(double, vd, "v")

} // namespace ddx::impl::literals

#undef DEFINE_CONST_UDL
#undef DEFINE_VAR_UDL

namespace std {
template <ddx::impl::Numeric T, auto... V>
struct tuple_size<ddx::impl::Lit<T, V...>> : integral_constant<std::size_t, 2> {
};

template <std::size_t I, ddx::impl::Numeric T, auto... V>
struct tuple_element<I, ddx::impl::Lit<T, V...>> {
  using type = typename ddx::impl::detail::expression_element<T, I>::type;
};

template <ddx::impl::Numeric T, ddx::impl::CFixedString auto C,
          ddx::impl::Freeze F>
struct tuple_size<ddx::impl::Variable<T, C, F>>
    : integral_constant<std::size_t, 2> {};

template <std::size_t I, ddx::impl::Numeric T, ddx::impl::CFixedString auto C,
          ddx::impl::Freeze F>
struct tuple_element<I, ddx::impl::Variable<T, C, F>> {
  using type = typename ddx::impl::detail::expression_element<T, I>::type;
};
} // namespace std
