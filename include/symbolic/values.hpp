#pragma once
#include "ops/operations.hpp"
#include "ops/scalar.hpp"
#include "symbolic/expressions.hpp"
#include "symbolic/simplify.hpp"
#include "symbolic/symbol.hpp"
#include "util/meta.hpp"
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

// Promotes a bare scalar to a zero-derivative constant of value_type, at any
// Dual nesting depth.
template <Numeric VT, CArithmetic S>
constexpr Constant<VT> promote_scalar(S s) noexcept {
  return Constant<VT>{
      ConstantEmbedder<VT>::embed(static_cast<scalar_base_t<VT>>(s))};
}

// Two expressions of compatible value type, or an expression and a bare scalar
// on either side.
template <typename A, typename B>
concept CBinaryOperands =
    (CExpression<A> && CExpression<B> && CompatibleValueTypes<A, B>) ||
    (CExpression<A> && CArithmetic<B>) || (CArithmetic<A> && CExpression<B>);

namespace detail {

// Promotes a bare scalar and folds two literals or two constants; anything
// else goes to build. A dual scalar can only fold back to the literals 0 and 1.
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
    // A literal zero divisor is left to the graph, since x / 0 is not a
    // constant expression.
    if constexpr (CLit<A> && CLit<B> &&
                  !(std::same_as<Fold, std::divides<>> && is_zero_v<B>)) {
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

// The node an operator builds when arithmetic() has nothing to fold.
template <template <Numeric> class Op>
inline constexpr auto node_of =
    []<CExpression L>(const L &l, const CExpression auto &r) {
      return simplify_node<Op<typename L::value_type>>(l, r);
    };

} // namespace detail

// c ? a : b, which folds to zero when both arms are zero.
template <CExpression C, CExpression A, CExpression B>
[[nodiscard]] constexpr auto selected(const C &c, const A &a,
                                      const B &b) noexcept {
  if constexpr (detail::is_zero_v<A> && detail::is_zero_v<B>) {
    return a;
  } else {
    return Expression<SelectOp<typename C::value_type>, C, A, B>{c, a, b};
  }
}

// x == 0 spelled as (x <= 0) * (0 <= x), as the graph does, so NaN gives 0.
template <CExpression E>
[[nodiscard]] constexpr auto zero_test(const E &e) noexcept {
  using T = typename E::value_type;
  const Lit<T, 0> zero{};
  return Expression<LeOp<T>, E, Lit<T, 0>>{e, zero} *
         Expression<LeOp<T>, Lit<T, 0>, E>{zero, e};
}

// Returns d, guarded to be exactly zero where a and factor both vanish. A
// nonzero literal or an unordered scalar needs no guard.
template <CExpression A, CExpression F, CExpression D>
[[nodiscard]] constexpr auto unless_vanishing(const A &a, const F &factor,
                                              const D &d) noexcept {
  using T = typename A::value_type;
  if constexpr (detail::is_zero_v<D> || !std::totally_ordered<T> ||
                (CLit<A> && !detail::is_zero_v<A>) ||
                (CLit<F> && !detail::is_zero_v<F>)) {
    return d;
  } else {
    return selected(zero_test(a) * zero_test(factor), Lit<T, 0>{}, d);
  }
}

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
// a - b is built as a + (-b), so one adjoint rule serves both and both
// operators apply their folding.
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

// One factory per unary math table row.
#define DDX_EXPR_UNFN(FN, OP, LABEL)                                           \
  template <CExpression Expr> constexpr auto FN(const Expr &a) noexcept {      \
    return MonoExpression<OP<typename Expr::value_type>, Expr>{a};             \
  }
DDX_UNARY_MATH_TABLE(DDX_EXPR_UNFN)
// abs is outside the math table, so its factory is spelled out.
DDX_EXPR_UNFN(abs, AbsOp, "abs")
#undef DDX_EXPR_UNFN

// Binary math functions, each with overloads that promote a bare scalar.
#define DDX_EXPR_BINFN(NAME, OP)                                               \
  template <CExpression LHS, CExpression RHS>                                  \
    requires CompatibleValueTypes<LHS, RHS>                                    \
  constexpr auto NAME(const LHS &a, const RHS &b) noexcept {                   \
    return detail::simplify_node<OP<typename LHS::value_type>>(a, b);          \
  }                                                                            \
  template <CArithmetic S, CExpression RHS>                                    \
  constexpr auto NAME(S s, const RHS &b) noexcept {                            \
    return NAME(promote_scalar<typename RHS::value_type>(s), b);               \
  }                                                                            \
  template <CExpression LHS, CArithmetic S>                                    \
  constexpr auto NAME(const LHS &a, S s) noexcept {                            \
    return NAME(a, promote_scalar<typename LHS::value_type>(s));               \
  }
#define DDX_EXPR_BINROW(fn, Name, ...) DDX_EXPR_BINFN(fn, Name##Op)
DDX_BINARY_MATH_TABLE(DDX_EXPR_BINROW)
#undef DDX_EXPR_BINROW
#undef DDX_EXPR_BINFN

namespace detail {

// Shared constant members; a specialisation supplies only read(). CRTP keeps
// the compile-time Lit an empty type.
template <typename Derived, Numeric T>
class ConstantOps : public EquationConvertible<Derived> {
public:
  using value_type = T;

  // A constant evaluates without a point, so its nullary eval() sits beside
  // the base's pack form.
  using EquationConvertible<Derived>::eval;

  [[nodiscard]] constexpr T get() const noexcept { return this->self().read(); }
  [[nodiscard]] constexpr T eval() const noexcept { return get(); }
  constexpr operator T() const noexcept { return get(); }

  // Lit with an int value works for every Numeric T.
  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Lit<T, 0>{};
  }

  // At its own type the value passes through; at a deeper dual it is embedded
  // with zero dual parts.
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
    if constexpr (meta::CTupleLike<T>) {
      return eval().template get<I>();
    } else if constexpr (I == 0) {
      return eval();
    } else {
      return static_cast<T>(derivative());
    }
  }
};

} // namespace detail

// A literal carried in the type, so the object is empty. V is a T, or an int
// when T cannot be a template argument.
template <Numeric T, auto V>
  requires std::same_as<std::remove_cv_t<decltype(V)>, T> ||
           std::same_as<std::remove_cv_t<decltype(V)>, int>
class Lit<T, V> : public detail::ConstantOps<Lit<T, V>, T> {
  friend detail::ConstantOps<Lit<T, V>, T>;
  [[nodiscard]] constexpr T read() const noexcept { return value; }

public:
  static constexpr T value = T(V);
};

// A literal stored as a member, so it can hold a run-time number.
template <Numeric T> class Lit<T> : public detail::ConstantOps<Lit<T>, T> {
  friend detail::ConstantOps<Lit<T>, T>;
  T value;
  [[nodiscard]] constexpr T read() const noexcept { return value; }

public:
  constexpr explicit Lit(T value) noexcept : value(value) {}
};

template <Numeric T> Lit(T) -> Lit<T>;

// A symbol. A frozen one still reads its value from the point but
// differentiates to zero.
template <Numeric T, CFixedString auto symbol, Freeze Kind>
class Variable : public EquationConvertible<Variable<T, symbol, Kind>> {
public:
  static constexpr auto label = symbol;
  static constexpr Freeze freeze = Kind;
  static constexpr bool frozen = Kind != Freeze::none;
  using value_type = T;
  template <Freeze K> using refrozen = Variable<T, symbol, K>;

  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Lit<T, frozen ? 0 : 1>{};
  }

  // A frozen symbol keeps the seed's value and drops its derivative parts.
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

// The literal's value is unused; only its type picks value_type.
#define DEFINE_VAR_UDL(type, suffix, label)                                    \
  consteval auto operator""_##suffix(unsigned long long) {                     \
    return ddx::impl::Variable<type, ddx::impl::FixedString{label}>{};         \
  }                                                                            \
  consteval auto operator""_##suffix(long double) {                            \
    return ddx::impl::Variable<type, ddx::impl::FixedString{label}>{};         \
  }

template <FixedString S, Numeric T = double>
inline constexpr Variable<T, S> var{};

// var_of<"x">(v) takes its type from v and never reads v, so a run-time value
// works as well as a literal.
template <FixedString S, Numeric T>
[[nodiscard]] constexpr auto var_of(const T &) noexcept {
  return Variable<T, S>{};
}

// The same two keyed by a symbol value: variable("x"_s), variable<dual>("x"_s)
// and var_of("x"_s, v).
template <Numeric T = double, FixedString S>
[[nodiscard]] constexpr auto variable(symbol_type<S>) noexcept {
  return Variable<T, S>{};
}

template <FixedString S, Numeric T>
[[nodiscard]] constexpr auto var_of(symbol_type<S>, const T &) noexcept {
  return Variable<T, S>{};
}

// constant(3.0) stores a value in the tree; a bare scalar is promoted anyway.
template <Numeric T> [[nodiscard]] constexpr auto constant(T v) noexcept {
  return Lit<T>{v};
}

} // namespace ddx::impl

// Reached with `using namespace ddx::literals;`, like "x"_s.
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
  using type = DDX_TYPE(ddx::impl::detail::expression_element<T, I>());
};

template <ddx::impl::Numeric T, ddx::impl::CFixedString auto C,
          ddx::impl::Freeze F>
struct tuple_size<ddx::impl::Variable<T, C, F>>
    : integral_constant<std::size_t, 2> {};

template <std::size_t I, ddx::impl::Numeric T, ddx::impl::CFixedString auto C,
          ddx::impl::Freeze F>
struct tuple_element<I, ddx::impl::Variable<T, C, F>> {
  using type = DDX_TYPE(ddx::impl::detail::expression_element<T, I>());
};
} // namespace std
