#pragma once

#include "ops/numeric.hpp"       // Numeric, COperation, CExpression
#include "symbolic/symbol.hpp"   // CSymbol, CSymbolList
#include "util/fixed_string.hpp" // FixedString, CFixedString

#include <concepts>
#include <cstdint>
#include <ranges>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ddx::impl {

template <FixedString S, typename V>
  requires std::is_object_v<V>
struct Entry;

// A symbol name lifted to a type, used as a symbol-list element.
template <auto S> struct symbol_type {
  static constexpr auto value = S;
  static constexpr std::string_view name = S.view();

  // "x"_s = 1.5 makes one entry of a point. The symbol has no state, so this
  // assignment builds an Entry instead of modifying anything.
  template <typename V>
    requires std::is_object_v<V>
  constexpr Entry<S, V> operator=(V v) const;
};

// The symbol as a value, for operator[].
template <FixedString S> inline constexpr symbol_type<S> sym{};

namespace literals {
template <FixedString S> [[nodiscard]] consteval auto operator""_s() noexcept {
  return sym<S>;
}

} // namespace literals

// partial holds the other symbols constant while one Jacobian column is taken,
// then is lifted, or the stored partial would differentiate to zero.
enum class Freeze : std::uint8_t { none, partial };

template <Numeric T, CFixedString auto, Freeze = Freeze::none> class Variable;

template <typename T> inline constexpr bool is_variable_v = false;
template <Numeric T, CFixedString auto C, Freeze F>
inline constexpr bool is_variable_v<Variable<T, C, F>> = true;

template <typename T>
concept CVariable = is_variable_v<std::remove_cvref_t<T>>;

// Unconstrained so that rt can specialise it; the compile-time definition
// carries its own CExpression constraint.
template <typename... Ts> class Equation;

template <Numeric T, auto... V>
inline constexpr bool is_expression_type_v<Lit<T, V...>> = true;

template <Numeric T, CFixedString auto C, Freeze F>
inline constexpr bool is_expression_type_v<Variable<T, C, F>> = true;

template <COperation Op, CExpression... Children>
inline constexpr bool is_expression_type_v<Expression<Op, Children...>> = true;

template <bool S, COperation Op, CExpression... Children>
inline constexpr bool is_expression_type_v<ExpressionImpl<S, Op, Children...>> =
    true;

template <typename T> inline constexpr bool is_constant_v = false;
template <Numeric T, auto... V>
inline constexpr bool is_constant_v<Lit<T, V...>> = true;
template <typename T>
concept CConstant = is_constant_v<std::remove_cvref_t<T>>;

template <typename T> inline constexpr bool is_lit_v = false;
template <Numeric T, auto V> inline constexpr bool is_lit_v<Lit<T, V>> = true;

template <typename T>
concept CLit = is_lit_v<std::remove_cvref_t<T>>;

// A node has children_t and a leaf does not.
template <typename T>
concept CExpressionNode =
    requires { typename std::remove_cvref_t<T>::children_t; };

template <CExpression T> consteval std::size_t node_count_fn() {
  using U = std::remove_cvref_t<T>;
  if constexpr (CExpressionNode<U>) {
    return
        []<CExpression... C>(std::type_identity<std::tuple<C...>>) consteval {
          return (1 + ... + node_count_fn<std::remove_cvref_t<C>>());
        }(std::type_identity<typename U::children_t>{});
  } else {
    return 1;
  }
}

template <CExpression T>
inline constexpr std::size_t node_count_v =
    node_count_fn<std::remove_cvref_t<T>>();

// One argument of a point: a number, a range, a ValueMap, or a named value.
template <typename T>
concept CEvalArg = Numeric<T> || std::ranges::input_range<T> || requires {
  typename std::remove_cvref_t<T>::symbols;
} || requires { std::remove_cvref_t<T>::symbol; };

namespace detail {
template <CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto eval_dispatch(const Expr &e,
                                           const Args &...args) noexcept;

template <auto Seed, CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto tangent_dispatch(const Expr &e,
                                              const Args &...args);
} // namespace detail

// The conversion is defined in equation.hpp. It stays a constrained template,
// since a plain one would recurse through every is_convertible query.
template <typename Derived> struct EquationConvertible {
  template <typename Eq>
    requires std::same_as<Eq, Equation<Derived>>
  constexpr operator Eq() const noexcept;

  // The tree stores no values, so evaluation always takes a point.
  [[nodiscard]] constexpr auto eval(const CEvalArg auto &...args) const {
    return detail::eval_dispatch(self(), args...);
  }

  template <FixedString Seed>
  [[nodiscard]] constexpr auto
  eval_with_tangent(const CEvalArg auto &...args) const {
    return detail::tangent_dispatch<Seed>(self(), args...);
  }

protected:
  [[nodiscard]] constexpr const Derived &self() const noexcept {
    return static_cast<const Derived &>(*this);
  }
};

template <CExpression Derived, COperation Op>
class ExpressionOps : public EquationConvertible<Derived> {
public:
  using op_type = Op;
  using value_type = typename Op::value_type;

  [[nodiscard]] constexpr auto derivative() const noexcept {
    return std::apply(
        [](const auto &...e) noexcept { return Op::derivative(e...); },
        this->self().expressions());
  }

  // The seeded sweep. U is deduced from the point, so the same tree evaluates
  // at a plain scalar or a dual.
  template <CSymbolList Syms, Numeric U, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded(const std::array<U, N> &vals) const noexcept {
    return std::apply(
        [&](const auto &...e) noexcept {
          return Op::eval(e.template eval_seeded<Syms>(vals)...);
        },
        this->self().expressions());
  }
};

// The CRTP parameter is the public Expression type, not this base.
template <COperation Op, CExpression... Children>
class ExpressionImpl<true, Op, Children...>
    : public ExpressionOps<Expression<Op, Children...>, Op> {
  friend ExpressionOps<Expression<Op, Children...>, Op>;

public:
  using children_t = std::tuple<Children...>;

  constexpr ExpressionImpl() noexcept = default;
  constexpr ExpressionImpl(Children...) noexcept {}

  [[nodiscard]] constexpr children_t expressions() const noexcept { return {}; }
};

template <COperation Op, CExpression... Children>
class ExpressionImpl<false, Op, Children...>
    : public ExpressionOps<Expression<Op, Children...>, Op> {
  std::tuple<Children...> operands;
  friend ExpressionOps<Expression<Op, Children...>, Op>;

public:
  using children_t = std::tuple<Children...>;

  constexpr ExpressionImpl(Children... c) noexcept
      : operands{std::move(c)...} {}
  [[nodiscard]] constexpr const children_t &expressions() const noexcept {
    return operands;
  }
};

// Derives rather than aliases, so partial specialisations can match Expression.
template <COperation Op, CExpression... Children>
class Expression : public ExpressionImpl<(std::is_empty_v<Children> && ...), Op,
                                         Children...> {
  using base_type =
      ExpressionImpl<(std::is_empty_v<Children> && ...), Op, Children...>;

public:
  using base_type::base_type;
  using children_t = typename base_type::children_t;
  static constexpr bool stateless = (std::is_empty_v<Children> && ...);
};

namespace detail {
template <Numeric V, std::size_t I, typename = void> struct expression_element {
  using type = V;
};

template <Numeric V, std::size_t I>
struct expression_element<V, I,
                          std::void_t<typename std::tuple_element_t<I, V>>> {
  using type = std::tuple_element_t<I, V>;
};
} // namespace detail

} // namespace ddx::impl

#define DDX_COMMUTATIVE_MULTIPLY(...)                                          \
  namespace ddx::impl {                                                        \
  template <>                                                                  \
  inline constexpr bool is_commutative_multiply_v<__VA_ARGS__> = true;         \
  }

namespace std {
template <ddx::impl::COperation Op, ddx::impl::CExpression... Children>
struct tuple_size<ddx::impl::Expression<Op, Children...>>
    : integral_constant<size_t, 2> {};

template <size_t I, ddx::impl::COperation Op,
          ddx::impl::CExpression... Children>
struct tuple_element<I, ddx::impl::Expression<Op, Children...>> {
  using type = typename ddx::impl::detail::expression_element<
      typename ddx::impl::Expression<Op, Children...>::value_type, I>::type;
};
} // namespace std
