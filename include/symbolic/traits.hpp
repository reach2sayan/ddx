#pragma once
#include "symbolic/values.hpp"
// CFixedString appears only as a constrained-auto NTTP placeholder, which
// include-cleaner does not count as a reference -- hence the pragma.
#include "symbolic/symbol.hpp"
#include "util/fixed_string.hpp" // IWYU pragma: keep
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace ddx::impl {

// Same leaf, same value lookup, zero derivative.  A pure type transform, which
// keeps the symbolic Jacobian made of empty types.
template <Freeze K, CVariable T> struct refrozen_variable;
template <Freeze K, Numeric T, CFixedString auto C, Freeze F>
struct refrozen_variable<K, Variable<T, C, F>> {
  using type = Variable<T, C, K>;
};
template <Freeze K, CVariable T>
using refrozen_variable_t = typename refrozen_variable<K, T>::type;

// The three leaf rewrites.  `hold_match` is the caller holding one symbol
// constant, `hold_others` is one Jacobian column, `lift_partials` undoes the
// second and never the first.
enum class Rewrite : std::uint8_t { hold_match, hold_others, lift_partials };

template <Rewrite R>
consteval Freeze next_freeze(Freeze now, bool match) noexcept {
  if constexpr (R == Rewrite::hold_match) {
    return match ? Freeze::held : now;
  } else if constexpr (R == Rewrite::hold_others) {
    // A held symbol stays held either way: only the machinery's own hold is
    // reversible.
    if (match) {
      return now == Freeze::partial ? Freeze::none : now;
    }
    return now == Freeze::none ? Freeze::partial : now;
  } else {
    return now == Freeze::partial ? Freeze::none : now;
  }
}

// Rebuilding a node from rewritten children, leaving any other leaf alone, is
// one walk, and a pure type transform, so the symbolic Jacobian stays made of
// empty types.
template <Rewrite R, CFixedString auto symbol = FixedString{""}, CExpression T>
constexpr auto refreeze(const T &e) noexcept {
  if constexpr (CVariable<T>) {
    constexpr Freeze kind = next_freeze<R>(T::freeze, T::label == symbol);
    if constexpr (kind == T::freeze) {
      return e;
    } else {
      return refrozen_variable_t<kind, T>{};
    }
  } else if constexpr (CExpressionNode<T>) {
    return std::apply(
        [](const auto &...child) {
          return Expression<typename T::op_type,
                            decltype(refreeze<R, symbol>(child))...>{
              refreeze<R, symbol>(child)...};
        },
        e.expressions());
  } else {
    return e;
  }
}

template <CFixedString auto symbol, CExpression E>
constexpr auto make_const_variable(const E &e) noexcept {
  return refreeze<Rewrite::hold_match, symbol>(e);
}

template <CFixedString auto symbol, CExpression E>
constexpr auto make_all_constant_except(const E &e) noexcept {
  return refreeze<Rewrite::hold_others, symbol>(e);
}

// A partial tree is differentiable again: what held the other symbols while it
// was formed is lifted, so d/dy of a stored d/dx is not zero.
template <CExpression E> constexpr auto thaw_partials(const E &e) noexcept {
  return refreeze<Rewrite::lift_partials>(e);
}

// Alphabetical by name; a metafunction because that is what mp_sort takes.
template <CSymbol A, CSymbol B>
struct symbol_less : std::bool_constant<(A::name < B::name)> {};

// mp_sort is stable, so a tie -- two symbols of the same name, collapsed by
// mp_unique below -- does not reorder the rest.
template <CSymbolList List> using sort_tuple_t = mp::mp_sort<List, symbol_less>;

template <CSymbolList List>
using unique_tuple_t = mp::mp_unique<sort_tuple_t<List>>;

template <CSymbolList... Lists>
using tuple_union_t = unique_tuple_t<mp::mp_append<Lists...>>;

template <CExpression T> consteval auto extract_symbols_impl() {
  if constexpr (CVariable<T>) {
    return std::type_identity<mp::mp_list<symbol_type<T::label>>>{};
  } else if constexpr (CExpressionNode<T>) {
    return []<COperation Op, CExpression... C>(
               std::type_identity<Expression<Op, C...>>) {
      return std::type_identity<tuple_union_t<
          typename decltype(extract_symbols_impl<C>())::type...>>{};
    }(std::type_identity<T>{});
  } else {
    return std::type_identity<mp::mp_list<>>{};
  }
}

template <CExpression T>
using extract_symbols_from_expr_t =
    typename decltype(extract_symbols_impl<T>())::type;

// Whether any leaf differentiates to zero, for either reason.
template <CExpression T> consteval bool holds_frozen() {
  if constexpr (CVariable<T>) {
    return T::frozen;
  } else if constexpr (CExpressionNode<T>) {
    return []<COperation Op, CExpression... C>(
               std::type_identity<Expression<Op, C...>>) {
      return (holds_frozen<C>() || ...);
    }(std::type_identity<T>{});
  } else {
    return false;
  }
}

namespace detail {

// The two things every driver asks of an expression: its canonical symbol list
// and how long that is.
template <CExpression Expr>
using expr_symbols_t = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;

template <CExpression Expr>
inline constexpr std::size_t expr_arity_v =
    mp::mp_size<expr_symbols_t<Expr>>::value;

} // namespace detail

template <std::size_t N> using idx_t = std::integral_constant<std::size_t, N>;

template <std::size_t N> [[nodiscard]] consteval idx_t<N> idx() noexcept {
  return {};
}

} // namespace ddx::impl
