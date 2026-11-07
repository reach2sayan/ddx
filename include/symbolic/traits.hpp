#pragma once
#include "symbolic/values.hpp"
// include-cleaner misses CFixedString used as a constrained auto parameter.
#include "symbolic/symbol.hpp"
#include "util/fixed_string.hpp" // IWYU pragma: keep
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace ddx::impl {

// Rebuilds the tree with each variable's Freeze reset, as a pure type change.
// With Hold, every symbol but `symbol` is held; without it, all are released.
template <bool Hold, CFixedString auto symbol = FixedString{""}, CExpression T>
constexpr auto refreeze(const T &e) noexcept {
  if constexpr (CVariable<T>) {
    return typename T::template refrozen<(
        Hold && T::label != symbol ? Freeze::partial : Freeze::none)>{};
  } else if constexpr (CExpressionNode<T>) {
    return std::apply(
        [](const auto &...child) {
          return Expression<typename T::op_type,
                            decltype(refreeze<Hold, symbol>(child))...>{
              refreeze<Hold, symbol>(child)...};
        },
        e.expressions());
  } else {
    return e;
  }
}

template <CFixedString auto symbol, CExpression E>
constexpr auto make_all_constant_except(const E &e) noexcept {
  return refreeze<true, symbol>(e);
}

// Releases the holds used to form a partial, so d/dy of a stored d/dx is not
// zero.
template <CExpression E> constexpr auto thaw_partials(const E &e) noexcept {
  return refreeze<false>(e);
}

// Orders symbols alphabetically, as a metafunction for mp_sort.
template <CSymbol A, CSymbol B>
struct symbol_less : std::bool_constant<(A::name < B::name)> {};

// mp_sort is stable, so duplicate names, which mp_unique then collapses, do
// not reorder the rest.
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

// Whether any variable in the tree is frozen.
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

// An expression's canonical symbol list and its length.
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
