#pragma once
#include "symbolic/values.hpp"
// include-cleaner misses CFixedString used as a constrained auto parameter.
#include "symbolic/symbol.hpp"
#include "util/fixed_string.hpp" // IWYU pragma: keep
#include "util/meta.hpp"
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
    return meta::unpack(e.expressions(), [](const auto &...child) {
      return Expression<typename T::op_type,
                        decltype(refreeze<Hold, symbol>(child))...>{
          refreeze<Hold, symbol>(child)...};
    });
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

// The union of symbol lists, alphabetical. The sort is stable, so unique
// keeps the first of equal names without reordering the rest.
consteval auto symbol_set(CSymbolList auto... lists) noexcept {
  return meta::unique(meta::sort(meta::concat(lists...), [](auto a, auto b) {
    return DDX_TYPE(a)::name < DDX_TYPE(b)::name;
  }));
}

consteval auto symbols_of(auto t) noexcept {
  using T = DDX_TYPE(t);
  if constexpr (CVariable<T>) {
    return meta::list_c<symbol_type<T::label>>;
  } else if constexpr (CExpressionNode<T>) {
    return meta::unpack(children(t),
                        [](auto... c) { return symbol_set(symbols_of(c)...); });
  } else {
    return meta::list_c<>;
  }
}

// An expression's canonical symbol list.
template <CExpression E>
using expr_symbols_t =
    meta::list_t<symbols_of(meta::type_c<std::remove_cvref_t<E>>)>;

// Whether any variable in the tree is frozen.
consteval bool holds_frozen(auto t) noexcept {
  using T = DDX_TYPE(t);
  if constexpr (CVariable<T>) {
    return T::frozen;
  } else if constexpr (CExpressionNode<T>) {
    return meta::unpack(children(t),
                        [](auto... c) { return (holds_frozen(c) || ...); });
  } else {
    return false;
  }
}

namespace detail {

// An expression's symbol count.
template <CExpression Expr>
inline constexpr std::size_t expr_arity_v = meta::size(expr_symbols_t<Expr>{});

} // namespace detail

} // namespace ddx::impl
