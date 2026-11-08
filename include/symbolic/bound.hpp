#pragma once

#include "symbolic/entry.hpp"
#include "symbolic/traits.hpp"
#include "util/config.hpp" // DDX_FWD
#include "util/error.hpp"
#include "util/fixed_string.hpp"
#include "util/meta.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ddx::impl {

// A compile-time map from symbol name to value, with one slot per symbol.
template <Numeric Scalar, CSymbolList SymList> struct ValueMap {
  using symbols = SymList;
  using value_type = Scalar;

  static constexpr bool kValueMap = true;
  static constexpr std::size_t arity = meta::size(SymList{});

  std::array<Scalar, arity> slots{};

  // Read or write a slot as get<"x">() or m["x"_s].
  template <FixedString S>
  [[nodiscard]] constexpr decltype(auto) get(this auto &&self) noexcept {
    constexpr auto idx = symbol_index<S, SymList>();
    static_assert(idx < arity, "ValueMap: symbol not present in map");
    if constexpr (std::is_lvalue_reference_v<decltype(self)>) {
      return DDX_FWD(self).slots[idx];
    } else {
      return Scalar{self.slots[idx]};
    }
  }
  template <CFixedString auto S>
  [[nodiscard]] constexpr decltype(auto) operator[](this auto &&self,
                                                    symbol_type<S>) noexcept {
    return DDX_FWD(self).template get<S>();
  }
  template <FixedString S> constexpr void set(const Scalar &v) noexcept {
    get<S>() = v;
  }
};

template <typename T>
concept CValueMap = requires {
  typename std::remove_cvref_t<T>::symbols;
  requires std::remove_cvref_t<T>::kValueMap;
};

namespace detail {

// For each canonical slot, the argument position that supplies it.
template <CSymbolList SymList, std::size_t N, CSymbol... ArgSyms>
consteval std::array<std::size_t, N> arg_of_canonical() noexcept {
  const std::array<std::size_t, N> canonical_of_arg{
      symbol_index<ArgSyms::value, SymList>()...};
  std::array<std::size_t, N> out{};
  for (const auto [j, slot] : std::views::enumerate(canonical_of_arg)) {
    out[slot] = static_cast<std::size_t>(j);
  }
  return out;
}

} // namespace detail

// Builds a ValueMap, as in values(named<"x">(1.0), named<"y">(0.5)).
template <FixedString... Syms, Numeric... Vs>
[[nodiscard]] constexpr auto values(Entry<Syms, Vs>... nv) noexcept {
  using SymList = decltype(symbol_set(meta::list_c<symbol_type<Syms>...>));
  constexpr std::size_t N = sizeof...(Syms);
  static_assert(meta::size(SymList{}) == N, "values: duplicate symbol");

  using Scalar = std::common_type_t<Vs...>;
  constexpr auto pos =
      detail::arg_of_canonical<SymList, N, symbol_type<Syms>...>();
  const auto args = std::tuple<Vs...>{nv.value...};
  return meta::unpack(meta::iota<N>, [&](auto... i) {
    return ValueMap<Scalar, SymList>{
        std::array<Scalar, N>{static_cast<Scalar>(std::get<pos[i]>(args))...}};
  });
}

// The canonical (alphabetical) symbol order that positional points use.
template <CExpression Expr>
[[nodiscard]] consteval auto symbol_order() noexcept {
  using SymList = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  return meta::unpack(SymList{}, [](auto... s) {
    return std::array<std::string_view, sizeof...(s)>{s.name...};
  });
}
template <CExpression Expr>
[[nodiscard]] consteval auto symbol_order(const Expr &) noexcept {
  return symbol_order<std::remove_cvref_t<Expr>>();
}

namespace detail {

// A point given as a runtime range, the only spelling whose length is checked
// at run time and so the only one that can return an error.
template <typename... Args>
concept CDynamicPoint =
    sizeof...(Args) == 1 && (std::ranges::input_range<Args> && ...) &&
    !(CTupleLike<Args> && ...);
// Copies the first N values of a range and returns how many it saw.
template <Numeric U, std::size_t N>
[[nodiscard]] constexpr std::pair<std::array<U, N>, std::size_t>
take_point(const std::ranges::input_range auto &r) noexcept {
  std::array<U, N> vals{};
  std::size_t i = 0;
  // Reads one past N to detect a range that is too long.
  std::ranges::for_each(r | std::views::take(N + 1), [&](const auto &v) {
    if (i < N) {
      vals[i] = static_cast<U>(v);
    }
    ++i;
  });
  return {vals, i};
}

// Reduces every spelling of a point to N values in canonical symbol order.
template <CSymbolList Syms, Numeric U, std::size_t N, CEvalArg... Args>
  requires(!CDynamicPoint<Args...>)
[[nodiscard]] constexpr std::array<U, N>
make_point(const Args &...args) noexcept {
  // clang-format off
  if constexpr (sizeof...(Args) == 0) { // constant-folded: no free symbols
    static_assert(N == 0,
                  "eval: this expression has free symbols, so it needs a point "
                  "(see symbol_order<Expr>())");
    return {};
  }

  // point(map) or point(named<"x">(..), ..), read by name.
  else if constexpr ((CValueMap<Args> && ...) || (CEntry<Args> && ...)) {
    const auto map = [&] {
      if constexpr ((CValueMap<Args> && ...)) {
        static_assert(sizeof...(Args) == 1, "eval: pass a single ValueMap");
        return std::get<0>(std::tuple{args...});
      } else {
        return values(args...);
      }
    }();
    std::array<U, N> vals{};
    meta::for_each(meta::iota<N>, [&](auto i) {
      vals[i] = static_cast<U>(map.template get<DDX_TYPE(meta::at<i>(Syms{}))::value>());
    });
    return vals;
  }

  // point(tuple-like range), whose size is in the type and checked statically.
  else if constexpr (sizeof...(Args) == 1 && (std::ranges::input_range<Args> && ...)) {
    return [&](const auto &r) {
      static_assert(
          std::tuple_size_v<std::remove_cvref_t<decltype(r)>> == N,
          "eval: range size must equal the expression's symbol count "
          "(see symbol_order<Expr>())");
      return take_point<U, N>(r).first;
    }(args...);
  }

  // point(x, y, z), positional in canonical order.
  else {
    static_assert(sizeof...(Args) == N,
                  "eval: supply exactly one value per symbol, in canonical "
                  "order (see symbol_order<Expr>())");
    return std::array<U, N>{static_cast<U>(args)...};
  }
  //clang-format on
}

// point(range), whose length is only known at run time and may be an error.
template <CSymbolList Syms, Numeric U, std::size_t N, CEvalArg... Args>
  requires(CDynamicPoint<Args...>)
[[nodiscard]] constexpr result<std::array<U, N>>
make_point(const Args &...args) noexcept {
  return [&](const auto &r) -> result<std::array<U, N>> {
    const auto [vals, got] = take_point<U, N>(r);
    if (got < N) {
      return fail(errc::short_point);
    }
    if (got > N) {
      return fail(errc::wrong_arity);
    }
    return vals;
  }(args...);
}

// Runs body on the point array, returning result<T> exactly when the point was
// given as a runtime range. The only place that decision is made.
template <CSymbolList Syms, Numeric U, std::size_t N>
[[nodiscard]] constexpr auto
with_point(auto &&body, const CEvalArg auto &...args) noexcept {
  if constexpr (CDynamicPoint<decltype(args)...>) {
    return make_point<Syms, U, N>(args...).transform(DDX_FWD(body));
  } else {
    return DDX_FWD(body)(make_point<Syms, U, N>(args...));
  }
}

template <CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto eval_dispatch(const Expr &e,
                                           const Args &...args) noexcept {
  using VT = typename std::remove_cvref_t<Expr>::value_type;
  using Syms = expr_symbols_t<Expr>;
  return with_point<Syms, VT, expr_arity_v<Expr>>(
      [&e](const auto &vals) { return e.template eval_seeded<Syms>(vals); },
      args...);
}

// Forward mode is the ordinary seeded sweep over Dual<VT>, with a unit tangent
// on the symbol being differentiated.
template <auto Seed, CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto tangent_dispatch(const Expr &e,
                                              const Args &...args) {
  using VT = typename std::remove_cvref_t<Expr>::value_type;
  using Syms = expr_symbols_t<Expr>;
  constexpr std::size_t N = expr_arity_v<Expr>;
  return with_point<Syms, VT, N>(
      [&e](const std::array<VT, N> &vals) {
        constexpr std::size_t seed = symbol_index<Seed, Syms>();
        const auto seeds = meta::unpack(meta::iota<N>, [&](auto... i) {
          return std::array<Dual<VT>, N>{
              Dual<VT>{vals[i], i == seed ? VT{1} : VT{}}...};
        });
        return e.template eval_seeded<Syms>(seeds);
      },
      args...);
}

} // namespace detail

template <CExpression Expr, CEvalArg... Args>
  requires(sizeof...(Args) > 0)
[[nodiscard]] constexpr auto eval(const Expr &e,
                                  const Args &...args) noexcept {
  return detail::eval_dispatch(e, args...);
}

} // namespace ddx::impl
