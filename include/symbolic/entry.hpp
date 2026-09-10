#pragma once

#include "symbolic/expressions.hpp" // Numeric, symbol_type, Variable
#include "util/fixed_string.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace ddx::impl {

namespace detail {

// sym<"x"> / "x"_s carry it as ::value, var<"x"> as ::label; undefined
// elsewhere, which is what CSymbolTag tests.
template <typename T> struct tag_key_of {};
template <auto S> struct tag_key_of<symbol_type<S>> {
  static constexpr auto value = S;
};
template <Numeric T, CFixedString auto S, Freeze F>
struct tag_key_of<Variable<T, S, F>> {
  static constexpr auto value = S;
};

template <typename T> using tag_key_t = tag_key_of<std::remove_cvref_t<T>>;

} // namespace detail

// A symbol named at the type level: "x"_s, sym<"x">, or var<"x">.
template <typename T>
concept CSymbolTag = requires { detail::tag_key_t<T>::value; };

// One keyword argument: a label bound to a value, the label a template
// parameter.
//
//   named<"x">(1.5)      Entry{"x"_s, 1.5}      Entry{var<"x">, 1.5}
//
// V is any object type: Record keys arbitrary values by this same spelling, and
// the numeric entry points constrain V to Numeric where they take it.
template <FixedString Sym, typename V>
  requires std::is_object_v<V>
struct Entry {
  using value_type = V;
  static constexpr auto symbol = Sym;

  V value;

  constexpr Entry()
    requires std::default_initializable<V>
  = default;

  // Not explicit: Record<...>{{1}, {2.5}} initialises its entries through this.
  constexpr Entry(V v) noexcept(std::is_nothrow_move_constructible_v<V>)
      : value{std::move(v)} {}

  template <CSymbolTag Tag>
    requires(detail::tag_key_t<Tag>::value == Sym)
  constexpr Entry(const Tag &,
                  V v) noexcept(std::is_nothrow_move_constructible_v<V>)
      : value{std::move(v)} {}

  constexpr bool operator==(const Entry &) const = default;
};

template <CSymbolTag Tag, typename V>
Entry(const Tag &, V) -> Entry<detail::tag_key_t<Tag>::value, V>;

// named<"x">(1.25) -- bind a value to a symbol by name.
template <FixedString Sym, typename V>
  requires std::is_object_v<V>
[[nodiscard]] constexpr Entry<Sym, V>
named(V v) noexcept(std::is_nothrow_move_constructible_v<V>) {
  return {std::move(v)};
}

// named("x"_s, 1.25) / named(var<"x">, 1.25) -- keyed by a tag in hand.
template <CSymbolTag Tag, typename V>
  requires std::is_object_v<V>
[[nodiscard]] constexpr auto
named(const Tag &, V v) noexcept(std::is_nothrow_move_constructible_v<V>) {
  return Entry<detail::tag_key_t<Tag>::value, V>{std::move(v)};
}

// "x"_s = 1.25, declared back in expressions.hpp where symbol_type is.
template <auto S>
template <typename V>
  requires std::is_object_v<V>
constexpr Entry<S, V> symbol_type<S>::operator=(V v) const {
  return Entry<S, V>{std::move(v)};
}

template <typename T> inline constexpr bool is_entry_v = false;
template <FixedString S, typename V>
inline constexpr bool is_entry_v<Entry<S, V>> = true;

template <typename T>
concept CEntry = is_entry_v<std::remove_cvref_t<T>>;

} // namespace ddx::impl
