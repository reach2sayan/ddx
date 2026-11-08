#pragma once

#include "symbolic/expressions.hpp" // Numeric, symbol_type, Variable
#include "util/fixed_string.hpp"
#include "util/meta.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace ddx::impl {

namespace detail {

// The symbol name of sym<"x">, "x"_s or var<"x">. Other types have no value,
// which is what CSymbolTag tests.
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

// A value bound to a symbol name, one entry of a point: named<"x">(1.5),
// Entry{"x"_s, 1.5} or Entry{var<"x">, 1.5}.
template <FixedString Sym, typename V>
  requires std::is_object_v<V>
struct Entry {
  using value_type = V;
  static constexpr auto symbol = Sym;

  V value;

  constexpr Entry()
    requires std::default_initializable<V>
  = default;

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

// named<"x">(1.25) binds a value to a symbol by name.
template <FixedString Sym, typename V>
  requires std::is_object_v<V>
[[nodiscard]] constexpr Entry<Sym, V>
named(V v) noexcept(std::is_nothrow_move_constructible_v<V>) {
  return {std::move(v)};
}

// named("x"_s, 1.25) or named(var<"x">, 1.25), keyed by a symbol value.
template <CSymbolTag Tag, typename V>
  requires std::is_object_v<V>
[[nodiscard]] constexpr auto
named(const Tag &, V v) noexcept(std::is_nothrow_move_constructible_v<V>) {
  return Entry<detail::tag_key_t<Tag>::value, V>{std::move(v)};
}

// "x"_s = 1.25, declared in expressions.hpp with symbol_type.
template <auto S>
template <typename V>
  requires std::is_object_v<V>
constexpr Entry<S, V> symbol_type<S>::operator=(V v) const {
  return Entry<S, V>{std::move(v)};
}

template <typename T>
concept CEntry = requires {
  []<FixedString S, typename V>(meta::Type<Entry<S, V>>) {
  }(meta::type_c<std::remove_cvref_t<T>>);
};

} // namespace ddx::impl
