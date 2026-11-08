#pragma once

#include "util/config.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/tuple.hpp>

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

// The type a meta::Type<T> value carries, as DDX_FWD is for forwarding.
#define DDX_TYPE(...) ::ddx::meta::type_of<decltype(__VA_ARGS__)>

// Types and indices as values, over Boost.Mp11. Type-level algorithms hand
// their callbacks Type<T>; for_each and unpack hand over the elements
// themselves, as mp_for_each does, so iota<N> yields integral_constants.
namespace ddx::meta {

namespace mp = boost::mp11;

template <typename T> struct Type : std::type_identity<T> {
  template <typename U>
  [[nodiscard]] friend consteval bool operator==(Type, Type<U>) noexcept {
    return std::same_as<T, U>;
  }
};

template <typename T> inline constexpr Type<T> type_c{};
template <typename V> using type_of = typename std::remove_cvref_t<V>::type;

// Has a std::tuple_size, so get<I> decomposes it.
template <typename T>
concept CTupleLike =
    requires { std::tuple_size<std::remove_cvref_t<T>>::value; };

// Any L<Ts...> but a tuple, which for_each and unpack walk as a value.
template <typename L>
concept CTypeList = mp::mp_is_list<std::remove_cvref_t<L>>::value && !CTupleLike<L>;

template <typename... Ts> using list = mp::mp_list<Ts...>;
template <typename... Ts> inline constexpr list<Ts...> list_c{};
// The type of a list value, as using L = meta::list_t<sort(l, less)>.
// L is a plain auto: a constrained auto parameter breaks later deductions on
// GCC 15.
template <auto L> using list_t = std::remove_cvref_t<decltype(L)>;
template <std::size_t N> using index_t = std::integral_constant<std::size_t, N>;
template <std::size_t N> inline constexpr mp::mp_iota_c<N> iota{};

// The types of a template's arguments, e.g. a tuple's, as Type<T> values.
template <typename T> consteval auto args_of(Type<T>) noexcept {
  return mp::mp_transform<Type, mp::mp_rename<T, mp::mp_list>>{};
}

namespace detail {
template <typename F> struct test {
  template <typename... A>
  using fn = std::bool_constant<static_cast<bool>(F{}(type_c<A>...))>;
};
} // namespace detail

consteval std::size_t size(CTypeList auto l) noexcept {
  return mp::mp_size<decltype(l)>::value;
}
template <std::size_t I> consteval auto at(CTypeList auto l) noexcept {
  return type_c<mp::mp_at_c<decltype(l), I>>;
}
consteval auto concat(CTypeList auto... l) noexcept {
  return mp::mp_append<mp::mp_list<>, decltype(l)...>{};
}
template <typename T>
consteval auto push_front(CTypeList auto l, Type<T>) noexcept {
  return mp::mp_push_front<decltype(l), T>{};
}
template <std::size_t N> consteval auto repeat(CTypeList auto l) noexcept {
  return mp::mp_repeat_c<decltype(l), N>{};
}
template <typename T>
consteval bool contains(CTypeList auto l, Type<T>) noexcept {
  return mp::mp_contains<decltype(l), T>::value;
}
// The position of T, or size(l) if absent.
template <typename T>
consteval std::size_t index_of(CTypeList auto l, Type<T>) noexcept {
  return mp::mp_find<decltype(l), T>::value;
}
template <typename F> consteval bool all_of(CTypeList auto l, F) noexcept {
  return mp::mp_all_of_q<decltype(l), detail::test<F>>::value;
}
// Stable, so equal elements keep their order.
template <typename F> consteval auto sort(CTypeList auto l, F) noexcept {
  return mp::mp_sort_q<decltype(l), detail::test<F>>{};
}
consteval auto unique(CTypeList auto l) noexcept {
  return mp::mp_unique<decltype(l)>{};
}
// Template<T, ..., T>, with T N times.
template <template <typename...> typename Template, typename T, std::size_t N>
using repeat_t = mp::mp_apply<Template, mp::mp_repeat_c<list<T>, N>>;

// Type<Template<Ts...>> for l = list_c<Ts...>.
template <template <typename...> typename Template>
consteval auto rebind(CTypeList auto l) noexcept {
  return type_c<mp::mp_apply<Template, decltype(l)>>;
}

// Calls f(T{}) for each T in order; over a tuple, f(element) left to right.
DDX_ALWAYS_INLINE constexpr void for_each(CTypeList auto l, auto &&f) {
  mp::mp_for_each<decltype(l)>(DDX_FWD(f));
}
DDX_ALWAYS_INLINE constexpr void for_each(CTupleLike auto &&t, auto &&f) {
  mp::tuple_for_each(DDX_FWD(t), DDX_FWD(f));
}

// f(Ts{}...) for a list; std::apply(f, t) for a tuple.
template <typename... Ts>
DDX_ALWAYS_INLINE constexpr decltype(auto) unpack(mp::mp_list<Ts...>,
                                                  auto &&f) {
  return DDX_FWD(f)(Ts{}...);
}
DDX_ALWAYS_INLINE constexpr decltype(auto) unpack(CTupleLike auto &&t,
                                                  auto &&f) {
  return std::apply(DDX_FWD(f), DDX_FWD(t));
}

// f(integral_constant<i>) for a runtime i < N.
template <std::size_t N>
DDX_ALWAYS_INLINE constexpr decltype(auto) with_index(std::size_t i,
                                                      auto &&f) {
  return mp::mp_with_index<N>(i, DDX_FWD(f));
}

} // namespace ddx::meta
