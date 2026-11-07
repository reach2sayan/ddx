#pragma once

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <cstddef>
#include <type_traits>

namespace ddx::impl {

namespace mp = boost::mp11;

template <auto S> struct symbol_type;

// Whether T is a symbol_type. The trait form exists for mp_all_of.
template <typename T> inline constexpr bool is_symbol_v = false;
template <auto S> inline constexpr bool is_symbol_v<symbol_type<S>> = true;
template <typename T> struct is_symbol : std::bool_constant<is_symbol_v<T>> {};

template <typename T>
concept CSymbol = is_symbol_v<std::remove_cvref_t<T>>;

template <typename L>
concept CTypeList = mp::mp_is_list<std::remove_cvref_t<L>>::value;

// An mp_list of symbol_types in alphabetical order. CTypeList is tested first
// because mp_all_of over a non-list is ill-formed rather than false.
template <typename L>
concept CSymbolList =
    CTypeList<L> && mp::mp_all_of<std::remove_cvref_t<L>, is_symbol>::value;

// The slot a name occupies in List, or the list's length if absent. S is a
// plain auto because a constrained one breaks later deductions on GCC 15.
template <auto S, CTypeList List>
consteval std::size_t symbol_index() noexcept {
  return mp::mp_find<List, symbol_type<S>>::value;
}

} // namespace ddx::impl
