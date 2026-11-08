#pragma once

#include "util/meta.hpp"

#include <cstddef>
#include <type_traits>

namespace ddx::impl {

template <auto S> struct symbol_type;

template <typename T>
concept CSymbol = requires {
  []<auto S>(meta::Type<symbol_type<S>>) {
  }(meta::type_c<std::remove_cvref_t<T>>);
};

// A meta list of symbol_types in alphabetical order.
template <typename L>
concept CSymbolList = meta::CTypeList<L> &&
                      meta::all_of(std::remove_cvref_t<L>{},
                                   [](auto t) { return CSymbol<DDX_TYPE(t)>; });

// The slot a name occupies in List, or the list's length if absent. S is a
// plain auto because a constrained one breaks later deductions on GCC 15.
template <auto S, meta::CTypeList List>
consteval std::size_t symbol_index() noexcept {
  return meta::index_of(List{}, meta::type_c<symbol_type<S>>);
}

} // namespace ddx::impl
