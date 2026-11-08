#pragma once

// The vendored reference mdspan, since no supported standard library ships
// <mdspan>. Named by explicit using-declarations rather than a namespace alias.

// Set before the include, or the vendored header declares itself in ::std.
#ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
#define MDSPAN_IMPL_STANDARD_NAMESPACE diff_mdspan
#endif
#include "third_party/mdspan.hpp"
#include "util/meta.hpp"
#define DDX_MDSPAN_NS MDSPAN_IMPL_STANDARD_NAMESPACE

#include <cstddef>
#include <type_traits>

namespace ddx::impl::md {

using DDX_MDSPAN_NS::dextents;
using DDX_MDSPAN_NS::extents;
using DDX_MDSPAN_NS::layout_right;
using DDX_MDSPAN_NS::mdspan;

template <typename E>
concept CExtents = requires {
  []<typename I, std::size_t... N>(meta::Type<extents<I, N...>>) {
  }(meta::type_c<std::remove_cvref_t<E>>);
};

// Every extent is known at compile time, so a tensor can own a std::array and
// stay usable in constant evaluation.
template <typename E>
concept CStaticExtents =
    CExtents<E> && (std::remove_cvref_t<E>::rank_dynamic() == 0);

static_assert(CExtents<extents<std::size_t, 2, 3>>);
static_assert(CStaticExtents<extents<std::size_t, 2, 3>>);
static_assert(!CStaticExtents<dextents<std::size_t, 2>>);

} // namespace ddx::impl::md
