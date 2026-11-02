#pragma once

// The one place the library says "mdspan".  It binds to the vendored reference
// implementation (third_party/mdspan.hpp, Apache-2.0 WITH LLVM-exception): no
// standard library ddx builds against ships <mdspan>.  Explicit `using`
// declarations, not a namespace alias: `namespace md = std` would make
// ddx::impl::md::vector compile.

// Before the include: otherwise the vendored header plants itself in ::std,
// which is UB and collides with a real <mdspan> in the same TU.
#ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
#define MDSPAN_IMPL_STANDARD_NAMESPACE diff_mdspan
#endif
#include "third_party/mdspan.hpp"
#define DDX_MDSPAN_NS MDSPAN_IMPL_STANDARD_NAMESPACE

#include <cstddef>
#include <type_traits>

namespace ddx::impl::md {

using DDX_MDSPAN_NS::dextents;
using DDX_MDSPAN_NS::extents;
using DDX_MDSPAN_NS::layout_right;
using DDX_MDSPAN_NS::mdspan;

namespace detail {
template <typename T> inline constexpr bool is_extents_v = false;
template <typename I, std::size_t... E>
inline constexpr bool is_extents_v<extents<I, E...>> = true;

} // namespace detail

template <typename E>
concept CExtents = detail::is_extents_v<std::remove_cvref_t<E>>;

// Every extent is known at compile time, so a tensor over it can own a
// std::array and stay usable in constant evaluation.
template <typename E>
concept CStaticExtents =
    CExtents<E> && (std::remove_cvref_t<E>::rank_dynamic() == 0);

static_assert(CExtents<extents<std::size_t, 2, 3>>);
static_assert(CStaticExtents<extents<std::size_t, 2, 3>>);
static_assert(!CStaticExtents<dextents<std::size_t, 2>>);

} // namespace ddx::impl::md
