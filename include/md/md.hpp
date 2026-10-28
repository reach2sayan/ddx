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

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ddx::impl::md {

using DDX_MDSPAN_NS::dextents;
using DDX_MDSPAN_NS::extents;
using DDX_MDSPAN_NS::mdspan;
using DDX_MDSPAN_NS::layout_right;

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

// Checked against the extents it will carry.  There is no unparameterised
// CLayoutPolicy: the layouts here are rank-constrained, so probing them with
// arbitrary extents would hard-error.
template <typename M, typename E>
concept CLayoutMappingOf = CExtents<E> && std::copyable<M> &&
                           std::equality_comparable<M> && requires(const M m) {
                             typename M::extents_type;
                             typename M::index_type;
                             typename M::rank_type;
                             typename M::layout_type;
                             // No `M(e)`: layout_stride's mapping needs the
                             // strides too.
                             { m.extents() } -> std::convertible_to<E>;
                             {
                               m.required_span_size()
                             } -> std::convertible_to<typename M::index_type>;
                             { M::is_always_unique() } -> std::same_as<bool>;
                             {
                               M::is_always_exhaustive()
                             } -> std::same_as<bool>;
                             { M::is_always_strided() } -> std::same_as<bool>;
                             { m.is_unique() } -> std::same_as<bool>;
                             { m.is_exhaustive() } -> std::same_as<bool>;
                             { m.is_strided() } -> std::same_as<bool>;
                           };

template <typename P, typename E>
concept CLayoutFor = CExtents<E> && requires {
  typename P::template mapping<E>;
} && CLayoutMappingOf<typename P::template mapping<E>, E>;

static_assert(CExtents<extents<std::size_t, 2, 3>>);
static_assert(CStaticExtents<extents<std::size_t, 2, 3>>);
static_assert(!CStaticExtents<dextents<std::size_t, 2>>);
static_assert(CLayoutFor<layout_right, extents<std::size_t, 2, 3>>);

} // namespace ddx::impl::md
