#pragma once

#include "md/md.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <ranges>

#include <boost/assert.hpp>

// A derivative tensor's symmetry as storage rather than convention.  (The
// sparse-pattern layout is in coupling.hpp, next to the pass deriving it.)
namespace ddx::impl {

namespace detail {

// C(n, k), integer arithmetic only: a rounded double would file a derivative in
// the wrong cell.  consteval; the subscript path uses binomial_fixed.
[[nodiscard]] consteval std::size_t binomial(std::size_t n,
                                             std::size_t k) noexcept {
  if (k > n) {
    return 0;
  }
  k = std::min(k, n - k); // C(n, k) == C(n, n - k); take the shorter fold
  return std::ranges::fold_left(
      std::views::iota(std::size_t{1}, k + 1), std::size_t{1},
      [n, k](std::size_t acc, std::size_t i) { return acc * (n - k + i) / i; });
}

static_assert(binomial(4, 2) == 6);
static_assert(binomial(8, 3) == 56);
static_assert(binomial(10, 0) == 1);

// C(n, K) with K fixed: the falling product over K!.  binomial()'s
// `k = min(k, n - k)` makes the fold's bound depend on the runtime n, so the
// loop cannot unroll; this collapses to a few multiplies and one divide.
template <std::size_t K>
[[nodiscard]] constexpr std::size_t binomial_fixed(std::size_t n) noexcept {
  if (n < K) {
    return 0;
  }
  std::size_t num = 1;
  std::size_t den = 1;
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    ((num *= (n - I), den *= (I + 1)), ...);
  }(std::make_index_sequence<K>{});
  return num / den;
}

static_assert(binomial_fixed<2>(4) == 6);
static_assert(binomial_fixed<3>(8) == 56);
static_assert(binomial_fixed<0>(10) == 1);
static_assert(binomial_fixed<1>(7) == 7);
static_assert(binomial_fixed<2>(1) == 0);

} // namespace detail

namespace detail {
// The typedefs, the extents and the six predicates [mdspan.layout.reqmts] asks
// of every mapping, so a layout writes only what it actually maps.  md.hpp's
// CLayoutMappingOf checks this contract; this supplies it.  operator== stays
// with the derived: hoisted, it would compare two different layouts sharing an
// Ext.
template <md::CExtents Ext, bool Unique, bool Exhaustive, bool Strided>
class mapping_base {
public:
  using extents_type = Ext;
  using index_type = typename Ext::index_type;
  using rank_type = typename Ext::rank_type;

  constexpr mapping_base() noexcept = default;
  constexpr explicit mapping_base(const Ext &e) noexcept : ext_(e) {}

  [[nodiscard]] constexpr const Ext &extents() const noexcept { return ext_; }

  static constexpr bool is_always_unique() noexcept { return Unique; }
  static constexpr bool is_always_exhaustive() noexcept { return Exhaustive; }
  static constexpr bool is_always_strided() noexcept { return Strided; }
  [[nodiscard]] constexpr bool is_unique() const noexcept { return Unique; }
  [[nodiscard]] constexpr bool is_exhaustive() const noexcept {
    return Exhaustive;
  }
  [[nodiscard]] constexpr bool is_strided() const noexcept { return Strided; }

protected:
  Ext ext_{};
};
} // namespace detail

// Lead dense leading axes over a simplex-packed remainder.  By Schwarz only
// non-decreasing multi-indices over the derivative axes carry distinct values:
// C(N + Order - 1, Order) against N^Order dense cells.  Lead covers a
// vector-valued Equation's output axis; Lead == 0 is the scalar case.  The
// symmetric part ranks the sorted multiset in colex order, via b_t = a_t + t.
template <std::size_t Lead> struct layout_leading_simplex {
  template <md::CExtents Ext>
  static constexpr std::size_t order_of = Ext::rank() - Lead;
  // Unique only when there is nothing to permute.
  template <md::CExtents Ext>
  using base_for = detail::mapping_base<Ext, order_of<Ext> == 1, true, false>;

  // Two axes can only be interchangeable if they have the same length, so the
  // mapping refuses extents whose symmetric axes differ: at compile time when
  // they are static, at construction otherwise.
  template <md::CExtents Ext> class mapping : public base_for<Ext> {
    static_assert(Ext::rank() > Lead,
                  "layout_leading_simplex needs at least one symmetric axis");
    static constexpr std::size_t kOrder = order_of<Ext>;
    using base_t = base_for<Ext>;

    [[nodiscard]] static constexpr bool symmetric_axes_agree(const Ext &e) {
      return std::ranges::all_of(
          std::views::iota(Lead, Ext::rank()),
          [&e](std::size_t r) { return e.extent(r) == e.extent(Lead); });
    }

  public:
    using typename base_t::index_type;
    using layout_type = layout_leading_simplex;

    constexpr mapping() noexcept : mapping(Ext{}) {}
    constexpr explicit mapping(const Ext &e) noexcept : base_t(e) {
      if constexpr (md::CStaticExtents<Ext>) {
        static_assert(symmetric_axes_agree(Ext{}),
                      "layout_leading_simplex: the symmetric axes must have "
                      "equal extents");
      } else {
        BOOST_ASSERT_MSG(symmetric_axes_agree(e),
                         "layout_leading_simplex: the symmetric axes must have "
                         "equal extents");
      }
    }

    [[nodiscard]] constexpr index_type required_span_size() const noexcept {
      const index_type lead = std::ranges::fold_left(
          std::views::iota(std::size_t{0}, Lead), index_type{1},
          [&](index_type acc, std::size_t r) {
            return acc * this->ext_.extent(r);
          });
      return lead * block_size();
    }

    [[nodiscard]] constexpr index_type
    operator()(std::integral auto... idx) const noexcept
      requires(sizeof...(idx) == Ext::rank())
    {
      const std::array<index_type, Ext::rank()> all{
          static_cast<index_type>(idx)...};

      const index_type lead = std::ranges::fold_left(
          std::views::iota(std::size_t{0}, Lead), index_type{0},
          [&](index_type acc, std::size_t r) {
            return acc * this->ext_.extent(r) + all[r];
          });

      std::array<index_type, kOrder> a{};
      std::ranges::copy(all | std::views::drop(Lead), a.begin());
      std::ranges::sort(a);
      // b_t = a_t + t is strictly increasing, so it is a plain combination;
      // its colex rank is the sum of C(b_t, t + 1).  Expanded over an
      // index_sequence so that `t` is a compile-time constant at each term.
      const index_type slot = [&]<std::size_t... T>(std::index_sequence<T...>) {
        return static_cast<index_type>(
            (detail::binomial_fixed<T + 1>(static_cast<std::size_t>(a[T]) + T) +
             ... + std::size_t{0}));
      }(std::make_index_sequence<kOrder>{});
      return lead * block_size() + slot;
    }

    [[nodiscard]] friend constexpr bool operator==(const mapping &a,
                                                   const mapping &b) noexcept {
      return a.extents() == b.extents();
    }

  private:
    // Cells per output block: C(N + Order - 1, Order).  The packing is what
    // this layout *is*; required_span_size and operator() are how it is asked.
    [[nodiscard]] constexpr index_type block_size() const noexcept {
      return static_cast<index_type>(detail::binomial_fixed<kOrder>(
          static_cast<std::size_t>(this->ext_.extent(Lead)) + kOrder - 1));
    }
  };
};

// The scalar case: every axis is a derivative axis.
using layout_simplex_packed = layout_leading_simplex<0>;

// clang-format off
// md.hpp asserts the standard layouts against the contract; these are ours.
static_assert(md::CLayoutFor<layout_simplex_packed, md::dextents<std::size_t, 2>>);
static_assert(md::CLayoutFor<layout_leading_simplex<1>, md::dextents<std::size_t, 3>>);
// clang-format on
} // namespace ddx::impl
