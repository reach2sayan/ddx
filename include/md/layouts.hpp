#pragma once

#include "md/md.hpp"
#include "util/meta.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>

// An mdspan layout that stores a derivative tensor's symmetry, keeping one
// cell per distinct mixed partial.
namespace ddx::impl {

namespace detail {

// C(n, K) in integer arithmetic, since a rounded double could pick the wrong
// cell: the falling product over K!, unrolled to a few multiplies and a divide.
template <std::size_t K>
[[nodiscard]] constexpr std::size_t binomial_fixed(std::size_t n) noexcept {
  if (n < K) {
    return 0;
  }
  std::size_t num = 1;
  std::size_t den = 1;
  meta::for_each(meta::iota<K>, [&](auto i) {
    num *= n - i;
    den *= i + 1;
  });
  return num / den;
}

static_assert(binomial_fixed<2>(4) == 6);
static_assert(binomial_fixed<3>(8) == 56);
static_assert(binomial_fixed<0>(10) == 1);
static_assert(binomial_fixed<1>(7) == 7);
static_assert(binomial_fixed<2>(1) == 0);

} // namespace detail

// Lead dense axes, for a vector output, over symmetric derivative axes packed
// into C(N + Order - 1, Order) cells, one per sorted multi-index.
template <std::size_t Lead> struct layout_leading_simplex {
  // Symmetric axes must have equal lengths to be interchangeable.
  template <md::CStaticExtents Ext> class mapping {
    static_assert(Ext::rank() > Lead,
                  "layout_leading_simplex needs at least one symmetric axis");
    static constexpr std::size_t kOrder = Ext::rank() - Lead;
    static_assert(std::ranges::all_of(std::views::iota(Lead, Ext::rank()),
                                      [](std::size_t r) {
                                        return Ext{}.extent(r) ==
                                               Ext{}.extent(Lead);
                                      }),
                  "layout_leading_simplex: the symmetric axes must have equal "
                  "extents");

  public:
    using index_type = typename Ext::index_type;

    constexpr mapping() noexcept = default;
    constexpr explicit mapping(const Ext &) noexcept {}

    [[nodiscard]] constexpr index_type required_span_size() const noexcept {
      const index_type lead = std::ranges::fold_left(
          std::views::iota(std::size_t{0}, Lead), index_type{1},
          [&](index_type acc, std::size_t r) { return acc * Ext{}.extent(r); });
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
            return acc * Ext{}.extent(r) + all[r];
          });

      std::array<index_type, kOrder> a{};
      std::ranges::copy(all | std::views::drop(Lead), a.begin());
      std::ranges::sort(a);
      // b_t = a_t + t is strictly increasing, and its colex rank is the sum of
      // C(b_t, t + 1), with t a compile-time constant in each term.
      const index_type slot = meta::unpack(meta::iota<kOrder>, [&](auto... t) {
        return static_cast<index_type>(
            (detail::binomial_fixed<t + 1>(static_cast<std::size_t>(a[t]) + t) +
             ... + std::size_t{0}));
      });
      return lead * block_size() + slot;
    }

  private:
    // Cells per output block: C(N + Order - 1, Order).
    [[nodiscard]] constexpr index_type block_size() const noexcept {
      return static_cast<index_type>(detail::binomial_fixed<kOrder>(
          static_cast<std::size_t>(Ext{}.extent(Lead)) + kOrder - 1));
    }
  };
};

// The scalar-output case, where every axis is a derivative axis.
using layout_simplex_packed = layout_leading_simplex<0>;

} // namespace ddx::impl
