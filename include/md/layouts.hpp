#pragma once

#include "md/md.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>

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

// Lead dense leading axes over a simplex-packed remainder.  By Schwarz only
// non-decreasing multi-indices over the derivative axes carry distinct values:
// C(N + Order - 1, Order) against N^Order dense cells.  Lead covers a
// vector-valued Equation's output axis; Lead == 0 is the scalar case.  The
// symmetric part ranks the sorted multiset in colex order, via b_t = a_t + t.
template <std::size_t Lead> struct layout_leading_simplex {
  template <md::CExtents Ext>
  static constexpr std::size_t order_of = Ext::rank() - Lead;
  // Two axes can only be interchangeable if they have the same length, so the
  // mapping refuses extents whose symmetric axes differ.
  template <md::CStaticExtents Ext> class mapping {
    static_assert(Ext::rank() > Lead,
                  "layout_leading_simplex needs at least one symmetric axis");
    static constexpr std::size_t kOrder = order_of<Ext>;
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

  private:
    // Cells per output block: C(N + Order - 1, Order).  The packing is what
    // this layout *is*; required_span_size and operator() are how it is asked.
    [[nodiscard]] constexpr index_type block_size() const noexcept {
      return static_cast<index_type>(detail::binomial_fixed<kOrder>(
          static_cast<std::size_t>(Ext{}.extent(Lead)) + kOrder - 1));
    }
  };
};

// The scalar case: every axis is a derivative axis.
using layout_simplex_packed = layout_leading_simplex<0>;

} // namespace ddx::impl
