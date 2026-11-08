#pragma once
// Shared by tests_tensor.cpp and dual/tests_tensor_dual.cpp.

#include "tests_common.hpp"

#include <boost/hash2/fnv1a.hpp>

#include <boost/hash2/flavor.hpp>
#include <boost/hash2/hash_append.hpp>

#include <bit>

// Hashes the IEEE bits of every result; the hash is pinned per toolchain with
// -DDDX_PIN_BIT_HASH=<value>, and a moved hash means results moved.

namespace {
struct BitHash {
  boost::hash2::fnv1a_64 h;
  void operator()(double d) noexcept {
    // Hash the bits, because hash2 appends a float as v + 0, which would fold
    // -0.0 onto +0.0 and hide a sign flip.
    const auto bits = std::bit_cast<std::uint64_t>(d);
    boost::hash2::hash_append(h, boost::hash2::little_endian_flavor{}, bits);
  }
  // result() advances the state, so the gate reads it exactly once.
  [[nodiscard]] std::uint64_t value() noexcept { return h.result(); }
};
} // namespace

// The packed tensor, which skips permutations, must equal the dense tensor
// evaluated at every multi-index.
template <std::size_t Order, CExpression Expr, ddx::meta::CTupleLike Values>
void ExpectPackedMatchesDense(const Expr &expr, const Values &values) {
  const auto packed = Equation{expr}.template derivative_tensor<Order>(values);
  constexpr std::size_t N = std::tuple_size_v<Values>;

  using S = double;
  using U = ddx::impl::nth_dual_t<S, Order>;
  using symbols = ddx::impl::expr_symbols_t<Expr>;
  // Every multi-index, in layout_right order.
  const auto grid = ddx::meta::unpack(ddx::meta::iota<Order>, [&](auto... d) {
    return std::views::cartesian_product(
        ((void)d, std::views::iota(std::size_t{0}, N))...);
  });
  for (const auto &multi : grid) {
    const auto idx = ddx::meta::unpack(
        multi, [](auto... i) { return std::array<std::size_t, Order>{i...}; });
    std::array<U, N> seeds{};
    for (const std::size_t k : std::views::iota(0uz, N)) {
      seeds[k] =
          ddx::impl::detail::make_mixed_seed<S, Order>(values[k], idx, k);
    }
    const U val = expr.template eval_seeded<symbols>(seeds);
    const double dense = ddx::impl::detail::extract_nth<Order>(val);
    EXPECT_DOUBLE_EQ(packed.at_index(idx), dense)
        << "order " << Order << " at index " << idx[0];
  }
}
