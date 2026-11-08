#include "graph_builder.hpp"
#include "ddx.hpp"
#include "jit/kernel.hpp"
#include "jit/tests_jit_common.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <ranges>
#include <vector>

// The kernel and the sweep must agree on which input column holds each seed,
// because a mix-up gives silently wrong arithmetic rather than a crash.

namespace {
using ddx::rt::Arena;
using ddx::rt::Graph;


// The inputs are x, y, then v0, v1, the order input_column() states.
Graph<> seeded_graph(Arena<> &b) {
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * x * seed(b, 0) + exp(y) * seed(b, 1);
  return ddx::test::GraphBuilder{b}.value(f).finish();
}

TEST(JitSeed, KernelArityCountsTheSeeds) {
  Arena<> b;
  const auto graph = seeded_graph(b);
  ASSERT_EQ(graph.arity(), 4u);
  EXPECT_EQ(must_compile(graph).shape().arity, 4u);
}

TEST(JitSeed, MatchesTheInterpreterToTheBit) {
  Arena<> b;
  const auto graph = seeded_graph(b);
  const auto kernel = must_compile(graph);

  // Enough points to cross the lane width with a ragged tail, since the masked
  // loads on the direction columns are their own code path.
  static constexpr std::size_t n = 37;
  std::vector<std::vector<double>> columns(4, std::vector<double>(n));
  for (const auto [j, column] : columns | std::views::enumerate) {
    for (const auto [i, cell] : column | std::views::enumerate) {
      cell = 0.3 + 0.4 * static_cast<double>((i + 3 * j) % 5) / 5.0;
    }
  }
  const auto xs =
      columns | std::views::transform([](const auto &c) { return c.data(); }) |
      ddx::impl::to<std::vector<const double *>>();

  std::vector<double> got(n);
  double *const values[]{got.data()};
  kernel(xs, values, {}, {}, n);

  for (const std::size_t i : std::views::iota(0uz, n)) {
    const auto point =
        columns | std::views::transform([i](const auto &c) { return c[i]; }) |
        ddx::impl::to<std::vector>();
    const auto want = ddx::rt::evaluate_all(b, point);
    EXPECT_EQ(got[i], want[graph.outputs()[0]]) << "at point " << i;
  }
}

// Reading the direction out of the symbol half of `xs` would still compile and
// run, so the seed slot is checked directly.
TEST(JitSeed, DirectionIsNotReadAsASymbol) {
  Arena<> b;
  const auto graph = seeded_graph(b);
  const auto kernel = must_compile(graph);

  // v0 = 1, v1 = 0 selects only the x*x term and nothing of exp(y).
  const std::array<double, 1> x{3.0};
  const std::array<double, 1> y{2.0};
  const std::array<double, 1> v0{1.0};
  const std::array<double, 1> v1{0.0};
  const double *const xs[]{x.data(), y.data(), v0.data(), v1.data()};

  double out = 0.0;
  double *const values[]{&out};
  kernel(xs, values, {}, {}, 1);
  EXPECT_DOUBLE_EQ(out, 9.0);
}
} // namespace
