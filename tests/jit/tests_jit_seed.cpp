#include "ddx.hpp"
#include "jit/kernel.hpp"
#include "jit/tests_jit_common.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <vector>

// The seed columns are the first inputs codegen loads that are not symbols, so
// what these pin is that the kernel and the sweep agree on which column is
// which -- getting it wrong is silent wrong arithmetic, not a crash.

namespace {
using ddx::rt::Arena;
using ddx::rt::Graph;


// x, y, then v0, v1 -- the order input_column() states.
Graph<> seeded_graph(Arena<> &b) {
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * x * seed(b, 0) + exp(y) * seed(b, 1);
  return ddx::rt::GraphBuilder{b}.value(f).finish();
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

  // Enough points to cross the lane width, and a tail that is not a multiple
  // of it: the masked loads on the direction columns are their own path.
  static constexpr std::size_t n = 37;
  std::vector<std::vector<double>> columns(4, std::vector<double>(n));
  for (const auto [j, column] : columns | std::views::enumerate) {
    for (const auto [i, cell] : column | std::views::enumerate) {
      cell = 0.3 + 0.4 * static_cast<double>((i + 3 * j) % 5) / 5.0;
    }
  }
  std::vector<const double *> xs;
  for (const auto &column : columns) {
    xs.push_back(column.data());
  }

  std::vector<double> got(n);
  double *const values[]{got.data()};
  kernel(xs, values, {}, {}, n);

  for (const std::size_t i : std::views::iota(0uz, n)) {
    std::vector<double> point;
    for (const auto &column : columns) {
      point.push_back(column[i]);
    }
    const auto want = ddx::rt::evaluate_all(b, point);
    EXPECT_EQ(got[i], want[graph.outputs()[0]]) << "at point " << i;
  }
}

// A seed slot is not a symbol slot: reading the direction out of the symbol
// half of `xs` would still typecheck and still run.
TEST(JitSeed, DirectionIsNotReadAsASymbol) {
  Arena<> b;
  const auto graph = seeded_graph(b);
  const auto kernel = must_compile(graph);

  // v0 = 1, v1 = 0 selects the first term alone: x*x, and nothing of exp(y).
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
