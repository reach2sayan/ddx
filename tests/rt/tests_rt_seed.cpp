#include "ddx.hpp"
#include "graph_builder.hpp"
#include "rt/equation.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

// A Seed is an input column that is never a symbol; these tests pin which
// column of the point a seed leaf reads.

namespace {
using ddx::rt::Graph;

TEST(RtSeed, ReadsTheColumnAfterTheSymbols) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto f = x * x + seed(b, 0);

  // Symbols come before directions, so with one symbol the seed is column 1.
  const std::array point{3.0, 7.0};
  const auto values = ddx::rt::evaluate_all(b, point);
  EXPECT_DOUBLE_EQ(values[f.id(b)], 16.0);
}

TEST(RtSeed, SlotsAreTheirOwnSpace) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * seed(b, 1) + y * seed(b, 0);

  const std::array point{2.0, 3.0, 5.0, 7.0}; // x, y, v0, v1
  const auto values = ddx::rt::evaluate_all(b, point);
  EXPECT_DOUBLE_EQ(values[f.id(b)], 2.0 * 7.0 + 3.0 * 5.0);
}

TEST(RtSeed, AskingTwiceInternsOnce) {
  ddx::rt::Arena<> b;
  const auto a = seed(b, 0);
  const auto c = seed(b, 0);
  EXPECT_EQ(a.id(b), c.id(b));
}

TEST(RtSeed, WidensTheGraphArity) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto f = x + seed(b, 0);
  const auto graph = ddx::test::freeze_values(b, std::array{f.id(b)});

  EXPECT_EQ(graph.inputs().count(ddx::rt::OpCode::Var), 1u);
  EXPECT_EQ(graph.arity(), 2u);
}

TEST(RtSeed, AGraphWithoutSeedsIsUnchanged) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * y;
  const auto graph = ddx::test::freeze_values(b, std::array{f.id(b)});

  EXPECT_EQ(graph.arity(), graph.inputs().count(ddx::rt::OpCode::Var));
}

TEST(RtSeed, ADeadSeedDoesNotWiden) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  [[maybe_unused]] const auto unused = seed(b, 0);
  const auto f = x * x;
  const auto graph = ddx::test::freeze_values(b, std::array{f.id(b)});

  EXPECT_EQ(graph.arity(), 1u);
}
} // namespace
