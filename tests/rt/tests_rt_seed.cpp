#include "ddx.hpp"
#include "rt/dot.hpp"
#include "rt/equation.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

// A Seed is an input column that is never a symbol.  What these pin is the one
// thing every consumer has to agree on: which column of the point a leaf reads.

namespace {
using ddx::rt::Graph;

TEST(RtSeed, ReadsTheColumnAfterTheSymbols) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto f = x * x + seed(b, 0);

  // Symbols first, then the directions: one symbol, so the seed is column 1.
  const std::array point{3.0, 7.0};
  const auto values = ddx::rt::evaluate_all(b, point);
  EXPECT_DOUBLE_EQ(values[f.id(b)], 16.0);
}

TEST(RtSeed, SlotsAreTheirOwnSpace) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * seed(b, 1) + y * seed(b, 0);

  const std::array point{2.0, 3.0, 5.0, 7.0}; // x, y, v0, v1
  const auto values = ddx::rt::evaluate_all(b, point);
  EXPECT_DOUBLE_EQ(values[f.id(b)], 2.0 * 7.0 + 3.0 * 5.0);
}

TEST(RtSeed, AskingTwiceInternsOnce) {
  ddx::rt::Builder<> b;
  const auto a = seed(b, 0);
  const auto c = seed(b, 0);
  EXPECT_EQ(a.id(b), c.id(b));
}

// The point of not being a Var: seal() gates symbols because a new one moves
// the slots above it, and a seed slot moves nothing.
TEST(RtSeed, ASealedArenaStillTakesOne) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  (void)ddx::rt::build_jacobian_impl(b, (x * x).id(b)); // every sweep seals
  EXPECT_TRUE(var(b, "late").poisoned()) << "symbols stay sealed";
  EXPECT_FALSE(seed(b, 0).poisoned());
}

TEST(RtSeed, WidensTheGraphArity) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto f = x + seed(b, 0);
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});

  EXPECT_EQ(graph.symbols().size(), 1u);
  EXPECT_EQ(graph.arity(), 2u);
}

// The regression guard for every graph that existed before seeds did.
TEST(RtSeed, AGraphWithoutSeedsIsUnchanged) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * y;
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});

  EXPECT_EQ(graph.arity(), graph.symbols().size());
}

TEST(RtSeed, ADeadSeedDoesNotWiden) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  [[maybe_unused]] const auto unused = seed(b, 0);
  const auto f = x * x;
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});

  EXPECT_EQ(graph.arity(), 1u);
}

TEST(RtSeed, DotNamesTheDirection) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto f = x + seed(b, 2);
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});

  const auto text = ddx::rt::Dot{graph}.str();
  EXPECT_NE(text.find("v[2]"), std::string::npos) << text;
}
} // namespace
