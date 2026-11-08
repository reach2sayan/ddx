#include "graph_builder.hpp"
#include "ddx.hpp"
#include "energy_models.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <format>
#include <ranges>
#include <vector>

// The freeze must keep each operand in its position and drop nothing reachable.

namespace {
using ddx::rt::Graph;
using ddx::rt::NodeId;

TEST(RtGraph, OperandOrderSurvivesCompression) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x / y; // not commutative: the slots must not be swapped
  const auto graph = ddx::test::freeze_values(b, std::array{f.id(b)});

  const auto [lhs, rhs, absent] = graph.operands(f.id(b));
  EXPECT_EQ(lhs, x.id(b));
  EXPECT_EQ(rhs, y.id(b));
}

TEST(RtGraph, UnaryNodeHasOneOperand) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto f = sin(x);
  const auto graph = ddx::test::freeze_values(b, std::array{f.id(b)});
  const auto [lhs, rhs, absent] = graph.operands(f.id(b));
  EXPECT_EQ(lhs, x.id(b));
  EXPECT_EQ(rhs, ddx::rt::no_node);
}

// When roots are not outputs, the value block is empty and nodes only the value
// needs are dead; d(x*y)/dx needs y, never the product.
TEST(RtGraph, RootsAloneLeaveTheValueOut) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * log(x) + 0.3 * (x * y);
  const NodeId root = f.id(b);
  const auto both = ddx::test::GraphBuilder{b}
                        .values_from(std::span<const NodeId>{&root, 1})
                        .build_jacobian()
                        .finish();
  const auto alone = ddx::test::GraphBuilder{b}
                         .values_from(std::span<const NodeId>{&root, 1}, false)
                         .build_jacobian()
                         .finish();
  EXPECT_EQ(alone.output_blocks().values.size(), 0u);
  EXPECT_EQ(alone.output_blocks().jacobian.size(),
            both.output_blocks().jacobian.size());
  EXPECT_FALSE(alone.live(root));
  EXPECT_TRUE(both.live(root));
  EXPECT_LT(alone.live_count(), both.live_count());
}

TEST(RtGraph, EverythingTheOutputsReachIsLive) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y);
  const auto dead = cos(x) + y; // built, never an output
  const auto graph = ddx::test::freeze_values(b, std::array{f.id(b)});

  EXPECT_TRUE(graph.live(f.id(b)));
  EXPECT_TRUE(graph.live(x.id(b)));
  EXPECT_TRUE(graph.live(y.id(b)));
  EXPECT_FALSE(graph.live(dead.id(b)));
  EXPECT_LT(graph.live_count(), graph.size());
}

TEST(RtGraph, CarriesSymbolsAndOutputs) {
  ddx::rt::Arena<> b;
  const auto root = ddx::rt::to_graph(b, ddx::var<"x"> * ddx::var<"y">);
  const auto graph =
      ddx::test::GraphBuilder{b}.value(root).build_jacobian().finish();

  EXPECT_EQ(graph.inputs().count(ddx::rt::OpCode::Var), 2u);
  EXPECT_EQ(graph.outputs().size(), 3u); // value plus two partials
}

TEST(RtGraph, IdOrderIsTopological) {
  ddx::rt::Arena<> b;
  const auto root = ddx::rt::to_graph(b, sin(ddx::var<"x"> * ddx::var<"y">) +
                                             exp(ddx::var<"x">));
  const auto graph =
      ddx::test::GraphBuilder{b}.value(root).build_jacobian().finish();

  // Codegen emits in id order in one pass, which is only correct if every child
  // precedes its parent.
  for (const NodeId v :
       std::views::iota(NodeId{0}, static_cast<NodeId>(graph.size()))) {
    for (const NodeId child : graph.operands(v)) {
      if (child != ddx::rt::no_node) {
        EXPECT_LT(child, v) << "node " << v << " has a forward reference";
      }
    }
  }
}

} // namespace

// Tape rows are reused once their last reader is done, so a dense model's sweep
// touches a fraction of its ids. Only the size is checked here.
TEST(RtGraph, PacksTheTapeByLiveness) {
  ddx::rt::Arena<> b;
  const auto x = std::views::iota(0uz, 64uz) |
                 std::views::transform([&](std::size_t i) {
                   return var(b, std::format("x{:02}", i));
                 }) |
                 ddx::impl::to<std::vector>();
  const NodeId root = models::mse(x).id(b);
  const auto g = ddx::test::GraphBuilder{b}
                     .values_from(std::span<const NodeId>{&root, 1})
                     .build_jacobian()
                     .finish();
  const double ratio =
      static_cast<double>(g.tape_rows()) / static_cast<double>(g.live_count());
  std::printf("[ ddx ] mse n = 64: %zu live nodes, %zu tape rows (%.3f)\n",
              g.live_count(), g.tape_rows(), ratio);
  EXPECT_LT(ratio, 0.25);
  auto rows = ddx::rt::in_order(g.output_blocks()) | std::views::join |
              std::views::transform([&g](NodeId o) { return g.rows()[o]; }) |
              ddx::impl::to<std::vector<std::uint32_t>>();
  for (const std::uint32_t r : rows) {
    EXPECT_LT(r, g.tape_rows());
  }
  std::ranges::sort(rows);
  EXPECT_EQ(std::ranges::adjacent_find(rows), rows.end());
  // No step reads the row it writes, since the lane loops assume no aliasing.
  for (const ddx::rt::Step &s : g.schedule()) {
    for (const std::uint32_t in : s.in) {
      EXPECT_NE(in, s.out);
    }
  }
}
