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

// What matters is that operand *position* survives the freeze and that
// nothing reachable is dropped.

namespace {
using ddx::rt::Graph;
using ddx::rt::NodeId;

TEST(RtGraph, OperandOrderSurvivesCompression) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x / y; // not commutative: the slots must not be swapped
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});

  const auto [lhs, rhs, absent] = graph.operands(f.id(b));
  EXPECT_EQ(lhs, x.id(b));
  EXPECT_EQ(rhs, y.id(b));
}

TEST(RtGraph, UnaryNodeHasOneOperand) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto f = sin(x);
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});
  const auto [lhs, rhs, absent] = graph.operands(f.id(b));
  EXPECT_EQ(lhs, x.id(b));
  EXPECT_EQ(rhs, ddx::rt::no_node);
}

// Roots that are not outputs: the value block is empty, and the nodes only the
// value needs are not live.  x*y is the case -- d/dx wants y, never the
// product.
TEST(RtGraph, RootsAloneLeaveTheValueOut) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * log(x) + 0.3 * (x * y);
  const NodeId root = f.id(b);
  const auto both = ddx::rt::GraphBuilder{b}
                        .values_from(std::span<const NodeId>{&root, 1})
                        .build_jacobian()
                        .finish();
  const auto alone = ddx::rt::GraphBuilder{b}
                         .roots_from(std::span<const NodeId>{&root, 1})
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
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y);
  const auto dead = cos(x) + y; // built, never an output
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});

  EXPECT_TRUE(graph.live(f.id(b)));
  EXPECT_TRUE(graph.live(x.id(b)));
  EXPECT_TRUE(graph.live(y.id(b)));
  EXPECT_FALSE(graph.live(dead.id(b)));
  EXPECT_LT(graph.live_count(), graph.size());
}

TEST(RtGraph, CarriesSymbolsAndOutputs) {
  ddx::rt::Builder<> b;
  const auto root = ddx::rt::to_graph(b, ddx::var<"x"> * ddx::var<"y">);
  const auto graph =
      ddx::rt::GraphBuilder{b}.value(root).build_jacobian().finish();

  EXPECT_EQ(graph.inputs().count(ddx::rt::OpCode::Var), 2u);
  EXPECT_EQ(graph.outputs().size(), 3u); // value plus two partials
}

TEST(RtGraph, IdOrderIsTopological) {
  ddx::rt::Builder<> b;
  const auto root = ddx::rt::to_graph(b, sin(ddx::var<"x"> * ddx::var<"y">) +
                                             exp(ddx::var<"x">));
  const auto graph =
      ddx::rt::GraphBuilder{b}.value(root).build_jacobian().finish();

  // Codegen emits in id order in one pass, which is only correct if every
  // child precedes its parent.
  for (NodeId v = 0; v < graph.size(); ++v) {
    for (const NodeId child : graph.operands(v)) {
      if (child != ddx::rt::no_node) {
        EXPECT_LT(child, v) << "node " << v << " has a forward reference";
      }
    }
  }
}

} // namespace

// The tape is the live set, not the arena: a row is freed after its last
// reader and taken by the next node, so a dense model's gradient sweeps over a
// fraction of the ids it schedules.  The arithmetic is untouched -- every
// kernel-against-sweep test is the proof -- so this asks only about size.
TEST(RtGraph, PacksTheTapeByLiveness) {
  ddx::rt::Builder<> b;
  std::vector<ddx::rt::RTExpression<>> x;
  for (std::size_t i = 0; i < 64; ++i) {
    x.push_back(var(b, std::format("x{:02}", i)));
  }
  const NodeId root = models::mse(x).id(b);
  const auto g = ddx::rt::GraphBuilder{b}
                     .values_from(std::span<const NodeId>{&root, 1})
                     .build_jacobian()
                     .finish();
  const double ratio = static_cast<double>(g.tape_slots()) /
                       static_cast<double>(g.live_count());
  std::printf("[ ddx ] mse n = 64: %zu live nodes, %zu tape rows (%.3f)\n",
              g.live_count(), g.tape_slots(), ratio);
  EXPECT_LT(ratio, 0.25);
  // Every output owns a distinct row inside the tape.
  auto rows = ddx::rt::in_order(g.output_slots()) | std::views::join |
              ddx::impl::to<std::vector<std::uint32_t>>();
  EXPECT_TRUE(std::ranges::all_of(
      rows, [&g](std::uint32_t r) { return r < g.tape_slots(); }));
  std::ranges::sort(rows);
  EXPECT_EQ(std::ranges::adjacent_find(rows), rows.end());
  // No step reads the row it writes: the lane loops promise no aliasing.
  for (const ddx::rt::Step &s : g.schedule()) {
    for (const std::uint32_t in : s.in) {
      EXPECT_NE(in, s.out);
    }
  }
}
