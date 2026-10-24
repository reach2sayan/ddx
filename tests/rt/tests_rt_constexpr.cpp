#include "ddx.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/equation.hpp"
#include "rt/interpret.hpp"

#include <algorithm>
#include <span>

#include <gtest/gtest.h>

// "Runtime" names when the *structure* is decided, not when the arithmetic
// runs, so a graph over constant-visible values is a constant expression.
// Everything here is a static_assert; the TEST bodies exist so the file
// reports.  The frozen graph is the boundary -- Boost.Graph's containers are
// not constexpr, and neither is the JIT.

namespace {
using ddx::rt::Arena;
using RE = ddx::rt::Expression<>;

consteval double build_and_evaluate() {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * x + y;
  return ddx::rt::evaluate(b, f.id(b), std::array{3.0, 4.0});
}
static_assert(build_and_evaluate() == 13.0);

// The rewrites fold at compile time exactly as they do at run time.
consteval bool rewrites_hold() {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  return (x + RE{0}).id(b) == x.id(b) && (x * RE{1}).id(b) == x.id(b) &&
         (-(-x)).id(b) == x.id(b) && ((y / x) * x).id(b) == y.id(b) &&
         ((x * y)).id(b) == (y * x).id(b);
}
static_assert(rewrites_hold());

// Interning is a compile-time property too: the same subtree twice is one node.
consteval std::size_t nodes_for_repeated_subtree() {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y);
  const auto g = exp(x) * sin(y);
  return f.id(b) == g.id(b) ? b.size() : 0;
}
static_assert(nodes_for_repeated_subtree() == 5);

// A Jacobian, swept and evaluated entirely during constant evaluation.
consteval double partial_of_product() {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * y;
  const auto g = ddx::rt::derive_jacobian(b, f.id(b));
  return ddx::rt::evaluate_all(b, std::array{3.0, 4.0})[g.partial[0]];
}
static_assert(partial_of_product() == 4.0);

// <cmath> is not constexpr; folding a libm call is a GCC extension and clang
// rejects it outright, so this one sweep is GCC-only.
#if defined(__GNUC__) && !defined(__clang__)
consteval double transcendental_jacobian() {
  Arena<> b;
  const auto x = var(b, "x");
  const auto f = sin(x) * x;
  const auto g = ddx::rt::derive_jacobian(b, f.id(b));
  const auto v = ddx::rt::evaluate_all(b, std::array{0.0});
  return v[g.partial[0]]; // d(x sin x)/dx at 0 is sin 0 + 0 cos 0
}
static_assert(transcendental_jacobian() == 0.0);
#endif

// Lowering a typed tree into a graph is a constant expression as well, so the
// two representations can be compared without running anything.
consteval bool bridge_agrees_with_equation() {
  constexpr auto x = ddx::var<"x">;
  constexpr auto y = ddx::var<"y">;
  Arena<> b;
  const auto root = ddx::rt::to_graph(b, x * y + x);
  const double viaGraph =
      ddx::rt::evaluate(b, root.id(b), std::array{3.0, 4.0});
  return viaGraph == ddx::Equation{x * y + x}.evaluate(3.0, 4.0);
}
static_assert(bridge_agrees_with_equation());

// Liveness is a property of the arena, so the pass the freeze runs is itself a
// constant expression -- what the freeze then does with it is not.
consteval std::size_t live_from_root() {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * x;
  (void)(x + y); // a node the root does not reach
  const ddx::rt::NodeId root = f.id(b);
  const auto live = ddx::rt::detail::reachable(b, std::span{&root, 1});
  return static_cast<std::size_t>(std::ranges::count(live, true));
}
static_assert(live_from_root() == 2); // x and x * x

// The colouring is Boost.Graph's, so it stops at the boundary -- but reading a
// Hessian back through one does not.
consteval ddx::rt::NodeId scatter_reads() {
  // Not const: the vectors are built during this evaluation, and a constant
  // evaluation may not write through a const object.
  ddx::rt::Hessian h{
      .value = 0,
      .partial = {},
      .compressed = {7, 8},
      .colouring = {.colour = {0, 0}, .scatter = {0, 1}, .cell = {0, 1}},
      .zero = 99};
  return h.colours() == 1 && h.at(0, 0) == 7 && h.at(1, 1) == 8 ? h.at(0, 1)
                                                                : 0;
}
static_assert(scatter_reads() == 99); // off the pattern: the zero node

// A colour that owns only one of its two rows: the unowned cell gets no
// storage at all, so `cells()` is short of `count() * n` and at() still
// answers.
consteval ddx::rt::NodeId unowned_cell_reads() {
  ddx::rt::Hessian h{.value = 0,
                     .partial = {},
                     .compressed = {7},
                     .colouring = {.colour = {0, 0},
                                   .scatter = {0, ddx::rt::no_column},
                                   .cell = {0, ddx::rt::no_column}},
                     .zero = 99};
  return h.at(0, 0) == 7 && h.at(1, 1) == 99 ? h.at(1, 0) : 0;
}
static_assert(unowned_cell_reads() == 99);

// The counts a caller sizes its buffers by come off the constructor's own
// sweep, so they answer without a frozen graph.
consteval std::size_t column_counts() {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * y, x + y);
  return *eq.value_columns() * 10 + *eq.jacobian_columns();
}
static_assert(column_counts() == 24); // 2 functions, 2 x 2 partials

TEST(RtConstexpr, EverythingAboveIsAStaticAssert) {
  EXPECT_DOUBLE_EQ(build_and_evaluate(), 13.0);
  EXPECT_DOUBLE_EQ(partial_of_product(), 4.0);
  EXPECT_EQ(nodes_for_repeated_subtree(), 5u);
  EXPECT_EQ(column_counts(), 24u);
}

} // namespace
