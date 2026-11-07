#include "rt/expressions.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <type_traits>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// The arena folds with ids: a rewrite returns an existing node, so the
// assertions here are on node identity rather than node count.

namespace {
using ddx::rt::Arena;
using RE = ddx::rt::Expression<>;

TEST(RtArena, IdenticalSubtreesInternToOneNode) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y);
  const std::size_t after_first = b.size();
  const auto g = exp(x) * sin(y);

  EXPECT_EQ(f.id(b), g.id(b));
  EXPECT_EQ(b.size(), after_first);
}

TEST(RtArena, ReserveChangesNeitherIdsNorInterning) {
  const auto build = [](Arena<> &b) {
    const auto x = var(b, "x");
    const auto y = var(b, "y");
    auto f = exp(x) * sin(y);
    for (int k = 1; k < 200; ++k) {
      f += pow(x, k) / (y + k);
    }
    return f.id(b);
  };
  Arena<> plain;
  Arena<> reserved;
  reserved.reserve(4096);
  EXPECT_EQ(reserved.size(), 0u);
  const auto f = build(plain);
  const auto g = build(reserved);

  EXPECT_EQ(f, g);
  EXPECT_EQ(plain.size(), reserved.size());
  EXPECT_EQ(exp(var(plain, "x")).id(plain),
            exp(var(reserved, "x")).id(reserved));
  EXPECT_EQ(plain.size(), reserved.size());
}

TEST(RtArena, CommutativeOperandsCanonicalise) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  EXPECT_EQ((x * y).id(b), (y * x).id(b));
  EXPECT_EQ((x + y).id(b), (y + x).id(b));
  EXPECT_NE((x / y).id(b), (y / x).id(b));
}

TEST(RtArena, IdentityRewrites) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");

  EXPECT_EQ((x + RE{0}).id(b), x.id(b));
  EXPECT_EQ((x * RE{1}).id(b), x.id(b));
  EXPECT_EQ((x / RE{1}).id(b), x.id(b));
  EXPECT_EQ((-(-x)).id(b), x.id(b));
  EXPECT_EQ(pow(x, RE{1}).id(b), x.id(b));

  EXPECT_TRUE(b.is_constant((x * RE{0}).id(b), 0.0));
  EXPECT_TRUE(b.is_constant((RE{0} / x).id(b), 0.0));
  EXPECT_TRUE(b.is_constant((x / x).id(b), 1.0));
  EXPECT_TRUE(b.is_constant(pow(x, RE{0}).id(b), 1.0));

  // (n/d)*d folds back to n by comparing the denominator's id.
  EXPECT_EQ(((y / x) * x).id(b), y.id(b));
  EXPECT_EQ((x * (y / x)).id(b), y.id(b));
}

TEST(RtArena, LiteralsFoldBeforeReachingTheGraph) {
  Arena<> b;
  const auto x = var(b, "x");
  const std::size_t before = b.size();
  const auto c = RE{2} * RE{3} + RE{1};
  EXPECT_TRUE(c.pending());
  EXPECT_DOUBLE_EQ(c.literal(), 7.0);
  EXPECT_EQ(b.size(), before);

  EXPECT_TRUE(b.is_constant((RE{2} * RE{3}).id(b), 6.0));
  (void)x;
}

TEST(RtArena, ConstantOperandsFold) {
  Arena<> b;
  const auto x = var(b, "x");
  EXPECT_TRUE(b.is_constant((lit(b, 2.0) * lit(b, 3.0)).id(b), 6.0));
  EXPECT_TRUE(b.is_constant(sqrt(lit(b, 9.0)).id(b), 3.0));
  (void)x;
}

// Constants intern by bit pattern, so +0.0 and -0.0 stay apart and a NaN is
// a single node.
TEST(RtArena, ConstantsInternByTheirBits) {
  Arena<> b;
  for (int i = 0; i < 64; ++i) {
    (void)lit(b, 1.0 + i).id(b);
  }
  const auto plus = lit(b, 0.0).id(b);
  const auto minus = lit(b, -0.0).id(b);
  EXPECT_NE(plus, minus);
  EXPECT_TRUE(std::signbit(b[minus].value));
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(lit(b, nan).id(b), lit(b, nan).id(b));
}

TEST(RtArena, SymbolsInternByName) {
  Arena<> b;
  EXPECT_EQ(var(b, "x").id(b), var(b, "x").id(b));
  EXPECT_NE(var(b, "x").id(b), var(b, "y").id(b));
  ASSERT_EQ(b.symbols().size(), 2u);
  EXPECT_EQ(b.symbols()[0], "x");
}

// A symbol named late takes the next slot and moves nothing already built.
TEST(RtArena, SlotsAreDeclarationOrder) {
  Arena<> b;
  const auto z = var(b, "z");
  const auto s = sin(z); // built while z is the only symbol
  const auto a = var(b, "a");

  EXPECT_EQ(b.symbols(), (std::vector<std::string>{"z", "a"}));
  EXPECT_EQ(ddx::rt::detail::get<ddx::rt::Slot>(b[z.id(b)]).value, 0u);
  EXPECT_EQ(ddx::rt::detail::get<ddx::rt::Slot>(b[a.id(b)]).value, 1u);
  EXPECT_EQ(b[s.id(b)].in[0], z.id(b)) << "the subexpression still names z";
}

// A parameter cannot reuse a symbol's name, and parameters number their slots
// after the symbols' columns.
TEST(RtArena, ParamsShareTheNamespaceAndNumberTheirOwnSlots) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto k = ddx::rt::param(b, "k");
  const auto y = var(b, "y");

  EXPECT_EQ(b.symbols(), (std::vector<std::string>{"x", "y"}));
  EXPECT_EQ(b.params(), (std::vector<std::string>{"k"}));
  EXPECT_EQ(b[k.id(b)].op, ddx::rt::OpCode::Param);
  EXPECT_EQ(ddx::rt::detail::get<ddx::rt::Slot>(b[k.id(b)]).value, 0u);
  EXPECT_EQ(ddx::rt::detail::get<ddx::rt::Slot>(b[y.id(b)]).value, 1u);
  EXPECT_EQ(ddx::rt::param(b, "k").id(b), k.id(b));
  EXPECT_TRUE(ddx::rt::param(b, "x").poisoned()) << "x is a symbol";
  EXPECT_TRUE(var(b, "k").poisoned()) << "k is a parameter";
  EXPECT_EQ(ddx::rt::input_column(b.inputs(), ddx::rt::OpCode::Param,
                                  ddx::rt::Slot{0}),
            2u);
  (void)x;
}

// Ops whose operands differ in meaning take named operands. Positional
// operands do not compile, and the names follow the opcode table's order.
namespace named {
using ddx::rt::NodeId;
using ddx::rt::OpCode;
using ddx::rt::opcode;
// Detects whether build(arena, u, m) compiles.
template <typename Build>
constexpr bool compiles = std::is_invocable_v<Build, Arena<> &, NodeId, NodeId>;
static_assert(!compiles<decltype([](auto &b, auto u, auto m)
                                     -> decltype(b.make(
                                         opcode<OpCode::RowScale>,
                                         ddx::rt::NoPayload{}, u, m)) {
  return {};
})>);
static_assert(compiles<decltype([](auto &b, auto u, auto m)
                                    -> decltype(b.make(
                                        opcode<OpCode::RowScale>,
                                        {.scale = u, .matrix = m})) {
  return {};
})>);
static_assert(!compiles<decltype([](auto &b, auto u, auto m)
                                     -> decltype(b.make(
                                         opcode<OpCode::RowScale>,
                                         {.matrix = m, .scale = u})) {
  return {};
})>);
static_assert(compiles<decltype([](auto &b, auto u, auto m)
                                    -> decltype(b.make(
                                        opcode<OpCode::ColScale>,
                                        {.matrix = m, .scale = u})) {
  return {};
})>);
} // namespace named

// Compound assignment builds the same node as the long form, which interning
// makes an id comparison. Past two terms, += builds a SumN instead.
TEST(RtArena, CompoundAssignmentBuildsWhatTheLongFormBuilds) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");

  auto acc = x;
  acc += y;
  acc *= exp(y);
  acc /= (x + y);
  acc -= sin(x);
  const auto id = acc.id(b); // a sum is formed when it is first read
  const std::size_t after = b.size();

  const auto spelled = ((x + y) * exp(y)) / (x + y) - sin(x);
  EXPECT_EQ(id, spelled.id(b));
  EXPECT_EQ(b.size(), after)
      << "the long form built a node the short one did not";
}

// The right operand converts as it does for the binary operators, and two
// literals fold without needing an arena.
TEST(RtArena, CompoundAssignmentTakesAScalar) {
  Arena<> b;
  const auto x = var(b, "x");

  auto e = x;
  e += 2.0;
  EXPECT_EQ(e.id(b), (x + RE{2.0}).id(b));

  RE lit = 3.0;
  lit *= 4.0;
  EXPECT_TRUE(lit.pending()) << "two literals reached the graph";
  EXPECT_EQ(lit.literal(), 12.0);
}

TEST(RtInterpreter, EvaluatesSharedNodesOnce) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = (x * y) + sin(x * y);
  const std::array<double, 2> pt{0.5, 1.5};
  const double xy = 0.75;
  EXPECT_NEAR(ddx::rt::evaluate(b, f.id(b), pt), xy + std::sin(xy), 1e-15);
}

// A literal condition picks its arm (NaN reads as true, as in select_impl),
// and equal arms need no select at all.
TEST(RtArena, SelectFoldsWhereABinaryWould) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  EXPECT_EQ(select(ddx::rt::lit(b, 1.0), x, y).id(b), x.id(b));
  EXPECT_EQ(select(ddx::rt::lit(b, 0.0), x, y).id(b), y.id(b));
  EXPECT_EQ(select(ddx::rt::lit(b, -0.0), x, y).id(b), y.id(b));
  EXPECT_EQ(select(ddx::rt::lit(b, std::nan("")), x, y).id(b), x.id(b));
  EXPECT_EQ(select(lt(x, y), x, x).id(b), x.id(b));
  const auto kept = select(lt(x, y), x, y);
  EXPECT_EQ(b[kept.id(b)].op, ddx::rt::OpCode::Select);
}

// The six comparison operators build only lt and le nodes, some with their
// operands swapped, and a bare number converts on either side.
TEST(RtArena, ComparisonOperatorsAreTheTwoNodes) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  EXPECT_EQ((x < y).id(b), lt(x, y).id(b));
  EXPECT_EQ((x <= y).id(b), le(x, y).id(b));
  EXPECT_EQ((x > y).id(b), lt(y, x).id(b));
  EXPECT_EQ((x >= y).id(b), le(y, x).id(b));
  EXPECT_EQ((x == y).id(b), (le(x, y) * le(y, x)).id(b));
  EXPECT_EQ((x != y).id(b), (1.0 - (x == y)).id(b));
  EXPECT_EQ((1.0 < x).id(b), (x > 1.0).id(b));
  EXPECT_EQ((x < 1).id(b), (x < 1.0).id(b));
  EXPECT_EQ(b[(x < y).id(b)].op, ddx::rt::OpCode::Lt);
  EXPECT_EQ(b[(x >= y).id(b)].op, ddx::rt::OpCode::Le);
}

// Float constants also intern by bits, so their two zeros stay apart both as
// scalars and inside a table.
TEST(RtArena, FloatConstantsInternByTheirBits) {
  Arena<float> b;
  const auto plus = lit(b, 0.0f).id(b);
  const auto minus = lit(b, -0.0f).id(b);
  EXPECT_NE(plus, minus);
  EXPECT_TRUE(std::signbit(b[minus].value));
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(lit(b, nan).id(b), lit(b, nan).id(b));

  const std::array<float, 2> zeros{0.0f, 0.0f};
  const std::array<float, 2> negative_zeros{-0.0f, -0.0f};
  EXPECT_NE(ddx::rt::vec_constant<float>(b, zeros).id(b),
            ddx::rt::vec_constant<float>(b, negative_zeros).id(b));
}

// Handles from two arenas cannot be combined, since a node built from both
// would read one handle against the wrong arena.
TEST(RtArena, HandlesOfTwoArenasDoNotCompose) {
  Arena<> a;
  Arena<> b;
  const auto x = var(a, "x");
  const auto y = var(b, "y");
  EXPECT_TRUE((x * y).poisoned());
  EXPECT_TRUE(select(x, x, y).poisoned());
  auto accumulated = x;
  accumulated += y;
  EXPECT_TRUE(accumulated.poisoned());
  EXPECT_FALSE((x * 2.0).poisoned());
}

} // namespace
