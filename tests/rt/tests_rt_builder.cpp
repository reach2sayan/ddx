#include "rt/expressions.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

// The builder folds with ids: a rewrite returns an existing node, so the
// assertions here are on node identity rather than node count.

namespace {
using ddx::rt::Builder;
using RE = ddx::rt::Expression<>;

TEST(RtBuilder, IdenticalSubtreesInternToOneNode) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y);
  const std::size_t after_first = b.size();
  const auto g = exp(x) * sin(y);

  EXPECT_EQ(f.id(b), g.id(b));
  EXPECT_EQ(b.size(), after_first);
}

TEST(RtBuilder, ReserveChangesNeitherIdsNorInterning) {
  const auto build = [](Builder<> &b) {
    const auto x = var(b, "x");
    const auto y = var(b, "y");
    auto f = exp(x) * sin(y);
    for (int k = 1; k < 200; ++k) {
      f += pow(x, k) / (y + k);
    }
    return f.id(b);
  };
  Builder<> plain;
  Builder<> reserved;
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

TEST(RtBuilder, CommutativeOperandsCanonicalise) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  EXPECT_EQ((x * y).id(b), (y * x).id(b));
  EXPECT_EQ((x + y).id(b), (y + x).id(b));
  EXPECT_NE((x / y).id(b), (y / x).id(b));
}

TEST(RtBuilder, IdentityRewrites) {
  Builder<> b;
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

  // (n/d)*d -> n, the DAG form of the compile-time rule: on interned nodes the
  // denominator match is an id compare.
  EXPECT_EQ(((y / x) * x).id(b), y.id(b));
  EXPECT_EQ((x * (y / x)).id(b), y.id(b));
}

TEST(RtBuilder, LiteralsFoldBeforeReachingTheGraph) {
  Builder<> b;
  const auto x = var(b, "x");
  const std::size_t before = b.size();
  const auto c = RE{2} * RE{3} + RE{1};
  EXPECT_TRUE(c.pending());
  EXPECT_DOUBLE_EQ(c.literal(), 7.0);
  EXPECT_EQ(b.size(), before);

  EXPECT_TRUE(b.is_constant((RE{2} * RE{3}).id(b), 6.0));
  (void)x;
}

TEST(RtBuilder, ConstantOperandsFold) {
  Builder<> b;
  const auto x = var(b, "x");
  EXPECT_TRUE(b.is_constant((lit(b, 2.0) * lit(b, 3.0)).id(b), 6.0));
  EXPECT_TRUE(b.is_constant(sqrt(lit(b, 9.0)).id(b), 3.0));
  (void)x;
}

// Constants intern by their bits, which is what they hash by: the two zeros
// stay apart whatever was built first, and a NaN is one node.
TEST(RtBuilder, ConstantsInternByTheirBits) {
  Builder<> b;
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

TEST(RtBuilder, SymbolsInternByName) {
  Builder<> b;
  EXPECT_EQ(var(b, "x").id(b), var(b, "x").id(b));
  EXPECT_NE(var(b, "x").id(b), var(b, "y").id(b));
  ASSERT_EQ(b.symbols().size(), 2u);
  EXPECT_EQ(b.symbols()[0], "x");
}

// A slot is the order the model named its symbols in, so a symbol named late
// takes the next slot and moves nothing already built.
TEST(RtBuilder, SlotsAreDeclarationOrder) {
  Builder<> b;
  const auto z = var(b, "z");
  const auto s = sin(z); // built while z is the only symbol
  const auto a = var(b, "a");

  EXPECT_EQ(b.symbols(), (std::vector<std::string>{"z", "a"}));
  EXPECT_EQ(b[z.id(b)].aux, 0u);
  EXPECT_EQ(b[a.id(b)].aux, 1u);
  EXPECT_EQ(b[s.id(b)].a, z.id(b)) << "the subexpression still names z";
}

// One namespace for every named input: a parameter cannot take a symbol's name,
// and parameters number their own slots after the symbols' columns.
TEST(RtBuilder, ParamsShareTheNamespaceAndNumberTheirOwnSlots) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto k = ddx::rt::param(b, "k");
  const auto y = var(b, "y");

  EXPECT_EQ(b.symbols(), (std::vector<std::string>{"x", "y"}));
  EXPECT_EQ(b.params(), (std::vector<std::string>{"k"}));
  EXPECT_EQ(b[k.id(b)].op, ddx::rt::OpCode::Param);
  EXPECT_EQ(b[k.id(b)].aux, 0u);
  EXPECT_EQ(b[y.id(b)].aux, 1u);
  EXPECT_EQ(ddx::rt::param(b, "k").id(b), k.id(b));
  EXPECT_TRUE(ddx::rt::param(b, "x").poisoned()) << "x is a symbol";
  EXPECT_TRUE(var(b, "k").poisoned()) << "k is a parameter";
  EXPECT_EQ(ddx::rt::input_column(b.inputs(), ddx::rt::OpCode::Param, 0), 2u);
  (void)x;
}

// Compound assignment rebinds the handle through the same operators, so an
// accumulator written either way lands on the same node -- which interning
// makes an id comparison.
// Two terms are the long form's own Add; the other compound operators build
// what their operator builds.  Past two terms `+=` is a SumN: tests_rt_sumn.
TEST(RtBuilder, CompoundAssignmentBuildsWhatTheLongFormBuilds) {
  Builder<> b;
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

// The right operand converts, as it does for the binary operators, and two
// pending literals fold without an arena to fold in.
TEST(RtBuilder, CompoundAssignmentTakesAScalar) {
  Builder<> b;
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
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = (x * y) + sin(x * y);
  const std::array<double, 2> pt{0.5, 1.5};
  const double xy = 0.75;
  EXPECT_NEAR(ddx::rt::evaluate(b, f.id(b), pt), xy + std::sin(xy), 1e-15);
}

// A literal condition has chosen, NaN reading as true the way select_impl
// reads it, and equal arms need no choosing.
TEST(RtBuilder, SelectFoldsWhereABinaryWould) {
  ddx::rt::Builder<> b;
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

// Two comparison opcodes carry six operators: each is `lt` or `le`, some read
// the other way round, and a bare number converts on either side.
TEST(RtBuilder, ComparisonOperatorsAreTheTwoNodes) {
  ddx::rt::Builder<> b;
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

} // namespace
