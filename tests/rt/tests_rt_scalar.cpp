#include "ddx.hpp"
#include "dual/taylor_dual.hpp"
#include "rt/derivative.hpp"
#include "rt/interpret.hpp"
#include <array>
#include <gtest/gtest.h>

#include <cmath>
#include <span>

// Numeric admits matrices and quaternions, so the rewrites that would assume a
// commuting product have to ask.  M2 is associative under + and *, commutative
// under + only.

namespace {

// A 2x2 matrix satisfying CFieldLike.
struct M2 {
  std::array<double, 4> a{};
  constexpr M2() = default;
  constexpr M2(int k) : a{double(k), 0, 0, double(k)} {} // k * identity
  constexpr M2(double x, double y, double z, double w) : a{x, y, z, w} {}
  friend constexpr M2 operator+(M2 l, M2 r) {
    return {l.a[0] + r.a[0], l.a[1] + r.a[1], l.a[2] + r.a[2], l.a[3] + r.a[3]};
  }
  friend constexpr M2 operator-(M2 l, M2 r) {
    return {l.a[0] - r.a[0], l.a[1] - r.a[1], l.a[2] - r.a[2], l.a[3] - r.a[3]};
  }
  friend constexpr M2 operator*(M2 l, M2 r) {
    return {
        l.a[0] * r.a[0] + l.a[1] * r.a[2], l.a[0] * r.a[1] + l.a[1] * r.a[3],
        l.a[2] * r.a[0] + l.a[3] * r.a[2], l.a[2] * r.a[1] + l.a[3] * r.a[3]};
  }
  friend constexpr M2 operator/(M2 l, M2 r) { // right division, l * r^-1
    const double d = r.a[0] * r.a[3] - r.a[1] * r.a[2];
    const M2 inv{r.a[3] / d, -r.a[1] / d, -r.a[2] / d, r.a[0] / d};
    return l * inv;
  }
  friend constexpr M2 operator-(M2 u) {
    return {-u.a[0], -u.a[1], -u.a[2], -u.a[3]};
  }
  friend constexpr bool operator==(M2 l, M2 r) = default;
};
// The one operator the library never reaches for: there is no Sub opcode, and a
// difference is built as the sum of a negation.  CFieldLike asks for it all the
// same, so it is here, and here is where it is checked to agree with the sum.
static_assert(M2{3} - M2{1} == M2{3} + -M2{1});

static_assert(ddx::impl::CFieldLike<M2>, "M2 is field-like");
static_assert(ddx::impl::Numeric<M2>, "so Numeric admits it");
static_assert(!ddx::impl::CCommutativeMultiply<M2>,
              "and its * does not commute");

TEST(RtScalar, CommutativeSwapIsAskedNotAssumed) {
  ddx::rt::Builder<M2> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  // A matrix product must not be canonicalised by swapping operands.
  EXPECT_NE((x * y).id(b), (y * x).id(b));

  ddx::rt::Builder<double> d;
  const auto dx = var(d, "x");
  const auto dy = var(d, "y");
  EXPECT_EQ((dx * dy).id(d), (dy * dx).id(d));
}

TEST(RtScalar, OnlyTheCancellationThatHoldsForAnyScalarFires) {
  ddx::rt::Builder<M2> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  // (n/d)*d is n whatever T is: `/` is right division, so d^-1 meets d.
  EXPECT_EQ(((y / x) * x).id(b), y.id(b));
  // d*(n/d) is d n d^-1, which is n only when the factors commute.
  EXPECT_NE((x * (y / x)).id(b), y.id(b));
}

TEST(RtScalar, EvaluatesInTheScalarsOwnArithmetic) {
  ddx::rt::Builder<M2> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const std::array<M2, 2> pt{M2{1, 2, 3, 4}, M2{5, 6, 7, 8}};
  const auto v = ddx::rt::evaluate(b, (x * y).id(b), pt);
  EXPECT_EQ(v, (M2{19, 22, 43, 50}));
}

// A transcendental over a matrix is meaningless, and must not stop the type
// being usable at all.
TEST(RtScalar, UnsupportedOpsDoNotBreakTheInstantiation) {
  ddx::rt::Builder<M2> b;
  const auto x = var(b, "x");
  // The assertion is that it instantiates and builds a node at all.
  EXPECT_NE((x + x * x - x / x).id(b), ddx::rt::no_node);
}

// What apply() cannot compute it answers with a zero, so a sweep asks first.
TEST(RtScalar, ComputesAtSaysWhichOpsTheScalarHas) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto field = (x + x * x - x / x).id(b);
  const auto kink = max(x, x * x).id(b);
  const auto wave = sin(x).id(b);
  EXPECT_TRUE(ddx::rt::computes_at<M2>(b, std::span{&field, 1}));
  EXPECT_FALSE(ddx::rt::computes_at<M2>(b, std::span{&kink, 1}));
  EXPECT_FALSE(ddx::rt::computes_at<M2>(b, std::span{&wave, 1}));
  using Taylor = ddx::impl::TaylorDual<double, 3>;
  EXPECT_TRUE(ddx::rt::computes_at<Taylor>(b, std::span{&kink, 1}));
  EXPECT_TRUE(ddx::rt::computes_at<Taylor>(b, std::span{&wave, 1}));
  EXPECT_TRUE(ddx::rt::supports<M2>(ddx::rt::OpCode::Mul));
  EXPECT_FALSE(ddx::rt::supports<M2>(ddx::rt::OpCode::Max));
  EXPECT_TRUE(ddx::rt::supports<Taylor>(ddx::rt::OpCode::Max));
}

} // namespace

namespace {

// Forward mode falls out of the interpreter rather than being a second engine:
// evaluate_all takes its arithmetic from the point's element type.
TEST(RtScalar, ADualPointCarriesDerivativesThroughTheSameWalk) {
  ddx::rt::Builder<> b; // the graph itself is over double
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y);

  using D = ddx::dual;
  const std::array<D, 2> seeded{D{1.0, 1.0}, D{2.0, 0.0}}; // seed x
  const auto v = ddx::rt::evaluate(b, f.id(b), seeded);

  EXPECT_NEAR(v.value(), std::exp(1.0) * std::sin(2.0), 1e-14);
  EXPECT_NEAR(v.deriv(), std::exp(1.0) * std::sin(2.0), 1e-14);

  // And it agrees with the reverse sweep over the same graph.
  const auto g = ddx::rt::derive_jacobian(b, f.id(b));
  const auto plain = ddx::rt::evaluate_all(b, std::array{1.0, 2.0});
  EXPECT_NEAR(plain[g.partial[0]], v.deriv(), 1e-14);
}

} // namespace
