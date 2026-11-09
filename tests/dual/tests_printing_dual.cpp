#include "dual/tests_dual_common.hpp"
#include "tests_printing_fixtures.hpp"

// A dual-valued constant is a reachable leaf, so the leaf printer must handle a
// value_type that is not arithmetic.
TEST(ExpressionPrinting, DualValuedLeavesPrint) {
  const ddx::impl::Constant<ddx::impl::Dual<double>> c{
      ddx::impl::Dual{1.5, 2.0}};
  EXPECT_EQ(std::format("{}", c), "1.5+2ε");
  EXPECT_EQ(std::format("{::.2f}", c), "1.50+2.00ε");
}

// All duals share one formatter base in util/fmt.hpp, so a spec means the same
// in each and iostream output matches std::format.
TEST(DualPrinting, EachDualRendersItsOwnShape) {
  const ddx::impl::Dual<double> d{1.5, 2.0};
  ddx::impl::TaylorDual<double, 3> t;
  t.c = {1.0, 2.0, 3.0, 4.0};

  EXPECT_EQ(std::format("{}", d), "1.5+2ε");
  EXPECT_EQ(std::format("{}", t), "1+2ε+3ε^2+4ε^3");
}
TEST(DualPrinting, OneSpecReachesEveryPart) {
  const ddx::impl::Dual<double> d{1.5, 2.0};
  ddx::impl::TaylorDual<double, 2> t;
  t.c = {1.0, 2.0, 3.0};

  EXPECT_EQ(std::format("{:.2f}", d), "1.50+2.00ε");
  EXPECT_EQ(std::format("{:.2f}", t), "1.00+2.00ε+3.00ε^2");
}
// Without bracketing, Dual<Dual<double>> reads as one four-term series.
TEST(DualPrinting, NestedCoefficientsAreBracketed) {
  const ddx::impl::Dual<ddx::impl::Dual<double>> d{ddx::impl::Dual{1.0, 2.0},
                                                   ddx::impl::Dual{3.0, 4.0}};
  EXPECT_EQ(std::format("{}", d), "(1+2ε)+(3+4ε)ε");
}
TEST(DualPrinting, StreamInserterMatchesFormat) {
  const ddx::impl::Dual<double> d{1.5, 2.0};
  ddx::impl::TaylorDual<double, 2> t;
  t.c = {1.0, 2.0, 3.0};

  std::ostringstream oss;
  oss << d << '|' << t;
  EXPECT_EQ(oss.str(), std::format("{}|{}", d, t));
}
// Every dual is reachable as an expression leaf, so both must be formattable.
TEST(DualPrinting, AllThreeWorkAsExpressionLeaves) {
  EXPECT_EQ(std::format("{}",
                        ddx::impl::Constant<ddx::impl::Dual<double>>{
                            ddx::impl::Dual{5.0, 1.0}}),
            "5+1ε");
  EXPECT_EQ(std::format("{}",
                        ddx::impl::Constant<ddx::impl::TaylorDual<double, 2>>{
                            ddx::impl::TaylorDual<double, 2>{5.0}}),
            "5+0ε+0ε^2");
}
