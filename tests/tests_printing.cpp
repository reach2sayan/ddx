#include "tests_printing_fixtures.hpp"

// A unary node must name its operator, not print as its bare child.
TEST(ExpressionPrinting, UnaryNodesNameTheirOperator) {
  EXPECT_EQ(std::format("{}", sin(px)), "sin(x)");
  EXPECT_EQ(std::format("{}", -px), "-x");
  EXPECT_EQ(std::format("{}", abs(px)), "abs(x)");
  EXPECT_EQ(std::format("{}", exp(px * py)), "exp(x * y)");
}
TEST(ExpressionPrinting, ParenthesesOnlyWherePrecedenceDemands) {
  EXPECT_EQ(std::format("{}", px * py + sin(px)), "x * y + sin(x)");
  EXPECT_EQ(std::format("{}", (px + py) * pz), "(x + y) * z");
  EXPECT_EQ(std::format("{}", px / (py / pz)), "x / (y / z)");
  EXPECT_EQ(std::format("{}", px / py / pz), "x / y / z");
  EXPECT_EQ(std::format("{}", -(px + py)), "-(x + y)");
  EXPECT_EQ(std::format("{}", -px * py), "-x * y");
}
// a - b is built as a + (-b); without the peephole every subtraction in the
// library would read "x + -y".
TEST(ExpressionPrinting, SubtractionSurvivesItsLowering) {
  EXPECT_EQ(std::format("{}", px - py), "x - y");
  EXPECT_EQ(std::format("{}", px - (py + pz)), "x - (y + z)");
  EXPECT_EQ(std::format("{}", px - py * pz), "x - y * z");
  EXPECT_EQ(std::format("{}", 2.0 * px * py + sin(px) - pz),
            "2 * x * y + sin(x) - z");
}
TEST(ExpressionPrinting, FunctionStyleOps) {
  EXPECT_EQ(std::format("{}", pow(px, 2.0)), "pow(x, 2)");
  EXPECT_EQ(std::format("{}", atan2(py, px)), "atan2(y, x)");
  EXPECT_EQ(std::format("{}", hypot(px, py)), "hypot(x, y)");
  EXPECT_EQ(std::format("{}", max(px, py + pz)), "max(x, y + z)");
}
// The only leaf needing a mark is the frozen variable: it prints as a name but
// differentiates to zero, which is what "_c" says.
TEST(ExpressionPrinting, FrozenVariablesPrintAsConstants) {
  EXPECT_EQ(std::format("{}", px * py), "x * y");
  EXPECT_EQ(
      std::format(
          "{}", make_all_constant_except<ddx::impl::FixedString{"x"}>(px * py)),
      "x * y_c");
}
TEST(ExpressionPrinting, ValueSpecReachesEveryNumber) {
  EXPECT_EQ(std::format("{::.3f}", 2.0 * px), "2.000 * x");
  EXPECT_EQ(std::format("{::.1e}", 2.0 * px + 3.0), "2.0e+00 * x + 3.0e+00");
  // The leading ':' is optional -- the whole spec is the value-spec.
  EXPECT_EQ(std::format("{:.3f}", 2.0 * px), "2.000 * x");
}
TEST(ExpressionPrinting, StreamInserterMatchesFormat) {
  const auto e = px * py + sin(px) - pz;
  std::ostringstream oss;
  oss << e;
  EXPECT_EQ(oss.str(), std::format("{}", e));
}

TEST(ExpressionPrinting, EquationPrintsFunctionsAndJacobianRows) {
  auto eq = ddx::Equation{px * py};
  const auto text = std::format("{}", eq);
  EXPECT_EQ(text, "f0: x * y\n  jac: y, x\n");
}
