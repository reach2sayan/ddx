#include "tests_core_fixtures.hpp"

TEST(MathFunctionTest, TanEvalAndDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(tan(x).eval(x0), std::tan(x0));
  ASSERT_DOUBLE_EQ(tan(x).derivative().eval(x0),
                   1.0 / (std::cos(x0) * std::cos(x0)));
}
TEST(MathFunctionTest, LogEvalAndDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(log(x).eval(x0), std::log(x0));
  ASSERT_DOUBLE_EQ(log(x).derivative().eval(x0), 1.0 / x0);
}
TEST(MathFunctionTest, SqrtEvalAndDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(sqrt(x).eval(x0), std::sqrt(x0));
  ASSERT_DOUBLE_EQ(sqrt(x).derivative().eval(x0), 0.5 / std::sqrt(x0));
}
TEST(MathFunctionTest, AsinDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(asin(x).eval(x0), std::asin(x0));
  ASSERT_DOUBLE_EQ(asin(x).derivative().eval(x0),
                   1.0 / std::sqrt(1.0 - x0 * x0));
}
TEST(MathFunctionTest, AcosDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(acos(x).eval(x0), std::acos(x0));
  ASSERT_DOUBLE_EQ(acos(x).derivative().eval(x0),
                   -1.0 / std::sqrt(1.0 - x0 * x0));
}
TEST(MathFunctionTest, AtanDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(atan(x).eval(x0), std::atan(x0));
  ASSERT_DOUBLE_EQ(atan(x).derivative().eval(x0), 1.0 / (1.0 + x0 * x0));
}
TEST(MathFunctionTest, SinhDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(sinh(x).eval(x0), std::sinh(x0));
  ASSERT_DOUBLE_EQ(sinh(x).derivative().eval(x0), std::cosh(x0));
}
TEST(MathFunctionTest, CoshDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(cosh(x).eval(x0), std::cosh(x0));
  ASSERT_DOUBLE_EQ(cosh(x).derivative().eval(x0), std::sinh(x0));
}
TEST(MathFunctionTest, TanhDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(tanh(x).eval(x0), std::tanh(x0));
  double c = std::cosh(x0);
  ASSERT_DOUBLE_EQ(tanh(x).derivative().eval(x0), 1.0 / (c * c));
}
TEST(MathFunctionTest, AbsEval) {
  ASSERT_DOUBLE_EQ(abs(var<"x">).eval(3.0), 3.0);
  ASSERT_DOUBLE_EQ(abs(var<"x">).eval(-3.0), 3.0);
  ASSERT_DOUBLE_EQ(abs(var<"x">).eval(0.0), 0.0);
}
TEST(MathFunctionTest, ChainRuleTan) {
  double x0 = 0.4;
  auto x = var_of<"x">(x0);
  auto expr = tan(constant(2.0) * x);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(x0),
                   2.0 / (std::cos(2.0 * x0) * std::cos(2.0 * x0)));
}
TEST(MathFunctionTest, ChainRuleLog) {
  double x0 = 0.8;
  auto x = var_of<"x">(x0);
  auto expr = log(x * x);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(x0), 2.0 / x0);
}
TEST(MathFunctionTest, ChainRuleSqrt) {
  double x0 = 1.0;
  auto x = var_of<"x">(x0);
  auto expr = sqrt(sin(x));
  double expected = std::cos(x0) / (2.0 * std::sqrt(std::sin(x0)));
  ASSERT_NEAR(expr.derivative().eval(x0), expected, 1e-12);
}
TEST(MathFunctionTest, PythagoreanIdentity) {
  for (double v : {0.0, 0.5, 1.0, 2.0}) {
    auto x = var_of<"x">(v);
    auto expr = sin(x) * sin(x) + cos(x) * cos(x);
    ASSERT_NEAR(expr.eval(v), 1.0, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(v), 0.0, 1e-12);
  }
}
TEST(MathFunctionTest, HyperbolicIdentity) {
  for (double v : {0.0, 0.5, 1.0, 2.0}) {
    auto x = var_of<"x">(v);
    auto expr = cosh(x) * cosh(x) - sinh(x) * sinh(x);
    ASSERT_NEAR(expr.eval(v), 1.0, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(v), 0.0, 1e-12);
  }
}
TEST(MathFunctionTest, ExpLogIdentity) {
  for (double v : {0.3, 0.5, 1.0, 2.0}) {
    auto x = var_of<"x">(v);
    auto expr = exp(log(x));
    ASSERT_NEAR(expr.eval(v), v, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(v), 1.0, 1e-12);
  }
}
TEST(MathFunctionTest, QuotientSelfIsConstant) {
  auto x = var_of<"x">(2.0);
  auto expr = x / x;
  // x/x folds to the literal 1, so it names no symbol and takes no point.
  static_assert(std::same_as<decltype(expr), ddx::impl::Lit<double, 1>>,
                "x / x folds to the literal 1 while the tree is built");
  ASSERT_NEAR(expr.eval(), 1.0, 1e-12);
  ASSERT_NEAR(expr.derivative().eval(), 0.0, 1e-12);
}
TEST(MathFunctionTest, TanEqualsRatio) {
  double x0 = 0.7;
  auto x1 = var_of<"x">(x0);
  auto x2 = var_of<"x">(x0);
  ASSERT_NEAR(tan(x1).derivative().eval(x0),
              (sin(x2) / cos(x2)).derivative().eval(x0), 1e-12);
}
TEST(ReverseModeAD, TanDerivative) {
  auto x = var<"x">;
  auto g = Equation{tan(x)}.jacobian(0.5);
  ASSERT_DOUBLE_EQ(g[0], 1.0 / (std::cos(0.5) * std::cos(0.5)));
}
TEST(ReverseModeAD, LogDerivative) {
  auto x = var<"x">;
  auto g = Equation{log(x)}.jacobian(0.5);
  ASSERT_DOUBLE_EQ(g[0], 2.0);
}
TEST(ReverseModeAD, SqrtDerivative) {
  auto x = var<"x">;
  auto g = Equation{sqrt(x)}.jacobian(4.0);
  ASSERT_DOUBLE_EQ(g[0], 0.25);
}
TEST(ReverseModeAD, AsinDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{asin(x)}.jacobian(x0);
  ASSERT_DOUBLE_EQ(g[0], 1.0 / std::sqrt(1.0 - x0 * x0));
}
TEST(ReverseModeAD, AcosDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{acos(x)}.jacobian(x0);
  ASSERT_DOUBLE_EQ(g[0], -1.0 / std::sqrt(1.0 - x0 * x0));
}
TEST(ReverseModeAD, AtanDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{atan(x)}.jacobian(x0);
  ASSERT_DOUBLE_EQ(g[0], 1.0 / (1.0 + x0 * x0));
}
TEST(ReverseModeAD, SinhDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{sinh(x)}.jacobian(x0);
  ASSERT_DOUBLE_EQ(g[0], std::cosh(x0));
}
TEST(ReverseModeAD, CoshDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{cosh(x)}.jacobian(x0);
  ASSERT_DOUBLE_EQ(g[0], std::sinh(x0));
}
TEST(ReverseModeAD, TanhDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{tanh(x)}.jacobian(x0);
  double c = std::cosh(x0);
  ASSERT_DOUBLE_EQ(g[0], 1.0 / (c * c));
}
TEST(ReverseModeAD, AbsDerivativePositive) {
  auto x = var<"x">;
  ASSERT_DOUBLE_EQ(Equation{abs(x)}.jacobian(1.0)[0], 1.0);
}
TEST(ReverseModeAD, AbsDerivativeNegative) {
  auto x = var<"x">;
  ASSERT_DOUBLE_EQ(Equation{abs(x)}.jacobian(-1.0)[0], -1.0);
}
TEST(ReverseModeAD, AbsDerivativeAtZero) {
  auto x = var<"x">;
  ASSERT_DOUBLE_EQ(Equation{abs(x)}.jacobian(0.0)[0], 0.0);
}
TEST(ConceptTest, NumericSatisfied) {
  static_assert(Numeric<int>);
  static_assert(Numeric<double>);
  static_assert(Numeric<float>);
  static_assert(Numeric<long double>);
  static_assert(!Numeric<std::string>);
}
TEST(ConceptTest, ExpressionConceptSatisfied) {
  static_assert(CExpression<Constant<double>>);
  static_assert(CExpression<Variable<double, ddx::impl::FixedString{"x"}>>);
  using SumExpr =
      decltype(std::declval<Variable<double, ddx::impl::FixedString{"x"}>>() +
               std::declval<Constant<double>>());
  static_assert(CExpression<SumExpr>);
  static_assert(!CExpression<int>);
  static_assert(!CExpression<double>);
}
TEST(ConceptTest, AnOpSatisfied) {
  static_assert(COperation<SumOp<double>>);
  static_assert(COperation<MultiplyOp<float>>);
  static_assert(COperation<SineOp<double>>);
  static_assert(COperation<CosineOp<double>>);
  static_assert(COperation<ExpOp<double>>);
  static_assert(COperation<NegateOp<int>>);
  static_assert(COperation<DivideOp<double>>);
}
TEST(SymbolTest, SingleVariable) {
  using E = Variable<double, ddx::impl::FixedString{"x"}>;
  using Syms = extract_symbols_from_expr_t<E>;
  static_assert(ddx::meta::size(Syms{}) == 1);
  static_assert(
      ddx::meta::at<0>(Syms{}) ==
                     ddx::meta::type_c<ddx::impl::symbol_type<ddx::impl::FixedString{"x"}>>);
}
TEST(SymbolTest, TwoVariables) {
  using E =
      decltype(std::declval<Variable<double, ddx::impl::FixedString{"x"}>>() *
               std::declval<Variable<double, ddx::impl::FixedString{"y"}>>());
  using Syms = extract_symbols_from_expr_t<E>;
  static_assert(ddx::meta::size(Syms{}) == 2);
  // Symbols are sorted lexicographically, so "x" comes before "y".
  static_assert(
      ddx::meta::at<0>(Syms{}) ==
                     ddx::meta::type_c<ddx::impl::symbol_type<ddx::impl::FixedString{"x"}>>);
  static_assert(
      ddx::meta::at<1>(Syms{}) ==
                     ddx::meta::type_c<ddx::impl::symbol_type<ddx::impl::FixedString{"y"}>>);
}
TEST(SymbolTest, DuplicateSymbolsDeduplicated) {
  using E =
      decltype(std::declval<Variable<double, ddx::impl::FixedString{"x"}>>() *
               std::declval<Variable<double, ddx::impl::FixedString{"x"}>>());
  using Syms = extract_symbols_from_expr_t<E>;
  static_assert(ddx::meta::size(Syms{}) == 1);
}
TEST(SymbolTest, ThreeVariables) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  auto expr = x + y + z;
  using Syms = extract_symbols_from_expr_t<decltype(expr)>;
  static_assert(ddx::meta::size(Syms{}) == 3);
}
// Symbols come back sorted and deduplicated however they are spread, or a
// Jacobian would carry the right numbers against the wrong names.
TEST(SymbolTest, CanonicalOrderIsAlphabetical) {
  auto expr = (var<"z"> * var<"a">)+(var<"m"> * var<"z">)+var<"b">;
  using Syms = extract_symbols_from_expr_t<decltype(expr)>;
  using ddx::impl::FixedString;
  static_assert(ddx::meta::size(Syms{}) == 4);
  static_assert(
      std::is_same_v<Syms, ddx::meta::list<
                               ddx::impl::symbol_type<FixedString{"a"}>,
                               ddx::impl::symbol_type<FixedString{"b"}>,
                               ddx::impl::symbol_type<FixedString{"m"}>,
                               ddx::impl::symbol_type<FixedString{"z"}>>>,
      "canonical symbol order must stay alphabetical and deduplicated");
}
TEST(ExpressionTest, StaticTests) {
  auto x = 4_vi;
  auto y = 2_vi;
  auto c = 2_ci;
  auto res = x * y + c;
  auto res2 = make_all_constant_except<ddx::impl::FixedString{"c"}>(res);
  ASSERT_EQ(res2.eval(4), res.eval(4));
}
TEST(ExpressionTest, SumTest) {
  double a = 1, b = 2, c = 3;
  auto sum_exp = a + b + c;
  ASSERT_EQ(sum_exp, 6);
}
TEST(ExpressionTest, MultiplyTest) {
  auto a = 1_ci;
  auto b = 2_vi;
  auto c = 3_ci;
  auto sum_exp = a * b * c;
  auto d = sum_exp.derivative();
  ASSERT_EQ(sum_exp.eval(2), 6);
  ASSERT_EQ(d.eval(), 3);
}
TEST(ExpressionTest, SubtractTest) {
  auto a = 1_ci;
  auto b = 2_vi;
  auto minus = a - b;
  auto d = minus.derivative();
  ASSERT_EQ(minus.eval(2), -1);
  ASSERT_EQ(d.eval(), -1);
}
TEST(ExpressionTest, DivideTest) {
  auto a = 4.0_vd;
  auto b = 2.0_cd;
  auto divide = a / b;
  auto d = divide.derivative();
  ASSERT_EQ(divide.eval(4.0), 2.0);
  ASSERT_EQ(d.eval(), 0.5);
}
TEST(ExpressionTest, ExpTest) {
  constexpr auto exp_exp = exp(2.0_cd);
  ASSERT_EQ(exp_exp.eval(), std::exp(2.0));
}
TEST(ExpressionTest, ExpSum) {
  constexpr auto target = exp(1_cd + 2_cd);
  ASSERT_EQ(target.eval(), std::exp(3.0));
}
TEST(ExpressionTest, ExpDerivative) {
  for (std::size_t i = 1; i < 1000; ++i) {
    Variable<double, ddx::impl::FixedString{"x"}> xv;
    auto target = exp(xv);
    const double at = static_cast<double>(i);
    ASSERT_EQ(target.derivative().eval(at), target.eval(at));
  }
}
TEST(ExpressionTest, Combination) {
  double target = (exp(1_cd + 2_cd + 3_cd) + 1_cd).eval();
  ASSERT_EQ(target, std::exp(6.0) + 1.0);
}
TEST(ExpressionTest, ConstantDerivative) {
  auto target = Constant<int>(1);
  ASSERT_EQ(target.derivative(), 0);
  ASSERT_EQ(target, 1);
}
TEST(ExpressionTest, VariableDerivative) {
  Variable<int, ddx::impl::FixedString{"x"}> x;
  ASSERT_EQ(x.derivative().eval(), 1);
  ASSERT_EQ(x.eval(5), 5);
}
TEST(ExpressionTest, DerivativeTest) {
  auto x = 4_vi;
  auto expr = x * 2_ci;
  auto derv = expr.derivative();
  ASSERT_EQ(expr.eval(4), 8);
  ASSERT_EQ(derv.eval(), 2);
}
TEST(DerivativeRuleTest, SumRule) {
  auto x = var<"x", int>;
  auto expr = 3_ci * x + 5_ci;
  ASSERT_EQ(expr.derivative().eval(), 3);
}
TEST(DerivativeRuleTest, ProductRule) {
  Variable<int, ddx::impl::FixedString{"x"}> x;
  auto expr = x * x;
  ASSERT_EQ(expr.derivative().eval(4), 8);
}
TEST(DerivativeRuleTest, QuotientRule) {
  Variable<double, ddx::impl::FixedString{"x"}> x;
  auto c = Constant<double>{3.0};
  auto expr = x / c;
  ASSERT_DOUBLE_EQ(expr.derivative().eval(), 1.0 / 3.0);
}
TEST(DerivativeRuleTest, ChainRule_ExpOfLinear) {
  Variable<double, ddx::impl::FixedString{"x"}> x;
  auto inner = constant(2.0) * x;
  auto expr = exp(inner);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(1.0), 2.0 * std::exp(2.0));
}
TEST(DerivativeRuleTest, ChainRule_SinOfLinear) {
  Variable<double, ddx::impl::FixedString{"x"}> x;
  auto inner = constant(3.0) * x;
  auto expr = sin(inner);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(0.0), 3.0 * std::cos(0.0));
}
TEST(VariableTest, GetValue) {
  auto a = 2_vi;
  ASSERT_EQ(a.eval(2), 2);
}
TEST(VariableTest, Assign) {
  // A variable holds no value, so "assignment" is supplying a point.
  Variable<int, FixedString{"a"}> a;
  ASSERT_EQ(a.eval(2), 2);
  ASSERT_EQ(a.eval(5), 5);
}
TEST(VariableTest, UdlCompAndAssign) {
  Variable<int, FixedString{"a"}> a;
  auto b = 4_vi;
  ASSERT_EQ(a.eval(4), b.eval(4));
}
TEST(TrigTest, SinTest) {
  auto b = sin(constant(0.5));
  ASSERT_EQ(b.eval(), std::sin(0.5));
}
TEST(TrigTest, CosTest) {
  auto b = cos(var<"x">);
  ASSERT_DOUBLE_EQ(b.eval(0.45), std::cos(0.45));
  ASSERT_DOUBLE_EQ(b.derivative().eval(0.45), -std::sin(0.45));
}
TEST(TrigTest, SinDerivative) {
  auto x = var<"x">;
  auto s = sin(x);
  ASSERT_DOUBLE_EQ(s.derivative().eval(0.7), std::cos(0.7));
}
TEST(TrigTest, SinCosIdentity) {
  for (double v : {0.0, 0.5, 1.0, std::numbers::pi / 4}) {
    Variable<double, ddx::impl::FixedString{"x"}> x;
    auto s = sin(x);
    auto c = cos(x);
    double lhs = (s * s).eval(v) + (c * c).eval(v);
    ASSERT_NEAR(lhs, 1.0, 1e-12);
  }
}
TEST(TrigTest, CosDerivativeIsNegSin) {
  for (double v : {0.0, 0.3, 1.0, 2.0}) {
    Variable<double, ddx::impl::FixedString{"x"}> x;
    ASSERT_DOUBLE_EQ(cos(x).derivative().eval(v), -std::sin(v));
  }
}
TEST(TrigTest, ExpDerivativeIsItself) {
  for (double v : {-1.0, 0.0, 0.5, 1.5}) {
    Variable<double, ddx::impl::FixedString{"x"}> x;
    ASSERT_DOUBLE_EQ(exp(x).derivative().eval(v), std::exp(v));
  }
}
TEST(EquationTest, SingleVariable) {
  auto x = var<"x", int>;
  auto expr = x * 2_ci;
  auto eq = Equation(expr);
  ASSERT_EQ(eq[idx<1>()].eval(), 2);
}
TEST(EquationTest, TwoVariables) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto expr = x * y;
  auto eq = Equation(expr);
  // Each folded partial names one symbol, so each takes just that value.
  ASSERT_EQ(eq[idx<1>()].eval(2), 2);
  ASSERT_EQ(eq[idx<2>()].eval(4), 4);
}
TEST(EquationTest, LinearCombination) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto expr = constant(1) * x + constant(2) * y;
  auto eq = Equation(expr);
  ASSERT_EQ(eq[idx<1>()].eval(), 1);
  ASSERT_EQ(eq[idx<2>()].eval(), 2);
}
TEST(EquationTest, DifferenceOfSquares) {
  constexpr auto x = var<"x", int>;
  constexpr auto y = var<"y", int>;
  constexpr auto expr = (x + y) * (x - y);
  auto eq = Equation(expr);
  auto [d1, d2] =
      eq.template jacobian<ddx::DiffMode::Symbolic>(std::array{4, 2});
  ASSERT_EQ(expr.eval(4, 2), 12);
  ASSERT_EQ(d1, 8);
  ASSERT_EQ(d2, -4);
}
TEST(EquationTest, ExpEquation) {
  auto x = var<"x">;
  auto expr = exp(x);
  auto eq = Equation(expr);
  ASSERT_DOUBLE_EQ(eq.evaluate(std::array{1.0}), std::exp(1.0));
  ASSERT_DOUBLE_EQ(eq[idx<1>()].eval(1.0), std::exp(1.0));
}
TEST(EquationTest, TrigEquation) {
  auto x = var<"x">;
  auto expr = sin(x) * cos(x);
  auto eq = Equation(expr);
  ASSERT_DOUBLE_EQ(eq[idx<1>()].eval(0.5), std::cos(2 * 0.5));
}
TEST(EquationTest, IdxEquivalence) {
  // eq[idx<N>()] is the subscript spelling of eq.get<N>().
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto eq = Equation(x * y);
  ASSERT_EQ(eq[idx<1>()].eval(5), eq.get<1>().eval(5));
  ASSERT_EQ(eq[idx<2>()].eval(3), eq.get<2>().eval(3));
}
TEST(EquationTest, ThreeVariablePartials) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto z = var<"z", int>;
  auto expr = x * y + y * z;
  auto eq = Equation(expr);
  auto [dx, dy, dz] =
      eq.template jacobian<ddx::DiffMode::Symbolic>(std::array{2, 3, 4});
  ASSERT_EQ(dx, 3);
  ASSERT_EQ(dy, 6);
  ASSERT_EQ(dz, 3);
}
TEST(EquationTest, UpdateAndReevaluate) {
  // Both copies of x in the derivative are live variables, so a new point
  // reaches both.
  Variable<int, ddx::impl::FixedString{"x"}> x;
  auto expr = x * x;
  auto eq = Equation(expr);

  ASSERT_EQ(eq.evaluate(std::array{3}), 9);
  ASSERT_EQ(eq[idx<1>()].eval(3), 6);

  ASSERT_EQ(eq.evaluate(std::array{5}), 25);
  ASSERT_EQ(eq[idx<1>()].eval(5), 10);
}
TEST(EquationTest, MixedTrigExpEquation) {
  auto x = var<"x">;
  auto expr = exp(x) * sin(x);
  auto eq = Equation(expr);
  double expected = std::exp(1.0) * (std::sin(1.0) + std::cos(1.0));
  ASSERT_DOUBLE_EQ(eq[idx<1>()].eval(1.0), expected);
}
TEST(EquationTest, NumberOfDerivatives) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  static_assert(Equation<decltype(x * y)>::number_of_derivatives == 2);
  static_assert(Equation<decltype(x * x)>::number_of_derivatives == 1);
}
TEST(EquationTest, Dimensions) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  using VE = Equation<decltype(x + y), decltype(x * y)>;
  static_assert(VE::output_dim == 2);
  static_assert(VE::input_dim == 2);
}
TEST(EquationTest, Eval) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto ve = Equation(x + y, x * y);
  auto v = ve.evaluate(std::array{3, 4});
  ASSERT_EQ(v[0], 7);
  ASSERT_EQ(v[1], 12);
}
TEST(EquationTest, JacobianLinear) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto ve = Equation(x + y, x * y);
  auto J = ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{3, 4});
  ASSERT_EQ((J[0, 0]), 1);
  ASSERT_EQ((J[0, 1]), 1);
  ASSERT_EQ((J[1, 0]), 4);
  ASSERT_EQ((J[1, 1]), 3);
}
TEST(EquationTest, JacobianWithTrig) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto ve = Equation(x * y, sin(x) + y * y);
  auto J = ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{2.0, 3.0});
  ASSERT_DOUBLE_EQ((J[0, 0]), 3.0);
  ASSERT_DOUBLE_EQ((J[0, 1]), 2.0);
  ASSERT_DOUBLE_EQ((J[1, 0]), std::cos(2.0));
  ASSERT_DOUBLE_EQ((J[1, 1]), 6.0);
}
TEST(EquationTest, SingleComponentIsJacobian) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto eq = Equation(x * y);
  auto g = eq.template jacobian<ddx::DiffMode::Symbolic>(std::array{2.0, 3.0});
  ASSERT_DOUBLE_EQ(g[0], 3.0);
  ASSERT_DOUBLE_EQ(g[1], 2.0);
}
TEST(EquationTest, SymbolUnionAcrossComponents) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto ve = Equation(x * x, y * y);
  static_assert(decltype(ve)::input_dim == 2);
  auto J = ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{4.0, 3.0});
  ASSERT_DOUBLE_EQ((J[0, 0]), 8.0);
  ASSERT_DOUBLE_EQ((J[0, 1]), 0.0);
  ASSERT_DOUBLE_EQ((J[1, 0]), 0.0);
  ASSERT_DOUBLE_EQ((J[1, 1]), 6.0);
}
TEST(EquationTest, ThreeOutputs) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto ve = Equation(x * x, x * y, y * y);
  static_assert(decltype(ve)::output_dim == 3);
  static_assert(decltype(ve)::input_dim == 2);
  auto J = ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{2.0, 5.0});
  ASSERT_DOUBLE_EQ((J[0, 0]), 4.0);
  ASSERT_DOUBLE_EQ((J[0, 1]), 0.0);
  ASSERT_DOUBLE_EQ((J[1, 0]), 5.0);
  ASSERT_DOUBLE_EQ((J[1, 1]), 2.0);
  ASSERT_DOUBLE_EQ((J[2, 0]), 0.0);
  ASSERT_DOUBLE_EQ((J[2, 1]), 10.0);
}
TEST(EquationTest, ReverseJacobianAgreesWithSymbolic) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  auto ve = Equation(x * y, sin(x) + y * z, exp(z));

  auto J_sym =
      ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{2.0, 3.0, 4.0});
  auto J_rev = ve.jacobian(std::array{2.0, 3.0, 4.0});

  for (std::size_t i = 0; i < decltype(ve)::output_dim; ++i)
    for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
      ASSERT_DOUBLE_EQ((J_rev[i, j]), (J_sym[i, j]));
}
TEST(EquationTest, ParallelReverseJacobian_FourOutputs) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  auto ve = Equation(x * y, y * z, x * z, x * y * z);
  static_assert(decltype(ve)::output_dim == 4);
  static_assert(decltype(ve)::input_dim == 3);

  auto J_sym =
      ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{1.0, 2.0, 3.0});
  auto J_rev = ve.jacobian(std::array{1.0, 2.0, 3.0});

  for (std::size_t i = 0; i < decltype(ve)::output_dim; ++i)
    for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
      ASSERT_DOUBLE_EQ((J_rev[i, j]), (J_sym[i, j]));
}
TEST(EquationTest, ParallelReverseJacobian_FiveOutputsTrigExp) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  auto ve = Equation(sin(x) * cos(y), exp(x + y), x * y + y * z,
                     cos(z) * sin(x), exp(x * z) + y * y);
  static_assert(decltype(ve)::output_dim == 5);
  static_assert(decltype(ve)::input_dim == 3);

  auto J_sym =
      ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{0.5, 1.0, 1.5});
  auto J_rev = ve.jacobian(std::array{0.5, 1.0, 1.5});

  for (std::size_t i = 0; i < decltype(ve)::output_dim; ++i)
    for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
      ASSERT_NEAR((J_rev[i, j]), (J_sym[i, j]), 1e-12);
}
TEST(EquationTest, ReverseJacobianSingleOutputMatchesFlatRow) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto expr = exp(x) * sin(y);
  // A two-component Equation selects the vector specialization.
  auto ve = Equation(expr, exp(x) * sin(y));

  auto J_rev = ve.jacobian(std::array{2.0, 5.0});
  auto g = Equation{expr}.jacobian(2.0, 5.0);

  for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
    ASSERT_DOUBLE_EQ((J_rev[0, j]), g[j]);
}
// canonicalise folds (y*x)/(x*y) to 1, so the sweep must be laid out by the
// Equation's symbols, or z's partial would land in x's slot.
TEST(EquationTest, ReverseJacobianIsLaidOutByTheEquationsSymbols) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  const Equation eq{(y * x) / (x * y) + z};
  static_assert(decltype(eq)::input_dim == 3);
  const auto reverse = eq.jacobian(1.0, 2.0, 3.0);
  const auto symbolic = eq.jacobian<DiffMode::Symbolic>(1.0, 2.0, 3.0);
  EXPECT_DOUBLE_EQ(reverse[0], 0.0);
  EXPECT_DOUBLE_EQ(reverse[1], 0.0);
  EXPECT_DOUBLE_EQ(reverse[2], 1.0);
  EXPECT_DOUBLE_EQ(symbolic[2], 1.0);
}
// a^0 is the constant 1 and 0^b the constant 0, so both partials are 0 there
// rather than the 0 * inf the general rules would give.
TEST(ReverseModeAD, PowAtAZeroBaseOrExponent) {
  auto x = var<"x">;
  auto y = var<"y">;
  const Equation eq{pow(x, y)};
  const auto at_zero_base = eq.jacobian(0.0, 2.0);
  EXPECT_DOUBLE_EQ(at_zero_base[0], 0.0);
  EXPECT_DOUBLE_EQ(at_zero_base[1], 0.0);
  const auto origin = eq.jacobian(0.0, 0.0);
  EXPECT_DOUBLE_EQ(origin[0], 0.0);
  const auto stored = Equation{pow(x, 0.0)}.jacobian(0.0);
  EXPECT_DOUBLE_EQ(stored[0], 0.0);
  const auto plain = eq.jacobian(2.0, 3.0);
  EXPECT_DOUBLE_EQ(plain[0], 12.0);
  EXPECT_DOUBLE_EQ(plain[1], 8.0 * std::log(2.0));
}
TEST(ForwardModeAD, ExpressionStructuredBinding) {
  // A bare expression carries no point, so value and derivative are taken
  // explicitly rather than by structured binding.
  auto x = var<"x">;
  const auto e = x * x;
  EXPECT_DOUBLE_EQ(e.eval(3.0), 9.0);
  EXPECT_DOUBLE_EQ(e.derivative().eval(3.0), 6.0);

  const auto s = sin(var<"x">);
  EXPECT_DOUBLE_EQ(s.eval(0.0), 0.0);
  EXPECT_DOUBLE_EQ(s.derivative().eval(0.0), 1.0);
}
TEST(ReverseModeAD, SingleVariableLinear) {
  auto x = var<"x">;
  auto expr = constant(3.0) * x;
  auto g = Equation{expr}.jacobian(5.0);
  EXPECT_DOUBLE_EQ(g[0], 3.0);
}
TEST(ReverseModeAD, ProductRule) {
  Variable<double, ddx::impl::FixedString{"x"}> x;
  auto expr = x * x;
  auto g = Equation{expr}.jacobian(std::array{4.0});
  EXPECT_DOUBLE_EQ(g[0], 8.0);
}
TEST(ReverseModeAD, TwoVariables) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto expr = x * y;
  auto g = Equation{expr}.jacobian(3.0, 4.0);
  static_assert(g.size() == 2);
  EXPECT_DOUBLE_EQ(g[0], 4.0);
  EXPECT_DOUBLE_EQ(g[1], 3.0);
}
TEST(ReverseModeAD, Sum) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto g = Equation{x + y}.jacobian(2.0, 5.0);
  EXPECT_DOUBLE_EQ(g[0], 1.0);
  EXPECT_DOUBLE_EQ(g[1], 1.0);
}
TEST(ReverseModeAD, LinearCombination) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto g = Equation{constant(2.0) * x + constant(3.0) * y}.jacobian(1.0, 1.0);
  EXPECT_DOUBLE_EQ(g[0], 2.0);
  EXPECT_DOUBLE_EQ(g[1], 3.0);
}
TEST(ReverseModeAD, Divide) {
  auto x = var<"x">;
  auto c = constant(3.0);
  auto g = Equation{x / c}.jacobian(6.0);
  EXPECT_DOUBLE_EQ(g[0], 1.0 / 3.0);
}
TEST(ReverseModeAD, NegateViaSubtract) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto g = Equation{x - y}.jacobian(5.0, 2.0);
  EXPECT_DOUBLE_EQ(g[0], 1.0);
  EXPECT_DOUBLE_EQ(g[1], -1.0);
}
TEST(ReverseModeAD, SinDerivative) {
  auto x = var<"x">;
  auto g = Equation{sin(x)}.jacobian(1.0);
  EXPECT_DOUBLE_EQ(g[0], std::cos(1.0));
}
TEST(ReverseModeAD, CosDerivative) {
  auto x = var<"x">;
  auto g = Equation{cos(x)}.jacobian(1.0);
  EXPECT_DOUBLE_EQ(g[0], -std::sin(1.0));
}
TEST(ReverseModeAD, ExpDerivative) {
  auto x = var<"x">;
  auto g = Equation{exp(x)}.jacobian(2.0);
  EXPECT_DOUBLE_EQ(g[0], std::exp(2.0));
}
TEST(ReverseModeAD, ChainRuleSinOfProduct) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto g = Equation{sin(x * y)}.jacobian(2.0, 3.0);
  EXPECT_DOUBLE_EQ(g[0], std::cos(6.0) * 3.0);
  EXPECT_DOUBLE_EQ(g[1], std::cos(6.0) * 2.0);
}
TEST(ReverseModeAD, ThreeVariables) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  auto g = Equation{x * y + y * z}.jacobian(2.0, 3.0, 4.0);
  EXPECT_DOUBLE_EQ(g[0], 3.0);
  EXPECT_DOUBLE_EQ(g[1], 6.0);
  EXPECT_DOUBLE_EQ(g[2], 3.0);
}
TEST(ReverseModeAD, ScalarOnRight) {
  auto x = var<"x">;
  auto g = Equation{x * 4.0 + 1.0}.jacobian(5.0);
  EXPECT_DOUBLE_EQ(g[0], 4.0);
}
TEST(ReverseModeAD, AgreesWithForwardMode) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto g = Equation{exp(x) * sin(y)}.jacobian(xv, yv);
  EXPECT_DOUBLE_EQ(g[0], std::exp(xv) * std::sin(yv));
  EXPECT_DOUBLE_EQ(g[1], std::exp(xv) * std::cos(yv));
}
TEST(TutorialForward, SingleVar_SymbolicValueAndDerivative) {
  double x0 = 2.0;
  auto x = var_of<"x">(x0);
  auto f = constant(1.0) + x + x * x + constant(1.0) / x + log(x);
  EXPECT_NEAR(f.eval(x0), 1.0 + x0 + x0 * x0 + 1.0 / x0 + std::log(x0), 1e-12);
  double expected_df = 1.0 + 2.0 * x0 - 1.0 / (x0 * x0) + 1.0 / x0;
  EXPECT_NEAR(f.derivative().eval(x0), expected_df, 1e-12);
}
TEST(TutorialForward, SingleVar_ReverseMode) {
  double x0 = 2.0;
  auto x = var_of<"x">(x0);
  auto f = constant(1.0) + x + x * x + constant(1.0) / x + log(x);
  auto g = Equation{f}.jacobian(x0);
  EXPECT_NEAR(g[0], 1.0 + 2.0 * x0 - 1.0 / (x0 * x0) + 1.0 / x0, 1e-12);
}
TEST(TutorialMultiVar, Value) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto f = constant(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
           exp(x / y + y / z);
  double expected = 1.0 + xv + yv + zv + xv * yv + yv * zv + xv * zv +
                    xv * yv * zv + std::exp(xv / yv + yv / zv);
  EXPECT_NEAR(f.eval(xv, yv, zv), expected, 1e-12);
}
TEST(TutorialMultiVar, SymbolicPartials) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto f = constant(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
           exp(x / y + y / z);
  auto eq = Equation(f);
  auto [dx, dy, dz] =
      eq.template jacobian<ddx::DiffMode::Symbolic>(std::array{xv, yv, zv});
  double e = std::exp(xv / yv + yv / zv);
  EXPECT_NEAR(dx, 1.0 + yv + zv + yv * zv + e / yv, 1e-10);
  EXPECT_NEAR(dy, 1.0 + xv + zv + xv * zv + e * (-xv / (yv * yv) + 1.0 / zv),
              1e-10);
  EXPECT_NEAR(dz, 1.0 + yv + xv + xv * yv + e * (-yv / (zv * zv)), 1e-10);
}
TEST(TutorialMultiVar, ReverseJacobian) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto f = constant(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
           exp(x / y + y / z);
  auto g = Equation{f}.jacobian(xv, yv, zv);
  double e = std::exp(xv / yv + yv / zv);
  EXPECT_NEAR(g[0], 1.0 + yv + zv + yv * zv + e / yv, 1e-10);
  EXPECT_NEAR(g[1], 1.0 + xv + zv + xv * zv + e * (-xv / (yv * yv) + 1.0 / zv),
              1e-10);
  EXPECT_NEAR(g[2], 1.0 + yv + xv + xv * yv + e * (-yv / (zv * zv)), 1e-10);
}
TEST(TutorialReverseConditionals, AbsJacobianChangesSign) {
  EXPECT_DOUBLE_EQ(Equation{abs(var<"x">)}.jacobian(3.0)[0], 1.0);
  EXPECT_DOUBLE_EQ(Equation{abs(var<"x">)}.jacobian(-3.0)[0], -1.0);
}
TEST(TutorialReverseConditionals, UpdateReevaluatesDerivatives) {
  Variable<double, FixedString{"x"}> x;
  auto eq = Equation(x * x + sin(x));
  EXPECT_NEAR(eq.evaluate(std::array{2.0}), 4.0 + std::sin(2.0), 1e-12);
  EXPECT_NEAR(eq.template jacobian<ddx::DiffMode::Symbolic>(std::array{2.0})[0],
              4.0 + std::cos(2.0), 1e-12);
  EXPECT_NEAR(eq.evaluate(std::array{5.0}), 25.0 + std::sin(5.0), 1e-12);
  EXPECT_NEAR(eq.template jacobian<ddx::DiffMode::Symbolic>(std::array{5.0})[0],
              10.0 + std::cos(5.0), 1e-12);
}
TEST(TutorialReverseParams, SingleParameter) {
  double xv = 1.0, p = 3.0;
  auto x = var_of<"x">(xv);
  auto g = Equation{constant(p) * sin(x)}.jacobian(xv);
  EXPECT_NEAR(g[0], p * std::cos(xv), 1e-12);
}
TEST(TutorialReverseParams, MultiVarWithParameter) {
  double xv = 2.0, yv = 3.0, p = 4.0;
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto g = Equation{constant(p) * x * y}.jacobian(xv, yv);
  EXPECT_NEAR(g[0], p * yv, 1e-12);
  EXPECT_NEAR(g[1], p * xv, 1e-12);
}
TEST(ScopedValue, SeedsOnEntryAndRestoresOnExit) {
  double slot = 7.5;
  {
    const scoped_value seed{slot, 1.0};
    EXPECT_DOUBLE_EQ(slot, 1.0);
  }
  EXPECT_DOUBLE_EQ(slot, 7.5);
}
TEST(ScopedValue, RestoreIsBitExactFromAnyBase) {
  // The guard saves the old bits, because slot += 1 then slot -= 1 is not the
  // identity in floating point.
  double slot = 0.1;
  {
    const scoped_value seed{slot, 1.0};
    EXPECT_DOUBLE_EQ(slot, 1.0);
  }
  EXPECT_EQ(slot, 0.1);
  EXPECT_NE(0.1 + 1.0 - 1.0, 0.1);
}

TEST(ScopedValue, IsPinned) {
  // Moving a guard would leave a second destructor writing back a value that
  // has already been carried off.
  using Guard = scoped_value<double>;
  static_assert(!std::is_copy_constructible_v<Guard>);
  static_assert(!std::is_move_constructible_v<Guard>);
  static_assert(!std::is_copy_assignable_v<Guard>);
  static_assert(!std::is_move_assignable_v<Guard>);
  SUCCEED();
}
TEST(ScopedValue, CarriesNothingForTheNewValue) {
  // The guard holds the reference and the saved scalar, and nothing else.
  static_assert(sizeof(scoped_value<double>) ==
                sizeof(double *) + sizeof(double));
  SUCCEED();
}
TEST(ScopedValue, HoldsARuntimeValueToo) {
  // rt::equation() uses the guard on a pointer slot with a non-constant value.
  int a = 1;
  int b = 2;
  int *slot = &a;
  {
    const ddx::impl::scoped_value hold{slot, &b};
    EXPECT_EQ(slot, &b);
    {
      const ddx::impl::scoped_value inner{slot, &a};
      EXPECT_EQ(slot, &a);
    }
    EXPECT_EQ(slot, &b);
  }
  EXPECT_EQ(slot, &a);
}

TEST(Ownership, EquationSubtreeAccessorsWorkOnTemporaries) {
  // The nodes are empty types, so the rvalue overload copies nothing.
  auto x = var<"x">;
  auto y = var<"y">;
  EXPECT_DOUBLE_EQ(Equation(x * y).get<1>().eval(2.0), 2.0);
  EXPECT_DOUBLE_EQ(Equation(x * y)[idx<2>()].eval(4.0), 4.0);
}

// The stored partial trees agree with the sweep at pow's zeros.
TEST(EquationTest, SymbolicPowAtAZeroBaseOrExponent) {
  auto x = var<"x">;
  auto y = var<"y">;
  const Equation eq{pow(x, y)};
  const auto at_zero_base = eq.jacobian<DiffMode::Symbolic>(0.0, 2.0);
  EXPECT_DOUBLE_EQ(at_zero_base[0], 0.0);
  EXPECT_DOUBLE_EQ(at_zero_base[1], 0.0);
  const auto origin = eq.jacobian<DiffMode::Symbolic>(0.0, 0.0);
  EXPECT_DOUBLE_EQ(origin[0], 0.0);
  const auto stored =
      Equation{pow(x, 0.0)}.jacobian<DiffMode::Symbolic>(0.0);
  EXPECT_DOUBLE_EQ(stored[0], 0.0);
  const auto plain = eq.jacobian<DiffMode::Symbolic>(2.0, 3.0);
  EXPECT_DOUBLE_EQ(plain[0], 12.0);
  EXPECT_DOUBLE_EQ(plain[1], 8.0 * std::log(2.0));
}
