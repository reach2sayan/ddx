#include "dual/tests_dual_common.hpp"
#include "tests_core_fixtures.hpp"

TEST(ForwardModeAD, TanDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = tan(x).eval(Dual{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::tan(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / (std::cos(x0) * std::cos(x0)));
}
TEST(ForwardModeAD, LogDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = log(x).eval(Dual{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::log(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / x0);
}
TEST(ForwardModeAD, SqrtDerivative) {
  double x0 = 4.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = sqrt(x).eval(Dual{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, 2.0);
  ASSERT_DOUBLE_EQ(df, 0.25);
}
TEST(ForwardModeAD, AsinDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = asin(x).eval(Dual{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::asin(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / std::sqrt(1.0 - x0 * x0));
}
TEST(ForwardModeAD, AcosDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = acos(x).eval(Dual{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::acos(x0));
  ASSERT_DOUBLE_EQ(df, -1.0 / std::sqrt(1.0 - x0 * x0));
}
TEST(ForwardModeAD, AtanDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = atan(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::atan(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / (1.0 + x0 * x0));
}
TEST(ForwardModeAD, SinhDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = sinh(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::sinh(x0));
  ASSERT_DOUBLE_EQ(df, std::cosh(x0));
}
TEST(ForwardModeAD, CoshDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = cosh(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::cosh(x0));
  ASSERT_DOUBLE_EQ(df, std::sinh(x0));
}
TEST(ForwardModeAD, TanhDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = tanh(x).eval(Dual{x0, 1.0});
  double c = std::cosh(x0);
  ASSERT_DOUBLE_EQ(f, std::tanh(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / (c * c));
}
TEST(ForwardModeAD, AbsDerivativePositive) {
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = abs(x).eval(Dual{2.0, 1.0});
  ASSERT_DOUBLE_EQ(f, 2.0);
  ASSERT_DOUBLE_EQ(df, 1.0);
}
TEST(ForwardModeAD, AbsDerivativeNegative) {
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = abs(x).eval(Dual{-2.0, 1.0});
  ASSERT_DOUBLE_EQ(f, 2.0);
  ASSERT_DOUBLE_EQ(df, -1.0);
}
TEST(ForwardModeAD, DualNumericConcept) {
  static_assert(Numeric<Dual<double>>);
  static_assert(Numeric<Dual<float>>);
}
TEST(ForwardModeAD, StructuredBinding) {
  Dual<double> d{3.0, 7.0};
  auto [v, dv] = d;
  EXPECT_DOUBLE_EQ(v, 3.0);
  EXPECT_DOUBLE_EQ(dv, 7.0);
  static_assert(std::tuple_size_v<Dual<double>> == 2);
  static_assert(std::is_same_v<std::tuple_element_t<0, Dual<double>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, Dual<double>>, double>);
}
TEST(ForwardModeAD, BasicArithmetic) {
  constexpr Dual a{3.0, 1.0};
  constexpr Dual b{2.0, 0.0};
  auto [sum_val, sum_deriv] = a + b;
  EXPECT_DOUBLE_EQ(sum_val, 5.0);
  EXPECT_DOUBLE_EQ(sum_deriv, 1.0);
  auto [prod_val, prod_deriv] = a * b;
  EXPECT_DOUBLE_EQ(prod_val, 6.0);
  EXPECT_DOUBLE_EQ(prod_deriv, 2.0); // 1*2 + 3*0 = 2
  auto [quot_val, quot_deriv] = a / b;
  EXPECT_DOUBLE_EQ(quot_val, 1.5);
  EXPECT_DOUBLE_EQ(quot_deriv, 0.5); // (1*2 - 3*0)/4 = 0.5
}
TEST(ForwardModeAD, DualScalarArithmetic) {
  // A bare scalar promotes to a zero-derivative Dual.
  constexpr Dual d{3.0, 1.0};

  auto [add_v, add_d] = d + 2.0;
  EXPECT_DOUBLE_EQ(add_v, 5.0);
  EXPECT_DOUBLE_EQ(add_d, 1.0);
  auto [radd_v, radd_d] = 2.0 + d;
  EXPECT_DOUBLE_EQ(radd_v, 5.0);
  EXPECT_DOUBLE_EQ(radd_d, 1.0);

  auto [sub_v, sub_d] = d - 2.0;
  EXPECT_DOUBLE_EQ(sub_v, 1.0);
  EXPECT_DOUBLE_EQ(sub_d, 1.0);
  auto [rsub_v, rsub_d] = 2.0 - d;
  EXPECT_DOUBLE_EQ(rsub_v, -1.0);
  EXPECT_DOUBLE_EQ(rsub_d, -1.0);

  auto [mul_v, mul_d] = 2.0 * d;
  EXPECT_DOUBLE_EQ(mul_v, 6.0);
  EXPECT_DOUBLE_EQ(mul_d, 2.0); // scalar scales the derivative

  auto [div_v, div_d] = d / 2.0;
  EXPECT_DOUBLE_EQ(div_v, 1.5);
  EXPECT_DOUBLE_EQ(div_d, 0.5);

  Dual<double> acc{3.0, 1.0};
  acc += 2.0;
  EXPECT_DOUBLE_EQ(acc.template get<0>(), 5.0);
  EXPECT_DOUBLE_EQ(acc.template get<1>(), 1.0);
}
TEST(ForwardModeAD, ScalarPromotionDeepDual) {
  // A zero derivative at every nesting level: one static_cast would need two
  // chained explicit conversions.
  using DD = Dual<Dual<double>>;
  Variable<DD, ddx::impl::FixedString{"x"}> x;
  auto expr = x + 2.0;
  DD result = expr.eval(DD{Dual{2.0, 0.0}, Dual{0.0, 0.0}});
  EXPECT_DOUBLE_EQ(get_real_part<2>(result), 4.0); // peel both Dual<> layers

  auto y = var<"y">;
  EXPECT_DOUBLE_EQ((y + 2.0).eval(3.0), 5.0);
}
TEST(ForwardModeAD, PolynomialDerivative) {
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = (x * x + x).eval(Dual{3.0, 1.0});
  EXPECT_DOUBLE_EQ(f, 12.0);
  EXPECT_DOUBLE_EQ(df, 7.0);
}
TEST(ForwardModeAD, PartialDerivativeX) {
  Variable<Dual<double>, FixedString{"x"}> x;
  Variable<Dual<double>, FixedString{"y"}> y;
  auto [f, df] = (x * y).eval(Dual{3.0, 1.0}, Dual{4.0, 0.0});
  EXPECT_DOUBLE_EQ(f, 12.0);
  EXPECT_DOUBLE_EQ(df, 4.0);
}
TEST(ForwardModeAD, PartialDerivativeY) {
  Variable<Dual<double>, FixedString{"x"}> x;
  Variable<Dual<double>, FixedString{"y"}> y;
  auto [f, df] = (x * y).eval(Dual{3.0, 0.0}, Dual{4.0, 1.0});
  EXPECT_DOUBLE_EQ(f, 12.0);
  EXPECT_DOUBLE_EQ(df, 3.0);
}
TEST(ForwardModeAD, SinDerivative) {
  double x0 = std::numbers::pi / 4.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = sin(x).eval(Dual{x0, 1.0});
  EXPECT_DOUBLE_EQ(f, std::sin(x0));
  EXPECT_DOUBLE_EQ(df, std::cos(x0));
}
TEST(ForwardModeAD, CosDerivative) {
  double x0 = std::numbers::pi / 3.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = cos(x).eval(Dual{x0, 1.0});
  EXPECT_DOUBLE_EQ(f, std::cos(x0));
  EXPECT_DOUBLE_EQ(df, -std::sin(x0));
}
TEST(ForwardModeAD, ExpDerivative) {
  double x0 = 2.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = exp(x).eval(Dual{x0, 1.0});
  EXPECT_DOUBLE_EQ(f, std::exp(x0));
  EXPECT_DOUBLE_EQ(df, std::exp(x0));
}
TEST(ForwardModeAD, ChainRule) {
  double x0 = 1.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = sin(x * x).eval(Dual{x0, 1.0});
  EXPECT_DOUBLE_EQ(f, std::sin(x0 * x0));
  EXPECT_DOUBLE_EQ(df, 2.0 * x0 * std::cos(x0 * x0));
}
TEST(ForwardModeAD, Equivalence) {
  double x0 = 1.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto xv = var_of<"x">(x0);
  auto l = sin(xv * xv);
  auto [f, df] = sin(x * x).eval(Dual{x0, 1.0});
  auto f2 = l.eval(x0);
  auto df2 = l.derivative().eval(x0);
  EXPECT_DOUBLE_EQ(f, std::sin(x0 * x0));
  EXPECT_DOUBLE_EQ(df, 2.0 * x0 * std::cos(x0 * x0));
  EXPECT_DOUBLE_EQ(df, df2);
  EXPECT_DOUBLE_EQ(f, f2);
}
TEST(EquationForward, TwoVariables) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, x + y);
  auto J = ve.derivative_tensor<1>(std::array{3.0, 4.0});
  EXPECT_DOUBLE_EQ((J[0, 0]), 4.0); // ∂(x*y)/∂x = y = 4
  EXPECT_DOUBLE_EQ((J[0, 1]), 3.0); // ∂(x*y)/∂y = x = 3
  EXPECT_DOUBLE_EQ((J[1, 0]), 1.0); // ∂(x+y)/∂x
  EXPECT_DOUBLE_EQ((J[1, 1]), 1.0); // ∂(x+y)/∂y
}
TEST(EquationForward, AgreesWithSymbolic) {
  double xv = 2.0, yv = 3.0;

  auto xs = var_of<"x">(xv);
  auto ys = var_of<"y">(yv);
  auto ve_sym = Equation(xs * xs, xs * ys, ys * ys);
  auto J_sym =
      ve_sym.template jacobian<ddx::DiffMode::Symbolic>(std::array{xv, yv});

  Variable<double, FixedString{"x"}> xd;
  Variable<double, FixedString{"y"}> yd;
  auto ve_fwd = Equation(xd * xd, xd * yd, yd * yd);
  auto J_fwd = ve_fwd.derivative_tensor<1>(std::array{xv, yv});

  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ((J_fwd[i, j]), (J_sym[i, j]));
}
TEST(EquationForward, TrigJacobian) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, sin(x) + y * y);
  auto J = ve.derivative_tensor<1>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((J[0, 0]), 3.0);
  EXPECT_DOUBLE_EQ((J[0, 1]), 2.0);
  EXPECT_NEAR((J[1, 0]), std::cos(2.0), 1e-12);
  EXPECT_DOUBLE_EQ((J[1, 1]), 6.0);
}
TEST(EquationForward, AgreesWithReverse) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve_fwd = Equation(x * y, sin(x) + y * y);
  auto J_fwd = ve_fwd.derivative_tensor<1>(std::array{2.0, 3.0});

  auto xs = var<"x">;
  auto ys = var<"y">;
  auto ve_rev = Equation(xs * ys, sin(xs) + ys * ys);
  auto J_rev = ve_rev.jacobian(std::array{2.0, 3.0});

  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR((J_rev[i, j]), (J_fwd[i, j]), 1e-12);
}
TEST(ReverseModeAD, ScalarLiteralCoercion) {
  auto x = var<"x", dual>;
  auto y = var<"y", dual>;
  auto z = var<"z", dual>;
  auto expe = 3.0 * x * y + y * z;
  auto g =
      Equation{expe}.jacobian(Dual{2.0, 0.0}, Dual{3.0, 0.0}, Dual{4.0, 0.0});
  EXPECT_DOUBLE_EQ(g[0], 9.0);  // df/dx = 3*y = 9
  EXPECT_DOUBLE_EQ(g[1], 10.0); // df/dy = 3*x + z = 10
  EXPECT_DOUBLE_EQ(g[2], 3.0);  // df/dz = y = 3
}
TEST(DualCompoundAssign, PlusEq) {
  Dual a{3.0, 1.0}, b{2.0, 0.5};
  a += b;
  EXPECT_DOUBLE_EQ(a.template get<0>(), 5.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 1.5);
}
TEST(DualCompoundAssign, MinusEq) {
  Dual a{3.0, 1.0}, b{2.0, 0.5};
  a -= b;
  EXPECT_DOUBLE_EQ(a.template get<0>(), 1.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 0.5);
}
TEST(DualCompoundAssign, TimesEq) {
  Dual a{3.0, 1.0}, b{2.0, 0.5};
  a *= b;
  EXPECT_DOUBLE_EQ(a.template get<0>(), 6.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 3.5);
}
TEST(DualCompoundAssign, DivEq) {
  Dual<double> a{4.0, 2.0}, b{2.0, 0.0};
  a /= b;
  EXPECT_DOUBLE_EQ(a.template get<0>(), 2.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 1.0);
}
// The four are gated on what Dual's own operators take, so the three operand
// shapes that compile for `+` compile here: a same-T dual, an arithmetic
// scalar, and -- one nesting up -- the inner Dual, which is dual2nd's scalar.
TEST(DualCompoundAssign, TakesEveryShapeTheOperatorDoes) {
  Dual a{3.0, 1.0};
  a += Dual{2.0, 0.5}; // same-T dual
  a += 2;              // integral
  a += 2.0f;           // another floating type
  EXPECT_DOUBLE_EQ(a.template get<0>(), 9.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 1.5);

  ddx::dual2nd d{Dual{2.0, 1.0}, Dual{1.0, 0.0}};
  d *= Dual{3.0, 0.0}; // the value type, one level down
  EXPECT_DOUBLE_EQ(d.template get<0>().template get<0>(), 6.0);
  EXPECT_DOUBLE_EQ(d.template get<1>().template get<0>(), 3.0);
}
TEST(ReverseModeAD_Dual, SingleVariable) {
  auto x = var<"x", dual>;
  auto g = Equation{3.0 * x}.jacobian(Dual{5.0, 0.0});
  EXPECT_DOUBLE_EQ(g[0], 3.0);
}
TEST(ReverseModeAD_Dual, TwoVariables) {
  auto x = var<"x", dual>;
  auto y = var<"y", dual>;
  auto g = Equation{x * y}.jacobian(Dual{3.0, 0.0}, Dual{4.0, 0.0});
  EXPECT_DOUBLE_EQ(g[0], 4.0);
  EXPECT_DOUBLE_EQ(g[1], 3.0);
}
TEST(ReverseModeAD_Dual, ThreeVariables) {
  auto x = var<"x", dual>;
  auto y = var<"y", dual>;
  auto z = var<"z", dual>;
  auto g = Equation{3.0 * x * y + y * z}.jacobian(
      Dual{2.0, 0.0}, Dual{3.0, 0.0}, Dual{4.0, 0.0});
  EXPECT_DOUBLE_EQ(g[0], 9.0);
  EXPECT_DOUBLE_EQ(g[1], 10.0);
  EXPECT_DOUBLE_EQ(g[2], 3.0);
}
TEST(ReverseModeAD_Dual, TrigExp) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  auto x = dual_var_of<"x">(xv);
  auto y = dual_var_of<"y">(yv);
  auto g = Equation{exp(x) * sin(y)}.jacobian(Dual{xv, 0.0},
                                              Dual{yv, 0.0});
  EXPECT_DOUBLE_EQ(g[0], std::exp(xv) * std::sin(yv));
  EXPECT_DOUBLE_EQ(g[1], std::exp(xv) * std::cos(yv));
}
TEST(ReverseModeAD_Dual, AgreesWithPVResult) {
  double xv = 1.3, yv = 0.7;
  auto xp = var_of<"x">(xv);
  auto yp = var_of<"y">(yv);
  auto xd = dual_var_of<"x">(xv);
  auto yd = dual_var_of<"y">(yv);
  auto gp =
      Equation{xp * yp + sin(xp) + yp * yp + exp(xp + yp)}.jacobian(xv, yv);
  auto gd = Equation{xd * yd + sin(xd) + yd * yd + exp(xd + yd)}.jacobian(
      Dual<double>{xv, 0.0}, Dual<double>{yv, 0.0});
  EXPECT_DOUBLE_EQ(gd[0], gp[0]);
  EXPECT_DOUBLE_EQ(gd[1], gp[1]);
}
TEST(HessianTest, ForwardOverReverse_FunctionValues) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  const std::array<D, 2> pt{D{2.0}, D{3.0}};
  (void)ve.hessian(std::array{2.0, 3.0});
  auto f = ve.evaluate(pt);
  EXPECT_DOUBLE_EQ(f[0].template get<0>(), 6.0);
  EXPECT_DOUBLE_EQ(f[1].template get<0>(), 4.0);
}
TEST(HessianTest, ForwardOverReverse_XY) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto H = ve.hessian(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((H[0, 0, 0]), 0.0); // ∂²(x*y)/∂x²
  EXPECT_DOUBLE_EQ((H[0, 0, 1]), 1.0); // ∂²(x*y)/∂x∂y
  EXPECT_DOUBLE_EQ((H[0, 1, 0]), 1.0); // ∂²(x*y)/∂y∂x
  EXPECT_DOUBLE_EQ((H[0, 1, 1]), 0.0); // ∂²(x*y)/∂y²
}
TEST(HessianTest, ForwardOverReverse_Quadratic) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto H = ve.hessian(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((H[1, 0, 0]), 2.0);
  EXPECT_DOUBLE_EQ((H[1, 0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((H[1, 1, 0]), 0.0);
  EXPECT_DOUBLE_EQ((H[1, 1, 1]), 0.0);
}
TEST(HessianTest, ForwardOverReverse_WithValues) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  std::array<double, 2> pt{2.0, 3.0};
  auto H = ve.hessian(pt);
  auto f = ve.evaluate(std::array{D{2.0}, D{3.0}});
  EXPECT_DOUBLE_EQ(f[0].template get<0>(), 6.0);
  EXPECT_DOUBLE_EQ((H[0, 0, 1]), 1.0);
  EXPECT_DOUBLE_EQ((H[1, 0, 0]), 2.0);
}
TEST(HessianTest, ForwardOverReverse_TrigFunction) {
  double xv = 1.0, yv = 2.0;
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(sin(x) * y, x + y);
  auto H = ve.hessian(std::array{xv, yv});
  EXPECT_NEAR((H[0, 0, 0]), -yv * std::sin(xv), 1e-12); // -y*sin(x)
  EXPECT_NEAR((H[0, 0, 1]), std::cos(xv), 1e-12);       // cos(x)
  EXPECT_NEAR((H[0, 1, 0]), std::cos(xv), 1e-12);       // symmetric
  EXPECT_NEAR((H[0, 1, 1]), 0.0, 1e-12);
}
TEST(HessianTest, ForwardOverReverse_Symmetric) {
  // Two outputs, so the vector specialisation is selected.
  double xv = 0.5, yv = 1.5;
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(exp(x * y), x + y);
  auto H = ve.hessian(std::array{xv, yv});
  EXPECT_NEAR((H[0, 0, 1]), (H[0, 1, 0]), 1e-12);
}
// The canonicalised tree loses x and y, the Equation keeps them, and the
// colouring and scatter are built over the Equation's list.
TEST(HessianTest, SingleOutputIsLaidOutByTheEquationsSymbols) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  Variable<D, FixedString{"z"}> z;
  const Equation eq{(y * x) / (x * y) + z * z * z};
  static_assert(decltype(eq)::input_dim == 3);
  const auto H = eq.hessian(1.0, 2.0, 3.0);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ((H[i, j]), (i == 2 && j == 2) ? 18.0 : 0.0) << i << j;
    }
  }
  const auto g = eq.jacobian(1.0, 2.0, 3.0);
  EXPECT_DOUBLE_EQ(g[2], 27.0);
}
TEST(ForwardModeAD, PowAtAZeroExponentIsFlat) {
  using D = Dual<double>;
  const D at_zero{0.0, 1.0};
  EXPECT_DOUBLE_EQ(pow(at_zero, 0.0).deriv(), 0.0);
  EXPECT_DOUBLE_EQ(pow(at_zero, 0u).deriv(), 0.0);
  EXPECT_DOUBLE_EQ(pow(at_zero, D{0.0, 0.0}).deriv(), 0.0);
  EXPECT_DOUBLE_EQ(pow(at_zero, 0.0).value(), 1.0);
  // The rule is untouched off the corner.
  EXPECT_DOUBLE_EQ(pow(D{2.0, 1.0}, 3.0).deriv(), 12.0);
  EXPECT_DOUBLE_EQ(pow(at_zero, 2.0).deriv(), 0.0);
}
TEST(HessianForwardTest, FunctionValues) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto f = ve.evaluate(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(f[0], 6.0);
  EXPECT_DOUBLE_EQ(f[1], 4.0);
}
TEST(HessianForwardTest, XY) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto H = ve.derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((H[0, 0, 0]), 0.0);
  EXPECT_DOUBLE_EQ((H[0, 0, 1]), 1.0);
  EXPECT_DOUBLE_EQ((H[0, 1, 0]), 1.0);
  EXPECT_DOUBLE_EQ((H[0, 1, 1]), 0.0);
}
TEST(HessianForwardTest, Quadratic) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto H = ve.derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((H[1, 0, 0]), 2.0);
  EXPECT_DOUBLE_EQ((H[1, 0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((H[1, 1, 0]), 0.0);
  EXPECT_DOUBLE_EQ((H[1, 1, 1]), 0.0);
}
TEST(HessianForwardTest, AgreesWithForwardOverReverse) {
  double xv = 1.5, yv = 2.5;

  using D = Dual<double>;
  Variable<D, FixedString{"x"}> xr;
  Variable<D, FixedString{"y"}> yr;
  auto ve_rev = Equation(xr * yr, xr * xr);
  auto H_rev = ve_rev.hessian(std::array{xv, yv});

  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto ve_fwd = Equation(xf * yf, xf * xf);
  auto H_fwd = ve_fwd.derivative_tensor<2>(std::array{xv, yv});

  for (const auto k : {0uz, 1uz})
    for (const auto i : {0uz, 1uz})
      for (const auto j : {0uz, 1uz})
        EXPECT_NEAR((H_fwd[k, i, j]), (H_rev[k, i, j]), 1e-12);
}
TEST(HessianForwardTest, WithValues) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto H = ve.derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((H[0, 0, 1]), 1.0);
  EXPECT_DOUBLE_EQ((H[1, 0, 0]), 2.0);
}
TEST(ScalarHessianTest, ReverseMode_XY) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto expr = x * y;
  auto H = Equation{expr}.hessian(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((H[0, 0]), 0.0);
  EXPECT_DOUBLE_EQ((H[0, 1]), 1.0);
  EXPECT_DOUBLE_EQ((H[1, 0]), 1.0);
  EXPECT_DOUBLE_EQ((H[1, 1]), 0.0);
}
TEST(ScalarHessianTest, ReverseMode_QuadraticForm) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto expr = x * x + constant(D{2.0}) * y * y;
  auto H = Equation{expr}.hessian(std::array{1.0, 1.0});
  EXPECT_DOUBLE_EQ((H[0, 0]), 2.0);
  EXPECT_DOUBLE_EQ((H[0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((H[1, 0]), 0.0);
  EXPECT_DOUBLE_EQ((H[1, 1]), 4.0);
}
TEST(ScalarHessianTest, ReverseMode_Symmetric) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto expr = exp(x * y);
  auto H = Equation{expr}.hessian(std::array{0.5, 1.5});
  EXPECT_NEAR((H[0, 1]), (H[1, 0]), 1e-12);
}
TEST(ScalarHessianTest, ForwardMode_XY) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = x * y;
  auto H = Equation{expr}.template derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((H[0, 0]), 0.0);
  EXPECT_DOUBLE_EQ((H[0, 1]), 1.0);
  EXPECT_DOUBLE_EQ((H[1, 0]), 1.0);
  EXPECT_DOUBLE_EQ((H[1, 1]), 0.0);
}
TEST(ScalarHessianTest, ForwardMode_QuadraticForm) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = x * x + constant(2.0) * y * y;
  auto H = Equation{expr}.template derivative_tensor<2>(std::array{1.0, 1.0});
  EXPECT_DOUBLE_EQ((H[0, 0]), 2.0);
  EXPECT_DOUBLE_EQ((H[0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((H[1, 0]), 0.0);
  EXPECT_DOUBLE_EQ((H[1, 1]), 4.0);
}
TEST(ScalarHessianTest, ReverseMode_NoValues) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto expr = x * y;
  auto H = Equation{expr}.hessian(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((H[0, 1]), 1.0);
  EXPECT_DOUBLE_EQ((H[1, 0]), 1.0);
  EXPECT_DOUBLE_EQ((H[0, 0]), 0.0);
  EXPECT_DOUBLE_EQ((H[1, 1]), 0.0);
}
TEST(ScalarHessianTest, ForwardMode_AtAPoint) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = x * y;
  auto H = Equation{expr}.template derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((H[0, 1]), 1.0);
  EXPECT_DOUBLE_EQ((H[1, 0]), 1.0);
  EXPECT_DOUBLE_EQ((H[0, 0]), 0.0);
  EXPECT_DOUBLE_EQ((H[1, 1]), 0.0);
}
TEST(ScalarHessianTest, ForwardAgreesWithReverse) {
  double xv = 0.5, yv = 1.5;

  using D = Dual<double>;
  Variable<D, FixedString{"x"}> xr;
  Variable<D, FixedString{"y"}> yr;
  auto expr_r = exp(xr * yr);
  auto H_rev = Equation{expr_r}.hessian(std::array{xv, yv});

  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto expr_f = exp(xf * yf);
  auto H_fwd =
      Equation{expr_f}.template derivative_tensor<2>(std::array{xv, yv});

  for (const auto i : {0uz, 1uz})
    for (const auto j : {0uz, 1uz})
      EXPECT_NEAR((H_fwd[i, j]), (H_rev[i, j]), 1e-12);
}
TEST(DerivativeTensorTest, Order1_ScalarVariable) {
  Variable<double, FixedString{"x"}> x;
  auto expr = x * x * x;
  auto T1 = Equation{expr}.template derivative_tensor<1>(std::array{3.0});
  EXPECT_NEAR(T1[0], 27.0, 1e-12);
}
TEST(DerivativeTensorTest, Order1_MatchesJacobian) {
  double xv = 1.5, yv = 2.5;
  (void)xv;
  (void)yv;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = x * x + x * y; // df/dx = 2x+y, df/dy = x

  auto T1 = Equation{expr}.template derivative_tensor<1>(std::array{xv, yv});
  EXPECT_NEAR(T1[0], 2 * xv + yv, 1e-12);
  EXPECT_NEAR(T1[1], xv, 1e-12);
}
TEST(DerivativeTensorTest, Order2_ScalarVariable) {
  Variable<double, FixedString{"x"}> x;
  auto expr = x * x * x;
  auto T2 = Equation{expr}.template derivative_tensor<2>(std::array{2.0});
  EXPECT_NEAR((T2[0, 0]), 12.0, 1e-12);
}
TEST(DerivativeTensorTest, Order2_MatchesHessian) {
  double xv = 0.7, yv = 1.3;
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> xr;
  Variable<D, FixedString{"y"}> yr;
  auto expr_r = exp(xr * yr);
  auto H_rev = Equation{expr_r}.hessian(std::array{xv, yv});

  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto expr_f = exp(xf * yf);
  auto H_fwd =
      Equation{expr_f}.template derivative_tensor<2>(std::array{xv, yv});

  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR((H_fwd[i, j]), (H_rev[i, j]), 1e-12);
}
TEST(DerivativeTensorTest, Order3_Polynomial) {
  Variable<double, FixedString{"x"}> x;
  auto expr = x * x * x * x;
  auto T3 = Equation{expr}.template derivative_tensor<3>(std::array{2.0});
  EXPECT_NEAR((T3[0, 0, 0]), 48.0, 1e-9);
}
TEST(DerivativeTensorTest, SecondOrderAtAPoint) {
  const double x0 = std::numbers::pi / 4.0;
  Variable<double, FixedString{"x"}> x;
  auto expr = sin(x);
  auto T2 = Equation{expr}.template derivative_tensor<2>(std::array{x0});
  EXPECT_NEAR((T2[0, 0]), -std::sin(x0), 1e-12);
}
TEST(DerivativeTensorTest, MixedPartials_Symmetric) {
  double xv = 0.5, yv = 1.5;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = exp(x * y);
  auto H = Equation{expr}.template derivative_tensor<2>(std::array{xv, yv});
  EXPECT_NEAR((H[0, 1]), (H[1, 0]), 1e-12);
}
TEST(DerivativeTensorTest, Equation_Order1_IsJacobian) {
  double xv = 1.0, yv = 2.0;
  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto ve_fwd = Equation(xf * yf, xf * xf);
  auto J_fwd = ve_fwd.derivative_tensor<1>(std::array{xv, yv});

  auto xr = var_of<"x">(xv);
  auto yr = var_of<"y">(yv);
  auto ve_rev = Equation(xr * yr, xr * xr);
  auto J_rev = ve_rev.jacobian(std::array{xv, yv});

  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR((J_fwd[i, j]), (J_rev[i, j]), 1e-12);
}
TEST(DerivativeTensorTest, Equation_Order2_IsHessianStack) {
  double xv = 1.5, yv = 2.5;
  (void)xv;
  (void)yv;
  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto ve_fwd = Equation(xf * yf, xf * xf);
  auto H_fwd = ve_fwd.derivative_tensor<2>(std::array{xv, yv});
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> xr;
  Variable<D, FixedString{"y"}> yr;
  auto ve_rev = Equation(xr * yr, xr * xr);
  auto H_rev = ve_rev.hessian(std::array{2.0, 3.0});

  for (std::size_t k = 0; k < 2; ++k)
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
        EXPECT_NEAR((H_fwd[k, i, j]), (H_rev[k, i, j]), 1e-12);
}
TEST(DerivativeTensorTest, Equation_Order3_TrigPolynomial) {
  Variable<double, FixedString{"x"}> x;
  auto ve = Equation(x * x * x * x);
  auto T3 = ve.derivative_tensor<3>(std::array{1.0});
  EXPECT_NEAR((T3[0, 0, 0]), 24.0, 1e-9); // one output: no leading axis
}
TEST(TutorialForward, SingleVar_DualNumbers) {
  double x0 = 2.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto f = 1.0 + x + x * x + 1.0 / x + log(x);
  auto [fv, df] = f.eval(Dual<double>{x0, 1.0});
  EXPECT_NEAR(fv, 1.0 + x0 + x0 * x0 + 1.0 / x0 + std::log(x0), 1e-12);
  EXPECT_NEAR(df, 1.0 + 2.0 * x0 - 1.0 / (x0 * x0) + 1.0 / x0, 1e-12);
}
TEST(TutorialMultiVar, ForwardReverseJacobianAgree) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  Variable<double, FixedString{"z"}> zf;
  auto f_fwd = 1.0 + xf + yf + zf + xf * yf + yf * zf + xf * zf + xf * yf * zf +
               exp(xf / yf + yf / zf);
  auto T1 =
      Equation{f_fwd}.template derivative_tensor<1>(std::array{xv, yv, zv});
  auto xr = var_of<"x">(xv);
  auto yr = var_of<"y">(yv);
  auto zr = var_of<"z">(zv);
  auto f_rev = constant(1.0) + xr + yr + zr + xr * yr + yr * zr + xr * zr +
               xr * yr * zr + exp(xr / yr + yr / zr);
  auto g = Equation{f_rev}.jacobian(xv, yv, zv);
  EXPECT_NEAR(T1[0], g[0], 1e-10);
  EXPECT_NEAR(T1[1], g[1], 1e-10);
  EXPECT_NEAR(T1[2], g[2], 1e-10);
}
TEST(TutorialJacobian, ForwardModeDerivativeTensor) {
  double xv = 1.0, yv = 0.5;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto f = sin(x) * cos(y) + exp(x * y);
  auto g = Equation{f}.template derivative_tensor<1>(std::array{xv, yv});
  EXPECT_NEAR(g[0], std::cos(xv) * std::cos(yv) + yv * std::exp(xv * yv),
              1e-12);
  EXPECT_NEAR(g[1], -std::sin(xv) * std::sin(yv) + xv * std::exp(xv * yv),
              1e-12);
}
TEST(TutorialJacobian, ForwardReverseAgree) {
  double xv = 1.0, yv = 0.5;
  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto T1 =
      Equation{sin(xf) * cos(yf) + exp(xf * yf)}.template derivative_tensor<1>(
          std::array{xv, yv});
  auto xr = var_of<"x">(xv);
  auto yr = var_of<"y">(yv);
  auto g = Equation{sin(xr) * cos(yr) + exp(xr * yr)}.jacobian(xv, yv);
  EXPECT_NEAR(T1[0], g[0], 1e-12);
  EXPECT_NEAR(T1[1], g[1], 1e-12);
}
TEST(TutorialHigherOrder, FourthOrder_SinX) {
  double x0 = std::numbers::pi / 4.0;
  Variable<double, FixedString{"x"}> x;
  auto T4 = Equation{sin(x)}.template derivative_tensor<4>(std::array{x0});
  EXPECT_NEAR((T4[0, 0, 0, 0]), std::sin(x0), 1e-9);
}
TEST(TutorialHigherOrder, FourthOrder_ExpX) {
  double x0 = 0.7;
  Variable<double, FixedString{"x"}> x;
  double ev = std::exp(x0);
  EXPECT_NEAR(
      (Equation{exp(x)}.template derivative_tensor<2>(std::array{x0})[0, 0]),
      ev, 1e-12);
  EXPECT_NEAR(
      (Equation{exp(x)}.template derivative_tensor<3>(std::array{x0})[0, 0, 0]),
      ev, 1e-12);
  EXPECT_NEAR((Equation{exp(x)}.template derivative_tensor<4>(
                  std::array{x0})[0, 0, 0, 0]),
              ev, 1e-9);
}
TEST(TutorialHigherOrder, FourthOrder_AllCrossPartialsOfSinXplusY) {
  double xv = 1.0, yv = 1.0;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto T4 =
      Equation{sin(x + y)}.template derivative_tensor<4>(std::array{xv, yv});
  double expected = std::sin(xv + yv);
  for (const auto i : {0uz, 1uz})
    for (const auto j : {0uz, 1uz})
      for (const auto k : {0uz, 1uz})
        for (const auto l : {0uz, 1uz})
          EXPECT_NEAR((T4[i, j, k, l]), expected, 1e-8);
}
TEST(TutorialHigherOrder, ThirdOrder_MixedPartial) {
  double xv = 1.0, yv = 2.0;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto T3 = Equation{exp(x) * y * y}.template derivative_tensor<3>(
      std::array{xv, yv});
  EXPECT_NEAR((T3[0, 0, 1]), 2.0 * yv * std::exp(xv), 1e-10);
  EXPECT_NEAR((T3[1, 1, 1]), 0.0, 1e-10); // d³(exp(x)*y²)/dy³ = 0
}
TEST(TutorialDirectional, FirstOrder_DualSeeding) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<Dual<double>, FixedString{"x"}> x;
  Variable<Dual<double>, FixedString{"y"}> y;
  auto [fv, dfdu] =
      (exp(x) * sin(y)).eval(Dual<double>{xv, u}, Dual<double>{yv, u});
  EXPECT_NEAR(fv, std::exp(xv) * std::sin(yv), 1e-12);
  EXPECT_NEAR(dfdu, std::exp(xv), 1e-12);
}
TEST(TutorialDirectional, FirstOrder_ViaJacobianDot) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto T1 = Equation{exp(x) * sin(y)}.template derivative_tensor<1>(
      std::array{xv, yv});
  EXPECT_NEAR(T1[0] * u + T1[1] * u, std::exp(xv), 1e-12);
}
TEST(TutorialDirectional, SecondOrder_HessianContraction) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto H = Equation{exp(x) * sin(y)}.template derivative_tensor<2>(
      std::array{xv, yv});
  double d2fdu2 = (H[0, 0]) * u * u + (H[0, 1]) * u * u + (H[1, 0]) * u * u +
                  (H[1, 1]) * u * u;
  EXPECT_NEAR(d2fdu2, std::exp(xv) * std::cos(yv), 1e-12);
}
TEST(TutorialDirectional, VectorFunction_JacobianTimesDirection) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(exp(x) * sin(y), x * y);
  auto J = ve.derivative_tensor<1>(std::array{xv, yv});
  double df0du = (J[0, 0]) * u + (J[0, 1]) * u;
  double df1du = (J[1, 0]) * u + (J[1, 1]) * u;
  EXPECT_NEAR(df0du, std::exp(xv), 1e-12);
  EXPECT_NEAR(df1du, (xv + yv) / std::sqrt(2.0), 1e-12);
}
TEST(TutorialTaylor, ScalarSin_FourthOrderAccuracy) {
  double x0 = std::numbers::pi / 4.0;
  double h = 0.05;
  Variable<double, FixedString{"x"}> x;
  auto f = sin(x);
  double f0 = std::sin(x0);
  auto T1 = Equation{f}.template derivative_tensor<1>(std::array{x0});
  auto T2 = Equation{f}.template derivative_tensor<2>(std::array{x0});
  auto T3 = Equation{f}.template derivative_tensor<3>(std::array{x0});
  auto T4 = Equation{f}.template derivative_tensor<4>(std::array{x0});
  double taylor = f0 + T1[0] * h + (T2[0, 0]) * h * h / 2.0 +
                  (T3[0, 0, 0]) * h * h * h / 6.0 +
                  (T4[0, 0, 0, 0]) * h * h * h * h / 24.0;
  EXPECT_NEAR(taylor, std::sin(x0 + h), 1e-7);
}
TEST(TutorialTaylor, VectorFunction_SecondOrderAccuracy) {
  double xv = 0.0, yv = 0.0, h = 0.1;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(sin(x) * cos(y), exp(x + y));
  auto f0 = ve.evaluate(std::array{xv, yv});
  auto J = ve.derivative_tensor<1>(std::array{xv, yv});
  auto H = ve.derivative_tensor<2>(std::array{xv, yv});
  double taylor1 =
      f0[1] + ((J[1, 0]) + (J[1, 1])) * h +
      0.5 * ((H[1, 0, 0]) + 2.0 * (H[1, 0, 1]) + (H[1, 1, 1])) * h * h;
  EXPECT_NEAR(taylor1, std::exp(2.0 * h), 2e-3);
}
TEST(TutorialReverseHessian, QuadraticForm) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto f = x * x + x * y + y * y;
  auto H = Equation{f}.hessian(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ((H[0, 0]), 2.0);
  EXPECT_DOUBLE_EQ((H[0, 1]), 1.0);
  EXPECT_DOUBLE_EQ((H[1, 0]), 1.0);
  EXPECT_DOUBLE_EQ((H[1, 1]), 2.0);
}
TEST(TutorialReverseHessian, TrigFunction) {
  using D = Dual<double>;
  double xv = std::numbers::pi / 4.0, yv = std::numbers::pi / 6.0;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto f = sin(x) * cos(y);
  auto H = Equation{f}.hessian(std::array{xv, yv});
  EXPECT_NEAR((H[0, 0]), -std::sin(xv) * std::cos(yv), 1e-12);
  EXPECT_NEAR((H[0, 1]), -std::cos(xv) * std::sin(yv), 1e-12);
  EXPECT_NEAR((H[1, 0]), -std::cos(xv) * std::sin(yv), 1e-12);
  EXPECT_NEAR((H[1, 1]), -std::sin(xv) * std::cos(yv), 1e-12);
}
TEST(TutorialReverseHessian, JacobianFromSameExpression) {
  using D = Dual<double>;
  double xv = 2.0, yv = 3.0;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto f = x * x + x * y + y * y;
  auto g = Equation{f}.jacobian(D{xv}, D{yv});
  EXPECT_NEAR(g[0], 2.0 * xv + yv, 1e-12); // df/dx = 2x + y
  EXPECT_NEAR(g[1], xv + 2.0 * yv, 1e-12); // df/dy = x + 2y
}
TEST(TutorialReverseHigherOrder, SingleVar_SecondAndThird) {
  double x0 = 1.0;
  Variable<double, FixedString{"x"}> x;
  EXPECT_NEAR(
      (Equation{sin(x)}.template derivative_tensor<2>(std::array{x0})[0, 0]),
      -std::sin(x0), 1e-12);
  EXPECT_NEAR(
      (Equation{sin(x)}.template derivative_tensor<3>(std::array{x0})[0, 0, 0]),
      -std::cos(x0), 1e-12);
}
TEST(TutorialReverseHigherOrder, MultiVar_Hessian) {
  using D = Dual<double>;
  double xv = 2.0, yv = 3.0;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto f = x * x * y + y * y * y;
  auto H = Equation{f}.hessian(std::array{xv, yv});
  EXPECT_NEAR((H[0, 0]), 2.0 * yv, 1e-12);
  EXPECT_NEAR((H[0, 1]), 2.0 * xv, 1e-12);
  EXPECT_NEAR((H[1, 0]), 2.0 * xv, 1e-12);
  EXPECT_NEAR((H[1, 1]), 6.0 * yv, 1e-12);
}
TEST(TutorialReverseHigherOrder, ForwardReverseHessianAgree) {
  double xv = 0.8, yv = 1.2;
  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto f_fwd = sin(xf) * exp(yf);
  auto H_fwd =
      Equation{f_fwd}.template derivative_tensor<2>(std::array{xv, yv});
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> xr;
  Variable<D, FixedString{"y"}> yr;
  auto f_rev = sin(xr) * exp(yr);
  auto H_rev = Equation{f_rev}.hessian(std::array{xv, yv});
  for (const auto i : {0uz, 1uz})
    for (const auto j : {0uz, 1uz})
      EXPECT_NEAR((H_fwd[i, j]), (H_rev[i, j]), 1e-12);
}
TEST(DualScalarContract, ValAndToDouble) {
  const dual2nd a = embed_constant<double, 2>(3.5);
  EXPECT_DOUBLE_EQ(val(a), 3.5);
  EXPECT_DOUBLE_EQ(to_double(a), 3.5);
  EXPECT_DOUBLE_EQ(val(2.0), 2.0); // identity on a plain scalar
}
TEST(DualScalarContract, ScalarMixingAtDepth) {
  // x = 2, seeded so the outer first-derivative slot tracks d/dx.
  const dual2nd x{Dual<double>{2.0, 1.0}, Dual<double>{1.0, 0.0}};
  // Bare doubles mix at depth 2.
  const dual2nd f = 3.0 * x + x * 2.0 - 1.0;
  EXPECT_DOUBLE_EQ(val(f), 9.0);
  EXPECT_DOUBLE_EQ(f.get<1>().get<0>(), 5.0);
}
TEST(DualScalarContract, PowMaxMinAndComparisons) {
  const dual2nd x = embed_constant<double, 2>(2.0);
  EXPECT_DOUBLE_EQ(val(pow(x, 3.0)), 8.0); // scalar exponent
  EXPECT_DOUBLE_EQ(val(pow(x, x)), 4.0);   // dual exponent
  EXPECT_DOUBLE_EQ(val(max(x, 5.0)), 5.0);
  EXPECT_DOUBLE_EQ(val(min(x, 5.0)), 2.0);
  EXPECT_TRUE(x < 3.0);
  EXPECT_TRUE(x <= 2.0);
  EXPECT_FALSE(x > 2.0);
  EXPECT_TRUE(x == 2.0);
  EXPECT_TRUE(x != 1.0);
}
TEST(DualScalarContract, ImplicitConstantFromScalarAtDepth) {
  // Generic numeric code written against a plain scalar relies on S{0} and
  // S c = 1.0 meaning a zero-derivative constant.
  const dual2nd a{0};    // brace-init from int
  const dual2nd b = 2.5; // copy-init needs a non-explicit ctor
  const std::vector<dual2nd> v(3, dual2nd{1}); // vector fill from a scalar
  EXPECT_DOUBLE_EQ(val(a), 0.0);
  EXPECT_DOUBLE_EQ(val(b), 2.5);
  EXPECT_DOUBLE_EQ(val(v[0]), 1.0);
  EXPECT_DOUBLE_EQ(b.get<1>().get<0>(), 0.0);
  EXPECT_DOUBLE_EQ(b.get<0>().get<1>(), 0.0);
  EXPECT_DOUBLE_EQ(b.get<1>().get<1>(), 0.0);
}
TEST(ForwardDriver, JacobianAndHessianCrossTerm) {
  auto f = [](auto x) { return x[0] * x[0] * x[1] + x[1] * x[1] * x[1]; };
  const std::array<double, 2> x{2.0, 3.0};
  const std::span<const double> xs{x.data(), x.size()};

  const auto H = ddx::impl::hessian(f, xs);
  EXPECT_DOUBLE_EQ(val_of(H), 39.0); // 4*3 + 27
  ASSERT_EQ(hess_n(H), 2u);
  EXPECT_DOUBLE_EQ(grad_at(H, 0), 12.0);   // 2 x0 x1
  EXPECT_DOUBLE_EQ(grad_at(H, 1), 31.0);   // x0^2 + 3 x1^2
  EXPECT_DOUBLE_EQ(hess_at(H, 0, 0), 6.0); // 2 x1
  EXPECT_DOUBLE_EQ(hess_at(H, 0, 1), 4.0); // 2 x0
  EXPECT_DOUBLE_EQ(hess_at(H, 1, 0), 4.0);
  EXPECT_DOUBLE_EQ(hess_at(H, 1, 1), 18.0); // 6 x1

  const auto g = ddx::impl::jacobian(f, xs);
  EXPECT_DOUBLE_EQ(g[0], 12.0);
  EXPECT_DOUBLE_EQ(g[1], 31.0);
}
TEST(ForwardDriver, IdealMixingHessianMatchesClosedForm) {
  // Mirrors the downstream CALPHAD oracle: G = R T (y0 ln y0 + y1 ln y1).
  const double R = 8.31446261815324;
  const double T = 1000.0;
  auto f = [R, T](auto y) {
    using std::log; // ADL still picks ddx::impl::log for the dual argument
    return R * T * (y[0] * log(y[0]) + y[1] * log(y[1]));
  };
  const std::array<double, 2> y{0.3, 0.7};
  const auto H =
      ddx::impl::hessian(f, std::span<const double>{y.data(), y.size()});

  EXPECT_NEAR(val_of(H), R * T * (0.3 * std::log(0.3) + 0.7 * std::log(0.7)),
              1e-6);
  EXPECT_NEAR(grad_at(H, 0), R * T * (std::log(0.3) + 1.0), 1e-6);
  EXPECT_NEAR(grad_at(H, 1), R * T * (std::log(0.7) + 1.0), 1e-6);
  EXPECT_NEAR(hess_at(H, 0, 0), R * T / 0.3, 1e-3);
  EXPECT_NEAR(hess_at(H, 1, 1), R * T / 0.7, 1e-3);
  EXPECT_NEAR(hess_at(H, 0, 1), 0.0, 1e-6); // no cross term
}
TEST(ScopedValue, NestedGuardsOverSiblingScalarsOfOneDual) {
  // Two independently scoped guards over the SAME dual2nd's seeds.
  dual2nd d{Dual<double>{2.0, 0.0}, Dual<double>{0.0, 0.0}};
  {
    const auto inner = scoped_seed<1.0>(d.value().deriv());
    {
      const auto outer = scoped_seed<1.0>(d.deriv().value());
      EXPECT_DOUBLE_EQ(d.value().deriv(), 1.0); // both live at once
      EXPECT_DOUBLE_EQ(d.deriv().value(), 1.0);
    }
    EXPECT_DOUBLE_EQ(d.deriv().value(), 0.0); // outer cleared
    EXPECT_DOUBLE_EQ(d.value().deriv(), 1.0); // inner survives, unclobbered
  }
  EXPECT_DOUBLE_EQ(d.value().deriv(), 0.0);
  EXPECT_DOUBLE_EQ(d.value().value(), 2.0); // the point never moved
}

TEST(ScopedValue, UsableDuringConstantEvaluation) {
  // reverse_mode_hessian and hessian_forward_over_reverse are constexpr, so
  // the guard must be too.
  static constexpr auto probe = []() constexpr {
    double slot = 3.0;
    double seen = 0.0;
    {
      const auto seed = scoped_seed<1.0>(slot);
      seen = slot;
    }
    return std::array<double, 2>{seen, slot};
  }();
  static_assert(probe[0] == 1.0);
  static_assert(probe[1] == 3.0);
  SUCCEED();
}
TEST(ForwardDriver, RepeatedCallsDoNotLeakSeeds) {
  // A seed left behind by one sweep would show up as a wrong second answer.
  auto f = [](auto x) { return x[0] * x[0] * x[1] + x[1] * x[1] * x[1]; };
  const std::array<double, 2> x{2.0, 3.0};
  const std::span<const double> xs{x.data(), x.size()};

  const auto g1 = ddx::impl::jacobian(f, xs);
  const auto g2 = ddx::impl::jacobian(f, xs);
  EXPECT_DOUBLE_EQ(g1[0], g2[0]);
  EXPECT_DOUBLE_EQ(g1[1], g2[1]);

  const auto H1 = ddx::impl::hessian(f, xs);
  const auto H2 = ddx::impl::hessian(f, xs);
  EXPECT_DOUBLE_EQ(val_of(H1), val_of(H2));
  for (std::size_t i = 0; i < hess_n(H1); ++i) {
    EXPECT_DOUBLE_EQ(grad_at(H1, i), grad_at(H2, i));
    for (std::size_t j = 0; j < hess_n(H1); ++j) {
      EXPECT_DOUBLE_EQ(hess_at(H1, i, j), hess_at(H2, i, j));
    }
  }
}
