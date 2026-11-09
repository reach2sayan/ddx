#include "dual/tests_dual_common.hpp"
#include "tests_math_fixtures.hpp"

TEST(NewMathFunctions, MaxMinTiesAverageAndNaNPropagatesSymmetrically) {
  auto x = var<"x">;
  auto y = var<"y">;
  // A tie is the mean of the two subgradients in every engine.
  auto tie = Equation{max(x, y)}.jacobian(2.0, 2.0);
  EXPECT_DOUBLE_EQ(tie[0], 0.5);
  EXPECT_DOUBLE_EQ(tie[1], 0.5);
  // A NaN operand propagates from either side, value and derivative both;
  // fmax-style dropping would depend on the arena's operand order.
  const double qnan = std::nan("");
  EXPECT_TRUE(std::isnan(max(x, y).eval(qnan, 2.0)));
  EXPECT_TRUE(std::isnan(max(x, y).eval(2.0, qnan)));
  EXPECT_TRUE(std::isnan(min(x, y).eval(qnan, 2.0)));
  EXPECT_TRUE(std::isnan(min(x, y).eval(2.0, qnan)));
  auto gnan = Equation{max(x, y)}.jacobian(2.0, qnan);
  EXPECT_TRUE(std::isnan(gnan[0]));
  EXPECT_TRUE(std::isnan(gnan[1]));
  // The derivative of abs uses sign(0) = 0 and sign(NaN) = NaN.
  EXPECT_DOUBLE_EQ(abs(Dual{0.0, 1.0}).deriv(), 0.0);
  EXPECT_TRUE(std::isnan(abs(Dual{qnan, 1.0}).deriv()));
}

// The scalar-first overloads compare in the opposite order, and Dual's
// operator== reads only the value, so the derivative is checked as well.
TEST(NewMathFunctions, MaxMinAgreeWhicheverSideTheScalarIsOn) {
  const auto same = [](const Dual<double> &l, const Dual<double> &r) {
    ASSERT_EQ(std::isnan(l.value()), std::isnan(r.value()));
    ASSERT_EQ(std::isnan(l.deriv()), std::isnan(r.deriv()));
    if (!std::isnan(l.value())) {
      EXPECT_DOUBLE_EQ(l.value(), r.value());
    }
    if (!std::isnan(l.deriv())) {
      EXPECT_DOUBLE_EQ(l.deriv(), r.deriv());
    }
  };
  const Dual<double> below{2.0, 1.0};
  const Dual<double> above{9.0, 3.0};
  const Dual<double> tie{5.0, 7.0};
  const double s = 5.0;
  const double qnan = std::nan("");

  for (const auto &d : {below, above, tie}) {
    same(max(d, s), max(s, d));
    same(min(d, s), min(s, d));
  }
  same(max(below, qnan), max(qnan, below));
  same(min(below, qnan), min(qnan, below));

  // The winner is taken whole, so a losing dual contributes no derivative and
  // the scalar none either way; a tie halves, as it does between two duals.
  EXPECT_DOUBLE_EQ(max(below, s).deriv(), 0.0);
  EXPECT_DOUBLE_EQ(min(below, s).deriv(), 1.0);
  EXPECT_DOUBLE_EQ(max(above, s).deriv(), 3.0);
  EXPECT_DOUBLE_EQ(min(above, s).deriv(), 0.0);
  EXPECT_DOUBLE_EQ(max(tie, s).deriv(), 3.5);
  EXPECT_DOUBLE_EQ(min(tie, s).deriv(), 3.5);
  EXPECT_TRUE(std::isnan(max(below, qnan).value()));
  EXPECT_TRUE(std::isnan(max(below, qnan).deriv()));
}
TEST(NewMathFunctions, ForwardDualUnary) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  {
    auto [f, df] = erf(x).eval(Dual{x0, 1.0});
    EXPECT_DOUBLE_EQ(f, std::erf(x0));
    EXPECT_DOUBLE_EQ(df, k2OverSqrtPi * std::exp(-x0 * x0));
  }
  {
    auto [f, df] = atanh(x).eval(Dual{x0, 1.0});
    EXPECT_DOUBLE_EQ(f, std::atanh(x0));
    EXPECT_DOUBLE_EQ(df, 1.0 / (1.0 - x0 * x0));
  }
}
TEST(NewMathFunctions, ForwardDualBinary) {
  double x0 = 3.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  {
    auto [f, df] = hypot(x, 4.0).eval(Dual{x0, 1.0});
    EXPECT_DOUBLE_EQ(f, 5.0);
    EXPECT_DOUBLE_EQ(df, x0 / 5.0);
  }
  {
    auto [f, df] = pow(x, 3.0).eval(Dual{x0, 1.0});
    EXPECT_DOUBLE_EQ(f, 27.0);
    EXPECT_DOUBLE_EQ(df, 3.0 * x0 * x0);
  }
}
TEST(NewMathFunctions, DualDirect) {
  Dual<double> x{0.7, 1.0};
  Dual<double> r = asinh(x);
  EXPECT_DOUBLE_EQ(r.get<0>(), std::asinh(0.7));
  EXPECT_DOUBLE_EQ(r.get<1>(), 1.0 / std::sqrt(0.49 + 1.0));

  Dual<double> a{1.0, 1.0}, b{2.0, 0.0}, c{2.0, 0.0};
  Dual<double> h = hypot(a, b, c);
  EXPECT_DOUBLE_EQ(h.get<0>(), 3.0);
  EXPECT_DOUBLE_EQ(h.get<1>(), 1.0 / 3.0);
}
TEST(NewMathFunctions, TaylorFirstAndSecondDerivative) {
  EXPECT_NEAR(Equation{asinh(var<"x">)}.template univariate_derivative<1>(0.7),
              1.0 / std::sqrt(0.49 + 1.0), 1e-12);
  EXPECT_NEAR(Equation{erf(var<"x">)}.template univariate_derivative<1>(0.5),
              k2OverSqrtPi * std::exp(-0.25), 1e-12);
  EXPECT_NEAR(Equation{cbrt(var<"x">)}.template univariate_derivative<1>(2.0),
              1.0 / (3.0 * std::cbrt(2.0) * std::cbrt(2.0)), 1e-12);

  {
    double x0 = 0.7;
    EXPECT_NEAR(
        Equation{asinh(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        -x0 / std::pow(1.0 + x0 * x0, 1.5), 1e-10);
  }
  {
    double x0 = 0.5;
    EXPECT_NEAR(
        Equation{erf(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        -2.0 * x0 * k2OverSqrtPi * std::exp(-x0 * x0), 1e-10);
  }
  {
    double x0 = 0.3;
    EXPECT_NEAR(
        Equation{atanh(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        2.0 * x0 / std::pow(1.0 - x0 * x0, 2.0), 1e-10);
  }
  {
    double x0 = 3.0;
    EXPECT_NEAR(
        Equation{log10(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        -1.0 / (x0 * x0 * kLn10), 1e-10);
  }
  {
    double x0 = 2.0;
    EXPECT_NEAR(
        Equation{cbrt(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        -2.0 / 9.0 * std::pow(x0, -5.0 / 3.0), 1e-10);
  }
  {
    double x0 = 2.0;
    EXPECT_NEAR(
        Equation{acosh(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        -x0 / std::pow(x0 * x0 - 1.0, 1.5), 1e-10);
  }
}
TEST(NewMathFunctions, TaylorBinaryUnivariate) {
  EXPECT_NEAR(
      Equation{pow(var<"x">, 3.0)}.template univariate_derivative<1>(2.0), 12.0,
      1e-10);
  EXPECT_NEAR(
      Equation{pow(var<"x">, 3.0)}.template univariate_derivative<2>(2.0), 12.0,
      1e-10);
  {
    double x0 = 3.0, h = std::sqrt(x0 * x0 + 4.0);
    EXPECT_NEAR(
        Equation{hypot(var_of<"x">(x0), 2.0)}.template univariate_derivative<1>(
            x0),
        x0 / h, 1e-10);
  }
  {
    double x0 = 2.0;
    EXPECT_NEAR(
        Equation{atan2(var_of<"x">(x0), 2.0)}.template univariate_derivative<1>(
            x0),
        2.0 / (4.0 + x0 * x0), 1e-10);
  }
}
// The asin/atan/asinh/acosh/atanh recurrences must use the argument's actual
// derivative series u', not assume the bare seed u'=1.
TEST(NewMathFunctions, TaylorCompositeArguments) {
  {
    double x0 = 0.6, x4 = x0 * x0 * x0 * x0, d = 1.0 + x4;
    auto x = var_of<"x">(x0);
    EXPECT_NEAR(Equation{atan(x * x)}.template univariate_derivative<1>(x0),
                2.0 * x0 / d, 1e-10);
    EXPECT_NEAR(Equation{atan(x * x)}.template univariate_derivative<2>(x0),
                (2.0 - 6.0 * x4) / (d * d), 1e-10);
  }
  {
    double x0 = 1.0, a = 0.5, r = 1.0 - a * a * x0 * x0;
    auto x = var_of<"x">(x0);
    EXPECT_NEAR(Equation{asin(0.5 * x)}.template univariate_derivative<1>(x0),
                a / std::sqrt(r), 1e-10);
    EXPECT_NEAR(Equation{asin(0.5 * x)}.template univariate_derivative<2>(x0),
                a * a * a * x0 * std::pow(r, -1.5), 1e-10);
  }
  {
    double x0 = 0.3, a = 2.0, r = 1.0 + a * a * x0 * x0;
    auto x = var_of<"x">(x0);
    EXPECT_NEAR(Equation{asinh(2.0 * x)}.template univariate_derivative<1>(x0),
                a / std::sqrt(r), 1e-10);
    EXPECT_NEAR(Equation{asinh(2.0 * x)}.template univariate_derivative<2>(x0),
                -a * a * a * x0 * std::pow(r, -1.5), 1e-10);
  }
  {
    auto x = var<"x">;
    EXPECT_NEAR(
        Equation{atan2(sin(x), cos(x))}.template univariate_derivative<1>(0.7),
        1.0, 1e-10);
    EXPECT_NEAR(
        Equation{atan2(sin(x), cos(x))}.template univariate_derivative<2>(0.7),
        0.0, 1e-10);
    EXPECT_NEAR(
        Equation{atan2(sin(x), cos(x))}.template univariate_derivative<3>(0.7),
        0.0, 1e-10);
  }
  {
    double x0 = 0.5;
    auto x = var_of<"x">(x0);
    EXPECT_NEAR(
        Equation{pow(sin(x), 2.0)}.template univariate_derivative<1>(x0),
        std::sin(2.0 * x0), 1e-10);
    EXPECT_NEAR(
        Equation{pow(sin(x), 2.0)}.template univariate_derivative<2>(x0),
        2.0 * std::cos(2.0 * x0), 1e-10);
  }
}

// Forward and reverse spell the quotient rule differently, so the tangent
// agrees with reverse mode to a few ULP rather than bit for bit.
#define EXPECT_TANGENT_MATCHES_REVERSE(expr_, xv_)                             \
  do {                                                                         \
    const auto e_ = (expr_);                                                   \
    const std::array<double, 1> pt_{xv_};                                      \
    const auto t_ = e_.template eval_with_tangent<"x">(xv_);                   \
    EXPECT_DOUBLE_EQ(t_.value(), e_.eval(xv_));                                \
    EXPECT_NEAR(t_.deriv(), Equation{e_}.jacobian(pt_)[0], 1e-12);             \
  } while (0)

TEST(ForwardTangentSweep, LeafSeeding) {
  auto x = var<"x">;
  auto y = var<"y">;

  const auto tx = x.eval_with_tangent<"x">(2.0);
  EXPECT_DOUBLE_EQ(tx.value(), 2.0);
  EXPECT_DOUBLE_EQ(tx.deriv(), 1.0);

  // Every other variable has tangent 0, including a name that differs from
  // the seeded one only in length.
  const auto ty = y.eval_with_tangent<"x">(3.0);
  EXPECT_DOUBLE_EQ(ty.value(), 3.0);
  EXPECT_DOUBLE_EQ(ty.deriv(), 0.0);
  EXPECT_DOUBLE_EQ(x.eval_with_tangent<"xx">(2.0).deriv(), 0.0);

  const auto tc = constant(5.0).eval_with_tangent<"x">();
  EXPECT_DOUBLE_EQ(tc.value(), 5.0);
  EXPECT_DOUBLE_EQ(tc.deriv(), 0.0);
}
TEST(ForwardTangentSweep, ArithmeticRules) {
  double x0 = 1.7;
  auto x = var_of<"x">(x0);

  EXPECT_TANGENT_MATCHES_REVERSE(x + x * x, 0.7);
  EXPECT_TANGENT_MATCHES_REVERSE(x - 3.0 * x * x, 0.7);
  EXPECT_TANGENT_MATCHES_REVERSE((x + 1.0) / x, 0.7);
  EXPECT_TANGENT_MATCHES_REVERSE(-(x / (x * x + 2.0)), 0.7);

  // derivative() is a total derivative, so this third check is only valid for
  // a single-variable expression.
  const auto e = (x * x + 1.0) / (x + 2.0);
  EXPECT_DOUBLE_EQ(e.eval_with_tangent<"x">(x0).deriv(),
                   e.derivative().eval(x0));
}
TEST(ForwardTangentSweep, UnaryMathRules) {
  // One probe per unary math function, each inside its domain.
  EXPECT_TANGENT_MATCHES_REVERSE(sin(var<"x">), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(cos(var<"x">), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(tan(var<"x">), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(exp(var<"x">), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(log(var<"x">), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(log10(var<"x">), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(sqrt(var<"x">), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(cbrt(var<"x">), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(asin(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(acos(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(atan(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(sinh(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(cosh(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(tanh(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(asinh(var<"x">), 0.7);
  EXPECT_TANGENT_MATCHES_REVERSE(acosh(var<"x">), 2.0);
  EXPECT_TANGENT_MATCHES_REVERSE(atanh(var<"x">), 0.3);
  EXPECT_TANGENT_MATCHES_REVERSE(erf(var<"x">), 0.5);

  EXPECT_DOUBLE_EQ(abs(var<"x">).eval_with_tangent<"x">(2.0).deriv(), 1.0);
  EXPECT_DOUBLE_EQ(abs(var<"x">).eval_with_tangent<"x">(-2.0).deriv(), -1.0);

  EXPECT_TANGENT_MATCHES_REVERSE(log(sin(var<"x">) + 2.0), 0.9);
}
TEST(ForwardTangentSweep, BinaryMathRules) {
  double x0 = 3.0, y0 = 4.0;
  auto x = var_of<"x">(x0);
  auto y = var_of<"y">(y0);

  // Symbols sort alphabetically, so Jacobian slot 0 is x and slot 1 is y.
  const auto check = [x0, y0](const auto &e, double dx, double dy) {
    EXPECT_DOUBLE_EQ(e.template eval_with_tangent<"x">(x0, y0).value(),
                     e.eval(x0, y0));
    EXPECT_DOUBLE_EQ(e.template eval_with_tangent<"x">(x0, y0).deriv(), dx);
    EXPECT_DOUBLE_EQ(e.template eval_with_tangent<"y">(x0, y0).deriv(), dy);
  };

  check(hypot(x, y), x0 / 5.0, y0 / 5.0);
  check(pow(x, y), y0 * std::pow(x0, y0 - 1.0),
        std::pow(x0, y0) * std::log(x0));
  // atan2(y, x) takes the numerator as its first operand.
  const double q = x0 * x0 + y0 * y0;
  check(atan2(y, x), -y0 / q, x0 / q);
  check(max(x, y), 0.0, 1.0);
  check(min(x, y), 1.0, 0.0);
}
TEST(ForwardTangentSweep, MultiVariableMatchesReverseMode) {
  auto x = var<"x">;
  auto y = var<"y">;

  const auto e = sin(x * y) * exp(x + y) + log(y * y + 1.0) / (x + 2.0) -
                 hypot(x, y) * tanh(x);

  const auto g = Equation{e}.jacobian(0.8, 1.3);
  const auto tx = e.eval_with_tangent<"x">(0.8, 1.3);
  const auto ty = e.eval_with_tangent<"y">(0.8, 1.3);

  EXPECT_DOUBLE_EQ(tx.value(), e.eval(0.8, 1.3));
  EXPECT_DOUBLE_EQ(ty.value(), e.eval(0.8, 1.3));
  EXPECT_NEAR(tx.deriv(), g[0], 1e-12);
  EXPECT_NEAR(ty.deriv(), g[1], 1e-12);

  EXPECT_DOUBLE_EQ(e.eval_with_tangent<"z">(0.8, 1.3).deriv(), 0.0);
}
TEST(ForwardTangentSweep, IsConstexpr) {
  // The expression uses no std:: transcendentals, so the sweep is constexpr on
  // every toolchain.
  constexpr auto x = var<"x">;
  constexpr auto e = x * x + 3.0 * x;
  constexpr auto t = e.eval_with_tangent<"x">(2.0);
  static_assert(t.value() == 10.0, "x² + 3x at x = 2");
  static_assert(t.deriv() == 7.0, "2x + 3 at x = 2");
  EXPECT_DOUBLE_EQ(t.deriv(), 7.0);
}
// A frozen symbol is a constant: same value lookup, zero derivative.  A tree
// holding one is a partial's row, not a model: Equation refuses to lower it.
TEST(FrozenVariable, EvaluatesAndSeesZeroDerivative) {
  auto e = make_all_constant_except<ddx::impl::FixedString{"y"}>(var<"x"> *
                                                                 var<"y">);

  EXPECT_DOUBLE_EQ(e.eval(2.0, 3.0), 6.0);
  EXPECT_DOUBLE_EQ(e.eval_with_tangent<"x">(2.0, 3.0).deriv(), 0.0);
  EXPECT_DOUBLE_EQ(e.eval_with_tangent<"y">(2.0, 3.0).deriv(), 2.0);
}

#undef EXPECT_TANGENT_MATCHES_REVERSE

// Every unary math function must instantiate at 3 variables and agree with
// reverse mode.
TEST(ForwardJacobian, CoversEveryUnaryMathFunction) {
  [[maybe_unused]] const std::array<double, 3> pt{0.6, 0.4, 1.3};
  [[maybe_unused]] auto y = var<"y">;
  [[maybe_unused]] auto z = var<"z">;
}

// The value coefficient of a Taylor log10 is std::log10's, not log's scaled.
TEST(NewMathFunctions, TaylorLog10ValueIsExact) {
  using T = ddx::impl::TaylorDual<double, 2>;
  const T w = log10(T::variable(1000.0));
  EXPECT_EQ(w.c[0], 3.0);
  EXPECT_DOUBLE_EQ(w.c[1], std::numbers::log10e / 1000.0);
}

// u^2.5 at u == 0 is zero below order 2.5, as Dual says for the first
// derivative, and has no coefficient past it.
TEST(NewMathFunctions, TaylorPowOfANonWholeExponentAtZero) {
  using T = ddx::impl::TaylorDual<double, 3>;
  const T w = pow(T::variable(0.0), 2.5);
  EXPECT_EQ(w.c[0], 0.0);
  EXPECT_EQ(w.c[1], 0.0);
  EXPECT_EQ(w.c[2], 0.0);
  EXPECT_TRUE(std::isnan(w.c[3]));
  EXPECT_EQ(pow(Dual{0.0, 1.0}, 2.5).deriv(), 0.0);
}

// A constant on either side agrees with the same constant embedded as a
// series, coefficient for coefficient.
TEST(NewMathFunctions, TaylorConstantOperandsMatchTheEmbeddedSeries) {
  using T = ddx::impl::TaylorDual<double, 3>;
  const T u = sin(T::variable(0.7));
  const T k{2.5};
  const auto same = [](const T &a, const T &b) {
    for (const std::size_t i : std::views::iota(0uz, a.c.size())) {
      EXPECT_DOUBLE_EQ(a.c[i], b.c[i]) << "coefficient " << i;
    }
  };
  same(u + 2.5, u + k);
  same(2.5 + u, k + u);
  same(u - 2.5, u - k);
  same(2.5 - u, k - u);
  same(u * 2.5, u * k);
  same(2.5 * u, k * u);
  same(u / 2.5, u / k);
  same(2.5 / u, k / u);
  same(u * 2, u * T{2.0});
  same(pow(u, 2.5), pow(u, k));
}

// atan2's recurrence never squares an operand past its range.
TEST(NewMathFunctions, TaylorAtan2OfHugeOperands) {
  using T = ddx::impl::TaylorDual<double, 2>;
  const double y0 = 3e200;
  const double x0 = 4e200;
  const T w = atan2(T::variable(y0), T{x0});
  EXPECT_DOUBLE_EQ(w.c[0], std::atan2(y0, x0));
  EXPECT_DOUBLE_EQ(w.c[1], x0 / std::hypot(x0, y0) / std::hypot(x0, y0));
  const T small = atan2(T::variable(0.3), T{0.4});
  EXPECT_DOUBLE_EQ(small.c[1], 0.4 / 0.25);
}

// A whole-number constant divided by a Dual: no narrowing in the quotient.
TEST(NewMathFunctions, AnIntOverADual) {
  const int two = 2;
  const auto q = two / Dual{4.0, 1.0};
  EXPECT_DOUBLE_EQ(q.value(), 0.5);
  EXPECT_DOUBLE_EQ(q.deriv(), -2.0 / 16.0);
}
