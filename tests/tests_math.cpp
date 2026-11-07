#include "tests_math_fixtures.hpp"

TEST(NewMathFunctions, ExprUnaryValueAndDerivative) {
  {
    double x0 = 3.0;
    auto x = var_of<"x">(x0);
    EXPECT_DOUBLE_EQ(log10(x).eval(x0), std::log10(x0));
    EXPECT_DOUBLE_EQ(log10(x).derivative().eval(x0), 1.0 / (x0 * kLn10));
  }
  {
    double x0 = 2.0;
    auto x = var_of<"x">(x0);
    EXPECT_DOUBLE_EQ(cbrt(x).eval(x0), std::cbrt(x0));
    EXPECT_DOUBLE_EQ(cbrt(x).derivative().eval(x0),
                     1.0 / (3.0 * std::cbrt(x0) * std::cbrt(x0)));
  }
  {
    double x0 = 0.7;
    auto x = var_of<"x">(x0);
    EXPECT_DOUBLE_EQ(asinh(x).eval(x0), std::asinh(x0));
    EXPECT_DOUBLE_EQ(asinh(x).derivative().eval(x0),
                     1.0 / std::sqrt(x0 * x0 + 1.0));
  }
  {
    double x0 = 2.0;
    auto x = var_of<"x">(x0);
    EXPECT_DOUBLE_EQ(acosh(x).eval(x0), std::acosh(x0));
    EXPECT_DOUBLE_EQ(acosh(x).derivative().eval(x0),
                     1.0 / std::sqrt(x0 * x0 - 1.0));
  }
  {
    double x0 = 0.3;
    auto x = var_of<"x">(x0);
    EXPECT_DOUBLE_EQ(atanh(x).eval(x0), std::atanh(x0));
    EXPECT_DOUBLE_EQ(atanh(x).derivative().eval(x0), 1.0 / (1.0 - x0 * x0));
  }
  {
    double x0 = 0.5;
    auto x = var_of<"x">(x0);
    EXPECT_DOUBLE_EQ(erf(x).eval(x0), std::erf(x0));
    EXPECT_DOUBLE_EQ(erf(x).derivative().eval(x0),
                     k2OverSqrtPi * std::exp(-x0 * x0));
  }
}
TEST(NewMathFunctions, ExprBinaryValue) {
  auto x = var<"x">;
  auto y = var<"y">;
  EXPECT_DOUBLE_EQ(hypot(x, y).eval(3.0, 4.0), 5.0);
  EXPECT_DOUBLE_EQ(pow(x, y).eval(3.0, 4.0), std::pow(3.0, 4.0));
  EXPECT_DOUBLE_EQ(atan2(y, x).eval(3.0, 4.0), std::atan2(4.0, 3.0));
  EXPECT_DOUBLE_EQ(max(x, y).eval(3.0, 4.0), 4.0);
  EXPECT_DOUBLE_EQ(min(x, y).eval(3.0, 4.0), 3.0);
  EXPECT_DOUBLE_EQ(hypot(x, 4.0).eval(3.0), 5.0);
  EXPECT_DOUBLE_EQ(pow(x, 2.0).eval(3.0), 9.0);
  EXPECT_DOUBLE_EQ(pow(2.0, x).eval(3.0), 8.0);
}
TEST(NewMathFunctions, ReverseUnary) {
  EXPECT_DOUBLE_EQ(Equation{log10(var<"x">)}.jacobian(3.0)[0],
                   1.0 / (3.0 * kLn10));
  EXPECT_DOUBLE_EQ(Equation{cbrt(var<"x">)}.jacobian(2.0)[0],
                   1.0 / (3.0 * std::cbrt(2.0) * std::cbrt(2.0)));
  EXPECT_DOUBLE_EQ(Equation{asinh(var<"x">)}.jacobian(0.7)[0],
                   1.0 / std::sqrt(0.49 + 1.0));
  EXPECT_DOUBLE_EQ(Equation{acosh(var<"x">)}.jacobian(2.0)[0],
                   1.0 / std::sqrt(4.0 - 1.0));
  EXPECT_DOUBLE_EQ(Equation{atanh(var<"x">)}.jacobian(0.3)[0],
                   1.0 / (1.0 - 0.09));
  EXPECT_DOUBLE_EQ(Equation{erf(var<"x">)}.jacobian(0.5)[0],
                   k2OverSqrtPi * std::exp(-0.25));
}
TEST(NewMathFunctions, ReverseBinaryPartials) {
  {
    auto x = var<"x">;
    auto y = var<"y">;
    auto g = Equation{hypot(x, y)}.jacobian(3.0, 4.0);
    EXPECT_DOUBLE_EQ(g[0], 3.0 / 5.0);
    EXPECT_DOUBLE_EQ(g[1], 4.0 / 5.0);
  }
  {
    auto x = var<"x">;
    auto y = var<"y">;
    auto g = Equation{pow(x, y)}.jacobian(2.0, 3.0);
    EXPECT_DOUBLE_EQ(g[0], 3.0 * std::pow(2.0, 2.0));
    EXPECT_DOUBLE_EQ(g[1], std::pow(2.0, 3.0) * std::log(2.0));
  }
  {
    // atan2(y, x) takes the numerator first, and the Jacobian is ordered by
    // symbol, so x is at index 0.
    auto y = var<"y">;
    auto x = var<"x">;
    auto g = Equation{atan2(y, x)}.jacobian(2.0, 1.0);
    const double q = 1.0 + 4.0;
    EXPECT_DOUBLE_EQ(g[0], -1.0 / q);
    EXPECT_DOUBLE_EQ(g[1], 2.0 / q);
  }
}
TEST(NewMathFunctions, ReverseMaxMinSelectsBranch) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto gmax = Equation{max(x, y)}.jacobian(3.0, 4.0);
  EXPECT_DOUBLE_EQ(gmax[0], 0.0);
  EXPECT_DOUBLE_EQ(gmax[1], 1.0);
  auto gmin = Equation{min(x, y)}.jacobian(3.0, 4.0);
  EXPECT_DOUBLE_EQ(gmin[0], 1.0);
  EXPECT_DOUBLE_EQ(gmin[1], 0.0);
}

TEST(ValueMapTest, SlotsAreInCanonicalOrderNotArgumentOrder) {
  // Canonical order is alphabetical, so the slots are {w, x, y} in any case.
  const auto m = values(named<"y">(3.0), named<"w">(1.0), named<"x">(2.0));
  EXPECT_EQ(decltype(m)::arity, 3u);
  EXPECT_DOUBLE_EQ(m.slots[0], 1.0);
  EXPECT_DOUBLE_EQ(m.slots[1], 2.0);
  EXPECT_DOUBLE_EQ(m.slots[2], 3.0);
  EXPECT_DOUBLE_EQ(m.get<"w">(), 1.0);
  EXPECT_DOUBLE_EQ(m.get<"x">(), 2.0);
  EXPECT_DOUBLE_EQ(m.get<"y">(), 3.0);
}
TEST(ValueMapTest, SetUpdatesTheNamedSlot) {
  auto m = values(named<"x">(2.0), named<"y">(3.0));
  m.set<"x">(7.0);
  EXPECT_DOUBLE_EQ(m.get<"x">(), 7.0);
  EXPECT_DOUBLE_EQ(m.get<"y">(), 3.0);
}
TEST(ValueMapTest, SubscriptReadsTheSameSlotAsGet) {
  const auto m = values(named<"y">(3.0), named<"w">(1.0), named<"x">(2.0));
  // Both spellings of the key are checked, since "x"_s expands to sym<"x">.
  EXPECT_DOUBLE_EQ(m[sym<"w">], m.get<"w">());
  EXPECT_DOUBLE_EQ(m["x"_s], 2.0);
  EXPECT_DOUBLE_EQ(m["y"_s], 3.0);
  static_assert(
      std::is_same_v<decltype(sym<"x">), const symbol_type<FixedString{"x"}>>,
      R"("x"_s and sym<"x"> must name the same key type)");
}
TEST(ValueMapTest, SubscriptAssignsIntoTheNamedSlot) {
  auto m = values(named<"x">(2.0), named<"y">(3.0));
  m["x"_s] = 7.0;
  EXPECT_DOUBLE_EQ(m.get<"x">(), 7.0);
  EXPECT_DOUBLE_EQ(m.get<"y">(), 3.0);
  m[sym<"y">] += 1.0;
  EXPECT_DOUBLE_EQ(m.slots[1], 4.0);
  EXPECT_DOUBLE_EQ(m.slots[0], 7.0);
}
TEST(ValueMapTest, SubscriptOwnershipMatchesGet) {
  const auto m = values(named<"x">(2.0), named<"y">(3.0));
  auto mut = m;
  static_assert(std::is_same_v<decltype(m["x"_s]), const double &>,
                "subscript on a const lvalue map borrows");
  static_assert(std::is_same_v<decltype(mut["x"_s]), double &>,
                "subscript on a mutable lvalue map hands out the slot");
  static_assert(
      std::is_same_v<decltype(values(named<"x">(2.0))["x"_s]), double>,
      "subscript on a temporary map must return by value");
  EXPECT_DOUBLE_EQ(values(named<"x">(2.0))["x"_s], 2.0);

  // get<S>() reaches the same slot with the same ownership.
  static_assert(std::is_same_v<decltype(m.get<"x">()), const double &>);
  static_assert(std::is_same_v<decltype(mut.get<"x">()), double &>);
  static_assert(
      std::is_same_v<decltype(values(named<"x">(2.0)).get<"x">()), double>,
      "get on a temporary map must return by value");
  EXPECT_DOUBLE_EQ(values(named<"x">(2.0)).get<"x">(), 2.0);
}
TEST(ValueMapTest, SubscriptIsConstexpr) {
  constexpr double folded = [] {
    auto m = values(named<"x">(2.0), named<"y">(3.0));
    m["x"_s] = 7.0;
    return m["x"_s] + m["y"_s];
  }();
  static_assert(folded == 10.0, "read and write both fold at compile time");
  EXPECT_DOUBLE_EQ(folded, 10.0);
}
TEST(NamedEvalTest, MatchesPositionalBitForBit) {
  // Named and positional evaluation differ only in storage, so they agree
  // exactly.
  auto x = var<"x">;
  auto y = var<"y">;
  const auto e = sin(x * y) * exp(x + y) + log(y * y + 1.0) / (x + 2.0) -
                 hypot(x, y) * tanh(x);

  EXPECT_EQ(eval(e, named<"x">(0.8), named<"y">(1.3)), e.eval(0.8, 1.3));
}
TEST(NamedEvalTest, ArgumentOrderIsIrrelevantAndSupersetsAreAllowed) {
  auto x = var<"x">;
  auto y = var<"y">;
  const auto e = (x + y) * x;

  EXPECT_DOUBLE_EQ(eval(e, named<"x">(2.0), named<"y">(3.0)), 10.0);
  EXPECT_DOUBLE_EQ(eval(e, named<"y">(3.0), named<"x">(2.0)), 10.0);

  // Extra symbols are skipped, since only the ones in use are looked up.
  const auto wide = values(named<"z">(9.0), named<"x">(2.0), named<"y">(3.0),
                           named<"a">(9.0));
  EXPECT_DOUBLE_EQ(eval(e, wide), 10.0);
}
TEST(NamedEvalTest, IsConstexpr) {
  constexpr auto x = var<"x">;
  constexpr auto y = var<"y">;
  static_assert(eval(x * x + y, named<"x">(3.0), named<"y">(4.0)) == 13.0,
                "x*x + y at (3,4)");
  SUCCEED();
}
TEST(PositionalEvalTest, SymbolOrderIsAlphabeticalNotSourceOrder) {
  auto y = var<"y">;
  auto x = var<"x">;
  const auto e = y * x;
  constexpr auto order = symbol_order<decltype(e)>();
  static_assert(order.size() == 2);
  EXPECT_EQ(order[0], "x");
  EXPECT_EQ(order[1], "y");
}
TEST(PositionalEvalTest, VariadicAndRangeAgreeWithNamed) {
  auto x = var<"x">;
  auto y = var<"y">;
  const auto e = (x + y) * x;

  const double expected = 10.0;
  EXPECT_DOUBLE_EQ(eval(e, 2.0, 3.0), expected);
  EXPECT_DOUBLE_EQ(eval(e, std::array{2.0, 3.0}), expected);
  EXPECT_DOUBLE_EQ(*eval(e, std::vector{2.0, 3.0}), expected);
  EXPECT_DOUBLE_EQ(eval(e, named<"x">(2.0), named<"y">(3.0)), expected);
}
TEST(PositionalEvalTest, AcceptsALazyNonSizedInputRange) {
  auto x = var<"x">;
  auto y = var<"y">;
  const auto e = (x + y) * x;

  // A single-pass view yielding 2.0 and 3.0.
  auto lazy = std::views::iota(2, 4) |
              std::views::transform([](int i) { return double(i); });
  EXPECT_DOUBLE_EQ(*eval(e, lazy), 10.0);
}
// A range's length is not in its type, so a short range is a runtime error,
// while the array spelling stays a static_assert.
TEST(PositionalEvalTest, AShortRangeIsAnError) {
  auto x = var<"x">;
  auto y = var<"y">;
  const auto e = (x + y) * x;
  const std::vector<double> too_few{2.0};
  const auto v = eval(e, too_few);
  EXPECT_FALSE(v.has_value());
  EXPECT_EQ(v.error().code, ddx::errc::short_point);
}
TEST(PositionalEvalTest, IsConstexpr) {
  constexpr auto x = var<"x">;
  constexpr auto y = var<"y">;
  constexpr auto e = (x + y) * x;
  static_assert(eval(e, 2.0, 3.0) == 10.0, "variadic positional eval");
  static_assert(eval(e, std::array{2.0, 3.0}) == 10.0, "range positional eval");
  EXPECT_DOUBLE_EQ(eval(e, 2.0, 3.0), 10.0);
}

// A long range is as wrong as a short one, because the extra values name no
// symbol.
TEST(PositionalEvalTest, ALongRangeIsAnError) {
  auto x = var<"x">;
  auto y = var<"y">;
  const auto e = (x + y) * x;
  const std::vector<double> too_many{2.0, 3.0, 4.0};
  const auto v = eval(e, too_many);
  ASSERT_FALSE(v.has_value());
  EXPECT_EQ(v.error().code, ddx::errc::wrong_arity);
  const std::vector<double> exact{2.0, 3.0};
  EXPECT_DOUBLE_EQ(*eval(e, exact), 10.0);
}
