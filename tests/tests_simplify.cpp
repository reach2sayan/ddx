#include "tests_simplify_fixtures.hpp"

// Evaluate a tuple of expressions at one point, in canonical symbol order.
template <ddx::impl::CSymbolList Syms>
constexpr auto eval_all(const auto &vals, const auto &...es) noexcept {
  return std::array{es.template eval_seeded<Syms>(vals)...};
}

TEST(Simplify, IdentitiesCollapseDerivativeTrees) {
  // Without folding, d(x*y)/dx would be 1*y + x*0, seven nodes for one lookup.
  static_assert(d_nodes<ddx::impl::FixedString{"x"}, decltype(sx * sy)>() == 1);

  // Without folding, the F4 Jacobian would be 268 nodes across four rows.
  constexpr std::size_t rows =
      d_nodes<ddx::impl::FixedString{"w"}, decltype(F4)>() +
      d_nodes<ddx::impl::FixedString{"x"}, decltype(F4)>() +
      d_nodes<ddx::impl::FixedString{"y"}, decltype(F4)>() +
      d_nodes<ddx::impl::FixedString{"z"}, decltype(F4)>();
  static_assert(rows == 68);
  static_assert(ddx::impl::node_count_v<decltype(F4)> == 26);
}
TEST(Simplify, OnlyCompileTimeLiteralsFold) {
  // Lit carries its value in the type, so it folds; a stored Constant does not.
  static_assert(ddx::impl::detail::is_zero_v<ddx::impl::Lit<double, 0>>);
  static_assert(ddx::impl::detail::is_zero_v<ddx::impl::Lit<double, 0.0>>);
  static_assert(ddx::impl::detail::is_one_v<ddx::impl::Lit<double, 1>>);
  static_assert(!ddx::impl::detail::is_zero_v<ddx::impl::Constant<double>>);
  static_assert(!ddx::impl::detail::is_one_v<ddx::impl::Constant<double>>);

  using X = std::remove_cvref_t<decltype(sx)>;
  auto folded = sx * ddx::impl::Lit<double, 1>{};
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(folded)>, X>);
  auto unfolded = sx * ddx::impl::Constant<double>{1.0};
  static_assert(!std::is_same_v<std::remove_cvref_t<decltype(unfolded)>, X>);
  EXPECT_DOUBLE_EQ(unfolded.eval(3.0), 3.0);
}
TEST(Simplify, StoredLiteralsNeverMakeTwoTreesTheSame) {
  // Every stored literal has the one type Lit<double>, so x*3.0 and x*5.0 share
  // a type; only a tree without literals may treat type identity as equality.
  using One = ddx::impl::Lit<double, 1>;
  static_assert(!std::is_same_v<decltype((sx * 3.0) / (sx * 5.0)), One>);
  static_assert(!std::is_same_v<decltype((sx + 3.0) / (sx + 5.0)), One>);
  using Y = std::remove_cvref_t<decltype(sy)>;
  static_assert(!std::is_same_v<decltype((sy / (sx + 2.0)) * (sx + 7.0)), Y>);
  static_assert(!std::is_same_v<decltype((sx + 7.0) * (sy / (sx + 2.0))), Y>);
  EXPECT_DOUBLE_EQ(((sx * 3.0) / (sx * 5.0)).eval(2.0), 0.6);
  EXPECT_DOUBLE_EQ(((sx + 3.0) / (sx + 5.0)).eval(1.0), 4.0 / 6.0);
  EXPECT_DOUBLE_EQ(((sy / (sx + 2.0)) * (sx + 7.0)).eval(1.0, 1.0), 8.0 / 3.0);
  // A tree without literals is fully described by its type, so it cancels.
  static_assert(std::is_same_v<decltype((sx * sy) / (sx * sy)), One>);
  using X = std::remove_cvref_t<decltype(sx)>;
  static_assert(std::is_same_v<decltype((sx / sy) * sy), X>);
}
TEST(Simplify, MaxMinDerivativeSizeIsOnTheLedger) {
  // The max/min derivative keeps a whole a-b subtree inside its sign node, and
  // that duplication is structural, so it shows in the node count.
  constexpr auto dmax =
      d_nodes<ddx::impl::FixedString{"x"}, decltype(max(sx, sy))>();
  constexpr auto dmin =
      d_nodes<ddx::impl::FixedString{"x"}, decltype(min(sx, sy))>();
  static_assert(dmax == 9);
  static_assert(dmin == 10);
}
TEST(Simplify, CommutativeOperandsCanonicalise) {
  // x+y and y+x become the same type, so type identity works as value
  // numbering; double declares * commutative, so x*y unifies with y*x too.
  static_assert(std::is_same_v<decltype(canonicalise(sx + sy)),
                               decltype(canonicalise(sy + sx))>);
  static_assert(std::is_same_v<decltype(canonicalise(sx * sy)),
                               decltype(canonicalise(sy * sx))>);
  static_assert(std::is_same_v<decltype(canonicalise((sx * sy) * sz)),
                               decltype(canonicalise(sz * (sy * sx)))>);
  EXPECT_DOUBLE_EQ(canonicalise(sy * sx).eval(3.0, 5.0), 15.0);
  EXPECT_DOUBLE_EQ(canonicalise(sx - sy).eval(5.0, 2.0), 3.0);
  // Literals sort ahead of symbols, so a scalar always lands on the left.
  EXPECT_EQ(std::format("{}", canonicalise(sx * 2.0)), "2 * x");
  EXPECT_EQ(std::format("{}", canonicalise(2.0 * sx)), "2 * x");
}
TEST(ReverseMode, MultiplyAdjointRespectsOperandSide) {
  // For c = a*b the adjoint reaching b is a*adj, multiplied on the right of a,
  // and not adj*a.
  static_assert(ddx::impl::Numeric<Mat2> &&
                !ddx::impl::CCommutativeMultiply<Mat2>);

  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"x"}> mx;
  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"y"}> my;
  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"z"}> mz;

  constexpr Mat2 X{1, 1, 0, 1}, Y{1, 2, 3, 4}, Z{1, 0, 1, 1};

  // In f = z*(x*y) the adjoint reaching the inner product is Z, not the
  // identity, so the two operand orders give different answers.
  const auto g = ddx::Equation{mz * (mx * my)}.jacobian(std::array{X, Y, Z});

  EXPECT_EQ(g[0], Z * Y);
  EXPECT_EQ(g[1], X * Z);
  EXPECT_EQ(g[2], X * Y);
  // Spelled out, so a failure prints the wrong matrix.
  EXPECT_EQ(g[1], (Mat2{2, 1, 1, 1}));
  EXPECT_NE(g[1], (Mat2{1, 1, 1, 2}));
}
TEST(ReverseMode, DivideAdjointRespectsOperandSide) {
  // DivideOp reads a/b as a*b^-1, so the b adjoint is -a*b^-1*adj*b^-1 and not
  // the commutative quotient rule's -adj*a/(b*b).
  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"x"}> mx;
  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"y"}> my;
  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"z"}> mz;

  constexpr Mat2 X{1, 1, 0, 1}, Y{1, 0, 1, 1}, Z{1, 2, 3, 4}; // det Y == 1

  const auto g = ddx::Equation{mz * (mx / my)}.jacobian(std::array{X, Y, Z});

  const Mat2 sided = -((X / Y) * Z) / Y;
  const Mat2 quotient = -(Z * X) / (Y * Y);
  // If the two rules ever agreed here, the checks below would prove nothing.
  ASSERT_NE(sided, quotient);

  EXPECT_EQ(g[0], Z / Y);
  EXPECT_EQ(g[1], sided);
  EXPECT_NE(g[1], quotient);
  EXPECT_EQ(g[2], X / Y);
}
TEST(Simplify, MaxAndMinHaveASymbolicDerivative) {
  // Selecting between lhs.derivative() and rhs.derivative() with a runtime
  // conditional cannot type-check: the two are different types.
  const auto dmax_dx =
      ddx::impl::make_all_constant_except<ddx::impl::FixedString{"x"}>(
          max(sx, sy))
          .derivative();
  const auto dmin_dx =
      ddx::impl::make_all_constant_except<ddx::impl::FixedString{"x"}>(
          min(sx, sy))
          .derivative();
  for (auto [a, b] : {std::pair{3.0, 1.0}, std::pair{1.0, 3.0}}) {
    const auto rmax = Equation{max(sx, sy)}.jacobian(std::array{a, b});
    const auto rmin = Equation{min(sx, sy)}.jacobian(std::array{a, b});
    EXPECT_DOUBLE_EQ(dmax_dx.eval(ddx::named<"x">(a), ddx::named<"y">(b)),
                     rmax[0]);
    EXPECT_DOUBLE_EQ(dmin_dx.eval(ddx::named<"x">(a), ddx::named<"y">(b)),
                     rmin[0]);
  }
}
TEST(Simplify, ReciprocalsCancelOnTheSideDivisionPutsThem) {
  // d(x*log x)/dx is born as log(x) + x*(1/x), and cancelling x*(1/x) costs
  // no accuracy.
  constexpr auto dxlogx =
      ddx::impl::make_all_constant_except<ddx::impl::FixedString{"x"}>(sx *
                                                                       log(sx))
          .derivative();
  static_assert(ddx::impl::node_count_v<decltype(dxlogx)> == 4);
  EXPECT_EQ(std::format("{}", canonicalise(dxlogx)), "1 + log(x)");
  EXPECT_DOUBLE_EQ(dxlogx.eval(2.0), std::log(2.0) + 1.0);

  // The numerator need not be a literal.
  static_assert(std::is_same_v<std::remove_cvref_t<decltype((sy / sx) * sx)>,
                               std::remove_cvref_t<decltype(sy)>>);

  // `/` is right division, so (n/a)*a cancels for any ring, while a*(n/a)
  // needs commutativity.
  ddx::impl::Variable<Mat2, ddx::impl::FixedString{"x"}> mx;
  ddx::impl::Variable<Mat2, ddx::impl::FixedString{"y"}> my;
  static_assert(std::is_same_v<std::remove_cvref_t<decltype((my / mx) * mx)>,
                               std::remove_cvref_t<decltype(my)>>);
  static_assert(ddx::impl::node_count_v<decltype(mx * (my / mx))> == 5);

  // For a commuting scalar both spellings fold, and x leaves the type entirely.
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(sx * (sy / sx))>,
                               std::remove_cvref_t<decltype(sy)>>);
  EXPECT_DOUBLE_EQ((sx * (sy / sx)).eval(12.0), 12.0);
}
// Jacobian rows hold every symbol but one, and that hold must come off the row
// so a second derivative is not zero; a hold the caller asked for stays.
TEST(Simplify, JacobianRowsDifferentiateAgain) {
  using ddx::impl::FixedString;
  using ddx::impl::Freeze;
  using ddx::impl::make_all_constant_except;
  using ddx::impl::thaw_partials;
  using Syms = ddx::meta::list<ddx::impl::symbol_type<FixedString{"x"}>,
                               ddx::impl::symbol_type<FixedString{"y"}>>;

  using Live = ddx::impl::Variable<double, FixedString{"y"}>;
  using Part = ddx::impl::Variable<double, FixedString{"y"}, Freeze::partial>;
  static_assert(
      std::is_same_v<std::remove_cvref_t<decltype(thaw_partials(Part{}))>,
                     Live>);

  // d2f/dxdy = 2x for f = x*x*y is non-zero only if the row came back thawed.
  constexpr auto row =
      ddx::impl::detail::make_derivatives(Syms{}, sx * sx * sy);
  constexpr auto d2 = canonicalise(
      thaw_partials(make_all_constant_except<FixedString{"y"}>(std::get<0>(row))
                        .derivative()));
  constexpr std::array at{3.0, 5.0};
  EXPECT_DOUBLE_EQ(eval_all<Syms>(at, std::get<0>(row))[0], 30.0);
  EXPECT_DOUBLE_EQ(eval_all<Syms>(at, d2)[0], 6.0);
}
