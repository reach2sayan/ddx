#include "tests_simplify_fixtures.hpp"

TEST(Simplify, IdentitiesCollapseDerivativeTrees) {
  // Unfolded, d(x*y)/dx is `1*y_c + x*0`: seven nodes and seven cache slots
  // for one symbol lookup.
  static_assert(d_nodes<ddx::impl::FixedString{"x"}, decltype(sx * sy)>() == 1);

  // Unfolded, the F4 Jacobian is 268 nodes across its four rows.
  constexpr std::size_t rows =
      d_nodes<ddx::impl::FixedString{"w"}, decltype(F4)>() +
      d_nodes<ddx::impl::FixedString{"x"}, decltype(F4)>() +
      d_nodes<ddx::impl::FixedString{"y"}, decltype(F4)>() +
      d_nodes<ddx::impl::FixedString{"z"}, decltype(F4)>();
  static_assert(rows == 68);
  static_assert(ddx::impl::node_count_v<decltype(F4)> ==
                26); // f itself is untouched
}
TEST(Simplify, OnlyCompileTimeLiteralsFold) {
  // Lit carries its value in the type, so it folds; a stored Constant does
  // not.
  static_assert(ddx::impl::detail::is_zero_v<ddx::impl::Lit<double, 0>>);
  static_assert(ddx::impl::detail::is_zero_v<ddx::impl::Lit<double, 0.0>>);
  static_assert(ddx::impl::detail::is_one_v<ddx::impl::Lit<double, 1>>);
  static_assert(!ddx::impl::detail::is_zero_v<ddx::impl::Constant<double>>);
  static_assert(!ddx::impl::detail::is_one_v<ddx::impl::Constant<double>>);

  using X = std::remove_cvref_t<decltype(sx)>;
  auto folded = sx * ddx::impl::Lit<double, 1>{}; // x*1 -> x, no node built
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(folded)>, X>);
  auto unfolded =
      sx * ddx::impl::Constant<double>{1.0}; // value unknown until run time
  static_assert(!std::is_same_v<std::remove_cvref_t<decltype(unfolded)>, X>);
  EXPECT_DOUBLE_EQ(unfolded.eval(3.0), 3.0);
}
TEST(Simplify, StoredLiteralsNeverMakeTwoTreesTheSame) {
  // Every stored literal is the one type Lit<double>, so x*3.0 and x*5.0 are
  // the same type while holding different numbers.  Same and AOverB may only
  // read type identity as structural identity for a stateless tree.
  using One = ddx::impl::Lit<double, 1>;
  static_assert(!std::is_same_v<decltype((sx * 3.0) / (sx * 5.0)), One>);
  static_assert(!std::is_same_v<decltype((sx + 3.0) / (sx + 5.0)), One>);
  using Y = std::remove_cvref_t<decltype(sy)>;
  static_assert(!std::is_same_v<decltype((sy / (sx + 2.0)) * (sx + 7.0)), Y>);
  static_assert(!std::is_same_v<decltype((sx + 7.0) * (sy / (sx + 2.0))), Y>);
  EXPECT_DOUBLE_EQ(((sx * 3.0) / (sx * 5.0)).eval(2.0), 0.6);
  EXPECT_DOUBLE_EQ(((sx + 3.0) / (sx + 5.0)).eval(1.0), 4.0 / 6.0);
  EXPECT_DOUBLE_EQ(((sy / (sx + 2.0)) * (sx + 7.0)).eval(1.0, 1.0), 8.0 / 3.0);
  // Stateless trees still cancel: no literal, so the type is the whole tree.
  static_assert(std::is_same_v<decltype((sx * sy) / (sx * sy)), One>);
  using X = std::remove_cvref_t<decltype(sx)>;
  static_assert(std::is_same_v<decltype((sx / sy) * sy), X>);
}
TEST(Simplify, MaxMinDerivativeSizeIsOnTheLedger) {
  // (a'+b'±sign(a-b)*(a'-b'))/2 keeps a whole a-b subtree inside the sign
  // node, and that duplication is structural, so it stays in the type.
  constexpr auto dmax =
      d_nodes<ddx::impl::FixedString{"x"}, decltype(max(sx, sy))>();
  constexpr auto dmin =
      d_nodes<ddx::impl::FixedString{"x"}, decltype(min(sx, sy))>();
  static_assert(dmax == 9);
  static_assert(dmin == 10);
}
TEST(Simplify, CommutativeOperandsCanonicalise) {
  // x+y and y+x become the same type, which is what makes type identity a
  // usable value numbering.  double declares * commutative, so x*y unifies
  // with y*x too.
  static_assert(std::is_same_v<decltype(canonicalise(sx + sy)),
                               decltype(canonicalise(sy + sx))>);
  static_assert(std::is_same_v<decltype(canonicalise(sx * sy)),
                               decltype(canonicalise(sy * sx))>);
  static_assert(std::is_same_v<decltype(canonicalise((sx * sy) * sz)),
                               decltype(canonicalise(sz * (sy * sx)))>);
  // Reordering is exact, so the values do not move.
  EXPECT_DOUBLE_EQ(canonicalise(sy * sx).eval(3.0, 5.0), 15.0);
  EXPECT_DOUBLE_EQ(canonicalise(sx - sy).eval(5.0, 2.0), 3.0);
  // Literals sort ahead of symbols, so a scalar always lands on the left.
  EXPECT_EQ(std::format("{}", canonicalise(sx * 2.0)), "2 * x");
  EXPECT_EQ(std::format("{}", canonicalise(2.0 * sx)), "2 * x");
}
TEST(ReverseMode, MultiplyAdjointRespectsOperandSide) {
  // For c = a*b the differential is da*b + a*db, so the adjoint reaching `b`
  // multiplies on the RIGHT of a: {adj*b, a*adj}, not {adj*b, adj*a}.
  static_assert(ddx::impl::Numeric<Mat2> &&
                !ddx::impl::CCommutativeMultiply<Mat2>);

  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"x"}> mx;
  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"y"}> my;
  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"z"}> mz;

  // X and Z do not commute: X*Z = {2,1,1,1} but Z*X = {1,1,1,2}.
  constexpr Mat2 X{1, 1, 0, 1}, Y{1, 2, 3, 4}, Z{1, 0, 1, 1};

  // f = z*(x*y).  Nothing reorders it -- Mat2 never opted in -- so the adjoint
  // arriving at the inner product is Z rather than the identity, which is what
  // makes the two sidings disagree.
  const auto g = ddx::Equation{mz * (mx * my)}.jacobian(std::array{X, Y, Z});

  // Partials come back in canonical symbol order: x, y, z.
  EXPECT_EQ(g[0], Z * Y); // dx = adj*y with adj = z
  EXPECT_EQ(g[1], X * Z); // dy = x*adj  <-- the sided one; the bug gives Z*X
  EXPECT_EQ(g[2], X * Y); // dz = adj*(x*y) with adj = I
  // Spelled out, so a failure reports the wrong matrix rather than a mismatch.
  EXPECT_EQ(g[1], (Mat2{2, 1, 1, 1}));
  EXPECT_NE(g[1], (Mat2{1, 1, 1, 2})); // what {adj*b, adj*a} would produce
}
TEST(ReverseMode, DivideAdjointRespectsOperandSide) {
  // DivideOp reads a/b as a*b^-1, so c = a*b^-1 differentiates as
  // da*b^-1 - a*b^-1*db*b^-1: the b-adjoint threads BETWEEN a/b and b^-1 and
  // does not collapse to the -adj*a/(b*b) of the quotient rule.
  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"x"}> mx;
  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"y"}> my;
  constexpr ddx::impl::Variable<Mat2, ddx::impl::FixedString{"z"}> mz;

  constexpr Mat2 X{1, 1, 0, 1}, Y{1, 0, 1, 1}, Z{1, 2, 3, 4}; // det Y == 1

  // f = z*(x/y), so the adjoint arriving at the quotient is Z, not the
  // identity -- which is what makes the two spellings disagree.
  const auto g = ddx::Equation{mz * (mx / my)}.jacobian(std::array{X, Y, Z});

  const Mat2 sided = -((X / Y) * Z) / Y;    // what the sided rule must produce
  const Mat2 quotient = -(Z * X) / (Y * Y); // what the commutative rule gives
  // Guard the guard: if these ever coincided the test below would be vacuous.
  ASSERT_NE(sided, quotient);

  EXPECT_EQ(g[0], Z / Y); // dx = adj*b^-1 with adj = z
  EXPECT_EQ(g[1], sided); // dy -- the whole point of the test
  EXPECT_NE(g[1], quotient);
  EXPECT_EQ(g[2], X / Y); // dz = adj*(x/y) with adj = I
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
  // d(x*log x)/dx is born as log(x) + x*(1/x): the chain rule multiplies log's
  // own 1/u straight back by u.  Cancelling costs no accuracy.
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

  // `/` is right division, so a^-1 only ever meets the operand on its right.
  // (n/a)*a cancels for any ring; a*(n/a) is a*n*a^-1 and needs commutativity.
  ddx::impl::Variable<Mat2, ddx::impl::FixedString{"x"}> mx;
  ddx::impl::Variable<Mat2, ddx::impl::FixedString{"y"}> my;
  static_assert(std::is_same_v<std::remove_cvref_t<decltype((my / mx) * mx)>,
                               std::remove_cvref_t<decltype(my)>>);
  static_assert(ddx::impl::node_count_v<decltype(mx * (my / mx))> == 5);

  // For a scalar that commutes both spellings fold, and what is left is a
  // one-symbol tree: x is gone from the signature, not just from the maths.
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(sx * (sy / sx))>,
                               std::remove_cvref_t<decltype(sy)>>);
  EXPECT_DOUBLE_EQ((sx * (sy / sx)).eval(12.0), 12.0);
}
// Two reasons a symbol differentiates to zero, and only one of them lifts.
// The Jacobian machinery holds every symbol but one to make a column; that
// hold has to come off the stored row, or differentiating the row a second
// time -- the Hessian -- would see held symbols and answer zero.  A hold the
// caller asked for outlives every rewrite.
TEST(Simplify, JacobianRowsDifferentiateAgainButHeldSymbolsStayHeld) {
  using ddx::impl::Freeze;
  using ddx::impl::make_all_constant_except;
  using ddx::impl::make_const_variable;
  using ddx::impl::thaw_partials;
  using ddx::impl::FixedString;
  using Syms = ddx::impl::mp::mp_list<ddx::impl::symbol_type<FixedString{"x"}>,
                                ddx::impl::symbol_type<FixedString{"y"}>>;

  // Only the machinery's own hold is reversible; the caller's is not.
  using Live = ddx::impl::Variable<double, FixedString{"y"}>;
  using Held = ddx::impl::Variable<double, FixedString{"y"}, Freeze::held>;
  using Part = ddx::impl::Variable<double, FixedString{"y"}, Freeze::partial>;
  static_assert(
      std::is_same_v<std::remove_cvref_t<decltype(thaw_partials(Part{}))>,
                     Live>);
  static_assert(
      std::is_same_v<std::remove_cvref_t<decltype(thaw_partials(Held{}))>,
                     Held>);

  // f = x*x*y: df/dx = 2xy, and d2f/dxdy = 2x is only non-zero if the row
  // came back thawed.
  constexpr auto row =
      ddx::impl::detail::make_derivatives(Syms{}, sx * sx * sy);
  constexpr auto d2 = canonicalise(thaw_partials(
      make_all_constant_except<FixedString{"y"}>(std::get<0>(row))
          .derivative()));
  constexpr std::array at{3.0, 5.0};
  EXPECT_DOUBLE_EQ(ddx::impl::detail::eval_all<Syms>(at, std::get<0>(row))[0],
                   30.0);
  EXPECT_DOUBLE_EQ(ddx::impl::detail::eval_all<Syms>(at, d2)[0], 6.0);

  // The same row, but with y held by the caller before the Jacobian was
  // taken: d/dy is zero and stays zero through the second derivative.
  constexpr auto held_row = ddx::impl::detail::make_derivatives(
      Syms{}, make_const_variable<FixedString{"y"}>(sx * sx * sy));
  constexpr auto held_d2 = canonicalise(thaw_partials(
      make_all_constant_except<FixedString{"y"}>(std::get<0>(held_row))
          .derivative()));
  EXPECT_DOUBLE_EQ(
      ddx::impl::detail::eval_all<Syms>(at, std::get<0>(held_row))[0], 30.0);
  EXPECT_DOUBLE_EQ(
      ddx::impl::detail::eval_all<Syms>(at, std::get<1>(held_row))[0], 0.0);
  EXPECT_DOUBLE_EQ(ddx::impl::detail::eval_all<Syms>(at, held_d2)[0], 0.0);
}
