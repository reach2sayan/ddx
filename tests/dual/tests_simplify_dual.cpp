#include "dual/tests_dual_common.hpp"
#include "tests_simplify_fixtures.hpp"

TEST(Simplify, DualLiteralPairsStayEmpty) {
  // Differentiation only makes 0 and 1, and those dual values fold back into
  // the type through the int spelling, so the spine stays empty.
  using D = ddx::impl::Dual<double>;
  static_assert(std::is_empty_v<decltype(ddx::impl::Lit<D, 0>{} +
                                         ddx::impl::Lit<D, 1>{})>);
  static_assert(std::is_empty_v<decltype(ddx::impl::Lit<D, 0>{} *
                                         ddx::impl::Lit<D, 1>{})>);
  static_assert(std::is_empty_v<decltype(ddx::impl::Lit<D, 1>{} +
                                         ddx::impl::Lit<D, 0>{})>);
  static_assert(std::is_empty_v<decltype(ddx::impl::Lit<double, 0>{} +
                                         ddx::impl::Lit<double, 1>{})>);
  // Any other value has to live in a stored Constant.
  EXPECT_DOUBLE_EQ(
      (ddx::impl::Lit<D, 1>{} + ddx::impl::Lit<D, 1>{}).get().value(), 2.0);
}
TEST(Simplify, DualValuedTreesAreStatelessToo) {
  // The int spelling works even for a T that cannot be an NTTP, so the
  // dual-valued path never falls back to a stored Constant.
  using D = ddx::impl::Dual<double>;
  ddx::impl::Variable<D, ddx::impl::FixedString{"x"}> dx;
  ddx::impl::Variable<D, ddx::impl::FixedString{"y"}> dy;
  static_assert(std::is_empty_v<decltype(dx * dy)>);
  static_assert(std::is_empty_v<decltype((dx * dy).derivative())>);
  static_assert(sizeof(dx * dy) == 1);
}
TEST(Concepts, NumericDemandsWhatTheSweepsActuallyUse) {
  // Differentiation makes 0 and 1, so the concept rejects a type that cannot
  // spell 1 instead of failing deep inside a sweep.
  static_assert(std::default_initializable<NoIdentity>);
  static_assert(!std::constructible_from<NoIdentity, int>);
  static_assert(!ddx::impl::Numeric<NoIdentity>);

  // Only abs, min and max compare their operands, so only they require an
  // ordering.
  static_assert(ddx::impl::Numeric<Undeclared>);
  static_assert(!std::totally_ordered<Undeclared>);
  static_assert(!OpAccepts<ddx::impl::AbsOp, Undeclared>);
  static_assert(!OpAccepts<ddx::impl::MaxOp, Undeclared>);
  static_assert(!OpAccepts<ddx::impl::MinOp, Undeclared>);
  static_assert(OpAccepts<ddx::impl::MultiplyOp, Undeclared>);
  static_assert(OpAccepts<ddx::impl::SumOp, Undeclared>);
  // Ordered scalars are accepted by all five ops.
  static_assert(OpAccepts<ddx::impl::AbsOp, double> &&
                OpAccepts<ddx::impl::MinOp, double>);

  // The shipped scalars satisfy both constraints, so neither narrows the
  // library's surface.
  static_assert(ddx::impl::Numeric<double> && std::totally_ordered<double>);
  static_assert(std::totally_ordered<ddx::impl::Dual<double>>);
  static_assert(std::totally_ordered<ddx::impl::TaylorDual<double, 3>>);
  static_assert(std::totally_ordered<ddx::impl::TaylorDual<double, 3>>);
}
TEST(Simplify, MultiplicationCommutesOnlyWhenTheScalarSaysSo) {
  // A type that says nothing is treated as non-commuting, which can cost a
  // shared subexpression but never gives a wrong result.
  static_assert(ddx::impl::Numeric<Undeclared> &&
                !ddx::impl::CCommutativeMultiply<Undeclared>);
  static_assert(ddx::impl::Numeric<Declared>);
  static_assert(ddx::impl::CCommutativeMultiply<double>);
  static_assert(ddx::impl::CCommutativeMultiply<ddx::impl::Dual<double>>);
  static_assert(
      ddx::impl::CCommutativeMultiply<ddx::impl::TaylorDual<double, 3>>);
  static_assert(
      ddx::impl::CCommutativeMultiply<ddx::impl::TaylorDual<double, 3>>);
  // The dual wrappers defer to the scalar underneath.
  static_assert(!ddx::impl::CCommutativeMultiply<ddx::impl::Dual<Undeclared>>);
  // DDX_COMMUTATIVE_MULTIPLY declares it for a concrete user type.
  static_assert(ddx::impl::CCommutativeMultiply<Declared>);

  // Canonicalisation reorders the same shape for one scalar and leaves it
  // alone for the other.
  ddx::impl::Variable<Undeclared, ddx::impl::FixedString{"x"}> ux;
  ddx::impl::Variable<Undeclared, ddx::impl::FixedString{"y"}> uy;
  static_assert(!std::is_same_v<decltype(canonicalise(ux * uy)),
                                decltype(canonicalise(uy * ux))>);
  // Addition needs no opt-in, because a ring's addition always commutes.
  static_assert(std::is_same_v<decltype(canonicalise(ux + uy)),
                               decltype(canonicalise(uy + ux))>);
}
