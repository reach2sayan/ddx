#include "ddx.hpp"
#include "rt/archive/snapshot.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/equation.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

// to_graph lowers a compile-time expression into the graph, so the same
// function can be differentiated by Equation and by the sweep, and the two
// must agree.

namespace {

// A pointer per element, which is the shape every batch call takes its columns
// in.  Two spellings because the inputs are read and the outputs written, and
// one overload taking both would be ambiguous for a non-const vector.
[[nodiscard]] std::vector<double *> out_columns(std::vector<double> &values) {
  return values | std::views::transform([](double &v) { return &v; }) |
         ddx::impl::to<std::vector<double *>>();
}

[[nodiscard]] std::vector<const double *> in_columns(const auto &values) {
  return values | std::views::transform([](const double &v) { return &v; }) |
         ddx::impl::to<std::vector<const double *>>();
}

// An overflow is an answer, and the two engines have to reach the same one:
// EXPECT_NEAR cannot say that, since inf - inf is NaN.
void expect_close(double got, double want) {
  if (std::isfinite(want) && std::isfinite(got)) {
    EXPECT_NEAR(got, want, 1e-12 * std::max(1.0, std::abs(want)));
  } else {
    EXPECT_EQ(std::isnan(got), std::isnan(want));
    if (!std::isnan(want)) {
      EXPECT_EQ(got, want);
    }
  }
}

// Symbols are matched by name: Equation reports partials alphabetically, the
// builder numbers them in first-seen order.
template <ddx::impl::CExpression E, std::size_t N>
void expect_agrees_with_ddx(const E &e, std::array<double, N> pt) {
  ddx::rt::Builder<> b;
  const auto root = ddx::rt::to_graph(b, e);
  const auto g = ddx::rt::build_jacobian_impl(b, root.id(b));
  const auto values = ddx::rt::evaluate_all(b, std::span<const double>{pt});

  const double expected = std::apply(
      [&](auto... a) { return ddx::Equation{e}.evaluate(a...); }, pt);
  expect_close(values[g.value], expected);

  const auto expected_grad = std::apply(
      [&](auto... a) { return ddx::Equation{e}.jacobian(a...); }, pt);

  // Both sides number their symbols alphabetically, so slot i is partial i.
  for (std::size_t i = 0; i < b.symbols().size(); ++i) {
    SCOPED_TRACE("partial d/d" + std::string{b.symbols()[i]});
    expect_close(values[g.partial[i]], expected_grad[i]);
  }
}

// A namespace of its own so the runtime tests below can call their variables
// x, y and z too without hiding these.
namespace sym {

constexpr auto x = ddx::var<"x">;
constexpr auto y = ddx::var<"y">;
constexpr auto z = ddx::var<"z">;

TEST(RtJacobian, Arithmetic) {
  expect_agrees_with_ddx(x * y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x + y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x - y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x / y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(-x * y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x * x * x + y * y - x * y, std::array{1.3, 2.1});
}

template <typename E>
constexpr bool bridges = requires(ddx::rt::Builder<> &b, const E &e) {
  ddx::rt::to_graph(b, e);
};

// rt has no leaf that reads a symbol and differentiates to zero, so a frozen
// one is refused rather than lowered live.  A Jacobian row comes back thawed,
// and bridges like any other tree.
TEST(RtJacobian, BridgeRefusesFrozenLeaves) {
  using ddx::impl::FixedString;
  const auto held = ddx::impl::make_const_variable<FixedString{"y"}>(x * y);
  const auto column =
      ddx::impl::make_all_constant_except<FixedString{"x"}>(x * x * y)
          .derivative();
  const auto row = ddx::impl::thaw_partials(column);

  static_assert(!bridges<decltype(held)>);
  static_assert(!bridges<decltype(column)>);
  static_assert(bridges<decltype(row)>);
  expect_agrees_with_ddx(row, std::array{1.3, 2.1});
}

TEST(RtJacobian, BinaryFunctions) {
  expect_agrees_with_ddx(pow(x, y), std::array{1.7, 2.3});
  expect_agrees_with_ddx(atan2(x, y), std::array{1.3, 2.1});
  expect_agrees_with_ddx(hypot(x, y), std::array{1.3, 2.1});
}

TEST(RtJacobian, Transcendentals) {
  expect_agrees_with_ddx(exp(x) * sin(y), std::array{0.7, 1.1});
  expect_agrees_with_ddx(log(x) * sqrt(y), std::array{1.3, 2.1});
  expect_agrees_with_ddx(tanh(x) + cbrt(y), std::array{0.4, 2.1});
  expect_agrees_with_ddx(erf(x) * atanh(y), std::array{0.4, 0.3});
  expect_agrees_with_ddx(asin(x) + acos(y), std::array{0.4, 0.3});
  expect_agrees_with_ddx(acosh(x) + asinh(y), std::array{1.9, 0.6});
  expect_agrees_with_ddx(sinh(x) * cosh(y), std::array{0.4, 0.6});
  expect_agrees_with_ddx(tan(x) + log10(y), std::array{0.4, 2.6});
  expect_agrees_with_ddx(cos(x) / exp(y), std::array{0.4, 0.6});
}

TEST(RtJacobian, SharedAndNested) {
  expect_agrees_with_ddx((x * y) * (x * y) + sin(x * y), std::array{0.3, 0.7});
  expect_agrees_with_ddx(sin(cos(exp(x * y))), std::array{0.3, 0.7});
  expect_agrees_with_ddx(exp(x * y) + log(z) / tanh(x),
                         std::array{0.3, 0.7, 1.4});
}

// max, min and abs have no entry in the compile-time derivative table, so they
// are checked against central differences on the interpreter instead.
TEST(RtJacobian, SelectingOpsAgainstFiniteDifferences) {
  using ddx::rt::Builder;
  using ddx::rt::RTExpression;
  const auto check = [](auto build, std::array<double, 2> pt) {
    Builder<> b;
    const auto vx = var(b, "x");
    const auto vy = var(b, "y");
    const auto f = build(vx, vy);
    const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
    const auto values = ddx::rt::evaluate_all(b, std::span<const double>{pt});
    for (std::size_t i = 0; i < 2; ++i) {
      const double h = 1e-6;
      auto hi = pt, lo = pt;
      hi[i] += h;
      lo[i] -= h;
      const double fd = (ddx::rt::evaluate(b, g.value, hi) -
                         ddx::rt::evaluate(b, g.value, lo)) /
                        (2 * h);
      EXPECT_NEAR(values[g.partial[i]], fd, 1e-5 * std::max(1.0, std::abs(fd)));
    }
  };
  check([](RTExpression<> a, RTExpression<> b2) { return max(a, b2); },
        {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return min(a, b2); },
        {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return abs(a * b2); },
        {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return abs(a * b2); },
        {-1.3, 2.1});
}

TEST(RtJacobian, DerivFromValueReusesThePrimalNode) {
  // exp's derivative is the primal, so the sweep must not build a second exp.
  ddx::rt::Builder<> b;
  const auto vx = var(b, "x");
  const auto f = exp(vx);
  const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
  EXPECT_EQ(g.partial[0], f.id(b));
}

// Every row of both tables must name a rule that instantiates at the number
// *and* at the graph.  That is the whole of ops/adjoints.hpp's contract: add an
// op without a shared rule and this stops compiling, which is what keeps a
// second spelling from being written for one of them.
#define DDX_TEST_UNARY_RULE(fn, Op, label, functor, Desc)                      \
  static_assert(requires(const double &a) {                                    \
    ddx::impl::detail::adjoints_of<ddx::impl::detail::Desc>(a, a);             \
  });                                                                          \
  static_assert(requires(const ddx::rt::RTExpression<> &a) {                   \
    ddx::impl::detail::adjoints_of<ddx::impl::detail::Desc>(a, a);             \
  });
DDX_RT_UNARY_TABLE(DDX_TEST_UNARY_RULE)
#undef DDX_TEST_UNARY_RULE

#define DDX_TEST_BINARY_RULE(fn, Op, label, functor, Desc)                     \
  static_assert(requires(const double &a) {                                    \
    ddx::impl::detail::adjoints_of<ddx::impl::detail::Desc>(a, a, a);          \
  });                                                                          \
  static_assert(requires(const ddx::rt::RTExpression<> &a) {                   \
    ddx::impl::detail::adjoints_of<ddx::impl::detail::Desc>(a, a, a);          \
  });
DDX_RT_BINARY_TABLE(DDX_TEST_BINARY_RULE)
#undef DDX_TEST_BINARY_RULE

// The comparison operators answer expressions, nothing boolean-testable, so
// neither concept holds: that is why ExtremumOp keeps a rule of its own, and
// why AbsOpFn's unqualified sign() reaches RTExpression's hidden friend rather
// than the one in ops/adjoints.hpp.  If either ever becomes true both facts go
// stale silently.
static_assert(!std::totally_ordered<ddx::rt::RTExpression<>>);
static_assert(!std::equality_comparable<ddx::rt::RTExpression<>>);

// A node stands for a double, so it answers the commutativity question the way
// a double does -- which is what picks DivideOpFn's branch.
static_assert(ddx::impl::CCommutativeMultiply<ddx::rt::RTExpression<>>);

TEST(RtJacobian, UnusedSymbolHasZeroPartial) {
  ddx::rt::Builder<> b;
  const auto vx = var(b, "x");
  const auto vy = var(b, "y");
  const auto f = vx * vx;
  const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
  ASSERT_EQ(g.partial.size(), 2u);
  EXPECT_TRUE(b.is_constant(g.partial[1], 0.0));
  (void)vy;
}

// Wider coverage of the ops that now read their rule from ops/adjoints.hpp,
// including the points the spellings were chosen for: hypot and atan2 are
// written to divide by the hypotenuse first so they survive where x*x + y*y
// overflows, and pow's left partial has to stay finite at a == 0.
TEST(RtJacobian, SharedRulesAgreeAcrossTheScaleRange) {
  for (const auto &[a, b] : std::vector<std::pair<double, double>>{
           {1.3, 2.1}, {1e200, 1e200}, {1e-200, 1e-200}, {3.0, 1e-8}}) {
    expect_agrees_with_ddx(hypot(x, y), std::array{a, b});
    expect_agrees_with_ddx(atan2(x, y), std::array{a, b});
    expect_agrees_with_ddx(x / y, std::array{a, b});
    expect_agrees_with_ddx(x * y, std::array{a, b});
    expect_agrees_with_ddx(x + y, std::array{a, b});
  }
  for (const auto &[a, b] :
       std::vector<std::pair<double, double>>{{1.7, 2.3}, {0.5, 3.0}}) {
    expect_agrees_with_ddx(pow(x, y), std::array{a, b});
  }
}

// neg had no cross-check against Equation at all; abs only had one against
// finite differences, which cannot see a sign convention that is wrong at 0.
TEST(RtJacobian, UnaryRulesAgree) {
  for (const double a : {1.3, -1.3, 1e-300, 4.0}) {
    expect_agrees_with_ddx(-x, std::array{a});
    expect_agrees_with_ddx(-(x * x), std::array{a});
  }
}

// build_symbolic_jacobian is a selectable public mode that nothing in the tree
// instantiated before this test: DiffMode::Symbolic carries a tangent up the
// graph rather than an adjoint down it, so it is the one caller that wants the
// bare partial.  It reads it from the same rules, seeded with a unit adjoint.
TEST(RtJacobian, ForwardModeAgreesWithReverse) {
  const auto check = [](auto build, std::array<double, 2> pt) {
    ddx::rt::Builder<> b;
    const auto vx = var(b, "x");
    const auto vy = var(b, "y");
    const auto f = build(vx, vy);
    const auto fwd =
        ddx::rt::build_jacobian_impl<ddx::DiffMode::Symbolic>(b, f.id(b));
    const auto rev =
        ddx::rt::build_jacobian_impl<ddx::DiffMode::Reverse>(b, f.id(b));
    const auto values = ddx::rt::evaluate_all(b, std::span<const double>{pt});
    for (std::size_t i = 0; i < 2; ++i) {
      const double want = values[rev.partial[i]];
      EXPECT_NEAR(values[fwd.partial[i]], want,
                  1e-12 * std::max(1.0, std::abs(want)))
          << "partial " << i;
    }
  };
  using ddx::rt::RTExpression;
  check([](RTExpression<> a, RTExpression<> b2) { return a * b2 + sin(a); },
        {0.3, 0.7});
  check([](RTExpression<> a, RTExpression<> b2) { return a / b2; }, {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return pow(a, b2); },
        {1.7, 2.3});
  check([](RTExpression<> a, RTExpression<> b2) { return hypot(a, b2); },
        {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return -(a * b2); },
        {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return max(a, b2); },
        {1.3, 2.1});
}

} // namespace sym
} // namespace

// A banded system, where the pattern is most of the point: residual i touches
// x[i-1], x[i] and x[i+1] and nothing else, so the Jacobian is tridiagonal and
// 16 of its 36 cells exist.  Everything the batch API hands a caller is sized
// by that, and the dense spelling has to put the other 20 back.
TEST(RtJacobian, ABandedSystemKeepsOnlyTheCellsThatExist) {
  static constexpr std::size_t n = 6;
  ddx::rt::Builder<> b;

  const auto name = [](std::size_t i) { return "x" + std::to_string(i); };
  std::vector<ddx::rt::RTExpression<double>> x;
  for (const std::size_t i : std::views::iota(0uz, n)) {
    x.push_back(var(b, name(i)));
  }
  // r_i = x_i^2 + x_{i-1} x_{i+1}, so the band is three wide and the ends are
  // two.  Squares rather than plain sums, or every partial would be constant
  // and the sweep would fold the whole row away.
  std::vector<ddx::rt::RTExpression<double>> rows;
  for (const std::size_t i : std::views::iota(0uz, n)) {
    auto r = x[i] * x[i];
    if (i > 0 && i + 1 < n) {
      r = r + x[i - 1] * x[i + 1];
    }
    rows.push_back(r);
  }

  const auto j = ddx::rt::build_jacobian_impl(b, [&] {
    std::vector<ddx::rt::NodeId> ids;
    for (auto &r : rows) {
      ids.push_back(r.id(b));
    }
    return ids;
  }());

  const auto &pattern = j.pattern;
  EXPECT_EQ(pattern.size(), n);
  EXPECT_EQ(pattern.columns, n);
  EXPECT_EQ(pattern.nonzeros(), j.partial.size());
  EXPECT_LT(pattern.nonzeros(), n * n) << "a dense pattern tests nothing here";

  // Exactly the band: |i - j| <= 1 for an interior row, the diagonal alone at
  // the ends, and no_column everywhere else.
  for (const std::size_t i : std::views::iota(0uz, n)) {
    for (const std::size_t k : std::views::iota(0uz, n)) {
      const bool interior = i > 0 && i + 1 < n;
      const bool banded = k == i || (interior && (k + 1 == i || k == i + 1));
      EXPECT_EQ(pattern.at(i, k).has_value(), banded)
          << "cell (" << i << ", " << k << ")";
    }
  }

  // A cell outside the pattern reads as the literal zero node, not as a hole.
  EXPECT_EQ(j.at(0, n - 1), j.zero);
  EXPECT_NE(j.at(0, 0), j.zero);
}

// The three ways to read one Jacobian -- dense, compressed-plus-pattern, and a
// sweep of the nodes -- are the same numbers.
TEST(RtJacobian, DenseCompressedAndTheNodesAllAgree) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto z = var(b, "z");
  // z appears in one row only, x in two, y in two: no row is dense and no
  // column is empty.
  const auto eq = ddx::rt::equation(x * log(x) + y, y * y + z, exp(x * z));

  const std::vector<double> at{0.6, 1.4, 0.9};
  const auto dense = *eq.jacobian(std::span<const double>{at});
  ASSERT_EQ(dense.size(), 3u * 3u);

  const ddx::rt::Sparsity &pattern = eq.jacobian_pattern()->get();
  ASSERT_EQ(pattern.nonzeros(), *eq.jacobian_columns());
  ASSERT_LT(pattern.nonzeros(), 9u);

  std::vector<double> cells(pattern.nonzeros());
  std::vector<double> values(3);
  const double *const columns[]{&at[0], &at[1], &at[2]};
  double *const vs[]{&values[0], &values[1], &values[2]};
  const auto ps = out_columns(cells);
  ASSERT_TRUE(eq.jacobian(columns, vs, ps, 1).has_value());

  for (const std::size_t i : std::views::iota(0uz, 3uz)) {
    for (const std::size_t k : std::views::iota(0uz, 3uz)) {
      const double want =
          pattern.at(i, k)
              .transform([&cells](std::size_t cell) { return cells[cell]; })
              .value_or(0.0);
      EXPECT_EQ(dense[i * 3 + k], want) << "cell (" << i << ", " << k << ")";
    }
  }
}

// A partial that *folds* to zero is as structural as one that never existed.
// In `a*b - a*b + c` the two adjoint contributions reaching `a*b` are 1 and -1,
// which constant-fold to zero before a node is built, so add_to refuses them
// and a and b get no column at all -- the pattern is exact here where a support
// walk would have called them live.
TEST(RtJacobian, AFoldedPartialIsAStructuralHole) {
  ddx::rt::Builder<> b;
  const auto a = var(b, "a");
  const auto bb = var(b, "b");
  const auto c = var(b, "c");
  const auto d = var(b, "d");
  const auto e = var(b, "e");
  const auto f = var(b, "f");
  (void)e;
  (void)f; // named, never used: their columns must not exist either
  const auto eq = ddx::rt::equation(a * bb - a * bb + c, d + d, c * 0.0 + 2.0);

  const ddx::rt::Sparsity &pattern = eq.jacobian_pattern()->get();
  EXPECT_EQ(pattern.size(), 3u);
  EXPECT_EQ(pattern.columns, 6u);
  // Row 0 keeps c alone, row 1 keeps d alone, row 2 is a constant.
  EXPECT_EQ(pattern.nonzeros(), 2u);
  const auto widths =
      pattern |
      std::views::transform([](const auto &row) { return row.size(); }) |
      ddx::impl::to<std::vector<std::size_t>>();
  EXPECT_EQ(widths, (std::vector<std::size_t>{1, 1, 0}));
  EXPECT_TRUE(pattern.at(0, 2).has_value());  // c
  EXPECT_FALSE(pattern.at(0, 0).has_value()); // a
  EXPECT_FALSE(pattern.at(0, 1).has_value()); // b

  // And the dense spelling still answers for every cell, with zeros.
  const std::vector<double> at{0.3, 0.7, 1.1, 0.5, 0.2, 0.9};
  const auto dense = *eq.jacobian(std::span<const double>{at});
  ASSERT_EQ(dense.size(), 18u);
  EXPECT_EQ(dense[0 * 6 + 2], 1.0); // d/dc of (... + c)
  EXPECT_EQ(dense[1 * 6 + 3], 2.0); // d/dd of (d + d)
  for (const std::size_t k : std::views::iota(0uz, 6uz)) {
    EXPECT_EQ(dense[2 * 6 + k], 0.0) << "row 2 is constant, column " << k;
  }
  EXPECT_EQ(dense[0 * 6 + 0], 0.0);
  EXPECT_EQ(dense[0 * 6 + 1], 0.0);
}

// An arrow Hessian: x0 couples with every other symbol and nothing else
// couples, so 16 of 36 cells exist.  hessian_cell is the only way a batch
// caller can place a column of the compressed block it is handed.
TEST(RtJacobian, HessianCellPlacesEveryCompressedColumn) {
  static constexpr std::size_t n = 6;
  ddx::rt::Builder<> b;
  std::vector<ddx::rt::RTExpression<double>> x;
  for (const std::size_t i : std::views::iota(0uz, n)) {
    x.push_back(var(b, "x" + std::to_string(i)));
  }
  auto energy = x[0] * x[0];
  for (const std::size_t i : std::views::iota(1uz, n)) {
    energy = energy + x[i] * x[i] + x[0] * x[i];
  }
  const auto eq = ddx::rt::equation(energy);

  const std::vector<double> at{0.3, 0.7, 1.1, 0.5, 0.2, 0.9};
  const auto dense = *eq.hessian(std::span<const double>{at});
  ASSERT_EQ(dense.size(), n * n);

  std::vector<double> cells(*eq.hessian_columns());
  std::vector<double> values(1);
  std::vector<double> partials(*eq.jacobian_columns());
  ASSERT_LT(cells.size(), n * n) << "a dense Hessian places nothing";

  const auto columns = in_columns(at);
  double *const vs[]{&values[0]};
  const auto ps = out_columns(partials);
  const auto hs = out_columns(cells);
  ASSERT_TRUE(eq.hessian(columns, vs, ps, hs, 1).has_value());

  std::size_t placed = 0;
  for (const std::size_t i : std::views::iota(0uz, n)) {
    for (const std::size_t k : std::views::iota(0uz, n)) {
      const auto cell = eq.hessian_cell(i, k);
      if (cell) {
        ASSERT_LT(*cell, cells.size());
        EXPECT_EQ(dense[i * n + k], cells[*cell])
            << "cell (" << i << ", " << k << ")";
        ++placed;
      } else {
        EXPECT_EQ(dense[i * n + k], 0.0)
            << "unplaced cell (" << i << ", " << k << ") is not zero";
      }
    }
  }
  EXPECT_GT(placed, 0u);
  EXPECT_LT(placed, n * n) << "an arrow leaves cells unplaced";
}

// --- select ------------------------------------------------------------------

// Both arms are evaluated and the condition picks one, so the derivative is the
// arm's derivative and the condition contributes nothing.  Checked either side
// of the switch, where a rule that leaked the other arm would show.
TEST(RtJacobian, SelectDifferentiatesTheArmTheConditionTakes) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return select(lt(x, y), x * x, y * y * y);
  });

  for (const auto &[px, py] : {std::pair{1.0, 3.0}, std::pair{3.0, 1.0}}) {
    const std::array<double, 2> at{px, py};
    const bool takes_x = px < py;
    EXPECT_DOUBLE_EQ(*eq.evaluate(std::span<const double>{at}),
                     takes_x ? px * px : py * py * py);

    const auto g = *eq.jacobian(std::span<const double>{at});
    EXPECT_DOUBLE_EQ(g[0], takes_x ? 2 * px : 0.0) << "d/dx at " << px;
    EXPECT_DOUBLE_EQ(g[1], takes_x ? 0.0 : 3 * py * py) << "d/dy at " << px;

    // Second order too: the condition still contributes nothing.
    const auto h = *eq.hessian(std::span<const double>{at});
    EXPECT_DOUBLE_EQ(h[0], takes_x ? 2.0 : 0.0);
    EXPECT_DOUBLE_EQ(h[3], takes_x ? 0.0 : 6 * py);
  }
}

// A condition is any nonzero value, as in C -- there is no separate boolean.
TEST(RtJacobian, SelectReadsAnyNonzeroConditionAsTrue) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return select(x, ddx::rt::RTExpression<double>{7.0},
                  ddx::rt::RTExpression<double>{9.0});
  });
  const std::array<double, 1> zero{0.0};
  const std::array<double, 1> two{2.0};
  const std::array<double, 1> negative{-3.0};
  EXPECT_DOUBLE_EQ(*eq.evaluate(std::span<const double>{zero}), 9.0);
  EXPECT_DOUBLE_EQ(*eq.evaluate(std::span<const double>{two}), 7.0);
  EXPECT_DOUBLE_EQ(*eq.evaluate(std::span<const double>{negative}), 7.0);
}

// gt/ge/eq/ne are lt and le read the other way round, so there are two
// comparison opcodes and not six.  eq is `a <= b && b <= a`, which answers 0 at
// a NaN operand where `1 - lt - lt` would have called two NaNs equal.
TEST(RtJacobian, ComparisonsComposeFromTheTwoOpcodes) {
  // Equation is move-only, so each spelling is built and read where it is made
  // rather than handed back out of a helper.
  const auto value = [](auto build, double a, double b) {
    const auto eq = ddx::rt::equation(
        [&build] { return build(ddx::rt::var("x"), ddx::rt::var("y")); });
    const std::array<double, 2> at{a, b};
    return *eq.evaluate(std::span<const double>{at});
  };
  const auto less = [](auto x, auto y) { return x < y; };
  const auto greater = [](auto x, auto y) { return x > y; };
  const auto at_least = [](auto x, auto y) { return x >= y; };
  const auto is_equal = [](auto x, auto y) { return x == y; };
  const auto is_unequal = [](auto x, auto y) { return x != y; };

  EXPECT_DOUBLE_EQ(value(less, 1.0, 2.0), 1.0);
  EXPECT_DOUBLE_EQ(value(less, 2.0, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(value(less, 2.0, 2.0), 0.0);
  EXPECT_DOUBLE_EQ(value(greater, 2.0, 1.0), 1.0);
  EXPECT_DOUBLE_EQ(value(at_least, 2.0, 2.0), 1.0);
  EXPECT_DOUBLE_EQ(value(is_equal, 2.0, 2.0), 1.0);
  EXPECT_DOUBLE_EQ(value(is_equal, 2.0, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(value(is_unequal, 2.0, 1.0), 1.0);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_DOUBLE_EQ(value(is_equal, nan, nan), 0.0)
      << "a NaN equals nothing, itself least";
  EXPECT_DOUBLE_EQ(value(less, nan, 1.0), 0.0)
      << "an ordered comparison is false at a NaN";
}

// A select node has three operands, so every walker that rebuilds a node has to
// carry all three: one that rebuilt it as a pair left the third `no_node` for
// the next walker to dereference.  The compiled path exercises rebalance, the
// colouring and codegen in one call.
TEST(RtJacobian, SelectSurvivesEveryWalkerThatRebuildsANode) {
  auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return select(le(x, y), exp(x) * y, log(y) + x * x);
  });
  const std::array<double, 2> at{0.6, 1.4};
  const auto swept = *eq.jacobian(std::span<const double>{at});
  const auto swept_h = *eq.hessian(std::span<const double>{at});

#ifdef DDX_HAS_JIT
  eq.options({.backend = ddx::jit::Backend::Compile, .points = 1});
  ASSERT_TRUE(eq.wait_for_kernel());
  const auto compiled = *eq.jacobian(std::span<const double>{at});
  EXPECT_EQ(swept, compiled) << "the kernel and the sweep disagree on a select";
#endif
  EXPECT_EQ(swept_h.size(), 4u);
}

#ifdef DDX_HAS_JIT
// The interpreter and the kernel have to answer alike at every value, and a
// comparison is where they most easily do not: `<=` written as `!(b < a)` calls
// a NaN less-or-equal to itself where `fcmp ole` says false.  A model that
// never sees a NaN would never notice.
TEST(RtJacobian, ComparisonsAgreeWithTheKernelAtNaN) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  auto model = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    // Every spelling, so one wrong predicate cannot hide behind the others.
    return (x < y) + (x <= y) * 2.0 + (x > y) * 4.0 + (x >= y) * 8.0 +
           (x == y) * 16.0 + (x != y) * 32.0 + select(x <= y, x, y) * 64.0;
  });

  const std::array<std::array<double, 2>, 7> points{{{1.0, 2.0},
                                                     {2.0, 1.0},
                                                     {2.0, 2.0},
                                                     {nan, 1.0},
                                                     {1.0, nan},
                                                     {nan, nan},
                                                     {inf, inf}}};

  // A raw value as the condition, which is the only way a NaN reaches one: a
  // comparison's output is 1.0 or 0.0 and never NaN, so the model above cannot
  // tell an ordered `fcmp` from an unordered one.  C's `!=` is unordered, so a
  // NaN condition is true; the kernel has to say the same.
  {
    auto raw = ddx::rt::equation([] {
      return select(ddx::rt::var("c"), ddx::rt::var("t"), ddx::rt::var("f"));
    });
    // Symbols sort alphabetically: c, f, t.
    const std::array<double, 3> at{nan, 1.0, 2.0};
    const double interpreted = *raw.evaluate(std::span<const double>{at});
    raw.options({.backend = ddx::jit::Backend::Compile, .points = 1});
    ASSERT_TRUE(raw.wait_for_kernel());
    // Through jacobian(), not evaluate(): wait_for_kernel() waits on the
    // Jacobian lane, and the values lane would still be interpreting.
    double value = 0.0;
    std::vector<double> cells(*raw.jacobian_columns());
    const auto gs = out_columns(cells);
    const double *const cols[]{&at[0], &at[1], &at[2]};
    double *const vs[]{&value};
    ASSERT_TRUE(raw.jacobian(cols, vs, gs, 1).has_value());
    EXPECT_EQ(cells.size(), 2u) << "the condition has no partial";
    EXPECT_EQ(value, interpreted) << "a NaN condition parted the two paths";
    EXPECT_EQ(interpreted, 2.0) << "a NaN condition is true, as in C";
  }

  std::array<double, 7> swept{};
  for (const auto [i, at] : std::views::enumerate(points)) {
    swept[static_cast<std::size_t>(i)] =
        *model.evaluate(std::span<const double>{at});
  }

  model.options({.backend = ddx::jit::Backend::Compile, .points = 1});
  ASSERT_TRUE(model.wait_for_kernel());
  for (const auto [i, at] : std::views::enumerate(points)) {
    const double compiled = *model.evaluate(std::span<const double>{at});
    const double interpreted = swept[static_cast<std::size_t>(i)];
    EXPECT_EQ(std::isnan(compiled), std::isnan(interpreted))
        << "at (" << at[0] << ", " << at[1] << ")";
    if (!std::isnan(interpreted)) {
      EXPECT_EQ(compiled, interpreted)
          << "at (" << at[0] << ", " << at[1] << ")";
    }
  }
}
#endif

// Two selects over one subexpression, with both conditions false at the first
// point and both true at the second -- so each arm is exercised over the same
// nodes, and the shared `s` collects an adjoint through whichever arms are
// live.
//
// This does *not* test a select whose two arms are the same node: `make` folds
// that to the arm before a node exists
// (RtBuilder.SelectFoldsWhereABinaryWould), so a test built through the
// factories would be checking the fold.  The rule still has to handle it -- see
// SharedArmsThroughTheLoaderStillSum below, which reaches it the one way that
// bypasses `make`.
TEST(RtJacobian, BothArmsOfASelectOverASharedSubexpression) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    const auto s = sin(x * y) + z;
    return s * s + select(le(x, y), s * s, -s);
  });

  // Condition false: f = s*s - s.
  {
    const std::array<double, 3> at{0.7, 0.3, 1.9};
    const double s = std::sin(at[0] * at[1]) + at[2];
    const double c = std::cos(at[0] * at[1]);
    const auto span = std::span<const double>{at};
    EXPECT_DOUBLE_EQ(*eq.evaluate(span), s * s - s);

    const auto g = *eq.jacobian(span);
    ASSERT_EQ(g.size(), 3u);
    EXPECT_DOUBLE_EQ(g[0], (2 * s - 1) * at[1] * c);
    EXPECT_DOUBLE_EQ(g[1], (2 * s - 1) * at[0] * c);
    EXPECT_DOUBLE_EQ(g[2], 2 * s - 1);

    const auto h = *eq.hessian(span);
    ASSERT_EQ(h.size(), 9u);
    EXPECT_DOUBLE_EQ(h[8], 2.0) << "d2f/dz2, the arms being linear in z";
  }

  // Condition true, same nodes, other arm: f = 2 s*s.
  {
    const std::array<double, 3> at{0.3, 0.7, 1.9};
    const double s = std::sin(at[0] * at[1]) + at[2];
    const auto span = std::span<const double>{at};
    EXPECT_DOUBLE_EQ(*eq.evaluate(span), 2 * s * s);
    EXPECT_DOUBLE_EQ((*eq.jacobian(span))[2], 4 * s);
    EXPECT_DOUBLE_EQ((*eq.hessian(span))[8], 4.0);
  }

  // x and y reach f through `s`, never only through the condition, so no
  // column is missing -- and s*s couples all three, so nothing compresses.
  const ddx::rt::Sparsity &pattern = eq.jacobian_pattern()->get();
  EXPECT_EQ(pattern.nonzeros(), 3u)
      << "a condition-only symbol would be absent";
  EXPECT_EQ(*eq.hessian_colors(), 3u);
}

// `make` folds select(c, s, s) to s, so the factories cannot build one -- but
// `restore` installs a node array verbatim, so a file written before the fold,
// or a forged one, still can.  The rule has to keep summing there: both
// contributions land on the one child, and either alone answers the arm the
// condition happens to name rather than the derivative.
TEST(RtJacobian, SharedArmsThroughTheLoaderStillSum) {
  using ddx::rt::Node;
  ddx::rt::Snapshot<double> snap;
  snap.symbols = {"x", "y"};
  // select(x < y, x, x), which is x -- and which make() would never form.
  snap.nodes = {
      Node<double>{.op = ddx::rt::OpCode::Var, .slot = 0},
      Node<double>{.op = ddx::rt::OpCode::Var, .slot = 1},
      Node<double>{.op = ddx::rt::OpCode::Lt, .a = 0, .b = 1},
      Node<double>{.op = ddx::rt::OpCode::Select, .a = 2, .b = 0, .c = 0}};
  snap.roots = {3};
  snap.model_nodes = 4;
  // What a file carries beside the model: the loader admits nothing less.
  snap.nodes.push_back(
      Node<double>{.op = ddx::rt::OpCode::Const, .value = 0.0});
  snap.jacobian = {.value = {3},
                   .partial = {},
                   .pattern = {.rowptr = {0, 0}, .col = {}, .columns = 2},
                   .zero = 4};

  auto sound = ddx::rt::verified(std::move(snap));
  ASSERT_TRUE(sound.has_value()) << "a shared arm is a sound file";
  const auto [arena, rest] = std::move(*sound).rebuild();
  ASSERT_EQ(arena->size(), 5u) << "restore installs the array as written";
  ASSERT_EQ((*arena)[3].b, (*arena)[3].c) << "the shared arm survived the load";

  const auto j = ddx::rt::build_jacobian_impl(*arena, rest.roots);
  // Both conditions, since a rule keeping one contribution answers the arm the
  // condition names and would be right at exactly one of these points.
  for (const auto &at :
       {std::array<double, 2>{1.0, 2.0}, std::array<double, 2>{2.0, 1.0}}) {
    const auto values = ddx::rt::evaluate_all(*arena, at);
    EXPECT_DOUBLE_EQ(values[j.at(0, 0)], 1.0)
        << "d select(c, x, x)/dx at (" << at[0] << ", " << at[1] << ")";
    EXPECT_DOUBLE_EQ(values[j.at(0, 1)], 0.0);
  }
}

// pow's exponent partial is a^b * log(a), 0 * -inf at a == 0, and its base
// partial b * a^(b-1) is 0 * inf there at b == 0.  On the graph the rule guards
// both with a select; with a literal exponent the builder folds the guard away
// and the partial is the node it always was.
TEST(RtJacobian, PowAtAZeroBaseIsGuardedOnTheGraph) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return pow(x, y);
  });
  const auto at_zero_base = *eq.jacobian(0.0, 2.0);
  EXPECT_EQ(at_zero_base[0], 0.0);
  EXPECT_EQ(at_zero_base[1], 0.0);
  const auto origin = *eq.jacobian(0.0, 0.0);
  EXPECT_EQ(origin[0], 0.0);
  // Away from zero the guard chooses the partial it always computed.
  const auto plain = *eq.jacobian(2.0, 3.0);
  EXPECT_DOUBLE_EQ(plain[0], 12.0);
  EXPECT_DOUBLE_EQ(plain[1], 8.0 * std::log(2.0));
  // The condition is not differentiated, so the Hessian is the arm's.
  const auto H = *eq.hessian(2.0, 3.0);
  EXPECT_NEAR(H[0], 12.0, 1e-12);
  EXPECT_NEAR(H[1], 4.0 * (1.0 + 3.0 * std::log(2.0)), 1e-12);
  EXPECT_NEAR(H[3], 8.0 * std::log(2.0) * std::log(2.0), 1e-12);

  // A literal exponent: equal(3, 0) folds to 0, the product to 0, and the
  // select to its false arm -- 3 * (x * x), no Select node anywhere in it.
  ddx::rt::Builder<> b;
  const auto x = ddx::rt::var(b, "x");
  const auto cubic = ddx::rt::build_jacobian_impl<ddx::impl::DiffMode::Reverse>(
      b, pow(x, 3.0).id(b));
  const auto live = ddx::rt::detail::reachable(
      b.size(), std::span<const ddx::rt::NodeId>{&cubic.partial[0], 1},
      [&b](ddx::rt::NodeId v, auto &&mark) {
        std::ranges::for_each(ddx::rt::detail::operands_of(b, v), mark);
      });
  for (std::size_t v = 0; v < b.size(); ++v) {
    EXPECT_FALSE(live[v] && b[static_cast<ddx::rt::NodeId>(v)].op ==
                                ddx::rt::OpCode::Select)
        << "node " << v;
  }
  EXPECT_DOUBLE_EQ(ddx::rt::evaluate(b, cubic.partial[0], std::array{2.0}),
                   12.0);
}
