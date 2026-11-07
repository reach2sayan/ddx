#include "ddx.hpp"
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

// to_graph lowers a compile-time expression into the graph, so Equation and the
// sweep can differentiate the same function and must agree.

namespace {

// Batch calls take one pointer per column. Inputs and outputs need separate
// overloads, since one taking both is ambiguous for a non-const vector.
[[nodiscard]] std::vector<double *> out_columns(std::vector<double> &values) {
  return values | std::views::transform([](double &v) { return &v; }) |
         ddx::impl::to<std::vector<double *>>();
}

[[nodiscard]] std::vector<const double *> in_columns(const auto &values) {
  return values | std::views::transform([](const double &v) { return &v; }) |
         ddx::impl::to<std::vector<const double *>>();
}

// An overflow is an answer both engines must reach, which EXPECT_NEAR cannot
// check because inf - inf is NaN.
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

// to_graph declares symbols in the compile-time Equation's canonical order, so
// the columns of both sides line up by name.
template <ddx::impl::CExpression E, std::size_t N>
void expect_agrees_with_ddx(const E &e, std::array<double, N> pt) {
  ddx::rt::Arena<> b;
  const auto root = ddx::rt::to_graph(b, e);
  const auto g = ddx::rt::derive_jacobian(b, root.id(b));
  const auto values = ddx::rt::evaluate_all(b, std::span<const double>{pt});

  const double expected = std::apply(
      [&](auto... a) { return ddx::Equation{e}.evaluate(a...); }, pt);
  expect_close(values[g.value], expected);

  const auto expected_grad = std::apply(
      [&](auto... a) { return ddx::Equation{e}.jacobian(a...); }, pt);

  for (std::size_t i = 0; i < b.symbols().size(); ++i) {
    SCOPED_TRACE("partial d/d" + std::string{b.symbols()[i]});
    expect_close(values[g.partial[i]], expected_grad[i]);
  }
}

// A separate namespace so the runtime tests below can also name their
// variables x, y and z.
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
constexpr bool bridges =
    requires(ddx::rt::Arena<> &b, const E &e) { ddx::rt::to_graph(b, e); };

// rt has no leaf that reads a symbol but differentiates to zero, so a frozen
// leaf is refused; a Jacobian row comes back thawed and lowers normally.
TEST(RtJacobian, BridgeRefusesFrozenLeaves) {
  using ddx::impl::FixedString;
  const auto column =
      ddx::impl::make_all_constant_except<FixedString{"x"}>(x * x * y)
          .derivative();
  const auto row = ddx::impl::thaw_partials(column);

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

// max, min and abs have no compile-time derivative entry, so they are checked
// against central differences on the interpreter.
TEST(RtJacobian, SelectingOpsAgainstFiniteDifferences) {
  using ddx::rt::Arena;
  using ddx::rt::Expression;
  const auto check = [](auto build, std::array<double, 2> pt) {
    Arena<> b;
    const auto vx = var(b, "x");
    const auto vy = var(b, "y");
    const auto f = build(vx, vy);
    const auto g = ddx::rt::derive_jacobian(b, f.id(b));
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
  check([](Expression<> a, Expression<> b2) { return max(a, b2); }, {1.3, 2.1});
  check([](Expression<> a, Expression<> b2) { return min(a, b2); }, {1.3, 2.1});
  check([](Expression<> a, Expression<> b2) { return abs(a * b2); },
        {1.3, 2.1});
  check([](Expression<> a, Expression<> b2) { return abs(a * b2); },
        {-1.3, 2.1});
}

TEST(RtJacobian, DerivFromValueReusesThePrimalNode) {
  // exp's derivative is its primal, so the sweep must not build a second exp.
  ddx::rt::Arena<> b;
  const auto vx = var(b, "x");
  const auto f = exp(vx);
  const auto g = ddx::rt::derive_jacobian(b, f.id(b));
  EXPECT_EQ(g.partial[0], f.id(b));
}

// Every op must have one shared rule in ops/adjoints.hpp that instantiates for
// both double and the graph; a missing rule stops this from compiling.
#define DDX_TEST_UNARY_RULE(fn, Op, label, functor, Desc)                      \
  static_assert(requires(const double &a) {                                    \
    ddx::impl::detail::adjoints_of<ddx::impl::detail::Desc>(a, a);             \
  });                                                                          \
  static_assert(requires(const ddx::rt::Expression<> &a) {                     \
    ddx::impl::detail::adjoints_of<ddx::impl::detail::Desc>(a, a);             \
  });
DDX_RT_UNARY_TABLE(DDX_TEST_UNARY_RULE)
#undef DDX_TEST_UNARY_RULE

#define DDX_TEST_BINARY_RULE(fn, Op, label, functor, Desc)                     \
  static_assert(requires(const double &a) {                                    \
    ddx::impl::detail::adjoints_of<ddx::impl::detail::Desc>(a, a, a);          \
  });                                                                          \
  static_assert(requires(const ddx::rt::Expression<> &a) {                     \
    ddx::impl::detail::adjoints_of<ddx::impl::detail::Desc>(a, a, a);          \
  });
DDX_RT_BINARY_TABLE(DDX_TEST_BINARY_RULE)
#undef DDX_TEST_BINARY_RULE

// Comparisons return expressions, so ExtremumOp needs its own rule and
// AbsOpFn's sign() resolves to rt::Expression's hidden friend.
static_assert(!std::totally_ordered<ddx::rt::Expression<>>);
static_assert(!std::equality_comparable<ddx::rt::Expression<>>);

// A node stands for a double, so it is commutative the way a double is, which
// selects DivideOpFn's branch.
static_assert(ddx::impl::CCommutativeMultiply<ddx::rt::Expression<>>);

TEST(RtJacobian, UnusedSymbolHasZeroPartial) {
  ddx::rt::Arena<> b;
  const auto vx = var(b, "x");
  const auto vy = var(b, "y");
  const auto f = vx * vx;
  const auto g = ddx::rt::derive_jacobian(b, f.id(b));
  ASSERT_EQ(g.partial.size(), 2u);
  EXPECT_TRUE(b.is_constant(g.partial[1], 0.0));
  (void)vy;
}

// hypot and atan2 divide by the hypotenuse first so they survive where
// x*x + y*y overflows, and pow's left partial must stay finite at a == 0.
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

// Finite differences cannot see a sign convention that is wrong at 0, so abs
// and neg are checked against Equation.
TEST(RtJacobian, UnaryRulesAgree) {
  for (const double a : {1.3, -1.3, 1e-300, 4.0}) {
    expect_agrees_with_ddx(-x, std::array{a});
    expect_agrees_with_ddx(-(x * x), std::array{a});
  }
}

// DiffMode::Symbolic carries a tangent up the graph rather than an adjoint down
// it, reading the same rules seeded with a unit adjoint.
TEST(RtJacobian, ForwardModeAgreesWithReverse) {
  const auto check = [](auto build, std::array<double, 2> pt) {
    ddx::rt::Arena<> b;
    const auto vx = var(b, "x");
    const auto vy = var(b, "y");
    const auto f = build(vx, vy);
    const auto fwd =
        ddx::rt::derive_jacobian<ddx::DiffMode::Symbolic>(b, f.id(b));
    const auto rev =
        ddx::rt::derive_jacobian<ddx::DiffMode::Reverse>(b, f.id(b));
    const auto values = ddx::rt::evaluate_all(b, std::span<const double>{pt});
    for (std::size_t i = 0; i < 2; ++i) {
      const double want = values[rev.partial[i]];
      EXPECT_NEAR(values[fwd.partial[i]], want,
                  1e-12 * std::max(1.0, std::abs(want)))
          << "partial " << i;
    }
  };
  using ddx::rt::Expression;
  check([](Expression<> a, Expression<> b2) { return a * b2 + sin(a); },
        {0.3, 0.7});
  check([](Expression<> a, Expression<> b2) { return a / b2; }, {1.3, 2.1});
  check([](Expression<> a, Expression<> b2) { return pow(a, b2); }, {1.7, 2.3});
  check([](Expression<> a, Expression<> b2) { return hypot(a, b2); },
        {1.3, 2.1});
  check([](Expression<> a, Expression<> b2) { return -(a * b2); }, {1.3, 2.1});
  check([](Expression<> a, Expression<> b2) { return max(a, b2); }, {1.3, 2.1});
}

} // namespace sym
} // namespace

// Residual i touches only x[i-1], x[i] and x[i+1], so 16 of the 36 Jacobian
// cells exist and the dense form must fill in the other 20.
TEST(RtJacobian, ABandedSystemKeepsOnlyTheCellsThatExist) {
  static constexpr std::size_t n = 6;
  ddx::rt::Arena<> b;

  const auto name = [](std::size_t i) { return "x" + std::to_string(i); };
  std::vector<ddx::rt::Expression<double>> x;
  for (const std::size_t i : std::views::iota(0uz, n)) {
    x.push_back(var(b, name(i)));
  }
  // Squares rather than plain sums, since otherwise every partial would be a
  // constant and the sweep would fold the whole row away.
  std::vector<ddx::rt::Expression<double>> rows;
  for (const std::size_t i : std::views::iota(0uz, n)) {
    auto r = x[i] * x[i];
    if (i > 0 && i + 1 < n) {
      r = r + x[i - 1] * x[i + 1];
    }
    rows.push_back(r);
  }

  const auto j = ddx::rt::derive_jacobian(b, [&] {
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

  // Exactly the band: |i - j| <= 1 for interior rows, the diagonal alone at the
  // ends, and no_column everywhere else.
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

TEST(RtJacobian, DenseCompressedAndTheNodesAllAgree) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto z = var(b, "z");
  // No row is dense and no column is empty.
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

// In a*b - a*b + c the adjoints reaching a*b fold to zero before a node is
// built, so a and b get no column, as if the partial never existed.
TEST(RtJacobian, AFoldedPartialIsAStructuralHole) {
  ddx::rt::Arena<> b;
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
  // Row 0 keeps c alone, row 1 keeps d alone, and row 2 is a constant.
  EXPECT_EQ(pattern.nonzeros(), 2u);
  const auto widths =
      pattern.rows() |
      std::views::transform([](const auto &row) { return row.size(); }) |
      ddx::impl::to<std::vector<std::size_t>>();
  EXPECT_EQ(widths, (std::vector<std::size_t>{1, 1, 0}));
  EXPECT_TRUE(pattern.at(0, 2).has_value());  // c
  EXPECT_FALSE(pattern.at(0, 0).has_value()); // a
  EXPECT_FALSE(pattern.at(0, 1).has_value()); // b

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

// x0 couples with every other symbol and nothing else couples, so 16 of 36
// cells exist and hessian_cell must place each compressed column.
TEST(RtJacobian, HessianCellPlacesEveryCompressedColumn) {
  static constexpr std::size_t n = 6;
  ddx::rt::Arena<> b;
  std::vector<ddx::rt::Expression<double>> x;
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

// The condition contributes nothing to the derivative. Checked on both sides of
// the switch, where a rule that leaked the other arm would show.
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

    const auto h = *eq.hessian(std::span<const double>{at});
    EXPECT_DOUBLE_EQ(h[0], takes_x ? 2.0 : 0.0);
    EXPECT_DOUBLE_EQ(h[3], takes_x ? 0.0 : 6 * py);
  }
}

// The unchosen arm's adjoint is a runtime zero, and log' and sqrt' are infinite
// at 0, so without the guard its contribution would be 0 * inf = NaN.
TEST(RtJacobian, SelectDiscardsTheUnchosenArmsPartials) {
  const auto clamp = [](auto arm) {
    return ddx::rt::equation([arm] {
      const auto x = ddx::rt::var("x");
      return select(x > 0.0, arm(x), ddx::rt::Expression<double>{0.0});
    });
  };
  const auto log_arm = clamp([](const auto &x) { return log(x); });
  const auto sqrt_arm = clamp([](const auto &x) { return sqrt(x); });
  const auto nested = clamp([](const auto &x) { return log(sqrt(x)); });

  for (const auto *eq : {&log_arm, &sqrt_arm, &nested}) {
    EXPECT_EQ((*eq->jacobian(0.0))[0], 0.0);
    EXPECT_EQ((*eq->hessian(0.0))[0], 0.0);
    EXPECT_EQ((*eq->hvp("x", 0.0))[0], 0.0);
    EXPECT_EQ((*eq->jacobian_jvp("x", 0.0))[0], 0.0);
    const std::array one{1.0};
    EXPECT_EQ((*eq->jvp(std::span<const double>{one}, 0.0))[0], 0.0);
    EXPECT_EQ((*eq->vjp(std::span<const double>{one}, 0.0))[0], 0.0);
    EXPECT_EQ((*eq->jacobian(-1.0))[0], 0.0);
  }
  EXPECT_DOUBLE_EQ((*log_arm.jacobian(2.0))[0], 0.5);
  EXPECT_DOUBLE_EQ((*log_arm.hessian(2.0))[0], -0.25);

  // Both arms are chosen within one batch.
  const std::array<double, 3> xs{-1.0, 0.0, 1.0};
  std::array<double, 3> f{};
  std::array<double, 3> g{};
  const std::array<const double *, 1> in{xs.data()};
  const std::array<double *, 1> fv{f.data()};
  const std::array<double *, 1> gv{g.data()};
  ASSERT_TRUE(log_arm.jacobian(in, fv, gv, xs.size()).has_value());
  EXPECT_EQ(g, (std::array{0.0, 0.0, 1.0}));
}

// The guard hides the arm not taken, never a singularity on the chosen arm; u
// is reached both ways, and at 0 the chosen arm is u itself.
TEST(RtJacobian, SelectKeepsTheChosenArmsSingularity) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto u = sqrt(x);
    return select(x > 0.0, log(u), u);
  });
  EXPECT_EQ((*eq.jacobian(0.0))[0], std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ((*eq.jacobian(4.0))[0], 0.125);
}

// A condition is true for any nonzero value, as in C; there is no boolean type.
TEST(RtJacobian, SelectReadsAnyNonzeroConditionAsTrue) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return select(x, ddx::rt::Expression<double>{7.0},
                  ddx::rt::Expression<double>{9.0});
  });
  const std::array<double, 1> zero{0.0};
  const std::array<double, 1> two{2.0};
  const std::array<double, 1> negative{-3.0};
  EXPECT_DOUBLE_EQ(*eq.evaluate(std::span<const double>{zero}), 9.0);
  EXPECT_DOUBLE_EQ(*eq.evaluate(std::span<const double>{two}), 7.0);
  EXPECT_DOUBLE_EQ(*eq.evaluate(std::span<const double>{negative}), 7.0);
}

// eq is a <= b && b <= a, which gives 0 when an operand is NaN, where
// 1 - lt - lt would have called two NaNs equal.
TEST(RtJacobian, ComparisonsComposeFromTheTwoOpcodes) {
  // Equation is move-only, so each one is built and read in place rather than
  // returned from a helper.
  const auto value = [](auto build, double a, double b) {
    const auto eq = ddx::rt::equation([&build] {
      const auto x = ddx::rt::var("x");
      return build(x, ddx::rt::var("y"));
    });
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

// A select node has three operands, and every walker that rebuilds a node must
// carry all three. The compiled path runs rebalance, colouring and codegen.
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
  ASSERT_TRUE(eq.uses_kernel());
  const auto compiled = *eq.jacobian(std::span<const double>{at});
  EXPECT_EQ(swept, compiled) << "the kernel and the sweep disagree on a select";
#endif
  EXPECT_EQ(swept_h.size(), 4u);
}

#ifdef DDX_HAS_JIT
// Writing <= as !(b < a) calls NaN less-or-equal to itself where fcmp ole says
// false, so interpreter and kernel are compared at NaN.
TEST(RtJacobian, ComparisonsAgreeWithTheKernelAtNaN) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  auto model = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    // Every operator is used, so one wrong predicate cannot hide.
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

  // A comparison never yields NaN, so a raw value is used as the condition. C's
  // != is unordered, so a NaN condition is true, and the kernel must agree.
  {
    // One declaration per statement, because argument evaluation order is
    // unspecified and symbols take the order they are named in.
    auto raw = ddx::rt::equation([] {
      const auto c = ddx::rt::var("c");
      const auto f = ddx::rt::var("f");
      const auto t = ddx::rt::var("t");
      return select(c, t, f);
    });
    const std::array<double, 3> at{nan, 1.0, 2.0}; // c, f, t
    const double interpreted = *raw.evaluate(std::span<const double>{at});
    raw.options({.backend = ddx::jit::Backend::Compile, .points = 1});
    ASSERT_TRUE(raw.uses_kernel());
    // Through jacobian() rather than evaluate(): uses_kernel() builds the
    // Jacobian lane, while the values lane would still be interpreting.
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
  ASSERT_TRUE(model.uses_kernel());
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

// Both conditions are false at the first point and true at the second, so the
// shared s collects its adjoint through each pair of arms over the same nodes.
TEST(RtJacobian, BothArmsOfASelectOverASharedSubexpression) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    const auto s = sin(x * y) + z;
    return s * s + select(le(x, y), s * s, -s);
  });

  // With the condition false, f = s*s - s.
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

  // With the condition true, the same nodes take the other arm: f = 2 s*s.
  {
    const std::array<double, 3> at{0.3, 0.7, 1.9};
    const double s = std::sin(at[0] * at[1]) + at[2];
    const auto span = std::span<const double>{at};
    EXPECT_DOUBLE_EQ(*eq.evaluate(span), 2 * s * s);
    EXPECT_DOUBLE_EQ((*eq.jacobian(span))[2], 4 * s);
    EXPECT_DOUBLE_EQ((*eq.hessian(span))[8], 4.0);
  }

  // x and y reach f through s and not only through the condition, so no column
  // is missing, and s*s couples all three so nothing compresses.
  const ddx::rt::Sparsity &pattern = eq.jacobian_pattern()->get();
  EXPECT_EQ(pattern.nonzeros(), 3u)
      << "a condition-only symbol would be absent";
  EXPECT_EQ(*eq.hessian_colours(), 3u);
}

// At a == 0 both pow partials are 0 * inf, so the graph guards them with a
// select; a literal exponent lets the arena fold the guard away.
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
  // Away from zero, the guard chooses the partial it always computed.
  const auto plain = *eq.jacobian(2.0, 3.0);
  EXPECT_DOUBLE_EQ(plain[0], 12.0);
  EXPECT_DOUBLE_EQ(plain[1], 8.0 * std::log(2.0));
  // The condition is not differentiated, so the Hessian is the chosen arm's.
  const auto H = *eq.hessian(2.0, 3.0);
  EXPECT_NEAR(H[0], 12.0, 1e-12);
  EXPECT_NEAR(H[1], 4.0 * (1.0 + 3.0 * std::log(2.0)), 1e-12);
  EXPECT_NEAR(H[3], 8.0 * std::log(2.0) * std::log(2.0), 1e-12);

  // With a literal exponent, equal(3, 0) folds to 0 and the select to its false
  // arm, 3 * (x * x), so no Select node remains.
  ddx::rt::Arena<> b;
  const auto x = ddx::rt::var(b, "x");
  const auto cubic = ddx::rt::derive_jacobian<ddx::impl::DiffMode::Reverse>(
      b, pow(x, 3.0).id(b));
  const auto live =
      ddx::rt::detail::reachable(b, std::span{&cubic.partial[0], 1});
  for (std::size_t v = 0; v < b.size(); ++v) {
    EXPECT_FALSE(live[v] && b[static_cast<ddx::rt::NodeId>(v)].op ==
                                ddx::rt::OpCode::Select)
        << "node " << v;
  }
  EXPECT_DOUBLE_EQ(ddx::rt::evaluate(b, cubic.partial[0], std::array{2.0}),
                   12.0);
}

// A partial taken during construction can be built on, as in Proton's
// d h/d Tr at the reference with the pressure moving with T.
TEST(RtJacobian, DerivativeBuildsOnTheGraph) {
  using RE = ddx::rt::Expression<>;
  const auto eq = ddx::rt::equation([] {
    const RE T = ddx::rt::var("T"), Tr = ddx::rt::var("Tr"),
             P = ddx::rt::var("P");
    const RE ps = exp(1.0 - 2.0 / T);
    const RE h = Tr * Tr * log(ps) + sin(Tr) * ps;
    EXPECT_EQ(ddx::rt::gradient(h).size(), 3u);
    return T * T * ddx::rt::derivative(h, "Tr") +
           P * (1.0 + ddx::rt::derivative(h, "P"));
  });
  const double T = 1.3, Tr = 0.7, P = 2.0;
  const double ps = std::exp(1.0 - 2.0 / T), D = 2 * Tr * std::log(ps) +
                                                 std::cos(Tr) * ps;
  expect_close(*eq.evaluate(T, Tr, P), T * T * D + P);
  const auto J = *eq.jacobian(T, Tr, P);
  expect_close(J[0], 2 * T * D + 4 * Tr + 2 * std::cos(Tr) * ps);
  expect_close(J[1], T * T * (2 * std::log(ps) - std::sin(Tr) * ps));
  expect_close(J[2], 1.0);
  EXPECT_EQ(eq.jacobian_pattern()->get().nonzeros(), 3u);
}

TEST(RtJacobian, DerivativeRefusals) {
  using RE = ddx::rt::Expression<>;
  const auto refused = [](auto make) {
    return ddx::rt::equation([&] { return make(); }).evaluate(1.0).error().code;
  };
  EXPECT_EQ(refused([] {
              return ddx::rt::derivative(ddx::rt::var("x") * 2.0, "y");
            }),
            ddx::errc::unknown_symbol);
  EXPECT_EQ(refused([] {
              const RE x = ddx::rt::var("x");
              return x + ddx::rt::derivative(x * ddx::rt::param("p"), "p");
            }),
            ddx::errc::unknown_symbol);
  EXPECT_EQ(refused([] {
              const auto v = ddx::rt::vec("v", 2);
              return sum(v) + ddx::rt::derivative(v, "v[0]");
            }),
            ddx::errc::tensor_shape);
  EXPECT_TRUE(ddx::rt::derivative(RE{2.0}, "x").poisoned());
}
