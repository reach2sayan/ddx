#include "energy_models.hpp"
#include "rt/equation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <format>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// dJ/dv: the derivative of every Jacobian cell along a direction, from one
// forward sweep over the Jacobian's cone -- and a view that keeps only some
// of the Jacobian's columns.
namespace {

using RE = ddx::rt::Expression<>;

// f = {x^2 y, sin(xy)}, so
//   J     = [[2xy, x^2], [y cos(xy), x cos(xy)]]
//   dJ/dx = [[2y, 2x], [-y^2 sin(xy), cos(xy) - xy sin(xy)]]
//   dJ/dy = [[2x, 0], [cos(xy) - xy sin(xy), -x^2 sin(xy)]]
[[nodiscard]] auto pair() {
  return ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return std::array{x * x * y, sin(x * y)};
  });
}

constexpr double px = 0.7;
constexpr double py = 1.3;

[[nodiscard]] std::array<double, 4> d_dx() {
  const double s = std::sin(px * py);
  const double c = std::cos(px * py);
  return {2 * py, 2 * px, -py * py * s, c - px * py * s};
}
[[nodiscard]] std::array<double, 4> d_dy() {
  const double s = std::sin(px * py);
  const double c = std::cos(px * py);
  return {2 * px, 0.0, c - px * py * s, -px * px * s};
}

void expect_near_all(std::span<const double> got, std::span<const double> want,
                     double tol) {
  ASSERT_EQ(got.size(), want.size());
  for (const auto [k, g] : got | std::views::enumerate) {
    EXPECT_NEAR(g, want[static_cast<std::size_t>(k)], tol) << "cell " << k;
  }
}

TEST(RtJacobianJvp, AlongABasisVectorIsTheAnalyticDerivative) {
  const auto eq = pair();
  expect_near_all(*eq.jacobian_jvp("x", px, py), d_dx(), 1e-14);
  expect_near_all(*eq.jacobian_jvp("y", px, py), d_dy(), 1e-14);
}

TEST(RtJacobianJvp, AlongAnyDirectionIsTheCombination) {
  const auto eq = pair();
  const std::array v{0.3, -1.7};
  const auto got = *eq.jacobian_jvp(std::span<const double>{v}, px, py);
  std::array<double, 4> want{};
  for (const std::size_t k : std::views::iota(0uz, 4uz)) {
    want[k] = v[0] * d_dx()[k] + v[1] * d_dy()[k];
  }
  expect_near_all(got, want, 1e-14);

  // And a central difference of the Jacobian along v.
  constexpr double h = 1e-6;
  const auto ahead = *eq.jacobian(px + h * v[0], py + h * v[1]);
  const auto behind = *eq.jacobian(px - h * v[0], py - h * v[1]);
  for (const std::size_t k : std::views::iota(0uz, 4uz)) {
    EXPECT_NEAR(got[k], (ahead[k] - behind[k]) / (2 * h), 1e-6) << k;
  }
}

TEST(RtJacobianJvp, ForOneFunctionItIsAHessianColumn) {
  const auto eq = ddx::rt::equation([] {
    const auto t = ddx::rt::var("T");
    const auto x = ddx::rt::var("x");
    return exp(t * x) * log(t) + x * x * t;
  });
  const std::array at{1.4, 0.6};
  const auto H = *eq.hessian(at);
  const auto column = *eq.jacobian_jvp("T", at);
  ASSERT_EQ(column.size(), 2u);
  for (const std::size_t j : std::views::iota(0uz, 2uz)) {
    EXPECT_NEAR(column[j], H[j * 2 + 0], 1e-12) << j;
  }
}

TEST(RtJacobianJvp, TheBatchMatchesThePointCalls) {
  const auto eq = pair();
  constexpr std::size_t n = 11;
  const std::size_t cells = *eq.jacobian_columns();
  std::vector<double> xs(n);
  std::vector<double> ys(n);
  std::vector<double> vx(n, 1.0);
  std::vector<double> vy(n, 0.5);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    xs[i] = 0.1 + 0.1 * static_cast<double>(i);
    ys[i] = 1.0 - 0.05 * static_cast<double>(i);
  }
  std::vector<std::vector<double>> f(2, std::vector<double>(n));
  std::vector<std::vector<double>> g(cells, std::vector<double>(n));
  std::vector<std::vector<double>> dg(cells, std::vector<double>(n));
  const auto ptrs = [](auto &columns) {
    return columns | std::views::transform([](auto &c) { return c.data(); }) |
           ddx::impl::to<std::vector<double *>>();
  };
  const std::array<const double *, 2> in{xs.data(), ys.data()};
  const std::array<const double *, 2> dir{vx.data(), vy.data()};
  ASSERT_TRUE(
      eq.jacobian_jvp(in, dir, ptrs(f), ptrs(g), ptrs(dg), n).has_value());

  const auto &pattern = eq.jacobian_pattern()->get();
  for (const std::size_t i : std::views::iota(0uz, n)) {
    const std::array v{vx[i], vy[i]};
    const auto want =
        *eq.jacobian_jvp(std::span<const double>{v}, xs[i], ys[i]);
    for (const auto [row, column, slot] :
         pattern.cells(std::views::iota(0uz, cells))) {
      EXPECT_EQ(dg[slot][i], want[row * 2 + column]) << "point " << i;
    }
  }
}

TEST(RtJacobianJvp, ASystemTakesItInOneSweep) {
  const auto sys = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return std::vector<RE>{x * x * y, sin(x * y)};
  });
  expect_near_all(*sys.jacobian_jvp("x", px, py), d_dx(), 1e-14);
}

TEST(RtJacobianJvp, AWrongDirectionOrNameIsRefused) {
  const auto eq = pair();
  const std::array v{1.0};
  EXPECT_EQ(eq.jacobian_jvp(std::span<const double>{v}, px, py).error().code,
            ddx::errc::wrong_direction);
  EXPECT_EQ(eq.jacobian_jvp("z", px, py).error().code,
            ddx::errc::unknown_symbol);
}

// --- restrict ---------------------------------------------------------------

// T, P and the mole fractions: the Jacobian of the MSE-shaped model, and the
// view a flash at fixed composition wants -- d/dT and d/dP only.
[[nodiscard]] auto mse(std::size_t species) {
  return ddx::rt::equation([species] {
    const auto t = ddx::rt::var("T");
    const auto p = ddx::rt::var("P");
    std::vector<RE> x;
    for (const std::size_t i : std::views::iota(0uz, species)) {
      x.push_back(ddx::rt::var(std::format("x{:02}", i)));
    }
    return models::mse(x) * t + log(p) * x[0];
  });
}

[[nodiscard]] std::vector<double> mse_point(std::size_t species) {
  std::vector<double> at{350.0, 2.0};
  for (const std::size_t i : std::views::iota(0uz, species)) {
    at.push_back(1.0 / static_cast<double>(species) +
                 0.01 * static_cast<double>(i % 3));
  }
  return at;
}

TEST(RtRestrict, KeepsOnlyTheChosenColumns) {
  constexpr std::size_t species = 6;
  const auto eq = mse(species);
  const auto tp = eq.restrict({"P", "T"});
  ASSERT_TRUE(tp.has_value());
  // Symbol order, however the names were given.
  EXPECT_TRUE(std::ranges::equal(tp->symbols(),
                                 std::array<std::string_view, 2>{"T", "P"}));

  std::size_t cells_in_tp = 0;
  for (const auto &row : eq.jacobian_pattern()->get().rows()) {
    cells_in_tp += static_cast<std::size_t>(
        std::ranges::count_if(row, [](std::uint32_t c) { return c < 2; }));
  }
  EXPECT_EQ(tp->jacobian_columns(), cells_in_tp);

  const auto at = mse_point(species);
  const auto full = *eq.jacobian(at);
  const auto part = *tp->jacobian(at);
  ASSERT_EQ(part.size(), 2u);
  EXPECT_EQ(part[0], full[0]);
  EXPECT_EQ(part[1], full[1]);

  const auto full_dt = *eq.jacobian_jvp("T", at);
  const auto part_dt = *tp->jacobian_jvp("T", at);
  EXPECT_EQ(part_dt[0], full_dt[0]);
  EXPECT_EQ(part_dt[1], full_dt[1]);
}

// A restriction that keeps no cell keeps none, not all of them: y's column is
// there, and zero.
TEST(RtRestrict, KeepingNoCellIsNoColumns) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    (void)ddx::rt::var("y");
    return x * x;
  });
  const auto none = eq.restrict({"y"});
  ASSERT_TRUE(none.has_value());
  EXPECT_EQ(none->jacobian_columns(), 0u);
  const auto j = none->jacobian(3.0, 4.0);
  ASSERT_TRUE(j.has_value()) << j.error().code;
  EXPECT_EQ(*j, std::vector{0.0});
  const auto dt = none->jacobian_jvp("y", 3.0, 4.0);
  ASSERT_TRUE(dt.has_value()) << dt.error().code;
  EXPECT_EQ(*dt, std::vector{0.0});
}

TEST(RtRestrict, ABatchThroughTheView) {
  constexpr std::size_t species = 4;
  const auto eq = mse(species);
  const auto tp = *eq.restrict({"T", "P"});
  constexpr std::size_t n = 13;
  const auto at = mse_point(species);
  std::vector<std::vector<double>> columns(at.size(), std::vector<double>(n));
  for (const auto [j, column] : columns | std::views::enumerate) {
    for (const auto [i, cell] : column | std::views::enumerate) {
      cell = at[static_cast<std::size_t>(j)] *
             (1.0 + 0.001 * static_cast<double>(i));
    }
  }
  const auto in =
      columns | std::views::transform([](const auto &c) { return c.data(); }) |
      ddx::impl::to<std::vector<const double *>>();
  std::vector<double> f(n);
  std::vector<std::vector<double>> g(tp.jacobian_columns(),
                                     std::vector<double>(n));
  const std::array<double *, 1> fs{f.data()};
  const auto gs = g | std::views::transform([](auto &c) { return c.data(); }) |
                  ddx::impl::to<std::vector<double *>>();
  ASSERT_TRUE(tp.jacobian(in, fs, gs, n).has_value());
  for (const std::size_t i : std::views::iota(0uz, n)) {
    std::vector<double> point;
    for (const auto &column : columns) {
      point.push_back(column[i]);
    }
    const auto full = *eq.jacobian(point);
    EXPECT_EQ(g[0][i], full[0]) << i;
    EXPECT_EQ(g[1][i], full[1]) << i;
  }
  // Two columns, so a block sized for the whole Jacobian is refused.
  std::vector<std::vector<double>> wide(*eq.jacobian_columns(),
                                        std::vector<double>(n));
  const auto ws = wide |
                  std::views::transform([](auto &c) { return c.data(); }) |
                  ddx::impl::to<std::vector<double *>>();
  EXPECT_EQ(tp.jacobian(in, fs, ws, n).error().code,
            ddx::errc::wrong_column_count);
}

TEST(RtRestrict, AnUnknownNameIsRefused) {
  const auto eq = pair();
  EXPECT_EQ(eq.restrict({"x", "nope"}).error().code, ddx::errc::unknown_symbol);
}

} // namespace
