#include "ddx.hpp"
#include "md/md.hpp"
#include "rt/equation.hpp"

#include <gtest/gtest.h>

#include <vector>

// w'J against the dense jacobian() contracted by hand.  A system is the case
// that matters: there the product is n columns where the Jacobian is one per
// structural nonzero.

namespace {

std::vector<double> dense_row_vector(const std::vector<double> &jac,
                                     std::size_t m, std::size_t n,
                                     std::span<const double> w) {
  const ddx::impl::md::mdspan J{jac.data(),
                                ddx::impl::md::dextents<std::size_t, 2>{m, n}};
  std::vector<double> out(n, 0.0);
  for (const std::size_t j : std::views::iota(0uz, n)) {
    for (const std::size_t i : std::views::iota(0uz, m)) {
      out[j] += w[i] * J[i, j];
    }
  }
  return out;
}

TEST(RtVjp, MatchesTheDenseJacobianForASystem) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq =
      ddx::rt::equation(x * x + y * y - 4.0, x * y - 1.0, exp(x) + sin(y));
  ASSERT_FALSE(eq.poisoned());
  static_assert(decltype(eq)::output_dim == 3);

  const std::array point{1.3, 0.7};
  const std::array<double, 3> w{2.0, -1.0, 0.5};

  const auto dense = *eq.jacobian(point);
  const auto want = dense_row_vector(dense, 3, 2, w);
  const auto got = *eq.vjp(std::span<const double>{w}, point);

  ASSERT_EQ(got.size(), 2u);
  for (const std::size_t j : std::views::iota(0uz, 2uz)) {
    EXPECT_NEAR(got[j], want[j], 1e-10 * std::max(1.0, std::abs(want[j])))
        << "symbol " << j;
  }
}

TEST(RtVjp, AUnitCovectorIsOneJacobianRow) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * x * y, log(x) + y);

  const std::array point{2.0, 3.0};
  const auto dense = *eq.jacobian(point);

  for (const std::size_t i : std::views::iota(0uz, 2uz)) {
    std::array<double, 2> w{0.0, 0.0};
    w[i] = 1.0;
    const auto row = *eq.vjp(std::span<const double>{w}, point);
    for (const std::size_t j : std::views::iota(0uz, 2uz)) {
      EXPECT_NEAR(row[j], dense[i * 2 + j], 1e-10) << "row " << i;
    }
  }
}

TEST(RtVjp, AtOneOutputItIsTheScaledGradient) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto eq = ddx::rt::equation(x * x * x);

  const std::array<double, 1> w{2.5};
  const auto got = *eq.vjp(std::span<const double>{w}, 2.0);
  EXPECT_NEAR(got[0], 2.5 * 3.0 * 4.0, 1e-12);
}

TEST(RtVjp, RefusesACovectorOfTheWrongLength) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x + y, x * y);

  const std::vector<double> tooLong{1.0, 2.0, 3.0};
  const auto r = eq.vjp(std::span<const double>{tooLong}, 1.0, 2.0);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().code, ddx::errc::wrong_direction);
}

TEST(RtVjp, BatchAgreesWithThePointSpelling) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(exp(x) * y, x * x - y);

  static constexpr std::size_t n = 35;
  std::array<std::vector<double>, 2> xs{std::vector<double>(n),
                                        std::vector<double>(n)};
  std::array<std::vector<double>, 2> ws{std::vector<double>(n),
                                        std::vector<double>(n)};
  for (const std::size_t i : std::views::iota(0uz, n)) {
    xs[0][i] = 0.2 + 0.01 * static_cast<double>(i);
    xs[1][i] = 0.9 - 0.005 * static_cast<double>(i);
    ws[0][i] = 1.0 + 0.25 * static_cast<double>(i % 3);
    ws[1][i] = -0.5 * static_cast<double>(i % 4);
  }
  const double *const xp[]{xs[0].data(), xs[1].data()};
  const double *const wp[]{ws[0].data(), ws[1].data()};

  std::array<std::vector<double>, 2> f{std::vector<double>(n),
                                       std::vector<double>(n)};
  std::array<std::vector<double>, 2> out{std::vector<double>(n),
                                         std::vector<double>(n)};
  double *const fp[]{f[0].data(), f[1].data()};
  double *const op[]{out[0].data(), out[1].data()};

  ASSERT_TRUE(eq.vjp(xp, wp, fp, op, n).has_value());

  for (const std::size_t i : std::views::iota(0uz, n)) {
    const std::array<double, 2> w{ws[0][i], ws[1][i]};
    const auto want =
        *eq.vjp(std::span<const double>{w}, std::array{xs[0][i], xs[1][i]});
    EXPECT_NEAR(out[0][i], want[0], 1e-10 * std::max(1.0, std::abs(want[0])));
    EXPECT_NEAR(out[1][i], want[1], 1e-10 * std::max(1.0, std::abs(want[1])));
  }
}

TEST(RtVjp, CostsFewerColumnsThanTheJacobian) {
  ddx::rt::Builder<> b;
  std::vector<ddx::rt::Expression<>> v;
  for (const char *name : {"a", "b", "c", "d"}) {
    v.push_back(var(b, name));
  }
  const auto eq = ddx::rt::equation(v[0] * v[1] + v[2] * v[3],
                                    exp(v[0]) + v[1] * v[2] * v[3],
                                    v[0] + v[1] + v[2] + v[3]);

  EXPECT_EQ(*eq.jacobian_columns(), 12u)
      << "three dense rows over four symbols";
  EXPECT_EQ(*eq.vjp_columns(), 4u);
}
} // namespace
