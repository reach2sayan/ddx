#include "ddx.hpp"
#include "md/md.hpp"
#include "rt/equation.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <vector>

// J v, the one product that is a forward sweep.  Checked two ways: against the
// dense jacobian() contracted by hand, and against evaluating the arena over
// Dual<double>, which carries the same directional derivative with no graph
// machinery at all.

namespace {

std::vector<double> dense_column(const std::vector<double> &jac, std::size_t m,
                                 std::size_t n, std::span<const double> v) {
  const ddx::impl::md::mdspan J{jac.data(),
                                ddx::impl::md::dextents<std::size_t, 2>{m, n}};
  std::vector<double> out(m, 0.0);
  for (const std::size_t i : std::views::iota(0uz, m)) {
    for (const std::size_t j : std::views::iota(0uz, n)) {
      out[i] += J[i, j] * v[j];
    }
  }
  return out;
}

TEST(RtJvp, MatchesTheDenseJacobianTimesTheDirection) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(exp(x * y), x * x - y, log(x) + sin(y));

  const std::array point{1.3, 0.7};
  const std::array<double, 2> v{2.0, -0.5};

  const auto dense = *eq.jacobian(point);
  const auto want = dense_column(dense, 3, 2, v);
  const auto got = *eq.jvp(std::span<const double>{v}, point);

  ASSERT_EQ(got.size(), 3u);
  for (const std::size_t i : std::views::iota(0uz, 3uz)) {
    EXPECT_NEAR(got[i], want[i], 1e-10 * std::max(1.0, std::abs(want[i])))
        << "function " << i;
  }
}

// The independent oracle: a dual point agrees without going near tangent_sweep.
TEST(RtJvp, AgreesWithADualPointThroughTheInterpreter) {
  using D = ddx::impl::Dual<double>;
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x * y) + sin(x) * y;

  const std::array point{0.6, 1.4};
  const std::array<double, 2> v{1.5, -0.25};

  const std::array<D, 2> seeded{D{point[0], v[0]}, D{point[1], v[1]}};
  const auto dual = ddx::rt::evaluate(b, f.id(b), seeded);

  const auto eq = ddx::rt::equation(f);
  const auto got = *eq.jvp(std::span<const double>{v}, point);
  EXPECT_NEAR(got[0], dual.deriv(),
              1e-12 * std::max(1.0, std::abs(dual.deriv())));
}

TEST(RtJvp, AUnitDirectionIsOneJacobianColumn) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * x * y, log(x) + y);

  const std::array point{2.0, 3.0};
  const auto dense = *eq.jacobian(point);

  for (const std::size_t j : std::views::iota(0uz, 2uz)) {
    std::array<double, 2> v{0.0, 0.0};
    v[j] = 1.0;
    const auto column = *eq.jvp(std::span<const double>{v}, point);
    for (const std::size_t i : std::views::iota(0uz, 2uz)) {
      EXPECT_NEAR(column[i], dense[i * 2 + j], 1e-10) << "column " << j;
    }
  }
}

TEST(RtJvp, RefusesADirectionOfTheWrongLength) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x + y);

  const std::vector<double> tooLong{1.0, 2.0, 3.0};
  const auto r = eq.jvp(std::span<const double>{tooLong}, 1.0, 2.0);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().code, ddx::errc::wrong_direction);
}

TEST(RtJvp, BatchAgreesWithThePointSpelling) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(exp(x) * y, x * x - y);

  static constexpr std::size_t n = 34;
  std::array<std::vector<double>, 2> xs{std::vector<double>(n),
                                        std::vector<double>(n)};
  std::array<std::vector<double>, 2> vs{std::vector<double>(n),
                                        std::vector<double>(n)};
  for (const std::size_t i : std::views::iota(0uz, n)) {
    xs[0][i] = 0.2 + 0.01 * static_cast<double>(i);
    xs[1][i] = 0.9 - 0.005 * static_cast<double>(i);
    vs[0][i] = 1.0 + 0.25 * static_cast<double>(i % 3);
    vs[1][i] = -0.5 * static_cast<double>(i % 4);
  }
  const double *const xp[]{xs[0].data(), xs[1].data()};
  const double *const vp[]{vs[0].data(), vs[1].data()};

  std::array<std::vector<double>, 2> f{std::vector<double>(n),
                                       std::vector<double>(n)};
  std::array<std::vector<double>, 2> out{std::vector<double>(n),
                                         std::vector<double>(n)};
  double *const fp[]{f[0].data(), f[1].data()};
  double *const op[]{out[0].data(), out[1].data()};

  ASSERT_TRUE(eq.jvp(xp, vp, fp, op, n).has_value());

  for (const std::size_t i : std::views::iota(0uz, n)) {
    const std::array<double, 2> v{vs[0][i], vs[1][i]};
    const auto want =
        *eq.jvp(std::span<const double>{v}, std::array{xs[0][i], xs[1][i]});
    EXPECT_NEAR(out[0][i], want[0], 1e-10 * std::max(1.0, std::abs(want[0])));
    EXPECT_NEAR(out[1][i], want[1], 1e-10 * std::max(1.0, std::abs(want[1])));
  }
}

TEST(RtJvp, CostsOneColumnPerFunction) {
  ddx::rt::Arena<> b;
  std::vector<ddx::rt::Expression<>> v;
  for (const char *name : {"a", "b", "c", "d", "e"}) {
    v.push_back(var(b, name));
  }
  auto acc = ddx::rt::Expression<>{0};
  for (const auto &s : v) {
    acc = acc + exp(s) * s;
  }
  const auto eq = ddx::rt::equation(acc);

  EXPECT_EQ(*eq.jacobian_columns(), 5u);
  EXPECT_EQ(*eq.jvp_columns(), 1u) << "one function, one column";
}

// The refactor gate: DiffMode::Symbolic is now n tangent_sweep calls with unit
// seeds, and must still emit the same arithmetic reverse mode does.
TEST(RtJvp, SymbolicModeStillAgreesWithReverse) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x * y) + log(x) * sin(y);

  const auto fwd =
      ddx::rt::derive_jacobian<ddx::DiffMode::Symbolic>(b, f.id(b));
  const auto rev = ddx::rt::derive_jacobian<ddx::DiffMode::Reverse>(b, f.id(b));

  const std::array point{1.4, 0.9};
  const auto values = ddx::rt::evaluate_all(b, point);
  for (const std::size_t j : std::views::iota(0uz, 2uz)) {
    EXPECT_NEAR(values[fwd.partial[j]], values[rev.partial[j]], 1e-12);
  }
}
} // namespace
