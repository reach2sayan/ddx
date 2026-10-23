#include "ddx.hpp"
#include "energy_models.hpp"
#include "rt/equation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <format>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// An implicit input: a value the caller solved for, differentiated as the root
// of the residual implicit() names -- dz/dx = -F_x / F_z, and second order by
// the same rule applied again.
namespace {

using RE = ddx::rt::Expression<>;
using namespace ddx::literals;

// z^2 = a, so z = sqrt(a): dz/da = 1 / (2 sqrt a), d2z/da2 = -1 / (4 a^1.5).
[[nodiscard]] auto root_of_a() {
  return ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto z =
        ddx::rt::implicit("z", [&](const RE &z) { return z * z - a; });
    return z;
  });
}

TEST(RtImplicit, FirstAndSecondOrderOfAClosedForm) {
  const auto eq = root_of_a();
  ASSERT_FALSE(eq.poisoned()) << eq.status()->code;
  EXPECT_EQ(eq.arity(), 1u) << "z is an input, not a symbol";
  EXPECT_EQ(*eq.inputs(), (std::vector<std::string>{"a", "z"}));
  EXPECT_EQ(eq.jacobian_columns(), 1u);

  // a = 4, z = 2, as positional (symbols, then implicits) and by name.
  EXPECT_NEAR((*eq.jacobian(4.0, 2.0))[0], 0.25, 1e-15);
  EXPECT_NEAR((*eq.gradient("z"_s = 2.0, "a"_s = 4.0))[0], 0.25, 1e-15);
  EXPECT_NEAR((*eq.hessian(4.0, 2.0))[0], -0.03125, 1e-15);
  EXPECT_NEAR((*eq.hvp("a", 4.0, 2.0))[0], -0.03125, 1e-15);
  EXPECT_NEAR((*eq.jacobian_jvp("a", 4.0, 2.0))[0], -0.03125, 1e-15);
  const std::array v{1.0};
  EXPECT_NEAR((*eq.jvp(std::span<const double>{v}, 4.0, 2.0))[0], 0.25, 1e-15);
}

TEST(RtImplicit, ResidualIsZeroAtTheRootTheCallerSupplies) {
  const auto eq = root_of_a();
  EXPECT_EQ(*eq.residual(4.0, 2.0), std::vector{0.0});
  EXPECT_EQ(*eq.residual(4.0, 3.0), std::vector{5.0});
}

// --- a cubic EoS -------------------------------------------------------------

constexpr std::size_t species = 4;

[[nodiscard]] std::vector<RE> fractions() {
  std::vector<RE> x;
  for (const std::size_t i : std::views::iota(0uz, species)) {
    x.push_back(ddx::rt::var(std::format("x{}", i)));
  }
  return x;
}

// What a caller differentiates: something of Z and of x directly.
[[nodiscard]] RE output(const std::vector<RE> &x, const RE &z) {
  return log(z) * x[0] + z * z * x[1] + x[2];
}

// The residual with Z a plain symbol, for the test's own Newton solve.
[[nodiscard]] auto cubic() {
  return ddx::rt::equation([] {
    const auto x = fractions();
    const auto z = ddx::rt::var("Z");
    return models::peng_robinson(x, z);
  });
}

[[nodiscard]] auto with_implicit_z() {
  return ddx::rt::equation([] {
    const auto x = fractions();
    const auto z = ddx::rt::implicit(
        "Z", [&](const RE &z) { return models::peng_robinson(x, z); });
    return output(x, z);
  });
}

// Newton from `z`, on the residual's own value and dF/dZ.
template <typename Cubic>
[[nodiscard]] double solve(const Cubic &f, std::vector<double> x, double z) {
  x.push_back(z);
  for (int it = 0; it < 100; ++it) {
    const double r = *f.evaluate(x);
    const double slope = (*f.jacobian(x))[species];
    x.back() -= r / slope;
    if (std::abs(r) < 1e-15) {
      break;
    }
  }
  return x.back();
}

[[nodiscard]] std::vector<double> with(std::vector<double> x, double z) {
  x.push_back(z);
  return x;
}

TEST(RtImplicit, ACubicAgainstDifferencesThatReSolveIt) {
  const auto f = cubic();
  const auto eq = with_implicit_z();
  ASSERT_FALSE(eq.poisoned()) << eq.status()->code;
  const std::vector<double> x{0.3, 0.25, 0.25, 0.2};
  const double z = solve(f, x, 1.0);
  ASSERT_NEAR((*eq.residual(with(x, z)))[0], 0.0, 1e-13);

  const auto g = *eq.gradient(with(x, z));
  const auto H = *eq.hessian(with(x, z));
  constexpr double h = 1e-6;
  for (const std::size_t j : std::views::iota(0uz, species)) {
    auto ahead = x;
    auto behind = x;
    ahead[j] += h;
    behind[j] -= h;
    const double za = solve(f, ahead, z);
    const double zb = solve(f, behind, z);
    const double fd =
        (*eq.evaluate(with(ahead, za)) - *eq.evaluate(with(behind, zb))) /
        (2 * h);
    EXPECT_NEAR(g[j], fd, 1e-6 * std::max(1.0, std::abs(fd))) << "d/dx" << j;

    // The Hessian column along x_j, from the gradient at the re-solved roots.
    const auto ga = *eq.gradient(with(ahead, za));
    const auto gb = *eq.gradient(with(behind, zb));
    for (const std::size_t i : std::views::iota(0uz, species)) {
      const double dd = (ga[i] - gb[i]) / (2 * h);
      EXPECT_NEAR(H[i * species + j], dd, 1e-5 * std::max(1.0, std::abs(dd)))
          << "d2/dx" << i << "dx" << j;
    }
  }
}

TEST(RtImplicit, ABatchCarriesAColumnOfRoots) {
  const auto f = cubic();
  const auto eq = with_implicit_z();
  constexpr std::size_t n = 64;
  std::vector<std::vector<double>> columns(species + 1, std::vector<double>(n));
  for (const std::size_t i : std::views::iota(0uz, n)) {
    const double t = static_cast<double>(i) / n;
    const std::vector<double> x{0.3 + 0.1 * t, 0.25, 0.25 - 0.1 * t, 0.2};
    const double z = solve(f, x, 1.0);
    for (const std::size_t j : std::views::iota(0uz, species)) {
      columns[j][i] = x[j];
    }
    columns[species][i] = z;
  }
  const auto in =
      columns | std::views::transform([](const auto &c) { return c.data(); }) |
      ddx::impl::to<std::vector<const double *>>();
  std::vector<double> values(n);
  std::vector<std::vector<double>> g(*eq.jacobian_columns(),
                                     std::vector<double>(n));
  const std::array<double *, 1> fs{values.data()};
  const auto gs = g | std::views::transform([](auto &c) { return c.data(); }) |
                  ddx::impl::to<std::vector<double *>>();
  ASSERT_TRUE(eq.jacobian(in, fs, gs, n).has_value());
  for (const std::size_t i : std::views::iota(0uz, n)) {
    std::vector<double> at;
    for (const auto &column : columns) {
      at.push_back(column[i]);
    }
    const auto J = *eq.jacobian(at);
    for (const auto [cell, j] :
         std::views::enumerate((*eq.jacobian_pattern()->get().begin()).cols)) {
      EXPECT_EQ(g[static_cast<std::size_t>(cell)][i], J[j]) << "point " << i;
    }
  }
}

TEST(RtImplicit, ZIsPartOfThePoint) {
  const auto eq = root_of_a();
  EXPECT_EQ(eq.gradient("a"_s = 4.0).error().code, ddx::errc::short_point);
  EXPECT_EQ(eq.gradient(4.0).error().code, ddx::errc::wrong_arity);
}

// --- refusals ----------------------------------------------------------------

TEST(RtImplicit, APoisonedResidualIsRefused) {
  const auto eq = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    return ddx::rt::implicit(
        "z", [&](const RE &z) { return z * ddx::rt::param("a") - a; });
  });
  ASSERT_TRUE(eq.poisoned());
  EXPECT_EQ(eq.status()->code, ddx::errc::name_clash);
}

TEST(RtImplicit, NamingTwiceIsRefused) {
  const auto eq = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto z =
        ddx::rt::implicit("z", [&](const RE &z) { return z * z - a; });
    const auto again =
        ddx::rt::implicit("z", [&](const RE &z) { return z - a; });
    return z + again;
  });
  ASSERT_TRUE(eq.poisoned());
  EXPECT_EQ(eq.status()->code, ddx::errc::closed_twice);
}

TEST(RtImplicit, AResidualOverAnotherImplicitIsRefused) {
  const auto eq = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto y = ddx::rt::implicit("y", [&](const RE &y) { return y - a; });
    const auto z =
        ddx::rt::implicit("z", [&](const RE &z) { return z * y - a; });
    return z;
  });
  ASSERT_TRUE(eq.poisoned());
  EXPECT_EQ(eq.status()->code, ddx::errc::nested_implicit);
}

TEST(RtImplicit, AnImplicitWithNoResidualIsRefused) {
  const auto eq = ddx::rt::equation([] {
    return ddx::rt::var("a") * ddx::rt::named(ddx::rt::OpCode::Implicit, "z");
  });
  ASSERT_TRUE(eq.poisoned());
  EXPECT_EQ(eq.status()->code, ddx::errc::unclosed_implicit);
}

TEST(RtImplicit, AnImplicitNamedLikeASymbolIsRefused) {
  const auto eq = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    return a * ddx::rt::implicit("a", [&](const RE &z) { return z - a; });
  });
  ASSERT_TRUE(eq.poisoned());
  EXPECT_EQ(eq.status()->code, ddx::errc::name_clash);
}

// An output that never reaches z differentiates as if it were not there.
TEST(RtImplicit, AnOutputNotThroughZIsUnaffected) {
  const auto eq = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    (void)ddx::rt::implicit("z", [&](const RE &z) { return z * z - a; });
    return a * a;
  });
  ASSERT_FALSE(eq.poisoned());
  EXPECT_EQ(*eq.gradient(3.0, 1.0), std::vector{6.0});
  EXPECT_EQ(*eq.hessian(3.0, 1.0), std::vector{2.0});
}

TEST(RtImplicit, ASystemOfOutputsThroughZ) {
  const auto sys = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto z =
        ddx::rt::implicit("z", [&](const RE &z) { return z * z - a; });
    return std::vector<RE>{z, z * a};
  });
  // d(z a)/da = z + a dz/da = 2 + 4 * 0.25.
  const auto J = *sys.jacobian(4.0, 2.0);
  EXPECT_NEAR(J[0], 0.25, 1e-15);
  EXPECT_NEAR(J[1], 3.0, 1e-15);
}

} // namespace
