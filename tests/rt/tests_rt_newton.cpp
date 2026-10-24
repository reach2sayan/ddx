#include "ddx.hpp"
#include "energy_models.hpp"
#include "rt/equation.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

// Build once, differentiate once, iterate -- the case the runtime path exists
// for.  A slightly wrong Jacobian still evaluates but converges linearly, so
// quadratic convergence is the sharp test.

namespace {
using ddx::rt::Arena;
using models::RE;

// By hand: a linear solver would be more machinery than the thing under test.
std::array<double, 2> solve2(const std::vector<double> &J,
                             const std::array<double, 2> &rhs) {
  const double det = J[0] * J[3] - J[1] * J[2];
  return {(rhs[0] * J[3] - J[1] * rhs[1]) / det,
          (J[0] * rhs[1] - rhs[0] * J[2]) / det};
}

TEST(RtNewton, SolvesASystemBuiltAtRuntime) {
  //   x^2 + y^2 = 4
  //   x * y     = 1
  // so x^2 = 2 + sqrt(3), and the root is (1.93185..., 0.51763...).
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto system = ddx::rt::equation(x * x + y * y - 4.0, x * y - 1.0);

  std::array<double, 2> at{1.5, 1.0};
  int taken = 0;
  for (int iteration = 0; iteration < 40; ++iteration) {
    const auto residual = *system.evaluate(at[0], at[1]);
    if (std::hypot(residual[0], residual[1]) < 1e-14) {
      break;
    }
    const auto step =
        solve2(*system.jacobian(at[0], at[1]), {-residual[0], -residual[1]});
    at[0] += step[0];
    at[1] += step[1];
    ++taken;
  }

  const double expected_x = std::sqrt(2.0 + std::sqrt(3.0));
  EXPECT_NEAR(at[0], expected_x, 1e-12);
  EXPECT_NEAR(at[1], 1.0 / expected_x, 1e-12);
  // Quadratic convergence: a wrong Jacobian needs far more iterations.
  EXPECT_LE(taken, 8);
}

// Peng-Robinson's cubic solved for Z at a composition read from "data" -- what
// a flash calculation does.
TEST(RtNewton, SolvesPengRobinsonForCompressibility) {
  constexpr std::size_t species = 6;
  Arena<> b;
  std::vector<RE> composition;
  for (std::size_t i = 0; i < species; ++i) {
    composition.push_back(var(b, "x" + std::to_string(i)));
  }
  const auto Z = var(b, "Z");
  const auto cubic = ddx::rt::equation(models::peng_robinson(composition, Z));

  ASSERT_EQ(cubic.arity(), species + 1);

  const auto names = *cubic.symbols();
  const std::size_t z_slot =
      static_cast<std::size_t>(std::ranges::find(names, "Z") - names.begin());

  std::vector<double> at(species + 1, 1.0 / species);
  at[z_slot] = 0.9; // Z, the unknown

  int taken = 0;
  for (int iteration = 0; iteration < 60; ++iteration) {
    const double residual = *cubic.evaluate(at);
    if (std::abs(residual) < 1e-14) {
      break;
    }
    // Only dF/dZ is wanted; the composition is held fixed.
    at[z_slot] -= residual / (*cubic.jacobian(at))[z_slot];
    ++taken;
  }

  EXPECT_LT(std::abs(*cubic.evaluate(at)), 1e-12) << "did not converge";
  EXPECT_LE(taken, 12);
  EXPECT_GT(at[z_slot], 0.0) << "a compressibility factor is positive";
}

// Build once, solve many: one Equation re-used across a parameter sweep.
TEST(RtNewton, OneEquationServesAWholeSweep) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto c = var(b, "c");
  const auto f = ddx::rt::equation(x * x - c); // root: x = sqrt(c)

  const auto names = *f.symbols();
  const std::size_t z_slot =
      static_cast<std::size_t>(std::ranges::find(names, "x") - names.begin());

  for (double target = 0.5; target < 8.0; target += 0.37) {
    std::vector<double> at(2, 1.0);
    at[z_slot] = 1.0;
    at[1 - z_slot] = target;

    for (int iteration = 0; iteration < 40; ++iteration) {
      const double residual = *f.evaluate(at);
      if (std::abs(residual) < 1e-15) {
        break;
      }
      at[z_slot] -= residual / (*f.jacobian(at))[z_slot];
    }
    EXPECT_NEAR(at[z_slot], std::sqrt(target), 1e-12) << "c=" << target;
  }
}

} // namespace
