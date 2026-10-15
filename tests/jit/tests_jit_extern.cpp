#include "rt/equation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <ranges>
#include <span>
#include <vector>

// A compiled kernel calls the program's function back through a trampoline,
// one call per lane, and answers what the sweep does.

namespace {

void g_of(std::span<const double> in, std::span<double> out) {
  const double a = in[0];
  const double b = in[1];
  const double e = std::exp(a * b);
  out[0] = e * std::sin(b);
  out[1] = b * e * std::sin(b);
  out[2] = e * (a * std::sin(b) + std::cos(b));
  out[3] = b * b * e * std::sin(b);
  out[4] = e * (std::sin(b) + a * b * std::sin(b) + b * std::cos(b));
  out[5] = e * (a * a * std::sin(b) + 2 * a * std::cos(b) - std::sin(b));
}

[[nodiscard]] auto model() {
  return ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    const auto g = ddx::rt::extern_fn({.name = "g", .arity = 2}, g_of)(a, b);
    return log(1.0 + g * g) + a * b;
  });
}

TEST(JitExtern, AKernelCallsTheFunctionAndAgreesWithTheSweep) {
  const auto swept = model();
  auto compiled = model();
  compiled.options({.backend = ddx::jit::Backend::Compile});
  ASSERT_TRUE(compiled.wait_for_kernel());
  ASSERT_TRUE(compiled.uses_kernel());

  // Enough points for the wide loop and a ragged tail.
  constexpr std::size_t n = 37;
  std::vector<double> as(n);
  std::vector<double> bs(n);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    as[i] = -0.5 + 0.03 * static_cast<double>(i);
    bs[i] = 0.2 + 0.05 * static_cast<double>(i);
  }
  const std::array<const double *, 2> in{as.data(), bs.data()};
  const auto run = [&](const auto &eq) {
    std::vector<double> f(n);
    std::vector<double> ga(n);
    std::vector<double> gb(n);
    const std::array<double *, 1> fs{f.data()};
    const std::array<double *, 2> gs{ga.data(), gb.data()};
    EXPECT_TRUE(eq.jacobian(in, fs, gs, n).has_value());
    return std::array{f, ga, gb};
  };
  const auto kernel = run(compiled);
  const auto sweep = run(swept);
  for (const auto [k, s] : std::views::zip(kernel, sweep)) {
    for (const auto [x, y] : std::views::zip(k, s)) {
      EXPECT_NEAR(x, y, 1e-14 * std::max(1.0, std::abs(y)));
    }
  }

  // The Hessian lane compiles from the swept graph, so it matches exactly.
  compiled.options({.backend = ddx::jit::Backend::Compile, .points = 1});
  ASSERT_TRUE(compiled.wait_for_kernel(ddx::rt::Want::Hessian));
  EXPECT_EQ(*compiled.hessian(0.3, 1.1), *swept.hessian(0.3, 1.1));
}

} // namespace
