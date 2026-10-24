#include "rt/equation.hpp"
#include "rt/tensor.hpp"

#include <gtest/gtest.h>

#include <ranges>
#include <span>
#include <vector>

// No kernel is lowered over a tensor row: the compile is refused up front, the
// lane keeps sweeping, and the answers are the sweep's.
namespace {
using RV = ddx::rt::Expression<>;

auto quadratic_form() {
  return ddx::rt::equation([] {
    const RV x = ddx::rt::vec("x", 4);
    const std::vector<double> m{1.0, 0.5, -0.25, 2.0, 0.0, 1.5,  0.75, -1.0,
                                0.3, 0.6, 0.9,   1.2, 2.1, -0.4, 0.8,  0.1};
    return dot(x, gemv(ddx::rt::mat_constant(4, 4, m), x)) + sum(log(x));
  });
}
} // namespace

TEST(JitTensor, RefusesTheCompileAndSweepsInstead) {
  const auto swept = quadratic_form();
  auto compiled = quadratic_form();
  compiled.options({.backend = ddx::jit::Backend::Compile});
  // The refusal lands like a kernel would: waited for, then not there.
  EXPECT_FALSE(compiled.wait_for_kernel());
  EXPECT_FALSE(compiled.uses_kernel());
  EXPECT_FALSE(compiled.kernel_level().has_value());

  constexpr std::size_t n = 37;
  std::vector<std::vector<double>> xs(4, std::vector<double>(n));
  for (const std::size_t p : std::views::iota(0uz, n)) {
    for (std::size_t i = 0; i < 4; ++i) {
      xs[i][p] = 0.2 + 0.03 * static_cast<double>(p + i);
    }
  }
  for (const std::size_t p : std::views::iota(0uz, n)) {
    const std::vector<double> at{xs[0][p], xs[1][p], xs[2][p], xs[3][p]};
    EXPECT_EQ(*compiled.evaluate(std::span<const double>{at}),
              *swept.evaluate(std::span<const double>{at}));
    EXPECT_EQ(*compiled.gradient(std::span<const double>{at}),
              *swept.gradient(std::span<const double>{at}));
  }
}
