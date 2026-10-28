#include "ddx.hpp"
#include "jit/kernel.hpp"
#include "rt/equation.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <ranges>
#include <vector>

// A parameter's column sits between the symbols' and the seeds', and the kernel
// has to read the same column the sweep does -- the wrong one is silent wrong
// arithmetic, not a crash.

namespace {

TEST(JitParam, ColumnsAreSymbolsThenParamsThenSeeds) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto a = ddx::rt::param(b, "a");
  const auto f = a * x * x + exp(x) * seed(b, 0);
  const auto graph = ddx::rt::GraphBuilder{b}.value(f).finish();
  ASSERT_EQ(graph.arity(), 3u);

  auto compiler = ddx::jit::Compiler::create();
  ASSERT_TRUE(compiler.has_value());
  const auto kernel = compiler->compile(graph);
  ASSERT_TRUE(kernel.has_value()) << kernel.error().detail;

  static constexpr std::size_t n = 37;
  std::vector<std::vector<double>> columns(3, std::vector<double>(n));
  for (const auto [j, column] : columns | std::views::enumerate) {
    for (const auto [i, cell] : column | std::views::enumerate) {
      cell = 0.3 + 0.4 * static_cast<double>((i + 3 * j) % 5) / 5.0 +
             static_cast<double>(j);
    }
  }
  std::vector<const double *> xs;
  for (const auto &column : columns) {
    xs.push_back(column.data());
  }
  std::vector<double> got(n);
  double *const values[]{got.data()};
  (*kernel)(xs, values, {}, {}, n);

  for (const std::size_t i : std::views::iota(0uz, n)) {
    const double xv = columns[0][i];
    const double av = columns[1][i];
    const double v = columns[2][i];
    const auto want = ddx::rt::evaluate_all(b, std::vector{xv, av, v});
    EXPECT_EQ(got[i], want[graph.outputs()[0]]) << "at point " << i;
  }
}

TEST(JitParam, AnEquationWithAParamCompilesAndAgreesWithTheSweep) {
  const auto build = [] {
    return ddx::rt::equation([] {
      const auto x = ddx::rt::var("x");
      const auto y = ddx::rt::var("y");
      const auto k = ddx::rt::param("k");
      return k * sin(x * y) + exp(k * y);
    });
  };
  const auto swept = build();
  auto compiled = build();
  compiled.options({.backend = ddx::jit::Backend::Compile, .points = 1});
  ASSERT_TRUE(compiled.uses_kernel());
  ASSERT_TRUE(compiled.uses_kernel());

  for (const double k : {0.5, 2.0}) {
    const std::array at{0.7, 1.3, k};
    EXPECT_EQ(*compiled.jacobian(at), *swept.jacobian(at)) << "k = " << k;
  }
}

// A partial taken during construction is ordinary nodes: compiled, every
// derivative of it is the sweep's.
TEST(JitParam, ADerivativeBuiltIntoTheGraphCompiles) {
  const auto build = [] {
    return ddx::rt::equation([] {
      const auto T = ddx::rt::var("T");
      const auto Tr = ddx::rt::var("Tr");
      const auto ps = exp(1.0 - 2.0 / T);
      const auto h = Tr * Tr * log(ps) + sin(Tr) * ps;
      return T * T * ddx::rt::derivative(h, "Tr");
    });
  };
  const auto swept = build();
  auto compiled = build();
  compiled.options({.backend = ddx::jit::Backend::Compile, .points = 1});
  ASSERT_TRUE(compiled.uses_kernel());
  ASSERT_TRUE(compiled.uses_kernel(ddx::rt::Want::Hessian));
  const std::array at{1.3, 0.7};
  const std::array v{0.4, -1.1};
  EXPECT_EQ(*compiled.jacobian(at), *swept.jacobian(at));
  EXPECT_EQ(*compiled.hessian(at), *swept.hessian(at));
  EXPECT_EQ(*compiled.hvp(v, at), *swept.hvp(v, at));
}

} // namespace
