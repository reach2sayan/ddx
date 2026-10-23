#include "ddx.hpp"
#include "jit/kernel.hpp"
#include "jit/tests_jit_common.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/equation.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "util/ranges.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <ranges>
#include <vector>

// A kernel computes exactly the outputs its graph was frozen with, so freezing
// {value, partials...} yields the Jacobian columns.

namespace {
using ddx::rt::Builder;
using ddx::rt::Graph;

void expect_jacobian_matches_interpreter(auto build, std::size_t nvars,
                                         std::size_t n = 32) {
  Builder<> b;
  std::vector<ddx::rt::Expression<>> vars;
  static constexpr const char *names[] = {"x", "y", "z"};
  for (std::size_t i = 0; i < nvars; ++i) {
    vars.push_back(var(b, names[i]));
  }
  const auto root = build(b, vars);
  // The sweep is repeated for the reference values; the builder runs its own.
  const auto reference_jacobian = ddx::rt::derive_jacobian(b, root.id(b));
  const auto kernel = must_compile(
      ddx::rt::GraphBuilder{b}.value(root).build_jacobian().finish());
  ASSERT_EQ(kernel.shape().outputs(), nvars + 1);

  std::vector<std::vector<double>> columns(nvars, std::vector<double>(n));
  for (std::size_t j = 0; j < nvars; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      columns[j][i] = 0.2 + 0.5 * static_cast<double>((i + 2 * j) % 7) / 7.0;
    }
  }
  std::vector<const double *> xs(nvars);
  for (std::size_t j = 0; j < nvars; ++j) {
    xs[j] = columns[j].data();
  }

  std::vector<double> value(n);
  std::vector<std::vector<double>> grad(nvars, std::vector<double>(n));
  std::vector<double *> gp(nvars);
  for (std::size_t j = 0; j < nvars; ++j) {
    gp[j] = grad[j].data();
  }
  double *const value_columns[]{value.data()};
  kernel(xs, value_columns, gp, {}, n);

  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> point(nvars);
    for (std::size_t j = 0; j < nvars; ++j) {
      point[j] = columns[j][i];
    }
    const auto reference = ddx::rt::evaluate_all(b, point);
    EXPECT_NEAR(value[i], reference[reference_jacobian.value], 1e-12);
    for (std::size_t j = 0; j < nvars; ++j) {
      const double want = reference[reference_jacobian.partial[j]];
      EXPECT_NEAR(grad[j][i], want, 1e-12 * std::max(1.0, std::abs(want)))
          << "d/d" << names[j] << " at point " << i;
    }
  }
}

TEST(JitJacobian, MatchesTheInterpreter) {
  expect_jacobian_matches_interpreter(
      [](Builder<> &, auto &v) { return exp(v[0]) * sin(v[1]); }, 2);
  expect_jacobian_matches_interpreter(
      [](Builder<> &, auto &v) { return v[0] / v[1]; }, 2);
  expect_jacobian_matches_interpreter(
      [](Builder<> &, auto &v) { return pow(v[0], v[1]); }, 2);
  expect_jacobian_matches_interpreter(
      [](Builder<> &, auto &v) { return log(v[0]) * sqrt(v[1]) + tanh(v[2]); },
      3);
  expect_jacobian_matches_interpreter(
      [](Builder<> &, auto &v) { return erf(v[0]) + cbrt(v[1]); }, 2);
  // abs and max/min put sign nodes in the Jacobian graph, which codegen emits
  // as compare-and-select rather than a libm call.
  expect_jacobian_matches_interpreter(
      [](Builder<> &, auto &v) { return abs(v[0] - v[1]) * v[1]; }, 2);
  expect_jacobian_matches_interpreter(
      [](Builder<> &, auto &v) {
        return max(v[0] * v[0], v[1]) + min(v[0], v[1]);
      },
      2);
}

TEST(JitJacobian, MatchesDdxThroughTheBridge) {
  constexpr auto x = ddx::var<"x">;
  constexpr auto y = ddx::var<"y">;
  const auto f = exp(x) * sin(y) + x * y;

  Builder<> b;
  const auto root = ddx::rt::to_graph(b, f);
  const auto kernel = must_compile(
      ddx::rt::GraphBuilder{b}.value(root).build_jacobian().finish());

  const std::array cx{1.0, 0.25, 2.5};
  const std::array cy{2.0, 1.75, 0.5};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};
  std::array<double, 3> value{};
  std::array<double, 3> dx{};
  std::array<double, 3> dy{};
  std::array<double *, 2> gp{dx.data(), dy.data()};
  double *const value_columns[]{value.data()};
  kernel(xs, value_columns, gp, {}, value.size());

  for (std::size_t i = 0; i < value.size(); ++i) {
    const auto expected = ddx::Equation{f}.jacobian(cx[i], cy[i]);
    EXPECT_NEAR(value[i], ddx::Equation{f}.evaluate(cx[i], cy[i]), 1e-12);
    EXPECT_NEAR(dx[i], expected[0], 1e-12);
    EXPECT_NEAR(dy[i], expected[1], 1e-12);
  }
}

} // namespace

namespace {

// The Hessian arrives compressed by colour, so the kernel reports its column
// count rather than leaving the caller to derive it from n.
TEST(JitHessian, ThirdBlockCarriesTheColouredHessian) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y) + x * x * y;

  const auto graph = ddx::rt::GraphBuilder{b}
                         .value(f)
                         .build_jacobian()
                         .build_hessian()
                         .finish();
  const auto kernel = must_compile(graph);

  ASSERT_EQ(kernel.shape().values, 1u);
  ASSERT_EQ(kernel.shape().jacobian, 2u);
  ASSERT_EQ(kernel.shape().hessian, graph.layout().hessian);
  ASSERT_GT(kernel.shape().hessian, 0u);

  constexpr std::size_t n = 4;
  std::array<double, n> cx{0.7, 0.8, 0.9, 1.0};
  std::array<double, n> cy{1.1, 1.2, 1.3, 1.4};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};

  std::array<double, n> value{};
  std::vector<std::vector<double>> jac(2, std::vector<double>(n));
  std::vector<std::vector<double>> hess(kernel.shape().hessian,
                                        std::vector<double>(n));
  double *const values[]{value.data()};
  const auto columns_of = [](auto &blocks) {
    return blocks | std::views::transform([](auto &c) { return c.data(); }) |
           ddx::impl::to<std::vector<double *>>();
  };
  auto jp = columns_of(jac);
  auto hp = columns_of(hess);

  kernel(xs, values, jp, hp, n);

  // Against the interpreter over the same graph, scattered the same way.
  const auto &colouring = graph.colouring();
  for (std::size_t point = 0; point < n; ++point) {
    const std::array<double, 2> at{cx[point], cy[point]};
    const auto reference = ddx::rt::evaluate_all(b, at);
    const auto outputs = graph.outputs();

    EXPECT_NEAR(value[point], reference[outputs[0]], 1e-12);
    for (std::size_t j = 0; j < 2; ++j) {
      EXPECT_NEAR(jac[j][point], reference[outputs[1 + j]], 1e-12);
    }
    for (std::size_t c = 0; c < kernel.shape().hessian; ++c) {
      EXPECT_NEAR(hess[c][point], reference[outputs[3 + c]], 1e-12)
          << "compressed Hessian column " << c;
    }
  }
  EXPECT_EQ(colouring.count() * 2, kernel.shape().hessian);
}

// The kernel is lowered from the rebalanced roots whatever the arena holds
// besides: a dense Hessian swept first leaves tensor nodes no scalar root
// reads, and the value kernel keeps the bits of one swept without it.
TEST(JitJacobian, ADenseHessianSweptFirstLeavesTheKernelsBits) {
  const auto build = [] {
    return ddx::rt::equation([] {
      const auto x = ddx::rt::var("x");
      const auto y = ddx::rt::var("y");
      ddx::rt::Expression<> f = x * y;
      for (int i = 1; i <= 64; ++i) {
        f = f + sin(x * (1.0 + 0.1 * i)) * exp(y * (i % 7)) / i;
      }
      return f;
    });
  };
  auto fresh = build();
  auto swept = build();
  ASSERT_TRUE(swept.dense_hessian(std::array{0.7, 1.3}).has_value());
  for (auto *eq : {&fresh, &swept}) {
    eq->options({.backend = ddx::jit::Backend::Compile, .points = 1});
    ASSERT_TRUE(eq->wait_for_kernel(ddx::rt::Want::Value));
  }
  for (const double x : {0.3, 0.7, 1.1, 2.9, 4.2}) {
    const std::array p{x, 1.3};
    EXPECT_EQ(*swept.evaluate(p), *fresh.evaluate(p)) << "x = " << x;
  }
}

} // namespace
