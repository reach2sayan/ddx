#include "jit/kernel.hpp"
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

// Two registered functions: the kernel's lanes are rebuilt over blocked spines,
// and every extern node has to come back naming its own function and
// component, not the first function's value.
TEST(JitExtern, EveryFunctionAndComponentSurvivesTheRebuild) {
  const auto h_of = [](std::span<const double> in, std::span<double> out) {
    out[0] = in[0] * in[1];
    out[1] = in[1];
    out[2] = in[0];
    out[3] = 0.0;
    out[4] = 1.0;
    out[5] = 0.0;
  };
  const auto model = [h_of] {
    return ddx::rt::equation([h_of] {
      const auto a = ddx::rt::var("a");
      const auto b = ddx::rt::var("b");
      const auto g = ddx::rt::extern_fn({.name = "g", .arity = 2}, g_of);
      const auto h = ddx::rt::extern_fn({.name = "h", .arity = 2}, h_of);
      auto spine = g(a, b) * h(b, a);
      for (int k = 1; k <= 20; ++k) {
        spine = spine + h(a, b) * static_cast<double>(k);
      }
      return spine;
    });
  };
  const auto swept = model();
  auto compiled = model();
  compiled.options({.backend = ddx::jit::Backend::Compile});
  ASSERT_TRUE(compiled.wait_for_kernel());
  ASSERT_TRUE(compiled.uses_kernel());
  // The spine is long enough to be blocked, so the last bits may differ; a
  // wrong function or component differs in the first.
  constexpr std::array<std::array<double, 2>, 3> at{
      {{0.3, 1.1}, {-0.7, 0.4}, {1.2, -2.0}}};
  for (const auto &[a, b] : at) {
    EXPECT_NEAR(*compiled.evaluate(a, b), *swept.evaluate(a, b),
                1e-12 * std::abs(*swept.evaluate(a, b)));
    const auto j = *swept.jacobian(a, b);
    for (const auto [x, y] : std::views::zip(*compiled.jacobian(a, b), j)) {
      EXPECT_NEAR(x, y, 1e-12 * std::max(1.0, std::abs(y)));
    }
  }
}

// The kernel holds the functions it calls and no address of its own: it runs
// after the graph it was lowered from is gone, and its object is kept like any
// other.
TEST(JitExtern, AKernelOutlivesItsGraphAndKeepsItsObject) {
  auto c = ddx::jit::Compiler::create();
  ASSERT_TRUE(c.has_value()) << c.error().detail;
  const auto kernel = [&] {
    ddx::rt::Arena<> b;
    const auto a = ddx::rt::var(b, "a");
    const auto bb = ddx::rt::var(b, "b");
    const auto g = ddx::rt::extern_fn(b, {.name = "g", .arity = 2}, g_of);
    const auto graph = ddx::rt::Graph<>::freeze(
        b, std::array{(g(a, bb) * 2.0).id(b)});
    auto k = c->compile(graph);
    EXPECT_TRUE(k.has_value()) << (k ? "" : k.error().detail);
    return k ? std::move(*k) : ddx::jit::Kernel{};
  }();
  ASSERT_TRUE(kernel);
  EXPECT_FALSE(kernel.object().empty()) << "an extern kernel kept no object";
  const std::array as{0.3, -0.7}, bs{1.1, 0.4};
  const std::array<const double *, 2> xs{as.data(), bs.data()};
  std::array<double, 2> f{};
  const std::array<double *, 1> fs{f.data()};
  kernel(xs, fs, {}, {}, 2);
  for (const auto [k, want] : std::views::zip(f, std::array{
                                                     2.0 * std::exp(0.3 * 1.1) * std::sin(1.1),
                                                     2.0 * std::exp(-0.7 * 0.4) * std::sin(0.4)})) {
    EXPECT_NEAR(k, want, 1e-14 * std::abs(want));
  }
}

} // namespace
