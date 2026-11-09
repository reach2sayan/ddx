#include "energy_models.hpp"
#include "rt/equation.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <latch>
#include <ranges>
#include <string>
#include <thread>
#include <vector>

// A system's output count is a run-time number, and every sweep past the
// Jacobian is built only when asked for.
namespace {

using RE = ddx::rt::Expression<>;

// Five functions of three symbols, like a model whose count comes from data.
[[nodiscard]] std::vector<RE> five(const RE &x, const RE &y, const RE &z) {
  return {x * y, sin(x) + z, exp(y * z), x * x * z, log(y) + x};
}

[[nodiscard]] auto five_system() {
  return ddx::rt::equation([] {
    return five(ddx::rt::var("x"), ddx::rt::var("y"), ddx::rt::var("z"));
  });
}

// Function k alone, as a one-output Equation over the same model.
[[nodiscard]] auto one_of_five(std::size_t k) {
  return ddx::rt::equation([k] {
    return five(ddx::rt::var("x"), ddx::rt::var("y"), ddx::rt::var("z"))[k];
  });
}

constexpr std::array system_point{0.7, 1.3, -0.4};

TEST(RtSystem, JacobianStacksTheScalarRows) {
  const auto sys = five_system();
  ASSERT_FALSE(sys.poisoned());
  ASSERT_EQ(sys.outputs(), 5u);
  EXPECT_EQ(sys.value_columns(), 5u);
  const auto J = *sys.jacobian(system_point);
  const auto f = *sys.evaluate(system_point);
  ASSERT_EQ(J.size(), 15u);
  ASSERT_EQ(f.size(), 5u);

  const auto &pattern = sys.jacobian_pattern()->get();
  for (const std::size_t k : std::views::iota(0uz, 5uz)) {
    const auto eq = one_of_five(k);
    EXPECT_DOUBLE_EQ(f[k], *eq.evaluate(system_point));
    const auto row = *eq.jacobian(system_point);
    for (const std::size_t j : std::views::iota(0uz, 3uz)) {
      EXPECT_DOUBLE_EQ(J[k * 3 + j], row[j]) << "d f" << k << " / d x" << j;
    }
    EXPECT_TRUE(std::ranges::equal(pattern.rows()[k],
                                   eq.jacobian_pattern()->get().rows()[0]))
        << "row " << k;
  }
}

TEST(RtSystem, BatchJacobianMatchesThePointCalls) {
  const auto sys = five_system();
  constexpr std::size_t n = 17;
  const std::size_t cells = *sys.jacobian_columns();
  std::array<std::vector<double>, 3> xs;
  for (const auto [j, column] : std::views::enumerate(xs)) {
    column.resize(n);
    for (const std::size_t i : std::views::iota(0uz, n)) {
      column[i] =
          0.5 + 0.1 * static_cast<double>(i) + 0.3 * static_cast<double>(j);
    }
  }
  std::vector<std::vector<double>> f(5, std::vector<double>(n));
  std::vector<std::vector<double>> g(cells, std::vector<double>(n));
  const auto pointers = [](auto &columns) {
    return columns | std::views::transform([](auto &c) { return c.data(); }) |
           ddx::impl::to<std::vector>();
  };
  const auto in = pointers(xs) | std::views::transform([](double *p) {
                    return static_cast<const double *>(p);
                  }) |
                  ddx::impl::to<std::vector>();
  ASSERT_TRUE(sys.jacobian(in, pointers(f), pointers(g), n).has_value());

  for (const std::size_t i : std::views::iota(0uz, n)) {
    const std::array at{xs[0][i], xs[1][i], xs[2][i]};
    const auto values = *sys.evaluate(at);
    const auto J = *sys.jacobian(at);
    for (const std::size_t k : std::views::iota(0uz, 5uz)) {
      EXPECT_EQ(f[k][i], values[k]);
    }
    for (const auto [k, i_j, v] :
         sys.jacobian_pattern()->get().cells(std::views::iota(0uz, cells))) {
      EXPECT_EQ(g[v][i], J[k * 3 + i_j]);
    }
  }
}

TEST(RtSystem, OneOutputMembersRefuseASystem) {
  const auto sys = five_system();
  const std::array v{1.0, 0.0, 0.0};
  EXPECT_EQ(sys.hvp(std::span<const double>{v}, system_point).error().code,
            ddx::errc::not_single_output);
  EXPECT_EQ(sys.hvp("x", system_point).error().code,
            ddx::errc::not_single_output);
  EXPECT_EQ(sys.univariate_derivative<1>(1.0).error().code,
            ddx::errc::not_single_output);
  EXPECT_EQ(sys.prepare({ddx::rt::Want::Hessian}).error().code,
            ddx::errc::not_single_output);
  EXPECT_FALSE(sys.hessian_columns().has_value());
  EXPECT_FALSE(sys.hessian_colours().has_value());
  EXPECT_FALSE(sys.hessian_cell(0, 0).has_value());
  EXPECT_TRUE(
      sys.vjp(std::span<const double>{std::array{1.0, 0.0, 0.0, 0.0, 0.0}},
              system_point)
          .has_value());
}

TEST(RtSystem, HessianStacksTheScalarHessians) {
  const auto sys = five_system();
  const auto H = *sys.hessian(system_point);
  ASSERT_EQ(H.size(), 5u * 3u * 3u);
  for (const std::size_t k : std::views::iota(0uz, 5uz)) {
    const auto h = *one_of_five(k).hessian(system_point);
    for (const std::size_t c : std::views::iota(0uz, 9uz)) {
      EXPECT_NEAR(H[k * 9 + c], h[c], 1e-14 * std::max(1.0, std::abs(h[c])))
          << "output " << k << " cell " << c;
    }
  }
}

// A system's hessian() walks the arena and contracts as the equation says, so
// with contraction off each block matches the one-output batch path to the bit.
TEST(RtSystem, AHessianAtAPointContractsAsTheEquationSays) {
  for (const bool contract : {true, false}) {
    ddx::jit::Options opt;
    opt.codegen.contract = contract;
    auto sys = five_system();
    sys.options(opt);
    const auto H = *sys.hessian(system_point);
    for (const std::size_t k : std::views::iota(0uz, 5uz)) {
      auto one = one_of_five(k);
      one.options(opt);
      const auto h = *one.hessian(system_point);
      for (const std::size_t c : std::views::iota(0uz, 9uz)) {
        EXPECT_EQ(H[k * 9 + c], h[c])
            << "contract " << contract << " output " << k << " cell " << c;
      }
    }
  }
}

// A literal output is constant wherever it stands, so the arena comes from the
// first output that names one.
TEST(RtSystem, ALiteralFirstOutputTakesTheArenaOfTheNext) {
  const auto first = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return std::vector{RE{2.5}, x * x};
  });
  const auto last = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return std::vector{x * x, RE{2.5}};
  });
  ASSERT_FALSE(first.poisoned()) << first.status()->code;
  ASSERT_FALSE(last.poisoned()) << last.status()->code;
  const std::array at{3.0};
  EXPECT_EQ(*first.evaluate(at), (std::vector{2.5, 9.0}));
  EXPECT_EQ(*first.jacobian(at), (std::vector{0.0, 6.0}));
  EXPECT_EQ(*last.jacobian(at), (std::vector{6.0, 0.0}));
}

TEST(RtSystem, AnEmptyRangeIsNoGraph) {
  const auto sys = ddx::rt::equation([] { return std::vector<RE>{}; });
  ASSERT_TRUE(sys.poisoned());
  EXPECT_EQ(sys.status()->code, ddx::errc::no_graph);
  EXPECT_FALSE(sys.outputs().has_value());
}

// Construction sweeps only the Jacobian; a coloured Hessian at n = 128 takes
// minutes in a debug build, so finishing is the check.
constexpr std::size_t many_species = 128;

TEST(RtSystem, BuildingSweepsOnlyTheJacobian) {
  const auto start = std::chrono::steady_clock::now();
  const auto eq = ddx::rt::equation([] {
    const auto x = std::views::iota(0uz, many_species) |
                   std::views::transform([](std::size_t i) {
                     return ddx::rt::var(std::format("x{:03}", i));
                   }) |
                   ddx::impl::to<std::vector>();
    return models::mse(x);
  });
  const std::chrono::duration<double, std::milli> built =
      std::chrono::steady_clock::now() - start;
  ASSERT_FALSE(eq.poisoned());
  EXPECT_EQ(eq.jacobian_columns(), many_species);
  std::printf("[ ddx ] n = %zu built in %.1f ms\n", many_species,
              built.count());
}

// prepare() builds the lane so the threads only read it. The latch forces the
// threads to overlap rather than hoping they do.
TEST(RtSystem, PreparedLanesServeManyThreads) {
  const auto sys = five_system();
  ASSERT_TRUE(sys.prepare({ddx::rt::Want::Jacobian}).has_value());
  constexpr std::size_t n = 9;
  const std::size_t cells = *sys.jacobian_columns();

  constexpr std::size_t threads = 8;
  std::latch start{threads};
  const auto run = [&](std::size_t salt, bool together) {
    std::array<std::vector<double>, 3> xs;
    for (const auto [j, column] : std::views::enumerate(xs)) {
      column.assign(n, 0.4 + 0.05 * static_cast<double>(salt) +
                           0.2 * static_cast<double>(j));
    }
    std::vector<std::vector<double>> f(5, std::vector<double>(n));
    std::vector<std::vector<double>> g(cells, std::vector<double>(n));
    std::array<const double *, 3> in{xs[0].data(), xs[1].data(), xs[2].data()};
    auto fs = f | std::views::transform([](auto &c) { return c.data(); }) |
              ddx::impl::to<std::vector>();
    auto gs = g | std::views::transform([](auto &c) { return c.data(); }) |
              ddx::impl::to<std::vector>();
    if (together) {
      start.arrive_and_wait();
    }
    for ([[maybe_unused]] const int rep : std::views::iota(0, 1000)) {
      EXPECT_TRUE(sys.jacobian(in, fs, gs, n).has_value());
    }
    return std::pair{f, g};
  };

  std::vector<std::pair<std::vector<std::vector<double>>,
                        std::vector<std::vector<double>>>>
      got(threads);
  {
    std::vector<std::jthread> pool;
    for (const std::size_t t : std::views::iota(0uz, threads)) {
      pool.emplace_back([&, t] { got[t] = run(t, true); });
    }
  }
  for (const std::size_t t : std::views::iota(0uz, threads)) {
    EXPECT_EQ(got[t], run(t, false)) << "thread " << t;
  }
}

} // namespace
