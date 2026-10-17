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
// Jacobian waits to be asked for.
namespace {

using RE = ddx::rt::Expression<>;

// Five functions of three symbols, as a model whose count comes from data.
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

constexpr std::array point{0.7, 1.3, -0.4};

TEST(RtSystem, JacobianStacksTheScalarRows) {
  const auto sys = five_system();
  ASSERT_FALSE(sys.poisoned());
  ASSERT_EQ(sys.outputs(), 5u);
  EXPECT_EQ(sys.value_columns(), 5u);
  const auto J = *sys.jacobian(point);
  const auto f = *sys.evaluate(point);
  ASSERT_EQ(J.size(), 15u);
  ASSERT_EQ(f.size(), 5u);

  const auto &pattern = sys.jacobian_pattern()->get();
  for (const std::size_t k : std::views::iota(0uz, 5uz)) {
    const auto eq = one_of_five(k);
    EXPECT_DOUBLE_EQ(f[k], *eq.evaluate(point));
    const auto row = *eq.jacobian(point);
    for (const std::size_t j : std::views::iota(0uz, 3uz)) {
      EXPECT_DOUBLE_EQ(J[k * 3 + j], row[j]) << "d f" << k << " / d x" << j;
    }
    EXPECT_TRUE(std::ranges::equal(
        *std::ranges::next(pattern.begin(), static_cast<std::ptrdiff_t>(k)),
        *eq.jacobian_pattern()->get().begin()))
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
           ddx::impl::to<std::vector<double *>>();
  };
  const auto in = pointers(xs) | std::views::transform([](double *p) {
                    return static_cast<const double *>(p);
                  }) |
                  ddx::impl::to<std::vector<const double *>>();
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

// The one-output members refuse a system rather than answer for output 0.
TEST(RtSystem, OneOutputMembersRefuseASystem) {
  const auto sys = five_system();
  const std::array v{1.0, 0.0, 0.0};
  EXPECT_EQ(sys.hvp(std::span<const double>{v}, point).error().code,
            ddx::errc::not_single_output);
  EXPECT_EQ(sys.hvp("x", point).error().code, ddx::errc::not_single_output);
  EXPECT_EQ(sys.univariate_derivative<1>(1.0).error().code,
            ddx::errc::not_single_output);
  EXPECT_EQ(sys.prepare({ddx::rt::Want::Hessian}).error().code,
            ddx::errc::not_single_output);
  EXPECT_FALSE(sys.hessian_columns().has_value());
  EXPECT_FALSE(sys.hessian_colours().has_value());
  EXPECT_FALSE(sys.hessian_cell(0, 0).has_value());
  EXPECT_TRUE(
      sys.vjp(std::span<const double>{std::array{1.0, 0.0, 0.0, 0.0, 0.0}},
              point)
          .has_value());
}

// m x n x n off the arena, each block the scalar equation's Hessian.
TEST(RtSystem, HessianStacksTheScalarHessians) {
  const auto sys = five_system();
  const auto H = *sys.hessian(point);
  ASSERT_EQ(H.size(), 5u * 3u * 3u);
  for (const std::size_t k : std::views::iota(0uz, 5uz)) {
    const auto h = *one_of_five(k).hessian(point);
    for (const std::size_t c : std::views::iota(0uz, 9uz)) {
      EXPECT_NEAR(H[k * 9 + c], h[c], 1e-14 * std::max(1.0, std::abs(h[c])))
          << "output " << k << " cell " << c;
    }
  }
}

TEST(RtSystem, AnEmptyRangeIsNoGraph) {
  const auto sys = ddx::rt::equation([] { return std::vector<RE>{}; });
  ASSERT_TRUE(sys.poisoned());
  EXPECT_EQ(sys.status()->code, ddx::errc::no_graph);
  EXPECT_FALSE(sys.outputs().has_value());
}

// Construction sweeps the Jacobian and nothing else: at n = 128 the coloured
// Hessian the constructor used to sweep is minutes in a debug build, so this
// finishing at all is the check.  The time is printed, not asserted.
constexpr std::size_t species = 128;

TEST(RtSystem, BuildingSweepsOnlyTheJacobian) {
  const auto start = std::chrono::steady_clock::now();
  const auto eq = ddx::rt::equation([] {
    std::vector<RE> x;
    for (const std::size_t i : std::views::iota(0uz, species)) {
      x.push_back(ddx::rt::var(std::format("x{:03}", i)));
    }
    return models::mse(x);
  });
  const std::chrono::duration<double, std::milli> built =
      std::chrono::steady_clock::now() - start;
  ASSERT_FALSE(eq.poisoned());
  EXPECT_EQ(eq.jacobian_columns(), species);
  std::printf("[ ddx ] n = %zu built in %.1f ms\n", species, built.count());
}

// prepare() builds the lane, so the threads below only ever read it -- and
// every one of them gets the single-thread answer to the bit.  The latch makes
// the overlap happen rather than hoping for it.
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
              ddx::impl::to<std::vector<double *>>();
    auto gs = g | std::views::transform([](auto &c) { return c.data(); }) |
              ddx::impl::to<std::vector<double *>>();
    if (together) {
      start.arrive_and_wait();
    }
    for (int rep = 0; rep < 1000; ++rep) {
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
