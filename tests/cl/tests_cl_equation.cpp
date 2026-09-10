// Backend::Device through the Equation a user holds: every lane, the choice of
// device, and what happens when there is none.
#include "tests_cl_common.hpp"

#include "rt/energy_models.hpp"
#include "rt/equation.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <format>
#include <latch>
#include <ranges>
#include <string>
#include <thread>
#include <vector>

namespace {

using ddx::rt::Backend;
using ddx::rt::Want;
using RE = ddx::rt::RTExpression<>;
using ClEquation = cltest::OnDevice;

constexpr std::size_t symbols = 12;

// Forty products on one spine, then the ops that branch.  A spine that long is
// what the LLVM kernel rebalances; the device takes the swept graph, so here
// it has to agree to the bit.
[[nodiscard]] auto arithmetic() {
  return ddx::rt::equation([] {
    const auto x = std::views::iota(0uz, symbols) |
                   std::views::transform([](std::size_t i) {
                     return ddx::rt::var(std::format("x{:02d}", i));
                   }) |
                   ddx::impl::to<std::vector<RE>>();
    RE acc = 0.0;
    for (const std::size_t i : std::views::iota(0uz, 40uz)) {
      acc = acc + x[i % symbols] * x[(i + 1) % symbols] *
                      (1.0 / (1.0 + static_cast<double>(i)));
    }
    return acc + max(x[0], x[1]) -
           select(x[2] < x[3], x[2] * x[3], x[4] / x[5]) + abs(x[6] - x[7]);
  });
}

[[nodiscard]] cltest::Columns points(std::size_t columns, std::size_t n) {
  cltest::Columns out(columns, std::vector<double>(n));
  for (const auto [j, column] : std::views::enumerate(out)) {
    for (const auto [i, v] : std::views::enumerate(column)) {
      v = 0.3 + 0.5 * static_cast<double>((i * 7 + j * 3) % 11) / 11.0;
    }
  }
  return out;
}

// Every batch call the equation has, end to end in one vector.
[[nodiscard]] std::vector<double> every_lane(const auto &eq, std::size_t n) {
  const auto xs_columns = points(symbols, n);
  const auto v_columns = points(symbols, n);
  const auto w_columns = points(1, n);
  const auto xs = cltest::pointers(xs_columns);
  const auto v = cltest::pointers(v_columns);
  const auto w = cltest::pointers(w_columns);
  std::vector<double> out;
  const auto keep = [&out](const cltest::Columns &c) {
    for (const auto &column : c) {
      out.insert(out.end(), column.begin(), column.end());
    }
  };
  const auto block = [n](std::size_t columns) {
    return cltest::Columns(columns, std::vector<double>(n));
  };
  auto f = block(1);
  auto g = block(*eq.jacobian_columns());
  auto h = block(*eq.hessian_columns());
  auto hv = block(*eq.hvp_columns());
  auto vj = block(*eq.vjp_columns());
  auto jv = block(*eq.jvp_columns());
  auto fp = cltest::pointers(f);
  auto gp = cltest::pointers(g);
  auto hp = cltest::pointers(h);
  auto hvp = cltest::pointers(hv);
  auto vjp = cltest::pointers(vj);
  auto jvp = cltest::pointers(jv);

  EXPECT_TRUE(eq.evaluate(xs, fp, n));
  keep(f);
  EXPECT_TRUE(eq.jacobian(xs, fp, gp, n));
  keep(f);
  keep(g);
  EXPECT_TRUE(eq.gradient(xs, gp, n));
  keep(g);
  EXPECT_TRUE(eq.hessian(xs, fp, gp, hp, n));
  keep(f);
  keep(g);
  keep(h);
  EXPECT_TRUE(eq.hvp(xs, v, fp, gp, hvp, n));
  keep(f);
  keep(g);
  keep(hv);
  EXPECT_TRUE(eq.vjp(xs, w, fp, vjp, n));
  keep(f);
  keep(vj);
  EXPECT_TRUE(eq.jvp(xs, v, fp, jvp, n));
  keep(f);
  keep(jv);
  return out;
}

void expect_same_bits(const std::vector<double> &want,
                      const std::vector<double> &got, std::size_t n) {
  ASSERT_EQ(want.size(), got.size());
  for (const auto [i, pair] : std::views::enumerate(std::views::zip(want, got))) {
    const auto [w, g] = pair;
    EXPECT_EQ(cltest::bits(w), cltest::bits(g))
        << "n = " << n << ", element " << i << ": " << w << " swept, " << g
        << " on the device";
  }
}

[[nodiscard]] ddx::jit::Options on_device(std::string selector = std::string{
                                              cltest::selector()}) {
  return {.backend = Backend::Device, .points = 1000, .device = std::move(selector)};
}

TEST_F(ClEquation, EveryLaneAnswersOnTheDeviceToTheBit) {
  const auto swept = arithmetic();
  auto dev = arithmetic();
  dev.options(on_device());
  for (const Want want : ddx::rt::want_values) {
    EXPECT_TRUE(dev.wait_for_kernel(want)) << ddx::rt::name_of(want);
  }
  EXPECT_TRUE(dev.uses_kernel());
  EXPECT_FALSE(dev.kernel_level().has_value());
  const auto status = dev.device_status();
  ASSERT_TRUE(status.has_value());
  ASSERT_TRUE(status->has_value()) << status->error().detail;
  EXPECT_EQ(**status, cltest::device()->identity());

  for (const std::size_t n : {1uz, 7uz, 8uz, 1000uz}) {
    expect_same_bits(every_lane(swept, n), every_lane(dev, n), n);
  }
}

// Transcendentals are the device's libm, so a real model agrees closely rather
// than exactly.
TEST_F(ClEquation, AModelWithTranscendentalsAgreesClosely) {
  const auto model = [] {
    return ddx::rt::equation([] {
      return models::uniquac(std::views::iota(0uz, 6uz) |
                             std::views::transform([](std::size_t i) {
                               return ddx::rt::var(std::format("x{}", i));
                             }) |
                             ddx::impl::to<std::vector<RE>>());
    });
  };
  const auto swept = model();
  auto dev = model();
  dev.options(on_device());
  ASSERT_TRUE(dev.wait_for_kernel());

  constexpr std::size_t n = 257;
  const auto xs_columns = points(6, n);
  const auto xs = cltest::pointers(xs_columns);
  const auto run = [&xs](const auto &eq) {
    cltest::Columns f(1, std::vector<double>(n));
    cltest::Columns g(*eq.jacobian_columns(), std::vector<double>(n));
    auto fp = cltest::pointers(f);
    auto gp = cltest::pointers(g);
    EXPECT_TRUE(eq.jacobian(xs, fp, gp, n));
    g.insert(g.begin(), f.front());
    return g;
  };
  const auto want = run(swept);
  const auto got = run(dev);
  for (const auto [k, pair] : std::views::enumerate(std::views::zip(want, got))) {
    const auto &[w, g] = pair;
    for (const auto [i, values] : std::views::enumerate(std::views::zip(w, g))) {
      const auto [expected, actual] = values;
      EXPECT_NEAR(actual, expected, 1e-10 * std::max(1.0, std::abs(expected)))
          << "column " << k << ", point " << i;
    }
  }
}

TEST_F(ClEquation, SwitchingBackendsTakesAndDropsTheKernel) {
  auto eq = arithmetic();
  eq.options(on_device());
  EXPECT_TRUE(eq.wait_for_kernel());
  eq.options({.backend = Backend::Interpret});
  EXPECT_FALSE(eq.uses_kernel());
  EXPECT_FALSE(eq.device_status().has_value());
  eq.options(on_device());
  EXPECT_TRUE(eq.wait_for_kernel());
}

// Each thread its own columns, one kernel between them: the launches are
// serialised inside, and every answer is whole.
TEST_F(ClEquation, ConcurrentBatchesEachGetTheirOwnAnswer) {
  auto dev = arithmetic();
  dev.options(on_device());
  ASSERT_TRUE(dev.wait_for_kernel());
  constexpr std::size_t n = 64;
  const auto xs_columns = points(symbols, n);
  const auto xs = cltest::pointers(xs_columns);
  const auto once = [&] {
    cltest::Columns f(1, std::vector<double>(n));
    cltest::Columns g(*dev.jacobian_columns(), std::vector<double>(n));
    auto fp = cltest::pointers(f);
    auto gp = cltest::pointers(g);
    EXPECT_TRUE(dev.jacobian(xs, fp, gp, n));
    g.push_back(f.front());
    return g;
  };
  const auto reference = once();

  constexpr std::size_t threads = 4;
  std::latch start{threads};
  std::vector<int> wrong(threads, 0);
  {
    std::vector<std::jthread> pool;
    for (const std::size_t t : std::views::iota(0uz, threads)) {
      pool.emplace_back([&, t] {
        start.arrive_and_wait();
        for ([[maybe_unused]] const int round : std::views::iota(0, 50)) {
          wrong[t] += once() != reference ? 1 : 0;
        }
      });
    }
  }
  EXPECT_EQ(wrong, std::vector<int>(threads, 0));
}

// No device needed: a selector nothing matches is the same as a host with
// none.  The equation still answers, from the sweep, and says why.
TEST(ClEquationSelection, ASelectorNothingMatchesSweepsAndSaysWhy) {
  const auto swept = arithmetic();
  auto eq = arithmetic();
  eq.options({.backend = Backend::Device, .device = "no-such-device-xyzzy"});
  EXPECT_FALSE(eq.wait_for_kernel());
  EXPECT_FALSE(eq.uses_kernel());
  const auto status = eq.device_status();
  ASSERT_TRUE(status.has_value());
  ASSERT_FALSE(status->has_value());
  EXPECT_EQ(status->error().code, ddx::errc::no_device);
  expect_same_bits(every_lane(swept, 9), every_lane(eq, 9), 9);
}

TEST(ClEquationSelection, DeviceStatusIsNothingUnderAnyOtherBackend) {
  auto eq = arithmetic();
  EXPECT_FALSE(eq.device_status().has_value());
  eq.options({.backend = Backend::Adapt});
  EXPECT_FALSE(eq.device_status().has_value());
}

} // namespace
