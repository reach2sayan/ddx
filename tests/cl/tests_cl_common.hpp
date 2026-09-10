#pragma once

#include "cl/device.hpp"
#include "cl/kernel.hpp"
#include "rt/builder.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

namespace cltest {

using Columns = std::vector<std::vector<double>>;

// Which device: DDX_TEST_DEVICE names one on a host with several -- "Intel"
// for its CPU runtime, "Portable" for pocl -- and unset takes the default.
[[nodiscard]] inline std::string_view selector() {
  const char *const named = std::getenv("DDX_TEST_DEVICE");
  return named != nullptr ? named : "";
}

[[nodiscard]] inline const ddx::cl::result<ddx::cl::Device> &device() {
  static const auto chosen = ddx::cl::Device::create(selector());
  return chosen;
}

// A host without a device has nothing to run a kernel on, which is not a
// failure of the library; the skip says which.
class OnDevice : public ::testing::Test {
protected:
  void SetUp() override {
    if (!device()) {
      GTEST_SKIP() << device().error().code << ": " << device().error().detail;
    }
  }
};

[[nodiscard]] inline std::uint64_t bits(double v) {
  return std::bit_cast<std::uint64_t>(v);
}

// Representable doubles between the two, across zero as well.
[[nodiscard]] inline std::uint64_t ulps(double a, double b) {
  const auto ordered = [](double v) {
    const auto u = std::bit_cast<std::uint64_t>(v);
    return (u >> 63) != 0 ? ~u : u | (std::uint64_t{1} << 63);
  };
  // The initializer-list form: the pair form would hold references to the
  // two temporaries.
  const auto [lo, hi] = std::minmax({ordered(a), ordered(b)});
  return hi - lo;
}

[[nodiscard]] inline std::vector<double *> pointers(Columns &c) {
  return c | std::views::transform([](std::vector<double> &v) { return v.data(); }) |
         std::ranges::to<std::vector<double *>>();
}
[[nodiscard]] inline std::vector<const double *> pointers(const Columns &c) {
  return c |
         std::views::transform([](const std::vector<double> &v) { return v.data(); }) |
         std::ranges::to<std::vector<const double *>>();
}

// The interpreter at every point, one column per output in the graph's order:
// what a device kernel has to reproduce.
[[nodiscard]] inline Columns swept(const ddx::rt::Builder<> &b,
                                   const ddx::rt::Graph<> &g, const Columns &xs,
                                   std::size_t n) {
  Columns out(g.outputs().size(), std::vector<double>(n));
  std::vector<double> at(xs.size());
  std::vector<double> tape(b.size());
  for (const std::size_t i : std::views::iota(0uz, n)) {
    std::ranges::transform(xs, at.begin(),
                           [i](const std::vector<double> &c) { return c[i]; });
    ddx::rt::evaluate_into(b, at, g.schedule(), std::span<double>{tape});
    for (const auto [column, o] : std::views::zip(out, g.outputs())) {
      column[i] = tape[o];
    }
  }
  return out;
}

// The same columns off the device.  A column the kernel never writes keeps the
// sentinel, so a store that went missing shows.
inline constexpr double sentinel = -12345.678;

[[nodiscard]] inline ddx::cl::result<Columns>
ran(const ddx::cl::Kernel &k, const Columns &xs, std::size_t n) {
  const auto &shape = k.shape();
  Columns out(shape.outputs(), std::vector<double>(n, sentinel));
  const auto in = pointers(xs);
  const auto all = pointers(out);
  const std::span<double *const> blocks{all};
  return k(in, blocks.first(shape.values),
           blocks.subspan(shape.values, shape.jacobian),
           blocks.subspan(shape.values + shape.jacobian, shape.hessian), n)
      .transform([&out] { return std::move(out); });
}

// To the bit, except that a NaN need only be a NaN: OpenCL does not promise a
// payload.
inline void expect_same_bits(const Columns &want, const Columns &got,
                             std::string_view what) {
  ASSERT_EQ(want.size(), got.size()) << what;
  for (const auto [k, pair] : std::views::enumerate(std::views::zip(want, got))) {
    const auto &[w, g] = pair;
    for (const auto [i, values] : std::views::enumerate(std::views::zip(w, g))) {
      const auto [expected, actual] = values;
      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual))
            << what << ": output " << k << ", point " << i << " is " << actual;
      } else {
        EXPECT_EQ(bits(expected), bits(actual))
            << what << ": output " << k << ", point " << i << ": " << expected
            << " swept, " << actual << " on the device";
      }
    }
  }
}

} // namespace cltest
