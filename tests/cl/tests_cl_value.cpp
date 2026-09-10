// Arithmetic on the device against the interpreter, to the bit.  IEEE 754
// rounds + - * / and fma correctly and OpenCL's double type keeps denormals,
// so a kernel that honours the graph's contraction has no room to differ.
#include "tests_cl_common.hpp"

#include "rt/derivative.hpp"

#include <array>
#include <functional>
#include <limits>
#include <ranges>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using ddx::rt::Builder;
using ddx::rt::GraphBuilder;
using RE = ddx::rt::RTExpression<>;
using ClValue = cltest::OnDevice;

struct Case {
  std::string_view name;
  std::function<RE(const RE &, const RE &, const RE &)> build;
};

// Every op a graph can hold short of the transcendentals, alone or nearly so,
// with its Jacobian: the derivatives bring the selects and comparisons.
const std::array cases{
    Case{"x*y + z", [](const RE &x, const RE &y, const RE &z) { return x * y + z; }},
    Case{"z - x*y", [](const RE &x, const RE &y, const RE &z) { return z - x * y; }},
    Case{"(x+y)/(x-z)",
         [](const RE &x, const RE &y, const RE &z) { return (x + y) / (x - z); }},
    Case{"-x*y", [](const RE &x, const RE &y, const RE &) { return -x * y; }},
    Case{"abs", [](const RE &x, const RE &y, const RE &z) { return abs(x - y) * z; }},
    Case{"sign", [](const RE &x, const RE &y, const RE &z) { return sign(x - y) + z; }},
    Case{"max", [](const RE &x, const RE &y, const RE &z) { return max(x, y) * z; }},
    Case{"min", [](const RE &x, const RE &y, const RE &z) { return min(x, -y) + z; }},
    Case{"select",
         [](const RE &x, const RE &y, const RE &z) {
           return select(x < y, x * z, y / z);
         }},
    Case{"compare",
         [](const RE &x, const RE &y, const RE &z) { return (x <= y) + (y < z) * x; }},
};

// The awkward doubles, every triple of them: signed zeros, infinities, NaN,
// denormals and the extremes.
[[nodiscard]] cltest::Columns awkward() {
  constexpr auto inf = std::numeric_limits<double>::infinity();
  constexpr auto tiny = std::numeric_limits<double>::denorm_min();
  const std::array values{0.0,  -0.0, 1.0,  -1.0, 0.5,   3.0,     -2.5, inf,
                          -inf, std::numeric_limits<double>::quiet_NaN(),
                          tiny, -tiny, 1e-310, 1e308, std::numeric_limits<double>::max()};
  cltest::Columns xs(3);
  for (const auto [x, y, z] : std::views::cartesian_product(values, values, values)) {
    xs[0].push_back(x);
    xs[1].push_back(y);
    xs[2].push_back(z);
  }
  return xs;
}

void expect_case_agrees(const Case &c, bool contract, const cltest::Columns &xs,
                        std::size_t n) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto z = var(b, "z");
  const auto g =
      GraphBuilder{b}.value(c.build(x, y, z)).build_jacobian().finish(contract);
  const auto k = cltest::device()->compile(g);
  ASSERT_TRUE(k) << c.name << ": " << k.error().detail;
  const auto got = cltest::ran(*k, xs, n);
  ASSERT_TRUE(got) << c.name << ": " << got.error().detail;
  cltest::expect_same_bits(cltest::swept(b, g, xs, n), *got,
                           std::format("{} (contract {})", c.name, contract));
}

TEST_F(ClValue, EveryArithmeticOpAgreesWithTheSweepToTheBit) {
  const auto xs = awkward();
  for (const Case &c : cases) {
    for (const bool contract : {true, false}) {
      expect_case_agrees(c, contract, xs, xs.front().size());
    }
  }
}

// A batch that is not a multiple of the work-group size leaves a tail of idle
// work-items; none of them may write.
TEST_F(ClValue, EveryBatchLengthAgrees) {
  const auto xs = awkward();
  for (const std::size_t n : {1uz, 2uz, 7uz, 8uz, 9uz, 63uz, 64uz, 65uz, 1000uz}) {
    expect_case_agrees(cases[0], true, xs, n);
    expect_case_agrees(cases[8], true, xs, n);
  }
}

// The same kernel called again with a longer batch, then a shorter one: the
// buffers grow and are reused, and nothing stale is read back.
TEST_F(ClValue, BuffersGrowAndAreReused) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto g = GraphBuilder{b}.value(x * y - y).build_jacobian().finish();
  const auto k = cltest::device()->compile(g);
  ASSERT_TRUE(k) << k.error().detail;
  for (const std::size_t n : {3uz, 500uz, 5uz}) {
    cltest::Columns xs(2);
    for (const std::size_t i : std::views::iota(0uz, n)) {
      xs[0].push_back(0.25 * static_cast<double>(i) + 1.0);
      xs[1].push_back(3.0 - 0.5 * static_cast<double>(i % 7));
    }
    const auto got = cltest::ran(*k, xs, n);
    ASSERT_TRUE(got) << got.error().detail;
    cltest::expect_same_bits(cltest::swept(b, g, xs, n), *got,
                             std::format("n = {}", n));
  }
}

} // namespace
