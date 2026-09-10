// The transcendentals are the device's own libm, which OpenCL bounds in ULPs
// rather than rounds correctly, so these are checked within a stated distance
// of glibc and not to the bit.
#include "tests_cl_common.hpp"

#include "ops/unary_math.hpp"

#include <cstdint>
#include <map>
#include <ranges>
#include <string_view>
#include <vector>

namespace {

using ddx::rt::Builder;
using ddx::rt::GraphBuilder;
using RE = ddx::rt::RTExpression<>;
using ClTranscendental = cltest::OnDevice;

struct Domain {
  double lo;
  double hi;
  std::uint64_t ulps; // OpenCL 1.2 section 7.4 for double, plus glibc's one
};

// Every row of DDX_UNARY_MATH_TABLE needs one, and the test below fails on a
// row without: a new transcendental cannot arrive untested.
const std::map<std::string_view, Domain> domains{
    {"sin", {-10.0, 10.0, 5}},    {"cos", {-10.0, 10.0, 5}},
    {"tan", {-1.5, 1.5, 6}},      {"exp", {-20.0, 20.0, 4}},
    {"log", {0.01, 100.0, 4}},    {"log10", {0.01, 100.0, 4}},
    {"sqrt", {0.0, 100.0, 0}},    {"cbrt", {-100.0, 100.0, 3}},
    {"asin", {-0.99, 0.99, 5}},   {"acos", {-0.99, 0.99, 5}},
    {"atan", {-5.0, 5.0, 6}},     {"sinh", {-5.0, 5.0, 5}},
    {"cosh", {-5.0, 5.0, 5}},     {"tanh", {-5.0, 5.0, 6}},
    {"asinh", {-5.0, 5.0, 5}},    {"acosh", {1.01, 50.0, 5}},
    {"atanh", {-0.99, 0.99, 6}},  {"erf", {-5.0, 5.0, 17}},
};

#define DDX_CL_TEST_MAKER(fn, Op, label) {label, [](const RE &x) { return fn(x); }},
const std::map<std::string_view, RE (*)(const RE &)> unary{
    DDX_UNARY_MATH_TABLE(DDX_CL_TEST_MAKER)};
#undef DDX_CL_TEST_MAKER

[[nodiscard]] std::vector<double> spread(double lo, double hi, std::size_t n) {
  return std::views::iota(0uz, n) | std::views::transform([=](std::size_t i) {
           return lo + (hi - lo) * (static_cast<double>(i) + 0.5) /
                           static_cast<double>(n);
         }) |
         ddx::impl::to<std::vector<double>>();
}

void expect_within(const Builder<> &b, const ddx::rt::Graph<> &g,
                   const cltest::Columns &xs, std::uint64_t bound,
                   std::string_view what) {
  const std::size_t n = xs.front().size();
  const auto k = cltest::device()->compile(g);
  ASSERT_TRUE(k) << what << ": " << k.error().detail;
  const auto got = cltest::ran(*k, xs, n);
  ASSERT_TRUE(got) << what << ": " << got.error().detail;
  const auto want = cltest::swept(b, g, xs, n);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    EXPECT_LE(cltest::ulps(want[0][i], (*got)[0][i]), bound)
        << what << " at " << xs[0][i] << ": " << want[0][i] << " swept, "
        << (*got)[0][i] << " on " << cltest::device()->identity();
  }
}

TEST(ClTranscendentalTable, EveryUnaryOpHasADomain) {
  for (const auto &[label, make] : unary) {
    EXPECT_TRUE(domains.contains(label)) << label;
  }
}

TEST_F(ClTranscendental, EveryUnaryOpIsWithinItsBound) {
  for (const auto &[label, make] : unary) {
    const auto it = domains.find(label);
    ASSERT_NE(it, domains.end()) << label;
    const Domain &d = it->second;
    Builder<> b;
    const auto x = var(b, "x");
    const auto g = GraphBuilder{b}.value(make(x)).finish();
    expect_within(b, g, {spread(d.lo, d.hi, 256)}, d.ulps, label);
  }
}

TEST_F(ClTranscendental, EveryBinaryOpIsWithinItsBound) {
  struct Binary {
    std::string_view label;
    RE (*make)(const RE &, const RE &);
    Domain x;
    Domain y;
  };
  const std::array binaries{
      Binary{"pow", [](const RE &x, const RE &y) { return pow(x, y); },
             {0.1, 10.0, 17}, {-3.0, 3.0, 17}},
      Binary{"atan2", [](const RE &x, const RE &y) { return atan2(x, y); },
             {-5.0, 5.0, 7}, {-5.0, 5.0, 7}},
      Binary{"hypot", [](const RE &x, const RE &y) { return hypot(x, y); },
             {-100.0, 100.0, 5}, {-100.0, 100.0, 5}},
  };
  for (const Binary &op : binaries) {
    cltest::Columns xs(2);
    for (const auto [x, y] : std::views::cartesian_product(
             spread(op.x.lo, op.x.hi, 16), spread(op.y.lo, op.y.hi, 16))) {
      xs[0].push_back(x);
      xs[1].push_back(y);
    }
    Builder<> b;
    const auto x = var(b, "x");
    const auto y = var(b, "y");
    const auto g = GraphBuilder{b}.value(op.make(x, y)).finish();
    expect_within(b, g, xs, op.x.ulps, op.label);
  }
}

} // namespace
