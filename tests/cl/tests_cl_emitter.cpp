// The OpenCL C the emitter writes, read as text: no device is needed, so these
// run on every host the backend is built on.
#include "tests_cl_common.hpp"

#include "rt/derivative.hpp"

#include <boost/algorithm/string/split.hpp> // find_all
#include <boost/range/iterator_range.hpp>

#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ddx::rt::Builder;
using ddx::rt::GraphBuilder;

[[nodiscard]] std::size_t count(const std::string &text, std::string_view what) {
  std::vector<boost::iterator_range<std::string::const_iterator>> hits;
  boost::algorithm::find_all(hits, text, what);
  return hits.size();
}

TEST(ClEmitter, BothPragmasPrecedeTheKernel) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const std::string src = ddx::cl::source_of(
      GraphBuilder{b}.value(x * y + sin(x)).build_jacobian().finish());
  const auto kernel = src.find("__kernel");
  ASSERT_NE(kernel, std::string::npos);
  EXPECT_LT(src.find("#pragma OPENCL EXTENSION cl_khr_fp64 : enable"), kernel);
  EXPECT_LT(src.find("#pragma OPENCL FP_CONTRACT OFF"), kernel);
}

// Contraction is the graph's decision, so the source fuses exactly the steps
// the schedule does -- and with it off, none.
TEST(ClEmitter, FmaWhereTheScheduleSaysAndNowhereElse) {
  for (const bool contract : {true, false}) {
    Builder<> b;
    const auto x = var(b, "x");
    const auto y = var(b, "y");
    const auto z = var(b, "z");
    const auto g = GraphBuilder{b}
                       .value(x * y + z - y * z)
                       .build_jacobian()
                       .finish(contract);
    const std::string src = ddx::cl::source_of(g);
    const auto fused = std::ranges::count_if(
        g.schedule(), [](const ddx::rt::Step &s) { return static_cast<bool>(s.fma); });
    const auto negated =
        std::ranges::count_if(g.schedule(), [](const ddx::rt::Step &s) {
          return static_cast<bool>(s.fma) && s.fma.negated;
        });
    EXPECT_EQ(count(src, "fma("), static_cast<std::size_t>(fused)) << contract;
    EXPECT_EQ(count(src, "fma(-v"), static_cast<std::size_t>(negated)) << contract;
    EXPECT_EQ(contract, fused > 0);
    EXPECT_EQ(count(src, "mad("), 0u);
  }
}

TEST(ClEmitter, ExtremaAreDefinedOnlyWhereTheGraphHasThem) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  EXPECT_NE(ddx::cl::source_of(GraphBuilder{b}.value(max(x, y) * y).finish())
                .find("double ddx_max("),
            std::string::npos);
  EXPECT_EQ(ddx::cl::source_of(GraphBuilder{b}.value(x * y).finish())
                .find("ddx_max"),
            std::string::npos);
}

TEST(ClEmitter, InputsReadTheColumnsTheInterpreterReads) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const std::string src =
      ddx::cl::source_of(GraphBuilder{b}.value(x / y).finish());
  EXPECT_NE(src.find("xs[0ul * n + i]"), std::string::npos);
  EXPECT_NE(src.find("xs[1ul * n + i]"), std::string::npos);
}

// Whatever constants survive the builder's folding, each is written as its
// own bits: hex for a finite value, the bit pattern for the rest.
TEST(ClEmitter, ConstantsRoundTripToTheBit) {
  std::size_t checked = 0;
  for (const double c :
       {1.0 / 3.0, 0.1, 7.0, -1.5, 1e308, std::numeric_limits<double>::max(),
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    Builder<> b;
    const auto x = var(b, "x");
    const auto g = GraphBuilder{b}.value(x * c + x).finish();
    const std::string src = ddx::cl::source_of(g);
    for (const ddx::rt::Step &s : g.schedule()) {
      if (g.op_of(s.node) != ddx::rt::OpCode::Const) {
        continue;
      }
      const std::string head = std::format("const double v{} = ", s.node);
      const auto at = src.find(head);
      ASSERT_NE(at, std::string::npos) << head;
      const auto from = at + head.size();
      const std::string literal = src.substr(from, src.find(';', from) - from);
      double parsed = 0;
      if (literal.starts_with("as_double(0x")) {
        std::uint64_t pattern = 0;
        const auto digits = std::string_view{literal}.substr(12);
        std::from_chars(digits.data(), digits.data() + digits.size(), pattern, 16);
        parsed = std::bit_cast<double>(pattern);
      } else {
        parsed = std::strtod(literal.c_str(), nullptr);
      }
      EXPECT_EQ(cltest::bits(parsed), cltest::bits(g[s.node].value))
          << literal << " for " << g[s.node].value;
      ++checked;
    }
  }
  EXPECT_GT(checked, 0u);
}

TEST(ClEmitter, AValuesOnlyGraphStoresNoDerivativeBlock) {
  Builder<> b;
  const auto x = var(b, "x");
  const std::string src =
      ddx::cl::source_of(GraphBuilder{b}.value(exp(x) * x).finish());
  EXPECT_NE(src.find("  f[0ul * n + i] = "), std::string::npos);
  EXPECT_EQ(src.find("  g["), std::string::npos);
  EXPECT_EQ(src.find("  h["), std::string::npos);
}

} // namespace
