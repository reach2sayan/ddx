#include "graph_builder.hpp"
#include "jit/kernel.hpp"
#include "jit/tests_jit_common.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"

#include <gtest/gtest.h>

#include <format>
#include <ranges>
#include <regex>
#include <string>

// A scalar fallback still gives right answers, only slowly, so vectorisation is
// asserted on the IR rather than inferred from timings.

namespace {
using ddx::rt::Arena;
using ddx::rt::Graph;

// libmvec offers widths 2, 4 and 8, and the host picks, so any width counts.
bool has_vector_doubles(const std::string &ir) {
  return std::regex_search(ir, std::regex{R"(<[0-9]+ x double>)"});
}

bool calls_vector_libm(const std::string &ir, const std::string &fn) {
  return std::regex_search(ir,
                           std::regex{"_ZGV[a-z]N[0-9]+v+_" + fn + R"(\()"});
}

constexpr bool host_has_libmvec =
#if defined(__x86_64__) && defined(__linux__)
    true;
#else
    false;
#endif

std::string ir_for(auto build, std::size_t nvars, ddx::jit::Options opt = {}) {
  Arena<> b;
  std::vector<ddx::rt::Expression<>> vars;
  static constexpr const char *names[] = {"x", "y"};
  for (const char *name : names | std::views::take(nvars)) {
    vars.push_back(var(b, name));
  }
  const auto root = build(vars);
  const auto graph = ddx::test::freeze_values(b, std::array{root.id(b)});
  return *compiler().ir(graph, opt);
}

// nounwind backs Kernel::operator()'s noexcept and willreturn lets the
// optimiser move code across the call; both are checked on the optimised IR.
TEST(JitVectorize, TheKernelIsNounwind) {
  const auto ir = ir_for([](auto &v) { return exp(v[0]) * sin(v[1]); }, 2);

  const std::smatch attrs = [&] {
    std::smatch m;
    // Find the attribute group the kernel definition names, then its body.
    std::smatch def;
    EXPECT_TRUE(std::regex_search(
        ir, def, std::regex{R"(define[^\n]*@ddx_kernel_\w+\([^\n]*#(\d+))"}))
        << "no kernel definition in the IR";
    std::regex_search(
        ir, m, std::regex{"attributes #" + def[1].str() + R"( = \{([^}]*)\})"});
    return m;
  }();
  ASSERT_FALSE(attrs.empty()) << "the kernel names no attribute group";

  const std::string got = attrs[1].str();
  EXPECT_NE(got.find("nounwind"), std::string::npos) << got;
  EXPECT_NE(got.find("willreturn"), std::string::npos) << got;
  // The kernel reaches its columns through loaded pointers, so an argmem claim
  // would be false and the optimiser might act on it.
  EXPECT_EQ(got.find("argmem"), std::string::npos) << got;
}

TEST(JitVectorize, ArithmeticLoopVectorises) {
  const auto ir = ir_for([](auto &v) { return v[0] * v[1] + v[0]; }, 2);
  EXPECT_TRUE(has_vector_doubles(ir)) << "the arithmetic loop stayed scalar";
}

// By default the vector library is declined, so a transcendental is the scalar
// libm routine that the interpreter also calls.
TEST(JitVectorize, TranscendentalsStayScalarByDefault) {
  const auto ir = ir_for([](auto &v) { return exp(v[0]) * sin(v[1]); }, 2);
  EXPECT_FALSE(calls_vector_libm(ir, "sin")) << "sin took a vector route";
  EXPECT_FALSE(calls_vector_libm(ir, "exp")) << "exp took a vector route";
}

// Opting in to libmvec still produces vector libm calls.
TEST(JitVectorize, VecLibLibmvecUsesVectorLibm) {
  if constexpr (!host_has_libmvec) {
    GTEST_SKIP() << "libmvec is glibc on x86-64";
  }
  ddx::jit::Options on;
  on.codegen.veclib = ddx::jit::VecLib::Libmvec;
  const auto ir = ir_for([](auto &v) { return exp(v[0]) * sin(v[1]); }, 2, on);
  EXPECT_TRUE(has_vector_doubles(ir)) << "loop stayed scalar";
  EXPECT_TRUE(calls_vector_libm(ir, "sin")) << "sin was not vectorised";
  EXPECT_TRUE(calls_vector_libm(ir, "exp")) << "exp was not vectorised";
}

// LLVM 20's libmvec stops at four doubles, so a derived width is capped there;
// an eight-wide sin would otherwise fall back to scalar calls.
TEST(JitVectorize, VecLibCapsADerivedWidth) {
  if constexpr (!host_has_libmvec) {
    GTEST_SKIP() << "libmvec is glibc on x86-64";
  }
  ddx::jit::Options on;
  on.codegen.veclib = ddx::jit::VecLib::Libmvec;
  const auto ir = ir_for([](auto &v) { return sin(v[0]) + v[1]; }, 2, on);
  EXPECT_FALSE(std::regex_search(ir, std::regex{R"(<(8|16) x double>)"}))
      << "wider than the library serves";
  // A stated width is used as stated, whatever the library supports.
  on.codegen.lanes = *ddx::jit::Lanes::exactly(8);
  const auto wide = ir_for([](auto &v) { return sin(v[0]) + v[1]; }, 2, on);
  EXPECT_NE(wide.find("<8 x double>"), std::string::npos) << wide;
}

TEST(JitVectorize, JacobianLoopVectorises) {
  // A wider Jacobian with shared subexpressions must still vectorise.
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto graph = ddx::test::GraphBuilder{b}
                         .value(x * y + x * x - y)
                         .build_jacobian()
                         .finish();
  const auto ir = *compiler().ir(graph);
  EXPECT_TRUE(has_vector_doubles(ir)) << "the Jacobian loop stayed scalar";
}

// A loop vectoriser would give up on many columns over alias checks, but the
// body is emitted wide, so the column count does not matter.
TEST(JitVectorize, WideTranscendentalJacobianIsVector) {
  Arena<> b;
  const auto v = std::views::iota(0uz, 16uz) |
                 std::views::transform([&](std::size_t i) {
                   return var(b, "x" + std::to_string(i));
                 }) |
                 ddx::impl::to<std::vector>();
  auto f = v[0] * log(v[0]);
  for (const auto [prev, cur] : v | std::views::pairwise) {
    f = f + cur * log(cur) + exp(prev * cur);
  }
  const auto graph =
      ddx::test::GraphBuilder{b}.value(f).build_jacobian().finish();
  ASSERT_EQ(graph.layout().values + graph.layout().jacobian, 17u);
  const auto ir = *compiler().ir(graph);
  EXPECT_TRUE(has_vector_doubles(ir)) << "33 columns stayed scalar";
  EXPECT_NE(ir.find("llvm.masked.load"), std::string::npos)
      << "the batch tail is not masked";
}

TEST(JitVectorize, OneLaneIsScalar) {
  ddx::jit::Options scalar;
  scalar.codegen.lanes = ddx::jit::Lanes::scalar();
  const auto ir =
      ir_for([](auto &v) { return exp(v[0]) * sin(v[1]) + v[0]; }, 2, scalar);
  EXPECT_FALSE(has_vector_doubles(ir)) << ir;
  EXPECT_EQ(ir.find("llvm.masked"), std::string::npos) << ir;
}

// LLVM's vector table targets a newer glibc than some hosts ship, so every
// symbol the vectoriser chose under the opt-in must resolve.
TEST(JitVectorize, EveryVectorSymbolResolves) {
  if constexpr (!host_has_libmvec) {
    GTEST_SKIP() << "libmvec is glibc on x86-64";
  }
  Arena<> b;
  const auto x = var(b, "x");
  auto f = sin(x) + cos(x) + exp(x) + log(x) + tanh(x) + erf(x) + cbrt(x);
  f = f + atan(x) + asinh(x) + pow(x, ddx::rt::Expression<>{3});

  const auto graph = ddx::test::freeze_values(b, std::array{f.id(b)});
  ddx::jit::Options on;
  on.codegen.veclib = ddx::jit::VecLib::Libmvec;
  const auto kernel = must_compile(graph, on);
  ASSERT_TRUE(static_cast<bool>(kernel));

  const std::array col{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  const std::array<const double *, 1> xs{col.data()};
  std::array<double, 8> got{};
  double *const value_columns[]{got.data()};
  kernel(xs, value_columns, {}, {}, got.size());
  for (const double v : got) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

} // namespace

namespace {

} // namespace
