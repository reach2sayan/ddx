#include "jit/format.hpp"
#include "jit/kernel.hpp"
#include "jit/tests_jit_common.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"

#include <gtest/gtest.h>

#include <format>
#include <regex>
#include <sstream>
#include <string>

// A scalar fallback is silent -- right answers, slowly -- so vectorisation is
// asserted on the IR, not inferred from a benchmark.  Transcendentals are
// asserted both ways: scalar under the default, vector when a caller opts in.

namespace {
using ddx::rt::Arena;
using ddx::rt::Graph;

// Any width: libmvec offers 2, 4 and 8, and the choice is the host's.
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
  for (std::size_t i = 0; i < nvars; ++i) {
    vars.push_back(var(b, names[i]));
  }
  const auto root = build(vars);
  const auto graph = Graph<>::freeze(b, std::array{root.id(b)});
  return std::format("{}", ddx::jit::Ir{compiler(), graph, opt});
}

// nounwind is what Kernel::operator()'s noexcept rests on, and willreturn lets
// the optimiser move code across the call.  Asserted on the *optimised* IR,
// which is what runs.
TEST(JitVectorize, TheKernelIsNounwind) {
  const auto ir = ir_for([](auto &v) { return exp(v[0]) * sin(v[1]); }, 2);

  const std::smatch attrs = [&] {
    std::smatch m;
    // The attribute group the kernel definition names, then that group's body.
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
  // The columns are reached through pointers the kernel loads, so no argmem
  // claim is true of it; one here would be a lie the optimiser may believe.
  EXPECT_EQ(got.find("argmem"), std::string::npos) << got;
}

TEST(JitVectorize, ArithmeticLoopVectorises) {
  const auto ir = ir_for([](auto &v) { return v[0] * v[1] + v[0]; }, 2);
  EXPECT_TRUE(has_vector_doubles(ir)) << "the arithmetic loop stayed scalar";
}

// The default declines the vector library, so a transcendental is the scalar
// libm one -- the same routine the interpreter calls.  This is the assertion
// that keeps the two on one implementation unless someone says otherwise.
TEST(JitVectorize, TranscendentalsStayScalarByDefault) {
  const auto ir = ir_for([](auto &v) { return exp(v[0]) * sin(v[1]); }, 2);
  EXPECT_FALSE(calls_vector_libm(ir, "sin")) << "sin took a vector route";
  EXPECT_FALSE(calls_vector_libm(ir, "exp")) << "exp took a vector route";
}

// And the opt-in still works: the machinery is kept, only its default changed.
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

// A derived width under the vector library is the widest it serves: LLVM 20's
// libmvec stops at four doubles, and an AVX-512 host would otherwise emit an
// eight-wide sin with no library form and get scalar calls back.
TEST(JitVectorize, VecLibCapsADerivedWidth) {
  if constexpr (!host_has_libmvec) {
    GTEST_SKIP() << "libmvec is glibc on x86-64";
  }
  ddx::jit::Options on;
  on.codegen.veclib = ddx::jit::VecLib::Libmvec;
  const auto ir = ir_for([](auto &v) { return sin(v[0]) + v[1]; }, 2, on);
  EXPECT_FALSE(std::regex_search(ir, std::regex{R"(<(8|16) x double>)"}))
      << "wider than the library serves";
  // Stated widths are taken as stated, whatever the library has.
  on.codegen.lanes = *ddx::jit::Lanes::exactly(8);
  const auto wide = ir_for([](auto &v) { return sin(v[0]) + v[1]; }, 2, on);
  EXPECT_NE(wide.find("<8 x double>"), std::string::npos) << wide;
}

TEST(JitVectorize, JacobianLoopVectorises) {
  // Wider and sharing subexpressions, and still has to vectorise.
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto graph = ddx::rt::GraphBuilder{b}
                         .value(x * y + x * x - y)
                         .build_jacobian()
                         .finish();
  const auto ir = std::format("{}", ddx::jit::Ir{compiler(), graph});
  EXPECT_TRUE(has_vector_doubles(ir)) << "the Jacobian loop stayed scalar";
}

// Many columns and a transcendental in every one.  A loop vectoriser would
// need a runtime alias check per pair of columns and give up past a handful;
// the body is emitted wide instead, so the column count is not a question.
TEST(JitVectorize, WideTranscendentalJacobianIsVector) {
  Arena<> b;
  std::vector<ddx::rt::Expression<>> v;
  for (std::size_t i = 0; i < 16; ++i) {
    v.push_back(var(b, "x" + std::to_string(i)));
  }
  auto f = v[0] * log(v[0]);
  for (std::size_t i = 1; i < v.size(); ++i) {
    f = f + v[i] * log(v[i]) + exp(v[i - 1] * v[i]);
  }
  const auto graph =
      ddx::rt::GraphBuilder{b}.value(f).build_jacobian().finish();
  ASSERT_EQ(graph.layout().values + graph.layout().jacobian, 17u);
  const auto ir = std::format("{}", ddx::jit::Ir{compiler(), graph});
  EXPECT_TRUE(has_vector_doubles(ir)) << "33 columns stayed scalar";
  EXPECT_NE(ir.find("llvm.masked.load"), std::string::npos)
      << "the batch tail is not masked";
}

// The scalar kernel has no vector type anywhere in it.
TEST(JitVectorize, OneLaneIsScalar) {
  ddx::jit::Options scalar;
  scalar.codegen.lanes = ddx::jit::Lanes::scalar();
  const auto ir =
      ir_for([](auto &v) { return exp(v[0]) * sin(v[1]) + v[0]; }, 2, scalar);
  EXPECT_FALSE(has_vector_doubles(ir)) << ir;
  EXPECT_EQ(ir.find("llvm.masked"), std::string::npos) << ir;
}

// LLVM's table is written against a newer glibc than some hosts ship, so under
// the opt-in every symbol the vectoriser chose has to resolve.
TEST(JitVectorize, EveryVectorSymbolResolves) {
  if constexpr (!host_has_libmvec) {
    GTEST_SKIP() << "libmvec is glibc on x86-64";
  }
  Arena<> b;
  const auto x = var(b, "x");
  auto f = sin(x) + cos(x) + exp(x) + log(x) + tanh(x) + erf(x) + cbrt(x);
  f = f + atan(x) + asinh(x) + pow(x, ddx::rt::Expression<>{3});

  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});
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

// operator<< goes through the formatter, as it does for an expression.
TEST(JitVectorize, IrStreamsAndFormatsAlike) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{sin(x).id(b)});
  const ddx::jit::Ir ir{compiler(), graph};

  std::ostringstream os;
  os << ir;
  EXPECT_EQ(os.str(), std::format("{}", ir));
  EXPECT_NE(os.str().find("define"), std::string::npos);
}

} // namespace
