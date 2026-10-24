#include "ddx.hpp"
#include "jit/kernel.hpp"
#include "jit/tests_jit_common.hpp"
#include "rt/bridge.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "scratch.hpp"

#include <boost/crc.hpp>
#include <boost/endian/conversion.hpp>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <latch>
#include <numeric>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// The interpreter in rt/interpret.hpp is the reference: it and the kernel walk
// the same graph, so a disagreement is a codegen bug.  Batch shape throughout.

namespace {
using ddx::rt::Arena;
using ddx::rt::Graph;

// The cache entry's prologue: tag, format, key, CRC, then the two lengths.  The
// one region no checksum covers, which is what the corruption sweep walks.
constexpr std::size_t kHeaderBytes = 8 + 4 + 8 + 4 + 4 + 8;

std::vector<std::byte> slurp(const std::filesystem::path &path) {
  std::ifstream in{path, std::ios::binary};
  const std::vector<char> raw{std::istreambuf_iterator<char>{in}, {}};
  const auto bytes = std::as_bytes(std::span{raw});
  return {bytes.begin(), bytes.end()};
}
void dump(const std::filesystem::path &path, std::span<const std::byte> bytes) {
  std::ofstream out{path, std::ios::binary | std::ios::trunc};
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

// For a failure message: the stated width, or that there is none.
std::string spelled(ddx::jit::Lanes lanes) {
  const auto w = lanes.stated();
  return w ? std::to_string(*w) : "derived";
}

// Compile `build` and compare a batch against the interpreter.
void expect_matches_interpreter(auto build, std::size_t nvars,
                                std::size_t n = 64) {
  Arena<> b;
  std::vector<ddx::rt::Expression<>> vars;
  static constexpr const char *names[] = {"x", "y", "z", "w"};
  for (std::size_t i = 0; i < nvars; ++i) {
    vars.push_back(var(b, names[i]));
  }
  const auto root = build(b, vars);
  const auto graph = Graph<>::freeze(b, std::array{root.id(b)});
  const auto kernel = must_compile(graph);
  ASSERT_TRUE(static_cast<bool>(kernel));
  ASSERT_EQ(kernel.shape().arity, nvars);

  // Spread the sample points over a range every op in the suite accepts.
  std::vector<std::vector<double>> columns(nvars, std::vector<double>(n));
  for (std::size_t j = 0; j < nvars; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      columns[j][i] = 0.15 + 0.6 * static_cast<double>((i + 3 * j) % 11) / 11.0;
    }
  }
  std::vector<const double *> xs(nvars);
  for (std::size_t j = 0; j < nvars; ++j) {
    xs[j] = columns[j].data();
  }

  std::vector<double> got(n, std::numeric_limits<double>::quiet_NaN());
  double *const value_columns[]{got.data()};
  kernel(xs, value_columns, {}, {}, n);

  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> point(nvars);
    for (std::size_t j = 0; j < nvars; ++j) {
      point[j] = columns[j][i];
    }
    const double want = ddx::rt::evaluate(b, root.id(b), point);
    ASSERT_NEAR(got[i], want, 1e-12 * std::max(1.0, std::abs(want)))
        << "at point " << i;
  }
}

// A Compiler is the LLJIT and a Kernel points into code it owns: one owner, but
// it must move -- Equation keeps one in a static.
TEST(JitValue, CompilerIsMovableButNotCopyable) {
  static_assert(!std::is_copy_constructible_v<ddx::jit::Compiler>);
  static_assert(!std::is_copy_assignable_v<ddx::jit::Compiler>);
  static_assert(std::is_move_constructible_v<ddx::jit::Compiler>);
  static_assert(std::is_move_assignable_v<ddx::jit::Compiler>);
  SUCCEED();
}

TEST(JitValue, Arithmetic) {
  expect_matches_interpreter([](Arena<> &, auto &v) { return v[0] + v[1]; },
                             2);
  expect_matches_interpreter([](Arena<> &, auto &v) { return v[0] - v[1]; },
                             2);
  expect_matches_interpreter([](Arena<> &, auto &v) { return v[0] * v[1]; },
                             2);
  expect_matches_interpreter([](Arena<> &, auto &v) { return v[0] / v[1]; },
                             2);
  expect_matches_interpreter([](Arena<> &, auto &v) { return -v[0] * v[1]; },
                             2);
}

// Split across tests so a failure names the op that broke.
TEST(JitValue, IntrinsicBackedFunctions) {
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return sin(v[0]) + cos(v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return tan(v[0]) * exp(v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return log(v[0]) - log10(v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return sqrt(v[0]) + asin(v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return acos(v[0]) + atan(v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return sinh(v[0]) + cosh(v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return tanh(v[0]) * abs(v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return pow(v[0], v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return atan2(v[0], v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return max(v[0], v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return min(v[0], v[1]); }, 2);
}

// A constant exponent or base is where LLVM's library-call simplifier would
// rewrite pow -- sqrt for 0.5, a division for -1, exp2/exp10 for a base -- and
// libm's pow is not correctly rounded, so any rewrite moves a bit.
TEST(JitValue, PowWithAConstantOperandIsTheSweepsPowToTheBit) {
  const auto bits_agree = [](auto build) {
    Arena<> b;
    const auto x = var(b, "x");
    const auto root = build(b, x);
    const auto graph = Graph<>::freeze(b, std::array{root.id(b)});
    const auto kernel = must_compile(graph);
    ASSERT_TRUE(static_cast<bool>(kernel));
    constexpr std::size_t n = 64;
    std::vector<double> column(n);
    for (std::size_t i = 0; i < n; ++i) {
      column[i] = 0.013 + 1.37 * static_cast<double>(i);
    }
    const double *const xs[]{column.data()};
    std::vector<double> got(n);
    double *const values[]{got.data()};
    kernel(xs, values, {}, {}, n);
    for (std::size_t i = 0; i < n; ++i) {
      const double want =
          ddx::rt::evaluate(b, root.id(b), std::span{&column[i], 1});
      EXPECT_EQ(std::bit_cast<std::uint64_t>(got[i]),
                std::bit_cast<std::uint64_t>(want))
          << "at x = " << column[i] << ": " << got[i] << " vs " << want;
    }
  };
  bits_agree([](Arena<> &, const auto &x) { return pow(x, 0.5); });
  bits_agree([](Arena<> &, const auto &x) { return pow(x, -1.0); });
  bits_agree([](Arena<> &, const auto &x) { return pow(x, 3.5); });
  bits_agree([](Arena<> &b, const auto &x) { return pow(lit(b, 10.0), x); });
  bits_agree([](Arena<> &b, const auto &x) { return pow(lit(b, 2.0), x); });
}

// No LLVM intrinsic: these go out as libm calls, a different emitter path.
TEST(JitValue, LibmBackedFunctions) {
  expect_matches_interpreter([](Arena<> &, auto &v) { return cbrt(v[0]); },
                             1);
  expect_matches_interpreter([](Arena<> &, auto &v) { return asinh(v[0]); },
                             1);
  expect_matches_interpreter([](Arena<> &, auto &v) { return atanh(v[0]); },
                             1);
  expect_matches_interpreter([](Arena<> &, auto &v) { return erf(v[0]); }, 1);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return hypot(v[0], v[1]); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return acosh(v[0] + 1.0); }, 1);
}

TEST(JitValue, MaxMinPropagateNaNSymmetrically) {
  // llvm.maximum/minimum propagate a NaN operand from either side, as
  // max_impl does; maxnum/minnum would return the other operand.
  Arena<> b;
  const auto root =
      max(var(b, "x"), var(b, "y")) + min(var(b, "x"), var(b, "y"));
  const auto kernel = must_compile(Graph<>::freeze(b, std::array{root.id(b)}));
  ASSERT_TRUE(static_cast<bool>(kernel));

  const double qnan = std::numeric_limits<double>::quiet_NaN();
  const std::array cx{qnan, 2.0, 1.0};
  const std::array cy{2.0, qnan, 3.0};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};
  std::array<double, 3> got{};
  double *const value_columns[]{got.data()};
  kernel(xs, value_columns, {}, {}, got.size());

  EXPECT_TRUE(std::isnan(got[0]));
  EXPECT_TRUE(std::isnan(got[1]));
  EXPECT_DOUBLE_EQ(got[2], 4.0);
}

// IEEE 754-2019 maximum/minimum order -0 below +0, which is what llvm.maximum
// and llvm.minimum compute; the interpreter has to say the same, or 1/min(x, y)
// changes sign between the two.
TEST(JitValue, MaxMinSignedZeroTiesMatchTheInterpreter) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto graph =
      Graph<>::freeze(b, std::array{max(x, y).id(b), min(x, y).id(b)});
  const auto kernel = must_compile(graph);
  ASSERT_TRUE(static_cast<bool>(kernel));

  const std::array cx{-0.0, 0.0, -0.0, 0.0};
  const std::array cy{0.0, -0.0, -0.0, 0.0};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};
  std::array<double, 4> got_max{};
  std::array<double, 4> got_min{};
  double *const value_columns[]{got_max.data(), got_min.data()};
  kernel(xs, value_columns, {}, {}, cx.size());

  for (std::size_t i = 0; i < cx.size(); ++i) {
    const auto want = ddx::rt::evaluate_all(b, std::array{cx[i], cy[i]});
    EXPECT_EQ(std::signbit(got_max[i]), std::signbit(want[graph.outputs()[0]]))
        << "max at " << i;
    EXPECT_EQ(std::signbit(got_min[i]), std::signbit(want[graph.outputs()[1]]))
        << "min at " << i;
  }
  EXPECT_FALSE(std::signbit(got_max[0])) << "max(-0, +0) is +0";
  EXPECT_TRUE(std::signbit(got_min[0])) << "min(-0, +0) is -0";
}

// Every width computes the same bits: a lane is an IEEE operation on its own,
// and a transcendental is the same scalar libm call per lane.  Every op
// A graph with no Jacobian block at all, which is what Equation::evaluate()
// freezes: the kernel takes an empty partials span, and the emitter has to
// produce a function that stores value columns and nothing else.
TEST(JitValue, AValuesOnlyGraphCompilesAndStoresNoPartials) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto graph =
      ddx::rt::GraphBuilder{b}.value(x * log(x) + exp(x * y)).finish();
  ASSERT_EQ(graph.layout().jacobian, 0u);

  constexpr std::size_t n = 5;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.1 * static_cast<double>(i);
    cy[i] = 1.1 - 0.05 * static_cast<double>(i);
  }
  const std::array<const double *, 2> xs{cx.data(), cy.data()};

  const auto kernel = must_compile(graph);
  std::vector<double> value(n);
  double *const values[]{value.data()};
  kernel(xs, values, {}, {}, n);

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(value[i], cx[i] * std::log(cx[i]) + std::exp(cx[i] * cy[i]),
                1e-12)
        << "value at " << i;
  }
}

// the emitter has, on a batch that is not a multiple of any width.
TEST(JitValue, EveryLaneWidthAgreesToTheBit) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto f = x / (y + 1.0) - x * y;
#define DDX_TEST_OP(fn, Op, label, ...) f = f + fn(x * 0.5);
  DDX_UNARY_MATH_TABLE(DDX_TEST_OP)
#undef DDX_TEST_OP
  f = f + pow(x, y) + atan2(x, y) + hypot(x, y) + abs(x - y) + max(x, y) +
      min(x, y) + sign(x - y);
  const auto graph =
      ddx::rt::GraphBuilder{b}.value(f).build_jacobian().finish();

  constexpr std::size_t n = 13;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.2 + 0.05 * static_cast<double>(i);
    cy[i] = 0.9 - 0.03 * static_cast<double>(i);
  }
  const std::array<const double *, 2> xs{cx.data(), cy.data()};

  const auto run = [&](ddx::jit::Lanes lanes) {
    ddx::jit::Options o;
    o.codegen.lanes = lanes;
    const auto kernel = must_compile(graph, o);
    std::vector<double> value(n), dx(n), dy(n);
    double *const values[]{value.data()};
    const std::array<double *, 2> partials{dx.data(), dy.data()};
    kernel(xs, values, partials, {}, n);
    return std::tuple{value, dx, dy};
  };

  const auto [v1, dx1, dy1] = run(ddx::jit::Lanes::scalar());
  for (const auto lanes :
       {*ddx::jit::Lanes::exactly(2), *ddx::jit::Lanes::exactly(4),
        *ddx::jit::Lanes::exactly(8), ddx::jit::Lanes::derived()}) {
    const auto [v, dx, dy] = run(lanes);
    const std::string width = spelled(lanes);
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(v[i]),
                std::bit_cast<std::uint64_t>(v1[i]))
          << "value, lanes " << width << " at " << i;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(dx[i]),
                std::bit_cast<std::uint64_t>(dx1[i]))
          << "d/dx, lanes " << width << " at " << i;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(dy[i]),
                std::bit_cast<std::uint64_t>(dy1[i]))
          << "d/dy, lanes " << width << " at " << i;
    }
  }
}

// The codegen level rides on the module, so one JIT serves compiles at several,
// and all four must agree to the bit -- which is what lets the default be the
// cheapest.  Level 0 qualifies only because contraction moved into the graph:
// FastISel forms no FMAs of its own but an llvm.fma it is handed is an
// operation like any other.
TEST(JitValue, EveryCodegenLevelAgreesToTheBit) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  // Multiply-add shapes throughout, so every level that forms an FMA does.
  const auto f = x * y + x * x - y + exp(x * y) * (y + 1.0) + log(x) * y;
  const auto graph =
      ddx::rt::GraphBuilder{b}.value(f).build_jacobian().finish();

  const std::array cx{0.25, 1.5, 2.75, 3.0};
  const std::array cy{1.25, 0.5, 2.0, 0.75};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};

  const auto run = [&](ddx::jit::OptLevel level) {
    ddx::jit::Options o;
    o.codegen.codegen_level = level;
    const auto kernel = must_compile(graph, o);
    std::vector<double> value(cx.size()), dx(cx.size()), dy(cx.size());
    double *const values[]{value.data()};
    const std::array<double *, 2> partials{dx.data(), dy.data()};
    kernel(xs, values, partials, {}, cx.size());
    return std::tuple{value, dx, dy};
  };

  const auto [v1, dx1, dy1] = run(ddx::jit::OptLevel::O1);
  for (const auto level : {ddx::jit::OptLevel::O0, ddx::jit::OptLevel::O2,
                           ddx::jit::OptLevel::O3}) {
    const auto [v, dx, dy] = run(level);
    const auto at = std::to_underlying(level);
    for (std::size_t i = 0; i < cx.size(); ++i) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(v[i]),
                std::bit_cast<std::uint64_t>(v1[i]))
          << "value, codegen " << at << " at " << i;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(dx[i]),
                std::bit_cast<std::uint64_t>(dx1[i]))
          << "d/dx, codegen " << at << " at " << i;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(dy[i]),
                std::bit_cast<std::uint64_t>(dy1[i]))
          << "d/dy, codegen " << at << " at " << i;
    }
  }
}

// Both vectorisers pack operations that were already independent and neither is
// given `reassoc`, so neither may move a bit.  This is the gate on turning
// either on.
TEST(JitValue, TheVectorisersDoNotMoveABit) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * log(x) + y * exp(x * y) + sqrt(x + y) + x * y * (x + y);
  const auto graph =
      ddx::rt::GraphBuilder{b}.value(f).build_jacobian().finish();

  constexpr std::size_t n = 9;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.05 * static_cast<double>(i);
    cy[i] = 0.9 - 0.03 * static_cast<double>(i);
  }
  const std::array<const double *, 2> xs{cx.data(), cy.data()};

  const auto run = [&](ddx::jit::Options o) {
    const auto kernel = must_compile(graph, o);
    std::vector<double> value(n), dx(n), dy(n);
    double *const values[]{value.data()};
    const std::array<double *, 2> partials{dx.data(), dy.data()};
    kernel(xs, values, partials, {}, n);
    return std::tuple{value, dx, dy};
  };

  for (const auto lanes :
       {ddx::jit::Lanes::scalar(), *ddx::jit::Lanes::exactly(4)}) {
    const auto [v0, dx0, dy0] = run({.codegen = {.lanes = lanes}});
    const std::string width = spelled(lanes);
    for (const auto &[what, o] :
         {std::pair{"slp_vectorize",
                    ddx::jit::Options{
                        .codegen = {.lanes = lanes, .slp_vectorize = true}}},
          std::pair{"loop_vectorize",
                    ddx::jit::Options{.codegen = {.lanes = lanes,
                                                  .loop_vectorize = true}}}}) {
      const auto [v, dx, dy] = run(o);
      for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(v[i]),
                  std::bit_cast<std::uint64_t>(v0[i]))
            << what << ", lanes " << width << ", value at " << i;
        EXPECT_EQ(std::bit_cast<std::uint64_t>(dx[i]),
                  std::bit_cast<std::uint64_t>(dx0[i]))
            << what << ", lanes " << width << ", d/dx at " << i;
        EXPECT_EQ(std::bit_cast<std::uint64_t>(dy[i]),
                  std::bit_cast<std::uint64_t>(dy0[i]))
            << what << ", lanes " << width << ", d/dy at " << i;
      }
    }
  }
}

TEST(JitValue, SharedAndNested) {
  expect_matches_interpreter(
      [](Arena<> &, auto &v) {
        return (v[0] * v[1]) * (v[0] * v[1]) + sin(v[0] * v[1]);
      },
      2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) { return sin(cos(exp(v[0] * v[1]))); }, 2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) {
        auto t = v[0];
        for (int i = 0; i < 6; ++i) {
          t = sin(t * v[1]) + exp(t / 3.0);
        }
        return t;
      },
      2);
  expect_matches_interpreter(
      [](Arena<> &, auto &v) {
        return exp(v[0] * v[1]) + log(v[2]) / tanh(v[3]);
      },
      4);
}

TEST(JitValue, AgreesWithDdxThroughTheBridge) {
  constexpr auto x = ddx::var<"x">;
  constexpr auto y = ddx::var<"y">;
  Arena<> b;
  const auto root = ddx::rt::to_graph(b, exp(x) * sin(y));
  const auto graph = Graph<>::freeze(b, std::array{root.id(b)});
  const auto kernel = must_compile(graph);

  const std::array cx{1.0, 0.25, 2.5};
  const std::array cy{2.0, 1.75, 0.5};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};
  std::array<double, 3> got{};
  double *const value_columns[]{got.data()};
  kernel(xs, value_columns, {}, {}, got.size());

  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], ddx::Equation{exp(x) * sin(y)}.evaluate(cx[i], cy[i]),
                1e-12);
  }
}

// rt::Equation::run() interprets when there is no kernel, so an empty Kernel
// must be falsy and bring-up must report its refusal rather than throw.
TEST(JitValue, AnUnbuiltKernelIsFalsyAndCreateAnswersWithAResult) {
  const ddx::jit::Kernel none;
  EXPECT_FALSE(static_cast<bool>(none));
  EXPECT_EQ(none.shape().arity, 0u);
  EXPECT_EQ(none.shape().outputs(), 0u);

  auto c = ddx::jit::Compiler::create();
  ASSERT_TRUE(c.has_value()) << c.error().detail;

  Arena<> b;
  const auto x = var(b, "x");
  auto k = c->compile(Graph<>::freeze(b, std::array{(x + x).id(b)}));
  ASSERT_TRUE(k.has_value()) << k.error().detail;
  EXPECT_TRUE(static_cast<bool>(*k));
}

TEST(JitValue, EmptyBatchWritesNothing) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{sin(x).id(b)});
  const auto kernel = must_compile(graph);

  const double *none = nullptr;
  const std::array<const double *, 1> xs{none};
  double sentinel = 42.0;
  double *const value_columns[]{&sentinel};
  kernel(xs, value_columns, {}, {}, 0);
  EXPECT_DOUBLE_EQ(sentinel, 42.0);
}

TEST(JitValue, ConstantFoldsToAStore) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto f = x * ddx::rt::Expression<>{0} + ddx::rt::Expression<>{7};
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});
  const auto kernel = must_compile(graph);

  const std::array cx{1.0, 2.0};
  const std::array<const double *, 1> xs{cx.data()};
  std::array<double, 2> got{};
  double *const value_columns[]{got.data()};
  kernel(xs, value_columns, {}, {}, got.size());
  EXPECT_DOUBLE_EQ(got[0], 7.0);
  EXPECT_DOUBLE_EQ(got[1], 7.0);
}

// The object a compile produced links back and answers the same bits, which is
// what makes a compile storable.
TEST(JitValue, AnObjectAdoptsBackIntoTheSameKernel) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto root = (x * log(x) + exp(x * y) + sqrt(y)).id(b);
  const auto graph = ddx::rt::GraphBuilder<double>{b}
                         .values_from(std::array{root})
                         .build_jacobian()
                         .finish();

  const auto compiled = must_compile(graph);
  ASSERT_TRUE(compiled);
  ASSERT_FALSE(compiled.object().empty()) << "the compile kept no object";

  // Unguessable, which is why the kernel carries it: whoever stores the bytes
  // stores the symbol beside them.
  ASSERT_FALSE(compiled.symbol().empty());
  const auto adopted =
      compiler().adopt(compiled.object(), compiled.symbol(), graph);
  ASSERT_TRUE(adopted.has_value()) << adopted.error().detail;

  const std::array cx{0.4, 0.9}, cy{1.3, 0.7};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};
  std::array<double, 2> f1{}, dx1{}, dy1{}, f2{}, dx2{}, dy2{};
  double *const v1[]{f1.data()};
  double *const g1[]{dx1.data(), dy1.data()};
  double *const v2[]{f2.data()};
  double *const g2[]{dx2.data(), dy2.data()};
  compiled(xs, v1, g1, {}, 2);
  (*adopted)(xs, v2, g2, {}, 2);

  EXPECT_EQ(f1, f2) << "the adopted kernel moved a value";
  EXPECT_EQ(dx1, dx2) << "the adopted kernel moved d/dx";
  EXPECT_EQ(dy1, dy2) << "the adopted kernel moved d/dy";

  // And it carries its own bytes, so a kernel that came off disk can go back.
  EXPECT_EQ(adopted->object().size(), compiled.object().size());
}

// The second compile of the same graph does no codegen at all.  The report is
// the assertion, not timing: a hit does not run the phases at all.
TEST(JitValue, ACachedObjectSkipsTheWholeCompile) {
  const Scratch dir{"jit_skips"};

  const auto build = [] {
    Arena<> b;
    const auto x = var(b, "x");
    const auto y = var(b, "y");
    const auto root = (x * log(x) + exp(x * y) + sqrt(y)).id(b);
    return std::pair{std::move(b),
                     root}; // the arena has to outlive the freeze that reads it
  };

  ddx::jit::Options opt;
  opt.cache_dir = dir.string();

  ddx::jit::CompileReport cold, warm;
  double cold_f = 0, warm_f = 0;
  const std::array c{0.7};
  const std::array<const double *, 2> xs{c.data(), c.data()};

  for (const auto &[report, out] :
       {std::pair{&cold, &cold_f}, std::pair{&warm, &warm_f}}) {
    auto [b, root] = build();
    const auto graph = Graph<>::freeze(b, std::array{root});
    auto k = compiler().compile(graph, opt, report);
    ASSERT_TRUE(k.has_value()) << k.error().detail;
    double *const values[]{out};
    (*k)(xs, values, {}, {}, 1);
  }

  EXPECT_GT(cold.codegen.count(), 0) << "the cold compile did no codegen";
  EXPECT_EQ(warm.codegen.count(), 0) << "the warm compile ran the backend";
  EXPECT_EQ(warm.emit.count(), 0) << "the warm compile emitted IR";
  EXPECT_EQ(warm.optimize.count(), 0) << "the warm compile ran the passes";
  EXPECT_EQ(cold_f, warm_f) << "the cached kernel moved a bit";
}

// The checksum is what stands between a corrupt file and machine code being
// jumped into.
TEST(JitValue, ACorruptCacheEntryIsAMissNotACrash) {
  const Scratch dir{"jit_corrupt"};

  ddx::jit::Options opt;
  opt.cache_dir = dir.string();

  Arena<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{(x * x + log(x)).id(b)});
  ASSERT_TRUE(compiler().compile(graph, opt).has_value());

  // Every entry the cold compile wrote, damaged in the payload.
  std::size_t damaged = 0;
  for (const auto &e : std::filesystem::directory_iterator{dir}) {
    std::fstream f{e.path(), std::ios::in | std::ios::out | std::ios::binary};
    ASSERT_TRUE(f.is_open());
    f.seekp(static_cast<std::streamoff>(kHeaderBytes + 8));
    f.put('\xff');
    ++damaged;
  }
  ASSERT_GT(damaged, 0u) << "the cold compile cached nothing";

  ddx::jit::CompileReport after;
  const auto k = compiler().compile(graph, opt, &after);
  ASSERT_TRUE(k.has_value()) << "a corrupt entry failed the compile";
  EXPECT_GT(after.codegen.count(), 0) << "a corrupt entry was adopted";

  double got = 0;
  const std::array c{0.7};
  const std::array<const double *, 1> xs{c.data()};
  double *const values[]{&got};
  (*k)(xs, values, {}, {}, 1);
  EXPECT_NEAR(got, 0.7 * 0.7 + std::log(0.7), 1e-15);
}

// A cache entry carries lengths read out of its own payload -- the shape where
// a hand-picked corrupt case proves nothing.  Every prologue byte, a stride
// through the payload and every truncation: each must degrade to a miss.
TEST(JitValue, ACacheEntrySurvivesCorruptionAndTruncation) {
  const Scratch dir{"jit_fuzz"};

  ddx::jit::Options opt;
  opt.cache_dir = dir.string();

  Arena<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{(x * x + log(x)).id(b)});
  ASSERT_TRUE(compiler().compile(graph, opt).has_value());

  std::filesystem::path entry;
  for (const auto &e : std::filesystem::directory_iterator{dir}) {
    entry = e.path();
  }
  ASSERT_FALSE(entry.empty()) << "the cold compile cached nothing";
  const auto pristine = slurp(entry);
  ASSERT_GT(pristine.size(), kHeaderBytes);

  const std::array c{0.7};
  const std::array<const double *, 1> xs{c.data()};
  const double want = 0.7 * 0.7 + std::log(0.7);

  // The second has the teeth: "returns a correct kernel" is satisfied by an
  // entry adopted and harmless, where "the backend ran" says it was refused.
  const auto rejected = [&](const char *what, std::size_t at) {
    ddx::jit::CompileReport rep;
    const auto k = compiler().compile(graph, opt, &rep);
    ASSERT_TRUE(k.has_value()) << what << " at " << at << " failed the compile";
    EXPECT_GT(rep.codegen.count(), 0)
        << what << " at " << at << " was adopted rather than refused";
    double got = 0;
    double *const values[]{&got};
    (*k)(xs, values, {}, {}, 1);
    EXPECT_NEAR(got, want, 1e-15) << what << " at " << at << " ran wrong code";
  };

  // Every prologue byte under three masks, then a stride through the payload;
  // no stride on the prologue, the one region no checksum can cover.  A sweep
  // rather than a spot check, for the *next* field somebody adds and forgets.
  //
  // Three masks, not one: one detects only an entirely unverified field --
  // relax `scalar_size` to a range check and `~8 = 247` is still refused while
  // `8 -> 9` is adopted.  Whole-byte, low-bit, high-bit.
  //
  // Three, not 255, is a trade.  Exhaustive was run once, 0 of 14280 accepted,
  // at 98 s against an 18 s suite -- a rejection here provokes a real
  // recompile. Three is adequate only because every field is checked by exact
  // `!=`; a field added with a range check needs the exhaustive sweep again.
  const std::size_t stride = std::max<std::size_t>(1, pristine.size() / 64);
  for (std::size_t i = 0; i < kHeaderBytes; ++i) {
    for (const std::byte mask :
         {std::byte{0xff}, std::byte{0x01}, std::byte{0x80}}) {
      auto damaged = pristine;
      damaged[i] ^= mask;
      if (damaged[i] == pristine[i]) {
        continue;
      }
      dump(entry, damaged);
      rejected("a flipped prologue byte", i);
    }
  }
  for (std::size_t i = kHeaderBytes; i < pristine.size(); i += stride) {
    auto damaged = pristine;
    damaged[i] = ~damaged[i];
    dump(entry, damaged);
    rejected("a flipped payload byte", i);
  }

  for (std::size_t n = 0; n < pristine.size(); n += stride) {
    dump(entry, std::span<const std::byte>{pristine}.first(n));
    rejected("a truncation to", n);
  }

  // What the sweep cannot reach: an enormous symbol length, the case damage
  // cannot produce because it fails the checksum first.  The entry is *rebuilt*
  // with a CRC that agrees -- which is what makes this not vacuous.  CRC32
  // detects accidents, it does not sign anything.
  {
    auto forged = pristine;
    std::ranges::fill_n(forged.begin() + 24, 4, std::byte{0xff}); // the length
    boost::crc_32_type crc;
    crc.process_bytes(forged.data() + 24, forged.size() - 24);
    const auto sum = boost::endian::native_to_little(crc.checksum());
    std::memcpy(forged.data() + 20, &sum, sizeof sum);
    dump(entry, forged);
    rejected("an impossible symbol length whose CRC agrees", 0);
  }
}

// The multi-writer case: a parallel build, or two workers warming one key, all
// missing together and writing one path at once.  Each must get a correct
// kernel, one whole entry must be left, and nothing may be left staged.
//
// It is *not* a regression test for the cache's per-writer staging name.
// Measured against the shared ".tmp" it replaced, this passes 0/10 while 120 of
// 160 cache writes fail -- so the writers collide constantly, and what hides it
// is that `write_entry` discards a losing rename's error by design: a cache
// that cannot be written is a cache that misses.  Nor is there a torn file to
// find at any size, rename being atomic.  What it covers is the end of the
// pipe: whatever the writers did, one whole entry loads afterwards.
TEST(JitValue, ConcurrentCompilesLeaveOneWholeCacheEntry) {
  const Scratch dir{"jit_writers"};

  ddx::jit::Options opt;
  opt.cache_dir = dir.string();

  constexpr std::size_t writers = 8;
  const double want = 0.7 * 0.7 + std::log(0.7);
  std::array<double, writers> got{};
  std::array<bool, writers> ok{};
  std::array<bool, writers> compiled{};

  // Graphs built *before* the rendezvous and compiles started after it, so
  // every writer reads the cache before any has written -- which makes "all
  // eight miss" structural rather than lucky.
  std::latch start{writers};

  {
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < writers; ++t) {
      threads.emplace_back([&, t] {
        // Its own arena: one Arena is not shared between threads.  Two
        // independent builds hashing alike is also the cache's premise.
        Arena<> b;
        const auto x = var(b, "x");
        const auto graph =
            Graph<>::freeze(b, std::array{(x * x + log(x)).id(b)});
        start.arrive_and_wait();
        ddx::jit::CompileReport rep;
        const auto k = compiler().compile(graph, opt, &rep);
        if (!k) {
          return;
        }
        compiled[t] = rep.codegen.count() > 0;
        const std::array c{0.7};
        const std::array<const double *, 1> xs{c.data()};
        double *const values[]{&got[t]};
        (*k)(xs, values, {}, {}, 1);
        ok[t] = true;
      });
    }
    for (auto &th : threads) {
      th.join();
    }
  }

  // Without this, writers serialising into one compile and seven hits would
  // leave every assertion below green.  An assertion about *work done*, not
  // timing -- which the latch above is what makes safe.
  EXPECT_EQ(std::ranges::count(compiled, true), writers)
      << "a writer hit the cache, so the writers did not all overlap";

  for (std::size_t t = 0; t < writers; ++t) {
    EXPECT_TRUE(ok[t]) << "writer " << t << " got no kernel";
    EXPECT_NEAR(got[t], want, 1e-15) << "writer " << t << " ran wrong code";
  }

  // One key, one entry, and nothing left staged by a losing rename.
  std::size_t entries = 0;
  for (const auto &e : std::filesystem::directory_iterator{dir}) {
    EXPECT_EQ(e.path().extension(), ".ddxjit")
        << "left staged: " << e.path().filename();
    ++entries;
  }
  EXPECT_EQ(entries, 1u) << "one graph and one option set should be one entry";

  // And what landed is a whole entry, not a mixture of eight.
  Arena<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{(x * x + log(x)).id(b)});
  ddx::jit::CompileReport warm;
  const auto k = compiler().compile(graph, opt, &warm);
  ASSERT_TRUE(k.has_value()) << k.error().detail;
  EXPECT_EQ(warm.codegen.count(), 0) << "what the writers left did not load";
}

// No directory named, nothing written.
TEST(JitValue, NoCacheDirectoryWritesNothing) {
  const Scratch dir{"jit_absent"};

  Arena<> b;
  const auto x = var(b, "x");
  ASSERT_TRUE(compiler()
                  .compile(Graph<>::freeze(b, std::array{(x * x).id(b)}))
                  .has_value());
  EXPECT_FALSE(std::filesystem::exists(dir));
}

TEST(JitValue, AForgedSymbolCannotReachThroughTheLinkOrder) {
  Arena<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{(x * x + exp(x)).id(b)});
  const auto compiled = must_compile(graph);
  ASSERT_TRUE(compiled);
  ASSERT_FALSE(compiled.object().empty());

  // Its own symbol links, so the negatives below are not vacuous.
  EXPECT_TRUE(compiler()
                  .adopt(compiled.object(), compiled.symbol(), graph)
                  .has_value());

  // Every one resolves in the main dylib -- libm through define_libm, memcpy
  // through the process generator -- and the object defines none of them.
  for (const std::string_view forged :
       {"sin", "exp", "log", "memcpy", "ddx_kernel_999"}) {
    EXPECT_FALSE(compiler().adopt(compiled.object(), forged, graph).has_value())
        << forged << " resolved through the link order";
  }
}

// A cache entry that will not link is a miss, not a crash and not an abort.
TEST(JitValue, AdoptingRubbishIsAnErrorNotACrash) {
  const std::array<std::byte, 8> rubbish{
      std::byte{0x7f}, std::byte{'E'}, std::byte{'L'}, std::byte{'F'},
      std::byte{0},    std::byte{0},   std::byte{0},   std::byte{0}};
  Arena<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{(x * x).id(b)});
  const auto k = compiler().adopt(rubbish, "ddx_kernel_0", graph);
  EXPECT_FALSE(k.has_value()) << "a truncated object linked";
}

TEST(JitValue, SeparateCompilesCoexist) {
  Arena<> b1;
  const auto x1 = var(b1, "x");
  const auto k1 = must_compile(Graph<>::freeze(b1, std::array{sin(x1).id(b1)}));

  Arena<> b2;
  const auto x2 = var(b2, "x");
  const auto k2 = must_compile(Graph<>::freeze(b2, std::array{cos(x2).id(b2)}));

  const std::array c{0.5};
  const std::array<const double *, 1> xs{c.data()};
  double a = 0, bb = 0;
  double *const first[]{&a};
  double *const second[]{&bb};
  k1(xs, first, {}, {}, 1);
  k2(xs, second, {}, {}, 1);
  EXPECT_NEAR(a, std::sin(0.5), 1e-15);
  EXPECT_NEAR(bb, std::cos(0.5), 1e-15);
}

// A Kernel shares ownership of the JIT its code lives in; without that share
// this call lands in an unmapped page.
TEST(JitValue, KernelOutlivesItsCompiler) {
  ddx::jit::Kernel kernel;
  {
    auto local = *ddx::jit::Compiler::create();
    Arena<> b;
    const auto x = var(b, "x");
    auto k = local.compile(Graph<>::freeze(b, std::array{(x * x).id(b)}));
    ASSERT_TRUE(k.has_value()) << k.error().detail;
    kernel = std::move(*k);
  }
  ASSERT_TRUE(static_cast<bool>(kernel));

  const std::array c{3.0};
  const std::array<const double *, 1> xs{c.data()};
  double got = 0;
  double *const value_columns[]{&got};
  kernel(xs, value_columns, {}, {}, 1);
  EXPECT_DOUBLE_EQ(got, 9.0);
}

} // namespace
