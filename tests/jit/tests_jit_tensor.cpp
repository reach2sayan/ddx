#include "rt/equation.hpp"
#include "rt/tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <functional>
#include <initializer_list>
#include <latch>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// The kernel's tensor loops sum in their own order, so it matches the sweep to
// a few ulps for every Want, every width and both scalars.
namespace jit_tensor {
namespace {
using ddx::rt::OpCode;
using ddx::rt::Want;

template <typename T>
std::vector<T> table(std::size_t count, std::size_t seed, double lo,
                     double hi) {
  return std::views::iota(0uz, count) |
         std::views::transform([=](std::size_t i) {
           return static_cast<T>(
               lo + (hi - lo) *
                        static_cast<double>((i * 37 + seed * 11) % 101) /
                        100.0);
         }) |
         ddx::impl::to<std::vector>();
}

// Gemv in both orientations off tables, plus a Dot, a Sum, Maps and Zips.
template <typename T> auto msevec(std::size_t n) {
  return ddx::rt::equation<T>([n] {
    using E = ddx::rt::Expression<T>;
    const E x = ddx::rt::vec<T>("x", n);
    const E r = ddx::rt::vec_constant<T>(table<T>(n, 1, 0.9, 5.2));
    const E q = ddx::rt::vec_constant<T>(table<T>(n, 2, 0.8, 4.4));
    const E ionic = dot(ddx::rt::vec_constant<T>(table<T>(n, 5, 0.1, 2.0)), x);
    const E lr = T(-0.3915) * sqrt(ionic) / (E{T(1)} + T(1.2) * sqrt(ionic));
    const E mr =
        dot(x,
            gemv(ddx::rt::mat_constant<T>(n, n, table<T>(n * n, 3, -0.4, 0.4)),
                 x)) *
        exp(-sqrt(ionic));
    const E phi = (r * x) / dot(r, x);
    const E theta = (q * x) / dot(q, x);
    const E inner =
        gemv(ddx::rt::mat_constant<T>(n, n, table<T>(n * n, 4, 0.2, 1.8)),
             theta, true);
    return lr + mr + sum(x * log(phi / x)) - sum(q * x * log(inner));
  });
}

// A matrix computed from a parameter, read in both orientations.
template <typename T> auto msevec_t(std::size_t n) {
  return ddx::rt::equation<T>([n] {
    using E = ddx::rt::Expression<T>;
    const E x = ddx::rt::vec<T>("x", n);
    const E t = ddx::rt::param<T>("T");
    const E tau =
        exp(-t * ddx::rt::mat_constant<T>(n, n, table<T>(n * n, 6, -0.2, 0.5)));
    return dot(x, gemv(tau, x)) + sum(x * log(gemv(tau, x, true)));
  });
}

// The ops a dense Hessian block is built from, with the Outer taken of a
// parameter so that it does not fold to a table, and summed with one so that
// the gemv reads it whole rather than as r (P r . x).
template <typename T> ddx::rt::Expression<T> dense_block(std::size_t n) {
  using E = ddx::rt::Expression<T>;
  const E x = ddx::rt::vec<T>("x", n);
  const E t = ddx::rt::var<T>("T");
  const auto m = [n](std::size_t seed) {
    return ddx::rt::mat_constant<T>(n, n, table<T>(n * n, seed, 0.2, 1.8));
  };
  const E r = ddx::rt::vec_constant<T>(table<T>(n, 10, 0.5, 1.5));
  const E s = dot(x, x);
  return dot(x, gemv(m(7), x * x)) + sum(log(x) * gemv(m(8), x, true)) -
         sum(x * log(gemv(m(9), x / sum(x), true))) +
         dot(x, gemv(ddx::rt::outer(r, ddx::rt::param<T>("P") * r) + m(10),
                     x)) +
         s * s * sum(x) +
         t * sum(x * log(x)) + t * log(t);
}
template <typename T> auto dense(std::size_t n) {
  return ddx::rt::equation<T>([n] { return dense_block<T>(n); });
}

// x^T A x over a table, whose dense block folds to a table with no node over
// it.
template <typename T> auto quadratic(std::size_t n) {
  return ddx::rt::equation<T>([n] {
    using E = ddx::rt::Expression<T>;
    const E x = ddx::rt::vec<T>("x", n);
    return dot(x, gemv(ddx::rt::mat_constant<T>(n, n, table<T>(n * n, 11, -0.5, 1.5)), x));
  });
}

// Selects on a scalar condition and on a vector one.
template <typename T> auto vselect(std::size_t n) {
  return ddx::rt::equation<T>([n] {
    using E = ddx::rt::Expression<T>;
    const E x = ddx::rt::vec<T>("x", n);
    return sum(select(x > T(0.4), x * log(x), x * x)) +
           sum(select(x[0] > T(0.3), log(x), exp(x * T(0.5))));
  });
}

template <typename T> using Columns = std::vector<std::vector<T>>;

// Builds `count` columns of n points inside every model's domain.
template <typename T>
Columns<T> points(std::size_t count, std::size_t n, std::size_t seed) {
  return std::views::iota(0uz, count) |
         std::views::transform([=](std::size_t j) {
           return std::views::iota(0uz, n) |
                  std::views::transform([=](std::size_t p) {
                    return static_cast<T>(
                        0.11 +
                        0.8 * static_cast<double>((j * 7 + p * 3 + seed) % 17) /
                            17.0);
                  }) |
                  ddx::impl::to<std::vector>();
         }) |
         ddx::impl::to<std::vector>();
}

template <typename T> auto pointers(Columns<T> &c) {
  return c | std::views::transform([](auto &v) { return v.data(); }) |
         ddx::impl::to<std::vector>();
}
template <typename T> auto pointers(const Columns<T> &c) {
  return c | std::views::transform([](const auto &v) { return v.data(); }) |
         ddx::impl::to<std::vector>();
}

// Runs one batch of `want` and returns every output column it writes.
template <typename T>
Columns<T> batch(const auto &eq, Want want, std::size_t n) {
  const std::size_t inputs = *eq.input_count();
  const std::size_t symbols = eq.symbols()->size();
  const std::size_t m = *eq.outputs();
  const Columns<T> xs = points<T>(inputs, n, 0);
  const Columns<T> v = points<T>(symbols, n, 5);
  const Columns<T> w = points<T>(m, n, 9);
  const std::size_t jac = *eq.jacobian_columns();
  const auto blocks = [&]() -> std::array<std::size_t, 3> {
    switch (want) {
    case Want::Value:
      return {m, 0, 0};
    case Want::Gradient:
      return {0, jac, 0};
    case Want::Jacobian:
      return {m, jac, 0};
    case Want::Hessian:
      return {m, jac, *eq.hessian_columns()};
    case Want::Hvp:
      return {m, jac, symbols};
    case Want::Vjp:
      return {m, *eq.vjp_columns(), 0};
    case Want::Jvp:
      return {m, *eq.jvp_columns(), 0};
    case Want::JacobianJvp:
      return {m, jac, jac};
    case Want::DenseHessian:
      return {m, jac, symbols * symbols};
    }
    std::unreachable();
  }();
  std::array<Columns<T>, 3> out;
  for (const auto [o, count] : std::views::zip(out, blocks)) {
    o.assign(count, std::vector<T>(n));
  }
  const auto x = pointers(xs);
  const auto f = pointers(out[0]);
  const auto g = pointers(out[1]);
  const auto h = pointers(out[2]);
  const auto ok = [&] {
    switch (want) {
    case Want::Value:
      return eq.evaluate(x, f, n);
    case Want::Gradient:
      return eq.gradient(x, g, n);
    case Want::Jacobian:
      return eq.jacobian(x, f, g, n);
    case Want::Hessian:
      return eq.hessian(x, f, g, h, n);
    case Want::Hvp:
      return eq.hvp(x, pointers(v), f, g, h, n);
    case Want::Vjp:
      return eq.vjp(x, pointers(w), f, g, n);
    case Want::Jvp:
      return eq.jvp(x, pointers(v), f, g, n);
    case Want::JacobianJvp:
      return eq.jacobian_jvp(x, pointers(v), f, g, h, n);
    case Want::DenseHessian:
      return eq.dense_hessian(x, f, g, h, n);
    }
    std::unreachable();
  }();
  EXPECT_TRUE(ok.has_value()) << ddx::rt::name_of(want);
  return std::views::join(out) | ddx::impl::to<std::vector>();
}

// The kernel and the sweep associate sums differently, so the error is relative
// to the larger magnitude, with a floor for results near zero.
template <typename T>
void expect_close(const Columns<T> &want, const Columns<T> &got,
                  const std::string &what) {
  const double tol = sizeof(T) == 8 ? 1e-12 : 1e-4;
  ASSERT_EQ(want.size(), got.size()) << what;
  for (const auto [c, pair] :
       std::views::zip(want, got) | std::views::enumerate) {
    const auto &[w, g] = pair;
    ASSERT_EQ(w.size(), g.size()) << what;
    for (const auto [p, wg] : std::views::zip(w, g) | std::views::enumerate) {
      const double a = std::get<0>(wg);
      const double b = std::get<1>(wg);
      EXPECT_LE(std::abs(a - b), tol * std::max({1.0, std::abs(a), std::abs(b)}))
          << what << ", column " << c << ", point " << p << ": " << a << " vs " << b;
    }
  }
}

std::string spelled(ddx::jit::Lanes lanes) {
  return lanes.stated() ? std::to_string(*lanes.stated()) : "host";
}

const auto every_want = ddx::rt::want_values;
const std::initializer_list<ddx::jit::Lanes> every_width{
    ddx::jit::Lanes::scalar(), *ddx::jit::Lanes::exactly(2),
    *ddx::jit::Lanes::exactly(4), *ddx::jit::Lanes::exactly(8),
    ddx::jit::Lanes::derived()};

ddx::jit::Options compiled(ddx::jit::Lanes lanes) {
  return {.backend = ddx::jit::Backend::Compile,
          .points = 64,
          .codegen = {.lanes = lanes}};
}

// Checks every Want at every width against the sweep, on a ragged batch and on
// a single point.
template <typename T> void expect_kernel_is_the_sweep(auto &&model) {
  const auto swept = model();
  ASSERT_FALSE(swept.poisoned()) << swept.status()->code;
  for (const auto lanes : every_width) {
    auto eq = model();
    eq.options(compiled(lanes));
    for (const Want want : every_want) {
      const std::string what =
          std::string{ddx::rt::name_of(want)} + ", lanes " + spelled(lanes);
      ASSERT_TRUE(eq.uses_kernel(want)) << what;
      for (const std::size_t n : {13uz, 1uz}) {
        expect_close(batch<T>(swept, want, n), batch<T>(eq, want, n),
                    what + ", n " + std::to_string(n));
      }
    }
  }
}

// Collects the ops that a model's dense block reads.
std::set<OpCode> dense_ops(auto &&expression) {
  ddx::rt::Arena<> b;
  const auto arena = ddx::rt::detail::scoped_arena(b);
  const auto f = expression();
  const auto h = ddx::rt::derive_dense_hessian(b, f.id(b));
  const auto live = ddx::rt::detail::reachable(b, std::span{&h.block, 1});
  return ddx::rt::detail::live_ids(live) |
         std::views::transform([&b](ddx::rt::NodeId v) { return b.op_of(v); }) |
         ddx::impl::to<std::set>();
}

} // namespace
} // namespace jit_tensor

using namespace jit_tensor;

TEST(JitTensor, CompilesAndAnswersTheSweep) {
  constexpr std::size_t n = 4;
  const auto model = [] {
    return ddx::rt::equation([] {
      const auto x = ddx::rt::vec("x", n);
      const std::vector<double> m{1.0, 0.5, -0.25, 2.0, 0.0, 1.5,  0.75, -1.0,
                                  0.3, 0.6, 0.9,   1.2, 2.1, -0.4, 0.8,  0.1};
      return dot(x, gemv(ddx::rt::mat_constant(n, n, m), x)) + sum(log(x));
    });
  };
  const auto swept = model();
  auto eq = model();
  eq.options({.backend = ddx::jit::Backend::Compile});
  EXPECT_TRUE(eq.uses_kernel());
  for (const std::size_t p : std::views::iota(0uz, 37uz)) {
    const auto at = std::views::iota(0uz, n) |
                    std::views::transform([p](std::size_t i) {
                      return 0.2 + 0.03 * static_cast<double>(p + i);
                    }) |
                    ddx::impl::to<std::vector>();
    EXPECT_NEAR(*eq.evaluate(at), *swept.evaluate(at), 1e-12);
    const auto g = *eq.gradient(at);
    for (const auto [a, b] : std::views::zip(g, *swept.gradient(at))) {
      EXPECT_NEAR(a, b, 1e-12);
    }
  }
}

TEST(JitTensor, MsevecEveryWantEveryWidth) {
  expect_kernel_is_the_sweep<double>([] { return msevec<double>(6); });
  expect_kernel_is_the_sweep<float>([] { return msevec<float>(6); });
}

TEST(JitTensor, ComputedMatrixEveryWantEveryWidth) {
  expect_kernel_is_the_sweep<double>([] { return msevec_t<double>(5); });
  expect_kernel_is_the_sweep<float>([] { return msevec_t<float>(5); });
}

TEST(JitTensor, DenseBlockEveryWantEveryWidth) {
  constexpr std::size_t n = 5;
  const auto ops = dense_ops([] { return dense_block<double>(n); });
  for (const OpCode op :
       {OpCode::Diag, OpCode::Outer, OpCode::RowScale, OpCode::ColScale,
        OpCode::MatMul, OpCode::StackRows, OpCode::RowOf}) {
    EXPECT_TRUE(ops.contains(op)) << ddx::rt::label_of(op);
  }
  expect_kernel_is_the_sweep<double>([] { return dense<double>(n); });
  expect_kernel_is_the_sweep<float>([] { return dense<float>(n); });
}

TEST(JitTensor, ATableAloneAsTheDenseBlockEveryWantEveryWidth) {
  expect_kernel_is_the_sweep<double>([] { return quadratic<double>(5); });
  expect_kernel_is_the_sweep<float>([] { return quadratic<float>(5); });
}

TEST(JitTensor, VectorSelectEveryWantEveryWidth) {
  expect_kernel_is_the_sweep<double>([] { return vselect<double>(6); });
  expect_kernel_is_the_sweep<float>([] { return vselect<float>(6); });
}

TEST(JitTensor, BatchesAndThreadsAreBitIdentical) {
  auto eq = msevec<double>(7);
  eq.options(compiled(ddx::jit::Lanes::derived()));
  ASSERT_TRUE(eq.prepare({Want::Jacobian, Want::Hvp}).has_value());
  ASSERT_TRUE(eq.uses_kernel(Want::Hvp));
  const auto call = [&] {
    return std::pair{batch<double>(eq, Want::Jacobian, 29),
                     batch<double>(eq, Want::Hvp, 29)};
  };
  const auto want = call();
  constexpr std::size_t threads = 8;
  std::latch start{threads};
  std::vector<char> ok(threads, 0);
  {
    std::vector<std::jthread> pool;
    for (const std::size_t t : std::views::iota(0uz, threads)) {
      pool.emplace_back([&, t] {
        start.arrive_and_wait();
        ok[t] = call() == want;
      });
    }
  }
  EXPECT_TRUE(std::ranges::all_of(ok, std::identity{}));
}

// The kernel never forms this matrix, so a gathered element must still map its
// flat index to the same row and column as the sweep.
TEST(JitTensor, AGatherFromAFusedMatrixReadsItsRowAndColumn) {
  using RE = ddx::rt::Expression<double>;
  const auto check = [](auto element, double want) {
    const auto swept = ddx::rt::equation(element);
    auto compiled = ddx::rt::equation(element);
    compiled.options({.backend = ddx::jit::Backend::Compile, .points = 1});
    ASSERT_TRUE(compiled.uses_kernel());
    const std::array<double, 5> at{2.0, 3.0, 5.0, 7.0, 11.0};
    const std::span<const double> point{at};
    EXPECT_DOUBLE_EQ(*swept.evaluate(point), want);
    EXPECT_DOUBLE_EQ(*compiled.evaluate(point), want);
    EXPECT_EQ(*compiled.jacobian(point), *swept.jacobian(point));
  };
  // In column-major order, element 4 of a 3 x 2 matrix is (1, 1).
  check(
      [] -> RE {
        const RE u = ddx::rt::vec("u", 3);
        const RE w = ddx::rt::vec("w", 2);
        return ddx::rt::outer(u, w)[4];
      },
      3.0 * 11.0);
  check(
      [] -> RE {
        const RE u = ddx::rt::vec("u", 3);
        const RE w = ddx::rt::vec("w", 2);
        return exp(ddx::rt::outer(u, w))[4];
      },
      std::exp(3.0 * 11.0));
}

// The kernel is checked against the sweep's layout, so rebalanced roots must
// keep that layout cell for cell even where their sums fold differently.
TEST(JitRebalance, AKernelKeepsTheLayoutOfTheSweep) {
  using RE = ddx::rt::Expression<double>;
  constexpr std::size_t count = 16;
  const auto names = [](std::size_t i) { return std::format("y{}", i); };
  const auto check = [&](auto model) {
    const auto swept = ddx::rt::equation(model);
    auto compiled = ddx::rt::equation(model);
    compiled.options({.backend = ddx::jit::Backend::Compile, .points = 1});
    ASSERT_TRUE(compiled.uses_kernel());
    const auto at = std::views::iota(0uz, *swept.input_count()) |
                    std::views::transform([](std::size_t i) {
                      return 0.5 + 0.125 * static_cast<double>(i);
                    }) |
                    ddx::impl::to<std::vector>();
    const std::span<const double> point{at};
    const auto want = *swept.jacobian(point);
    const auto got = *compiled.jacobian(point);
    ASSERT_EQ(got.size(), want.size());
    for (const auto [g, w] : std::views::zip(got, want)) {
      EXPECT_NEAR(g, w, 1e-9);
    }
  };
  // Two spellings of one sum become one node, and a/b then folds to 1.
  check([&] {
    const auto y = std::views::iota(0uz, count) |
                   std::views::transform([&](std::size_t i) {
                     return ddx::rt::var(names(i));
                   }) |
                   ddx::impl::to<std::vector>();
    RE chain = y[0];
    for (const RE &term : y | std::views::drop(1)) {
      chain = chain + term;
    }
    RE folded{0};
    for (const RE &term : y) {
      folded += term;
    }
    return std::vector{chain / folded, chain};
  });
  // A partial whose constant adjoints cancel in only one summation order.
  check([&]() -> RE {
    const RE x = ddx::rt::var("x");
    const double big = 9007199254740992.0;
    const RE y0 = ddx::rt::var(names(0));
    const RE y1 = ddx::rt::var(names(1));
    RE f = y0 + y1 + x;
    f = f + (-big) * x;
    f = f + big * x;
    for (const std::size_t i : std::views::iota(2uz, count)) {
      f = f + ddx::rt::var(names(i));
    }
    return f;
  });
}
