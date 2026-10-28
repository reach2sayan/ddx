#include "rt/equation.hpp"
#include "rt/kernels.hpp"
#include "rt/tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
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

// A tensor row compiles to a call into the sweep's own step, so the kernel
// answers the sweep's bits: every Want, every width, both scalars, the table
// the equation was given.
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
         std::ranges::to<std::vector>();
}

// Gemv both ways off tables, a Dot, a Sum, Maps and Zips.
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

// A matrix computed from a parameter, read both ways.
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

// What the dense Hessian's block is built from: a Diag, an Outer, a RowScale
// and a ColScale, a MatMul over StackRows, and RowOf for the scalar symbol
// beside the vector.
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
         dot(x, gemv(ddx::rt::outer(r, r * r), x)) + s * s * sum(x) +
         t * sum(x * log(x)) + t * log(t);
}
template <typename T> auto dense(std::size_t n) {
  return ddx::rt::equation<T>([n] { return dense_block<T>(n); });
}

// A scalar condition and a vector one.
template <typename T> auto vselect(std::size_t n) {
  return ddx::rt::equation<T>([n] {
    using E = ddx::rt::Expression<T>;
    const E x = ddx::rt::vec<T>("x", n);
    return sum(select(x > T(0.4), x * log(x), x * x)) +
           sum(select(x[0] > T(0.3), log(x), exp(x * T(0.5))));
  });
}

template <typename T> using Columns = std::vector<std::vector<T>>;

// `count` columns of n points, inside every model's domain.
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
                  std::ranges::to<std::vector>();
         }) |
         std::ranges::to<std::vector>();
}

template <typename T> auto pointers(Columns<T> &c) {
  return c | std::views::transform([](auto &v) { return v.data(); }) |
         std::ranges::to<std::vector>();
}
template <typename T> auto pointers(const Columns<T> &c) {
  return c | std::views::transform([](const auto &v) { return v.data(); }) |
         std::ranges::to<std::vector>();
}

// One batch of `want`, every output column it writes.
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
  return std::views::join(out) | std::ranges::to<std::vector>();
}

template <typename T>
void expect_bits(const Columns<T> &want, const Columns<T> &got,
                 const std::string &what) {
  using Bits = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
  ASSERT_EQ(want.size(), got.size()) << what;
  for (const auto [c, pair] :
       std::views::zip(want, got) | std::views::enumerate) {
    const auto &[w, g] = pair;
    ASSERT_EQ(w.size(), g.size()) << what;
    for (const auto [p, wg] : std::views::zip(w, g) | std::views::enumerate) {
      EXPECT_EQ(std::bit_cast<Bits>(std::get<0>(wg)),
                std::bit_cast<Bits>(std::get<1>(wg)))
          << what << ", column " << c << ", point " << p;
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

// Every Want at every width, against the sweep, on a ragged batch and a
// single point.
template <typename T> void expect_kernel_is_the_sweep(auto &&model) {
  const auto swept = model();
  ASSERT_FALSE(swept.poisoned()) << swept.status()->code;
  for (const auto lanes : every_width) {
    auto eq = model();
    eq.options(compiled(lanes));
    for (const Want want : every_want) {
      const std::string what =
          std::string{ddx::rt::name_of(want)} + ", lanes " + spelled(lanes);
      ASSERT_TRUE(eq.wait_for_kernel(want)) << what;
      for (const std::size_t n : {13uz, 1uz}) {
        expect_bits(batch<T>(swept, want, n), batch<T>(eq, want, n),
                    what + ", n " + std::to_string(n));
      }
    }
  }
}

// The ops a model's dense block reads.
std::set<OpCode> dense_ops(auto &&expression) {
  ddx::rt::Arena<> b;
  const auto arena = ddx::rt::detail::scoped_arena(b);
  const auto f = expression();
  const auto h = ddx::rt::derive_dense_hessian(b, f.id(b));
  const auto live = ddx::rt::detail::reachable(b, std::span{&h.block, 1});
  return ddx::rt::detail::live_ids(live) |
         std::views::transform([&b](ddx::rt::NodeId v) { return b.op_of(v); }) |
         std::ranges::to<std::set>();
}

// The table's sum off by one: a kernel that skipped the table would answer
// the reference's bits instead.
template <typename T> struct Shifted : ddx::rt::reference::Kernels<T> {
  static constexpr std::string_view name = "shifted";
  [[nodiscard]] static constexpr T sum(std::span<const T> v) noexcept {
    return ddx::rt::reference::Kernels<T>::sum(v) + T{1};
  }
};
template <typename T>
constexpr auto shifted_table = ddx::rt::tensor_kernel_table<Shifted<T>, T>();
} // namespace
} // namespace jit_tensor

using namespace jit_tensor;

TEST(JitTensor, CompilesAndAnswersTheSweepsBits) {
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
  EXPECT_TRUE(eq.wait_for_kernel());
  EXPECT_TRUE(eq.uses_kernel());
  EXPECT_TRUE(eq.kernel_level().has_value());
  for (const std::size_t p : std::views::iota(0uz, 37uz)) {
    const auto at = std::views::iota(0uz, n) |
                    std::views::transform([p](std::size_t i) {
                      return 0.2 + 0.03 * static_cast<double>(p + i);
                    }) |
                    std::ranges::to<std::vector>();
    EXPECT_EQ(*eq.evaluate(std::span<const double>{at}),
              *swept.evaluate(std::span<const double>{at}));
    EXPECT_EQ(*eq.gradient(std::span<const double>{at}),
              *swept.gradient(std::span<const double>{at}));
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

TEST(JitTensor, VectorSelectEveryWantEveryWidth) {
  expect_kernel_is_the_sweep<double>([] { return vselect<double>(6); });
  expect_kernel_is_the_sweep<float>([] { return vselect<float>(6); });
}

TEST(JitTensor, TheEquationsTableIsTheKernels) {
  const auto with_table = [] {
    auto eq = msevec<double>(6);
    eq.tensor_kernels(shifted_table<double>);
    return eq;
  };
  const auto reference = msevec<double>(6);
  auto eq = with_table();
  eq.options(compiled(ddx::jit::Lanes::derived()));
  ASSERT_TRUE(eq.wait_for_kernel(Want::Jacobian));
  const auto got = batch<double>(eq, Want::Jacobian, 13);
  expect_bits(batch<double>(with_table(), Want::Jacobian, 13), got, "shifted");
  EXPECT_NE(batch<double>(reference, Want::Jacobian, 13), got);
}

TEST(JitTensor, BatchesAndThreadsAreBitIdentical) {
  auto eq = msevec<double>(7);
  eq.options(compiled(ddx::jit::Lanes::derived()));
  ASSERT_TRUE(eq.prepare({Want::Jacobian, Want::Hvp}).has_value());
  ASSERT_TRUE(eq.wait_for_kernel(Want::Hvp));
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
    for (std::size_t t = 0; t < threads; ++t) {
      pool.emplace_back([&, t] {
        start.arrive_and_wait();
        ok[t] = call() == want;
      });
    }
  }
  EXPECT_TRUE(std::ranges::all_of(ok, std::identity{}));
}
