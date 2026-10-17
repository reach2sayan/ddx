#include "dual/dual.hpp"
#include "energy_models.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "rt/rebalance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <format>
#include <ranges>
#include <vector>

// A loop of `acc += term` is one SumN node over its terms; the same loop
// spelled `acc = acc + term` is a chain of Add.  The two agree on every number
// to rounding, and the sum is the smaller arena.

namespace {
using ddx::rt::Builder;
using ddx::rt::OpCode;
using models::RE;

std::vector<RE> symbols(Builder<> &b, std::size_t n) {
  std::vector<RE> x;
  for (std::size_t i = 0; i < n; ++i) {
    x.push_back(var(b, std::format("x{:02}", i)));
  }
  return x;
}

// k terms over six symbols, a product for every third term, the rest not.
constexpr std::size_t kTerms = 200;
RE term(const std::vector<RE> &x, std::size_t t) {
  const RE &a = x[t % x.size()];
  const RE &c = x[(t * 7) % x.size()];
  const double w = 0.1 + 0.01 * static_cast<double>(t);
  switch (t % 3) {
  case 0:
    return w * a * c; // a product: fused into the running sum
  case 1:
    return sin(a * w) * exp(c * 0.1);
  default:
    return log(a + 1.5) / (c + 2.0);
  }
}
RE summed(const std::vector<RE> &x) {
  RE acc{0};
  for (std::size_t t = 0; t < kTerms; ++t) {
    acc += term(x, t);
  }
  return acc;
}
RE chained(const std::vector<RE> &x) {
  RE acc{0};
  for (std::size_t t = 0; t < kTerms; ++t) {
    acc = acc + term(x, t);
  }
  return acc;
}

std::vector<double> point(std::size_t n) {
  std::vector<double> p(n);
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = 0.3 + 0.17 * static_cast<double>(i);
  }
  return p;
}

void expect_close(double got, double want, double rel) {
  EXPECT_NEAR(got, want, rel * std::max(1.0, std::abs(want)));
}

TEST(RtSumN, ALoopOfPlusEqualsIsOneNode) {
  Builder<> sum;
  const auto root = summed(symbols(sum, 6));
  const auto &node = sum[root.id(sum)];
  EXPECT_EQ(node.op, OpCode::SumN);
  EXPECT_EQ(node.aux, kTerms);
  EXPECT_EQ(ddx::rt::arguments(sum, root.id(sum)).size(), kTerms);

  Builder<> chain;
  (void)chained(symbols(chain, 6)).id(chain);
  // The chain's k - 1 Add nodes against the one SumN, and nothing else.
  EXPECT_EQ(chain.size(), sum.size() + kTerms - 2);
}

TEST(RtSumN, TwoTermsAreTheChainsAdd) {
  Builder<> b;
  const auto x = symbols(b, 2);
  auto acc = x[0];
  acc += x[1];
  EXPECT_EQ(acc.id(b), (x[0] + x[1]).id(b));
  acc += x[0] * x[1];
  EXPECT_EQ(b[acc.id(b)].op, OpCode::SumN);
}

TEST(RtSumN, ZeroTermsAreDroppedAndMinusNegates) {
  Builder<> b;
  const auto x = symbols(b, 2);
  auto acc = x[0];
  acc += 0.0;
  acc += RE{b, b.constant(0.0)};
  EXPECT_EQ(acc.id(b), x[0].id(b));
  acc -= x[1];
  acc -= x[0] * x[1];
  const auto root = acc.id(b); // formed here, so the sweep below has it
  const auto p = point(2);
  const auto values = ddx::rt::evaluate_all(b, p);
  EXPECT_DOUBLE_EQ(values[root], p[0] - p[1] - p[0] * p[1]);
}

TEST(RtSumN, ACopyOfAnAccumulatorKeepsItsValue) {
  Builder<> b;
  const auto x = symbols(b, 3);
  auto acc = x[0];
  acc += x[1];
  acc += x[2];
  const RE before = acc;
  acc += x[0] * x[1];
  acc += x[1] * x[2];
  const auto kept = before.id(b);
  const auto grown = acc.id(b);
  // Reading a sum again is the node it formed, not another.
  const std::size_t n = b.size();
  EXPECT_EQ(acc.id(b), grown);
  EXPECT_EQ(b.size(), n);
  const auto p = point(3);
  const auto values = ddx::rt::evaluate_all(b, p);
  EXPECT_DOUBLE_EQ(values[kept], p[0] + p[1] + p[2]);
  EXPECT_DOUBLE_EQ(values[grown],
                   p[0] + p[1] + p[2] + p[0] * p[1] + p[1] * p[2]);
}

TEST(RtSumN, ValueGradientHvpAndHessianMatchTheChain) {
  constexpr std::size_t n = 6;
  Builder<> sum, chain;
  const auto fs = summed(symbols(sum, n)).id(sum);
  const auto fc = chained(symbols(chain, n)).id(chain);
  const auto gs = ddx::rt::derive_jacobian(sum, fs);
  const auto gc = ddx::rt::derive_jacobian(chain, fc);
  const auto hs = ddx::rt::derive_hessian(sum, fs);
  const auto hc = ddx::rt::derive_hessian(chain, fc);
  const auto ps = ddx::rt::derive_hvp(sum, fs);
  const auto pc = ddx::rt::derive_hvp(chain, fc);

  auto at = point(n);
  for (std::size_t i = 0; i < n; ++i) {
    at.push_back(1.0 - 0.1 * static_cast<double>(i)); // the direction
  }
  const auto vs = ddx::rt::evaluate_all(sum, at);
  const auto vc = ddx::rt::evaluate_all(chain, at);
  expect_close(vs[fs], vc[fc], 1e-14);
  for (std::size_t i = 0; i < n; ++i) {
    expect_close(vs[gs.partial[i]], vc[gc.partial[i]], 1e-13);
    expect_close(vs[ps.product[i]], vc[pc.product[i]], 1e-12);
    for (std::size_t j = 0; j < n; ++j) {
      expect_close(vs[hs.at(i, j)], vc[hc.at(i, j)], 1e-12);
    }
  }
}

// The frozen graph fuses a product term into the running sum where it
// contracts an Add; unfused it adds the term's own row.  Both are the sum.
TEST(RtSumN, FusedAndUnfusedSchedulesAgree) {
  Builder<> b;
  const auto x = symbols(b, 6);
  const auto f = summed(x).id(b);
  const auto fused = ddx::rt::Graph<>::freeze(b, std::array{f}, true);
  const auto plain = ddx::rt::Graph<>::freeze(b, std::array{f}, false);
  // Fused, the product terms' multiplies are nobody's to compute.
  EXPECT_LT(fused.schedule().size(), plain.schedule().size());

  const auto p = point(6);
  std::vector<double> tf(fused.tape_rows()), tp(plain.tape_rows());
  ddx::rt::evaluate_into(
      b, p, fused.schedule(), std::span<double>{tf},
      ddx::rt::detail::TapeView<double>{.scalar_rows = fused.rows()});
  ddx::rt::evaluate_into(
      b, p, plain.schedule(), std::span<double>{tp},
      ddx::rt::detail::TapeView<double>{.scalar_rows = plain.rows()});
  expect_close(tf[fused.rows()[f]], tp[plain.rows()[f]], 1e-14);
  expect_close(tf[fused.rows()[f]], ddx::rt::evaluate(b, f, p), 1e-14);
}

// A sum adds at any scalar the sweep runs at, so a dual point carries its
// derivative through the sum.
TEST(RtSumN, SweepsAtADualScalar) {
  using D = ddx::impl::Dual<double>;
  Builder<> b;
  const auto x = symbols(b, 2);
  auto acc = x[0] * x[1];
  acc += x[0] * x[0];
  acc += x[1];
  const auto root = acc.id(b);
  const std::vector<D> p{D{2.0, 1.0}, D{3.0, 0.0}}; // d/dx0
  const auto values = ddx::rt::evaluate_all(b, p);
  EXPECT_DOUBLE_EQ(values[root].get<0>(), 2.0 * 3.0 + 4.0 + 3.0);
  EXPECT_DOUBLE_EQ(values[root].get<1>(), 3.0 + 4.0);
}

// The kernel's blocked summation reads a SumN as the spine it already is.
TEST(RtSumN, RebalanceBlocksTheSum) {
  Builder<> b;
  const auto x = symbols(b, 6);
  const auto f = summed(x).id(b);
  const auto moved = ddx::rt::detail::rebalance(b, std::span{&f, 1});
  ASSERT_EQ(moved.size(), 1u);
  EXPECT_NE(moved[0], f);
  EXPECT_NE(b[moved[0]].op, OpCode::SumN);
  const auto p = point(6);
  const auto values = ddx::rt::evaluate_all(b, p);
  expect_close(values[moved[0]], values[f], 1e-13);
}

// models::mse at n = 64 writes its n^2 pair terms with `+=`: one SumN of n^2
// terms where the chain was n^2 Add nodes.  The arena size is printed for the
// record; the terms and their derivatives are what remains of it.
TEST(RtSumN, MseSumsItsPairTermsInOneNode) {
  constexpr std::size_t n = 64;
  Builder<> b;
  const auto f = models::mse(symbols(b, n)).id(b);
  const auto g = ddx::rt::derive_jacobian(b, f);
  std::printf("[ ARENA    ] mse n=%zu: %zu nodes with the gradient\n", n,
              b.size());
  EXPECT_EQ(g.partial.size(), n);
  // All but the pairs whose coefficient pseudo() makes exactly zero, which
  // fold out of the sum as they fold out of `x + 0`.
  const auto widest = std::ranges::max(
      b.nodes() | std::views::filter([](const auto &node) {
        return node.op == OpCode::SumN;
      }) |
      std::views::transform([](const auto &node) { return node.aux; }));
  std::printf("[ ARENA    ] widest sum: %u terms\n", widest);
  EXPECT_GT(widest, n * n * 9 / 10);
}

} // namespace
