#include "dual/dual.hpp"
#include "energy_models.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/equation.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "rt/rebalance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <format>
#include <ranges>
#include <string>
#include <vector>

// A loop of `acc op= term` is one fold node over its terms -- SumN, ProdN or
// QuotN -- where the same loop spelled `acc = acc op term` is a chain of the
// binary op.  The two agree on every number to rounding, and the fold is the
// smaller arena.

namespace {
using ddx::rt::Arena;
using ddx::rt::OpCode;
using models::RE;

std::vector<RE> fold_symbols(Arena<> &b, std::size_t n) {
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

std::vector<double> ramp(std::size_t n) {
  std::vector<double> p(n);
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = 0.3 + 0.17 * static_cast<double>(i);
  }
  return p;
}

void expect_close(double got, double want, double rel) {
  EXPECT_NEAR(got, want, rel * std::max(1.0, std::abs(want)));
}

// One fold: its node, its `op=`, the binary op its chain is built of, and the
// term a long loop of it stays finite on -- near one for a product or a
// quotient.
struct Fold {
  const char *name;
  OpCode op;
  OpCode binary;
  void (*assign)(RE &, const RE &);
  RE (*chain)(const RE &, const RE &);
  RE (*term)(const std::vector<RE> &, std::size_t);
};
RE near_one(const std::vector<RE> &x, std::size_t t) {
  return 1.0 + 0.05 * sin(term(x, t));
}
const Fold kFolds[]{
    {"Sum", OpCode::SumN, OpCode::Add, [](RE &a, const RE &t) { a += t; },
     [](const RE &a, const RE &t) { return a + t; }, term},
    {"Prod", OpCode::ProdN, OpCode::Mul, [](RE &a, const RE &t) { a *= t; },
     [](const RE &a, const RE &t) { return a * t; }, near_one},
    {"Quot", OpCode::QuotN, OpCode::Div, [](RE &a, const RE &t) { a /= t; },
     [](const RE &a, const RE &t) { return a / t; }, near_one},
};

class RtFold : public ::testing::TestWithParam<Fold> {
protected:
  [[nodiscard]] RE folded(const std::vector<RE> &x,
                          std::size_t k = kTerms) const {
    const Fold &f = GetParam();
    RE acc = x[0] + 1.0;
    for (std::size_t t = 0; t < k; ++t) {
      f.assign(acc, f.term(x, t));
    }
    return acc;
  }
  [[nodiscard]] RE chained(const std::vector<RE> &x,
                           std::size_t k = kTerms) const {
    const Fold &f = GetParam();
    RE acc = x[0] + 1.0;
    for (std::size_t t = 0; t < k; ++t) {
      acc = f.chain(acc, f.term(x, t));
    }
    return acc;
  }
};

TEST_P(RtFold, ALoopIsOneNode) {
  Arena<> fold;
  const auto root = folded(fold_symbols(fold, 6));
  const auto &node = fold[root.id(fold)];
  EXPECT_EQ(node.op, GetParam().op);
  EXPECT_EQ(node.aux, kTerms + 1);
  EXPECT_EQ(ddx::rt::arguments(fold, root.id(fold)).size(), kTerms + 1);

  Arena<> chain;
  (void)chained(fold_symbols(chain, 6)).id(chain);
  // The chain's k binary nodes against the one fold, and nothing else.
  EXPECT_EQ(chain.size(), fold.size() + kTerms - 1);
}

TEST_P(RtFold, TwoTermsAreTheChainsBinaryNode) {
  Arena<> b;
  const auto x = fold_symbols(b, 2);
  auto acc = x[0];
  GetParam().assign(acc, x[1]);
  EXPECT_EQ(acc.id(b), GetParam().chain(x[0], x[1]).id(b));
  EXPECT_EQ(b[acc.id(b)].op, GetParam().binary);
  GetParam().assign(acc, x[0] * x[1]);
  EXPECT_EQ(b[acc.id(b)].op, GetParam().op);
}

TEST_P(RtFold, TheIdentityOnTheRightIsDropped) {
  Arena<> b;
  const auto x = fold_symbols(b, 2);
  const double identity = GetParam().op == OpCode::SumN ? 0.0 : 1.0;
  auto acc = x[0];
  GetParam().assign(acc, RE{identity});
  GetParam().assign(acc, RE{b, b.constant(identity)});
  EXPECT_EQ(acc.id(b), x[0].id(b));
}

TEST_P(RtFold, ACopyOfAnAccumulatorKeepsItsValue) {
  Arena<> b;
  const auto x = fold_symbols(b, 3);
  const Fold &f = GetParam();
  auto acc = x[0];
  f.assign(acc, x[1]);
  f.assign(acc, x[2]);
  const RE before = acc;
  f.assign(acc, x[0] * x[1]);
  f.assign(acc, x[1] * x[2]);
  const auto kept = before.id(b);
  const auto grown = acc.id(b);
  // Reading a fold again is the node it formed, not another.
  const std::size_t n = b.size();
  EXPECT_EQ(acc.id(b), grown);
  EXPECT_EQ(b.size(), n);
  const auto p = ramp(3);
  const auto values = ddx::rt::evaluate_all(b, p);
  const auto step = [&](double l, double r) {
    return f.op == OpCode::SumN ? l + r : f.op == OpCode::ProdN ? l * r : l / r;
  };
  const double three = step(step(p[0], p[1]), p[2]);
  EXPECT_DOUBLE_EQ(values[kept], three);
  EXPECT_DOUBLE_EQ(values[grown],
                   step(step(three, p[0] * p[1]), p[1] * p[2]));
}

TEST_P(RtFold, ValueGradientHvpAndHessianMatchTheChain) {
  constexpr std::size_t n = 6;
  Arena<> fold, chain;
  const auto fs = folded(fold_symbols(fold, n)).id(fold);
  const auto fc = chained(fold_symbols(chain, n)).id(chain);
  const auto gs = ddx::rt::derive_jacobian(fold, fs);
  const auto gc = ddx::rt::derive_jacobian(chain, fc);
  const auto hs = ddx::rt::derive_hessian(fold, fs);
  const auto hc = ddx::rt::derive_hessian(chain, fc);
  const auto ps = ddx::rt::derive_hvp(fold, fs);
  const auto pc = ddx::rt::derive_hvp(chain, fc);

  auto at = ramp(n);
  for (std::size_t i = 0; i < n; ++i) {
    at.push_back(1.0 - 0.1 * static_cast<double>(i)); // the direction
  }
  const auto vs = ddx::rt::evaluate_all(fold, at);
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

// A fold's sparsity is its chain's: a sum couples nothing between its terms,
// a product every pair, a quotient its divisors with themselves as well.
TEST_P(RtFold, CouplingPatternIsTheChains) {
  constexpr std::size_t n = 6;
  Arena<> fold, chain;
  const auto x = fold_symbols(fold, n);
  const auto y = fold_symbols(chain, n);
  // Terms each reading one symbol, so the coupling is the fold's own.
  const auto build = [&](const std::vector<RE> &s, auto &&combine) {
    RE acc = s[0] * s[0];
    for (std::size_t i = 1; i < n; ++i) {
      combine(acc, s[i] + 1.0);
    }
    return acc;
  };
  const auto fs = build(x, GetParam().assign).id(fold);
  const auto fc =
      build(y, [&](RE &a, const RE &t) { a = GetParam().chain(a, t); })
          .id(chain);
  ASSERT_EQ(fold[fs].op, GetParam().op);
  EXPECT_EQ(ddx::rt::coupling_pattern(fold, fs),
            ddx::rt::coupling_pattern(chain, fc));
}

// A fold computes at any scalar the sweep runs at, so a dual point carries
// its derivative through it.
TEST_P(RtFold, SweepsAtADualScalar) {
  using D = ddx::impl::Dual<double>;
  Arena<> fold, chain;
  const auto fs = folded(fold_symbols(fold, 2), 20).id(fold);
  const auto fc = chained(fold_symbols(chain, 2), 20).id(chain);
  const std::vector<D> p{D{0.7, 1.0}, D{1.3, 0.0}}; // d/dx0
  const auto vs = ddx::rt::evaluate_all(fold, p);
  const auto vc = ddx::rt::evaluate_all(chain, p);
  expect_close(vs[fs].get<0>(), vc[fc].get<0>(), 1e-14);
  expect_close(vs[fs].get<1>(), vc[fc].get<1>(), 1e-13);
}

// The whole pipeline: every derivative equation() answers, and the dense
// Hessian -- whose sweep has its own fold arm -- against the coloured one.
TEST_P(RtFold, TheEquationMatchesTheChains) {
  constexpr std::size_t n = 6;
  const auto build = [&](bool in_place) {
    return ddx::rt::equation([&] {
      std::vector<RE> x;
      for (std::size_t i = 0; i < n; ++i) {
        x.push_back(ddx::rt::var(std::format("x{:02}", i)));
      }
      return in_place ? folded(x, 30) : chained(x, 30);
    });
  };
  const auto fold = build(true);
  const auto chain = build(false);
  ASSERT_FALSE(fold.poisoned());
  ASSERT_FALSE(chain.poisoned());
  const auto at = ramp(n);
  const auto close = [](const auto &want, const auto &got, double tol,
                        const char *what) {
    ASSERT_EQ(want.size(), got.size()) << what;
    for (std::size_t i = 0; i < want.size(); ++i) {
      EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::abs(want[i])))
          << what << " at " << i;
    }
  };
  expect_close(*fold.evaluate(at), *chain.evaluate(at), 1e-14);
  close(*chain.gradient(at), *fold.gradient(at), 1e-13, "gradient");
  close(*chain.hessian(at), *fold.hessian(at), 1e-12, "hessian");
  const auto dense = fold.dense_hessian(at);
  ASSERT_TRUE(dense.has_value());
  close(*fold.hessian(at), *dense, 1e-12, "dense vs coloured");
}

// A product with a zero term: the reverse rule takes each term's adjoint off
// the running products, never by dividing the whole by the term.
TEST(RtFoldProd, AZeroTermHasAFiniteGradient) {
  Arena<> b;
  const auto x = fold_symbols(b, 3);
  auto acc = x[0];
  acc *= x[1];
  acc *= x[2];
  acc *= x[0] + x[1];
  const auto f = acc.id(b);
  ASSERT_EQ(b[f].op, OpCode::ProdN);
  const auto g = ddx::rt::derive_jacobian(b, f);
  const std::vector<double> p{2.0, 0.0, 3.0}; // x1 = 0
  const auto v = ddx::rt::evaluate_all(b, p);
  // d/dx1 of x0 x1 x2 (x0 + x1) at x1 = 0 is x0 x2 (x0 + x1) = 12.
  EXPECT_DOUBLE_EQ(v[g.partial[0]], 0.0);
  EXPECT_DOUBLE_EQ(v[g.partial[1]], 12.0);
  EXPECT_DOUBLE_EQ(v[g.partial[2]], 0.0);
}

// Changing op mid-loop closes the fold so far as the first term of the next.
TEST(RtFoldMixed, AnotherOpStartsANewFold) {
  Arena<> b;
  const auto x = fold_symbols(b, 3);
  auto acc = x[0];
  acc += x[1];
  acc += x[2];
  acc *= x[0];
  acc *= x[1];
  acc *= x[2];
  acc /= x[1] + 1.0;
  acc -= x[0];
  acc -= x[2];
  const auto f = acc.id(b);
  EXPECT_EQ(b[f].op, OpCode::SumN);
  const auto p = ramp(3);
  const auto v = ddx::rt::evaluate_all(b, p);
  const double sum = p[0] + p[1] + p[2];
  const double prod = sum * p[0] * p[1] * p[2];
  EXPECT_DOUBLE_EQ(v[f], prod / (p[1] + 1.0) - p[0] - p[2]);
}

INSTANTIATE_TEST_SUITE_P(Folds, RtFold, ::testing::ValuesIn(kFolds),
                         [](const auto &info) {
                           return std::string{info.param.name};
                         });

// The sum alone: what contraction and rebalancing do with it.
RE summed(const std::vector<RE> &x) {
  RE acc{0};
  for (std::size_t t = 0; t < kTerms; ++t) {
    acc += term(x, t);
  }
  return acc;
}

TEST(RtSumN, ZeroTermsAreDroppedAndMinusNegates) {
  Arena<> b;
  const auto x = fold_symbols(b, 2);
  auto acc = x[0];
  acc += 0.0;
  acc += RE{b, b.constant(0.0)};
  EXPECT_EQ(acc.id(b), x[0].id(b));
  acc -= x[1];
  acc -= x[0] * x[1];
  const auto root = acc.id(b); // formed here, so the sweep below has it
  const auto p = ramp(2);
  const auto values = ddx::rt::evaluate_all(b, p);
  EXPECT_DOUBLE_EQ(values[root], p[0] - p[1] - p[0] * p[1]);
}

// The frozen graph fuses a product term into the running sum where it
// contracts an Add; unfused it adds the term's own row.  Both are the sum.
TEST(RtSumN, FusedAndUnfusedSchedulesAgree) {
  Arena<> b;
  const auto x = fold_symbols(b, 6);
  const auto f = summed(x).id(b);
  const auto fused = ddx::rt::Graph<>::freeze(b, std::array{f}, true);
  const auto plain = ddx::rt::Graph<>::freeze(b, std::array{f}, false);
  // Fused, the product terms' multiplies are nobody's to compute.
  EXPECT_LT(fused.schedule().size(), plain.schedule().size());

  const auto p = ramp(6);
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

// The kernel's blocked summation reads a SumN as the spine it already is.
TEST(RtSumN, RebalanceBlocksTheSum) {
  Arena<> b;
  const auto x = fold_symbols(b, 6);
  const auto f = summed(x).id(b);
  const auto moved = ddx::rt::detail::rebalance(b, std::span{&f, 1});
  ASSERT_EQ(moved.size(), 1u);
  EXPECT_NE(moved[0], f);
  EXPECT_NE(b[moved[0]].op, OpCode::SumN);
  const auto p = ramp(6);
  const auto values = ddx::rt::evaluate_all(b, p);
  expect_close(values[moved[0]], values[f], 1e-13);
}

// models::mse at n = 64 writes its n^2 pair terms with `+=`: one SumN of n^2
// terms where the chain was n^2 Add nodes.  The arena size is printed for the
// record; the terms and their derivatives are what remains of it.
TEST(RtSumN, MseSumsItsPairTermsInOneNode) {
  constexpr std::size_t n = 64;
  Arena<> b;
  const auto f = models::mse(fold_symbols(b, n)).id(b);
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
