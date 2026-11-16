#include "dual/dual.hpp"
#include "energy_models.hpp"
#include "graph_builder.hpp"
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
#include <initializer_list>
#include <ranges>
#include <string>
#include <vector>

// acc op= term is acc = acc op term, and an equation flattens each
// single-use chain of it into one fold node (SumN, ProdN or QuotN) that
// rounds as the chain does.

namespace {
using ddx::rt::OpCode;
using ddx::rt::detail::Arena;
using models::RE;

std::vector<RE> fold_symbols(Arena<> &b, std::size_t n) {
  return std::views::iota(0uz, n) | std::views::transform([&](std::size_t i) {
           return var(b, std::format("x{:02}", i));
         }) |
         ddx::impl::to<std::vector>();
}

// k terms over six symbols, with every third term a product.
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
  return std::views::iota(0uz, n) | std::views::transform([](std::size_t i) {
           return 0.3 + 0.17 * static_cast<double>(i);
         }) |
         ddx::impl::to<std::vector>();
}

// The roots as an equation holds them, with their chains flattened.
std::vector<ddx::rt::detail::NodeId> flat(Arena<> &b,
                                          std::initializer_list<RE> roots) {
  const auto ids = roots | std::views::transform([&b](const RE &r) {
                     return r.id(b);
                   }) |
                   ddx::impl::to<std::vector>();
  return ddx::rt::detail::flatten(b, ids);
}
ddx::rt::detail::NodeId flat(Arena<> &b, const RE &root) {
  return flat(b, {root}).front();
}
// The fold node under a flattened root. A product's chain sits under its last
// Mul, which an add still contracts into an fma.
ddx::rt::detail::NodeId fold_of(const Arena<> &b, ddx::rt::detail::NodeId f) {
  if (b[f].op != OpCode::Mul) {
    return f;
  }
  const auto in = b[f].in;
  return b[in[0]].op == OpCode::ProdN ? in[0] : in[1];
}
// The terms of a chain of `total` in its fold node: a product's last factor
// stays outside.
std::size_t listed_terms(OpCode op, std::size_t total) {
  return op == OpCode::ProdN ? total - 1 : total;
}

void expect_close(double got, double want, double rel) {
  EXPECT_NEAR(got, want, rel * std::max(1.0, std::abs(want)));
}

// One fold: its node, its op=, the binary op of its chain, and a term near one
// for products and quotients so that a long loop stays finite.
struct Fold {
  const char *name;
  OpCode op;
  OpCode binary;
  void (*assign)(RE &, const RE &);
  RE (*chain)(const RE &, const RE &);
  RE (*term)(const std::vector<RE> &, std::size_t);
  double (*step)(double, double);
};
RE near_one(const std::vector<RE> &x, std::size_t t) {
  return 1.0 + 0.05 * sin(term(x, t));
}
const Fold kFolds[]{
    {"Sum", OpCode::SumN, OpCode::Add, [](RE &a, const RE &t) { a += t; },
     [](const RE &a, const RE &t) { return a + t; }, term,
     [](double a, double t) { return a + t; }},
    {"Prod", OpCode::ProdN, OpCode::Mul, [](RE &a, const RE &t) { a *= t; },
     [](const RE &a, const RE &t) { return a * t; }, near_one,
     [](double a, double t) { return a * t; }},
    {"Quot", OpCode::QuotN, OpCode::Div, [](RE &a, const RE &t) { a /= t; },
     [](const RE &a, const RE &t) { return a / t; }, near_one,
     [](double a, double t) { return a / t; }},
    // -= sums the negated term, just as - is the add of the negation.
    {"Diff", OpCode::SumN, OpCode::Add, [](RE &a, const RE &t) { a -= t; },
     [](const RE &a, const RE &t) { return a - t; }, term,
     [](double a, double t) { return a - t; }},
};

class RtFold : public ::testing::TestWithParam<Fold> {
protected:
  [[nodiscard]] RE folded(const std::vector<RE> &x,
                          std::size_t k = kTerms) const {
    const Fold &f = GetParam();
    RE acc = x[0] + 1.0;
    for (const std::size_t t : std::views::iota(0uz, k)) {
      f.assign(acc, f.term(x, t));
    }
    return acc;
  }
  [[nodiscard]] RE chained(const std::vector<RE> &x,
                           std::size_t k = kTerms) const {
    const Fold &f = GetParam();
    RE acc = x[0] + 1.0;
    for (const std::size_t t : std::views::iota(0uz, k)) {
      acc = f.chain(acc, f.term(x, t));
    }
    return acc;
  }
};

// The loop starts from x0 + 1, which a sum's chain absorbs as two terms.
TEST_P(RtFold, ALoopIsOneNode) {
  Arena<> b;
  const auto x = fold_symbols(b, 6);
  const auto root = folded(x);
  EXPECT_EQ(root.id(b), chained(x).id(b)); // the very same chain
  const auto f = fold_of(b, flat(b, root));
  const std::size_t terms = listed_terms(
      GetParam().op, kTerms + (GetParam().binary == OpCode::Add ? 2 : 1));
  EXPECT_EQ(b[f].op, GetParam().op);
  EXPECT_EQ(ddx::rt::detail::get<ddx::rt::detail::Count>(b[f]).value, terms);
  EXPECT_EQ(ddx::rt::detail::arguments(b, f).size(), terms);
}

TEST_P(RtFold, TwoTermsStayTheBinaryNode) {
  Arena<> b;
  const auto x = fold_symbols(b, 4);
  auto acc = x[0];
  GetParam().assign(acc, x[1]);
  EXPECT_EQ(flat(b, acc), acc.id(b));
  EXPECT_EQ(b[acc.id(b)].op, GetParam().binary);
  GetParam().assign(acc, x[2]);
  GetParam().assign(acc, x[3]);
  EXPECT_EQ(b[fold_of(b, flat(b, acc))].op, GetParam().op);
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
  const auto both = flat(b, {before, acc});
  const auto kept = both[0];
  const auto grown = both[1];
  const auto p = ramp(3);
  const auto values = ddx::rt::detail::evaluate_all(b, p);
  const auto step = f.step;
  const double three = step(step(p[0], p[1]), p[2]);
  EXPECT_DOUBLE_EQ(values[kept], three);
  EXPECT_DOUBLE_EQ(values[grown], step(step(three, p[0] * p[1]), p[1] * p[2]));
}

// The value is the chain's to the bit. Derivatives agree to rounding: a fold
// hands its terms their adjoints first to last, the chain last to first.
TEST_P(RtFold, ValueGradientHvpAndHessianMatchTheChain) {
  constexpr std::size_t n = 6;
  Arena<> b;
  const auto root = chained(fold_symbols(b, n));
  const auto fc = root.id(b);
  const auto fs = flat(b, root);
  ASSERT_EQ(b[fold_of(b, fs)].op, GetParam().op);
  const auto gs = ddx::rt::detail::derive_jacobian(b, fs);
  const auto gc = ddx::rt::detail::derive_jacobian(b, fc);
  const auto hs = ddx::rt::detail::derive_hessian(b, fs);
  const auto hc = ddx::rt::detail::derive_hessian(b, fc);
  const auto ps = ddx::rt::detail::derive_hvp(b, fs);
  const auto pc = ddx::rt::detail::derive_hvp(b, fc);

  auto at = ramp(n);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    at.push_back(1.0 - 0.1 * static_cast<double>(i)); // the direction
  }
  const auto v = ddx::rt::detail::evaluate_all(b, at);
  EXPECT_EQ(v[fs], v[fc]);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    expect_close(v[gs.partial[i]], v[gc.partial[i]], 1e-13);
    expect_close(v[ps.product[i]], v[pc.product[i]], 1e-12);
    for (const std::size_t j : std::views::iota(0uz, n)) {
      expect_close(v[hs.at(i, j)], v[hc.at(i, j)], 1e-12);
    }
  }
}

// A partial that is itself a root ends the chain below it: the fold above
// takes it as its first term, and the partial is a fold of its own.
TEST_P(RtFold, ASharedPartialSplitsTheChain) {
  constexpr std::size_t n = 6;
  const Fold &f = GetParam();
  Arena<> b;
  const auto x = fold_symbols(b, n);
  RE acc = x[0] + 1.0;
  for (const std::size_t t : std::views::iota(0uz, 5uz)) {
    f.assign(acc, f.term(x, t));
  }
  const RE shared = acc;
  for (const std::size_t t : std::views::iota(5uz, 10uz)) {
    f.assign(acc, f.term(x, t));
  }
  const auto roots = flat(b, {acc, shared});
  EXPECT_EQ(b[fold_of(b, roots[0])].op, f.op);
  EXPECT_EQ(b[fold_of(b, roots[1])].op, f.op);
  const auto upper = ddx::rt::detail::arguments(b, fold_of(b, roots[0]));
  EXPECT_EQ(upper.size(), listed_terms(f.op, 6));
  EXPECT_EQ(upper.front(), roots[1]);
  const auto v = ddx::rt::detail::evaluate_all(b, ramp(n));
  EXPECT_EQ(v[roots[0]], v[acc.id(b)]);
  EXPECT_EQ(v[roots[1]], v[shared.id(b)]);
}

// Only a running operand is followed: (a + b) + (c + d) keeps c + d as one
// term, and a divisor that is a quotient is never part of the chain.
TEST(RtFoldShape, NothingIsReassociated) {
  Arena<> b;
  const auto x = fold_symbols(b, 4);
  const auto sum = flat(b, (x[0] + x[1]) + (x[2] + x[3]));
  ASSERT_EQ(b[sum].op, OpCode::SumN);
  const auto terms = ddx::rt::detail::arguments(b, sum);
  ASSERT_EQ(terms.size(), 3u);
  EXPECT_EQ(b[terms[2]].op, OpCode::Add);
  const RE nested = x[0] / (x[1] / (x[2] / x[3]));
  EXPECT_EQ(flat(b, nested), nested.id(b));
  const auto quot = flat(b, x[0] / x[1] / x[2] / x[3]);
  EXPECT_EQ(b[quot].op, OpCode::QuotN);
  const auto v = ddx::rt::detail::evaluate_all(b, ramp(4));
  const auto p = ramp(4);
  EXPECT_EQ(v[sum], (p[0] + p[1]) + (p[2] + p[3]));
  EXPECT_EQ(v[quot], p[0] / p[1] / p[2] / p[3]);
}

// A sum written right-nested is still one running value plus a term at each
// step, so it is a chain too.
TEST(RtFoldShape, ARightNestedSumIsAChain) {
  Arena<> b;
  const auto x = fold_symbols(b, 4);
  const RE root = x[0] + (x[1] + (x[2] + x[3]));
  const auto f = flat(b, root);
  EXPECT_EQ(b[f].op, OpCode::SumN);
  const auto v = ddx::rt::detail::evaluate_all(b, ramp(4));
  EXPECT_EQ(v[f], v[root.id(b)]);
}

// A sum couples none of its terms, a product couples every pair, and a quotient
// also couples each divisor with itself.
TEST_P(RtFold, CouplingPatternIsTheChains) {
  constexpr std::size_t n = 6;
  Arena<> b;
  const auto x = fold_symbols(b, n);
  // Each term reads one symbol, so the coupling is the fold's own.
  RE acc = x[0] * x[0];
  for (const RE &xi : x | std::views::drop(1)) {
    GetParam().assign(acc, xi + 1.0);
  }
  const auto fc = acc.id(b);
  const auto fs = flat(b, acc);
  ASSERT_EQ(b[fold_of(b, fs)].op, GetParam().op);
  EXPECT_EQ(ddx::rt::detail::coupling_pattern(b, fs),
            ddx::rt::detail::coupling_pattern(b, fc));
}

TEST_P(RtFold, SweepsAtADualScalar) {
  using D = ddx::impl::Dual<double>;
  Arena<> b;
  const auto root = chained(fold_symbols(b, 2), 20);
  const auto fc = root.id(b);
  const auto fs = flat(b, root);
  const std::vector p{D{0.7, 1.0}, D{1.3, 0.0}}; // d/dx0
  const auto v = ddx::rt::detail::evaluate_all(b, p);
  EXPECT_EQ(v[fs].get<0>(), v[fc].get<0>());
  expect_close(v[fs].get<1>(), v[fc].get<1>(), 1e-13);
}

// Checks every derivative equation() answers, and the dense Hessian, whose
// sweep has its own fold arm, against the coloured one.
TEST_P(RtFold, TheEquationMatchesTheChains) {
  constexpr std::size_t n = 6;
  const auto build = [&](bool in_place) {
    return ddx::rt::equation([&] {
      const auto x = std::views::iota(0uz, n) |
                     std::views::transform([](std::size_t i) {
                       return ddx::rt::var(std::format("x{:02}", i));
                     }) |
                     ddx::impl::to<std::vector>();
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
    for (const std::size_t i : std::views::iota(0uz, want.size())) {
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

// The reverse rule takes each term's adjoint from the running products, never
// by dividing the whole product by the term.
TEST(RtFoldProd, AZeroTermHasAFiniteGradient) {
  Arena<> b;
  const auto x = fold_symbols(b, 3);
  auto acc = x[0];
  acc *= x[1];
  acc *= x[2];
  acc *= x[0] + x[1];
  const auto f = flat(b, acc);
  ASSERT_EQ(b[fold_of(b, f)].op, OpCode::ProdN);
  const auto g = ddx::rt::detail::derive_jacobian(b, f);
  const std::vector p{2.0, 0.0, 3.0}; // x1 = 0
  const auto v = ddx::rt::detail::evaluate_all(b, p);
  // d/dx1 of x0 x1 x2 (x0 + x1) at x1 = 0 is x0 x2 (x0 + x1) = 12.
  EXPECT_DOUBLE_EQ(v[g.partial[0]], 0.0);
  EXPECT_DOUBLE_EQ(v[g.partial[1]], 12.0);
  EXPECT_DOUBLE_EQ(v[g.partial[2]], 0.0);
}

// Changing op mid-loop makes the fold so far the first term of the next.
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
  const auto f = flat(b, acc);
  EXPECT_EQ(b[f].op, OpCode::SumN);
  const auto p = ramp(3);
  const auto v = ddx::rt::detail::evaluate_all(b, p);
  const double sum = p[0] + p[1] + p[2];
  const double prod = sum * p[0] * p[1] * p[2];
  EXPECT_DOUBLE_EQ(v[f], prod / (p[1] + 1.0) - p[0] - p[2]);
}

INSTANTIATE_TEST_SUITE_P(Folds, RtFold, ::testing::ValuesIn(kFolds),
                         [](const auto &info) {
                           return std::string{info.param.name};
                         });

// The sum alone, to see what contraction and rebalancing do with it.
RE summed(const std::vector<RE> &x) {
  RE acc{0};
  for (const std::size_t t : std::views::iota(0uz, kTerms)) {
    acc += term(x, t);
  }
  return acc;
}

// Lists intern by their whole content, so a list that is only a prefix of
// another is a separate list.
TEST(RtSumN, ListsInternByContent) {
  Arena<> b;
  const auto x = fold_symbols(b, 4);
  const auto sum = [&](std::size_t terms) {
    RE acc = x[0];
    for (const std::size_t t : std::views::iota(1uz, terms)) {
      acc += x[t];
    }
    return flat(b, acc);
  };
  const auto four = sum(4);
  const std::size_t pool = b.lists().size();
  EXPECT_EQ(sum(4), four);
  EXPECT_EQ(b.lists().size(), pool);
  const auto three = sum(3);
  EXPECT_NE(three, four);
  EXPECT_EQ(ddx::rt::detail::arguments(b, three).size(), 3u);
  EXPECT_EQ(ddx::rt::detail::arguments(b, four).size(), 4u);
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
  const auto values = ddx::rt::detail::evaluate_all(b, p);
  EXPECT_DOUBLE_EQ(values[root], p[0] - p[1] - p[0] * p[1]);
}

// The frozen graph fuses product terms into the running sum where it contracts
// an Add; unfused, it adds each term's own row. Both compute the same sum.
TEST(RtSumN, FusedAndUnfusedSchedulesAgree) {
  Arena<> b;
  const auto x = fold_symbols(b, 6);
  const auto f = flat(b, summed(x));
  ASSERT_EQ(b[f].op, OpCode::SumN);
  const auto fused = ddx::test::freeze_values(b, std::array{f}, true);
  const auto plain = ddx::test::freeze_values(b, std::array{f}, false);
  // When fused, the product terms need no multiplies of their own.
  EXPECT_LT(fused.schedule().size(), plain.schedule().size());

  const auto p = ramp(6);
  std::vector<double> tf(fused.tape_rows()), tp(plain.tape_rows());
  ddx::rt::detail::evaluate_into(
      b, p, fused.schedule(), std::span{tf},
      ddx::rt::detail::TapeView<double>::of(fused, nullptr, nullptr));
  ddx::rt::detail::evaluate_into(
      b, p, plain.schedule(), std::span{tp},
      ddx::rt::detail::TapeView<double>::of(plain, nullptr, nullptr));
  expect_close(tf[fused.rows()[f]], tp[plain.rows()[f]], 1e-14);
  expect_close(tf[fused.rows()[f]], ddx::rt::detail::evaluate(b, f, p), 1e-14);
}

// The kernel's blocked summation splits a long SumN into partial sums.
TEST(RtSumN, RebalanceBlocksTheSum) {
  Arena<> b;
  const auto x = fold_symbols(b, 6);
  const auto f = flat(b, summed(x));
  const auto moved = ddx::rt::detail::rebalance(b, std::span{&f, 1});
  ASSERT_EQ(moved.size(), 1u);
  EXPECT_NE(moved[0], f);
  EXPECT_NE(b[moved[0]].op, OpCode::SumN);
  const auto p = ramp(6);
  const auto values = ddx::rt::detail::evaluate_all(b, p);
  expect_close(values[moved[0]], values[f], 1e-13);
}

// models::mse at n = 64 writes its n^2 pair terms with +=, which flattens
// into one SumN instead of n^2 Add nodes.
TEST(RtSumN, MseSumsItsPairTermsInOneNode) {
  constexpr std::size_t n = 64;
  Arena<> b;
  const auto f = flat(b, models::mse(fold_symbols(b, n)));
  const auto g = ddx::rt::detail::derive_jacobian(b, f);
  std::printf("[ ARENA    ] mse n=%zu: %zu nodes with the gradient\n", n,
              b.size());
  EXPECT_EQ(g.partial.size(), n);
  // Pairs whose coefficient pseudo() makes exactly zero fold out of the sum, as
  // they would out of x + 0.
  const auto widest = std::ranges::max(
      b.nodes() | std::views::filter([](const auto &node) {
        return node.op == OpCode::SumN;
      }) |
      std::views::transform([](const auto &node) {
        return ddx::rt::detail::get<ddx::rt::detail::Count>(node).value;
      }));
  std::printf("[ ARENA    ] widest sum: %u terms\n", widest);
  EXPECT_GT(widest, n * n * 9 / 10);
}

} // namespace
