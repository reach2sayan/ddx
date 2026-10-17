#include "jit/kernel.hpp"
#include "jit/tests_jit_common.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

// A SumN is one step in the kernel as in the sweep: the terms into the running
// sum in list order, a product term as the same fma.  Bit identity, fused and
// not, value and gradient.

namespace {
using ddx::rt::Builder;
using ddx::rt::Graph;
using RE = ddx::rt::RTExpression<>;

RE summed(const std::vector<RE> &x) {
  RE acc{0};
  for (std::size_t t = 0; t < 40; ++t) {
    const RE &a = x[t % x.size()];
    const RE &c = x[(t * 5) % x.size()];
    const double w = 0.2 + 0.03 * static_cast<double>(t);
    switch (t % 4) {
    case 0:
      acc += w * a * c;
      break;
    case 1:
      acc -= a * c;
      break;
    case 2:
      acc += exp(a * 0.1);
      break;
    default:
      acc += a / (c + 2.0);
    }
  }
  return acc;
}

void expect_bit_identical(bool contract) {
  constexpr std::size_t nvars = 3, n = 64;
  Builder<> b;
  std::vector<RE> x;
  for (const char *name : {"x", "y", "z"}) {
    x.push_back(var(b, name));
  }
  const auto f = summed(x).id(b);
  ASSERT_EQ(b[f].op, ddx::rt::OpCode::SumN);
  const auto g = ddx::rt::build_jacobian_impl(b, f);
  const auto graph = Graph<>::freeze(
      b, {.values = {f}, .jacobian = g.partial, .hessian = {}}, {}, {},
      contract);
  const auto kernel = must_compile(graph);
  ASSERT_TRUE(static_cast<bool>(kernel));

  std::array<std::vector<double>, nvars> columns;
  std::vector<const double *> xs;
  for (std::size_t j = 0; j < nvars; ++j) {
    columns[j].resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      columns[j][i] = 0.2 + 0.5 * static_cast<double>((i + 7 * j) % 13) / 13.0;
    }
    xs.push_back(columns[j].data());
  }
  std::vector<double> value(n);
  std::array<std::vector<double>, nvars> grad;
  std::vector<double *> gs;
  for (auto &col : grad) {
    col.resize(n);
    gs.push_back(col.data());
  }
  double *const vs[]{value.data()};
  kernel(xs, vs, gs, {}, n);

  std::vector<double> tape(graph.tape_slots());
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> p(nvars);
    for (std::size_t j = 0; j < nvars; ++j) {
      p[j] = columns[j][i];
    }
    ddx::rt::evaluate_into(
        b, p, graph.schedule(), std::span<double>{tape},
        ddx::rt::detail::Scratch<double>{.slots = graph.slots()});
    EXPECT_EQ(std::bit_cast<std::uint64_t>(value[i]),
              std::bit_cast<std::uint64_t>(tape[graph.slots()[f]]))
        << "value at point " << i << (contract ? " fused" : " plain");
    for (std::size_t j = 0; j < nvars; ++j) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(grad[j][i]),
                std::bit_cast<std::uint64_t>(tape[graph.slots()[g.partial[j]]]))
          << "d/dx" << j << " at point " << i;
    }
  }
}

TEST(JitSumN, FusedSumMatchesTheSweepToTheBit) { expect_bit_identical(true); }
TEST(JitSumN, PlainSumMatchesTheSweepToTheBit) { expect_bit_identical(false); }

} // namespace
