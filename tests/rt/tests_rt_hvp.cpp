#include "ddx.hpp"
#include "rt/derivative.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <vector>

// The gate: H(x)v built by re-rooting, against derive_hessian's own
// Hessian contracted with the same v.  Two different sweeps of the same arena,
// so a wrong rule for the Seed leaf shows up as a wrong number here.

namespace {
using ddx::rt::Builder;

// The point, then the direction, in the order input_column() states.
std::vector<double> widened(std::span<const double> x,
                            std::span<const double> v) {
  std::vector<double> at{x.begin(), x.end()};
  at.insert(at.end(), v.begin(), v.end());
  return at;
}

// Every cell of the coloured Hessian, scattered, times v.
std::vector<double> reference_product(const ddx::rt::Hessian &h,
                                      std::span<const double> values,
                                      std::span<const double> v) {
  const std::size_t n = h.partial.size();
  std::vector<double> out(n, 0.0);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    for (const std::size_t j : std::views::iota(0uz, n)) {
      out[i] += values[h.at(i, j)] * v[j];
    }
  }
  return out;
}

void expect_hvp_matches_hessian(auto build, std::size_t nvars,
                                std::span<const double> x,
                                std::span<const double> v) {
  Builder<> b;
  std::vector<ddx::rt::Expression<>> vars;
  static constexpr const char *names[] = {"a", "b", "c", "d", "e", "f"};
  for (const std::size_t i : std::views::iota(0uz, nvars)) {
    vars.push_back(var(b, names[i]));
  }
  const auto root = build(b, vars).id(b);

  const auto hess = ddx::rt::derive_hessian(b, root);
  const auto hvp = ddx::rt::derive_hvp(b, root);
  ASSERT_EQ(hvp.product.size(), nvars);

  const auto values = ddx::rt::evaluate_all(b, widened(x, v));
  const auto want = reference_product(hess, values, v);
  for (const std::size_t i : std::views::iota(0uz, nvars)) {
    EXPECT_NEAR(values[hvp.product[i]], want[i],
                1e-11 * std::max(1.0, std::abs(want[i])))
        << "component " << i;
  }
}

// Dense coupling: colours == n, the worst case for the coloured path.
auto dense = [](Builder<> &, auto &v) {
  auto acc = ddx::rt::Expression<>{0};
  for (const auto &a : v) {
    for (const auto &c : v) {
      acc = acc + exp(a * c);
    }
  }
  return acc;
};

// Separable: one colour, and every off-diagonal cell structurally zero, so the
// fold's terms mostly vanish before a node exists.
auto separable = [](Builder<> &, auto &v) {
  auto acc = ddx::rt::Expression<>{0};
  for (const auto &a : v) {
    acc = acc + sin(a) * sin(a);
  }
  return acc;
};

TEST(RtHvp, MatchesTheColouredHessianDenselyCoupled) {
  const std::array x{0.3, 0.7, 0.5};
  const std::array v{1.0, -2.0, 0.25};
  expect_hvp_matches_hessian(dense, 3, x, v);
}

TEST(RtHvp, MatchesTheColouredHessianSeparable) {
  const std::array x{0.3, 0.7, 0.5, 1.1};
  const std::array v{1.0, -2.0, 0.25, 3.0};
  expect_hvp_matches_hessian(separable, 4, x, v);
}

TEST(RtHvp, AUnitDirectionIsOneHessianColumn) {
  const std::array x{0.3, 0.7, 0.5};
  for (const std::size_t j : std::views::iota(0uz, 3uz)) {
    std::array v{0.0, 0.0, 0.0};
    v[j] = 1.0;
    expect_hvp_matches_hessian(dense, 3, x, v);
  }
}

TEST(RtHvp, IsLinearInTheDirection) {
  const std::array x{0.4, 0.9, 0.2};
  const std::array v{2.0, -1.0, 0.5};
  const std::array scaled{4.0, -2.0, 1.0};

  Builder<> b;
  const auto p = var(b, "a");
  const auto q = var(b, "b");
  const auto r = var(b, "c");
  const auto root = (exp(p * q) + r * r * p).id(b);
  const auto hvp = ddx::rt::derive_hvp(b, root);

  const auto one = ddx::rt::evaluate_all(b, widened(x, v));
  const auto two = ddx::rt::evaluate_all(b, widened(x, scaled));
  for (const auto node : hvp.product) {
    EXPECT_NEAR(two[node], 2.0 * one[node],
                1e-11 * std::max(1.0, std::abs(one[node])));
  }
}

TEST(RtHvp, CarriesTheGradient) {
  Builder<> b;
  const auto p = var(b, "a");
  const auto q = var(b, "b");
  const auto root = (p * p * q).id(b);

  const auto hvp = ddx::rt::derive_hvp(b, root);
  const auto reverse = ddx::rt::build_reverse_jacobian(b, root);
  EXPECT_EQ(hvp.partial, reverse.partial) << "interning shares one gradient";

  const std::array x{3.0, 5.0};
  const std::array v{0.0, 0.0};
  const auto values = ddx::rt::evaluate_all(b, widened(x, v));
  EXPECT_DOUBLE_EQ(values[hvp.partial[0]], 2.0 * 3.0 * 5.0); // d/da
  EXPECT_DOUBLE_EQ(values[hvp.partial[1]], 9.0);             // d/db
}
} // namespace
