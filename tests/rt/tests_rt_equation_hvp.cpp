#include "ddx.hpp"
#include "md/md.hpp"
#include "rt/equation.hpp"

#include <gtest/gtest.h>

#include <vector>

// The Equation surface for H(x)v: the same product the dense hessian() gives,
// reached without building the matrix, and the same answer whether the lane is
// swept or compiled.

namespace {

// The dense Hessian this is checked against, read as a matrix rather than as
// an index expression.
std::vector<double> dense_product(const std::vector<double> &h, std::size_t n,
                                  std::span<const double> v) {
  const ddx::impl::md::mdspan m{h.data(),
                                ddx::impl::md::dextents<std::size_t, 2>{n, n}};
  std::vector<double> out(n, 0.0);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    for (const std::size_t j : std::views::iota(0uz, n)) {
      out[i] += m[i, j] * v[j];
    }
  }
  return out;
}

TEST(RtEquationHvp, MatchesTheDenseHessianTimesTheDirection) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    return exp(x * y) + z * z * x + sin(y * z);
  });
  ASSERT_FALSE(eq.poisoned());
  ASSERT_EQ(eq.arity(), 3u);

  const std::array point{0.3, 0.7, 0.5};
  const std::array<double, 3> v{1.0, -2.0, 0.25};

  const auto dense = *eq.hessian(point);
  const auto want = dense_product(dense, 3, v);
  const auto got = *eq.hvp(std::span<const double>{v}, point);

  ASSERT_EQ(got.size(), 3u);
  for (const std::size_t i : std::views::iota(0uz, 3uz)) {
    EXPECT_NEAR(got[i], want[i], 1e-10 * std::max(1.0, std::abs(want[i])))
        << "component " << i;
  }
}

TEST(RtEquationHvp, ANamedSymbolIsThatHessianColumn) {
  const auto eq = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    return a * a * b + exp(b);
  });
  const std::array point{2.0, 0.5};
  const auto dense = *eq.hessian(point);

  // Symbols are in declaration order, so "a" is column 0 and "b" is column 1.
  for (const auto [j, name] : std::array{"a", "b"} | std::views::enumerate) {
    const auto column = *eq.hvp(name, point);
    ASSERT_EQ(column.size(), 2u);
    for (const std::size_t i : std::views::iota(0uz, 2uz)) {
      EXPECT_NEAR(column[i], dense[i * 2 + static_cast<std::size_t>(j)], 1e-10)
          << name << " at row " << i;
    }
  }
}

// A seed the product never reads still has its column: y's is dead in
// H(x*x + y), and the call takes the full direction all the same.
TEST(RtEquationHvp, ASeedTheProductDropsKeepsItsColumn) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return x * x + y;
  });
  const std::vector<double> v{1.0, 1.0};
  const auto got = eq.hvp(std::span<const double>{v}, 3.0, 4.0);
  ASSERT_TRUE(got.has_value()) << got.error().code;
  EXPECT_EQ((*got)[0], 2.0);
  EXPECT_EQ((*got)[1], 0.0);
  const auto linear = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return 2.0 * x + y;
  });
  const auto dt = linear.jacobian_jvp("x", 3.0, 4.0);
  ASSERT_TRUE(dt.has_value()) << dt.error().code;
  EXPECT_EQ((*dt)[0], 0.0);
  EXPECT_EQ((*dt)[1], 0.0);
}

TEST(RtEquationHvp, RefusesADirectionOfTheWrongLength) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return x * y;
  });
  const std::vector<double> tooLong{1.0, 2.0, 3.0};
  const auto r = eq.hvp(std::span<const double>{tooLong}, 1.0, 2.0);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().code, ddx::errc::wrong_direction);
}

TEST(RtEquationHvp, RefusesAnUnknownSymbol) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return x * x;
  });
  const auto r = eq.hvp("nope", 1.0);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().code, ddx::errc::unknown_symbol);
}

TEST(RtEquationHvp, TakesEveryPointSpelling) {
  using namespace ddx::literals;
  const auto eq = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    return a * a * b;
  });
  const std::array<double, 2> v{1.0, 1.0};
  const auto d = std::span<const double>{v};

  const auto positional = *eq.hvp(d, 2.0, 3.0);
  EXPECT_EQ(*eq.hvp(d, std::array{2.0, 3.0}), positional);
  EXPECT_EQ(*eq.hvp(d, "a"_s = 2.0, "b"_s = 3.0), positional);
}

TEST(RtEquationHvp, BatchAgreesWithThePointSpelling) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return exp(x) * sin(y) + x * y * y;
  });

  static constexpr std::size_t n = 40; // past the lane width, with a tail
  std::array<std::vector<double>, 2> xs{std::vector<double>(n),
                                        std::vector<double>(n)};
  std::array<std::vector<double>, 2> vs{std::vector<double>(n),
                                        std::vector<double>(n)};
  for (const std::size_t i : std::views::iota(0uz, n)) {
    xs[0][i] = 0.2 + 0.01 * static_cast<double>(i);
    xs[1][i] = 0.5 - 0.005 * static_cast<double>(i);
    vs[0][i] = 1.0 + 0.1 * static_cast<double>(i % 3);
    vs[1][i] = -0.5 * static_cast<double>(i % 4);
  }
  const double *const xp[]{xs[0].data(), xs[1].data()};
  const double *const vp[]{vs[0].data(), vs[1].data()};

  std::vector<double> f(n);
  std::array<std::vector<double>, 2> g{std::vector<double>(n),
                                       std::vector<double>(n)};
  std::array<std::vector<double>, 2> hv{std::vector<double>(n),
                                        std::vector<double>(n)};
  double *const fp[]{f.data()};
  double *const gp[]{g[0].data(), g[1].data()};
  double *const hp[]{hv[0].data(), hv[1].data()};

  ASSERT_TRUE(eq.hvp(xp, vp, fp, gp, hp, n).has_value());

  for (const std::size_t i : std::views::iota(0uz, n)) {
    const std::array<double, 2> v{vs[0][i], vs[1][i]};
    const auto want =
        *eq.hvp(std::span<const double>{v}, std::array{xs[0][i], xs[1][i]});
    EXPECT_NEAR(hv[0][i], want[0], 1e-10 * std::max(1.0, std::abs(want[0])));
    EXPECT_NEAR(hv[1][i], want[1], 1e-10 * std::max(1.0, std::abs(want[1])));
    // The gradient rides along in the same sweep.
    const auto grad = *eq.jacobian(std::array{xs[0][i], xs[1][i]});
    EXPECT_NEAR(g[0][i], grad[0], 1e-10 * std::max(1.0, std::abs(grad[0])));
  }
}

// The lane is exempted from the rebalanced compile graph, so the kernel and the
// sweep are the same arithmetic in the same order and bit equality is honest.
#ifdef DDX_HAS_JIT
TEST(RtEquationHvp, TheKernelAgreesWithTheSweepToTheBit) {
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto z = var(b, "z");
  auto eq = ddx::rt::equation(exp(x * y) + z * z * x + log(y + z));

  static constexpr std::size_t n = 33;
  std::array<std::vector<double>, 3> xs;
  std::array<std::vector<double>, 3> vs;
  for (const auto [j, pair] : std::views::zip(xs, vs) | std::views::enumerate) {
    auto &[col, dir] = pair;
    col.resize(n);
    dir.resize(n);
    for (const std::size_t i : std::views::iota(0uz, n)) {
      col[i] = 0.4 + 0.01 * static_cast<double>(i + 2 * std::size_t(j));
      dir[i] = 1.0 - 0.03 * static_cast<double>((i + std::size_t(j)) % 5);
    }
  }
  const double *const xp[]{xs[0].data(), xs[1].data(), xs[2].data()};
  const double *const vp[]{vs[0].data(), vs[1].data(), vs[2].data()};

  const auto run = [&](ddx::rt::Backend which) {
    std::vector<double> f(n);
    std::array<std::vector<double>, 3> g{
        std::vector<double>(n), std::vector<double>(n), std::vector<double>(n)};
    std::array<std::vector<double>, 3> hv{
        std::vector<double>(n), std::vector<double>(n), std::vector<double>(n)};
    double *const fp[]{f.data()};
    double *const gp[]{g[0].data(), g[1].data(), g[2].data()};
    double *const hp[]{hv[0].data(), hv[1].data(), hv[2].data()};
    eq.options({.backend = which});
    const bool kernel = eq.uses_kernel();
    EXPECT_TRUE(eq.hvp(xp, vp, fp, gp, hp, n).has_value());
    return std::tuple{hv, kernel};
  };

  const auto [compiled, used_kernel] = run(ddx::rt::Backend::Compile);
  const auto [swept, used_none] = run(ddx::rt::Backend::Interpret);
  EXPECT_FALSE(used_none);

  for (const auto [j, pair] :
       std::views::zip(compiled, swept) | std::views::enumerate) {
    const auto &[c, s] = pair;
    for (const std::size_t i : std::views::iota(0uz, n)) {
      EXPECT_EQ(c[i], s[i]) << "component " << j << " at point " << i;
    }
  }
}
#endif

TEST(RtEquationHvp, ColumnCountIsTheSymbolCount) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return x * y;
  });
  EXPECT_EQ(eq.hvp_columns(), std::optional{2uz});
}
} // namespace
