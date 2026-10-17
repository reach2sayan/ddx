#include "rt/eigen.hpp"
#include "rt/eigen_kernels.hpp"
#include "rt/equation.hpp"
#include "rt/tensor.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <ranges>
#include <span>
#include <vector>

// A batch straight out of and back into Eigen matrices: the columns are the
// matrices' own storage, and scatter() lays one point's block out dense.
namespace {

TEST(RtEigen, ABatchThroughColumnMajorMatrices) {
  const auto sys = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    return std::array{x * y + sin(z), exp(x) * z, y * y};
  });
  constexpr Eigen::Index n = 50;
  Eigen::MatrixXd X(n, 3);
  for (const Eigen::Index i : std::views::iota(Eigen::Index{0}, n)) {
    X.row(i) << 0.01 * static_cast<double>(i), 1.0 - 0.02 * static_cast<double>(i),
        0.5 + std::sin(static_cast<double>(i));
  }
  Eigen::MatrixXd F(n, 3);
  Eigen::MatrixXd G(n, static_cast<Eigen::Index>(*sys.jacobian_columns()));
  const auto xs = ddx::rt::eigen::columns_of(X);
  ASSERT_TRUE(sys.jacobian(xs, ddx::rt::eigen::columns_of(F),
                           ddx::rt::eigen::columns_of(G),
                           static_cast<std::size_t>(n))
                  .has_value());

  Eigen::MatrixXd J;
  for (const Eigen::Index i : std::views::iota(Eigen::Index{0}, n)) {
    ddx::rt::eigen::scatter(sys.jacobian_pattern()->get(), G, i, J);
    const auto want = *sys.jacobian(X(i, 0), X(i, 1), X(i, 2));
    ASSERT_EQ(J.rows(), 3);
    ASSERT_EQ(J.cols(), 3);
    for (const auto [r, c] : std::views::cartesian_product(
             std::views::iota(Eigen::Index{0}, Eigen::Index{3}),
             std::views::iota(Eigen::Index{0}, Eigen::Index{3}))) {
      EXPECT_EQ(J(r, c), want[static_cast<std::size_t>(r * 3 + c)]);
    }
  }
}

TEST(RtEigen, TheHessianBlockScattersDense) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return exp(x * y) + x * x * x;
  });
  constexpr Eigen::Index n = 9;
  Eigen::MatrixXd X(n, 2);
  X.col(0).setLinSpaced(0.1, 0.9);
  X.col(1).setLinSpaced(-0.4, 0.4);
  Eigen::MatrixXd F(n, 1);
  Eigen::MatrixXd G(n, 2);
  Eigen::MatrixXd H(n, static_cast<Eigen::Index>(*eq.hessian_columns()));
  ASSERT_TRUE(eq.hessian(ddx::rt::eigen::columns_of(X),
                         ddx::rt::eigen::columns_of(F),
                         ddx::rt::eigen::columns_of(G),
                         ddx::rt::eigen::columns_of(H),
                         static_cast<std::size_t>(n))
                  .has_value());
  Eigen::MatrixXd dense;
  for (const Eigen::Index i : std::views::iota(Eigen::Index{0}, n)) {
    ddx::rt::eigen::scatter(eq.hessian_pattern()->get(), H, i, dense);
    const auto want = *eq.hessian(X(i, 0), X(i, 1));
    EXPECT_EQ(dense(0, 0), want[0]);
    EXPECT_EQ(dense(0, 1), want[1]);
    EXPECT_EQ(dense(1, 0), want[2]);
    EXPECT_EQ(dense(1, 1), want[3]);
  }
}

} // namespace

// The Eigen kernel table under a tensor graph: the same numbers as the
// reference table to rounding, and the table the equation says it ran on.
TEST(RtEigen, KernelsAgreeWithTheReferenceTable) {
  constexpr std::size_t n = 30;
  std::vector<double> m(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      m[i + j * n] = std::sin(static_cast<double>(i * 7 + j * 3));
    }
  }
  const auto model = [&] {
    const auto x = ddx::rt::vec("x", n);
    const auto M = ddx::rt::mat_const(n, n, m);
    return dot(x, gemv(M, x)) + sum(log(x) * gemv(M, x, true)) * 0.5;
  };
  const auto reference = ddx::rt::equation(model);
  auto eigen = ddx::rt::equation(model);
  eigen.kernels(ddx::rt::eigen::table);
  EXPECT_EQ(eigen.tensor_kernels().name, "eigen");

  std::vector<double> at(n);
  for (std::size_t i = 0; i < n; ++i) {
    at[i] = 0.5 + 0.01 * static_cast<double>(i);
  }
  const auto want_g = *reference.gradient(std::span<const double>{at});
  const auto got_g = *eigen.gradient(std::span<const double>{at});
  const auto want_h = *reference.hvp(at, std::span<const double>{at});
  const auto got_h = *eigen.hvp(at, std::span<const double>{at});
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(got_g[i], want_g[i], 1e-13 * std::max(1.0, std::abs(want_g[i])));
    EXPECT_NEAR(got_h[i], want_h[i], 1e-13 * std::max(1.0, std::abs(want_h[i])));
  }
}
