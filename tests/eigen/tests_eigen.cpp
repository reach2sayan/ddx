#include "rt/coupling.hpp"
#include "rt/equation.hpp"
#include "rt/kernels.hpp"
#include "rt/tensor.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

// Eigen on both sides of ddx, as a consumer writes it: batches straight out of
// and back into Eigen matrices, and Eigen computing the tensor nodes.  ddx takes
// column pointers and spans, so nothing in ddx knows Eigen.
namespace {

// A batch's inputs are column pointers, each n long, which is what a
// column-major n x inputs matrix holds; its output blocks are the same.  A
// row-major matrix's columns are strided, which no column pointer can say.
template <typename D>
  requires(!D::IsRowMajor && D::InnerStrideAtCompileTime == 1)
std::vector<const double *> columns_of(const Eigen::PlainObjectBase<D> &m) {
  return std::views::iota(Eigen::Index{0}, m.cols()) |
         std::views::transform(
             [&m](Eigen::Index j) { return m.data() + j * m.outerStride(); }) |
         std::ranges::to<std::vector<const double *>>();
}
template <typename D>
  requires(!D::IsRowMajor && D::InnerStrideAtCompileTime == 1)
std::vector<double *> columns_of(Eigen::PlainObjectBase<D> &m) {
  return std::views::iota(Eigen::Index{0}, m.cols()) |
         std::views::transform(
             [&m](Eigen::Index j) { return m.data() + j * m.outerStride(); }) |
         std::ranges::to<std::vector<double *>>();
}

// One point's Jacobian, dense m x symbols, from a jacobian block: one column
// per pattern cell, every cell the pattern leaves out zero.
template <typename B, typename J>
void scatter(const ddx::rt::Sparsity &pattern, const Eigen::MatrixBase<B> &block,
             Eigen::Index point, Eigen::MatrixBase<J> &dense) {
  dense.derived().setZero(static_cast<Eigen::Index>(pattern.size()),
                          static_cast<Eigen::Index>(pattern.columns));
  const auto row = block.row(point);
  for (const auto [i, j, v] : pattern.cells(
           std::views::iota(Eigen::Index{0}, block.cols()) |
           std::views::transform([&row](Eigen::Index k) { return row(k); }))) {
    dense(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = v;
  }
}

// One point's Hessian, dense symbols x symbols, from a colour-compressed block.
template <typename B, typename H>
void scatter(const ddx::rt::Colouring &colouring,
             const Eigen::MatrixBase<B> &block, Eigen::Index point,
             Eigen::MatrixBase<H> &dense) {
  const auto n = static_cast<Eigen::Index>(colouring.colour.size());
  dense.derived().setZero(n, n);
  for (const ddx::rt::Cell &cell : colouring.entries()) {
    dense(static_cast<Eigen::Index>(cell.row),
          static_cast<Eigen::Index>(cell.column)) =
        block(point, static_cast<Eigen::Index>(cell.slot));
  }
}

// A kernel table: the reference rows, with the four Eigen helps with replaced.
// Over an Eigen built with EIGEN_USE_BLAS the products land in that BLAS.
// map and zip stay the reference rows: a transcendental per element is the same
// work under any library.
struct EigenKernels : ddx::rt::reference::Kernels {
  static constexpr std::string_view name = "eigen";
  using Vector = Eigen::Map<const Eigen::VectorXd>;
  using Matrix = Eigen::Map<const Eigen::MatrixXd>;

  static void gemv(std::size_t r, std::size_t c, In m, In v, Out out,
                   bool transposed) noexcept {
    const auto rows = static_cast<Eigen::Index>(r);
    const auto cols = static_cast<Eigen::Index>(c);
    const Matrix M(m.data(), rows, cols);
    if (transposed) {
      Eigen::Map<Eigen::VectorXd> y(out.data(), cols);
      y.noalias() = M.transpose() * Vector(v.data(), rows);
    } else {
      Eigen::Map<Eigen::VectorXd> y(out.data(), rows);
      y.noalias() = M * Vector(v.data(), cols);
    }
  }
  [[nodiscard]] static double dot(In u, In v) noexcept {
    const auto n = static_cast<Eigen::Index>(u.size());
    return Vector(u.data(), n).dot(Vector(v.data(), n));
  }
  static void scale(double s, In v, Out out) noexcept {
    const auto n = static_cast<Eigen::Index>(v.size());
    Eigen::Map<Eigen::VectorXd>(out.data(), n) = s * Vector(v.data(), n);
  }
  [[nodiscard]] static double sum(In v) noexcept {
    return Vector(v.data(), static_cast<Eigen::Index>(v.size())).sum();
  }
  static void gemm(std::size_t r, std::size_t k, std::size_t c, In a, In b,
                   Out out) noexcept {
    const auto rows = static_cast<Eigen::Index>(r);
    const auto inner = static_cast<Eigen::Index>(k);
    const auto cols = static_cast<Eigen::Index>(c);
    Eigen::Map<Eigen::MatrixXd> y(out.data(), rows, cols);
    y.noalias() = Matrix(a.data(), rows, inner) * Matrix(b.data(), inner, cols);
  }
};

constexpr ddx::rt::TensorKernelTable eigen_table =
    ddx::rt::tensor_kernel_table<EigenKernels>();

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
    X.row(i) << 0.01 * static_cast<double>(i),
        1.0 - 0.02 * static_cast<double>(i),
        0.5 + std::sin(static_cast<double>(i));
  }
  Eigen::MatrixXd F(n, 3);
  Eigen::MatrixXd G(n, static_cast<Eigen::Index>(*sys.jacobian_columns()));
  const auto xs = columns_of(X);
  ASSERT_TRUE(sys.jacobian(xs, columns_of(F),
                           columns_of(G),
                           static_cast<std::size_t>(n))
                  .has_value());

  Eigen::MatrixXd J;
  for (const Eigen::Index i : std::views::iota(Eigen::Index{0}, n)) {
    scatter(sys.jacobian_pattern()->get(), G, i, J);
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
  ASSERT_TRUE(
      eq.hessian(columns_of(X), columns_of(F),
                 columns_of(G), columns_of(H),
                 static_cast<std::size_t>(n))
          .has_value());
  Eigen::MatrixXd dense;
  for (const Eigen::Index i : std::views::iota(Eigen::Index{0}, n)) {
    scatter(eq.hessian_pattern()->get(), H, i, dense);
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
    const auto M = ddx::rt::mat_constant(n, n, m);
    return dot(x, gemv(M, x)) + sum(log(x) * gemv(M, x, true)) * 0.5;
  };
  const auto reference = ddx::rt::equation(model);
  auto eigen = ddx::rt::equation(model);
  eigen.tensor_kernels(eigen_table);
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
    EXPECT_NEAR(got_g[i], want_g[i],
                1e-13 * std::max(1.0, std::abs(want_g[i])));
    EXPECT_NEAR(got_h[i], want_h[i],
                1e-13 * std::max(1.0, std::abs(want_h[i])));
  }
}
