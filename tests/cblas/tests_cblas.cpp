#include "rt/equation.hpp"
#include "rt/kernels.hpp"
#include "rt/tensor.hpp"

#include <gtest/gtest.h>

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

// Any CBLAS under the tensor nodes -- OpenBLAS, MKL, Accelerate, the reference
// netlib -- as a consumer writes it.  A table's matrices are column-major with
// contiguous columns, so the leading dimension is the row count.
namespace {

// The reference rows, with the three BLAS has a routine for replaced.
struct CblasKernels : ddx::rt::reference::Kernels {
  static constexpr std::string_view name = "cblas";

  static void gemv(std::size_t r, std::size_t c, In m, In v, Out out,
                   bool transposed) noexcept {
    const auto rows = static_cast<int>(r);
    cblas_dgemv(CblasColMajor, transposed ? CblasTrans : CblasNoTrans, rows,
                static_cast<int>(c), 1.0, m.data(), rows, v.data(), 1, 0.0,
                out.data(), 1);
  }
  [[nodiscard]] static double dot(In u, In v) noexcept {
    return cblas_ddot(static_cast<int>(u.size()), u.data(), 1, v.data(), 1);
  }
  static void gemm(std::size_t r, std::size_t k, std::size_t c, In a, In b,
                   Out out) noexcept {
    const auto rows = static_cast<int>(r);
    const auto inner = static_cast<int>(k);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows,
                static_cast<int>(c), inner, 1.0, a.data(), rows, b.data(),
                inner, 0.0, out.data(), rows);
  }
};

constexpr ddx::rt::TensorKernelTable cblas_table =
    ddx::rt::tensor_kernel_table<CblasKernels>();

void expect_close(std::span<const double> got, std::span<const double> want) {
  ASSERT_EQ(got.size(), want.size());
  for (const auto [g, w] : std::views::zip(got, want)) {
    EXPECT_NEAR(g, w, 1e-13 * std::max(1.0, std::abs(w)));
  }
}

std::vector<double> filled(std::size_t count, std::size_t seed) {
  return std::views::iota(0uz, count) |
         std::views::transform([seed](std::size_t i) {
           return std::sin(static_cast<double>(i * 7 + seed));
         }) |
         std::ranges::to<std::vector<double>>();
}

// The same numbers as the reference table to rounding, through both products
// and the transposed one the reverse sweep reads.
TEST(RtCblas, KernelsAgreeWithTheReferenceTable) {
  constexpr std::size_t n = 30;
  const auto m = filled(n * n, 3);
  const auto model = [&] {
    const auto x = ddx::rt::vec("x", n);
    const auto M = ddx::rt::mat_constant(n, n, m);
    return dot(x, gemv(M, x)) + sum(log(x) * gemv(M, x, true)) * 0.5;
  };
  const auto reference = ddx::rt::equation(model);
  auto cblas = ddx::rt::equation(model);
  cblas.tensor_kernels(cblas_table);
  EXPECT_EQ(cblas.tensor_kernels().name, "cblas");

  const auto at = std::views::iota(0uz, n) |
                  std::views::transform([](std::size_t i) {
                    return 0.5 + 0.01 * static_cast<double>(i);
                  }) |
                  std::ranges::to<std::vector<double>>();
  expect_close(*cblas.gradient(std::span<const double>{at}),
               *reference.gradient(std::span<const double>{at}));
  expect_close(*cblas.hvp(at, std::span<const double>{at}),
               *reference.hvp(at, std::span<const double>{at}));
}

// gemm is reached only where the dense Hessian's shape algebra runs out, so it
// is checked against the reference row directly.
TEST(RtCblas, GemmAgreesWithTheReferenceRow) {
  constexpr std::size_t r = 7, k = 5, c = 4;
  const auto a = filled(r * k, 1);
  const auto b = filled(k * c, 2);
  std::vector<double> got(r * c), want(r * c);
  cblas_table.gemm(r, k, c, a, b, got);
  ddx::rt::reference_table.gemm(r, k, c, a, b, want);
  expect_close(got, want);
}

} // namespace
