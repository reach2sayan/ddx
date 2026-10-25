#include "rt/equation.hpp"
#include "rt/kernels.hpp"
#include "rt/tensor.hpp"

#include <gtest/gtest.h>

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

// Any CBLAS under the tensor nodes -- OpenBLAS, MKL, Accelerate, the reference
// netlib -- as a consumer writes it.  A table's matrices are column-major with
// contiguous columns, so the leading dimension is the row count.
namespace {

// The reference rows, with the three BLAS has a routine for replaced: the d
// routines at double, the s ones at float.
template <typename T> struct CblasKernels : ddx::rt::reference::Kernels<T> {
  using typename ddx::rt::reference::Kernels<T>::In;
  using typename ddx::rt::reference::Kernels<T>::Out;
  static constexpr std::string_view name = "cblas";
  static constexpr bool single = std::same_as<T, float>;

  static void gemv(std::size_t r, std::size_t c, In m, In v, Out out,
                   bool transposed) noexcept {
    const auto rows = static_cast<int>(r);
    const auto how = transposed ? CblasTrans : CblasNoTrans;
    if constexpr (single) {
      cblas_sgemv(CblasColMajor, how, rows, static_cast<int>(c), 1.0f, m.data(),
                  rows, v.data(), 1, 0.0f, out.data(), 1);
    } else {
      cblas_dgemv(CblasColMajor, how, rows, static_cast<int>(c), 1.0, m.data(),
                  rows, v.data(), 1, 0.0, out.data(), 1);
    }
  }
  [[nodiscard]] static T dot(In u, In v) noexcept {
    const auto n = static_cast<int>(u.size());
    if constexpr (single) {
      return cblas_sdot(n, u.data(), 1, v.data(), 1);
    } else {
      return cblas_ddot(n, u.data(), 1, v.data(), 1);
    }
  }
  static void gemm(std::size_t r, std::size_t k, std::size_t c, In a, In b,
                   Out out) noexcept {
    const auto rows = static_cast<int>(r);
    const auto inner = static_cast<int>(k);
    if constexpr (single) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows,
                  static_cast<int>(c), inner, 1.0f, a.data(), rows, b.data(),
                  inner, 0.0f, out.data(), rows);
    } else {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rows,
                  static_cast<int>(c), inner, 1.0, a.data(), rows, b.data(),
                  inner, 0.0, out.data(), rows);
    }
  }
};

template <typename T>
constexpr auto cblas_table = ddx::rt::tensor_kernel_table<CblasKernels<T>, T>();

// Relative to the scalar's own rounding.
template <typename T> constexpr double tolerance = 1e-13;
template <> constexpr double tolerance<float> = 1e-4;

template <typename T>
void expect_close(std::span<const T> got, std::span<const T> want) {
  ASSERT_EQ(got.size(), want.size());
  for (const auto [g, w] : std::views::zip(got, want)) {
    EXPECT_NEAR(g, w, tolerance<T> * std::max(1.0, std::abs(double{w})));
  }
}

template <typename T>
std::vector<T> filled(std::size_t count, std::size_t seed) {
  return std::views::iota(0uz, count) |
         std::views::transform([seed](std::size_t i) {
           return static_cast<T>(std::sin(static_cast<double>(i * 7 + seed)));
         }) |
         std::ranges::to<std::vector<T>>();
}

// The same numbers as the reference table to rounding, through both products
// and the transposed one the reverse sweep reads.
template <typename T> void cblas_agrees_with_the_reference_table() {
  constexpr std::size_t n = 30;
  const auto m = filled<T>(n * n, 3);
  const auto model = [&] {
    const auto x = ddx::rt::vec<T>("x", n);
    const auto M = ddx::rt::mat_constant<T>(n, n, m);
    return dot(x, gemv(M, x)) + sum(log(x) * gemv(M, x, true)) * T(0.5);
  };
  const auto reference = ddx::rt::equation<T>(model);
  auto cblas = ddx::rt::equation<T>(model);
  cblas.tensor_kernels(cblas_table<T>);
  EXPECT_EQ(cblas.tensor_kernels().name, "cblas");

  const auto at = std::views::iota(0uz, n) |
                  std::views::transform([](std::size_t i) {
                    return static_cast<T>(0.5 + 0.01 * static_cast<double>(i));
                  }) |
                  std::ranges::to<std::vector<T>>();
  expect_close<T>(*cblas.gradient(std::span<const T>{at}),
                  *reference.gradient(std::span<const T>{at}));
  expect_close<T>(*cblas.hvp(at, std::span<const T>{at}),
                  *reference.hvp(at, std::span<const T>{at}));
}

TEST(RtCblas, KernelsAgreeWithTheReferenceTable) {
  cblas_agrees_with_the_reference_table<double>();
}
TEST(RtCblas, FloatKernelsAgreeWithTheReferenceTable) {
  cblas_agrees_with_the_reference_table<float>();
}

// gemm is reached only where the dense Hessian's shape algebra runs out, so it
// is checked against the reference row directly.
template <typename T> void gemm_agrees_with_the_reference_row() {
  constexpr std::size_t r = 7, k = 5, c = 4;
  const auto a = filled<T>(r * k, 1);
  const auto b = filled<T>(k * c, 2);
  std::vector<T> got(r * c), want(r * c);
  cblas_table<T>.gemm(r, k, c, a, b, got);
  ddx::rt::reference_table<T>.gemm(r, k, c, a, b, want);
  expect_close<T>(got, want);
}

TEST(RtCblas, GemmAgreesWithTheReferenceRow) {
  gemm_agrees_with_the_reference_row<double>();
}
TEST(RtCblas, FloatGemmAgreesWithTheReferenceRow) {
  gemm_agrees_with_the_reference_row<float>();
}

} // namespace
