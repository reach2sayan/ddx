#pragma once

// Eigen under the tensor nodes.  Optional, as rt/eigen.hpp is: nothing in ddx
// includes this, and a consumer that has Eigen does, once --
//
//   ddx::rt::use_kernels(ddx::rt::eigen::table);      // every equation
//   eq.kernels(ddx::rt::eigen::table);                // this one
//
// If that Eigen is built over a BLAS (EIGEN_USE_BLAS), the gemv lands there;
// ddx knows nothing about it.  The elementwise rows stay the reference ones:
// a transcendental per element is the same work under any name.  Last bits
// may differ from the reference table's, since a summation order is the one
// thing a table chooses.
//
// Inert without Eigen rather than a hard error: this header is installed with
// the rest, and tests/package compiles every installed header.

#include "rt/eigen.hpp" // DDX_HAS_EIGEN, probed once

#if DDX_HAS_EIGEN

#include "rt/kernels.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <span>
#include <string_view>

namespace ddx::rt::eigen {

struct Kernels : reference::Kernels {
  static constexpr std::string_view name = "eigen";
  using In = KernelTable::In;
  using Out = KernelTable::Out;
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
static_assert(TensorKernels<Kernels>);

inline constexpr KernelTable table = kernel_table<Kernels>();

} // namespace ddx::rt::eigen

#endif // DDX_HAS_EIGEN
