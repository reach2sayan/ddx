#pragma once

// Batch columns as Eigen matrices, without a copy either way.  Optional:
// nothing in ddx includes this, and only a consumer that has Eigen does.
//
// A batch's inputs are column pointers, each n long, which is exactly what a
// column-major n x inputs matrix holds; its output blocks are the same, so an
// n x columns matrix takes a whole block.  A row-major matrix's columns are
// strided, which no column pointer can say -- refused at compile time.
//
// "Optional" is enforced, not just stated: this header is installed with the
// rest, and tests/package compiles every installed header in one translation
// unit.  Without Eigen it is inert rather than a hard error.

#if defined(__has_include) && __has_include(<Eigen/Core>)
#define DDX_HAS_EIGEN 1
#else
#define DDX_HAS_EIGEN 0
#endif

#if DDX_HAS_EIGEN

#include "rt/coupling.hpp"
#include "util/ranges.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <ranges>
#include <vector>

namespace ddx::rt::eigen {

// One pointer per column of `m`, into `m`'s own storage.
template <typename D>
  requires(!D::IsRowMajor && D::InnerStrideAtCompileTime == 1)
[[nodiscard]] std::vector<const double *>
columns_of(const Eigen::PlainObjectBase<D> &m) {
  return std::views::iota(Eigen::Index{0}, m.cols()) |
         std::views::transform(
             [&m](Eigen::Index j) { return m.data() + j * m.outerStride(); }) |
         impl::to<std::vector<const double *>>();
}
template <typename D>
  requires(!D::IsRowMajor && D::InnerStrideAtCompileTime == 1)
[[nodiscard]] std::vector<double *> columns_of(Eigen::PlainObjectBase<D> &m) {
  return std::views::iota(Eigen::Index{0}, m.cols()) |
         std::views::transform(
             [&m](Eigen::Index j) { return m.data() + j * m.outerStride(); }) |
         impl::to<std::vector<double *>>();
}

// The Jacobian at one point of a batch, dense m x symbols, from a jacobian
// block: n rows, one column per pattern cell.  Every cell the pattern leaves
// out is zero.
template <typename B, typename J>
void scatter(const Sparsity &pattern, const Eigen::MatrixBase<B> &block,
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

// The Hessian at one point, dense symbols x symbols, from a hessian block
// compressed by `coloring`.
template <typename B, typename H>
void scatter(const Coloring &coloring, const Eigen::MatrixBase<B> &block,
             Eigen::Index point, Eigen::MatrixBase<H> &dense) {
  const auto n = static_cast<Eigen::Index>(coloring.color.size());
  dense.derived().setZero(n, n);
  for (const Cell &cell : coloring.entries()) {
    dense(static_cast<Eigen::Index>(cell.row),
          static_cast<Eigen::Index>(cell.column)) =
        block(point, static_cast<Eigen::Index>(cell.slot));
  }
}

} // namespace ddx::rt::eigen

#endif // DDX_HAS_EIGEN
