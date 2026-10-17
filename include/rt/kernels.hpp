#pragma once

// What computes a tensor node: a backend is a type satisfying TensorKernels,
// a KernelTable is that type erased to function pointers, and the interpreter
// holds one table.  The reference backend below is plain loops with nothing
// behind it; a consumer with Eigen or a BLAS includes its own header (one
// ships beside this: rt/eigen_kernels.hpp) and hands the table over once.
//
// Spans at the boundary, not a library's types; a matrix is column-major, and
// every vector a kernel sees is contiguous.  Results may differ between
// backends in the last bits -- a summation order is the only thing a table
// chooses -- which is why a saved file records the table's name.

#include "rt/apply.hpp"
#include "rt/opcode.hpp"
#include "util/config.hpp"
#include "util/export.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace ddx::rt {

// `M` is r x c column-major; gemv writes M v (c in, r out) or, transposed,
// M^T v (r in, c out).  map and zip are the scalar table's rows, elementwise.
// gemm(r, k, c, A, B, out) writes the r x c product of A (r x k) and B (k x c),
// the one cubic kernel, which the dense Hessian sweep reaches only where its
// shape algebra ran out.
template <class B>
concept TensorKernels =
    requires(std::span<const double> u, std::span<const double> v,
             std::span<double> out, std::size_t r, std::size_t c, bool t,
             double s, OpCode op) {
      { B::name } -> std::convertible_to<std::string_view>;
      { B::gemv(r, c, u, v, out, t) } -> std::same_as<void>;
      { B::dot(u, v) } -> std::convertible_to<double>;
      { B::scale(s, v, out) } -> std::same_as<void>;
      { B::map(op, v, out) } -> std::same_as<void>;
      { B::zip(op, u, v, out) } -> std::same_as<void>;
      { B::sum(v) } -> std::convertible_to<double>;
      { B::gemm(r, c, r, u, v, out) } -> std::same_as<void>;
    };

// A backend as the interpreter holds it: one record of function pointers, so
// no backend type reaches an equation's type.
struct KernelTable {
  using In = std::span<const double>;
  using Out = std::span<double>;
  void (*gemv)(std::size_t, std::size_t, In, In, Out, bool) = nullptr;
  double (*dot)(In, In) = nullptr;
  void (*scale)(double, In, Out) = nullptr;
  void (*map)(OpCode, In, Out) = nullptr;
  void (*zip)(OpCode, In, In, Out) = nullptr;
  double (*sum)(In) = nullptr;
  void (*gemm)(std::size_t, std::size_t, std::size_t, In, In, Out) = nullptr;
  std::string_view name;
};

// The only way to make one: a backend that misses a member fails here, against
// the concept, and not at the first call.
template <TensorKernels B>
[[nodiscard]] consteval KernelTable kernel_table() noexcept {
  return {.gemv = &B::gemv,
          .dot = &B::dot,
          .scale = &B::scale,
          .map = &B::map,
          .zip = &B::zip,
          .sum = &B::sum,
          .gemm = &B::gemm,
          .name = B::name};
}

namespace reference {

// Straight loops, each written so the compiler vectorises it; the one
// backend ddx ships and tests against.  The dot and gemv inner loops take one
// rounding per term, as the scalar sweep's contraction does.
struct Kernels {
  static constexpr std::string_view name = "reference";
  using In = KernelTable::In;
  using Out = KernelTable::Out;

  static constexpr void gemv(std::size_t r, std::size_t c, In m, In v, Out out,
                             bool transposed) noexcept {
    if (transposed) {
      // out[j] = column_j . v
      for (std::size_t j = 0; j < c; ++j) {
        out[j] = dot(m.subspan(j * r, r), v);
      }
      return;
    }
    // out = sum_j v[j] column_j: one pass per column, contiguous both sides.
    std::ranges::fill(out, 0.0);
    for (std::size_t j = 0; j < c; ++j) {
      const double x = v[j];
      const double *DDX_RESTRICT col = m.data() + j * r;
      double *DDX_RESTRICT y = out.data();
      for (std::size_t i = 0; i < r; ++i) {
        y[i] = std::fma(col[i], x, y[i]);
      }
    }
  }

  [[nodiscard]] static constexpr double dot(In u, In v) noexcept {
    return std::transform_reduce(u.begin(), u.end(), v.begin(), 0.0,
                                 std::plus<>{},
                                 [](double a, double b) { return a * b; });
  }

  static constexpr void scale(double s, In v, Out out) noexcept {
    std::ranges::transform(v, out.begin(), [s](double x) { return s * x; });
  }

  // The scalar table's row, picked once and then looped: apply<double> per
  // element would re-dispatch the opcode inside the loop.
  static constexpr void map(OpCode op, In v, Out out) noexcept {
    detail::dispatch<double>(op, [&]<typename R>(R) {
      using Fn = typename R::functor;
      if constexpr (!std::is_void_v<Fn> && R::arity == 1) {
        std::ranges::transform(v, out.begin(), [](double x) {
          return detail::supported<Fn, double, R::ok>(x);
        });
      } else {
        std::ranges::fill(out, 0.0);
      }
    });
  }

  static constexpr void zip(OpCode op, In u, In v, Out out) noexcept {
    detail::dispatch<double>(op, [&]<typename R>(R) {
      using Fn = typename R::functor;
      if constexpr (!std::is_void_v<Fn> && R::arity == 2) {
        std::ranges::transform(u, v, out.begin(), [](double a, double b) {
          return detail::supported<Fn, double, R::ok>(a, b);
        });
      } else {
        std::ranges::fill(out, 0.0);
      }
    });
  }

  [[nodiscard]] static constexpr double sum(In v) noexcept {
    return std::reduce(v.begin(), v.end(), 0.0);
  }

  // out = A B, one column of B at a time: a gemv per column.
  static constexpr void gemm(std::size_t r, std::size_t k, std::size_t c, In a,
                             In bm, Out out) noexcept {
    for (std::size_t j = 0; j < c; ++j) {
      gemv(r, k, a, bm.subspan(j * k, k), out.subspan(j * r, r), false);
    }
  }
};
static_assert(TensorKernels<Kernels>);

} // namespace reference

inline constexpr KernelTable reference_table =
    kernel_table<reference::Kernels>();

// The process default: what every equation's tensor nodes run on unless the
// equation was handed a table of its own.  Set once, before any sweep -- a
// race to set it is the caller's, as it is for a jit::Compiler.
DDX_API void use_kernels(const KernelTable &table) noexcept;
[[nodiscard]] DDX_API const KernelTable &kernels() noexcept;

} // namespace ddx::rt
