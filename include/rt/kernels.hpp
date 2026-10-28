#pragma once

// What computes a tensor node: a backend is a type satisfying TensorKernels,
// a TensorKernelTable is that type erased to function pointers, and the
// interpreter holds one table per scalar, the equation's own.  The reference
// backend below is plain loops with nothing behind it; a consumer with Eigen or
// a BLAS writes a backend over it (tests/eigen and tests/cblas each hold one)
// and hands the table over once.
//
// Spans at the boundary, not a library's types; a matrix is column-major, and
// every vector a kernel sees is contiguous.  Results may differ between
// backends in the last bits: a summation order is the only thing a table
// chooses.

#include "rt/apply.hpp"
#include "rt/opcode.hpp"
#include "util/config.hpp"
#include "util/export.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
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
template <class B, typename T = double>
concept TensorKernels =
    requires(std::span<const T> u, std::span<const T> v, std::span<T> out,
             std::size_t r, std::size_t c, bool t, T s, OpCode op) {
      { B::name } -> std::convertible_to<std::string_view>;
      { B::gemv(r, c, u, v, out, t) } -> std::same_as<void>;
      { B::dot(u, v) } -> std::convertible_to<T>;
      { B::scale(s, v, out) } -> std::same_as<void>;
      { B::map(op, v, out) } -> std::same_as<void>;
      { B::zip(op, u, v, out) } -> std::same_as<void>;
      { B::sum(v) } -> std::convertible_to<T>;
      { B::gemm(r, c, r, u, v, out) } -> std::same_as<void>;
    };

// A backend as the interpreter holds it: one record of function pointers, so
// no backend type reaches an equation's type.
template <impl::Numeric T = double> struct TensorKernelTable {
  using In = std::span<const T>;
  using Out = std::span<T>;
  void (*gemv)(std::size_t, std::size_t, In, In, Out, bool) = nullptr;
  T (*dot)(In, In) = nullptr;
  void (*scale)(T, In, Out) = nullptr;
  void (*map)(OpCode, In, Out) = nullptr;
  void (*zip)(OpCode, In, In, Out) = nullptr;
  T (*sum)(In) = nullptr;
  void (*gemm)(std::size_t, std::size_t, std::size_t, In, In, Out) = nullptr;
  std::string_view name;
};

// The only way to make one: a backend that misses a member fails here, against
// the concept, and not at the first call.
template <typename B, impl::Numeric T = double>
  requires TensorKernels<B, T>
[[nodiscard]] consteval TensorKernelTable<T> tensor_kernel_table() noexcept {
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
// backend ddx ships and tests against, at any scalar.  The dot and gemv inner
// loops take one rounding per term, as the scalar sweep's contraction does.
template <impl::Numeric T = double> struct Kernels {
  static constexpr std::string_view name = "reference";
  using In = typename TensorKernelTable<T>::In;
  using Out = typename TensorKernelTable<T>::Out;

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
    std::ranges::fill(out, T{});
    for (std::size_t j = 0; j < c; ++j) {
      const T x = v[j];
      const T *DDX_RESTRICT col = m.data() + j * r;
      T *DDX_RESTRICT y = out.data();
      for (std::size_t i = 0; i < r; ++i) {
        if constexpr (std::floating_point<T>) {
          y[i] = std::fma(col[i], x, y[i]);
        } else { // a field-like scalar: no fma, and no rounding to keep
          y[i] = y[i] + col[i] * x;
        }
      }
    }
  }

  // The order libstdc++'s transform_reduce takes, and the fmas GCC forms in
  // it at -O2, spelled out so neither the library nor -ffp-contract decides:
  // blocks of four as two products each finished by an fma, then a fused tail.
  [[nodiscard]] static constexpr T dot(In u, In v) noexcept {
    const auto fma = [](T a, T b, T c) -> T {
      if constexpr (std::floating_point<T>) {
        return std::fma(a, b, c);
      } else {
        return a * b + c;
      }
    };
    const std::size_t whole = u.size() / 4 * 4;
    const T blocks = std::ranges::fold_left(
        std::views::zip(u.first(whole) | std::views::chunk(4),
                        v.first(whole) | std::views::chunk(4)),
        T{}, [&fma](T acc, const auto &block) {
          const auto &[a, b] = block;
          return acc +
                 (fma(a[1], b[1], a[0] * b[0]) + fma(a[3], b[3], a[2] * b[2]));
        });
    return std::ranges::fold_left(
        std::views::zip(u.subspan(whole), v.subspan(whole)), blocks,
        [&fma](T acc, const auto &ab) {
          return fma(std::get<0>(ab), std::get<1>(ab), acc);
        });
  }

  static constexpr void scale(T s, In v, Out out) noexcept {
    std::ranges::transform(v, out.begin(), [s](T x) { return s * x; });
  }

  // The scalar table's row, picked once and then looped: apply<T> per
  // element would re-dispatch the opcode inside the loop.
  static constexpr void map(OpCode op, In v, Out out) noexcept {
    detail::dispatch<T>(op, [&]<typename R>(R) {
      using Fn = typename R::functor;
      if constexpr (!std::is_void_v<Fn> && R::arity == 1) {
        std::ranges::transform(v, out.begin(), [](T x) {
          return detail::supported<Fn, T, R::ok>(x);
        });
      } else {
        std::ranges::fill(out, T{});
      }
    });
  }

  static constexpr void zip(OpCode op, In u, In v, Out out) noexcept {
    detail::dispatch<T>(op, [&]<typename R>(R) {
      using Fn = typename R::functor;
      if constexpr (!std::is_void_v<Fn> && R::arity == 2) {
        std::ranges::transform(u, v, out.begin(), [](T a, T b) {
          return detail::supported<Fn, T, R::ok>(a, b);
        });
      } else {
        std::ranges::fill(out, T{});
      }
    });
  }

  // std::reduce's order: blocks of four summed as pairs, then a left fold.
  [[nodiscard]] static constexpr T sum(In v) noexcept {
    const std::size_t whole = v.size() / 4 * 4;
    const T blocks = std::ranges::fold_left(
        v.first(whole) | std::views::chunk(4), T{}, [](T acc, const auto &b) {
          return acc + ((b[0] + b[1]) + (b[2] + b[3]));
        });
    return std::ranges::fold_left(v.subspan(whole), blocks, std::plus<>{});
  }

  // out = A B, one column of B at a time: a gemv per column.
  static constexpr void gemm(std::size_t r, std::size_t k, std::size_t c, In a,
                             In bm, Out out) noexcept {
    for (std::size_t j = 0; j < c; ++j) {
      gemv(r, k, a, bm.subspan(j * k, k), out.subspan(j * r, r), false);
    }
  }
};
static_assert(TensorKernels<Kernels<>>);
static_assert(TensorKernels<Kernels<float>, float>);

} // namespace reference

template <impl::Numeric T = double>
inline constexpr TensorKernelTable<T> reference_table =
    tensor_kernel_table<reference::Kernels<T>, T>();

namespace detail {
// One per scalar.  libddx holds double's and float's, so a program and the
// library set and read the same one; any other scalar's is the reader's own.
template <impl::Numeric T>
DDX_TEMPLATE_API std::atomic<const TensorKernelTable<T> *> &
default_tensor_kernels() noexcept {
  static std::atomic<const TensorKernelTable<T> *> current{&reference_table<T>};
  return current;
}
} // namespace detail

// The process default at T: what every T equation's tensor nodes run on unless
// the equation was handed a table of its own.  Set once, before any sweep -- a
// race to set it is the caller's, as it is for a jit::Compiler.
template <impl::Numeric T>
void use_tensor_kernels(const TensorKernelTable<T> &table) noexcept {
  detail::default_tensor_kernels<T>().store(&table, std::memory_order_release);
}
template <impl::Numeric T = double>
[[nodiscard]] const TensorKernelTable<T> &tensor_kernels() noexcept {
  return *detail::default_tensor_kernels<T>().load(std::memory_order_acquire);
}

} // namespace ddx::rt

DDX_EXTERN_INSTANCE(
    DDX_INSTANCE_API std::atomic<const ::ddx::rt::TensorKernelTable<double> *>
        & ::ddx::rt::detail::default_tensor_kernels<double>() noexcept)
DDX_EXTERN_INSTANCE(
    DDX_INSTANCE_API std::atomic<const ::ddx::rt::TensorKernelTable<float> *>
        & ::ddx::rt::detail::default_tensor_kernels<float>() noexcept)
