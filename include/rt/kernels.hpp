#pragma once

// Tensor kernel backends: a TensorKernels type, erased into a TensorKernelTable
// of function pointers. Matrices are column-major and vectors contiguous.

#include "rt/apply.hpp"
#include "rt/opcode.hpp"
#include "util/config.hpp"
#include "util/export.hpp"
#include "util/meta.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <string_view>

namespace ddx::rt {

// gemv writes M v, or M^T v when transposed, for an r x c column-major M.
// gemm writes the r x c product of A (r x k) and B (k x c).
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

// A type-erased backend, so that no backend type leaks into an equation's
// type.
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
  // Optional M v and M^T v of a square M in one pass; null means two gemvs.
  void (*gemv_pair)(std::size_t, In, In, Out, Out) = nullptr;
  std::string_view name;
};

// Builds a table, so a backend missing a member fails here against the
// concept rather than at the first call.
template <typename B, impl::Numeric T = double>
  requires TensorKernels<B, T>
[[nodiscard]] consteval TensorKernelTable<T> tensor_kernel_table() noexcept {
  TensorKernelTable<T> t{.gemv = &B::gemv,
                         .dot = &B::dot,
                         .scale = &B::scale,
                         .map = &B::map,
                         .zip = &B::zip,
                         .sum = &B::sum,
                         .gemm = &B::gemm,
                         .name = B::name};
  if constexpr (requires { B::gemv_pair; }) {
    t.gemv_pair = &B::gemv_pair;
  }
  return t;
}

namespace reference {

// The built-in backend of plain vectorisable loops. Reductions use kLanes
// partial sums, and every product goes through madd so results never vary.
template <impl::Numeric T = double> struct Kernels {
  static constexpr std::string_view name = "reference";
  using In = typename TensorKernelTable<T>::In;
  using Out = typename TensorKernelTable<T>::Out;
  static constexpr std::size_t kLanes = 16;

  [[nodiscard]] static constexpr T madd(T a, T b, T c) noexcept {
    if constexpr (std::floating_point<T>) {
      return std::fma(a, b, c);
    } else { // a field-like scalar has no fma and no rounding to match
      return a * b + c;
    }
  }

  // Unrolled at compile time, since a loop over a spilled array costs more
  // than a short row.
  template <std::size_t N>
  [[nodiscard]] static constexpr T fold(std::array<T, N> &acc) noexcept {
    meta::for_each(meta::iota<std::bit_width(N) - 1>, [&](auto level) {
      constexpr std::size_t w = N >> (level + 1);
      meta::for_each(meta::iota<w>,
                     [&](auto l) { acc[l] = acc[l] + acc[l + w]; });
    });
    return acc[0];
  }

  static constexpr void gemv(std::size_t r, std::size_t c, In m, In v, Out out,
                             bool transposed) noexcept {
    if (transposed) {
      // out[j] = column_j . v
      for (std::size_t j = 0; j < c; ++j) {
        out[j] = dot(m.subspan(j * r, r), v);
      }
      return;
    }
    // out = sum_j v[j] column_j, four columns per pass to cut traffic on out.
    // Each out[i] still accumulates j in order.
    std::ranges::fill(out, T{});
    T *DDX_RESTRICT y = out.data();
    std::size_t j = 0;
    for (; j + 4 <= c; j += 4) {
      const T *DDX_RESTRICT c0 = m.data() + j * r;
      const T *DDX_RESTRICT c1 = c0 + r;
      const T *DDX_RESTRICT c2 = c1 + r;
      const T *DDX_RESTRICT c3 = c2 + r;
      const T x0 = v[j], x1 = v[j + 1], x2 = v[j + 2], x3 = v[j + 3];
      for (std::size_t i = 0; i < r; ++i) {
        y[i] = madd(c3[i], x3,
                    madd(c2[i], x2, madd(c1[i], x1, madd(c0[i], x0, y[i]))));
      }
    }
    for (; j < c; ++j) {
      const T *DDX_RESTRICT col = m.data() + j * r;
      const T x = v[j];
      for (std::size_t i = 0; i < r; ++i) {
        y[i] = madd(col[i], x, y[i]);
      }
    }
  }

  // Writes M v into plain and M^T v into transposed in one pass over n x n M.
  static constexpr void gemv_pair(std::size_t n, In m, In v, Out plain,
                                  Out transposed) noexcept {
    constexpr std::size_t L = 8;
    std::ranges::fill(plain, T{});
    T *DDX_RESTRICT y = plain.data();
    const T *DDX_RESTRICT x = v.data();
    const std::size_t whole = n / L * L;
    const auto columns =
        [&]<std::size_t C>(std::integral_constant<std::size_t, C>,
                           std::size_t j0) {
          std::array<const T *, C> col{};
          std::array<T, C> xj{};
          for (std::size_t d = 0; d < C; ++d) {
            col[d] = m.data() + (j0 + d) * n;
            xj[d] = x[j0 + d];
          }
          std::array<std::array<T, L>, C> acc{};
          for (std::size_t i = 0; i < whole; i += L) {
            for (std::size_t l = 0; l < L; ++l) {
              T yi = y[i + l];
              for (std::size_t d = 0; d < C; ++d) {
                const T mi = col[d][i + l];
                yi = madd(mi, xj[d], yi);
                acc[d][l] = madd(mi, x[i + l], acc[d][l]);
              }
              y[i + l] = yi;
            }
          }
          for (std::size_t i = whole; i < n; ++i) {
            for (std::size_t d = 0; d < C; ++d) {
              y[i] = madd(col[d][i], xj[d], y[i]);
              acc[d][i - whole] = madd(col[d][i], x[i], acc[d][i - whole]);
            }
          }
          for (std::size_t d = 0; d < C; ++d) {
            transposed[j0 + d] = fold(acc[d]);
          }
        };
    std::size_t j = 0;
    for (; j + 4 <= n; j += 4) {
      columns(std::integral_constant<std::size_t, 4>{}, j);
    }
    for (; j < n; ++j) {
      columns(std::integral_constant<std::size_t, 1>{}, j);
    }
  }

  // Reduces n elements into kLanes accumulators, then folds them pairwise.
  // step(acc, i) adds element i into acc.
  [[nodiscard]] static constexpr T reduce(std::size_t n, auto step) noexcept {
    std::array<T, kLanes> acc{};
    const std::size_t whole = n / kLanes * kLanes;
    for (std::size_t i = 0; i < whole; i += kLanes) {
      for (std::size_t l = 0; l < kLanes; ++l) {
        acc[l] = step(acc[l], i + l);
      }
    }
    for (std::size_t i = whole; i < n; ++i) {
      acc[i - whole] = step(acc[i - whole], i);
    }
    return fold(acc);
  }

  [[nodiscard]] static constexpr T dot(In u, In v) noexcept {
    return reduce(u.size(), [a = u.data(), b = v.data()](T acc, std::size_t i) {
      return madd(a[i], b[i], acc);
    });
  }

  static constexpr void scale(T s, In v, Out out) noexcept {
    std::ranges::transform(v, out.begin(), [s](T x) { return s * x; });
  }

  // Dispatches the opcode once and then loops, rather than calling apply<T>
  // per element.
  static constexpr void map(OpCode op, In v, Out out) noexcept {
    detail::dispatch<T>(op, [&]<typename R>(R) {
      using Fn = typename R::functor;
      if constexpr (!std::is_void_v<Fn> && R::arity == 1) {
        std::ranges::transform(v, out.begin(), [](T x) {
          return detail::supported<Fn, T, R::computes>(x);
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
          return detail::supported<Fn, T, R::computes>(a, b);
        });
      } else {
        std::ranges::fill(out, T{});
      }
    });
  }

  [[nodiscard]] static constexpr T sum(In v) noexcept {
    return reduce(v.size(),
                  [a = v.data()](T acc, std::size_t i) { return acc + a[i]; });
  }

  // out = A B, computed as one gemv per column of B.
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
// The default table per scalar. libddx exports the double and float ones so
// a program and the library share them.
template <impl::Numeric T>
DDX_TEMPLATE_API std::atomic<const TensorKernelTable<T> *> &
default_tensor_kernels() noexcept {
  constinit static std::atomic<const TensorKernelTable<T> *> current{
      &reference_table<T>};
  return current;
}
} // namespace detail

// Sets the process-wide default table for T. Call it once before any sweep;
// concurrent setting is the caller's responsibility.
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
