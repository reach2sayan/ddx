#pragma once

#include "ops/numeric.hpp"

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <utility>

// Reverse-mode rules for the ops outside unary_math.hpp, shared by the rt graph
// and Dual. adjoint<I> is what reaches operand I; no rule reads its own primal.
namespace ddx::impl::detail {

// sign(0) is 0, the usual subgradient convention. A graph handle finds its own
// Sign node through ADL instead.
template <Numeric T>
  requires std::totally_ordered<T>
[[nodiscard]] constexpr T sign(const T &a) noexcept {
  return a > T{} ? T{1} : a < T{} ? T{-1} : T{a - a};
}

// A rule that takes only the adjoint reads no primal, which tells the sweep it
// may skip storing the operands.
template <Numeric T> struct SumOpFn {
  template <std::size_t>
  [[nodiscard]] static constexpr T adjoint(const auto &adj) noexcept {
    return adj;
  }
};

// d(a*b) = da*b + a*db, so a's adjoint sits left of b and b's right of a. A
// non-commutative scalar would also need a transpose.
template <Numeric T> struct MultiplyOpFn {
  template <std::size_t I>
  [[nodiscard]] static constexpr T adjoint(const auto &adj, const auto &a,
                                           const auto &b) noexcept {
    if constexpr (I == 0) {
      return adj * b;
    } else {
      return a * adj;
    }
  }
};

template <Numeric T> struct NegateOpFn {
  template <std::size_t>
  [[nodiscard]] static constexpr T adjoint(const auto &adj) noexcept {
    return -adj;
  }
};

// a / b is right division (see DivideOp::derivative), so a non-commutative
// scalar needs its own operand order.
template <Numeric T> struct DivideOpFn {
  template <std::size_t I>
  [[nodiscard]] static constexpr T adjoint(const auto &adj, const auto &a,
                                           const auto &b) noexcept {
    // Divides twice by b, because b*b overflows past ~1.3e154 and a/b is
    // already the primal on a graph.
    if constexpr (I == 0) {
      return adj / b;
    } else if constexpr (CCommutativeMultiply<T>) {
      return -adj * (a / b) / b;
    } else {
      return -((a / b) * adj) / b;
    }
  }
};

template <Numeric T> struct PowOpFn {
  template <std::size_t I>
  [[nodiscard]] static constexpr T adjoint(const auto &adj, const auto &a,
                                           const auto &b) noexcept {
    using std::pow, std::log;
    // At a == 0 each partial is 0·inf, but its vanishing factor makes it an
    // exact zero. A graph handle tests with a select node instead of a branch.
    const auto unless_vanishing = [&a](const T &d, const T &factor) -> T {
      if constexpr (std::equality_comparable<T>) {
        return a == T{} && factor == T{} ? T{} : d;
      } else if constexpr (requires { select(a == T{0}, d, d); }) {
        return select((a == T{0}) * (factor == T{0}), T{0}, d);
      } else {
        return d;
      }
    };
    if constexpr (I == 0) {
      return unless_vanishing(adj * b * pow(a, b - T{1}), b);
    } else {
      const T p = pow(a, b);
      return unless_vanishing(adj * p * log(a), p);
    }
  }
};

// atan2(y, x): operand 0 is y and operand 1 is x.
template <Numeric T> struct Atan2OpFn {
  template <std::size_t I>
  [[nodiscard]] static constexpr T adjoint(const auto &adj, const auto &y,
                                           const auto &x) noexcept {
    using std::hypot;
    // Divides twice by hypot, because x² + y² overflows past 1e154.
    const T h = hypot(x, y);
    if constexpr (I == 0) {
      return adj * (x / h) / h;
    } else {
      return -adj * (y / h) / h;
    }
  }
};

template <Numeric T> struct HypotOpFn {
  template <std::size_t I>
  [[nodiscard]] static constexpr T adjoint(const auto &adj, const auto &x,
                                           const auto &y) noexcept {
    using std::hypot;
    // x / h first, since that quotient lies in [-1, 1] and x * adj can
    // overflow.
    return adj * ((I == 0 ? x : y) / hypot(x, y));
  }
};

template <Numeric T> struct AbsOpFn {
  template <std::size_t>
  [[nodiscard]] static constexpr T adjoint(const auto &adj,
                                           const auto &u) noexcept {
    return adj * sign(u);
  }
};

template <Numeric T> struct SignOpFn {
  template <std::size_t>
  [[nodiscard]] static constexpr T adjoint(const auto &) noexcept {
    return T{};
  }
};

// Branch-free through sign(l - r): exact away from a tie and half to each side
// at one, matching ExtremumOp::derivative.
template <Numeric T, bool IsMax> struct ExtremumOpFn {
  template <std::size_t I>
  [[nodiscard]] static constexpr T adjoint(const auto &adj, const auto &l,
                                           const auto &r) noexcept {
    const T s = sign(l - r);
    if constexpr ((I == 0) == IsMax) {
      return adj * ((T{1} + s) / T{2});
    } else {
      return adj * ((T{1} - s) / T{2});
    }
  }
};

template <Numeric T> using MaxOpFn = ExtremumOpFn<T, true>;
template <Numeric T> using MinOpFn = ExtremumOpFn<T, false>;

// Whether a rule reads its operands' primals, judged from its signature.
template <template <Numeric> class Fn, Numeric T>
inline constexpr bool reads_primals_v =
    !requires(const T &adj) { Fn<T>::template adjoint<0>(adj); };

// Operand I's adjoint. The caller passes every operand, and the rule's
// signature decides whether any is read.
template <template <Numeric> class Fn, std::size_t I, Numeric T>
[[nodiscard]] constexpr T
adjoint_of(const T &adj, [[maybe_unused]] const auto &...operands) noexcept {
  if constexpr (reads_primals_v<Fn, T>) {
    return Fn<T>::template adjoint<I>(adj, operands...);
  } else {
    return Fn<T>::template adjoint<I>(adj);
  }
}

// The adjoints of every operand, in operand order.
template <template <Numeric> class Fn, Numeric T>
[[nodiscard]] constexpr auto adjoints_of(const T &adj,
                                         const auto &...operands) noexcept {
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::array<T, sizeof...(I)>{adjoint_of<Fn, I>(adj, operands...)...};
  }(std::make_index_sequence<sizeof...(operands)>{});
}

} // namespace ddx::impl::detail
