#pragma once

#include "ops/numeric.hpp"

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <utility>

// The reverse-mode rule for every op outside unary_math.hpp's eighteen, one
// operand at a time: adjoint<I> is what reaches operand I.  The arena
// instantiates them at rt::Expression and asks only for the operands a
// derivative moves; Dual instantiates them at its component.  No rule reads
// its own primal, so pow and hypot recompute it, which costs a graph nothing.
namespace ddx::impl::detail {

// The number for an arithmetic T, the Sign node for a graph handle: ordinary
// lookup finds this and ADL finds rt::Expression's hidden friend.  sign(0) is
// 0, the usual subgradient convention, and a - a reaches only ±0 and NaN.
template <Numeric T>
  requires std::totally_ordered<T>
[[nodiscard]] constexpr T sign(const T &a) noexcept {
  return a > T{} ? T{1} : a < T{} ? T{-1} : T{a - a};
}

// A rule that takes only the adjoint reads no primal, and the sweep derives
// from that signature which stores it may skip -- a sum, and so a difference,
// is the biggest source of skippable stores.
template <Numeric T> struct SumOpFn {
  template <std::size_t>
  [[nodiscard]] static constexpr T adjoint(const auto &adj) noexcept {
    return adj;
  }
};

// d(a*b) = da*b + a*db, so `a`'s adjoint multiplies on the LEFT of b and `b`'s
// on the RIGHT of a.  Sided, not complete: a non-commutative scalar would also
// need a transpose.
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

// `a / b` is right division, so the two spellings do not fold into one; see
// DivideOp::derivative.  A graph handle's branch is decided by its scalar, and
// the negation's place decides contraction, so the commutative one stays.
template <Numeric T> struct DivideOpFn {
  template <std::size_t I>
  [[nodiscard]] static constexpr T adjoint(const auto &adj, const auto &a,
                                           const auto &b) noexcept {
    // Twice by b, never once by b*b: the square overflows past ~1.3e154, and
    // on a graph a/b is the primal quotient already.
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
    // Two 0 * inf at a == 0: b * a^(b-1) with b == 0, where a^0 is the
    // constant 1, and a^b * log(a) with a^b == 0, where 0^b is the constant 0.
    // Both are the zero the vanishing factor says.  A number asks with a
    // comparison; a graph handle's comparison is a node, so it asks with a
    // select, which the arena folds to the bare partial wherever the
    // factor is a literal.
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

// lhs is y, rhs is x.
template <Numeric T> struct Atan2OpFn {
  template <std::size_t I>
  [[nodiscard]] static constexpr T adjoint(const auto &adj, const auto &y,
                                           const auto &x) noexcept {
    using std::hypot;
    // Scaled by hypot: x² + y² overflows past 1e154.
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
    // x / h first: the quotient is in [-1, 1] where x * adj can overflow.
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

// Branch-free, since a node has no comparisons: the kink is a sign -- exact
// away from the tie, half to each side at it, as ExtremumOp's symbolic
// derivative spells the same expansion.
template <Numeric T, bool IsMax> struct ExtremumOpFn {
  template <std::size_t I>
  [[nodiscard]] static constexpr T adjoint(const auto &adj, const auto &l,
                                           const auto &r) noexcept {
    const T s = sign(l - r);
    // max's left operand takes the high half, min's the low.
    if constexpr ((I == 0) == IsMax) {
      return adj * ((T{1} + s) / T{2});
    } else {
      return adj * ((T{1} - s) / T{2});
    }
  }
};

template <Numeric T> using MaxOpFn = ExtremumOpFn<T, true>;
template <Numeric T> using MinOpFn = ExtremumOpFn<T, false>;

// Whether a rule reads its operands' primals, read off its signature.
template <template <Numeric> class Fn, Numeric T>
inline constexpr bool reads_primals_v =
    !requires(const T &adj) { Fn<T>::template adjoint<0>(adj); };

// Operand I's adjoint.  The one call shape for both sides: the caller hands
// every operand and the descriptor's arity decides whether any is read.
template <template <Numeric> class Fn, std::size_t I, Numeric T>
[[nodiscard]] constexpr T
adjoint_of(const T &adj, [[maybe_unused]] const auto &...operands) noexcept {
  if constexpr (reads_primals_v<Fn, T>) {
    return Fn<T>::template adjoint<I>(adj, operands...);
  } else {
    return Fn<T>::template adjoint<I>(adj);
  }
}

// Every operand's, in operand order.
template <template <Numeric> class Fn, Numeric T>
[[nodiscard]] constexpr auto adjoints_of(const T &adj,
                                         const auto &...operands) noexcept {
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::array<T, sizeof...(I)>{adjoint_of<Fn, I>(adj, operands...)...};
  }(std::make_index_sequence<sizeof...(operands)>{});
}

} // namespace ddx::impl::detail
