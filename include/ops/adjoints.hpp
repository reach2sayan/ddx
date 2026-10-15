#pragma once

#include "ops/numeric.hpp"

#include <array>
#include <cmath>
#include <concepts>

// The reverse-mode rule for every op outside unary_math.hpp's eighteen.  Both
// sides call these: the sweep instantiates them at a number, the graph builder
// at RTExpression.  No rule reads its own primal -- the compile-time cache
// never stores a node's own slot -- so pow and hypot recompute it, which costs
// a graph nothing.  max/min are the one pair not shared; see ExtremumOpFn.
namespace ddx::impl::detail {

// The number for an arithmetic T, the Sign node for a graph handle: ordinary
// lookup finds this and ADL finds RTExpression's hidden friend.  sign(0) is 0,
// the usual subgradient convention, and a - a reaches only ±0 and NaN.
template <Numeric T>
  requires std::totally_ordered<T>
[[nodiscard]] constexpr T sign(const T &a) noexcept {
  return a > T{} ? T{1} : a < T{} ? T{-1} : T{a - a};
}

// A rule that takes only the adjoint reads no primal, and the sweep derives
// from that signature which stores it may skip -- a sum, and so a difference,
// is the biggest source of skippable stores.
template <Numeric T> struct SumOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj) noexcept {
    return {adj, adj};
  }
};

// d(a*b) = da*b + a*db, so `a`'s adjoint multiplies on the LEFT of b and `b`'s
// on the RIGHT of a.  Sided, not complete: a non-commutative scalar would also
// need a transpose.
template <Numeric T> struct MultiplyOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &a, const auto &b) noexcept {
    return {adj * b, a * adj};
  }
};

template <Numeric T> struct NegateOpFn {
  [[nodiscard]] static constexpr std::array<T, 1>
  adjoints(const auto &adj) noexcept {
    return {-adj};
  }
};

// `a / b` is right division, so the two spellings do not fold into one; see
// DivideOp::derivative.  A graph handle's branch is decided by its scalar.
template <Numeric T> struct DivideOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &a, const auto &b) noexcept {
    // Twice by b, never once by b*b: the square overflows past ~1.3e154, and
    // on a graph a/b is the primal quotient already.
    if constexpr (CCommutativeMultiply<T>) {
      return {adj / b, -adj * (a / b) / b};
    } else {
      return {adj / b, -((a / b) * adj) / b};
    }
  }
};

template <Numeric T> struct PowOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &a, const auto &b) noexcept {
    using std::pow, std::log;
    const T p = pow(a, b);
    const T da = adj * b * pow(a, b - T{1});
    const T db = adj * p * log(a);
    // Two 0 * inf at a == 0: b * a^(b-1) with b == 0, where a^0 is the
    // constant 1, and a^b * log(a) with a^b == 0, where 0^b is the constant 0.
    // Both are the zero the vanishing factor says.  A number asks with a
    // comparison; a graph handle's comparison is a node, so it asks with a
    // select, which the builder folds to the bare partial wherever the
    // exponent is a literal.
    if constexpr (std::equality_comparable<T>) {
      if (a == T{}) {
        return {b == T{} ? T{} : da, p == T{} ? T{} : db};
      }
      return {da, db};
    } else if constexpr (requires { select(a == T{0}, da, db); }) {
      const T at_zero = a == T{0};
      return {select(at_zero * (b == T{0}), T{0}, da),
              select(at_zero * (p == T{0}), T{0}, db)};
    } else {
      return {da, db};
    }
  }
};

// lhs is y, rhs is x.
template <Numeric T> struct Atan2OpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &y, const auto &x) noexcept {
    using std::hypot;
    // Scaled by hypot: x² + y² overflows past 1e154.
    const T h = hypot(x, y);
    return {adj * (x / h) / h, -adj * (y / h) / h};
  }
};

template <Numeric T> struct HypotOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &x, const auto &y) noexcept {
    using std::hypot;
    const T h = hypot(x, y);
    // x / h first: the quotient is in [-1, 1] where x * adj can overflow.
    return {adj * (x / h), adj * (y / h)};
  }
};

template <Numeric T> struct AbsOpFn {
  [[nodiscard]] static constexpr std::array<T, 1>
  adjoints(const auto &adj, const auto &u) noexcept {
    return {adj * sign(u)};
  }
};

template <Numeric T> struct SignOpFn {
  [[nodiscard]] static constexpr std::array<T, 1>
  adjoints(const auto &) noexcept {
    return {T{}};
  }
};

// The one rule the two sides do not share: a node has no comparisons, so the
// kink is a sign -- exact away from the tie, NaN at it.  ExtremumOp keeps a
// branching rule for numbers, which is cheaper and can halve a tie.
template <Numeric T, bool IsMax> struct ExtremumOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &l, const auto &r) noexcept {
    const T s = sign(l - r);
    const T lo = (T{1} - s) / T{2};
    const T hi = (T{1} + s) / T{2};
    if constexpr (IsMax) {
      return {adj * hi, adj * lo};
    } else {
      return {adj * lo, adj * hi};
    }
  }
};

template <Numeric T> using MaxOpFn = ExtremumOpFn<T, true>;
template <Numeric T> using MinOpFn = ExtremumOpFn<T, false>;

// Whether a rule reads its operands' primals, read off its signature.
template <template <Numeric> class Fn, Numeric T>
inline constexpr bool reads_primals_v =
    !requires(const T &adj) { Fn<T>::adjoints(adj); };

// The one call shape for both sides: the caller hands every operand and the
// descriptor's arity decides whether any is read.
template <template <Numeric> class Fn, Numeric T>
[[nodiscard]] constexpr auto
adjoints_of(const T &adj, [[maybe_unused]] const auto &...operands) noexcept {
  if constexpr (reads_primals_v<Fn, T>) {
    return Fn<T>::adjoints(adj, operands...);
  } else {
    return Fn<T>::adjoints(adj);
  }
}

} // namespace ddx::impl::detail
