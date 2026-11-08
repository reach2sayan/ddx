#pragma once

#include "ops/adjoints.hpp"
#include "ops/algebra.hpp" // RuleOp
#include "ops/numeric.hpp"
#include "ops/unary_math.hpp"
#include "util/fixed_string.hpp"
#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <numeric> // std::midpoint
#include <optional>
#include <string_view>
#include <utility>

namespace ddx::impl {

// How format.hpp prints an op: Prefix -a, Infix a + b, or Function pow(a, b).
enum class Notation : std::uint8_t { Prefix, Infix, Function };

// Precedence of function calls and leaves; a higher value binds tighter.
inline constexpr int precedence_atom = 100;

// The printing and rewriting traits every op states besides its evaluation.
template <Numeric T, typename func, CFixedString auto symbol, Notation note,
          int prec>
struct OpDescriptor {
  using value_type = T;
  static constexpr std::string_view label = symbol.view();
  static constexpr Notation notation = note;
  static constexpr int precedence = prec;
  static_assert(note == Notation::Function || prec < precedence_atom,
                "an op printed without its own parentheses must state a "
                "precedence");

  // The algebraic identities the simplifier may apply; none for most ops.
  static constexpr std::optional<algebra::RuleOp> rule_op = std::nullopt;
};

template <Numeric T, typename func, CFixedString auto symbol,
          Notation note = Notation::Function, int prec = precedence_atom>
  requires std::regular_invocable<func, const T &> &&
           std::convertible_to<std::invoke_result_t<func, const T &>, T>
struct UnaryOp : OpDescriptor<T, func, symbol, note, prec> {
  [[nodiscard]] static constexpr auto eval(const Numeric auto &lhs) noexcept {
    return std::invoke(func{}, lhs);
  }
};

template <Numeric T, typename func, CFixedString auto symbol,
          Notation note = Notation::Function, int prec = precedence_atom>
  requires std::regular_invocable<func, const T &, const T &> &&
           std::convertible_to<std::invoke_result_t<func, const T &, const T &>,
                               T>
struct BinaryOp : OpDescriptor<T, func, symbol, note, prec> {
  [[nodiscard]] static constexpr auto eval(const Numeric auto &lhs,
                                           const Numeric auto &rhs) noexcept {
    return std::invoke(func{}, lhs, rhs);
  }
};

template <Numeric T>
struct SumOp
    : BinaryOp<T, std::plus<void>, FixedString{"+"}, Notation::Infix, 10> {
  static constexpr std::optional rule_op = algebra::RuleOp::Add;

  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    return lhs.derivative() + rhs.derivative();
  }
};

template <Numeric T>
struct MultiplyOp : BinaryOp<T, std::multiplies<void>, FixedString{"*"},
                             Notation::Infix, 20> {
  static constexpr std::optional rule_op = algebra::RuleOp::Mul;
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    auto lmul = lhs.derivative() * rhs;
    auto rmul = lhs * rhs.derivative();
    return std::move(lmul) + std::move(rmul);
  }
};

template <Numeric T>
struct NegateOp
    : UnaryOp<T, std::negate<void>, FixedString{"-"}, Notation::Prefix, 30> {
  static constexpr std::optional rule_op = algebra::RuleOp::Neg;

  // Uses operator- so that values.hpp's folding turns -0 into a Lit.
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept {
    return -lhs.derivative();
  }
};

// Shared by the rewriter and the printer.
namespace detail {
template <typename E> inline constexpr bool is_negation_expr_v = false;
template <Numeric T, CExpression C>
inline constexpr bool is_negation_expr_v<Expression<NegateOp<T>, C>> = true;
} // namespace detail

template <Numeric T>
struct DivideOp
    : BinaryOp<T, std::divides<void>, FixedString{"/"}, Notation::Infix, 20> {
  static constexpr std::optional rule_op = algebra::RuleOp::Div;
  // a / b means a * b^-1, so d(a/b) = (da - (a/b)·db) / b holds even for a
  // non-commuting scalar, and never forms b*b, which overflows past ~1.3e154.
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    return (lhs.derivative() - (lhs / rhs) * rhs.derivative()) / rhs;
  }
};

// Binary math ops. derivative() builds the forward symbolic tree; the reverse
// rule comes from the descriptor in adjoints.hpp.
namespace detail {
// Returns the scalar's 1 or 0 rather than a bool, so the tape keeps one
// element type and a condition is an ordinary value.
template <bool OrEqual> struct compare_impl {
  template <Numeric T>
    requires std::totally_ordered<T>
  constexpr T operator()(const T &a, const T &b) const noexcept {
    // a < b || a == b, never !(b < a), which would call NaN <= NaN true where
    // the JIT kernel's `fcmp ole` says false.
    return T(static_cast<int>(OrEqual ? (a < b || a == b) : (a < b)));
  }
};
using lt_impl = compare_impl<false>;
using le_impl = compare_impl<true>;

// c != 0 ? t : f with both arms already computed, so each lane of a batch is a
// blend on the same instruction path.
struct select_impl {
  template <Numeric T>
    requires std::equality_comparable<T>
  constexpr T operator()(const T &c, const T &t, const T &f) const noexcept {
    return c != T{0} ? t : f;
  }
};

struct abs_impl {
  constexpr auto operator()(const Numeric auto &a) const noexcept {
    using std::abs;
    return abs(a);
  }
};
// The value of the sign rule that adjoints.hpp differentiates through.
struct sign_impl {
  template <Numeric T> constexpr auto operator()(const T &a) const noexcept {
    return sign(a);
  }
};
// An unqualified call after `using std::FN`, so ADL finds the overloads of
// Dual and TaylorDual and std:: handles the rest.
#define DDX_ADL_BINARY_IMPL(NAME, FN)                                          \
  struct NAME {                                                                \
    constexpr auto operator()(const Numeric auto &a,                           \
                              const Numeric auto &b) const noexcept {          \
      using std::FN;                                                           \
      return FN(a, b);                                                         \
    }                                                                          \
  };
DDX_ADL_BINARY_IMPL(pow_impl, pow)
DDX_ADL_BINARY_IMPL(atan2_impl, atan2)
DDX_ADL_BINARY_IMPL(hypot_impl, hypot)
// max and min average a tie through midpoint, which Dual and TaylorDual
// overload.
DDX_ADL_BINARY_IMPL(midpoint_impl, midpoint)
#undef DDX_ADL_BINARY_IMPL
// A tie averages and an unordered pair gives NaN, both symmetric under operand
// reordering. Signed zeros follow IEEE 754-2019 maximum/minimum, via -0 + +0.
template <bool IsMax> struct extremum_impl {
  constexpr auto operator()(const Numeric auto &a,
                            const Numeric auto &b) const noexcept {
    using R = std::remove_cvref_t<decltype(a < b ? b : a)>;
    if (a == b) {
      if constexpr (CArithmetic<R>) {
        if (a == R{}) {
          return IsMax ? R{a + b} : R{-((-a) + (-b))};
        }
        return R{a};
      } else {
        return R{midpoint_impl{}(R{a}, R{b})};
      }
    }
    if (a < b) {
      return IsMax ? R{b} : R{a};
    }
    if (b < a) {
      return IsMax ? R{a} : R{b};
    }
    return R{(a - b) * R{}};
  }
};
using max_impl = extremum_impl<true>;
using min_impl = extremum_impl<false>;
} // namespace detail

// One op per math table row, with value and derivative from its descriptor.
#define DDX_UNARY_MATH_OP(FN, NAME, LABEL)                                     \
  template <Numeric T>                                                         \
    requires(!detail::needs_real_constants_v<detail::NAME##Fn<T>> ||           \
             std::constructible_from<T, double>)                               \
  struct NAME : UnaryOp<T, detail::NAME##Fn<T>, FixedString{LABEL}> {          \
    [[nodiscard]] static constexpr auto                                        \
    derivative(const CExpression auto &lhs) noexcept {                         \
      return detail::NAME##Fn<T>::deriv(lhs) * lhs.derivative();               \
    }                                                                          \
  };

DDX_UNARY_MATH_TABLE(DDX_UNARY_MATH_OP)
#undef DDX_UNARY_MATH_OP

// Internal: keeps the derivative trees of abs, max and min finite at the kink,
// where u/|u| would be 0/0.
template <Numeric T>
  requires std::totally_ordered<T>
struct SignOp : UnaryOp<T, detail::sign_impl, FixedString{"sign"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &) noexcept {
    return Lit<T, 0>{};
  }
};

// abs is outside the math table because its derivative is a sign, not a
// function of the primal.
template <Numeric T>
  requires std::totally_ordered<T>
struct AbsOp : UnaryOp<T, detail::abs_impl, FixedString{"abs"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept {
    auto s = MonoExpression<SignOp<T>, std::decay_t<decltype(lhs)>>{lhs};
    return std::move(s) * lhs.derivative();
  }
};

// The comparison and select that guard the partials of pow. A comparison is
// piecewise constant, so its derivative is zero.
template <Numeric T>
struct LeOp
    : BinaryOp<T, detail::le_impl, FixedString{"<="}, Notation::Infix, 5> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &, const CExpression auto &) noexcept {
    return Lit<T, 0>{};
  }
};
template <Numeric T>
struct SelectOp : OpDescriptor<T, detail::select_impl, FixedString{"select"},
                               Notation::Function, precedence_atom> {
  [[nodiscard]] static constexpr auto eval(const Numeric auto &c,
                                           const Numeric auto &a,
                                           const Numeric auto &b) noexcept {
    return detail::select_impl{}(c, a, b);
  }
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &c, const CExpression auto &a,
             const CExpression auto &b) noexcept {
    return selected(c, a.derivative(), b.derivative());
  }
};

template <Numeric T>
struct PowOp : BinaryOp<T, detail::pow_impl, FixedString{"pow"}> {
  static constexpr std::optional rule_op = algebra::RuleOp::Pow;
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    // d(a^b) = b'·ln(a)·a^b + b·a^(b-1)·a', split so a == 0 never gives 0/0.
    // Each term is an exact zero when its own factor vanishes.
    return unless_vanishing(lhs, pow(lhs, rhs),
                            rhs.derivative() * log(lhs) * pow(lhs, rhs)) +
           unless_vanishing(
               lhs, rhs, rhs * pow(lhs, rhs - Lit<T, 1>{}) * lhs.derivative());
  }
};

// atan2(y, x) with lhs y and rhs x: d = (x y' - y x') / (x² + y²).
template <Numeric T>
struct Atan2Op : BinaryOp<T, detail::atan2_impl, FixedString{"atan2"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    // Divides twice by hypot, because x² + y² overflows past 1e154.
    return ((rhs * lhs.derivative() - lhs * rhs.derivative()) /
            hypot(lhs, rhs)) /
           hypot(lhs, rhs);
  }
};

// hypot(x, y) = sqrt(x² + y²), so d = (x x' + y y') / hypot.
template <Numeric T>
struct HypotOp : BinaryOp<T, detail::hypot_impl, FixedString{"hypot"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    return (lhs * lhs.derivative() + rhs * rhs.derivative()) / hypot(lhs, rhs);
  }
};

// With s = sign(a - b), max' = (a' + b' + s·(a' - b')) / 2, and min' subtracts.
// Branch-free because the two arms are different types; a tie gives the mean.
template <Numeric T, bool IsMax, typename func, CFixedString auto symbol>
  requires std::totally_ordered<T>
struct ExtremumOp : BinaryOp<T, func, symbol> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    const auto d = lhs - rhs;
    const auto s = MonoExpression<SignOp<T>, std::decay_t<decltype(d)>>{d};
    const auto sum = lhs.derivative() + rhs.derivative();
    const auto kink = s * (lhs.derivative() - rhs.derivative());
    if constexpr (IsMax) {
      return (sum + kink) / Lit<T, 2>{};
    } else {
      return (sum - kink) / Lit<T, 2>{};
    }
  }
};

template <Numeric T>
  requires std::totally_ordered<T>
struct MaxOp : ExtremumOp<T, true, detail::max_impl, FixedString{"max"}> {};

template <Numeric T>
  requires std::totally_ordered<T>
struct MinOp : ExtremumOp<T, false, detail::min_impl, FixedString{"min"}> {};

} // namespace ddx::impl
