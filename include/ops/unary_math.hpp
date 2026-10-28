#pragma once

#include "ops/numeric.hpp" // Numeric

#include <cmath>
#include <numbers>

namespace ddx::impl::detail {

#define DDX_UNARY_MATH_FNS                                                     \
  using std::sin, std::cos, std::tan, std::exp, std::log, std::log10,          \
      std::sqrt, std::cbrt, std::asin, std::acos, std::atan, std::sinh,        \
      std::cosh, std::tanh, std::asinh, std::acosh, std::atanh, std::erf
#define DDX_UNARY_MATH_DESC_(NAME, VAL, DERIV, EXTRA)                          \
  template <Numeric T> struct NAME {                                           \
    constexpr auto operator()(const auto &u) const noexcept {                  \
      DDX_UNARY_MATH_FNS;                                                      \
      return (VAL);                                                            \
    }                                                                          \
    static constexpr auto deriv(const auto &u) noexcept {                      \
      DDX_UNARY_MATH_FNS;                                                      \
      return (DERIV);                                                          \
    }                                                                          \
    EXTRA                                                                      \
  };
// MSVC's default preprocessor has no __VA_OPT__,
#define DDX_UNARY_MATH_DESC(NAME, VAL, DERIV)                                  \
  DDX_UNARY_MATH_DESC_(NAME, VAL, DERIV, )
#define DDX_UNARY_MATH_DESC_FV(NAME, VAL, DERIV, FV)                           \
  DDX_UNARY_MATH_DESC_(                                                        \
      NAME, VAL, DERIV,                                                        \
      static constexpr auto deriv_from_value(const auto &,                     \
                                             const auto &fu) noexcept {        \
        DDX_UNARY_MATH_FNS;                                                    \
        return (FV);                                                           \
      })

// clang-format off
DDX_UNARY_MATH_DESC(SineOpFn,    sin(u), cos(u))
DDX_UNARY_MATH_DESC(CosineOpFn,  cos(u), -sin(u))
DDX_UNARY_MATH_DESC_FV(ExpOpFn,  exp(u), exp(u), fu)
// sec^2 = 1 + tan^2 — reassociates, so not bit-identical to 1/cos^2.
DDX_UNARY_MATH_DESC_FV(TanOpFn,  tan(u),  T{1} / (cos(u) * cos(u)), T{1} + fu * fu)
DDX_UNARY_MATH_DESC(LogOpFn,     log(u),  T{1} / u)
DDX_UNARY_MATH_DESC_FV(SqrtOpFn, sqrt(u), T{1} / (T{2} * sqrt(u)), T{1} / (T{2} * fu))
// (1-u)(1+u), not 1-u*u: near |u| = 1 the subtraction cancels against a
// rounded square and costs half the significand.  Likewise acosh and atanh.
DDX_UNARY_MATH_DESC(AsinOpFn,    asin(u), T{1} / sqrt((T{1} - u) * (T{1} + u)))
DDX_UNARY_MATH_DESC(AcosOpFn,    acos(u), T{-1} / sqrt((T{1} - u) * (T{1} + u)))
DDX_UNARY_MATH_DESC(AtanOpFn,    atan(u), T{1} / (T{1} + u * u))
DDX_UNARY_MATH_DESC(SinhOpFn,    sinh(u), cosh(u))
DDX_UNARY_MATH_DESC(CoshOpFn,    cosh(u), sinh(u))
// sech^2 = 1 - tanh^2 — reassociates, so not bit-identical to 1/cosh^2.
DDX_UNARY_MATH_DESC_FV(TanhOpFn, tanh(u),  T{1} / (cosh(u) * cosh(u)), T{1} - fu * fu)
DDX_UNARY_MATH_DESC(Log10OpFn,   log10(u), T{1} / (u * static_cast<T>(std::numbers::ln10)))
DDX_UNARY_MATH_DESC_FV(CbrtOpFn, cbrt(u),  T{1} / (T{3} * cbrt(u) * cbrt(u)), T{1} / (T{3} * fu * fu))
DDX_UNARY_MATH_DESC(AsinhOpFn,   asinh(u), T{1} / sqrt(u * u + T{1}))
DDX_UNARY_MATH_DESC(AcoshOpFn,   acosh(u), T{1} / (sqrt(u - T{1}) * sqrt(u + T{1})))
DDX_UNARY_MATH_DESC(AtanhOpFn,   atanh(u), T{1} / ((T{1} - u) * (T{1} + u)))
DDX_UNARY_MATH_DESC(ErfOpFn,     erf(u),   static_cast<T>(2.0 * std::numbers::inv_sqrtpi) * exp(-(u * u)))
#undef DDX_UNARY_MATH_DESC_FV
#undef DDX_UNARY_MATH_DESC
#undef DDX_UNARY_MATH_DESC_
#undef DDX_UNARY_MATH_FNS
// clang-format on

// A descriptor's `deriv` needs strictly more of T than its value does (sin
// needs cos, asin needs sqrt), and that is not expressible as a constraint:
// `deriv` returns auto, so a failure hard-errors rather than SFINAE-ing.  log10
// and erf are the exception -- their irrational constants need
// static_cast<T>(double), which constructible_from<T, int> does not give, and
// that IS a constraint.
template <typename Fn> inline constexpr bool needs_real_constants_v = false;
template <Numeric T>
inline constexpr bool needs_real_constants_v<Log10OpFn<T>> = true;
template <Numeric T>
inline constexpr bool needs_real_constants_v<ErfOpFn<T>> = true;

template <typename Fn, Numeric T>
inline constexpr bool has_deriv_from_value_v =
    requires(const T &u, const T &fu) { Fn::deriv_from_value(u, fu); };

} // namespace ddx::impl::detail

// The two-argument functions every front end spells, as rt's binary rows:
// (factory, enumerator, label, value functor, ops/adjoints.hpp descriptor).
#define DDX_BINARY_MATH_TABLE(X)                                               \
  X(pow, Pow, "pow", impl::detail::pow_impl, PowOpFn)                          \
  X(atan2, Atan2, "atan2", impl::detail::atan2_impl, Atan2OpFn)                \
  X(hypot, Hypot, "hypot", impl::detail::hypot_impl, HypotOpFn)                \
  X(max, Max, "max", impl::detail::max_impl, MaxOpFn)                          \
  X(min, Min, "min", impl::detail::min_impl, MinOpFn)

// `abs` is absent: its derivative is a sign, not a function of the primal.
#define DDX_UNARY_MATH_TABLE(X)                                                \
  X(sin, SineOp, "sin")                                                        \
  X(cos, CosineOp, "cos")                                                      \
  X(exp, ExpOp, "exp")                                                         \
  X(tan, TanOp, "tan")                                                         \
  X(log, LogOp, "log")                                                         \
  X(log10, Log10Op, "log10")                                                   \
  X(sqrt, SqrtOp, "sqrt")                                                      \
  X(cbrt, CbrtOp, "cbrt")                                                      \
  X(asin, AsinOp, "asin")                                                      \
  X(acos, AcosOp, "acos")                                                      \
  X(atan, AtanOp, "atan")                                                      \
  X(sinh, SinhOp, "sinh")                                                      \
  X(cosh, CoshOp, "cosh")                                                      \
  X(tanh, TanhOp, "tanh")                                                      \
  X(asinh, AsinhOp, "asinh")                                                   \
  X(acosh, AcoshOp, "acosh")                                                   \
  X(atanh, AtanhOp, "atanh")                                                   \
  X(erf, ErfOp, "erf")
