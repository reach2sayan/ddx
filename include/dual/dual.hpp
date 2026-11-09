#pragma once

#include "ops/adjoints.hpp"
#include "ops/operations.hpp" // the value functors
#include "ops/scalar.hpp"
#include "ops/unary_math.hpp"
#include "symbolic/expressions.hpp"
#include "util/config.hpp"
#include "util/fmt.hpp"
#include <array>
#include <cmath>
#include <format>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ddx::impl {

// (a+bε)(c+dε) = ac + (ad+bc)ε commutes exactly when the scalar does.
template <Numeric T> class Dual;
template <Numeric T>
inline constexpr bool is_commutative_multiply_v<Dual<T>> =
    is_commutative_multiply_v<T>;

// A zero-derivative operand for a dual whose components are V.
template <typename C, typename V>
concept ConstOperand =
    CArithmetic<C> || std::same_as<std::remove_cvref_t<C>, V>;

// Ref: Clifford, Proc. LMS s1-4 (1873) 381 -- adjoin ε with ε² = 0.
template <Numeric T> class Dual : public compound_from_binary<Dual<T>> {
private:
  T val_{};
  T deriv_{};

public:
  constexpr Dual() noexcept = default;
  constexpr explicit Dual(T v, T d = T{}) noexcept : val_{v}, deriv_{d} {}
  // deriv_ keeps its default initializer, because MSVC cannot lower T{} here
  // once T is itself a Dual (see dual_div).
  constexpr Dual(CArithmetic auto s) noexcept : val_{T(s)} {}

  constexpr Dual &operator++() noexcept {
    ++val_;
    return *this;
  }

  // get() on an rvalue returns an rvalue, as std::get does.
  template <std::size_t Index>
  [[nodiscard]] constexpr decltype(auto) get(this auto &&self) noexcept {
    static_assert(Index < 2, "Dual index out of bounds");
    if constexpr (Index == 0) {
      return (DDX_FWD(self).val_);
    } else {
      return (DDX_FWD(self).deriv_);
    }
  }
  [[nodiscard]] constexpr decltype(auto) value(this auto &&self) noexcept {
    return DDX_FWD(self).template get<0>();
  }
  [[nodiscard]] constexpr decltype(auto) deriv(this auto &&self) noexcept {
    return DDX_FWD(self).template get<1>();
  }
};

// The primary template is in util/fmt.hpp, beside the formatter that reads it.
namespace detail {
template <Numeric T> inline constexpr bool is_dual_family_v<Dual<T>> = true;
} // namespace detail

// The Dual cases of the recursions that ops/scalar.hpp defines for scalars.
template <Numeric T> constexpr auto val(const Dual<T> &d) noexcept {
  const auto &[real, _] = d;
  return val(real);
}
template <Numeric T> constexpr bool all_zero(const Dual<T> &d) noexcept {
  const auto &[real, dual] = d;
  return all_zero(real) && all_zero(dual);
}
template <Numeric T>
constexpr bool same_value(const Dual<T> &a, const Dual<T> &b) noexcept {
  const auto &[a_real, a_dual] = a;
  const auto &[b_real, b_dual] = b;
  return same_value(a_real, b_real) && same_value(a_dual, b_dual);
}

// dual_var_of<"x">(v) lives here rather than beside var_of() because it needs
// Dual to be complete.
template <FixedString S, Numeric T>
[[nodiscard]] constexpr auto dual_var_of(const T &) noexcept {
  return Variable<Dual<T>, S>{};
}

template <FixedString S, Numeric T>
[[nodiscard]] constexpr auto dual_var_of(symbol_type<S>, const T &) noexcept {
  return Variable<Dual<T>, S>{};
}

// One division per nesting level, by multiplying with the reciprocal. Locals
// use the alias V because MSVC ICEs on bare T when T is itself a Dual.
template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_div(const Dual<T> &a,
                                             const Dual<T> &b) noexcept {
  using V = T;
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  const V inv = V{1} / bv;
  const V q = av * inv;
  return Dual{q, (ad - q * bd) * inv};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T>
dual_div(const Dual<T> &a, const ConstOperand<T> auto &s) noexcept {
  using V = T;
  const auto &[av, ad] = a;
  const V inv = V{1} / V(s);
  return Dual{av * inv, ad * inv};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_div(const ConstOperand<T> auto &s,
                                             const Dual<T> &a) noexcept {
  using V = T;
  const auto &[av, ad] = a;
  const V inv = V{1} / av;
  const V q = V(s) * inv;
  return Dual{q, -(q * ad) * inv};
}

// Each operator for (Dual, Dual), (Dual, scalar) and (scalar, Dual). A scalar
// is not promoted, which would leave an `ad + 0` that IEEE forbids folding.
#define DDX_DUAL_BINOP(OP, BOTH, RIGHT, LEFT)                                  \
  template <DualLike A>                                                        \
  DDX_ALWAYS_INLINE constexpr auto operator OP(                                \
      A &&a, DualCompatible<A> auto &&b) noexcept {                            \
    using T [[maybe_unused]] = dual_value_t<A>;                                \
    [[maybe_unused]] const auto &[av, ad] = a;                                 \
    [[maybe_unused]] const auto &[bv, bd] = b;                                 \
    return BOTH;                                                               \
  }                                                                            \
  template <DualLike A>                                                        \
  DDX_ALWAYS_INLINE constexpr auto operator OP(                                \
      A &&a, ConstOperand<dual_value_t<A>> auto &&s) noexcept {                \
    using T [[maybe_unused]] = dual_value_t<A>;                                \
    [[maybe_unused]] const auto &[av, ad] = a;                                 \
    return RIGHT;                                                              \
  }                                                                            \
  template <DualLike A>                                                        \
  DDX_ALWAYS_INLINE constexpr auto operator OP(                                \
      ConstOperand<dual_value_t<A>> auto &&s, A &&a) noexcept {                \
    using T [[maybe_unused]] = dual_value_t<A>;                                \
    [[maybe_unused]] const auto &[av, ad] = a;                                 \
    return LEFT;                                                               \
  }
DDX_DUAL_BINOP(+, (Dual<T>{av + bv, ad + bd}), (Dual<T>{av + s, ad}),
               (Dual<T>{av + s, ad}))
DDX_DUAL_BINOP(-, (Dual<T>{av - bv, ad - bd}), (Dual<T>{av - s, ad}),
               (Dual<T>{-(av - s), -ad})) // s - a == -(a - s)
DDX_DUAL_BINOP(*, (Dual<T>{av * bv, ad * bv + av * bd}),
               (Dual<T>{av * s, ad * s}), (Dual<T>{av * s, ad * s}))
DDX_DUAL_BINOP(/, dual_div(a, b), dual_div(a, s), dual_div(s, a))
#undef DDX_DUAL_BINOP

constexpr auto operator-(DualLike auto &&a) noexcept {
  const auto &[v, d] = a;
  using DT = std::remove_cvref_t<decltype(a)>;
  return DT{-v, -d};
}

// The unary chain rule, reusing f(u) when the descriptor's derivative can be
// written in terms of it.
template <template <Numeric> class Fn> struct unary_dual_combine {
  DDX_ALWAYS_INLINE constexpr auto
  operator()(const DualLike auto &x) const noexcept {
    const auto &[v, d] = x;
    // The descriptor is instantiated at the base scalar S, because at Comp its
    // constants would become duals and leave a `0.0 - x` behind.
    using Comp = std::remove_cvref_t<decltype(v)>;
    using DT = std::remove_cvref_t<decltype(x)>;
    using S = scalar_base_t<Comp>;
    const Comp fv = Fn<S>{}(v);
    // A zero tangent skips the partial, so an infinite one such as sqrt's at 0
    // gives zero rather than NaN.
    if (all_zero(d)) {
      return DT{fv};
    }
    if constexpr (detail::has_deriv_from_value_v<Fn<S>, Comp>) {
      return DT{fv, Fn<S>::deriv_from_value(v, fv) * d};
    } else {
      return DT{fv, Fn<S>::deriv(v) * d};
    }
  }
};
// One chain-rule overload per unary math table row.
#define DDX_DUAL_UNARY(FN, OP, LABEL)                                          \
  constexpr auto FN(DualLike auto &&a) noexcept {                              \
    return unary_dual_combine<detail::OP##Fn>{}(a);                            \
  }
DDX_UNARY_MATH_TABLE(DDX_DUAL_UNARY)
#undef DDX_DUAL_UNARY

constexpr auto abs(DualLike auto &&x) noexcept {
  using std::abs;
  const auto &[v, d] = x;
  using DT = std::remove_cvref_t<decltype(x)>;
  return DT{abs(v), detail::sign(v) * d};
}

// Comparisons look at the value level only.
template <typename A, typename B>
concept DualComparable =
    DualOrArithmetic<A> && DualOrArithmetic<B> && (DualLike<A> || DualLike<B>);

template <DualOrArithmetic A>
constexpr auto operator<=>(const A &a,
                           const DualComparable<A> auto &b) noexcept {
  return val(a) <=> val(b);
}
template <DualOrArithmetic A>
constexpr bool operator==(const A &a,
                          const DualComparable<A> auto &b) noexcept {
  return val(a) == val(b);
}

// The binary chain rule f' = da·a' + db·b'. A term with a zero tangent is
// skipped, not multiplied by zero, since pow's 0·log(a) at a <= 0 is NaN.
template <template <Numeric> class Desc, typename Value>
struct binary_dual_combine {
  template <DualLike DT>
  DDX_ALWAYS_INLINE constexpr DT operator()(const DT &x,
                                            const DT &y) const noexcept {
    const auto &[xv, xd] = x;
    const auto &[yv, yd] = y;
    using Comp = std::remove_cvref_t<decltype(xv)>;
    const Comp f = Value{}(xv, yv);
    if (all_zero(xd) && all_zero(yd)) {
      return DT{f};
    }
    if (all_zero(xd)) {
      return DT{f, detail::adjoint_of<Desc, 1, Comp>(Comp{1}, xv, yv) * yd};
    }
    if (all_zero(yd)) {
      return DT{f, detail::adjoint_of<Desc, 0, Comp>(Comp{1}, xv, yv) * xd};
    }
    const auto [dx, dy] = detail::adjoints_of<Desc, Comp>(Comp{1}, xv, yv);
    return DT{f, dx * xd + dy * yd};
  }
};

// Each binary function for two duals or a constant on either side. Unqualified
// inner calls let ADL handle any nesting depth.
#define DDX_DUAL_BINARY(FN, Desc, Value)                                       \
  template <DualLike A>                                                        \
  constexpr auto FN(A &&a, DualCompatible<A> auto &&b) noexcept {              \
    using DT = std::remove_cvref_t<A>;                                         \
    return binary_dual_combine<Desc, Value>{}(DT{a}, DT{b});                   \
  }                                                                            \
  template <DualLike A>                                                        \
  constexpr auto FN(A &&a, CArithmetic auto s) noexcept {                      \
    using DT = std::remove_cvref_t<A>;                                         \
    return binary_dual_combine<Desc, Value>{}(DT{a}, DT{s});                   \
  }                                                                            \
  template <DualLike A>                                                        \
  constexpr auto FN(CArithmetic auto s, A &&a) noexcept {                      \
    using DT = std::remove_cvref_t<A>;                                         \
    return binary_dual_combine<Desc, Value>{}(DT{s}, DT{a});                   \
  }
#define DDX_DUAL_BINROW(fn, Op, label, functor, Desc)                          \
  DDX_DUAL_BINARY(fn, detail::Desc, functor)
DDX_BINARY_MATH_TABLE(DDX_DUAL_BINROW)
#undef DDX_DUAL_BINROW
#undef DDX_DUAL_BINARY

// midpoint is linear, so it applies to value and tangent alike.
template <DualLike A>
constexpr auto midpoint(A &&a, DualCompatible<A> auto &&b) noexcept {
  using std::midpoint;
  const auto &[xv, xd] = a;
  const auto &[yv, yd] = b;
  using DT = std::remove_cvref_t<A>;
  return DT{midpoint(xv, yv), midpoint(xd, yd)};
}

// Three-argument hypot, for duals only.
template <DualLike A>
constexpr auto hypot(A &&a, DualCompatible<A> auto &&b,
                     DualCompatible<A> auto &&c) noexcept {
  using std::hypot;
  using T = dual_value_t<A>;
  const Dual<T> x = a, y = b, z = c;
  const auto &[xv, xd] = x;
  const auto &[yv, yd] = y;
  const auto &[zv, zd] = z;
  const T h = hypot(xv, yv, zv);
  return Dual{h, (xv / h) * xd + (yv / h) * yd + (zv / h) * zd};
}

static_assert(Numeric<Dual<double>>);
static_assert(Numeric<Dual<float>>);

using dual = nth_dual_t<double, 1>;
using dual2nd = nth_dual_t<double, 2>;

// An N-deep dual has 2^N coefficients, one per subset of its ε's; bit l of a
// mask is ε_l, with l = 0 outermost. dual_seed sets 1 on each ε in mask.
template <std::size_t N, Numeric T>
[[nodiscard]] constexpr nth_dual_t<T, N> dual_seed(T v,
                                                   std::size_t mask) noexcept {
  if constexpr (N == 0) {
    return v;
  } else {
    return nth_dual_t<T, N>{dual_seed<N - 1>(v, mask >> 1),
                            embed_constant<T, N - 1>((mask & 1) ? T{1} : T{})};
  }
}

// The coefficient of the ε's in Mask: 0 is the value and 2^N - 1 the mixed
// Nth derivative.
template <std::size_t Mask, Numeric D>
[[nodiscard]] constexpr auto dual_part(const D &x) noexcept {
  if constexpr (dual_depth_v<D> == 0) {
    static_assert(Mask == 0, "dual_part: the mask names a level D lacks");
    return x;
  } else {
    return dual_part<(Mask >> 1)>(x.template get<Mask & 1>());
  }
}

} // namespace ddx::impl

// Prints a Dual as `v+dε` through the shared series renderer.
template <ddx::impl::Numeric T>
struct std::formatter<ddx::impl::Dual<T>, char>
    : ddx::impl::detail::dual_formatter_base<T> {
  auto format(const ddx::impl::Dual<T> &d, std::format_context &ctx) const {
    this->series(ctx, std::array{d.value(), d.deriv()});
    return ctx.out();
  }
};
