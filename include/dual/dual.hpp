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

// (a+be)(c+de) = ac + (ad+bc)e
template <Numeric T> class Dual;
template <Numeric T>
inline constexpr bool is_commutative_multiply_v<Dual<T>> =
    is_commutative_multiply_v<T>;

// A zero-derivative operand for the dual A.
template <typename C, typename A>
concept ConstOperand =
    CArithmetic<C> || std::same_as<std::remove_cvref_t<C>, dual_value_t<A>>;

// Ref: Clifford, Proc. LMS s1-4 (1873) 381 -- adjoin ε with ε² = 0.
template <Numeric T> class Dual : public compound_from_binary<Dual<T>> {
private:
  T val_{};
  T deriv_{};

public:
  constexpr Dual() noexcept = default;
  constexpr explicit Dual(T v, T d = T{}) noexcept : val_{v}, deriv_{d} {}
  // deriv_ is left to its NSDMI: naming T{} here is one of the spellings
  // MSVC's front end cannot lower once T is itself a Dual (see dual_div).
  constexpr Dual(CArithmetic auto s) noexcept : val_{T(s)} {}

  constexpr Dual &operator++() noexcept {
    ++val_;
    return *this;
  }

private:
  template <std::size_t Index>
  [[nodiscard]] static constexpr decltype(auto) slot(auto &&self) noexcept {
    static_assert(Index < 2, "Dual index out of bounds");
    if constexpr (Index == 0) {
      return (DDX_FWD(self).val_);
    } else {
      return (DDX_FWD(self).deriv_);
    }
  }

public:
  DDX_SLOT_ACCESSOR(value, 0)
  DDX_SLOT_ACCESSOR(deriv, 1)

  // get() on an rvalue yields an rvalue, as std::get does.
  DDX_KEYED_GET(std::size_t Index, Index)
};

// The primary lives in util/fmt.hpp, next to the formatter that reads it.
namespace detail {
template <Numeric T> inline constexpr bool is_dual_family_v<Dual<T>> = true;
} // namespace detail

// The Dual ends of the recursions ops/scalar.hpp starts for a plain scalar.
template <Numeric T> constexpr auto val(const Dual<T> &d) noexcept {
  const auto &[real, _] = d;
  return val(real);
}
template <Numeric T> constexpr bool all_zero(const Dual<T> &d) noexcept {
  const auto &[real, dual] = d;
  return all_zero(real) && all_zero(dual);
}

// dual_var_of<"x">(v) — here rather than beside var_of() because it needs Dual
// complete, not declared.
template <FixedString S, Numeric T>
[[nodiscard]] constexpr auto dual_var_of(const T &) noexcept {
  return Variable<Dual<T>, S>{};
}

template <FixedString S, Numeric T>
[[nodiscard]] constexpr auto dual_var_of(symbol_type<S>, const T &) noexcept {
  return Variable<Dual<T>, S>{};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_add(const Dual<T> &a,
                                             const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av + bv, ad + bd};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T>
dual_add(const Dual<T> &a, const ConstOperand<Dual<T>> auto &s) noexcept {
  const auto &[av, ad] = a;
  return Dual<T>{av + s, ad};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_sub(const Dual<T> &a,
                                             const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av - bv, ad - bd};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T>
dual_sub(const Dual<T> &a, const ConstOperand<Dual<T>> auto &s) noexcept {
  const auto &[av, ad] = a;
  return Dual<T>{av - s, ad};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T>
dual_sub(const ConstOperand<Dual<T>> auto &s, const Dual<T> &a) noexcept {
  const auto &[av, ad] = a; // s - a == -(a - s);
  return Dual<T>{-(av - s), -ad};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_mul(const Dual<T> &a,
                                             const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av * bv, ad * bv + av * bd};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T>
dual_mul(const Dual<T> &a, const ConstOperand<Dual<T>> auto &s) noexcept {
  const auto &[av, ad] = a; // scalar distributes; no zero-derivative term
  return Dual<T>{av * s, ad * s};
}

// Reciprocal form: one hardware division per nesting level instead of two.
// V, not T, throughout the bodies below: MSVC's front end ICEs in lower-trees
// on a local whose type is spelled as the bare template parameter once that
// parameter is itself a Dual, and the alias is what it will lower.
template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_div(const Dual<T> &a,
                                             const Dual<T> &b) noexcept {
  using V = T;
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  const V inv = V{1} / bv;
  const V q = av * inv; // value = a / b
  return Dual{q, (ad - q * bd) * inv};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T>
dual_div(const Dual<T> &a, const ConstOperand<Dual<T>> auto &s) noexcept {
  using V = T;
  const auto &[av, ad] = a; // s is a zero-derivative constant
  const V inv = V{1} / V(s);
  return Dual{av * inv, ad * inv};
}

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T>
dual_div(const ConstOperand<Dual<T>> auto &s, const Dual<T> &a) noexcept {
  using V = T;
  const auto &[av, ad] = a; // s / a; inner kept T-on-left (wide-scalar-safe)
  const V inv = V{1} / av;
  const V q = V(s) * inv; // value = s / a
  return Dual{q, -(q * ad) * inv};
}

// All three shapes of each operator; LEFT spells (scalar, Dual), the only one
// that differs between operators.  Separate kernels rather than a promotion,
// which would leave an `ad + 0` IEEE will not let the compiler fold.
#define DDX_DUAL_BINOP(OP, COMB, LEFT)                                         \
  template <DualLike A>                                                        \
  constexpr auto operator OP(A &&a, DualCompatible<A> auto &&b) noexcept {     \
    return COMB(a, b);                                                         \
  }                                                                            \
  template <DualLike A>                                                        \
  constexpr auto operator OP(A &&a, ConstOperand<A> auto &&s) noexcept {       \
    return COMB(a, s);                                                         \
  }                                                                            \
  template <DualLike A>                                                        \
  constexpr auto operator OP(ConstOperand<A> auto &&s, A &&a) noexcept {       \
    return LEFT;                                                               \
  }
DDX_DUAL_BINOP(+, dual_add, dual_add(a, s))
DDX_DUAL_BINOP(-, dual_sub, dual_sub(s, a))
DDX_DUAL_BINOP(*, dual_mul, dual_mul(a, s))
DDX_DUAL_BINOP(/, dual_div, dual_div(s, a))
#undef DDX_DUAL_BINOP

constexpr auto operator-(DualLike auto &&a) noexcept {
  const auto &[v, d] = a;
  using DT = std::remove_cvref_t<decltype(a)>;
  return DT{-v, -d};
}

// The primal is reused when the descriptor can express its derivative in terms
// of f(u).
template <template <Numeric> class Fn> struct unary_dual_combine {
  DDX_ALWAYS_INLINE constexpr auto
  operator()(const DualLike auto &x) const noexcept {
    const auto &[v, d] = x;
    // Comp must stay a Dual at depth >= 2 or fv truncates.  The descriptor is
    // instantiated at the base scalar S: at Comp its constants would become
    // duals and leave a `0.0 - x` behind.
    using Comp = std::remove_cvref_t<decltype(v)>;
    using DT = std::remove_cvref_t<decltype(x)>;
    using S = scalar_base_t<Comp>;
    if constexpr (detail::has_deriv_from_value_v<Fn<S>, Comp>) {
      const Comp fv = Fn<S>{}(v);
      return DT{fv, Fn<S>::deriv_from_value(v, fv) * d};
    } else {
      return DT{Fn<S>{}(v), Fn<S>::deriv(v) * d};
    }
  }
};
struct abs_combine {
  constexpr auto operator()(const DualLike auto &x) const noexcept {
    using std::abs;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    return DT{abs(v), detail::sign(v) * d};
  }
};
// One chain-rule overload per unary math function, from the registry.
#define DDX_DUAL_UNARY(FN, OP, LABEL)                                          \
  constexpr auto FN(DualLike auto &&a) noexcept {                              \
    return unary_dual_combine<detail::OP##Fn>{}(a);                            \
  }
DDX_UNARY_MATH_TABLE(DDX_DUAL_UNARY)
#undef DDX_DUAL_UNARY

constexpr auto abs(DualLike auto &&a) noexcept { return abs_combine{}(a); }

// ---- comparisons (operate on materialized values) -------------------------
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

// The binary chain rule off the shared descriptor, f' = da a' + db b', with
// the primal from the value functor.  A term whose operand has a zero
// derivative -- a constant, or a constant exponent handed over embedded in a
// dual -- is left out rather than multiplied by zero: at a <= 0 pow's exponent
// partial is 0 * log(a), which is NaN, not nothing.  The descriptor is
// instantiated at the component, so a nested dual's rule recurses.
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
    const auto [dx, dy] = detail::adjoints_of<Desc, Comp>(Comp{1}, xv, yv);
    if (all_zero(xd)) {
      return DT{f, dy * yd};
    }
    if (all_zero(yd)) {
      return DT{f, dx * xd};
    }
    return DT{f, dx * xd + dy * yd};
  }
};

// Each in three shapes: all-dual, and a constant on either side, which
// arrives embedded with a zero derivative.  The inner calls are unqualified,
// so ADL makes these work at any nesting depth.
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
DDX_DUAL_BINARY(pow, detail::PowOpFn, detail::pow_impl)
DDX_DUAL_BINARY(atan2, detail::Atan2OpFn, detail::atan2_impl)
DDX_DUAL_BINARY(hypot, detail::HypotOpFn, detail::hypot_impl)
DDX_DUAL_BINARY(max, detail::MaxOpFn, detail::max_impl)
DDX_DUAL_BINARY(min, detail::MinOpFn, detail::min_impl)
#undef DDX_DUAL_BINARY

// Linear, so it is its own derivative rule.  std::midpoint takes arithmetic
// types only, which is why max/min go through midpoint_impl instead.
template <DualLike A>
constexpr auto midpoint(A &&a, DualCompatible<A> auto &&b) noexcept {
  using std::midpoint;
  const auto &[xv, xd] = a;
  const auto &[yv, yd] = b;
  using DT = std::remove_cvref_t<A>;
  return DT{midpoint(xv, yv), midpoint(xd, yd)};
}

// 3-argument hypot, all-dual only.
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
  return Dual<T>{h, (xv / h) * xd + (yv / h) * yd + (zv / h) * zd};
}

static_assert(Numeric<Dual<double>>);
static_assert(Numeric<Dual<float>>);

using dual = nth_dual_t<double, 1>;    // first-order forward dual
using dual2nd = nth_dual_t<double, 2>; // second-order (Hessian-capable) dual

} // namespace ddx::impl

// `v+de` -- the two-term case of the shared series renderer.
template <ddx::impl::Numeric T>
struct std::formatter<ddx::impl::Dual<T>, char>
    : ddx::impl::detail::dual_formatter_base<T> {
  auto format(const ddx::impl::Dual<T> &d, std::format_context &ctx) const {
    this->series(ctx, std::array{d.value(), d.deriv()});
    return ctx.out();
  }
};
