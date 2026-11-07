#pragma once
// The includes, model expressions and helpers shared by tests_simplify.cpp and
// dual/tests_simplify_dual.cpp.

#include "tests_common.hpp"

// Folding runs as trees are built and every sweep visits every node, so these
// tests assert on node_count_v at compile time.

namespace {
using ddx::impl::detail::canonicalise;
constexpr auto sx = ddx::var<"x">;
constexpr auto sy = ddx::var<"y">;
constexpr auto sz = ddx::var<"z">;
constexpr auto sw = ddx::var<"w">;

template <ddx::impl::FixedString S, class E> consteval std::size_t d_nodes() {
  return ddx::impl::node_count_v<
      decltype(ddx::impl::make_all_constant_except<S>(E{}).derivative())>;
}
constexpr auto F4 =
    (sx + sy) * (sz - sw) + exp(sx * sz) + sin(sy * sw) + sx * sy * sz * sw;
} // namespace

// Two scalars that differ only in whether their multiplication declares itself
// commutative.
template <int Tag> struct Ring {
  double v{};
  friend constexpr Ring operator+(Ring a, Ring b) noexcept {
    return {a.v + b.v};
  }
  friend constexpr Ring operator-(Ring a, Ring b) noexcept {
    return {a.v - b.v};
  }
  friend constexpr Ring operator*(Ring a, Ring b) noexcept {
    return {a.v * b.v};
  }
  friend constexpr Ring operator/(Ring a, Ring b) noexcept {
    return {a.v / b.v};
  }
  constexpr Ring operator-() const noexcept { return {-v}; }
};
using Undeclared = Ring<0>;
using Declared = Ring<1>;
DDX_COMMUTATIVE_MULTIPLY(Ring<1>)

// A non-commutative ring, the only kind of scalar that can observe which side
// an adjoint multiplies on; scalars embed as s*I.
struct Mat2 {
  double a{}, b{}, c{}, d{};
  constexpr Mat2() = default;
  constexpr Mat2(double s) noexcept : a{s}, d{s} {}
  constexpr Mat2(double a_, double b_, double c_, double d_) noexcept
      : a{a_}, b{b_}, c{c_}, d{d_} {}
  friend constexpr Mat2 operator+(Mat2 x, Mat2 y) noexcept {
    return {x.a + y.a, x.b + y.b, x.c + y.c, x.d + y.d};
  }
  friend constexpr Mat2 operator-(Mat2 x, Mat2 y) noexcept {
    return {x.a - y.a, x.b - y.b, x.c - y.c, x.d - y.d};
  }
  friend constexpr Mat2 operator*(Mat2 x, Mat2 y) noexcept {
    return {x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d, x.c * y.a + x.d * y.c,
            x.c * y.b + x.d * y.d};
  }
  friend constexpr Mat2 operator/(Mat2 x, Mat2 y) noexcept {
    const double det = y.a * y.d - y.b * y.c;
    return x * Mat2{y.d / det, -y.b / det, -y.c / det, y.a / det};
  }
  constexpr Mat2 operator-() const noexcept { return {-a, -b, -c, -d}; }
  // Mat2 has no operator+=, which pins that the reverse sweep accumulates with
  // + and assignment and never demands one.
  friend constexpr bool operator==(Mat2, Mat2) noexcept = default;
};

// Checks whether an op accepts a scalar without instantiating it, because a
// constrained specialisation named directly is a hard error.
template <template <class> class Op, class T>
concept OpAccepts = requires { typename Op<T>; };

// A type closed under the five CFieldLike operators but with no way to spell 1.
class NoIdentity {
  [[maybe_unused]] double v{};

public:
  constexpr NoIdentity() = default;
  friend constexpr NoIdentity operator+(NoIdentity, NoIdentity) noexcept {
    return {};
  }
  friend constexpr NoIdentity operator-(NoIdentity, NoIdentity) noexcept {
    return {};
  }
  friend constexpr NoIdentity operator*(NoIdentity, NoIdentity) noexcept {
    return {};
  }
  friend constexpr NoIdentity operator/(NoIdentity, NoIdentity) noexcept {
    return {};
  }
  constexpr NoIdentity operator-() const noexcept { return {}; }
};
