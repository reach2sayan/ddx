#pragma once

#include "dual/dual.hpp"
#include "ops/unary_math.hpp" // DDX_UNARY_MATH_TABLE
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numbers>
#include <numeric> // std::midpoint, std::transform_reduce
#include <ranges>
#include <utility>

namespace ddx::impl {

// Flat N+1 normalized coefficients c[k] = f^(k)(x)/k!.  Multiply is truncated
// convolution, O(N^2) against the O(2^N) of nested Dual.  Every transcendental
// below comes from the ODE the function satisfies, differentiated and
// coefficient-matched -- never Faa di Bruno, whose term count is the set
// partitions of k.  Ref: Jorba & Zou, Exp. Math. 14(1) (2005) 99 §2; Berz,
// Particle Accelerators 24 (1989) 109.

// The coefficient product is a convolution, sum_i a_i b_(k-i), so it commutes
// exactly when the scalar underneath does.
template <Numeric S, std::size_t N> struct TaylorDual;
template <Numeric S, std::size_t N>
inline constexpr bool is_commutative_multiply_v<TaylorDual<S, N>> =
    is_commutative_multiply_v<S>;

template <Numeric S, std::size_t N>
struct TaylorDual : compound_from_binary<TaylorDual<S, N>> {
  std::array<S, N + 1> c{};

  // 1/k, folded once rather than a ~13-cycle vdivsd on the dependency chain.
  static constexpr std::array<S, N + 1> inv_k = []() consteval {
    std::array<S, N + 1> r{};
    for (auto &&[k, rk] : r | std::views::enumerate | std::views::drop(1)) {
      rk = S{1} / static_cast<S>(k);
    }
    return r;
  }();

  constexpr TaylorDual() noexcept = default;
  constexpr explicit TaylorDual(S val) noexcept : c{} { c[0] = val; }

  // The independent variable at x0: unit first coefficient, so the sweep's
  // c[k] come out as f^(k)(x0)/k!.
  [[nodiscard]] static constexpr TaylorDual variable(S x0) noexcept
    requires(N >= 1)
  {
    TaylorDual r{x0};
    r.c[1] = S{1};
    return r;
  }

  constexpr TaylorDual operator+(const TaylorDual &o) const noexcept {
    TaylorDual r;
    std::ranges::transform(c, o.c, r.c.begin(), std::plus{});
    return r;
  }
  constexpr TaylorDual operator-(const TaylorDual &o) const noexcept {
    TaylorDual r;
    std::ranges::transform(c, o.c, r.c.begin(), std::minus{});
    return r;
  }
  constexpr TaylorDual operator-() const noexcept {
    TaylorDual r;
    std::ranges::transform(c, r.c.begin(), std::negate{});
    return r;
  }

  // Σ_{j=j0}^{j1-1} a[j]*b[k-j], the truncated convolution every product and
  // recurrence below is built from.
  [[nodiscard]] static constexpr S conv(const std::array<S, N + 1> &a,
                                        const std::array<S, N + 1> &b,
                                        std::size_t k, std::size_t j0,
                                        std::size_t j1) noexcept {
    const auto js = std::views::iota(j0, j1);
    return std::transform_reduce(
        js.begin(), js.end(), S{}, std::plus{},
        [&](std::size_t j) { return a[j] * b[k - j]; });
  }

  // (uv)[k] = Σ_{j=0}^{k} u[j]*v[k-j].  Ref: Knuth, TAOCP Vol. 2 §4.7; the
  // sub-quadratic forms only pay off far above this N.
  constexpr TaylorDual operator*(const TaylorDual &o) const noexcept {
    TaylorDual r;
    for (std::size_t k = 0; k <= N; ++k) {
      r.c[k] = conv(c, o.c, k, 0, k + 1);
    }
    return r;
  }

  // Polynomial division via r*o = this -- forward substitution, Knuth §4.7:
  //   r[k] = (c[k] - Σ_{j=0}^{k-1} r[j]*o[k-j]) / o[0]
  constexpr TaylorDual operator/(const TaylorDual &o) const noexcept {
    TaylorDual r;
    const S inv0 = S{1} / o.c[0];
    // The k loop is a recurrence -- r.c[j] below was written by an earlier
    // pass -- so only the inner convolution, whose j are all settled, folds.
    for (std::size_t k = 0; k <= N; ++k) {
      r.c[k] = (c[k] - conv(r.c, o.c, k, 0, k)) * inv0;
    }
    return r;
  }

  // Every coefficient times s, s on the side it was written.
  template <bool Left>
  static constexpr void scale(TaylorDual &u, const auto &s) noexcept {
    std::ranges::transform(u.c, u.c.begin(), [&s](const S &x) noexcept {
      if constexpr (Left) {
        return s * x;
      } else {
        return x * s;
      }
    });
  }

  // A constant moves c[0] alone or scales every coefficient: O(N), and no
  // `c[k] + 0` left that IEEE will not fold -- Dual's reasoning.  RIGHT and
  // LEFT update u in place for `u OP s` and `s OP u`; s / u stays the
  // recurrence, the reciprocal being a whole series.
#define DDX_TAYLOR_CONST_BINOP(OP, RIGHT, LEFT)                                \
  [[nodiscard]] friend constexpr TaylorDual operator OP(                       \
      TaylorDual u, const ConstOperand<S> auto &s) noexcept {                  \
    RIGHT;                                                                     \
    return u;                                                                  \
  }                                                                            \
  [[nodiscard]] friend constexpr TaylorDual operator OP(                       \
      const ConstOperand<S> auto &s, TaylorDual u) noexcept {                  \
    LEFT;                                                                      \
    return u;                                                                  \
  }
  DDX_TAYLOR_CONST_BINOP(+, u.c[0] += s, u = u + s)
  DDX_TAYLOR_CONST_BINOP(-, u.c[0] -= s, u = -(u - s))
  DDX_TAYLOR_CONST_BINOP(*, scale<false>(u, s), scale<true>(u, s))
  DDX_TAYLOR_CONST_BINOP(/, u = u * (S{1} / S(s)), u = TaylorDual{S(s)} / u)
#undef DDX_TAYLOR_CONST_BINOP

  constexpr TaylorDual &operator++() noexcept {
    ++c[0];
    return *this;
  }

  // On the value coefficient, as Dual's comparisons use val().  The branching
  // kernels in operations.hpp need it.
  [[nodiscard]] friend constexpr auto
  operator<=>(const TaylorDual &a, const TaylorDual &b) noexcept {
    return a.c[0] <=> b.c[0];
  }
  [[nodiscard]] friend constexpr bool operator==(const TaylorDual &a,
                                                 const TaylorDual &b) noexcept {
    return a.c[0] == b.c[0];
  }

  // Coefficient k-1 of a·b': Σ_{j=j0}^{k-1} a[j]·(k-j)·b[k-j].  j0 = 1 drops
  // the a[0] term a recurrence has already moved to the left.
  [[nodiscard]] static constexpr S dconv(const std::array<S, N + 1> &a,
                                         const std::array<S, N + 1> &b,
                                         std::size_t k,
                                         std::size_t j0 = 0) noexcept {
    const auto js = std::views::iota(j0, k);
    return std::transform_reduce(
        js.begin(), js.end(), S{}, std::plus{},
        [&](std::size_t j) { return a[j] * static_cast<S>(k - j) * b[k - j]; });
  }

  // Solve g·w' = u' for w, given w[0].  For k >= 1:
  //   g[0]·k·w[k] = k·u[k] − Σ_{j=1}^{k-1} (k-j)·g[j]·w[k-j]
  // Stated over the whole series g, so it holds for a composite u; every
  // inverse function below is this recurrence with a different g.
  [[nodiscard]] static constexpr TaylorDual
  implicit_recurrence(const TaylorDual &u, S w0, const TaylorDual &g) noexcept {
    const S invg0 = S{1} / g.c[0];
    return series(w0, [&](const auto &w, std::size_t k) {
      return (static_cast<S>(k) * u.c[k] - dconv(g.c, w, k, 1)) * inv_k[k] *
             invg0;
    });
  }

  // w[0] = w0, then w[k] = next(w, k) for k = 1..N: each coefficient reads
  // only those already written.
  [[nodiscard]] static constexpr TaylorDual series(S w0, auto next) noexcept {
    TaylorDual w{w0};
    for (std::size_t k = 1; k <= N; ++k) {
      w.c[k] = next(w.c, k);
    }
    return w;
  }

  // exp: w[0]=exp(u[0]),  k*w[k] = Σ_{j=1}^{k} j*u[j]*w[k-j]
  [[nodiscard]] friend constexpr TaylorDual exp(const TaylorDual &u) noexcept {
    using std::exp;
    return series(exp(u.c[0]), [&](const auto &w, std::size_t k) {
      return dconv(w, u.c, k) * inv_k[k];
    });
  }

  // log: w[0]=log(u[0]),  w[k] = (u[k] - (1/k)Σ_{j=1}^{k-1} j*w[j]*u[k-j]) /
  // u[0]
  [[nodiscard]] friend constexpr TaylorDual log(const TaylorDual &u) noexcept {
    using std::log;
    const S invu0 = S{1} / u.c[0];
    return series(log(u.c[0]), [&](const auto &w, std::size_t k) {
      return (u.c[k] - dconv(u.c, w, k, 1) * inv_k[k]) * invu0;
    });
  }

  // The coupled sine/cosine recurrence; circular and hyperbolic differ in one
  // sign:
  //   k*s[k] = Σ j*u[j]*c[k-j],   k*c[k] = ±Σ j*u[j]*s[k-j]
  // Circular = -1, hyperbolic = +1.  Either series needs both, so they are
  // computed together.
  template <int Sign>
  [[nodiscard]] static constexpr std::pair<TaylorDual, TaylorDual>
  coupled_pair(const TaylorDual &u, S s0, S c0) noexcept {
    static_assert(Sign == 1 || Sign == -1);
    TaylorDual s{s0}, cc{c0};
    for (std::size_t k = 1; k <= N; ++k) {
      S sv{}, cv{};
      for (std::size_t j = 1; j <= k; ++j) {
        const S ju = static_cast<S>(j) * u.c[j];
        sv += ju * cc.c[k - j];
        if constexpr (Sign < 0) {
          cv -= ju * s.c[k - j];
        } else {
          cv += ju * s.c[k - j];
        }
      }
      s.c[k] = sv * inv_k[k];
      cc.c[k] = cv * inv_k[k];
    }
    return {s, cc};
  }

  [[nodiscard]] friend constexpr std::pair<TaylorDual, TaylorDual>
  sincos_td(const TaylorDual &u) noexcept {
    using std::sin, std::cos;
    return coupled_pair<-1>(u, sin(u.c[0]), cos(u.c[0]));
  }

  [[nodiscard]] constexpr friend TaylorDual sin(const TaylorDual &u) noexcept {
    return sincos_td(u).first;
  }
  [[nodiscard]] constexpr friend TaylorDual cos(const TaylorDual &u) noexcept {
    return sincos_td(u).second;
  }
  [[nodiscard]] constexpr friend TaylorDual tan(const TaylorDual &u) noexcept {
    const auto [sn, cn] = sincos_td(u);
    return sn / cn;
  }

  // sqrt: w[0]=sqrt(u[0]),  w[k] = (u[k] - Σ_{j=1}^{k-1} w[j]*w[k-j]) /
  // (2*w[0])
  [[nodiscard]] constexpr friend TaylorDual sqrt(const TaylorDual &u) noexcept {
    using std::sqrt;
    const S w0 = sqrt(u.c[0]);
    const auto inv2w0 = S{1} / (S{2} * w0);
    return series(w0, [&](const auto &w, std::size_t k) {
      return (u.c[k] - conv(w, w, k, 1, k)) * inv2w0;
    });
  }

  [[nodiscard]] constexpr friend TaylorDual abs(const TaylorDual &u) noexcept {
    if (u.c[0] == S{}) {
      return TaylorDual{}; // sign(0) is 0, the convention every engine shares
    } else if (u.c[0] > S{}) {
      return u;
    } else if (u.c[0] < S{}) {
      return -u;
    }
    // NaN value: a series zero, not S{} -- the convolution poisons every c[k].
    return u * TaylorDual{};
  }

  [[nodiscard]] constexpr friend std::pair<TaylorDual, TaylorDual>
  sinhcosh_td(const TaylorDual &u) noexcept {
    using std::sinh, std::cosh;
    return coupled_pair<1>(u, sinh(u.c[0]), cosh(u.c[0]));
  }

  [[nodiscard]] constexpr friend TaylorDual sinh(const TaylorDual &u) noexcept {
    return sinhcosh_td(u).first;
  }
  [[nodiscard]] constexpr friend TaylorDual cosh(const TaylorDual &u) noexcept {
    return sinhcosh_td(u).second;
  }
  // tanh: w' = (1 − w²)·u', seeded from libm rather than sinh/cosh -- both
  // overflow near |u[0]| ≈ 710 where tanh is ±1.
  [[nodiscard]] constexpr friend TaylorDual tanh(const TaylorDual &u) noexcept {
    using std::tanh;
    TaylorDual w{tanh(u.c[0])};
    std::array<S, N + 1> sq{}; // w², extended as each w[k] lands
    sq[0] = w.c[0] * w.c[0];
    for (std::size_t k = 1; k <= N; ++k) {
      w.c[k] = (static_cast<S>(k) * u.c[k] - dconv(sq, u.c, k)) * inv_k[k];
      sq[k] = conv(w.c, w.c, k, 0, k + 1);
    }
    return w;
  }

  // asin: sqrt(1-u²)·w' = u'  (general, composite-safe):
  //   s[0]·k·w[k] = k·u[k] − Σ_{j=1}^{k-1} (k-j)·s[k-j]·w[k-j]
  [[nodiscard]] constexpr friend TaylorDual asin(const TaylorDual &u) noexcept {
    using std::asin;
    // (1-u)(1+u), not 1-u*u: near |u| = 1 the subtraction cancels against a
    // rounded square.  Ref: Higham, Accuracy and Stability §1.7.
    return implicit_recurrence(u, asin(u.c[0]), sqrt((S{1} - u) * (S{1} + u)));
  }

  // acos = pi/2 - asin: asin's derivative coefficients, negated.
  [[nodiscard]] friend constexpr TaylorDual acos(const TaylorDual &u) noexcept {
    using std::acos;
    TaylorDual w = -asin(u);
    w.c[0] = acos(u.c[0]);
    return w;
  }

  // atan: (1+u²)·w' = u'  (general, composite-safe):
  //   p[0]·k·w[k] = k·u[k] − Σ_{j=1}^{k-1} (k-j)·p[k-j]·w[k-j]
  [[nodiscard]] friend constexpr TaylorDual atan(const TaylorDual &u) noexcept {
    using std::atan;
    return implicit_recurrence(u, atan(u.c[0]), S{1} + u * u);
  }

  // Real-power series for w = u^p via the ODE  u·w' = p·u'·w, given w[0].
  //   n·u[0]·w[n] = p·Σ_{j=1}^{n} j·u[j]·w[n-j] − Σ_{j=1}^{n-1} j·u[n-j]·w[j]
  [[nodiscard]] static constexpr TaylorDual powser(const TaylorDual &u, S p,
                                                   S w0) noexcept {
    const S invu0 = S{1} / u.c[0];
    return series(w0, [&](const auto &w, std::size_t n) {
      return (p * dconv(w, u.c, n) - dconv(u.c, w, n, 1)) * inv_k[n] * invu0;
    });
  }

  // A constant exponent goes through powser, which needs no logarithm and
  // allows u[0] < 0; at u[0] == 0 powser is 0/0, so a whole non-negative
  // exponent takes the truncated product instead, and any other positive one
  // is zero below order p and has no coefficient from there on.
  [[nodiscard]] friend constexpr TaylorDual
  pow(const TaylorDual &u, const ConstOperand<S> auto &exponent) noexcept {
    using std::pow, std::trunc;
    const S p = S(exponent);
    if constexpr (CArithmetic<S>) {
      if (u.c[0] == S{} && !(p < S{}) && p == trunc(p)) {
        TaylorDual w;
        if (p > static_cast<S>(N)) {
          return w; // u has valuation >= 1, so u^p starts past order N
        }
        w.c[0] = S{1};
        for (auto e = static_cast<std::size_t>(p); e > 0; --e) {
          w = w * u;
        }
        return w;
      }
      if (u.c[0] == S{} && p > S{}) {
        TaylorDual w;
        for (std::size_t k = 1; k <= N; ++k) {
          if (!(static_cast<S>(k) < p)) {
            w.c[k] = std::numeric_limits<S>::quiet_NaN();
          }
        }
        return w;
      }
    }
    return powser(u, p, pow(u.c[0], p));
  }

  // exp(v·log u), requires u[0] > 0 -- unless v is a constant that arrived
  // embedded, which the sweep hands over as a series with an all-zero tail.
  [[nodiscard]] friend constexpr TaylorDual pow(const TaylorDual &u,
                                                const TaylorDual &v) noexcept {
    if (std::ranges::all_of(v.c | std::views::drop(1),
                            [](const S &e) { return e == S{}; })) {
      return pow(u, v.c[0]);
    }
    return exp(v * log(u));
  }

  // cbrt via the same recurrence; w[0] = std::cbrt handles a negative argument.
  [[nodiscard]] friend constexpr TaylorDual cbrt(const TaylorDual &u) noexcept {
    using std::cbrt;
    return powser(u, S{1} / S{3}, cbrt(u.c[0]));
  }

  // log10(u) = log(u) / ln(10); the value is std::log10's, the derivative
  // coefficients log's scaled by log10(e).
  [[nodiscard]] friend constexpr TaylorDual
  log10(const TaylorDual &u) noexcept {
    using std::log10;
    TaylorDual w = log(u) * static_cast<S>(std::numbers::log10e);
    w.c[0] = log10(u.c[0]);
    return w;
  }

  // asinh: sqrt(1+u²)·w' = u'  (mirrors asin with s = sqrt(1 + u²)).
  [[nodiscard]] friend constexpr TaylorDual
  asinh(const TaylorDual &u) noexcept {
    using std::asinh;
    return implicit_recurrence(u, asinh(u.c[0]), sqrt(S{1} + u * u));
  }

  // acosh: sqrt(u²-1)·w' = u'  (requires u[0] > 1); (u-1)(u+1) for the same
  // cancellation reason as asin.
  [[nodiscard]] friend constexpr TaylorDual
  acosh(const TaylorDual &u) noexcept {
    using std::acosh;
    return implicit_recurrence(u, acosh(u.c[0]), sqrt((u - S{1}) * (u + S{1})));
  }

  // atanh: (1-u²)·w' = u'  (mirrors atan); (1-u)(1+u) for the same
  // cancellation reason as asin.
  [[nodiscard]] friend constexpr TaylorDual
  atanh(const TaylorDual &u) noexcept {
    using std::atanh;
    return implicit_recurrence(u, atanh(u.c[0]), (S{1} - u) * (S{1} + u));
  }

  // erf: erf' = (2/√π)·exp(-u²); integrate w' = (2/√π)·exp(-u²)·u'.
  [[nodiscard]] friend constexpr TaylorDual erf(const TaylorDual &u) noexcept {
    using std::erf;
    constexpr S cc = static_cast<S>(S{2} * std::numbers::inv_sqrtpi);
    const TaylorDual g = exp(-(u * u)); // exp(-u²)
    return series(erf(u.c[0]), [&](const auto &, std::size_t k) {
      return cc * dconv(g.c, u.c, k) * inv_k[k];
    });
  }

  // (x² + y²)·w' = x·y' − y·x', so the recurrence never divides by x.  The
  // squares overflow past ~1e154, so both sides first scale by the power of
  // two that brings the larger value near 1: exact, and the ratio unchanged.
  [[nodiscard]] friend constexpr TaylorDual atan2(TaylorDual y,
                                                  TaylorDual x) noexcept {
    using std::atan2;
    if constexpr (CArithmetic<S>) {
      using std::abs, std::fmax, std::ilogb, std::isfinite, std::scalbn;
      const S big = fmax(abs(x.c[0]), abs(y.c[0]));
      if (isfinite(big) && big != S{}) {
        const int e = -ilogb(big);
        const auto by = [e](const S &v) noexcept { return scalbn(v, e); };
        std::ranges::transform(x.c, x.c.begin(), by);
        std::ranges::transform(y.c, y.c.begin(), by);
      }
    }
    // n integrates x·y' − y·x', which is all implicit_recurrence reads of u.
    TaylorDual n;
    for (std::size_t k = 1; k <= N; ++k) {
      n.c[k] = (dconv(x.c, y.c, k) - dconv(y.c, x.c, k)) * inv_k[k];
    }
    return implicit_recurrence(n, atan2(y.c[0], x.c[0]), x * x + y * y);
  }

  // Never sqrt(x² + y²): the value is std::hypot's and the coefficients come
  // from h·h' = x·x' + y·y', so nothing is squared.
  [[nodiscard]] friend constexpr TaylorDual
  hypot(const TaylorDual &x, const TaylorDual &y) noexcept {
    using std::hypot;
    const S h0 = hypot(x.c[0], y.c[0]);
    const S invh = S{1} / h0;
    return series(h0, [&](const auto &w, std::size_t k) {
      return (dconv(x.c, x.c, k) + dconv(y.c, y.c, k) - dconv(w, w, k, 1)) *
             inv_k[k] * invh;
    });
  }
};

// Linear, so it applies coefficient-wise -- the same reasoning as Dual's.
template <Numeric S, std::size_t N>
[[nodiscard]] constexpr TaylorDual<S, N>
midpoint(const TaylorDual<S, N> &a, const TaylorDual<S, N> &b) noexcept {
  using std::midpoint;
  TaylorDual<S, N> r;
  std::ranges::transform(a.c, b.c, r.c.begin(),
                         [](const S &x, const S &y) { return midpoint(x, y); });
  return r;
}

static_assert(Numeric<TaylorDual<double, 3>>,
              "TaylorDual must satisfy Numeric to nest in an expression");

// A table row without a recurrence above fails here, not in the first sweep
// that reaches it.
#define DDX_TAYLOR_HAS(FN, NAME, LABEL)                                        \
  static_assert(                                                               \
      requires(const TaylorDual<double, 3> &t) { FN(t); },                     \
      LABEL " has no TaylorDual overload");
DDX_UNARY_MATH_TABLE(DDX_TAYLOR_HAS)
#undef DDX_TAYLOR_HAS

template <Numeric T, std::size_t N>
auto scalar_base_impl(std::type_identity<TaylorDual<T, N>>) -> T;

// dual_depth_v<TaylorDual<S,N>> = N  (mirrors nth_dual_t<S,N> depth)
template <Numeric S, std::size_t N>
inline constexpr std::size_t dual_depth_v<TaylorDual<S, N>> = N;

// Lets Constant::eval_seeded produce a zero-derivative TaylorDual.

template <Numeric S, std::size_t N> struct ConstantEmbedder<TaylorDual<S, N>> {
  static constexpr TaylorDual<S, N> embed(S val) noexcept {
    return TaylorDual<S, N>{val};
  }
};

namespace detail {
template <Numeric S, std::size_t N>
inline constexpr bool is_dual_family_v<TaylorDual<S, N>> = true;
} // namespace detail

} // namespace ddx::impl

// Over the normalized coefficients f^(k)/k!; univariate_derivative puts the
// factorial back.
template <ddx::impl::Numeric S, std::size_t N>
struct std::formatter<ddx::impl::TaylorDual<S, N>, char>
    : ddx::impl::detail::dual_formatter_base<S> {
  auto format(const ddx::impl::TaylorDual<S, N> &t,
              std::format_context &ctx) const {
    this->series(ctx, t.c);
    return ctx.out();
  }
};
