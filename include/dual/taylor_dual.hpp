#pragma once

#include "dual/dual.hpp"
#include "ops/unary_math.hpp" // DDX_UNARY_MATH_TABLE
#include "util/config.hpp"    // DDX_ALWAYS_INLINE
#include "util/meta.hpp"
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

// Taylor coefficients c[k] = f^(k)(x)/k!, multiplied by truncated convolution
// in O(N^2). Each function comes from its ODE, never from Faa di Bruno.
// Ref: Jorba & Zou, Exp. Math. 14(1) (2005) 99 §2.
// Ref: Berz, Particle Accelerators 24 (1989) 109.

// The coefficient product is a convolution, sum_i a_i b_(k-i), so it commutes
// exactly when the scalar underneath does.
template <Numeric S, std::size_t N> struct TaylorDual;
template <Numeric S, std::size_t N>
inline constexpr bool is_commutative_multiply_v<TaylorDual<S, N>> =
    is_commutative_multiply_v<S>;

template <Numeric S, std::size_t N>
struct TaylorDual : compound_from_binary<TaylorDual<S, N>> {
  std::array<S, N + 1> c{};

  // Precomputed 1/k, so the recurrences multiply instead of dividing.
  static constexpr std::array<S, N + 1> inv_k = []() consteval {
    std::array<S, N + 1> r{};
    for (auto &&[k, rk] : r | std::views::enumerate | std::views::drop(1)) {
      rk = S{1} / static_cast<S>(k);
    }
    return r;
  }();

  // Calls f(k) for k = First..N, each k a std::integral_constant, so every
  // convolution an order runs has a fixed length and unrolls. Forced inline
  // with generic callers, so each order's body is called once and inlined
  // into the step that runs it.
  template <std::size_t First>
  DDX_ALWAYS_INLINE static constexpr void orders(auto &&f) {
    meta::unpack(meta::iota<N + 1 - First>, [&](auto... k) DDX_LAMBDA_INLINE {
      (f(meta::index_t<First + k>{}), ...);
    });
  }

  constexpr TaylorDual() noexcept = default;
  constexpr explicit TaylorDual(S val) noexcept : c{} { c[0] = val; }

  // The independent variable at x0. Its unit first coefficient makes the
  // sweep's c[k] come out as f^(k)(x0)/k!.
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

  // The truncated convolution Σ_{j=j0}^{j1-1} a[j]·b[k-j], which every product
  // and recurrence below is built from.
  [[nodiscard]] DDX_ALWAYS_INLINE static constexpr S
  conv(const std::array<S, N + 1> &a, const std::array<S, N + 1> &b,
       std::size_t k, std::size_t j0, std::size_t j1) noexcept {
    const auto js = std::views::iota(j0, j1);
    return std::transform_reduce(
        js.begin(), js.end(), S{}, std::plus{},
        [&](std::size_t j) { return a[j] * b[k - j]; });
  }

  // (uv)[k] = Σ_{j=0}^{k} u[j]·v[k-j], quadratic at any practical N.
  // Ref: Knuth, TAOCP Vol. 2 §4.7.
  DDX_ALWAYS_INLINE constexpr TaylorDual
  operator*(const TaylorDual &o) const noexcept {
    TaylorDual r;
    orders<0>([&](auto k)
                  DDX_LAMBDA_INLINE { r.c[k] = conv(c, o.c, k, 0, k + 1); });
    return r;
  }

  // Division solves r·o = this by forward substitution (Knuth §4.7):
  //   r[k] = (c[k] - Σ_{j=0}^{k-1} r[j]·o[k-j]) / o[0]
  DDX_ALWAYS_INLINE constexpr TaylorDual
  operator/(const TaylorDual &o) const noexcept {
    TaylorDual r;
    const S inv0 = S{1} / o.c[0];
    orders<0>([&](auto k) DDX_LAMBDA_INLINE {
      r.c[k] = (c[k] - conv(r.c, o.c, k, 0, k)) * inv0;
    });
    return r;
  }

  // Multiplies every coefficient by s, keeping s on the side it was written.
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

  // A constant shifts c[0] or scales every coefficient. RIGHT and LEFT update
  // u for `u OP s` and `s OP u`; s / u still divides, as 1/u is a series.
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

  // Comparisons read the value coefficient only, as Dual's do. The branching
  // kernels in operations.hpp rely on them.
  [[nodiscard]] friend constexpr auto
  operator<=>(const TaylorDual &a, const TaylorDual &b) noexcept {
    return a.c[0] <=> b.c[0];
  }
  [[nodiscard]] friend constexpr bool operator==(const TaylorDual &a,
                                                 const TaylorDual &b) noexcept {
    return a.c[0] == b.c[0];
  }

  [[nodiscard]] friend constexpr bool same_value(const TaylorDual &a,
                                                 const TaylorDual &b) noexcept {
    return a.c == b.c;
  }

  // Coefficient k-1 of a·b', Σ_{j=j0}^{k-1} a[j]·(k-j)·b[k-j]. Passing j0 = 1
  // drops the a[0] term that a recurrence has already moved to the left.
  [[nodiscard]] DDX_ALWAYS_INLINE static constexpr S
  dconv(const std::array<S, N + 1> &a, const std::array<S, N + 1> &b,
        std::size_t k, std::size_t j0 = 0) noexcept {
    const auto js = std::views::iota(j0, k);
    return std::transform_reduce(
        js.begin(), js.end(), S{}, std::plus{},
        [&](std::size_t j) { return a[j] * static_cast<S>(k - j) * b[k - j]; });
  }

  // Solves g·w' = u' for w given w[0]; every inverse function below uses it:
  //   g[0]·k·w[k] = k·u[k] − Σ_{j=1}^{k-1} (k-j)·g[j]·w[k-j]
  [[nodiscard]] static constexpr TaylorDual
  implicit_recurrence(const TaylorDual &u, S w0, const TaylorDual &g) noexcept {
    const S invg0 = S{1} / g.c[0];
    return series(w0, [&](const std::array<S, N + 1> &w, auto k) {
      return (static_cast<S>(k) * u.c[k] - dconv(g.c, w, k, 1)) * inv_k[k] *
             invg0;
    });
  }

  // Sets w[0] = w0, then w[k] = next(w, k) for k = 1..N. Each coefficient
  // reads only the ones already written.
  [[nodiscard]] DDX_ALWAYS_INLINE static constexpr TaylorDual
  series(S w0, auto next) noexcept {
    TaylorDual w{w0};
    orders<1>([&](auto k) DDX_LAMBDA_INLINE { w.c[k] = next(w.c, k); });
    return w;
  }

  // exp: w[0] = exp(u[0]),  k·w[k] = Σ_{j=1}^{k} j·u[j]·w[k-j]
  [[nodiscard]] friend constexpr TaylorDual exp(const TaylorDual &u) noexcept {
    using std::exp;
    return series(exp(u.c[0]), [&](const std::array<S, N + 1> &w, auto k) {
      return TaylorDual::dconv(w, u.c, k) * TaylorDual::inv_k[k];
    });
  }

  // log: w[0] = log(u[0]),
  //   w[k] = (u[k] - (1/k)·Σ_{j=1}^{k-1} j·w[j]·u[k-j]) / u[0]
  [[nodiscard]] friend constexpr TaylorDual log(const TaylorDual &u) noexcept {
    using std::log;
    const S invu0 = S{1} / u.c[0];
    return series(log(u.c[0]), [&](const std::array<S, N + 1> &w, auto k) {
      return (u.c[k] - TaylorDual::dconv(u.c, w, k, 1) * TaylorDual::inv_k[k]) *
             invu0;
    });
  }

  // The coupled recurrence, Sign = -1 for sin/cos and +1 for sinh/cosh:
  //   k·s[k] = Σ j·u[j]·c[k-j],   k·c[k] = Sign·Σ j·u[j]·s[k-j]
  template <int Sign>
  [[nodiscard]] DDX_ALWAYS_INLINE static constexpr std::pair<TaylorDual,
                                                             TaylorDual>
  coupled_pair(const TaylorDual &u, S s0, S c0) noexcept {
    static_assert(Sign == 1 || Sign == -1);
    TaylorDual s{s0}, cc{c0};
    orders<1>([&](auto k) DDX_LAMBDA_INLINE {
      S sv{}, cv{};
      // Oldest coefficient first, so the newest, s[k-1] and c[k-1], enter
      // last and the chain to c[k] is one fma long rather than k.
      for (std::size_t j = k; j >= 1; --j) {
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
    });
    return {s, cc};
  }

  [[nodiscard]] friend DDX_ALWAYS_INLINE constexpr std::pair<TaylorDual,
                                                             TaylorDual>
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

  // sqrt: w[0] = sqrt(u[0]),
  //   w[k] = (u[k] - Σ_{j=1}^{k-1} w[j]·w[k-j]) / (2·w[0])
  [[nodiscard]] constexpr friend TaylorDual sqrt(const TaylorDual &u) noexcept {
    using std::sqrt;
    const S w0 = sqrt(u.c[0]);
    const auto inv2w0 = S{1} / (S{2} * w0);
    return series(w0, [&](const std::array<S, N + 1> &w, auto k) {
      return (u.c[k] - TaylorDual::conv(w, w, k, 1, k)) * inv2w0;
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
    // A NaN value multiplies a zero series, so the NaN reaches every c[k].
    return u * TaylorDual{};
  }

  [[nodiscard]] friend DDX_ALWAYS_INLINE constexpr std::pair<TaylorDual,
                                                             TaylorDual>
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
  // tanh: w' = (1 − w²)·u', seeded from std::tanh because sinh and cosh both
  // overflow near |u[0]| ≈ 710, where tanh is ±1.
  [[nodiscard]] constexpr friend TaylorDual tanh(const TaylorDual &u) noexcept {
    using std::tanh;
    TaylorDual w{tanh(u.c[0])};
    std::array<S, N + 1> sq{}; // w², extended as each w[k] lands
    sq[0] = w.c[0] * w.c[0];
    orders<1>([&](auto k) DDX_LAMBDA_INLINE {
      // Qualified: MSVC does not find statics from a lambda in a hidden friend.
      w.c[k] = (static_cast<S>(k) * u.c[k] - TaylorDual::dconv(sq, u.c, k)) *
               TaylorDual::inv_k[k];
      sq[k] = TaylorDual::conv(w.c, w.c, k, 0, k + 1);
    });
    return w;
  }

  // asin: sqrt(1-u²)·w' = u'.
  [[nodiscard]] constexpr friend TaylorDual asin(const TaylorDual &u) noexcept {
    using std::asin;
    // (1-u)(1+u) rather than 1-u*u avoids cancellation near |u| = 1.
    // Ref: Higham, Accuracy and Stability §1.7.
    return implicit_recurrence(u, asin(u.c[0]), sqrt((S{1} - u) * (S{1} + u)));
  }

  // acos = pi/2 - asin, so its derivative coefficients are asin's, negated.
  [[nodiscard]] friend constexpr TaylorDual acos(const TaylorDual &u) noexcept {
    using std::acos;
    TaylorDual w = -asin(u);
    w.c[0] = acos(u.c[0]);
    return w;
  }

  // atan: (1+u²)·w' = u'.
  [[nodiscard]] friend constexpr TaylorDual atan(const TaylorDual &u) noexcept {
    using std::atan;
    return implicit_recurrence(u, atan(u.c[0]), S{1} + u * u);
  }

  // Real power w = u^p from the ODE u·w' = p·u'·w, given w[0]:
  //   n·u[0]·w[n] = p·Σ_{j=1}^{n} j·u[j]·w[n-j] − Σ_{j=1}^{n-1} j·u[n-j]·w[j]
  [[nodiscard]] static constexpr TaylorDual powser(const TaylorDual &u, S p,
                                                   S w0) noexcept {
    const S invu0 = S{1} / u.c[0];
    return series(w0, [&](const std::array<S, N + 1> &w, auto n) {
      return (p * dconv(w, u.c, n) - dconv(u.c, w, n, 1)) * inv_k[n] * invu0;
    });
  }

  // powser needs no log and allows u[0] < 0 but is 0/0 at u[0] == 0, where a
  // whole power multiplies out and any other is 0 below order p and NaN above.
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
        orders<1>([&](auto k) {
          if (!(static_cast<S>(k) < p)) {
            w.c[k] = std::numeric_limits<S>::quiet_NaN();
          }
        });
        return w;
      }
    }
    return powser(u, p, pow(u.c[0], p));
  }

  // exp(v·log u) needs u[0] > 0, so a constant v, which arrives as a series
  // with an all-zero tail, takes the constant-exponent path instead.
  [[nodiscard]] friend constexpr TaylorDual pow(const TaylorDual &u,
                                                const TaylorDual &v) noexcept {
    if (std::ranges::all_of(v.c | std::views::drop(1),
                            [](const S &e) { return e == S{}; })) {
      return pow(u, v.c[0]);
    }
    return exp(v * log(u));
  }

  // cbrt uses powser, seeded from std::cbrt so a negative argument works.
  [[nodiscard]] friend constexpr TaylorDual cbrt(const TaylorDual &u) noexcept {
    using std::cbrt;
    return powser(u, S{1} / S{3}, cbrt(u.c[0]));
  }

  // log10 takes its value from std::log10 and its derivative coefficients
  // from log's, scaled by log10(e).
  [[nodiscard]] friend constexpr TaylorDual
  log10(const TaylorDual &u) noexcept {
    using std::log10;
    TaylorDual w = log(u) * static_cast<S>(std::numbers::log10e);
    w.c[0] = log10(u.c[0]);
    return w;
  }

  // asinh: sqrt(1+u²)·w' = u'.
  [[nodiscard]] friend constexpr TaylorDual
  asinh(const TaylorDual &u) noexcept {
    using std::asinh;
    return implicit_recurrence(u, asinh(u.c[0]), sqrt(S{1} + u * u));
  }

  // acosh: sqrt(u²-1)·w' = u' with u[0] > 1. (u-1)(u+1) avoids cancellation,
  // as in asin.
  [[nodiscard]] friend constexpr TaylorDual
  acosh(const TaylorDual &u) noexcept {
    using std::acosh;
    return implicit_recurrence(u, acosh(u.c[0]), sqrt((u - S{1}) * (u + S{1})));
  }

  // atanh: (1-u²)·w' = u'. (1-u)(1+u) avoids cancellation, as in asin.
  [[nodiscard]] friend constexpr TaylorDual
  atanh(const TaylorDual &u) noexcept {
    using std::atanh;
    return implicit_recurrence(u, atanh(u.c[0]), (S{1} - u) * (S{1} + u));
  }

  // erf integrates w' = (2/√π)·exp(-u²)·u'.
  [[nodiscard]] friend constexpr TaylorDual erf(const TaylorDual &u) noexcept {
    using std::erf;
    constexpr S cc = static_cast<S>(S{2} * std::numbers::inv_sqrtpi);
    const TaylorDual g = exp(-(u * u));
    return series(erf(u.c[0]), [&](const std::array<S, N + 1> &, auto k) {
      return cc * TaylorDual::dconv(g.c, u.c, k) * TaylorDual::inv_k[k];
    });
  }

  // atan2 from (x² + y²)·w' = x·y' − y·x', which never divides by x. Both
  // inputs first scale by a power of two so the squares cannot overflow.
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
    // n integrates x·y' − y·x', the only part of u implicit_recurrence reads.
    TaylorDual n;
    orders<1>([&](auto k) DDX_LAMBDA_INLINE {
      // Qualified, as in tanh.
      n.c[k] =
          (TaylorDual::dconv(x.c, y.c, k) - TaylorDual::dconv(y.c, x.c, k)) *
          TaylorDual::inv_k[k];
    });
    return implicit_recurrence(n, atan2(y.c[0], x.c[0]), x * x + y * y);
  }

  // hypot takes its value from std::hypot and its coefficients from
  // h·h' = x·x' + y·y', so nothing is squared and nothing overflows.
  [[nodiscard]] friend constexpr TaylorDual
  hypot(const TaylorDual &x, const TaylorDual &y) noexcept {
    using std::hypot;
    const S h0 = hypot(x.c[0], y.c[0]);
    const S invh = S{1} / h0;
    return series(h0, [&](const std::array<S, N + 1> &w, auto k) {
      return (TaylorDual::dconv(x.c, x.c, k) + TaylorDual::dconv(y.c, y.c, k) -
              TaylorDual::dconv(w, w, k, 1)) *
             TaylorDual::inv_k[k] * invh;
    });
  }
};

// midpoint is linear, so it applies coefficient-wise, as for Dual.
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

// A math table row with no TaylorDual overload fails here rather than in the
// first sweep that reaches it.
#define DDX_TAYLOR_HAS(FN, NAME, LABEL)                                        \
  static_assert(                                                               \
      requires(const TaylorDual<double, 3> &t) { FN(t); },                     \
      LABEL " has no TaylorDual overload");
DDX_UNARY_MATH_TABLE(DDX_TAYLOR_HAS)
#undef DDX_TAYLOR_HAS

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

// Prints the normalized coefficients f^(k)/k!; univariate_derivative puts the
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
