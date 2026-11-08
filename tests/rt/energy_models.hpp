#pragma once
#include "rt/derivative.hpp"
#include <cmath>
#include <numeric>
#include <ranges>
#include <vector>

// Thermodynamic models built as runtime graphs. The parameters are arbitrary
// values of a realistic magnitude, since only the differentiation is tested.
namespace models {

using RE = ddx::rt::Expression<>;

inline double pseudo(std::size_t i, std::size_t j, double lo, double hi) {
  const double t = static_cast<double>((i * 7919 + j * 104729) % 1000) / 1000.0;
  return lo + t * (hi - lo);
}

// UNIQUAC excess Gibbs energy over n species at a fixed temperature.
inline RE uniquac(const std::vector<RE> &x) {
  const std::size_t n = x.size();
  std::vector<double> r(n), q(n);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    r[i] = pseudo(i, 1, 0.9, 5.2);
    q[i] = pseudo(i, 2, 0.8, 4.4);
  }
  RE sum_r{0}, sum_q{0};
  for (const std::size_t i : std::views::iota(0uz, n)) {
    sum_r += r[i] * x[i];
    sum_q += q[i] * x[i];
  }

  std::vector<RE> phi, theta;
  phi.reserve(n);
  theta.reserve(n);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    phi.push_back(r[i] * x[i] / sum_r);
    theta.push_back(q[i] * x[i] / sum_q);
  }

  RE comb{0};
  for (const std::size_t i : std::views::iota(0uz, n)) {
    comb += x[i] * log(phi[i] / x[i]) +
            5.0 * q[i] * x[i] * log(theta[i] / phi[i]); // z/2 = 5
  }

  RE res{0};
  for (const std::size_t i : std::views::iota(0uz, n)) {
    RE inner{0};
    for (const std::size_t j : std::views::iota(0uz, n)) {
      inner += theta[j] * pseudo(j, i, 0.2, 1.8); // tau_ji, T folded in
    }
    res -= q[i] * x[i] * log(inner);
  }
  return comb + res;
}

// Peng-Robinson cubic in the compressibility factor Z, which is a variable,
// with van der Waals mixing rules.
inline RE peng_robinson(const std::vector<RE> &x, const RE &Z) {
  const std::size_t n = x.size();
  std::vector<double> ai(n), bi(n);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    ai[i] = pseudo(i, 3, 0.3, 2.6);
    bi[i] = pseudo(i, 4, 0.02, 0.09);
  }
  RE a_mix{0};
  for (const auto [i, j] : std::views::cartesian_product(
           std::views::iota(0uz, n), std::views::iota(0uz, n))) {
    const double aij =
        std::sqrt(ai[i] * ai[j]) * (1.0 - pseudo(i, j, 0.0, 0.12));
    a_mix += x[i] * x[j] * aij;
  }
  RE b_mix{0};
  for (const auto [xi, b] : std::views::zip(x, bi)) {
    b_mix += xi * b;
  }

  const RE A = a_mix * 0.45724; // P/(RT)^2 folded into the constant
  const RE B = b_mix * 0.07780;
  return Z * Z * Z - (RE{1} - B) * Z * Z + (A - 3.0 * B * B - 2.0 * B) * Z -
         (A * B - B * B - B * B * B);
}

// An MSE-like model: Debye-Huckel, a middle-range ionic double sum and a
// short-range UNIQUAC term. The double sum makes the Hessian dense.
inline RE mse(const std::vector<RE> &x) {
  const std::size_t n = x.size();
  const auto zc = std::views::iota(0uz, n) |
                  std::views::transform([](std::size_t i) {
                    return pseudo(i, 5, -2.0, 2.0);
                  }) |
                  ddx::impl::to<std::vector>();

  RE ionic{0};
  for (const auto [z, xi] : std::views::zip(zc, x)) {
    ionic += 0.5 * z * z * xi;
  }

  const RE lr = -0.3915 * sqrt(ionic) / (RE{1} + 1.2 * sqrt(ionic));

  RE mr{0};
  for (const auto [i, j] : std::views::cartesian_product(
           std::views::iota(0uz, n), std::views::iota(0uz, n))) {
    mr += x[i] * x[j] * pseudo(i, j, -0.4, 0.4) * exp(-sqrt(ionic));
  }
  return lr + mr + uniquac(x);
}

// The models below couple each site only to its neighbours, so their Hessian
// is banded and the colouring has something to exploit.

// Cluster expansion truncated at nearest and next-nearest neighbours on a ring.
inline RE cluster_expansion(const std::vector<RE> &s, std::size_t range = 2) {
  const std::size_t n = s.size();
  RE e{0};
  for (const std::size_t i : std::views::iota(0uz, n)) {
    e += pseudo(i, 6, -0.5, 0.5) * s[i]; // point term
    for (const std::size_t d : std::views::iota(1uz, range + 1)) {
      const std::size_t j = (i + d) % n;
      e += pseudo(i, j, -0.3, 0.3) * s[i] * s[j]; // pair term
    }
  }
  return e;
}

// A bonded chain: stretch and bend terms, each local to a few sites.
inline RE bonded_chain(const std::vector<RE> &r) {
  RE e{0};
  for (const auto [k, w] : std::views::enumerate(std::views::slide(r, 2))) {
    const RE d = w[1] - w[0];
    e += pseudo(static_cast<std::size_t>(k), 7, 0.5, 2.0) * d * d; // stretch
  }
  for (const auto [k, w] : std::views::enumerate(std::views::slide(r, 3))) {
    const RE bend = w[2] - 2.0 * w[1] + w[0];
    e += pseudo(static_cast<std::size_t>(k), 8, 0.2, 1.0) * bend * bend;
  }
  return e;
}

// Regular-solution free energy f(x) = c x(1-x) + k(x ln x + (1-x) ln(1-x)).
// With k > 0 and c > 2k, x = 1/2 is a maximum between two minima.
inline RE regular_solution(const RE &x, double c, double k) {
  return c * x * (RE{1} - x) + k * (x * log(x) + (RE{1} - x) * log(RE{1} - x));
}

// An equimolar-ish composition summing to one.
inline std::vector<double> composition(std::size_t n) {
  auto pt = std::views::iota(0uz, n) |
            std::views::transform([n](std::size_t i) {
              return (1.0 + 0.3 * static_cast<double>(i % 5)) /
                     static_cast<double>(n);
            }) |
            ddx::impl::to<std::vector>();
  const double total = std::accumulate(pt.begin(), pt.end(), 0.0);
  for (double &v : pt) {
    v /= total;
  }
  return pt;
}

} // namespace models
