#pragma once
#include "rt/derivative.hpp"
#include <cmath>
#include <numeric>
#include <vector>

// UNIQUAC, Peng-Robinson and an MSE-shaped electrolyte model as runtime graphs:
// the species list comes from a database, so the expression cannot be written
// in source.  The parameters are deterministic nonsense of the right magnitude
// -- the machinery is what is under test, not the thermodynamics.
namespace models {

using RE = ddx::rt::Expression<>;

inline double pseudo(std::size_t i, std::size_t j, double lo, double hi) {
  const double t = static_cast<double>((i * 7919 + j * 104729) % 1000) / 1000.0;
  return lo + t * (hi - lo);
}

// UNIQUAC excess Gibbs energy over n species at fixed T.
//   comb = sum xi ln(Phi_i/xi) + (z/2) sum qi xi ln(theta_i/Phi_i)
//   res  = -sum qi xi ln(sum_j theta_j tau_ji)
inline RE uniquac(const std::vector<RE> &x) {
  const std::size_t n = x.size();
  std::vector<double> r(n), q(n);
  for (std::size_t i = 0; i < n; ++i) {
    r[i] = pseudo(i, 1, 0.9, 5.2);
    q[i] = pseudo(i, 2, 0.8, 4.4);
  }
  RE sum_r{0}, sum_q{0};
  for (std::size_t i = 0; i < n; ++i) {
    sum_r += r[i] * x[i];
    sum_q += q[i] * x[i];
  }

  std::vector<RE> phi, theta;
  phi.reserve(n);
  theta.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    phi.push_back(r[i] * x[i] / sum_r);
    theta.push_back(q[i] * x[i] / sum_q);
  }

  RE comb{0};
  for (std::size_t i = 0; i < n; ++i) {
    comb += x[i] * log(phi[i] / x[i]) +
            5.0 * q[i] * x[i] * log(theta[i] / phi[i]); // z/2 = 5
  }

  RE res{0};
  for (std::size_t i = 0; i < n; ++i) {
    RE inner{0};
    for (std::size_t j = 0; j < n; ++j) {
      inner += theta[j] * pseudo(j, i, 0.2, 1.8); // tau_ji, T folded in
    }
    res -= q[i] * x[i] * log(inner);
  }
  return comb + res;
}

// Peng-Robinson: the cubic in Z, with van der Waals mixing.  Z is a variable.
//   A = a_mix P/(RT)^2,  B = b_mix P/(RT)
//   f(Z) = Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3)
inline RE peng_robinson(const std::vector<RE> &x, const RE &Z) {
  const std::size_t n = x.size();
  std::vector<double> ai(n), bi(n);
  for (std::size_t i = 0; i < n; ++i) {
    ai[i] = pseudo(i, 3, 0.3, 2.6);
    bi[i] = pseudo(i, 4, 0.02, 0.09);
  }
  RE a_mix{0};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double aij =
          std::sqrt(ai[i] * ai[j]) * (1.0 - pseudo(i, j, 0.0, 0.12));
      a_mix += x[i] * x[j] * aij;
    }
  }
  RE b_mix{0};
  for (std::size_t i = 0; i < n; ++i) {
    b_mix += x[i] * bi[i];
  }

  const RE A = a_mix * 0.45724; // P/(RT)^2 folded into the constant
  const RE B = b_mix * 0.07780;
  return Z * Z * Z - (RE{1} - B) * Z * Z + (A - 3.0 * B * B - 2.0 * B) * Z -
         (A * B - B * B - B * B * B);
}

// MSE-flavoured: long-range Debye-Huckel, a middle-range ionic double sum and
// a short-range UNIQUAC.  The double sum is what makes the Hessian dense.
inline RE mse(const std::vector<RE> &x) {
  const std::size_t n = x.size();
  std::vector<double> zc(n);
  for (std::size_t i = 0; i < n; ++i) {
    zc[i] = pseudo(i, 5, -2.0, 2.0);
  }

  RE ionic{0};
  for (std::size_t i = 0; i < n; ++i) {
    ionic += 0.5 * zc[i] * zc[i] * x[i];
  }

  const RE lr = -0.3915 * sqrt(ionic) / (RE{1} + 1.2 * sqrt(ionic));

  RE mr{0};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      mr += x[i] * x[j] * pseudo(i, j, -0.4, 0.4) * exp(-sqrt(ionic));
    }
  }
  return lr + mr + uniquac(x);
}

// --- sparse models ---------------------------------------------------------
// A finite interaction range couples each site to its neighbours only, so the
// Hessian is banded and the colouring has something to exploit.

// Cluster expansion truncated at nearest and next-nearest neighbours on a ring.
inline RE cluster_expansion(const std::vector<RE> &s, std::size_t range = 2) {
  const std::size_t n = s.size();
  RE e{0};
  for (std::size_t i = 0; i < n; ++i) {
    e += pseudo(i, 6, -0.5, 0.5) * s[i]; // point term
    for (std::size_t d = 1; d <= range; ++d) {
      const std::size_t j = (i + d) % n;
      e += pseudo(i, j, -0.3, 0.3) * s[i] * s[j]; // pair term
    }
  }
  return e;
}

// A bonded chain: stretch and bend terms, each local to a few sites.
inline RE bonded_chain(const std::vector<RE> &r) {
  const std::size_t n = r.size();
  RE e{0};
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const RE d = r[i + 1] - r[i];
    e += pseudo(i, 7, 0.5, 2.0) * d * d; // stretch
  }
  for (std::size_t i = 0; i + 2 < n; ++i) {
    const RE bend = r[i + 2] - 2.0 * r[i + 1] + r[i];
    e += pseudo(i, 8, 0.2, 1.0) * bend * bend;
  }
  return e;
}

// Regular-solution free energy of a binary mixture:
//
//   f(x) = c x(1-x) + k( x ln x + (1-x) ln(1-x) )
//
// Symmetric about x = 1/2, so f'(1/2) is exactly zero for every c and k, and
// f''(x) = -2c + k/(x(1-x)) in closed form.  With k > 0 and c > 2k, x = 1/2 is
// a maximum flanked by two minima -- a miscibility gap.  k < 0 inverts that and
// admits no interior double well for any c.
inline RE regular_solution(const RE &x, double c, double k) {
  return c * x * (RE{1} - x) + k * (x * log(x) + (RE{1} - x) * log(RE{1} - x));
}

// An equimolar-ish composition summing to one.
inline std::vector<double> composition(std::size_t n) {
  std::vector<double> pt(n);
  for (std::size_t i = 0; i < n; ++i) {
    pt[i] = (1.0 + 0.3 * static_cast<double>(i % 5)) / static_cast<double>(n);
  }
  const double total = std::accumulate(pt.begin(), pt.end(), 0.0);
  for (double &v : pt) {
    v /= total;
  }
  return pt;
}

} // namespace models
