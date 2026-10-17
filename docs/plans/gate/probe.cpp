// Bit gate: every Want on every model at one point, as hex bits, plus the
#include "energy_models.hpp"
#include "rt/equation.hpp"
#include "rt/tensor.hpp"
#include <cstdio>
#include <format>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>

using RE = ddx::rt::RTExpression<>;
#ifdef DDX_ONE_HANDLE
using RV = RE;
#else
using RV = ddx::rt::RTVector<>;
#endif

std::vector<RE> symbols(std::size_t n) {
  std::vector<RE> x;
  for (const std::size_t i : std::views::iota(0uz, n)) x.push_back(ddx::rt::var(std::format("x[{}]", i)));
  return x;
}
std::vector<double> composition(std::size_t n) {
  std::vector<double> pt(n);
  for (std::size_t i = 0; i < n; ++i) pt[i] = (1.0 + 0.3 * static_cast<double>(i % 5)) / static_cast<double>(n);
  const double total = std::accumulate(pt.begin(), pt.end(), 0.0);
  for (double &v : pt) v /= total;
  return pt;
}
std::vector<double> direction(std::size_t n) {
  std::vector<double> d(n);
  for (std::size_t i = 0; i < n; ++i) d[i] = std::sin(1.0 + static_cast<double>(i));
  return d;
}
RE uniquac_tensor(const RV &x) {
  const std::size_t n = x.size();
  std::vector<double> r(n), q(n), tau(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    r[i] = models::pseudo(i, 1, 0.9, 5.2);
    q[i] = models::pseudo(i, 2, 0.8, 4.4);
    for (std::size_t j = 0; j < n; ++j) tau[j + i * n] = models::pseudo(j, i, 0.2, 1.8);
  }
  const RV rv = ddx::rt::vec_const(r);
  const RV qv = ddx::rt::vec_const(q);
  const RE sum_r = dot(rv, x);
  const RE sum_q = dot(qv, x);
  const RV phi = (rv * x) / sum_r;
  const RV theta = (qv * x) / sum_q;
  const RE comb = sum(x * log(phi / x)) + 5.0 * sum(qv * x * log(theta / phi));
  const RV inner = gemv_t(ddx::rt::mat_const(n, n, tau), theta);
  const RE res = -sum(qv * x * log(inner));
  return comb + res;
}
RE mse_tensor(const RV &x) {
  const std::size_t n = x.size();
  std::vector<double> half_zc2(n), p(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    const double zc = models::pseudo(i, 5, -2.0, 2.0);
    half_zc2[i] = 0.5 * zc * zc;
    for (std::size_t j = 0; j < n; ++j) p[i + j * n] = models::pseudo(i, j, -0.4, 0.4);
  }
  const RE ionic = dot(ddx::rt::vec_const(half_zc2), x);
  const RE lr = -0.3915 * sqrt(ionic) / (RE{1} + 1.2 * sqrt(ionic));
  const RE mr = dot(x, gemv(ddx::rt::mat_const(n, n, p), x)) * exp(-sqrt(ionic));
  return lr + mr + uniquac_tensor(x);
}

void dump(const char *what, const auto &r) {
  std::printf("  %s:", what);
  if (!r) { std::printf(" err %d\n", static_cast<int>(r.error().code)); return; }
  if constexpr (std::is_same_v<std::decay_t<decltype(*r)>, double>) std::printf(" %a", *r);
  else for (double v : *r) std::printf(" %a", v);
  std::printf("\n");
}

void run(const char *name, auto &&eq, std::size_t n) {
  std::printf("%s\n", name);
  const auto at = composition(n);
  const auto dir = direction(n);
  const std::span<const double> x{at};
  dump("evaluate", eq.evaluate(x));
  dump("jacobian", eq.jacobian(x));
  dump("gradient", eq.gradient(x));
  dump("hessian", eq.hessian(x));
  dump("hvp", eq.hvp(std::span<const double>{dir}, x));
  dump("jvp", eq.jvp(std::span<const double>{dir}, x));
  const std::vector<double> w(1, 0.7);
  dump("vjp", eq.vjp(std::span<const double>{w}, x));
  dump("jacobian_jvp", eq.jacobian_jvp(std::span<const double>{dir}, x));
}

int main() {
  constexpr std::size_t n = 6;
  run("uniquac", ddx::rt::equation([] { return models::uniquac(symbols(n)); }), n);
  run("mse", ddx::rt::equation([] { return models::mse(symbols(n)); }), n);
  run("cluster", ddx::rt::equation([] { return models::cluster_expansion(symbols(n)); }), n);
  run("chain", ddx::rt::equation([] { return models::bonded_chain(symbols(n)); }), n);
  run("uniquac_tensor", ddx::rt::equation([] { return uniquac_tensor(ddx::rt::vec("x", n)); }), n);
  run("mse_tensor", ddx::rt::equation([] { return mse_tensor(ddx::rt::vec("x", n)); }), n);
  run("mixed", ddx::rt::equation([] { const RV x = ddx::rt::vec("x", n); const RE s = sum(exp(-x) * x) / dot(x, x);
      return s * s + sum(pow(x, 2.0)) + sum(pow(x, x)) + sum(max(x, 0.1)) + sum(abs(x - 0.2)) + x[2] * sum(add_at(x, x[0], 1)); }), n);
  return 0;
}
