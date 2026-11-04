// Temporary audit probe: census ratio and build time per Want, per model, vs n.
#include "rt/equation.hpp"
#include "rt/tensor.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
namespace {
namespace rt = ddx::rt;
using RE = rt::Expression<double>;
using W = rt::Want;
const std::pair<W, const char *> kWants[] = {
    {W::Value, "value"}, {W::Jacobian, "jac"}, {W::Gradient, "grad"},
    {W::Hessian, "hess"}, {W::Hvp, "hvp"}, {W::Vjp, "vjp"}, {W::Jvp, "jvp"},
    {W::JacobianJvp, "jjvp"}, {W::DenseHessian, "dense"}};

std::vector<RE> scalars(std::size_t n) {
  std::vector<RE> x;
  for (std::size_t i = 0; i < n; ++i) x.push_back(rt::var("x" + std::to_string(100000 + i)));
  return x;
}
using Maker = std::function<std::vector<RE>(std::size_t)>;
const std::pair<const char *, Maker> kModels[] = {
    {"scalar-chain", [](std::size_t n) {
       auto x = scalars(n); RE acc{0.0};
       for (std::size_t i = 0; i + 1 < n; ++i) acc = acc + x[i] * x[i + 1] + sin(x[i]);
       return std::vector{acc}; }},
    {"vec-index", [](std::size_t n) {
       auto x = rt::vec("x", n); RE acc{0.0};
       for (std::size_t i = 0; i + 1 < n; ++i) acc = acc + x[i] * x[i + 1] + sin(x[i]);
       return std::vector{acc}; }},
    {"vec-elementwise", [](std::size_t n) {
       auto x = rt::vec("x", n); auto acc = x;
       for (int i = 0; i < 20; ++i) acc = acc * 0.999 + sin(acc);
       return std::vector{sum(acc * x)}; }},
    {"vec-dot-gather", [](std::size_t n) {
       auto x = rt::vec("x", n); auto y = exp(x);
       RE acc = dot(y, x);
       for (std::size_t i = 0; i + 1 < n; ++i) acc = acc + y[i] * x[i + 1];
       return std::vector{acc}; }},
    {"sys-scalar", [](std::size_t n) {
       auto x = scalars(n); std::vector<RE> f;
       for (std::size_t i = 0; i + 1 < n; ++i) f.push_back(sin(x[i]) * x[i + 1]);
       return f; }},
    {"sys-vec", [](std::size_t n) {
       auto x = rt::vec("x", n); std::vector<RE> f;
       for (std::size_t i = 0; i + 1 < n; ++i) f.push_back(sin(x[i]) * x[i + 1]);
       return f; }},
};

TEST(ZZProbe, Audit) {
  for (const auto &[name, make] : kModels) {
    std::printf("%s\n", name);
    for (std::size_t n : {50uz, 100uz, 200uz, 400uz}) {
      std::printf("  n=%4zu", n);
      double base = 0;
      for (const auto &[w, label] : kWants) {
        const auto eq = rt::equation([&] { return make(n); });
        const auto t0 = std::chrono::steady_clock::now();
        const auto c = eq.census(w);
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!c) { std::printf(" %s:-", label); continue; }
        if (w == W::Value) base = double(c->values);
        std::printf(" %s:%.1fx/%.0fms", label, double(c->values) / base, ms);
      }
      std::printf("\n");
    }
  }
}
}
