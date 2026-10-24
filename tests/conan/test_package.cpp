// The README's first example, through the package Conan installed.
#include "ddx.hpp"
#include "rt/equation.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  using namespace ddx;

  struct Sample { double t, y; };
  const std::vector<Sample> data{{0.0, 2.00}, {1.0, 1.21}, {2.0, 0.74}, {3.0, 0.45}};
  const double delta = 0.05;
  const auto eq = rt::equation([&] {
    const auto a = rt::var("a");
    const auto b = rt::var("b");
    rt::Expression<double> loss = 0.0;
    for (const auto &s : data) {
      const auto r = s.y - a * exp(-b * s.t);
      loss += select(abs(r) < delta, r * r, delta * (2.0 * abs(r) - delta));
    }
    return loss;
  });

  const auto value = eq.evaluate(2.0, 0.5);
  if (!value || std::fabs(*value - 4.1343959887e-05) > 1e-12) {
    std::fprintf(stderr, "test_package: evaluate is wrong\n");
    return 1;
  }
  std::printf("ddx %.*s: f = %.10e\n", static_cast<int>(version.size()), version.data(), *value);
  return 0;
}
