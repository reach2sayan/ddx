// Headers come through the imported targets' include paths, and ddx::jit
// resolves as a real shared object.
#include "ddx.hpp"

#include <cmath>
#include <cstdio>

#ifdef DDX_CONSUMER_RT
#include "rt/equation.hpp"
#endif

namespace {

bool close(double a, double b) { return std::fabs(a - b) < 1e-12; }

int fail(const char *what) {
  std::fprintf(stderr, "consumer: %s\n", what);
  return 1;
}

} // namespace

int main() {
  using namespace ddx;

  const auto x = var_of<"x">(1.25);
  const auto y = var_of<"y">(0.5);
  const auto eq = Equation(x * y + sin(x));

  const auto g = eq.jacobian(named<"x">(1.25), named<"y">(0.5));
  if (!close(g[0], 0.5 + std::cos(1.25)) || !close(g[1], 1.25)) {
    return fail("compile-time jacobian is wrong");
  }

#ifdef DDX_CONSUMER_RT
  // The runtime equation is the header set that needs Boost.
  auto f = rt::equation([] {
    const auto rx = rt::var("x");
    const auto ry = rt::var("y");
    return rx * ry + sin(rx);
  });
  const auto rg = f.jacobian(1.25, 0.5);
  if (!rg || !close((*rg)[0], 0.5 + std::cos(1.25)) || !close((*rg)[1], 1.25)) {
    return fail("runtime jacobian is wrong");
  }
#endif

#ifdef DDX_HAS_JIT
  // The compile must resolve out of the installed library.
  f.options({.backend = rt::Backend::Compile});
  if (const auto status = f.compile_status(); !status) {
    return fail(status.error().detail.c_str());
  }
  const auto compiled = f.jacobian(1.25, 0.5);
  if (!compiled || *compiled != *rg) {
    return fail("JIT-compiled jacobian is wrong");
  }
#endif

  return 0;
}
