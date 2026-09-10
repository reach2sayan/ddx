// Headers reached through the imported targets' include paths, and ddx::jit
// resolved as a real shared object.
#include "ddx.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <span>

#ifdef DDX_CONSUMER_RT
#include "rt/bridge.hpp"
#include "rt/equation.hpp"
#endif

#ifdef DDX_HAS_JIT
#include "jit/kernel.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"
#endif

#ifdef DDX_HAS_OPENCL
#include "cl/device.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"
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

  // f(x, y) = x*y + sin(x); df/dx = y + cos(x), df/dy = x.
  const auto x = var_of<"x">(1.25);
  const auto y = var_of<"y">(0.5);
  const auto eq = Equation(x * y + sin(x));

  const auto g = eq.jacobian(named<"x">(1.25), named<"y">(0.5));
  if (!close(g[0], 0.5 + std::cos(1.25)) || !close(g[1], 1.25)) {
    return fail("compile-time jacobian is wrong");
  }

#ifdef DDX_CONSUMER_RT
  // The runtime graph is the header set that needs Boost.
  rt::Builder<> b;
  const auto root = rt::to_graph(b, x * y + sin(x));
  const auto rg = rt::build_jacobian_impl(b, root.id(b));
  const auto values = rt::evaluate_all(b, std::array{1.25, 0.5});
  if (!close(values[rg.partial[0]], 0.5 + std::cos(1.25)) ||
      !close(values[rg.partial[1]], 1.25)) {
    return fail("runtime-graph jacobian is wrong");
  }
#endif

#ifdef DDX_HAS_JIT
  // The one call that has to resolve out of libddx_jit.so.
  auto compiler = jit::Compiler::create();
  if (!compiler) {
    return fail(compiler.error().detail.c_str());
  }
  auto kernel = compiler->compile(
      rt::GraphBuilder{b}.value(root).build_jacobian().finish());
  if (!kernel) {
    return fail(kernel.error().detail.c_str());
  }

  double xs0 = 1.25, xs1 = 0.5, out_f = 0, out_gx = 0, out_gy = 0;
  const double *const in[] = {&xs0, &xs1};
  double *const outs[] = {&out_f, &out_gx, &out_gy};
  (*kernel)(in, std::span{outs, 1}, std::span{outs + 1, 2}, {}, 1);
  if (!close(out_gx, 0.5 + std::cos(1.25)) || !close(out_gy, 1.25)) {
    return fail("JIT-compiled jacobian is wrong");
  }
#endif

#ifdef DDX_HAS_OPENCL
  // The symbols resolving is the claim; a host with no device has nothing to
  // run the kernel on, and a wrong answer from one that does is the failure.
  if (const auto device = cl::Device::create()) {
    const auto kernel = device->compile(
        rt::GraphBuilder{b}.value(root).build_jacobian().finish());
    if (!kernel) {
      return fail(kernel.error().detail.c_str());
    }
    double xs0 = 1.25, xs1 = 0.5, out_f = 0, out_gx = 0, out_gy = 0;
    const double *const in[] = {&xs0, &xs1};
    double *const outs[] = {&out_f, &out_gx, &out_gy};
    if (!(*kernel)(in, std::span{outs, 1}, std::span{outs + 1, 2}, {}, 1)) {
      return fail("the device refused a launch");
    }
    if (!close(out_gx, 0.5 + std::cos(1.25)) || !close(out_gy, 1.25)) {
      return fail("device-compiled jacobian is wrong");
    }
  }
#endif

  return 0;
}
