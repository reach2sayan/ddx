// One model's batch Jacobian through the sweep, the JIT and the device, as the
// batch grows.  One process, so the machine's drift lands on every arm alike;
// bytes are reported beside points because the device's floor is the copy.
#include "rt/energy_models.hpp"
#include "rt/equation.hpp"
#include "util/ranges.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <format>
#include <ranges>
#include <string>
#include <vector>

namespace {

using ddx::rt::Backend;
using RE = ddx::rt::RTExpression<>;

constexpr std::size_t species = 8;

[[nodiscard]] auto model() {
  return ddx::rt::equation([] {
    return models::uniquac(std::views::iota(0uz, species) |
                           std::views::transform([](std::size_t i) {
                             return ddx::rt::var(std::format("x{}", i));
                           }) |
                           ddx::impl::to<std::vector<RE>>());
  });
}

void jacobian(benchmark::State &state, Backend backend) {
  const auto n = static_cast<std::size_t>(state.range(0));
  auto eq = model();
  eq.options({.backend = backend, .points = n});
  if (backend != Backend::Interpret && !eq.wait_for_kernel()) {
    const auto why = eq.device_status();
    state.SkipWithMessage(why && !*why ? why->error().detail
                                       : std::string{"no kernel landed"});
    return;
  }

  std::vector<std::vector<double>> x(species, std::vector<double>(n));
  for (const auto [j, column] : std::views::enumerate(x)) {
    for (const auto [i, v] : std::views::enumerate(column)) {
      v = 0.05 + 0.9 * static_cast<double>((i * 7 + j * 13) % 101) / 101.0;
    }
  }
  std::vector<std::vector<double>> f(1, std::vector<double>(n));
  std::vector<std::vector<double>> g(*eq.jacobian_columns(), std::vector<double>(n));
  const auto xs = x | std::views::transform([](auto &c) -> const double * { return c.data(); }) |
                  ddx::impl::to<std::vector<const double *>>();
  const auto out = [](auto &cs) {
    return cs | std::views::transform([](auto &c) { return c.data(); }) |
           ddx::impl::to<std::vector<double *>>();
  };
  const auto fs = out(f);
  const auto gs = out(g);

  for (auto _ : state) {
    auto done = eq.jacobian(xs, fs, gs, n);
    benchmark::DoNotOptimize(done);
    benchmark::ClobberMemory();
  }
  const auto points = static_cast<std::int64_t>(state.iterations()) *
                      static_cast<std::int64_t>(n);
  state.SetItemsProcessed(points);
  state.SetBytesProcessed(points * static_cast<std::int64_t>(
                                       (species + 1 + gs.size()) * sizeof(double)));
}

BENCHMARK_CAPTURE(jacobian, interpret, Backend::Interpret)
    ->RangeMultiplier(10)
    ->Range(1'000, 1'000'000)
    ->Unit(benchmark::kMillisecond);
#ifdef DDX_HAS_JIT
BENCHMARK_CAPTURE(jacobian, jit, Backend::Compile)
    ->RangeMultiplier(10)
    ->Range(1'000, 1'000'000)
    ->Unit(benchmark::kMillisecond);
#endif
BENCHMARK_CAPTURE(jacobian, device, Backend::Device)
    ->RangeMultiplier(10)
    ->Range(1'000, 1'000'000)
    ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
