// Abandons background compiles mid-flight from many threads, then returns from
// main.  The assertion is the exit status: LLVM registers its atexit entries
// while it compiles, and a pool destroyed after them runs its workers against
// freed state -- a fault or a hang at exit, which no in-process test can see.
// Not a gtest: the thing under test happens after every test has returned.

#include "jit/kernel.hpp"
#include "rt/equation.hpp"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

// Wide enough that a compile takes milliseconds, so a round drops its equations
// while the pool is still inside LLVM.  Each equation owns its arena: the
// Hessian sweep appends, so two threads may not share one Builder.
auto model(std::size_t n) {
  return ddx::rt::equation([n] {
    std::vector<ddx::rt::Expression<double>> v;
    for (std::size_t i = 0; i < n; ++i) {
      v.push_back(ddx::rt::var("x" + std::to_string(i)));
    }
    auto f = v[0] * log(v[0]);
    for (std::size_t i = 1; i < n; ++i) {
      f = f + v[i] * log(v[i]) + exp(v[i - 1] * v[i]);
    }
    return f;
  });
}

} // namespace

int main() {
  constexpr std::size_t rounds = 8;
  constexpr std::size_t threads = 16;

  for (std::size_t round = 0; round < rounds; ++round) {
    std::vector<std::thread> racers;
    for (std::size_t t = 0; t < threads; ++t) {
      racers.emplace_back([] {
        auto eq = model(48);
        eq.options({.backend = ddx::rt::Backend::Compile});
        (void)eq.uses_kernel(); // launches the compile; the equation dies now
      });
    }
    for (auto &r : racers) {
      r.join();
    }
  }
  std::puts("exit_stress: compiles abandoned, exiting");
  return 0;
}
