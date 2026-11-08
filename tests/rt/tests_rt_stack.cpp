// Runs every call over a thousand symbols on a 1 MiB stack, the Windows
// default, so nothing may recurse once per symbol or per term.

#include "rt/equation.hpp"
#include "rt/tensor.hpp"

#include <gtest/gtest.h>

#if __has_include(<pthread.h>)
#include <pthread.h>
#define DDX_TEST_SMALL_STACK 1
#endif

#include <cstddef>
#include <ranges>
#include <string>
#include <vector>

namespace {

// Runs body on a thread with a 1 MiB stack, which std::thread cannot size.
void on_small_stack([[maybe_unused]] auto body) {
#ifndef DDX_TEST_SMALL_STACK
  GTEST_SKIP() << "no pthreads to size a stack with";
#else
  pthread_attr_t attr;
  ASSERT_EQ(pthread_attr_init(&attr), 0);
  ASSERT_EQ(pthread_attr_setstacksize(&attr, 1uz << 20), 0);
  pthread_t thread;
  ASSERT_EQ(pthread_create(
                &thread, &attr,
                [](void *p) -> void * {
                  (*static_cast<decltype(body) *>(p))();
                  return nullptr;
                },
                &body),
            0);
  pthread_join(thread, nullptr);
  pthread_attr_destroy(&attr);
#endif
}

constexpr std::size_t kSymbols = 1000;
constexpr std::size_t kDepth = 1000;

TEST(RtStack, AThousandChainedSymbolsOnAOneMegabyteStack) {
  on_small_stack([] {
    const auto eq = ddx::rt::equation([] {
      const auto x = std::views::iota(0uz, kSymbols) |
                     std::views::transform([](std::size_t i) {
                       return ddx::rt::var("x" + std::to_string(10000 + i));
                     }) |
                     ddx::impl::to<std::vector>();
      // Built the way a loop builds it, one term deeper each time.
      ddx::rt::Expression<double> acc{0.0};
      for (const auto [a, b] : x | std::views::pairwise) {
        acc = acc + a * b + sin(a);
      }
      return acc;
    });
    ASSERT_FALSE(eq.poisoned());
    const std::vector<double> at(kSymbols, 0.5);
    const std::span<const double> p{at};
    EXPECT_TRUE(eq.evaluate(p).has_value());
    EXPECT_TRUE(eq.gradient(p).has_value());
    EXPECT_TRUE(eq.hessian(p).has_value());
    EXPECT_TRUE(eq.hvp(p, p).has_value());
    EXPECT_TRUE(eq.census(ddx::rt::Want::Hessian).has_value());
  });
}

TEST(RtStack, AVectorChainDeepOnAOneMegabyteStack) {
  on_small_stack([] {
    const auto eq = ddx::rt::equation([] {
      const auto x = ddx::rt::vec("x", 8);
      auto acc = x;
      for ([[maybe_unused]] const std::size_t i :
           std::views::iota(0uz, kDepth)) {
        acc = acc * 0.999 + sin(acc);
      }
      return sum(acc * x);
    });
    ASSERT_FALSE(eq.poisoned());
    const std::vector<double> at(8, 0.5);
    const std::span<const double> p{at};
    EXPECT_TRUE(eq.gradient(p).has_value());
    EXPECT_TRUE(eq.hvp(p, p).has_value());
  });
}

} // namespace
