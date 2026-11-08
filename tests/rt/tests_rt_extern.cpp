#include "rt/equation.hpp"
#include "rt/text/equation.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <latch>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <vector>

// A function the program supplies with its own partials: ddx puts it in the
// graph and applies the chain rule around it, to any order its partials reach.
namespace {

using RE = ddx::rt::Expression<>;

// g(a, b) = exp(ab) sin(b) and its partials, written out by hand.
void g_of(std::span<const double> in, std::span<double> out) {
  const double a = in[0];
  const double b = in[1];
  const double e = std::exp(a * b);
  const double s = std::sin(b);
  const double c = std::cos(b);
  out[0] = e * s;
  out[1] = b * e * s;       // g_a
  out[2] = e * (a * s + c); // g_b
  if (out.size() > 3) {
    out[3] = b * b * e * s;                   // g_aa
    out[4] = e * (s + a * b * s + b * c);     // g_ab
    out[5] = e * (a * a * s + 2 * a * c - s); // g_bb
  }
}

[[nodiscard]] auto native() {
  return ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    return exp(a * b) * sin(b);
  });
}

[[nodiscard]] auto supplied(std::uint32_t order = 2) {
  return ddx::rt::equation([order] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    const auto g =
        ddx::rt::extern_fn({.name = "g", .arity = 2, .order = order}, g_of);
    return g(a, b);
  });
}

constexpr std::array<std::array<double, 2>, 5> points{
    {{0.3, 1.1}, {-0.7, 0.4}, {1.2, -2.0}, {0.0, 0.5}, {0.9, 3.0}}};

void expect_close(std::span<const double> got, std::span<const double> want,
                  double tol) {
  ASSERT_EQ(got.size(), want.size());
  for (const auto [k, g] : got | std::views::enumerate) {
    const double w = want[static_cast<std::size_t>(k)];
    EXPECT_NEAR(g, w, tol * std::max(1.0, std::abs(w))) << "cell " << k;
  }
}

TEST(RtExtern, AgreesWithTheSameFunctionWrittenOut) {
  const auto ref = native();
  const auto eq = supplied();
  ASSERT_FALSE(eq.poisoned()) << eq.status()->code;
  for (const auto &at : points) {
    EXPECT_NEAR(*eq.evaluate(at), *ref.evaluate(at), 1e-13);
    expect_close(*eq.jacobian(at), *ref.jacobian(at), 1e-13);
    expect_close(*eq.hessian(at), *ref.hessian(at), 1e-13);
    expect_close(*eq.hvp("b", at), *ref.hvp("b", at), 1e-13);
    expect_close(*eq.jacobian_jvp("a", at), *ref.jacobian_jvp("a", at), 1e-13);
  }
}

TEST(RtExtern, TheChainRuleRunsAroundIt) {
  const auto ref = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    const auto g = exp(a * b) * sin(b);
    return log(1.0 + g * g) * a;
  });
  const auto eq = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    const auto g = ddx::rt::extern_fn({.name = "g", .arity = 2}, g_of)(a, b);
    return log(1.0 + g * g) * a;
  });
  for (const auto &at : points) {
    expect_close(*eq.jacobian(at), *ref.jacobian(at), 1e-12);
    expect_close(*eq.hessian(at), *ref.hessian(at), 1e-12);
  }
}

// f(x, y, z, w) = x y + z w^2 as a function of four, called with w = x.
TEST(RtExtern, AnyArityAndARepeatedArgument) {
  const auto f = [](std::span<const double> in, std::span<double> out) {
    const double x = in[0], y = in[1], z = in[2], w = in[3];
    out[0] = x * y + z * w * w;
    // First partials, in the order x, y, z, w.
    out[1] = y;
    out[2] = x;
    out[3] = w * w;
    out[4] = 2 * z * w;
    // Second partials fill the upper triangle in row-major order.
    std::ranges::fill(out.subspan(5), 0.0);
    out[5 + 1] = 1.0;   // xy
    out[5 + 8] = 2 * w; // zw
    out[5 + 9] = 2 * z; // ww
  };
  const auto eq = ddx::rt::equation([&f] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    return ddx::rt::extern_fn({.name = "f", .arity = 4}, f)(x, y, z, x);
  });
  const auto ref = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    const auto z = ddx::rt::var("z");
    return x * y + z * x * x;
  });
  const std::array at{0.7, -1.3, 2.1};
  EXPECT_DOUBLE_EQ(*eq.evaluate(at), *ref.evaluate(at));
  expect_close(*eq.jacobian(at), *ref.jacobian(at), 1e-14);
  expect_close(*eq.hessian(at), *ref.hessian(at), 1e-14);
}

TEST(RtExtern, FirstPartialsOnlyRefuseSecondOrder) {
  const auto ref = native();
  const auto eq = supplied(1);
  const auto &at = points[0];
  expect_close(*eq.jacobian(at), *ref.jacobian(at), 1e-13);
  const std::array v{1.0, 0.0};
  expect_close(*eq.jvp(std::span<const double>{v}, at),
               *ref.jvp(std::span<const double>{v}, at), 1e-13);
  EXPECT_EQ(eq.hessian(at).error().code, ddx::errc::extern_order);
  EXPECT_EQ(eq.hvp("a", at).error().code, ddx::errc::extern_order);
  EXPECT_EQ(eq.jacobian_jvp("a", at).error().code, ddx::errc::extern_order);
  EXPECT_EQ(eq.prepare({ddx::rt::Want::Hessian}).error().code,
            ddx::errc::extern_order);
}

TEST(RtExtern, AMisshapenRegistrationOrCallIsRefused) {
  const auto wrong_count = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    return ddx::rt::extern_fn({.name = "g", .arity = 2}, g_of)(a);
  });
  EXPECT_EQ(wrong_count.status()->code, ddx::errc::wrong_argument_count);
  const auto third = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    return ddx::rt::extern_fn({.name = "g", .arity = 1, .order = 3}, g_of)(a);
  });
  EXPECT_EQ(third.status()->code, ddx::errc::extern_shape);
  // At arity 400, 1 + 400 + 80200 components overflow a slot's half-word count.
  const auto wide = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    return ddx::rt::extern_fn({.name = "g", .arity = 400, .order = 2}, g_of)(a);
  });
  EXPECT_EQ(wide.status()->code, ddx::errc::extern_shape);
  for (const auto *text : {"extern(x)", "extern_d(x)", "extern_args(x, y)"}) {
    EXPECT_EQ(ddx::rt::equation(text).error().code, ddx::errc::unknown_function)
        << text;
  }
}

TEST(RtExtern, ABatchMatchesThePointCalls) {
  const auto eq = supplied();
  constexpr std::size_t n = 1000;
  std::vector<double> as(n);
  std::vector<double> bs(n);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    as[i] = -1.0 + 2.0 * static_cast<double>(i) / n;
    bs[i] = 0.5 + std::sin(static_cast<double>(i));
  }
  std::vector<double> f(n);
  std::vector<double> ga(n);
  std::vector<double> gb(n);
  const std::array<const double *, 2> in{as.data(), bs.data()};
  const std::array<double *, 1> fs{f.data()};
  const std::array<double *, 2> gs{ga.data(), gb.data()};
  ASSERT_TRUE(eq.jacobian(in, fs, gs, n).has_value());
  for (const std::size_t i : std::views::iota(0uz, n)) {
    const auto J = *eq.jacobian(as[i], bs[i]);
    EXPECT_EQ(f[i], *eq.evaluate(as[i], bs[i])) << i;
    EXPECT_EQ(ga[i], J[0]) << i;
    EXPECT_EQ(gb[i], J[1]) << i;
  }
}

// Every sweeping thread calls the function at once, so it only has to be
// re-entrant, and each thread must get the single-thread answers.
TEST(RtExtern, CalledFromManyThreads) {
  std::atomic<std::size_t> calls{0};
  const auto eq = ddx::rt::equation([&calls] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    return ddx::rt::extern_fn(
        {.name = "g", .arity = 2},
        [&calls](std::span<const double> in, std::span<double> out) {
          calls.fetch_add(1, std::memory_order_relaxed);
          g_of(in, out);
        })(a, b);
  });
  ASSERT_TRUE(eq.prepare({ddx::rt::Want::Jacobian}).has_value());
  constexpr std::size_t threads = 4;
  std::latch start{threads};
  std::vector<std::vector<double>> got(threads);
  {
    std::vector<std::jthread> pool;
    for (const std::size_t t : std::views::iota(0uz, threads)) {
      pool.emplace_back([&, t] {
        start.arrive_and_wait();
        for (int rep = 0; rep < 200; ++rep) {
          got[t] = *eq.jacobian(0.1 * static_cast<double>(t), 1.0);
        }
      });
    }
  }
  for (const std::size_t t : std::views::iota(0uz, threads)) {
    EXPECT_EQ(got[t], *eq.jacobian(0.1 * static_cast<double>(t), 1.0));
  }
  EXPECT_GT(calls.load(), 0u);
}

TEST(RtExtern, ASystemThroughOneFunction) {
  const auto sys = ddx::rt::equation([] {
    const auto a = ddx::rt::var("a");
    const auto b = ddx::rt::var("b");
    const auto g = ddx::rt::extern_fn({.name = "g", .arity = 2}, g_of)(a, b);
    return std::vector<RE>{g, g * a};
  });
  const auto ref = native();
  const auto &at = points[2];
  const auto J = *sys.jacobian(at);
  const auto j = *ref.jacobian(at);
  EXPECT_NEAR(J[0], j[0], 1e-13);
  EXPECT_NEAR(J[1], j[1], 1e-13);
}

// The inner equation shares the outer sweep's scalar type and thread-local
// buffers, and is larger, so it would grow them during the outer sweep.
void expect_reentry_matches(auto inner_model, auto native_model,
                            auto outer_model, std::size_t symbols,
                            ddx::rt::Backend backend) {
  auto inner = ddx::rt::equation(inner_model);
  inner.options({.backend = backend});
  auto native = ddx::rt::equation(native_model);
  auto eq = ddx::rt::equation([&] {
    const auto g = ddx::rt::extern_fn(
        {.name = "nested", .arity = 2, .order = 1},
        [&inner](std::span<const double> in, std::span<double> out) {
          out[0] = *inner.evaluate(in);
          const auto partials = *inner.jacobian(in);
          out[1] = partials[0];
          out[2] = partials[1];
        });
    return outer_model(g);
  });
  ASSERT_FALSE(eq.poisoned());
  eq.options({.backend = backend});

  constexpr std::size_t n = 37;
  const auto column = [](std::size_t j, std::size_t p) {
    const auto phase = static_cast<double>(j * 7 + p);
    return 0.25 + 0.5 * std::abs(std::sin(1.0 + phase));
  };
  for (const std::size_t p : std::views::iota(0uz, 3uz)) {
    const auto at = std::views::iota(0uz, symbols) |
                    std::views::transform(
                        [&](std::size_t j) { return column(j, p); }) |
                    ddx::impl::to<std::vector>();
    const std::span<const double> point{at};
    EXPECT_NEAR(*eq.evaluate(point), *native.evaluate(point), 1e-10);
    expect_close(*eq.jacobian(point), *native.jacobian(point), 1e-10);
  }
  for (const std::size_t count : {3uz, n}) {
    std::vector<std::vector<double>> columns(symbols,
                                             std::vector<double>(count));
    std::vector<std::vector<double>> partials(symbols,
                                              std::vector<double>(count));
    std::vector<std::vector<double>> want = partials;
    std::vector<double> f(count);
    std::vector<double> want_f(count);
    std::vector<const double *> xs;
    std::vector<double *> gs;
    std::vector<double *> want_gs;
    for (const std::size_t j : std::views::iota(0uz, symbols)) {
      for (const std::size_t p : std::views::iota(0uz, count)) {
        columns[j][p] = column(j, p);
      }
      xs.push_back(columns[j].data());
      gs.push_back(partials[j].data());
      want_gs.push_back(want[j].data());
    }
    const std::array<double *, 1> fs{f.data()};
    const std::array<double *, 1> want_fs{want_f.data()};
    ASSERT_TRUE(eq.jacobian(xs, fs, gs, count).has_value());
    ASSERT_TRUE(native.jacobian(xs, want_fs, want_gs, count).has_value());
    expect_close(f, want_f, 1e-10);
    for (const std::size_t j : std::views::iota(0uz, symbols)) {
      expect_close(partials[j], want[j], 1e-10);
    }
  }
}

[[nodiscard]] RE nested_terms(const RE &a, const RE &b) {
  RE acc{0};
  for (const int k : std::views::iota(0, 96)) {
    const double c = 1.0 + 0.01 * k;
    acc += sin(a * c) * b + exp(b * c * 0.1);
  }
  return acc;
}

[[nodiscard]] std::vector<ddx::rt::Backend> reentry_backends() {
#ifdef DDX_HAS_JIT
  return {ddx::rt::Backend::Interpret, ddx::rt::Backend::Compile};
#else
  return {ddx::rt::Backend::Interpret};
#endif
}

TEST(RtExtern, AFunctionMaySweepAnotherEquation) {
  for (const ddx::rt::Backend backend : reentry_backends()) {
    expect_reentry_matches(
        [] {
          const auto a = ddx::rt::var("a");
          const auto b = ddx::rt::var("b");
          return nested_terms(a, b);
        },
        [] {
          const auto a = ddx::rt::var("a");
          const auto b = ddx::rt::var("b");
          return nested_terms(a, b) * a + log(a + b);
        },
        [](const auto &g) {
          const auto a = ddx::rt::var("a");
          const auto b = ddx::rt::var("b");
          return g(a, b) * a + log(a + b);
        },
        2, backend);
  }
}

TEST(RtExtern, AFunctionMaySweepAnotherTensorEquation) {
  const auto tensor_terms = [](const RE &v) {
    return nested_terms(v[0], v[1]) + sum(v * log(v)) + dot(v, v * v);
  };
  for (const ddx::rt::Backend backend : reentry_backends()) {
    expect_reentry_matches(
        [=] { return tensor_terms(ddx::rt::vec("v", 2)); },
        [=] {
          const auto x = ddx::rt::vec("x", 3);
          return tensor_terms(ddx::rt::pack(std::vector{x[0], x[1]})) *
                     dot(x, x) +
                 sum(x * x);
        },
        [](const auto &g) {
          const auto x = ddx::rt::vec("x", 3);
          return g(x[0], x[1]) * dot(x, x) + sum(x * x);
        },
        3, backend);
  }
}

} // namespace
