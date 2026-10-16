#include "energy_models.hpp"
#include "rt/archive/archive.hpp"
#include "rt/equation.hpp"
#include "rt/kernels.hpp"
#include "rt/tensor.hpp"
#include "scratch.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <latch>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <vector>

// Tensor nodes against the scalar graph.  The same model written twice -- once
// as scalars, once over vectors -- must agree on every derivative, and the
// scalar graph is the oracle: its rules are tested elsewhere.  Nothing here is
// bit-exact across the two spellings, since a Dot sums in another order; the
// bit-exactness asked for is between one spelling and itself -- batches,
// threads, a file.
namespace {

using RE = ddx::rt::RTExpression<>;
using RV = ddx::rt::RTVector<>;

static_assert(ddx::impl::Numeric<RV>);
static_assert(ddx::rt::detail::CGraphHandle<RV>);

// The symbols as vec() names them, so both spellings share one point layout.
std::vector<RE> symbols(std::size_t n) {
  std::vector<RE> x;
  for (const std::size_t i : std::views::iota(0uz, n)) {
    x.push_back(ddx::rt::var(std::format("x[{}]", i)));
  }
  return x;
}

std::vector<double> composition(std::size_t n) {
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
std::vector<double> direction(std::size_t n) {
  std::vector<double> d(n);
  for (std::size_t i = 0; i < n; ++i) {
    d[i] = std::sin(1.0 + static_cast<double>(i));
  }
  return d;
}

// models::uniquac over vectors: phi and theta are elementwise, the inner sum
// over tau is one transposed Gemv.
RE uniquac_tensor(const RV &x) {
  const std::size_t n = x.size();
  std::vector<double> r(n), q(n), tau(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    r[i] = models::pseudo(i, 1, 0.9, 5.2);
    q[i] = models::pseudo(i, 2, 0.8, 4.4);
    for (std::size_t j = 0; j < n; ++j) {
      tau[j + i * n] = models::pseudo(j, i, 0.2, 1.8); // tau_ji at (j, i)
    }
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

// models::mse over vectors.
RE mse_tensor(const RV &x) {
  const std::size_t n = x.size();
  std::vector<double> half_zc2(n), p(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    const double zc = models::pseudo(i, 5, -2.0, 2.0);
    half_zc2[i] = 0.5 * zc * zc;
    for (std::size_t j = 0; j < n; ++j) {
      p[i + j * n] = models::pseudo(i, j, -0.4, 0.4);
    }
  }
  const RE ionic = dot(ddx::rt::vec_const(half_zc2), x);
  const RE lr = -0.3915 * sqrt(ionic) / (RE{1} + 1.2 * sqrt(ionic));
  const RE mr =
      dot(x, gemv(ddx::rt::mat_const(n, n, p), x)) * exp(-sqrt(ionic));
  return lr + mr + uniquac_tensor(x);
}

auto scalar_mse(std::size_t n) {
  return ddx::rt::equation([n] { return models::mse(symbols(n)); });
}
auto tensor_mse(std::size_t n) {
  return ddx::rt::equation([n] { return mse_tensor(ddx::rt::vec("x", n)); });
}

void expect_close(std::span<const double> want, std::span<const double> got,
                  double tol, const char *what) {
  ASSERT_EQ(want.size(), got.size()) << what;
  for (const auto [i, pair] : std::views::zip(want, got) | std::views::enumerate) {
    const auto [w, g] = pair;
    const double scale = std::max({1.0, std::abs(w), std::abs(g)});
    EXPECT_LE(std::abs(w - g), tol * scale) << what << " at " << i;
  }
}
void expect_close(double want, double got, double tol, const char *what) {
  expect_close(std::span{&want, 1}, std::span{&got, 1}, tol, what);
}

// Every derivative the equation answers, both spellings, one point.
void expect_same_derivatives(const auto &want, const auto &got,
                             std::span<const double> at, double tol = 1e-12) {
  ASSERT_FALSE(want.poisoned());
  ASSERT_FALSE(got.poisoned()) << got.status()->code;
  ASSERT_TRUE(std::ranges::equal(*want.symbols(), *got.symbols()));
  const auto d = direction(at.size());
  expect_close(*want.evaluate(at), *got.evaluate(at), tol, "value");
  expect_close(*want.gradient(at), *got.gradient(at), tol, "gradient");
  expect_close(*want.jacobian(at), *got.jacobian(at), tol, "jacobian");
  expect_close(*want.hvp(d, at), *got.hvp(d, at), tol, "hvp");
  expect_close(*want.jacobian_jvp(d, at), *got.jacobian_jvp(d, at), tol,
               "jacobian_jvp");
  expect_close(*want.jvp(d, at), *got.jvp(d, at), tol, "jvp");
  const std::array one{1.0};
  expect_close(*want.vjp(one, at), *got.vjp(one, at), tol, "vjp");
  expect_close(*want.hessian(at), *got.hessian(at), tol, "hessian");
}

} // namespace

TEST(RtTensor, MseMatchesTheScalarGraph) {
  for (const std::size_t n : {5uz, 20uz, 50uz}) {
    const auto at = composition(n);
    expect_same_derivatives(scalar_mse(n), tensor_mse(n), at, 1e-11);
  }
}

// Each node alone, against the same arithmetic spelled over scalars.
namespace {
template <typename Scalar, typename Tensor>
void each_node(std::size_t n, Scalar scalar, Tensor tensor, double tol = 1e-12) {
  const auto want = ddx::rt::equation([&] { return scalar(symbols(n)); });
  const auto got =
      ddx::rt::equation([&] { return tensor(ddx::rt::vec("x", n)); });
  expect_same_derivatives(want, got, composition(n), tol);
}
std::vector<double> matrix(std::size_t rows, std::size_t cols) {
  std::vector<double> m(rows * cols);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      m[i + j * rows] = models::pseudo(i, j, -1.0, 1.0);
    }
  }
  return m;
}
} // namespace

TEST(RtTensor, DotSumAndScale) {
  each_node(
      6,
      [](const std::vector<RE> &x) {
        RE d{0}, s{0};
        for (std::size_t i = 0; i < x.size(); ++i) {
          d += x[i] * x[(i + 1) % x.size()];
          s += 3.0 * x[0] * x[i];
        }
        return d * s;
      },
      [](const RV &x) {
        std::vector<RE> rolled;
        for (std::size_t i = 0; i < x.size(); ++i) {
          rolled.push_back(x[(i + 1) % x.size()]);
        }
        return dot(x, ddx::rt::pack(rolled)) * sum((3.0 * x[0]) * x);
      });
}

TEST(RtTensor, MapAndZip) {
  each_node(
      5,
      [](const std::vector<RE> &x) {
        RE acc{0};
        for (std::size_t i = 0; i < x.size(); ++i) {
          acc += log(x[i]) / (x[i] + 1.0) + exp(x[i]) * sqrt(x[i]) -
                 pow(x[i], 1.5) + abs(x[i] - 0.3);
        }
        return acc;
      },
      [](const RV &x) {
        return sum(log(x) / (x + 1.0) + exp(x) * sqrt(x) - pow(x, 1.5) +
                   abs(x - 0.3));
      });
}

TEST(RtTensor, GemvBothWays) {
  constexpr std::size_t rows = 4, cols = 6;
  const auto m = matrix(rows, cols);
  each_node(
      cols,
      [&](const std::vector<RE> &x) {
        // y = M x, then sum_i y_i^2 ; z = M^T (first rows of x), dot with x
        RE acc{0};
        for (std::size_t i = 0; i < rows; ++i) {
          RE y{0};
          for (std::size_t j = 0; j < cols; ++j) {
            y += m[i + j * rows] * x[j];
          }
          acc += y * y;
        }
        for (std::size_t j = 0; j < cols; ++j) {
          RE z{0};
          for (std::size_t i = 0; i < rows; ++i) {
            z += m[i + j * rows] * x[i];
          }
          acc += z * x[j];
        }
        return acc;
      },
      [&](const RV &x) {
        const auto M = ddx::rt::mat_const(rows, cols, m);
        std::vector<RE> head;
        for (std::size_t i = 0; i < rows; ++i) {
          head.push_back(x[i]);
        }
        const RV y = gemv(M, x);
        return sum(y * y) + dot(gemv_t(M, ddx::rt::pack(head)), x);
      });
}

TEST(RtTensor, GatherAndAddAtRoundTrip) {
  each_node(
      5,
      [](const std::vector<RE> &x) {
        // v = x with v[2] += x[0]*x[1]; then sum v*v and read v[2] back
        std::vector<RE> v = x;
        v[2] = v[2] + x[0] * x[1];
        RE acc{0};
        for (const auto &e : v) {
          acc += e * e;
        }
        return acc + v[2] * x[4];
      },
      [](const RV &x) {
        const RV v = add_at(x, x[0] * x[1], 2);
        return sum(v * v) + v[2] * x[4];
      });
}

TEST(RtTensor, AScalarSelectFeedsAScale) {
  each_node(
      4,
      [](const std::vector<RE> &x) {
        const RE k = select(x[0] < 0.2, x[1], x[2] * x[2]);
        RE acc{0};
        for (const auto &e : x) {
          acc += k * e * e;
        }
        return acc;
      },
      [](const RV &x) {
        const RE k = select(x[0] < 0.2, x[1], x[2] * x[2]);
        return sum(k * (x * x));
      });
}

TEST(RtTensor, PendingLiteralsTakeTheVectorsWidth) {
  each_node(
      3,
      [](const std::vector<RE> &x) {
        RE acc{0};
        for (const auto &e : x) {
          acc += (2.0 * e + 1.0) / (e - 0.5) * 3.0;
        }
        return acc;
      },
      [](const RV &x) { return sum((2.0 * x + 1.0) / (x - 0.5) * 3.0); });
}

TEST(RtTensor, RefusesMismatchedWidthsAndVectorRoots) {
  const auto mismatched = ddx::rt::equation(
      [] { return dot(ddx::rt::vec("a", 2), ddx::rt::vec("b", 3)); });
  EXPECT_TRUE(mismatched.poisoned());
  EXPECT_EQ(mismatched.status()->code, ddx::errc::tensor_shape);

  const auto out_of_range =
      ddx::rt::equation([] { return ddx::rt::vec("a", 2)[2]; });
  EXPECT_EQ(out_of_range.status()->code, ddx::errc::tensor_shape);

  const auto gemv_shape = ddx::rt::equation([] {
    const std::vector<double> m(6, 1.0);
    return sum(gemv(ddx::rt::mat_const(2, 3, m), ddx::rt::vec("a", 2)));
  });
  EXPECT_EQ(gemv_shape.status()->code, ddx::errc::tensor_shape);

  // A vector where an output must be a scalar, through the arena directly.
  ddx::rt::Builder<> b;
  const RV v = ddx::rt::vec(b, "x", 3);
  const auto rooted = ddx::rt::equation(RE{b, v.id()});
  EXPECT_EQ(rooted.status()->code, ddx::errc::tensor_shape);

  // No arena to put a vector in.
  EXPECT_EQ(ddx::rt::equation([] { return sum(ddx::rt::vec("x", 2)); })
                .poisoned(),
            false);
  const RV orphan = ddx::rt::vec("x", 2);
  EXPECT_EQ(ddx::rt::equation(sum(orphan)).status()->code, ddx::errc::no_arena);
}

TEST(RtTensor, UnivariateTaylorSweepsRefuseTensors) {
  const auto eq = ddx::rt::equation([] {
    const RV x = ddx::rt::vec("x", 1);
    return sum(x * x * x);
  });
  ASSERT_FALSE(eq.poisoned());
  EXPECT_EQ(eq.template univariate_derivative<2>(0.7).error().code,
            ddx::errc::unsupported_scalar);
}

// One graph, one sweep: the batch call, the block sweep and a remembered call
// all read the same bits.
TEST(RtTensor, BatchesAndThreadsAreBitIdentical) {
  constexpr std::size_t n = 7;
  const auto eq = tensor_mse(n);
  ASSERT_FALSE(eq.poisoned());
  constexpr std::size_t points = 64; // past block_lanes, with a ragged tail
  std::vector<std::vector<double>> xs(n, std::vector<double>(points));
  for (std::size_t p = 0; p < points; ++p) {
    const auto at = composition(n);
    for (std::size_t i = 0; i < n; ++i) {
      xs[i][p] = at[i] * (1.0 + 0.01 * static_cast<double>(p));
    }
  }
  std::vector<const double *> columns(n);
  std::ranges::transform(xs, columns.begin(),
                         [](const auto &c) { return c.data(); });
  std::vector<double> f(points);
  std::vector<std::vector<double>> g(n, std::vector<double>(points));
  std::vector<double *> gcols(n);
  std::ranges::transform(g, gcols.begin(), [](auto &c) { return c.data(); });
  double *const values[]{f.data()};
  ASSERT_TRUE(eq.jacobian(columns, values, gcols, points).has_value());
  for (std::size_t p = 0; p < points; ++p) {
    std::vector<double> at(n);
    for (std::size_t i = 0; i < n; ++i) {
      at[i] = xs[i][p];
    }
    EXPECT_EQ(f[p], *eq.evaluate(std::span<const double>{at})) << p;
    const auto grad = *eq.gradient(std::span<const double>{at});
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_EQ(g[i][p], grad[i]) << p << "," << i;
    }
  }

  ASSERT_TRUE(eq.prepare({ddx::rt::Want::Jacobian, ddx::rt::Want::Hvp})
                  .has_value());
  const auto at = composition(n);
  const auto d = direction(n);
  const auto want_g = *eq.gradient(std::span<const double>{at});
  const auto want_h = *eq.hvp(d, std::span<const double>{at});
  constexpr std::size_t threads = 8;
  std::latch start{threads};
  std::vector<std::jthread> pool;
  std::vector<bool> ok(threads, false);
  for (std::size_t t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      start.arrive_and_wait();
      const auto g2 = *eq.gradient(std::span<const double>{at});
      const auto h2 = *eq.hvp(d, std::span<const double>{at});
      ok[t] = std::ranges::equal(g2, want_g) && std::ranges::equal(h2, want_h);
    });
  }
  pool.clear();
  EXPECT_TRUE(std::ranges::all_of(ok, std::identity{}));
}

TEST(RtTensor, RemembersTheLastCallThroughTensorRows) {
  constexpr std::size_t n = 6;
  const auto cold = tensor_mse(n);
  const auto warm = ddx::rt::equation(
      [] { return mse_tensor(ddx::rt::vec("x", n)); }, ddx::rt::LastValue{});
  auto at = composition(n);
  for (const std::size_t moved : {0uz, 3uz, 5uz, 0uz}) {
    at[moved] *= 1.1;
    EXPECT_EQ(*cold.gradient(std::span<const double>{at}),
              *warm.gradient(std::span<const double>{at}))
        << moved;
    EXPECT_EQ(*cold.evaluate(std::span<const double>{at}),
              *warm.evaluate(std::span<const double>{at}))
        << moved;
  }
}

TEST(RtTensor, RoundTripsThroughAFile) {
  const Scratch file{"tensor"};
  constexpr std::size_t n = 40;
  const auto built = tensor_mse(n);
  ASSERT_FALSE(built.poisoned());
  ASSERT_TRUE(built.save(file).has_value());
  const auto loaded = ddx::rt::load(file.path());
  ASSERT_TRUE(loaded.has_value()) << loaded.error().code;
  const auto at = composition(n);
  const auto d = direction(n);
  EXPECT_EQ(*built.evaluate(std::span<const double>{at}),
            *loaded->evaluate(std::span<const double>{at}));
  EXPECT_EQ(*built.gradient(std::span<const double>{at}),
            *loaded->gradient(std::span<const double>{at}));
  EXPECT_EQ(*built.hvp(d, std::span<const double>{at}),
            *loaded->hvp(d, std::span<const double>{at}));
  EXPECT_TRUE(built.verify(file).has_value());
}

// A table of one's own: the reference kernels under another name answer the
// same bits, and the equation records which table it ran on.
namespace {
struct Renamed : ddx::rt::reference::Kernels {
  static constexpr std::string_view name = "renamed";
};
static_assert(ddx::rt::TensorKernels<Renamed>);
constexpr auto renamed_table = ddx::rt::kernel_table<Renamed>();
} // namespace

TEST(RtTensor, TakesAKernelTable) {
  const auto &table = renamed_table;
  EXPECT_EQ(ddx::rt::kernels().name, "reference");

  constexpr std::size_t n = 5;
  const auto plain = tensor_mse(n);
  auto own = tensor_mse(n);
  own.kernels(table);
  const auto at = composition(n);
  EXPECT_EQ(*plain.gradient(std::span<const double>{at}),
            *own.gradient(std::span<const double>{at}));
  EXPECT_EQ(own.tensor_kernels().name, "renamed");
  EXPECT_EQ(plain.tensor_kernels().name, "reference");
}

// The point of the exercise: the graph is O(n) nodes and the sweep a few GEMVs
// where the scalar spelling was O(n^2) nodes.  Counts are asserted; the times
// are printed.
TEST(RtTensor, StaysSmallAtAThousandSpecies) {
  constexpr std::size_t n = 1000;
  const auto start = std::chrono::steady_clock::now();
  ddx::rt::Builder<> b;
  const RE f = mse_tensor(ddx::rt::vec(b, "x", n));
  const auto g = ddx::rt::build_reverse_jacobian(b, f.id(b));
  const auto gradient_nodes = b.size();
  const auto along = ddx::rt::detail::seeded_sum(b, g.partial);
  (void)ddx::rt::build_reverse_jacobian(b, along.id(b));
  const auto hvp_nodes = b.size();
  const std::chrono::duration<double, std::milli> built =
      std::chrono::steady_clock::now() - start;
  EXPECT_LT(gradient_nodes, 50'000u);
  EXPECT_LT(hvp_nodes, 50'000u);

  const auto eq = tensor_mse(n);
  ASSERT_FALSE(eq.poisoned());
  const auto at = composition(n);
  const auto t0 = std::chrono::steady_clock::now();
  const auto grad = eq.gradient(std::span<const double>{at});
  const auto t1 = std::chrono::steady_clock::now();
  const auto h = eq.hvp(direction(n), std::span<const double>{at});
  const auto t2 = std::chrono::steady_clock::now();
  ASSERT_TRUE(grad.has_value());
  ASSERT_TRUE(h.has_value());
  std::printf("[ ddx ] tensor mse n = %zu: %zu gradient nodes, %zu hvp nodes, "
              "swept in %.1f ms; gradient %.2f ms, hvp %.2f ms (first call)\n",
              n, gradient_nodes, hvp_nodes, built.count(),
              std::chrono::duration<double, std::milli>(t1 - t0).count(),
              std::chrono::duration<double, std::milli>(t2 - t1).count());
}
