#include "energy_models.hpp"
#include "rt/equation.hpp"
#include "rt/kernels.hpp"
#include "rt/tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <functional>
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

using RE = ddx::rt::Expression<>;
using RV = RE; // a handle known to be a vector

// The symbols as vec() names them, so both spellings share one point layout.
std::vector<RE> symbols(std::size_t n) {
  std::vector<RE> x;
  for (const std::size_t i : std::views::iota(0uz, n)) {
    x.push_back(ddx::rt::var(std::format("x[{}]", i)));
  }
  return x;
}

using models::composition;

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
  const RV rv = ddx::rt::vec_constant(r);
  const RV qv = ddx::rt::vec_constant(q);
  const RE sum_r = dot(rv, x);
  const RE sum_q = dot(qv, x);
  const RV phi = (rv * x) / sum_r;
  const RV theta = (qv * x) / sum_q;
  const RE comb = sum(x * log(phi / x)) + 5.0 * sum(qv * x * log(theta / phi));
  const RV inner = gemv(ddx::rt::mat_constant(n, n, tau), theta, true);
  const RE res = -sum(qv * x * log(inner));
  return comb + res;
}

// models::mse over vectors: the Debye-Hueckel and double-sum terms, then
// UNIQUAC.
RE debye_hueckel_and_double_sum(const RV &x) {
  const std::size_t n = x.size();
  std::vector<double> half_zc2(n), p(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    const double zc = models::pseudo(i, 5, -2.0, 2.0);
    half_zc2[i] = 0.5 * zc * zc;
    for (std::size_t j = 0; j < n; ++j) {
      p[i + j * n] = models::pseudo(i, j, -0.4, 0.4);
    }
  }
  const RE ionic = dot(ddx::rt::vec_constant(half_zc2), x);
  const RE lr = -0.3915 * sqrt(ionic) / (RE{1} + 1.2 * sqrt(ionic));
  const RE mr =
      dot(x, gemv(ddx::rt::mat_constant(n, n, p), x)) * exp(-sqrt(ionic));
  return lr + mr;
}
RE mse_tensor(const RV &x) {
  return debye_hueckel_and_double_sum(x) + uniquac_tensor(x);
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
  for (const auto [i, pair] :
       std::views::zip(want, got) | std::views::enumerate) {
    const auto [w, g] = pair;
    const double scale = std::max({1.0, std::abs(w), std::abs(g)});
    EXPECT_LE(std::abs(w - g), tol * scale) << what << " at " << i;
  }
}
void expect_close(double want, double got, double tol, const char *what) {
  expect_close(std::span{&want, 1}, std::span{&got, 1}, tol, what);
}

// dense_hessian() at `at` is the coloured hessian(), to `tol`.
void expect_dense_matches(const auto &eq, std::span<const double> at,
                          double tol = 1e-10) {
  const auto dense = eq.dense_hessian(at);
  ASSERT_TRUE(dense.has_value()) << dense.error().code;
  expect_close(*eq.hessian(at), *dense, tol, "dense vs coloured");
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

// composition(n), moved a little further at each point p.
std::vector<double> drifted(std::size_t n, std::size_t p) {
  auto at = composition(n);
  for (double &v : at) {
    v *= 1.0 + 0.01 * static_cast<double>(p);
  }
  return at;
}
// `points` points, at(p) each, swept as one batch Jacobian: x, f and g column
// by column, as the batch call takes them.
struct Batch {
  std::vector<std::vector<double>> xs;
  std::vector<double> f;
  std::vector<std::vector<double>> g;
  [[nodiscard]] std::vector<double> point(std::size_t p) const {
    return xs | std::views::transform([p](const auto &c) { return c[p]; }) |
           ddx::impl::to<std::vector<double>>();
  }
  [[nodiscard]] std::vector<double> gradient(std::size_t p) const {
    return g | std::views::transform([p](const auto &c) { return c[p]; }) |
           ddx::impl::to<std::vector<double>>();
  }
};
Batch batch_jacobian(const auto &eq, std::size_t width, std::size_t points,
                     auto &&at) {
  Batch r{.xs = std::vector(width, std::vector<double>(points)),
          .f = std::vector<double>(points),
          .g = std::vector(width, std::vector<double>(points))};
  for (std::size_t p = 0; p < points; ++p) {
    const auto x = at(p);
    for (std::size_t i = 0; i < width; ++i) {
      r.xs[i][p] = x[i];
    }
  }
  const auto columns = r.xs | std::views::transform([](const auto &c) {
                         return c.data();
                       }) |
                       ddx::impl::to<std::vector<const double *>>();
  const auto gcols =
      r.g | std::views::transform([](auto &c) { return c.data(); }) |
      ddx::impl::to<std::vector<double *>>();
  double *const values[]{r.f.data()};
  EXPECT_TRUE(eq.jacobian(columns, values, gcols, points).has_value());
  return r;
}
// Every point of a batch is the point calls' answer, to the bit.
void expect_batch_matches(const auto &eq, const Batch &r) {
  for (std::size_t p = 0; p < r.f.size(); ++p) {
    const auto at = r.point(p);
    EXPECT_EQ(r.f[p], *eq.evaluate(std::span<const double>{at})) << p;
    EXPECT_EQ(r.gradient(p), *eq.gradient(std::span<const double>{at})) << p;
  }
}
// Eight threads at once, each answering what `call` answers alone.
void expect_threads_agree(auto &&call) {
  const auto want = call();
  constexpr std::size_t threads = 8;
  std::latch start{threads};
  // One byte per thread: vector<bool> packs bits, and eight threads writing one
  // word is a race the comparison would then report as a moved bit.
  std::vector<char> ok(threads, 0);
  {
    std::vector<std::jthread> pool;
    for (std::size_t t = 0; t < threads; ++t) {
      pool.emplace_back([&, t] {
        start.arrive_and_wait();
        ok[t] = call() == want;
      });
    }
  }
  EXPECT_TRUE(std::ranges::all_of(ok, std::identity{}));
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
void each_node(std::size_t n, Scalar scalar, Tensor tensor,
               double tol = 1e-12) {
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
        const auto M = ddx::rt::mat_constant(rows, cols, m);
        std::vector<RE> head;
        for (std::size_t i = 0; i < rows; ++i) {
          head.push_back(x[i]);
        }
        const RV y = gemv(M, x);
        return sum(y * y) + dot(gemv(M, ddx::rt::pack(head), true), x);
      });
}

// Nodes are interned, so the operator and the call agreeing on an id is the
// same node -- the same kernel, the same bits.
TEST(RtTensor, OperatorsAreTheCalls) {
  constexpr std::size_t rows = 4, cols = 6;
  const auto m = matrix(rows, cols);
  ddx::rt::Arena<> b;
  const RV x = ddx::rt::vec(b, "x", cols);
  const RV y = ddx::rt::vec(b, "y", rows);
  const RE T = ddx::rt::var(b, "T");
  const auto M = ddx::rt::mat_constant(b, rows, cols, std::span{m});
  const auto C = exp(T * M);

  EXPECT_EQ((x % x).id(), dot(x, x).id());
  EXPECT_EQ((M % x).id(), gemv(M, x).id());
  EXPECT_EQ((C % x).id(), gemv(C, x).id());
  EXPECT_EQ((ddx::rt::transpose(M) % y).id(), gemv(M, y, true).id());
  EXPECT_EQ((ddx::rt::transpose(C) % y).id(), gemv(C, y, true).id());
  EXPECT_EQ((y % (M % x) + x % x).id(),
            (dot(y, gemv(M, x)) + dot(x, x)).id());
  EXPECT_EQ((2.0 * x % x).id(), dot(2.0 * x, x).id());

  const auto why = [](const RE &e) {
    return ddx::rt::equation(e).status()->code;
  };
  const std::array refused_all{
      M % M, x % M, y % C, M % y, ddx::rt::transpose(M) % x, x % y};
  for (const RE &refused : refused_all) {
    EXPECT_EQ(why(refused), ddx::errc::tensor_shape);
  }
  const RE orphan = ddx::rt::vec("z", 2);
  EXPECT_EQ(why(orphan % M), ddx::errc::outside_equation);
  EXPECT_EQ(why(M % orphan), ddx::errc::outside_equation);
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
        const RV v = ddx::rt::detail::add_at(x, x[0] * x[1], 2);
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
    return sum(gemv(ddx::rt::mat_constant(2, 3, m), ddx::rt::vec("a", 2)));
  });
  EXPECT_EQ(gemv_shape.status()->code, ddx::errc::tensor_shape);

  // A vector where an output must be a scalar, through the arena directly.
  ddx::rt::Arena<> b;
  const RV v = ddx::rt::vec(b, "x", 3);
  const auto rooted = ddx::rt::equation(RE{b, v.id()});
  EXPECT_EQ(rooted.status()->code, ddx::errc::tensor_shape);

  // No arena to put a vector in.
  EXPECT_EQ(
      ddx::rt::equation([] { return sum(ddx::rt::vec("x", 2)); }).poisoned(),
      false);
  const RV orphan = ddx::rt::vec("x", 2);
  EXPECT_EQ(ddx::rt::equation(sum(orphan)).status()->code,
            ddx::errc::outside_equation);
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

// One graph, one sweep: the batch call and the block sweep read the same bits.
TEST(RtTensor, BatchesAndThreadsAreBitIdentical) {
  constexpr std::size_t n = 7;
  const auto eq = tensor_mse(n);
  ASSERT_FALSE(eq.poisoned());
  // Past block_lanes, with a ragged tail.
  expect_batch_matches(eq, batch_jacobian(eq, n, 64, [](std::size_t p) { return drifted(n, p); }));

  ASSERT_TRUE(
      eq.prepare({ddx::rt::Want::Jacobian, ddx::rt::Want::Hvp}).has_value());
  const auto at = composition(n);
  const auto d = direction(n);
  expect_threads_agree([&] {
    return std::pair{*eq.gradient(std::span<const double>{at}),
                     *eq.hvp(d, std::span<const double>{at})};
  });
}

// A table of one's own: the reference kernels under another name answer the
// same bits, and the equation says which table it runs on.
namespace {
struct Renamed : ddx::rt::reference::Kernels<> {
  static constexpr std::string_view name = "renamed";
};
static_assert(ddx::rt::TensorKernels<Renamed>);
constexpr auto renamed_table = ddx::rt::tensor_kernel_table<Renamed>();
} // namespace

TEST(RtTensor, TakesAKernelTable) {
  const auto &table = renamed_table;
  EXPECT_EQ(ddx::rt::tensor_kernels().name, "reference");

  constexpr std::size_t n = 5;
  const auto plain = tensor_mse(n);
  auto own = tensor_mse(n);
  own.tensor_kernels(table);
  const auto at = composition(n);
  EXPECT_EQ(*plain.gradient(std::span<const double>{at}),
            *own.gradient(std::span<const double>{at}));
  EXPECT_EQ(own.tensor_kernels().name, "renamed");
  EXPECT_EQ(plain.tensor_kernels().name, "reference");
}

// Tensor rows compute at the equation's own scalar, through a table of it.
namespace {
auto narrowed(std::span<const double> v) {
  return v |
         std::views::transform([](double d) { return static_cast<float>(d); }) |
         std::ranges::to<std::vector>();
}

template <typename T> auto scalar_typed_tensor_model(std::size_t n) {
  return ddx::rt::equation<T>([n] {
    const auto x = ddx::rt::vec<T>("x", n);
    std::vector<T> m(n * n);
    for (std::size_t i = 0; i < m.size(); ++i) {
      m[i] = static_cast<T>(models::pseudo(i % n, i / n, -1.0, 1.0));
    }
    const auto M = ddx::rt::mat_constant<T>(n, n, m);
    const auto y = gemv(M, x);
    return dot(x, y) + sum(log(x) * x) +
           x[1] * sum(select(x - T(0.3), exp(x * T(0.5)), sqrt(x))) +
           dot(gemv(M, x, true), y);
  });
}
} // namespace

TEST(RtTensor, AFloatEquationComputesItsTensorRows) {
  constexpr std::size_t n = 6;
  const auto wide = scalar_typed_tensor_model<double>(n);
  const auto narrow = scalar_typed_tensor_model<float>(n);
  const auto at = composition(n);
  const auto at_f = narrowed(at);
  const double v = *wide.evaluate(std::span<const double>{at});
  const float vf = *narrow.evaluate(std::span<const float>{at_f});
  expect_close(v, double{vf}, 1e-5, "value");
  const auto g = *wide.gradient(std::span<const double>{at});
  const auto gf = *narrow.gradient(std::span<const float>{at_f});
  expect_close(g, std::vector<double>(gf.begin(), gf.end()), 1e-5, "gradient");
  const auto h = *wide.dense_hessian(std::span<const double>{at});
  const auto hf = *narrow.dense_hessian(std::span<const float>{at_f});
  expect_close(h, std::vector<double>(hf.begin(), hf.end()), 1e-4,
               "dense hessian");
}

// One default per scalar: a float table set for the process leaves double's
// alone, and an equation's own table wins over either.
namespace {
struct RenamedF : ddx::rt::reference::Kernels<float> {
  static constexpr std::string_view name = "renamed-f";
};
static_assert(ddx::rt::TensorKernels<RenamedF, float>);
static_assert(!ddx::rt::TensorKernels<RenamedF>);
constexpr auto renamed_f_table =
    ddx::rt::tensor_kernel_table<RenamedF, float>();
} // namespace

TEST(RtTensor, EachScalarHasItsOwnDefaultTable) {
  constexpr std::size_t n = 5;
  const auto at = composition(n);
  const auto at_f = narrowed(at);
  const auto plain = scalar_typed_tensor_model<float>(n);
  const auto want = *plain.gradient(std::span<const float>{at_f});

  ddx::rt::use_tensor_kernels(renamed_f_table);
  EXPECT_EQ(ddx::rt::tensor_kernels<float>().name, "renamed-f");
  EXPECT_EQ(ddx::rt::tensor_kernels().name, "reference");
  const auto defaulted = scalar_typed_tensor_model<float>(n);
  EXPECT_EQ(defaulted.tensor_kernels().name, "renamed-f");
  EXPECT_EQ(*defaulted.gradient(std::span<const float>{at_f}), want);
  ddx::rt::use_tensor_kernels(ddx::rt::reference_table<float>);

  auto own = scalar_typed_tensor_model<float>(n);
  own.tensor_kernels(renamed_f_table);
  EXPECT_EQ(own.tensor_kernels().name, "renamed-f");
  EXPECT_EQ(ddx::rt::tensor_kernels<float>().name, "reference");
  EXPECT_EQ(*own.gradient(std::span<const float>{at_f}), want);
}

// The point of the exercise: the graph is O(n) nodes and the sweep a few GEMVs
// where the scalar spelling was O(n^2) nodes.  Counts are asserted; the times
// are printed.
TEST(RtTensor, StaysSmallAtAThousandSpecies) {
  constexpr std::size_t n = 1000;
  const auto start = std::chrono::steady_clock::now();
  ddx::rt::Arena<> b;
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

// --- D10b: second order through tensor nodes -------------------------------
// Nothing new is implemented: hvp is a reverse sweep over the seeded gradient,
// and every first-order rule emits nodes the second sweep has rules for.  What
// follows pins that down against the scalar graph, against the coloured
// Hessian, against differences, and across a pool.

namespace {
// Every Hessian column by one batch call of n points, each with its own unit
// direction: the shape a pool uses.  `from`/`count` pick a column range, so a
// thread takes a slice and the slices tile the batch.
std::vector<double> hessian_by_hvp(const auto &eq, std::span<const double> at,
                                   std::size_t from, std::size_t count) {
  const std::size_t n = at.size();
  std::vector<std::vector<double>> xs(n, std::vector<double>(count));
  std::vector<std::vector<double>> vs(n, std::vector<double>(count, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    std::ranges::fill(xs[i], at[i]);
  }
  for (std::size_t k = 0; k < count; ++k) {
    vs[from + k][k] = 1.0;
  }
  std::vector<const double *> xcols(n), vcols(n);
  std::ranges::transform(xs, xcols.begin(),
                         [](const auto &c) { return c.data(); });
  std::ranges::transform(vs, vcols.begin(),
                         [](const auto &c) { return c.data(); });
  std::vector<double> f(count);
  std::vector<std::vector<double>> g(n, std::vector<double>(count));
  std::vector<std::vector<double>> hv(n, std::vector<double>(count));
  std::vector<double *> gcols(n), hcols(n);
  std::ranges::transform(g, gcols.begin(), [](auto &c) { return c.data(); });
  std::ranges::transform(hv, hcols.begin(), [](auto &c) { return c.data(); });
  double *const fcol[]{f.data()};
  EXPECT_TRUE(eq.hvp(xcols, vcols, fcol, gcols, hcols, count).has_value());
  // Row-major H(i, j): point k answers column from + k.
  std::vector<double> h(n * count);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < count; ++k) {
      h[i * count + k] = hv[i][k];
    }
  }
  return h;
}
} // namespace

TEST(RtTensorHvp, MatchesTheScalarGraphAtAHundredSpecies) {
  constexpr std::size_t n = 100;
  const auto want = scalar_mse(n);
  const auto got = tensor_mse(n);
  ASSERT_FALSE(got.poisoned());
  const auto at = composition(n);
  for (const std::size_t j : {0uz, 37uz, 99uz}) {
    const auto name = std::format("x[{}]", j);
    expect_close(*want.hvp(name, std::span<const double>{at}),
                 *got.hvp(name, std::span<const double>{at}), 1e-10,
                 name.c_str());
  }
  const auto d = direction(n);
  expect_close(*want.hvp(d, std::span<const double>{at}),
               *got.hvp(d, std::span<const double>{at}), 1e-10, "hvp(d)");
}

TEST(RtTensorHvp, ColumnsAssembleTheColouredHessian) {
  constexpr std::size_t n = 48;
  const auto eq = tensor_mse(n);
  ASSERT_FALSE(eq.poisoned());
  const auto at = composition(n);
  const auto by_hvp = hessian_by_hvp(eq, at, 0, n);
  const auto coloured = *eq.hessian(std::span<const double>{at});
  expect_close(coloured, by_hvp, 1e-10, "H by n hvp vs coloured");
  // Symmetric, as a Hessian is, to the same rounding.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      expect_close(by_hvp[i * n + j], by_hvp[j * n + i], 1e-10, "symmetry");
    }
  }
}

TEST(RtTensorHvp, APoolSplitsTheBatchByColumnAndAgreesToTheBit) {
  constexpr std::size_t n = 64;
  const auto eq = tensor_mse(n);
  ASSERT_TRUE(eq.prepare({ddx::rt::Want::Hvp}).has_value());
  const auto at = composition(n);
  const auto whole = hessian_by_hvp(eq, at, 0, n);

  constexpr std::size_t threads = 8;
  constexpr std::size_t slice = n / threads;
  std::vector<std::vector<double>> parts(threads);
  std::latch start{threads};
  {
    std::vector<std::jthread> pool;
    for (std::size_t t = 0; t < threads; ++t) {
      pool.emplace_back([&, t] {
        start.arrive_and_wait();
        parts[t] = hessian_by_hvp(eq, at, t * slice, slice);
      });
    }
  }
  for (std::size_t t = 0; t < threads; ++t) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t k = 0; k < slice; ++k) {
        EXPECT_EQ(parts[t][i * slice + k], whole[i * n + t * slice + k])
            << t << "," << i << "," << k;
      }
    }
  }
}

TEST(RtTensorHvp, JacobianJvpAlongTemperatureMatchesDifferences) {
  constexpr std::size_t n = 12;
  const auto eq = ddx::rt::equation([] {
    const RE T = ddx::rt::var("T");
    const RV x = ddx::rt::vec("x", n);
    return T * mse_tensor(x) + log(T) * sum(x * x);
  });
  ASSERT_FALSE(eq.poisoned());
  // T is declared first, so it is the first column.
  auto at = composition(n);
  at.insert(at.begin(), 1.7);
  const auto dj = *eq.jacobian_jvp("T", std::span<const double>{at});
  const double h = 1e-6;
  auto hi = at, lo = at;
  hi[0] += h;
  lo[0] -= h;
  const auto ghi = *eq.gradient(std::span<const double>{hi});
  const auto glo = *eq.gradient(std::span<const double>{lo});
  ASSERT_EQ(dj.size(), ghi.size());
  for (std::size_t j = 0; j < dj.size(); ++j) {
    const double fd = (ghi[j] - glo[j]) / (2 * h);
    EXPECT_NEAR(dj[j], fd, 1e-6 * std::max(1.0, std::abs(fd))) << j;
  }
}

TEST(RtTensorHvp, CostAtAThousandSpecies) {
  constexpr std::size_t n = 1000;
  const auto eq = tensor_mse(n);
  ASSERT_FALSE(eq.poisoned());
  ASSERT_TRUE(eq.prepare({ddx::rt::Want::Hvp}).has_value());
  const auto at = composition(n);
  const auto d = direction(n);
  (void)eq.hvp(d, std::span<const double>{at}); // warm
  const auto t0 = std::chrono::steady_clock::now();
  const auto h = eq.hvp(d, std::span<const double>{at});
  const auto t1 = std::chrono::steady_clock::now();
  ASSERT_TRUE(h.has_value());
  constexpr std::size_t columns = 16;
  const auto t2 = std::chrono::steady_clock::now();
  const auto block = hessian_by_hvp(eq, at, 0, columns);
  const auto t3 = std::chrono::steady_clock::now();
  EXPECT_EQ(block.size(), n * columns);
  std::printf(
      "[ ddx ] tensor mse n = %zu: hvp %.2f ms warm; %zu Hessian "
      "columns by batch hvp %.2f ms (%.2f ms/column)\n",
      n, std::chrono::duration<double, std::milli>(t1 - t0).count(), columns,
      std::chrono::duration<double, std::milli>(t3 - t2).count(),
      std::chrono::duration<double, std::milli>(t3 - t2).count() / columns);
}

// --- D10c: the dense Hessian in O(n^2) ---------------------------------------
// One forward sweep over the gradient's cone with matrix-valued tangents, in
// place of n + 1 coloured reverse sweeps.  The oracle is the coloured block
// itself (tested elsewhere against the scalar graph), and for the scalar
// spelling the two run over the same nodes.

namespace {
// hessian() is row-major n x n; so is dense_hessian().
void expect_dense_matches_coloured(const auto &eq, std::span<const double> at,
                                   double tol) {
  ASSERT_FALSE(eq.poisoned());
  const auto coloured = eq.hessian(at);
  const auto dense = eq.dense_hessian(at);
  ASSERT_TRUE(coloured.has_value());
  ASSERT_TRUE(dense.has_value()) << dense.error().code;
  expect_close(*coloured, *dense, tol, "dense vs coloured");
}
} // namespace

// A select over tensors picks the arm taken, by element under a vector
// condition and whole under a scalar one, so where the other arm is infinite
// -- log at 0 -- no derivative reads 0 * inf.
TEST(RtTensor, SelectOverTensorsPicksTheArmTaken) {
  const std::vector<double> at{0.0, 0.3, 0.5, 0.2};
  const auto want = ddx::rt::equation([] {
    const auto x = symbols(4);
    RE acc{0};
    for (const auto &e : x) {
      acc += select(e > 0.0, e * log(e), RE{0.0}) +
             select(x[0] > 0.1, log(e), e * e);
    }
    return acc;
  });
  const auto got = ddx::rt::equation([] {
    const RV x = ddx::rt::vec("x", 4);
    return sum(select(x > 0.0, x * log(x), RE{0.0})) +
           sum(select(x[0] > 0.1, log(x), x * x));
  });
  expect_same_derivatives(want, got, at);
  expect_dense_matches_coloured(got, at, 1e-12);
  EXPECT_EQ((*got.gradient(at))[0], 0.0);

  const auto mismatched = ddx::rt::equation([] {
    const RV x = ddx::rt::vec("x", 3);
    return sum(select(ddx::rt::vec("c", 2), x, x));
  });
  EXPECT_EQ(mismatched.status()->code, ddx::errc::tensor_shape);
}

TEST(RtTensorDense, MatchesTheColouredHessianOnTheVectorModel) {
  for (const std::size_t n : {5uz, 20uz, 50uz}) {
    const auto eq = tensor_mse(n);
    expect_dense_matches_coloured(eq, composition(n), 1e-10);
    // The UNIQUAC residual is log(tau^T theta), whose Hessian is
    // tau diag(.) tau^T: one dense product, which no algebra avoids.  The
    // rest stays quadratic.
    const auto cost = eq.dense_hessian_cost();
    ASSERT_TRUE(cost.has_value());
    EXPECT_EQ(cost->n3_terms, 1u);
    EXPECT_GT(cost->n2_terms, 0u);
  }
}

TEST(RtTensorDense, MatchesTheColouredHessianOnTheScalarModel) {
  // No vector in the model: every tangent is a row, and the block is the rows
  // stacked.
  const auto eq = scalar_mse(12);
  expect_dense_matches_coloured(eq, composition(12), 1e-10);
}

TEST(RtTensorDense, EachNodeAlone) {
  constexpr std::size_t n = 6;
  const auto m = matrix(n, n);
  const auto at = composition(n);
  const auto models = {
      std::function<RE(const RV &)>{[&](const RV &x) {
        return dot(x, gemv(ddx::rt::mat_constant(n, n, m), x));
      }},
      {[&](const RV &x) {
        return sum(log(x) * gemv(ddx::rt::mat_constant(n, n, m), x, true));
      }},
      {[&](const RV &x) {
        const RE s = dot(x, x);
        return sum(exp(x / s)) * s + x[2] * x[3];
      }},
      {[&](const RV &x) {
        return sum(ddx::rt::detail::add_at(x * x, x[0] * x[1], 4) *
                   pow(x, 1.5));
      }},
      {[&](const RV &x) {
        const RE k = select(x[0] < 0.3, x[1], x[2] * x[2]);
        return sum(k * (x * log(x)));
      }},
  };
  for (const auto &model : models) {
    const auto eq =
        ddx::rt::equation([&] { return model(ddx::rt::vec("x", n)); });
    expect_dense_matches_coloured(eq, at, 1e-10);
  }
  // A quadratic form and a transposed product each stay in the algebra: the
  // identity through a constant matrix is that matrix, copied.
  const auto quadratic = ddx::rt::equation([&] {
    const RV x = ddx::rt::vec("x", n);
    const auto C = ddx::rt::mat_constant(n, n, m);
    return dot(x, gemv(C, x)) + sum(x * gemv(C, x, true));
  });
  EXPECT_EQ(quadratic.dense_hessian_cost()->n3_terms, 0u);
}

TEST(RtTensorDense, BatchColumnsAreThePointsBlockRowMajor) {
  constexpr std::size_t n = 5;
  const auto eq = tensor_mse(n);
  const auto at = composition(n);
  const auto want = *eq.dense_hessian(std::span<const double>{at});
  ASSERT_EQ(*eq.dense_hessian_columns(), n * n);
  constexpr std::size_t points = 3;
  std::vector<std::vector<double>> xs(n, std::vector<double>(points));
  for (std::size_t i = 0; i < n; ++i) {
    std::ranges::fill(xs[i], at[i]);
  }
  std::vector<const double *> xcols(n);
  std::ranges::transform(xs, xcols.begin(),
                         [](const auto &c) { return c.data(); });
  std::vector<double> f(points);
  std::vector<std::vector<double>> g(n, std::vector<double>(points));
  std::vector<std::vector<double>> h(n * n, std::vector<double>(points));
  std::vector<double *> gcols(n), hcols(n * n);
  std::ranges::transform(g, gcols.begin(), [](auto &c) { return c.data(); });
  std::ranges::transform(h, hcols.begin(), [](auto &c) { return c.data(); });
  double *const fcol[]{f.data()};
  ASSERT_TRUE(eq.dense_hessian(xcols, fcol, gcols, hcols, points).has_value());
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t p = 0; p < points; ++p) {
        EXPECT_EQ(h[i * n + j][p], want[i * n + j]) << i << "," << j;
      }
    }
  }
}

TEST(RtTensorDense, RefusesASystem) {
  const auto sys = ddx::rt::equation([] {
    const RV x = ddx::rt::vec("x", 3);
    return std::vector<RE>{sum(x * x), dot(x, x) * x[0]};
  });
  EXPECT_EQ(
      sys.dense_hessian(std::span<const double>{composition(3)}).error().code,
      ddx::errc::not_single_output);
  EXPECT_FALSE(sys.dense_hessian_cost().has_value());
}

TEST(RtTensorDense, CostAtManySpecies) {
  constexpr std::size_t n = 300;
  const auto eq = tensor_mse(n);
  ASSERT_FALSE(eq.poisoned());
  const auto at = composition(n);
  const auto t0 = std::chrono::steady_clock::now();
  const auto first = eq.dense_hessian(std::span<const double>{at});
  const auto t1 = std::chrono::steady_clock::now();
  const auto second = eq.dense_hessian(std::span<const double>{at});
  const auto t2 = std::chrono::steady_clock::now();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  const auto cost = *eq.dense_hessian_cost();
  std::printf("[ ddx ] dense hessian n = %zu: %zu quadratic terms, %zu cubic; "
              "first %.1f ms, warm %.2f ms\n",
              n, cost.n2_terms, cost.n3_terms,
              std::chrono::duration<double, std::milli>(t1 - t0).count(),
              std::chrono::duration<double, std::milli>(t2 - t1).count());
  EXPECT_EQ(cost.n3_terms, 1u);
}

// --- Computed matrices -------------------------------------------------------
// A pair matrix whose every entry is a function of T: arithmetic over a table
// reads it as its row, so exp(T * B) is a handful of nodes where the scalar
// spelling is n^2.  The scalar spelling is the oracle.

namespace {
// B as n^2 scalars, (i, j) at i + j n.
std::vector<double> pair_table(std::size_t n, std::size_t seed, double lo,
                               double hi) {
  std::vector<double> m(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      m[i + j * n] = models::pseudo(i + seed, j, lo, hi);
    }
  }
  return m;
}
// T first, then x: the order both spellings declare them in.
std::vector<double> at_temperature(std::size_t n, double T) {
  auto at = composition(n);
  at.insert(at.begin(), T);
  return at;
}
void expect_same_with_temperature(const auto &want, const auto &got,
                                  std::span<const double> at,
                                  double tol = 1e-12) {
  expect_same_derivatives(want, got, at, tol);
  expect_close(*want.jacobian_jvp("T", at), *got.jacobian_jvp("T", at), tol,
               "jacobian_jvp along T");
}
} // namespace

// select over every pairing of shapes: a scalar choosing a whole matrix, a
// matrix condition choosing element by element, and a scalar arm -- a literal
// or a node -- spread to the other's shape.
TEST(RtMatrix, SelectAcrossShapes) {
  constexpr std::size_t n = 4;
  const auto b1 = pair_table(n, 0, -0.5, 0.5);
  const auto b2 = pair_table(n, 3, 0.2, 1.1);
  const auto want = ddx::rt::equation([&] {
    const RE T = ddx::rt::var("T");
    const auto x = symbols(n);
    RE acc{0};
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const double p = b1[i + j * n];
        const RE m = select(T > 1.0, exp(T * p), T * b2[i + j * n]);
        const RE e = p > 0.0 ? exp(T * p) : RE{0.0};
        acc += x[i] * (m + e) * x[j];
      }
      acc += select(x[i] > 0.25, x[i], T) * log(x[i]);
    }
    return acc;
  });
  const auto got = ddx::rt::equation([&] {
    const RE T = ddx::rt::var("T");
    const RV x = ddx::rt::vec("x", n);
    const auto B1 = ddx::rt::mat_constant(n, n, b1);
    const auto B2 = ddx::rt::mat_constant(n, n, b2);
    const auto M = select(T > 1.0, exp(T * B1), T * B2);
    const auto E = select(B1 > 0.0, exp(T * B1), RE{0.0});
    return x % (M % x) + x % (E % x) + sum(select(x > 0.25, x, T) * log(x));
  });
  for (const double T : {1.3, 0.7}) {
    expect_same_with_temperature(want, got, at_temperature(n, T));
  }

  const auto refused = [](auto model) {
    return ddx::rt::equation(model).status()->code;
  };
  EXPECT_EQ(refused([] {
              const RV x = ddx::rt::vec("x", n);
              const auto B = ddx::rt::mat_constant(n, n, pair_table(n, 0, 0, 1));
              return x % (select(x > 0.5, exp(B * x[0]), B) % x);
            }),
            ddx::errc::tensor_shape); // a vector condition, matrix arms
  EXPECT_EQ(refused([] {
              const RV x = ddx::rt::vec("x", n);
              const auto B = ddx::rt::mat_constant(n, n, pair_table(n, 0, 0, 1));
              return sum(select(x[0] > 0.5, exp(B * x[0]), x));
            }),
            ddx::errc::tensor_shape); // arms of two shapes

  ddx::rt::Arena<> b;
  const RV x = ddx::rt::vec(b, "x", 3);
  EXPECT_EQ(select(RE{1.0}, x, x * x).id(), x.id()); // folded as a scalar's is
}

TEST(RtMatrix, ExpOfTemperatureTimesATableMatchesTheScalarGraph) {
  for (const std::size_t n : {5uz, 20uz, 50uz}) {
    const auto b = pair_table(n, 0, -0.5, 0.5);
    const auto want = ddx::rt::equation([&] {
      const RE T = ddx::rt::var("T");
      const auto x = symbols(n);
      RE acc{0};
      for (std::size_t i = 0; i < n; ++i) {
        RE row{0};
        for (std::size_t j = 0; j < n; ++j) {
          row += x[j] * exp(T * b[i + j * n]);
        }
        acc += x[i] * row;
      }
      return acc;
    });
    const auto got = ddx::rt::equation([&] {
      const RE T = ddx::rt::var("T");
      const RV x = ddx::rt::vec("x", n);
      return dot(x, gemv(exp(T * ddx::rt::mat_constant(n, n, b)), x));
    });
    expect_same_with_temperature(want, got, at_temperature(n, 1.3), 1e-11);
  }
}

// The UNIQUAC-type tau_ij = exp(-(A0 + A1 T + A2 T^2) / (R T)), read
// transposed as sum_j theta_j tau_ji.
TEST(RtMatrix, TemperatureDependentTauMatchesTheScalarGraph) {
  constexpr std::size_t n = 12;
  constexpr double R = 8.314;
  const auto a0 = pair_table(n, 1, -200.0, 400.0);
  const auto a1 = pair_table(n, 2, -1.0, 1.0);
  const auto a2 = pair_table(n, 3, -1e-3, 1e-3);
  const auto want = ddx::rt::equation([&] {
    const RE T = ddx::rt::var("T");
    const auto x = symbols(n);
    RE acc{0};
    for (std::size_t i = 0; i < n; ++i) {
      RE inner{0};
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t ji = j + i * n;
        inner +=
            x[j] * exp(-(a0[ji] + T * a1[ji] + (T * T) * a2[ji]) / (R * T));
      }
      acc += x[i] * log(inner);
    }
    return acc;
  });
  const auto got = ddx::rt::equation([&] {
    const RE T = ddx::rt::var("T");
    const RV x = ddx::rt::vec("x", n);
    const auto A0 = ddx::rt::mat_constant(n, n, a0);
    const auto A1 = ddx::rt::mat_constant(n, n, a1);
    const auto A2 = ddx::rt::mat_constant(n, n, a2);
    const RE tau_scale = -1.0 / (R * T);
    const RV tau = exp(tau_scale * (A0 + T * A1 + (T * T) * A2));
    return sum(x * log(gemv(tau, x, true)));
  });
  expect_same_with_temperature(want, got, at_temperature(n, 310.0), 1e-11);
}

// Each rule alone: Scale, Map, Zip, a literal meeting a matrix, a reduction
// over one, MatVec both ways and Outer.
TEST(RtMatrix, EachRuleAlone) {
  constexpr std::size_t n = 5;
  const auto b = pair_table(n, 4, -1.0, 1.0);
  const auto c = pair_table(n, 5, 0.5, 1.5);
  const auto at = at_temperature(n, 0.8);
  const auto pair = [](auto &&entry) {
    // sum_ij x_i entry(i, j) x_j, written out.
    const auto x = symbols(n);
    RE acc{0};
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        acc += x[i] * entry(i, j) * x[j];
      }
    }
    return acc;
  };
  struct Case {
    std::function<RE(const RE &)> scalar; // given T
    std::function<RE(const RE &, const RV &)> tensor;
  };
  const std::vector<Case> cases{
      {[&](const RE &T) {
         return pair([&](std::size_t i, std::size_t j) {
           return (T * b[i + j * n]) * c[i + j * n] + 2.0;
         });
       },
       [&](const RE &T, const RV &x) {
         const auto B = ddx::rt::mat_constant(n, n, b);
         const auto C = ddx::rt::mat_constant(n, n, c);
         return dot(x, gemv(T * B * C + 2.0, x));
       }},
      {[&](const RE &T) {
         return pair([&](std::size_t i, std::size_t j) {
           return sqrt(c[i + j * n] * T) / (b[i + j * n] - 3.0);
         });
       },
       [&](const RE &T, const RV &x) {
         const auto B = ddx::rt::mat_constant(n, n, b);
         const auto C = ddx::rt::mat_constant(n, n, c);
         return dot(x, gemv(sqrt(C * T) / (B - 3.0), x));
       }},
      {[&](const RE &T) {
         // sum_ij exp(T b_ij) x_i x_j, as the dot of two matrices.
         return pair([&](std::size_t i, std::size_t j) {
           return exp(T * b[i + j * n]);
         });
       },
       [&](const RE &T, const RV &x) {
         return dot(exp(T * ddx::rt::mat_constant(n, n, b)),
                    ddx::rt::outer(x, x));
       }},
      {[&](const RE &T) {
         // sum_ij exp(T x_i x_j): x inside the exponential.
         const auto x = symbols(n);
         RE acc{0};
         for (std::size_t i = 0; i < n; ++i) {
           for (std::size_t j = 0; j < n; ++j) {
             acc += exp(T * x[i] * x[j]);
           }
         }
         return acc;
       },
       [&](const RE &T, const RV &x) {
         return sum(exp(T * ddx::rt::outer(x, x)));
       }},
      {[&](const RE &T) {
         // x^T M^T (log x) with M = T B: the transposed product.
         const auto x = symbols(n);
         RE acc{0};
         for (std::size_t i = 0; i < n; ++i) {
           for (std::size_t j = 0; j < n; ++j) {
             acc += x[j] * (T * b[i + j * n]) * log(x[i]);
           }
         }
         return acc;
       },
       [&](const RE &T, const RV &x) {
         return dot(x, gemv(T * ddx::rt::mat_constant(n, n, b), log(x), true));
       }},
  };
  for (const auto &[k, rule] : cases | std::views::enumerate) {
    SCOPED_TRACE(k);
    const auto &[scalar, tensor] = rule;
    const auto want = ddx::rt::equation([&] {
      const RE T = ddx::rt::var("T");
      return scalar(T);
    });
    const auto got = ddx::rt::equation([&] {
      const RE T = ddx::rt::var("T");
      return tensor(T, ddx::rt::vec("x", n));
    });
    expect_same_with_temperature(want, got, at, 1e-11);
  }
}

// M(T) = B1 + B2 / T + B5 exp(T B6): no x inside, so its tangent is empty and
// the dense Hessian stays in the algebra, M^T included.
TEST(RtMatrix, DenseHessianOverATemperatureDependentMatrix) {
  for (const std::size_t n : {20uz, 100uz}) {
    const auto b1 = pair_table(n, 6, -1.0, 1.0);
    const auto b2 = pair_table(n, 7, -50.0, 50.0);
    const auto b5 = pair_table(n, 8, -0.3, 0.3);
    const auto b6 = pair_table(n, 9, -0.01, 0.01);
    const auto eq = ddx::rt::equation([&] {
      const RE T = ddx::rt::param("T");
      const RV x = ddx::rt::vec("x", n);
      const auto M = ddx::rt::mat_constant(n, n, b1) +
                     ddx::rt::mat_constant(n, n, b2) / T +
                     ddx::rt::mat_constant(n, n, b5) *
                         exp(T * ddx::rt::mat_constant(n, n, b6));
      return dot(x, gemv(M, x)) + sum(log(x) * gemv(M, x, true));
    });
    ASSERT_FALSE(eq.poisoned()) << eq.status()->code;
    auto at = composition(n);
    at.push_back(298.15); // the param follows the vars
    expect_dense_matches(eq, at);
    EXPECT_EQ(eq.dense_hessian_cost()->n3_terms, 0u);
  }
}

// The same M(T) with T a symbol: the block is (n + 1) x (n + 1), and its x
// block is the one T held fixed gives.
TEST(RtMatrix, DenseHessianOverAMatrixThatMovesWithTemperature) {
  for (const std::size_t n : {4uz, 20uz, 50uz, 100uz}) {
    SCOPED_TRACE(n);
    const auto b1 = pair_table(n, 6, -1.0, 1.0);
    const auto b2 = pair_table(n, 7, -50.0, 50.0);
    const auto b5 = pair_table(n, 8, -0.3, 0.3);
    const auto b6 = pair_table(n, 9, -0.01, 0.01);
    const auto model = [&](const RE &T) {
      const RV x = ddx::rt::vec("x", n);
      const auto M = ddx::rt::mat_constant(n, n, b1) +
                     ddx::rt::mat_constant(n, n, b2) / T +
                     ddx::rt::mat_constant(n, n, b5) *
                         exp(T * ddx::rt::mat_constant(n, n, b6));
      return dot(x, gemv(M, x)) + sum(x * log(x)) +
             sum(log(x) * gemv(M, x, true));
    };
    const auto var =
        ddx::rt::equation([&] { return model(ddx::rt::var("T")); });
    const auto fixed =
        ddx::rt::equation([&] { return model(ddx::rt::param("T")); });
    ASSERT_FALSE(var.poisoned()) << var.status()->code;
    const auto at = at_temperature(n, 298.15);
    expect_dense_matches(var, at);
    EXPECT_EQ(var.dense_hessian_cost()->n3_terms, 0u);

    auto point = composition(n);
    point.push_back(298.15);
    const auto x_block = *fixed.dense_hessian(std::span<const double>{point});
    const auto dense = *var.dense_hessian(std::span<const double>{at});
    const std::size_t w = n + 1;
    const auto from_var =
        std::views::cartesian_product(std::views::iota(1uz, w),
                                      std::views::iota(1uz, w)) |
        std::views::transform([&](const auto &ij) {
          const auto [j, i] = ij; // column-major: i fastest
          return dense[i + j * w];
        }) |
        ddx::impl::to<std::vector<double>>();
    expect_close(x_block, from_var, 1e-12, "x block vs T held");
  }
}

// UNIQUAC's residual with T and P symbols: tau = exp(-(A0 + T A1 + P AP) /
// (R T)), read transposed.
TEST(RtMatrix, DenseHessianOverAUniquacResidualInTemperatureAndPressure) {
  constexpr std::size_t n = 15;
  constexpr double R = 8.314;
  const auto a0 = pair_table(n, 13, -200.0, 400.0);
  const auto a1 = pair_table(n, 14, -1.0, 1.0);
  const auto ap = pair_table(n, 15, -1e-2, 1e-2);
  std::vector<double> q(n);
  for (std::size_t i = 0; i < n; ++i) {
    q[i] = 1.0 + 0.1 * static_cast<double>(i % 7);
  }
  const auto eq = ddx::rt::equation([&] {
    const RE T = ddx::rt::var("T");
    const RE P = ddx::rt::var("P");
    const RV x = ddx::rt::vec("x", n);
    const auto Q = ddx::rt::vec_constant(q);
    const RV theta = Q * x / dot(Q, x);
    const auto tau = exp(-(ddx::rt::mat_constant(n, n, a0) +
                           T * ddx::rt::mat_constant(n, n, a1) +
                           P * ddx::rt::mat_constant(n, n, ap)) /
                         (R * T));
    return -sum(Q * x * log(gemv(tau, theta, true)));
  });
  ASSERT_FALSE(eq.poisoned()) << eq.status()->code;
  auto at = composition(n);
  at.insert(at.begin(), {310.0, 2.0});
  expect_dense_matches(eq, at);
  EXPECT_LE(eq.dense_hessian_cost()->n3_terms, 1u);
}

// A scalar that moves with x and T multiplying the matrix's product.
TEST(RtMatrix, DenseHessianOverAScaleThatMovesWithBoth) {
  constexpr std::size_t n = 12;
  const auto b = pair_table(n, 16, -0.5, 0.5);
  const auto eq = ddx::rt::equation([&] {
    const RE T = ddx::rt::var("T");
    const RV x = ddx::rt::vec("x", n);
    const RE I = sqrt(sum(x * x));
    const auto B = exp(T * ddx::rt::mat_constant(n, n, b));
    return (I / (1.0 + T * I)) * dot(x, gemv(B, x));
  });
  ASSERT_FALSE(eq.poisoned()) << eq.status()->code;
  const auto at = at_temperature(n, 1.1);
  expect_dense_matches(eq, at);
}

// Not only matrices: any vector beside a scalar symbol.  At fixed T the
// double sum's gemv reads the identity, where a stacked tangent was a MatMul.
TEST(RtMatrix, DenseHessianOverAVectorBesideAScalar) {
  constexpr std::size_t n = 30;
  const auto eq = ddx::rt::equation([] {
    const RV x = ddx::rt::vec("x", n);
    const RE T = ddx::rt::var("T");
    return debye_hueckel_and_double_sum(x) * (1.0 + 0.01 * T) + T * log(T);
  });
  ASSERT_FALSE(eq.poisoned()) << eq.status()->code;
  auto at = composition(n);
  at.push_back(1.3);
  expect_dense_matches(eq, at);
  EXPECT_EQ(eq.dense_hessian_cost()->n3_terms, 0u);
}

// Mostly scalars: every symbol is swept, as when none is a vector.
TEST(RtMatrix, DenseHessianOverMostlyScalars) {
  const auto eq = ddx::rt::equation([] {
    const RV x = ddx::rt::vec("x", 2);
    const RE a = ddx::rt::var("a");
    const RE b = ddx::rt::var("b");
    return sum(exp(a * x)) * b + a * a * b;
  });
  ASSERT_FALSE(eq.poisoned()) << eq.status()->code;
  const std::vector<double> at{0.3, 0.7, 1.1, -0.4};
  expect_dense_matches(eq, at, 1e-12);
}

// x inside the exponential: the matrix's tangent is rows x cols x n, so the
// dense Hessian refuses, and the coloured one and hvp still answer.
TEST(RtMatrix, DenseHessianRefusesACompositionDependentMatrix) {
  constexpr std::size_t n = 20;
  const auto b = pair_table(n, 10, -1.0, 1.0);
  const auto got = ddx::rt::equation([&] {
    const RV x = ddx::rt::vec("x", n);
    return dot(
        x, gemv(exp(ddx::rt::outer(x, x)) * ddx::rt::mat_constant(n, n, b), x));
  });
  const auto want = ddx::rt::equation([&] {
    const auto x = symbols(n);
    RE acc{0};
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        acc += x[i] * (exp(x[i] * x[j]) * b[i + j * n]) * x[j];
      }
    }
    return acc;
  });
  const auto at = composition(n);
  EXPECT_EQ(got.dense_hessian(std::span<const double>{at}).error().code,
            ddx::errc::tensor_shape);
  EXPECT_FALSE(got.dense_hessian_cost().has_value()) << "no block, no cost";
  expect_same_derivatives(want, got, at, 1e-11);
}

// No adjoint or tangent is built for what nothing moves: a param, a constant
// vector, a table.  Counted on the arena, as a node a sweep builds and nobody
// reads is computed nowhere but costs every later sweep its interning.
TEST(RtTensor, SweepsBuildNothingForWhatHoldsStill) {
  const auto count = [](const ddx::rt::Arena<> &b, ddx::rt::OpCode op) {
    return std::ranges::count(b.nodes(), op, &ddx::rt::Node<double>::op);
  };
  {
    ddx::rt::Arena<> b;
    const RE s = ddx::rt::param(b, "s");
    const RV x = ddx::rt::vec(b, "x", 3);
    const auto root = sum(s * x).id(b);
    (void)ddx::rt::derive_jacobian(b, std::span{&root, 1});
    EXPECT_EQ(count(b, ddx::rt::OpCode::Dot), 0) << "an adjoint for the param";
  }
  {
    ddx::rt::Arena<> b;
    const std::vector<double> c{1.0, 2.0, 3.0};
    const RV x = ddx::rt::vec(b, "x", 3);
    const RV k = ddx::rt::vec_constant(b, std::span<const double>{c});
    const auto root = sum(gemv(ddx::rt::outer(x, k), k)).id(b);
    (void)ddx::rt::tangent_sweep(b, std::span{&root, 1},
                                 ddx::rt::detail::seeds_along(b));
    EXPECT_EQ(count(b, ddx::rt::OpCode::Outer), 2)
        << "the primal and d(x) k^T, and no x d(k)^T over a zero row";
  }
}

TEST(RtMatrix, RefusesMismatchedShapesAndMatrixRoots) {
  const std::vector<double> six(6, 1.0);
  const auto transposed = ddx::rt::equation([&] {
    const RE T = ddx::rt::var("T");
    const auto m =
        ddx::rt::mat_constant(2, 3, six) + T * ddx::rt::mat_constant(3, 2, six);
    return sum(m);
  });
  EXPECT_EQ(transposed.status()->code, ddx::errc::tensor_shape);

  const auto not_a_vector = ddx::rt::equation([&] {
    const auto m = exp(ddx::rt::var("T") * ddx::rt::mat_constant(2, 3, six));
    return sum(ddx::rt::outer(m, ddx::rt::vec("x", 2)));
  });
  EXPECT_EQ(not_a_vector.status()->code, ddx::errc::tensor_shape);

  ddx::rt::Arena<> b;
  const RE T = ddx::rt::var(b, "T");
  const RE m = T * ddx::rt::mat_constant(b, 2, 3, std::span<const double>{six});
  EXPECT_EQ(ddx::rt::equation(RE{b, m.id()}).status()->code,
            ddx::errc::tensor_shape);

  // A table has no width, so nothing that counts widths may take it for a
  // scalar: not as an output, an element, or a moved entry.
  const auto table =
      ddx::rt::equation([&] { return ddx::rt::mat_constant(2, 3, six); });
  EXPECT_EQ(table.status()->code, ddx::errc::tensor_shape);
  const auto element = ddx::rt::equation([&] {
    const std::array elements{ddx::rt::var("T"),
                              ddx::rt::mat_constant(2, 3, six)};
    return sum(ddx::rt::pack(std::span<const RE>{elements}));
  });
  EXPECT_EQ(element.status()->code, ddx::errc::tensor_shape);
  const auto moved = ddx::rt::equation([&] {
    return sum(ddx::rt::detail::add_at(ddx::rt::vec("x", 2),
                                       ddx::rt::mat_constant(2, 3, six), 0));
  });
  EXPECT_EQ(moved.status()->code, ddx::errc::tensor_shape);
}

TEST(RtMatrix, BatchesAndThreadsAreBitIdentical) {
  constexpr std::size_t n = 9;
  const auto b = pair_table(n, 11, -0.5, 0.5);
  const auto eq = ddx::rt::equation([&] {
    const RE T = ddx::rt::var("T");
    const RV x = ddx::rt::vec("x", n);
    return dot(x, gemv(exp(T * ddx::rt::mat_constant(n, n, b)), x)) * log(T);
  });
  ASSERT_FALSE(eq.poisoned());
  expect_batch_matches(eq, batch_jacobian(eq, n + 1, 64, [](std::size_t p) {
                         return at_temperature(
                             n, 1.0 + 0.01 * static_cast<double>(p));
                       }));

  const auto at = at_temperature(n, 1.2);
  const auto d = direction(n + 1);
  expect_threads_agree([&] { return *eq.hvp(d, std::span<const double>{at}); });
}

// What the plan claims of the structure, counted rather than timed.
namespace {
// The block's own cone, past the gradient's: how many elements a point writes
// and how many of the row-by-row nodes it holds.
struct Census {
  std::size_t elements = 0;
  std::size_t row_reads = 0; // RowOf: a row read back out of a matrix
};
Census census(const ddx::rt::Arena<> &b, const ddx::rt::DenseHessian &h) {
  const auto block = ddx::rt::detail::reachable(b, std::span{&h.block, 1});
  const auto gradient = ddx::rt::detail::reachable(b, h.partial);
  Census c;
  for (const ddx::rt::NodeId v : ddx::rt::detail::live_ids(block)) {
    const auto op = b.op_of(v);
    // A table's MatRow is read in place, never written.
    const bool in_place =
        ddx::rt::is_table(b, v) && !ddx::rt::detail::is_transposed(b[v]);
    if (gradient[v] || !ddx::rt::detail::is_tensor(b[v]) || in_place) {
      continue;
    }
    c.elements += ddx::rt::width_of(b, v);
    c.row_reads += op == ddx::rt::OpCode::RowOf;
  }
  return c;
}
} // namespace

// The gradient of a vector model is the Pack's adjoint read element by
// element, so the block is that vector's tangent summed as matrices: no row is
// read back.  Stacking the partials' rows wrote 18.2 n^2 here, 2 n^2 of it
// strided; the sum writes 7.3 n^2 -- the Dense terms (4: C read in place, C^T
// copied), the outer products as one low-rank product (1), one add per
// further matrix (2).
TEST(RtTensorDense, TheBlockIsSummedNotStacked) {
  constexpr std::size_t n = 200;
  ddx::rt::Arena<> b;
  const auto arena = ddx::rt::detail::scoped_arena(b); // as equation() runs it
  const RE f = debye_hueckel_and_double_sum(ddx::rt::vec(b, "x", n));
  ASSERT_FALSE(f.poisoned());
  const auto h = ddx::rt::derive_dense_hessian(b, f.id(b));
  const Census c = census(b, h);
  EXPECT_EQ(c.row_reads, 0u);
  std::printf("[ ddx ] dense block, Debye-Hueckel + double sum, n = %zu: "
              "%.2f n^2 elements written\n",
              n, static_cast<double>(c.elements) / (n * n));
  EXPECT_LE(c.elements, 8 * n * n);
  EXPECT_NE(b.op_of(h.block), ddx::rt::OpCode::StackRows);

  // A scalar model's partials are no vector's elements: rows, stacked.
  ddx::rt::Arena<> scalar;
  const auto x = std::views::iota(0uz, 6uz) |
                 std::views::transform([&](std::size_t i) {
                   return ddx::rt::var(scalar, std::format("x[{}]", i));
                 }) |
                 ddx::impl::to<std::vector<RE>>();
  const RE g = models::mse(x);
  ASSERT_FALSE(g.poisoned());
  const auto hs = ddx::rt::derive_dense_hessian(scalar, g.id(scalar));
  EXPECT_EQ(scalar.op_of(hs.block), ddx::rt::OpCode::StackRows);
}

// exp(T B) is a few nodes whatever n is, and its gradient stays O(n): the
// matrix is one row, not n^2 scalars.
TEST(RtMatrix, NodesDoNotGrowWithTheSquare) {
  const auto nodes = [](std::size_t n) {
    const auto table = pair_table(n, 12, -0.5, 0.5);
    ddx::rt::Arena<> b;
    const RE T = ddx::rt::var(b, "T");
    const RV x = ddx::rt::vec(b, "x", n);
    const auto B =
        ddx::rt::mat_constant(b, n, n, std::span<const double>{table});
    const RE f = dot(x, gemv(exp(T * B), x));
    EXPECT_FALSE(f.poisoned());
    const std::size_t model = b.size() - n; // past the n Var leaves
    (void)ddx::rt::build_reverse_jacobian(b, f.id(b));
    return std::pair{model, b.size()};
  };
  const auto [model_10, gradient_10] = nodes(10);
  const auto [model_100, gradient_100] = nodes(100);
  EXPECT_EQ(model_10, model_100);
  // Each further species costs its Var and its gradient element, and nothing
  // per pair.
  EXPECT_LE(gradient_100 - gradient_10, 3 * 90u);
}

// A released thread's tapes are empty, and the next sweep grows them back to
// the same bits.
TEST(RtTensor, ReleasedTapesGrowBackToTheSameBits) {
  constexpr std::size_t n = 7;
  const auto eq = tensor_mse(n);
  ASSERT_FALSE(eq.poisoned());
  const auto sweep = [&] { // the block_lanes path, with a tail
    const Batch r = batch_jacobian(eq, n, 64, [&](std::size_t p) { return drifted(n, p); });
    return std::pair{r.f, r.g};
  };
  const auto before = sweep();
  const auto &kept = ddx::rt::detail::thread_tapes<double>();
  ASSERT_GT(kept.lane_vtape.capacity(), 0u);
  ddx::rt::release_tapes();
  EXPECT_EQ(kept.lane_tape.capacity(), 0u);
  EXPECT_EQ(kept.lane_vtape.capacity(), 0u);
  EXPECT_EQ(sweep(), before);
}

// The numbers X2 and X1 are judged by, printed for a release build to read:
// the dense block without the UNIQUAC gemm, and a matrix of temperature.
TEST(RtMatrix, CostAtAThousandSpecies) {
  constexpr std::size_t n = 1000;
  const auto at = composition(n);
  const auto ms = [](auto t0, auto t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  const auto dh = ddx::rt::equation(
      [] { return debye_hueckel_and_double_sum(ddx::rt::vec("x", n)); });
  ASSERT_TRUE(dh.prepare({ddx::rt::Want::DenseHessian}).has_value());
  (void)dh.dense_hessian(std::span<const double>{at}); // warm
  const auto t0 = std::chrono::steady_clock::now();
  ASSERT_TRUE(dh.dense_hessian(std::span<const double>{at}).has_value());
  const auto t1 = std::chrono::steady_clock::now();

  const auto b1 = pair_table(n, 6, -1.0, 1.0);
  const auto b2 = pair_table(n, 7, -50.0, 50.0);
  const auto b5 = pair_table(n, 8, -0.3, 0.3);
  const auto b6 = pair_table(n, 9, -0.01, 0.01);
  const auto mt = ddx::rt::equation([&] {
    const RE T = ddx::rt::param("T");
    const RV x = ddx::rt::vec("x", n);
    const auto M = ddx::rt::mat_constant(n, n, b1) +
                   ddx::rt::mat_constant(n, n, b2) / T +
                   ddx::rt::mat_constant(n, n, b5) *
                       exp(T * ddx::rt::mat_constant(n, n, b6));
    return dot(x, gemv(M, x)) + sum(x * log(x));
  });
  auto point = at;
  point.push_back(298.15);
  const std::span<const double> p{point};
  ASSERT_TRUE(mt.prepare({ddx::rt::Want::Gradient, ddx::rt::Want::DenseHessian})
                  .has_value());
  (void)mt.gradient(p);
  (void)mt.dense_hessian(p);
  const auto t2 = std::chrono::steady_clock::now();
  ASSERT_TRUE(mt.gradient(p).has_value());
  const auto t3 = std::chrono::steady_clock::now();
  ASSERT_TRUE(mt.dense_hessian(p).has_value());
  const auto t4 = std::chrono::steady_clock::now();

  const auto mv = ddx::rt::equation([&] {
    const RE T = ddx::rt::var("T");
    const RV x = ddx::rt::vec("x", n);
    const auto M = ddx::rt::mat_constant(n, n, b1) +
                   ddx::rt::mat_constant(n, n, b2) / T +
                   ddx::rt::mat_constant(n, n, b5) *
                       exp(T * ddx::rt::mat_constant(n, n, b6));
    return dot(x, gemv(M, x)) + sum(x * log(x));
  });
  auto tx = at;
  tx.insert(tx.begin(), 298.15);
  const std::span<const double> q{tx};
  ASSERT_TRUE(mv.prepare({ddx::rt::Want::DenseHessian}).has_value());
  (void)mv.dense_hessian(q);
  const auto t5 = std::chrono::steady_clock::now();
  ASSERT_TRUE(mv.dense_hessian(q).has_value());
  const auto t6 = std::chrono::steady_clock::now();
  std::printf("[ ddx ] n = %zu: Debye-Hueckel + double sum dense_hessian "
              "%.2f ms; M(T) gradient %.2f ms, dense_hessian %.2f ms; T a "
              "var, dense_hessian %.2f ms\n",
              n, ms(t0, t1), ms(t2, t3), ms(t3, t4), ms(t5, t6));
}
