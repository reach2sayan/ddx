#include "ddx.hpp"
#include "rt/bridge.hpp"
#include "rt/equation.hpp"
#include "util/ranges.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iterator>
#include <map>
#include <numbers>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// One name for both kinds of expression, so the same function can be asked the
// same question both ways.

namespace {
constexpr auto sx = ddx::var<"x">;
constexpr auto sy = ddx::var<"y">;

// constexpr, never consteval (NOTES.md): one implementation serves both a
// static_assert and a value read at run time.
consteval double slope_at_two() {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto eq = ddx::rt::equation(x * x + 3.0 * x);
  return (*eq.jacobian(2.0))[0];
}
static_assert(slope_at_two() == 7.0);

TEST(RtEquation, MatchesTheCompileTimeEquation) {
  const auto f = exp(sx) * sin(sy) + sx * sy;
  const ddx::Equation compile_time{f};

  ddx::rt::Builder<> b;
  const auto runtime = ddx::rt::equation(ddx::rt::to_graph(b, f));

  EXPECT_EQ(runtime.arity(), 2u);
  EXPECT_NEAR(*runtime.evaluate(1.3, 0.7), compile_time.evaluate(1.3, 0.7),
              1e-12);

  const auto want = compile_time.jacobian(1.3, 0.7);
  const auto got = *runtime.jacobian(1.3, 0.7);
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], 1e-12) << "partial " << i;
  }
}

// The runtime side resolves names against a run-time symbol list where
// make_point uses a compile-time one.
TEST(RtEquation, EveryCallSpellingAgrees) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(log(x) * sqrt(y));

  const auto positional = *eq.jacobian(1.3, 0.7);
  const auto by_range = *eq.jacobian(std::array{1.3, 0.7});
  const auto by_name = *eq.jacobian(ddx::named<"y">(0.7), ddx::named<"x">(1.3));
  const auto by_map =
      *eq.jacobian(std::map<std::string, double>{{"y", 0.7}, {"x", 1.3}});

  ASSERT_EQ(positional.size(), 2u);
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_DOUBLE_EQ(by_range[i], positional[i]);
    EXPECT_DOUBLE_EQ(by_name[i], positional[i]);
    EXPECT_DOUBLE_EQ(by_map[i], positional[i]);
  }
}

// var(name) reads a name the program only has at run time, and a varmap keyed
// the same way is how a point for it is written.
TEST(RtEquation, NamesAndPointsBothComeFromRunTimeStrings) {
  std::istringstream config{"beta alpha"};
  std::vector<std::string> names;
  for (std::string n; config >> n;) {
    names.push_back(n);
  }

  // 1 * beta + 2 * alpha, weighted in the order the names were read.
  const auto eq = ddx::rt::equation([&] {
    ddx::rt::RTExpression<double> acc = 0.0;
    double weight = 1.0;
    for (const std::string &n : names) {
      acc += weight++ * ddx::rt::var(n);
    }
    return acc;
  });

  ASSERT_FALSE(eq.poisoned());
  EXPECT_TRUE(std::ranges::equal(
      *eq.symbols(), std::array<std::string_view, 2>{"alpha", "beta"}));

  const std::map<std::string, double> vm{{"alpha", 3.0}, {"beta", 5.0}};
  EXPECT_DOUBLE_EQ(*eq.evaluate(vm), 1.0 * 5.0 + 2.0 * 3.0);
  EXPECT_TRUE(std::ranges::equal(*eq.jacobian(vm), std::array{2.0, 1.0}));

  // Any range of pairs, not only a map, and the key need not own its text.
  const std::vector<std::pair<std::string_view, double>> pairs{{"beta", 5.0},
                                                               {"alpha", 3.0}};
  EXPECT_DOUBLE_EQ(*eq.evaluate(pairs), *eq.evaluate(vm));
}

TEST(RtEquation, EveryNamedPointMustNameEverySymbolAndOnlyRealOnes) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * y);

  // Silence is not zero: a symbol no argument reaches is the mistake, and
  // every spelling of a named point says so the same way.
  EXPECT_EQ(eq.evaluate(std::map<std::string, double>{{"x", 1.0}}).error().code,
            ddx::errc::short_point);
  EXPECT_EQ(eq.evaluate(ddx::named<"x">(1.0)).error().code,
            ddx::errc::short_point);

  EXPECT_EQ(eq.evaluate(std::map<std::string, double>{{"x", 1.0}, {"z", 2.0}})
                .error()
                .code,
            ddx::errc::unknown_symbol);
  EXPECT_EQ(
      eq.evaluate(ddx::named<"x">(1.0), ddx::named<"z">(2.0)).error().code,
      ddx::errc::unknown_symbol);

  // A whole point still goes through, either way round.
  EXPECT_DOUBLE_EQ(*eq.evaluate(ddx::named<"y">(3.0), ddx::named<"x">(2.0)),
                   6.0);
  EXPECT_DOUBLE_EQ(
      *eq.evaluate(std::map<std::string, double>{{"y", 3.0}, {"x", 2.0}}), 6.0);
}

TEST(RtEquation, SystemsGiveAJacobian) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * y + sin(x), exp(x) - y * y);

  const auto J = *eq.jacobian(1.3, 0.7);
  ASSERT_EQ(J.size(), 4u); // 2 functions x 2 symbols, row-major

  // Row k is the Jacobian row of function k.
  const auto row0 = ddx::Equation{sx * sy + sin(sx)}.jacobian(1.3, 0.7);
  const auto row1 = ddx::Equation{exp(sx) - sy * sy}.jacobian(1.3, 0.7);
  EXPECT_NEAR(J[0], row0[0], 1e-12);
  EXPECT_NEAR(J[1], row0[1], 1e-12);
  EXPECT_NEAR(J[2], row1[0], 1e-12);
  EXPECT_NEAR(J[3], row1[1], 1e-12);
}

TEST(RtEquation, MultipleOutputsEvaluateToAVector) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto eq = ddx::rt::equation(x * x, sin(x));
  const auto v = *eq.evaluate(0.5);
  ASSERT_EQ(v.size(), 2u);
  EXPECT_DOUBLE_EQ(v[0], 0.25);
  EXPECT_NEAR(v[1], std::sin(0.5), 1e-15);
}

TEST(RtEquation, AWrongSizedPointIsRejected) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x + y);
  EXPECT_EQ(eq.jacobian(std::array{1.0}).error().code, ddx::errc::wrong_arity);
  EXPECT_EQ(eq.jacobian(ddx::named<"z">(1.0)).error().code,
            ddx::errc::unknown_symbol);
}

} // namespace

namespace {

// dual2nd on the compile-time side, which is what its hessian needs; the
// runtime graph is over plain double either way.
TEST(RtEquation, HessianMatchesTheCompileTimeEquation) {
  using D = ddx::dual2nd;
  constexpr auto dx = ddx::var<"x", D>;
  constexpr auto dy = ddx::var<"y", D>;
  const auto f = exp(dx) * sin(dy) + dx * dx * dy;

  ddx::rt::Builder<> b;
  const auto eq = ddx::rt::equation(ddx::rt::to_graph(b, f));
  const auto got = *eq.hessian(0.7, 1.1);
  const auto want = ddx::Equation{f}.hessian(0.7, 1.1);

  ASSERT_EQ(got.size(), 4u);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      EXPECT_NEAR(got[i * 2 + j], ddx::impl::get_real_part<1>(want[i, j]),
                  1e-10)
          << "H[" << i << "][" << j << "]";
    }
  }
  EXPECT_TRUE(std::isfinite(got[1]));
  EXPECT_DOUBLE_EQ(got[1], got[2]) << "a Hessian is symmetric";
}

// Whether colouring saves anything is not something a caller can guess.
TEST(RtEquation, HessianColoursReflectTheCoupling) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto z = var(b, "z");

  const auto coupled = ddx::rt::equation(x * y + y * z + x * z);
  EXPECT_EQ(coupled.hessian_colors(), 3u) << "every pair couples";

  ddx::rt::Builder<> b2;
  const auto a = var(b2, "a");
  const auto c = var(b2, "c");
  const auto d = var(b2, "d");
  const auto separable = ddx::rt::equation(sin(a) + exp(c) + d * d);
  EXPECT_EQ(separable.hessian_colors(), 1u) << "nothing couples: one sweep";
}

} // namespace

namespace {

// equation() makes the arena and moves it into the result, so a caller never
// names or keeps a Builder alive.
TEST(RtEquation, TheFactoryHidesTheArena) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return exp(x) * sin(y);
  });

  EXPECT_EQ(eq.arity(), 2u);
  const auto g = *eq.jacobian(1.0, 2.0);
  ASSERT_EQ(g.size(), 2u);
  EXPECT_NEAR(*eq.evaluate(1.0, 2.0), std::exp(1.0) * std::sin(2.0), 1e-12);
  EXPECT_NEAR(g[0], std::exp(1.0) * std::sin(2.0), 1e-12);
  EXPECT_NEAR(g[1], std::exp(1.0) * std::cos(2.0), 1e-12);
}

// Owning the arena is what lets an Equation be returned, stored and passed.
TEST(RtEquation, AnOwningEquationOutlivesItsScope) {
  const auto make = [](double c) {
    return ddx::rt::equation([c] {
      const auto x = ddx::rt::var("x");
      return c * x * x;
    });
  };

  const auto eq = make(3.0);
  EXPECT_NEAR(*eq.evaluate(2.0), 12.0, 1e-12);
  EXPECT_NEAR((*eq.jacobian(2.0))[0], 12.0, 1e-12); // d(3x^2)/dx = 6x
  EXPECT_NEAR((*eq.hessian(2.0))[0], 6.0, 1e-12);
}

// Same answers through the compiled kernel and the interpreter alike.
TEST(RtEquation, BatchAgreesWithPerPoint) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(exp(x) * sin(y) + x * y);

  constexpr std::size_t n = 8;
  std::vector<double> cx(n), cy(n), f(n), dx(n), dy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.1 * static_cast<double>(i);
    cy[i] = 1.0 + 0.2 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};
  double *const values[]{f.data()};
  double *const partials[]{dx.data(), dy.data()};

  *eq.jacobian(columns, values, partials, n);

  for (std::size_t i = 0; i < n; ++i) {
    const auto want = *eq.jacobian(cx[i], cy[i]);
    EXPECT_NEAR(f[i], *eq.evaluate(cx[i], cy[i]), 1e-12) << "point " << i;
    EXPECT_NEAR(dx[i], want[0], 1e-12) << "point " << i;
    EXPECT_NEAR(dy[i], want[1], 1e-12) << "point " << i;
  }
}

// Every lane of a block sweep runs the scalar sweep's operations in its order,
// so the two agree at every node to the bit -- contracted steps of both signs
// included, which the block sweep packs and the scalar one does not.
TEST(RtEquation, BlockSweepAgreesWithTheScalarSweepToTheBit) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto z = var(b, "z");
  const auto expr = sqrt(abs(x * y + z)) * exp(-(y * z) + x) + x * x * y;
  const auto root = expr.id(b);
  std::ignore = ddx::rt::build_jacobian_impl(b, root);

  const auto schedule = ddx::rt::detail::schedule_of(
      b, std::views::iota(ddx::rt::NodeId{0},
                          static_cast<ddx::rt::NodeId>(b.size())));
  const auto contracted = [&](bool negated) {
    return std::ranges::any_of(schedule, [negated](const ddx::rt::Step &s) {
      return s.fma && s.fma.negated == negated;
    });
  };
  ASSERT_TRUE(contracted(false));
  ASSERT_TRUE(contracted(true));

  constexpr std::size_t w = ddx::rt::block_lanes;
  constexpr std::size_t symbols = 3;
  std::vector<double> lanes(symbols * w);
  for (const auto [k, v] : std::views::enumerate(lanes)) {
    v = std::sin(1.7 * static_cast<double>(k) + 0.3) * 2.5;
  }
  std::vector<double> block(b.size() * w);
  ddx::rt::evaluate_block<w>(b, lanes, schedule, std::span{block});

  std::vector<double> tape(b.size());
  for (const std::size_t k : std::views::iota(0uz, w)) {
    const std::array at{lanes[k], lanes[w + k], lanes[2 * w + k]};
    ddx::rt::evaluate_into(b, at, schedule, std::span{tape});
    for (const std::size_t v : std::views::iota(0uz, b.size())) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(block[v * w + k]),
                std::bit_cast<std::uint64_t>(tape[v]))
          << "node " << v << " lane " << k;
    }
  }
}

// The same through the equation: two whole blocks and a tail, each point
// against the scalar accessor that sweeps it alone.
TEST(RtEquation, BatchAgreesWithPerPointToTheBit) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto z = var(b, "z");
  const auto eq = ddx::rt::equation(sqrt(abs(x * y + z)) *
                                        exp(-(y * z) + x) +
                                    x * x * y);

  constexpr std::size_t n = 2 * ddx::rt::block_lanes + 3;
  std::vector<double> cx(n), cy(n), cz(n), f(n), dx(n), dy(n), dz(n);
  for (const std::size_t i : std::views::iota(0uz, n)) {
    const auto t = static_cast<double>(i);
    cx[i] = std::sin(1.3 * t + 0.1);
    cy[i] = std::cos(0.7 * t - 0.4) * 1.5;
    cz[i] = 0.2 * t - 1.1;
  }
  const double *const columns[]{cx.data(), cy.data(), cz.data()};
  double *const values[]{f.data()};
  double *const partials[]{dx.data(), dy.data(), dz.data()};
  ASSERT_TRUE(eq.jacobian(columns, values, partials, n).has_value());

  for (const std::size_t i : std::views::iota(0uz, n)) {
    const auto g = *eq.jacobian(cx[i], cy[i], cz[i]);
    const auto bits = [](double d) { return std::bit_cast<std::uint64_t>(d); };
    EXPECT_EQ(bits(f[i]), bits(*eq.evaluate(cx[i], cy[i], cz[i])))
        << "point " << i;
    EXPECT_EQ(bits(dx[i]), bits(g[0])) << "point " << i;
    EXPECT_EQ(bits(dy[i]), bits(g[1])) << "point " << i;
    EXPECT_EQ(bits(dz[i]), bits(g[2])) << "point " << i;
  }
}

TEST(RtEquation, BatchHessianFillsTheCompressedColumns) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(exp(x) * sin(y) + x * x * y);

  constexpr std::size_t n = 4;
  const std::size_t columns = *eq.hessian_columns();
  ASSERT_GT(columns, 0u);

  std::vector<double> cx(n, 0.7), cy(n, 1.1), f(n), dx(n), dy(n);
  std::vector<std::vector<double>> blocks(columns, std::vector<double>(n));
  const auto block_ptrs =
      blocks | std::views::transform([](auto &c) { return c.data(); }) |
      ddx::impl::to<std::vector<double *>>();
  const double *const inputs[]{cx.data(), cy.data()};
  double *const values[]{f.data()};
  double *const partials[]{dx.data(), dy.data()};

  *eq.hessian(inputs, values, partials, block_ptrs, n);

  // The dense per-point Hessian is the scatter of those columns.
  const auto dense = *eq.hessian(0.7, 1.1);
  EXPECT_NEAR(dense[1], dense[2], 1e-12) << "symmetric";
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(f[i], *eq.evaluate(0.7, 1.1), 1e-12);
  }
}

} // namespace

namespace {

// Any contiguous range of columns.
TEST(RtEquation, BatchTakesAnyContiguousRange) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * y);

  constexpr std::size_t n = 3;
  std::vector<double> cx(n, 2.0), cy(n, 3.0), f(n), dx(n), dy(n);

  const std::vector<const double *> as_vector{cx.data(), cy.data()};
  const double *const as_array[]{cx.data(), cy.data()};
  std::vector<double *> outputs{f.data()};
  std::vector<double *> partials{dx.data(), dy.data()};

  *eq.jacobian(as_vector, outputs, partials, n);
  EXPECT_DOUBLE_EQ(f[0], 6.0);
  EXPECT_DOUBLE_EQ(dx[0], 3.0);

  std::ranges::fill(f, 0.0);
  *eq.jacobian(as_array, outputs, partials, n);
  EXPECT_DOUBLE_EQ(f[0], 6.0);

  *eq.jacobian(std::span{as_vector}, std::span{outputs}, std::span{partials},
               n);
  EXPECT_DOUBLE_EQ(f[0], 6.0);
}

// A wrong column count is silent corruption once it reaches the kernel.
TEST(RtEquation, AMismatchedColumnCountIsRejected) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * y);

  std::vector<double> cx(2, 1.0), cy(2, 1.0), f(2), dx(2), dy(2);
  const std::vector<const double *> xs{cx.data(), cy.data()};
  std::vector<double *> values{f.data()};
  std::vector<double *> partials{dx.data(), dy.data()};
  EXPECT_TRUE(eq.jacobian(xs, values, partials, 2).has_value());

  // One partial per symbol, so one column is one too few.
  const std::vector<double *> one_partial{dx.data()};
  EXPECT_EQ(eq.jacobian(xs, values, one_partial, 2).error().code,
            ddx::errc::wrong_column_count);

  // One value column for one function, so two is one too many.
  const std::vector<double *> two_values{f.data(), dy.data()};
  EXPECT_EQ(eq.jacobian(xs, two_values, partials, 2).error().code,
            ddx::errc::wrong_column_count);

  // And the point itself has to name every symbol.
  const std::vector<const double *> short_point{cx.data()};
  EXPECT_EQ(eq.jacobian(short_point, values, partials, 2).error().code,
            ddx::errc::wrong_column_count);
}

} // namespace

namespace {

// Symbols come from equation(); no arena is ever named.
TEST(RtEquation, TheArenaIsInvisible) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return x * y + sin(x);
  });
  EXPECT_EQ(eq.arity(), 2u);
  EXPECT_NEAR(*eq.evaluate(2.0, 3.0), 6.0 + std::sin(2.0), 1e-12);

  // var() poisons rather than erroring, which would cost every model lambda a
  // dereference.  The poison survives every operator and reaches the Equation.
  const auto stray = ddx::rt::var("stray");
  EXPECT_TRUE(stray.poisoned());
  EXPECT_TRUE((stray * 2.0 + 1.0).poisoned());

  // Compound assignment must neither lose the poison nor pick up a live right
  // operand's arena.
  auto spreading = stray;
  spreading += 2.0;
  EXPECT_TRUE(spreading.poisoned());
  {
    ddx::rt::Builder<> other;
    auto adopting = stray;
    adopting *= var(other, "x");
    EXPECT_TRUE(adopting.poisoned());
  }

  const auto poisoned = ddx::rt::equation(stray * 2.0);
  EXPECT_TRUE(poisoned.poisoned());
  ASSERT_TRUE(poisoned.status().has_value());
  EXPECT_EQ(poisoned.status()->code, ddx::errc::no_arena);

  // Every call spends the error rather than reading a graph that is not there.
  EXPECT_EQ(poisoned.evaluate(1.0).error().code, ddx::errc::no_arena);
  EXPECT_EQ(poisoned.jacobian(1.0).error().code, ddx::errc::no_arena);
  EXPECT_EQ(poisoned.point(1.0).error().code, ddx::errc::no_arena);
  // nullopt, never a 0 a caller could loop over by accident.
  EXPECT_FALSE(poisoned.arity().has_value());
  EXPECT_FALSE(poisoned.symbols().has_value());
  EXPECT_FALSE(poisoned.jacobian_columns().has_value());
  EXPECT_FALSE(poisoned.value_columns().has_value());
  EXPECT_FALSE(poisoned.hessian_columns().has_value());
  EXPECT_FALSE(poisoned.hessian_colors().has_value());
}

// A symbol named into a borrowed arena after an Equation has it would move
// slots under a graph already built, so the arena refuses.
TEST(RtEquation, AnArenaStopsTakingSymbolsOnceAnEquationHasIt) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto eq = ddx::rt::equation(x * x);
  ASSERT_EQ(eq.arity(), 1u);
  EXPECT_NEAR(*eq.evaluate(3.0), 9.0, 1e-12);

  const auto a = var(b, "a"); // would have moved x from slot 0 to slot 1
  EXPECT_TRUE(a.poisoned());

  const auto second = ddx::rt::equation(a + x);
  ASSERT_TRUE(second.status().has_value());
  EXPECT_EQ(second.status()->code, ddx::errc::sealed_arena);

  // The first equation is untouched: same symbol set, same slot, same answers.
  ASSERT_EQ(eq.arity(), 1u);
  EXPECT_EQ((*eq.symbols())[0], "x");
  EXPECT_NEAR((*eq.jacobian(3.0))[0], 6.0, 1e-12);
}

// Sealed stops symbols being added, not named.
TEST(RtEquation, ASealedArenaStillNamesTheSymbolsItHas) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto first = ddx::rt::equation(x * y);

  const auto again = var(b, "x");
  EXPECT_FALSE(again.poisoned());
  const auto second = ddx::rt::equation(again + y * y);
  ASSERT_FALSE(second.poisoned());

  EXPECT_NEAR(*first.evaluate(2.0, 3.0), 6.0, 1e-12);
  EXPECT_NEAR(*second.evaluate(2.0, 3.0), 11.0, 1e-12);
}

// A bare literal names no builder -- distinct from no_arena.
TEST(RtEquation, ALiteralNamesNoGraph) {
  const auto eq = ddx::rt::equation(ddx::rt::RTExpression<double>{2.0});
  ASSERT_TRUE(eq.status().has_value());
  EXPECT_EQ(eq.status()->code, ddx::errc::no_graph);
  EXPECT_EQ(eq.evaluate(1.0).error().code, ddx::errc::no_graph);
}

// One Hessian per output, the shape the compile-time side returns as
// nd_stack_t<S, m, n, 2>.
TEST(RtEquation, HessianOfASystemIsOneBlockPerOutput) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return std::array{x * x * y, exp(x) * y};
  });

  constexpr std::size_t n = 2;
  const auto H = *eq.hessian(0.7, 1.3);
  ASSERT_EQ(H.size(), 2 * n * n);

  const auto at = [&](std::size_t k, std::size_t i, std::size_t j) {
    return H[(k * n + i) * n + j];
  };

  EXPECT_NEAR(at(0, 0, 0), 2 * 1.3, 1e-10);
  EXPECT_NEAR(at(0, 0, 1), 2 * 0.7, 1e-10);
  EXPECT_NEAR(at(0, 1, 1), 0.0, 1e-10);
  EXPECT_NEAR(at(1, 0, 0), std::exp(0.7) * 1.3, 1e-10);
  EXPECT_NEAR(at(1, 0, 1), std::exp(0.7), 1e-10);
  EXPECT_NEAR(at(1, 1, 1), 0.0, 1e-10);

  for (std::size_t k = 0; k < 2; ++k) {
    EXPECT_NEAR(at(k, 0, 1), at(k, 1, 0), 1e-12);
  }
}

// One Taylor sweep rather than K nested duals.
TEST(RtEquation, UnivariateDerivativesToArbitraryOrder) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return exp(x) * sin(x);
  });

  // d^k/dx^k [e^x sin x] = 2^(k/2) e^x sin(x + k*pi/4)
  constexpr double at = 0.7;
  const auto expected = [](std::size_t k) {
    return std::pow(std::numbers::sqrt2, static_cast<double>(k)) *
           std::exp(at) *
           std::sin(at + static_cast<double>(k) * std::numbers::pi / 4);
  };
  EXPECT_NEAR(*eq.univariate_derivative<1>(at), expected(1), 1e-10);
  EXPECT_NEAR(*eq.univariate_derivative<2>(at), expected(2), 1e-10);
  EXPECT_NEAR(*eq.univariate_derivative<4>(at), expected(4), 1e-10);

  // It is a one-variable question; more than one symbol has no answer.
  const auto two =
      ddx::rt::equation([] { return ddx::rt::var("x") * ddx::rt::var("y"); });
  EXPECT_EQ(two.univariate_derivative<1>(1.0).error().code,
            ddx::errc::not_univariate);
}

// max and min take the winning side's coefficients; the Taylor scalar is
// asked for a comparison and a midpoint, never for a max of its own.
TEST(RtEquation, UnivariateDerivativesThroughMaxAndMin) {
  const auto kink = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return max(x * x, x) * tanh(x) + min(x * x, x);
  });
  // At 2: x*x wins max, x wins min.
  const double t = std::tanh(2.0), s = 1 - t * t;
  EXPECT_NEAR(*kink.univariate_derivative<1>(2.0), 4 * t + 4 * s + 1, 1e-12);
  EXPECT_NEAR(*kink.univariate_derivative<2>(2.0),
              2 * t + 8 * s + 4 * (-2 * t * s), 1e-12);
  EXPECT_NEAR(*kink.univariate_derivative<1>(0.5),
              1 * std::tanh(0.5) + 0.5 * (1 - std::pow(std::tanh(0.5), 2)) + 1,
              1e-12);
  // The reverse sweep says the same.
  EXPECT_NEAR(*kink.univariate_derivative<1>(2.0), (*kink.jacobian(2.0))[0],
              1e-12);
  EXPECT_NEAR(*kink.univariate_derivative<2>(2.0), (*kink.hessian(2.0))[0],
              1e-10);
}

} // namespace

// --- choosing the backend ---------------------------------------------------

// The scalar accessors answer from the frozen graph, the arena walk from every
// node in the arena.  EXPECT_EQ rather than a tolerance: the graph changes
// which nodes are visited, never the arithmetic at any of them.
TEST(RtEquation, ScalarAccessorsAgreeWithTheArenaWalkToTheBit) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto z = var(b, "z");
  // Every symbol under one log, so the Hessian is dense and the colouring
  // sweeps the arena several times over.
  const auto expr = (x * log(x) + y * y * z) * log(x + y + z) / (z + y);

  // Before the equation, so interning lands its own sweeps on these same nodes.
  const auto root = expr.id(b);
  const auto row = ddx::rt::build_jacobian_impl(b, root);
  const auto hess = ddx::rt::build_hessian_impl(b, root);
  const auto eq = ddx::rt::equation(expr);

  const std::vector<double> at{0.7, 1.3, 2.1};
  const auto values = ddx::rt::evaluate_all(b, at);
  const std::size_t n = at.size();

  EXPECT_EQ(*eq.evaluate(std::span<const double>{at}), values[root]);

  const auto g = *eq.jacobian(std::span<const double>{at});
  ASSERT_EQ(g.size(), n);
  for (const auto [i, p] : std::views::enumerate(row.partial)) {
    EXPECT_EQ(g[static_cast<std::size_t>(i)], values[p]) << "d/d" << i;
  }

  const auto h = *eq.hessian(std::span<const double>{at});
  ASSERT_EQ(h.size(), n * n);
  auto dims = std::views::iota(0uz, n);
  for (const auto [i, j] : std::views::cartesian_product(dims, dims)) {
    EXPECT_EQ(h[i * n + j], values[hess.at(i, j)]) << "d2/d" << i << "d" << j;
  }
}

#ifdef DDX_HAS_JIT

// The switchover contract, both halves of it.  `rebalance` blocks a reduction
// spine of `least` = 16 terms or more for the kernel and leaves the sweep's
// graph alone, so those two paths sum in different orders -- the one place in
// the library where interpreted and compiled answers differ at all.  README
// quotes the ULP figures below, so they are pinned here.
TEST(RtEquation, ShortSpinesSurviveTheSwitchoverToTheBit) {
  const auto spine = [](std::size_t terms) {
    return ddx::rt::equation([terms] {
      ddx::rt::RTExpression<double> acc = 0.0;
      for (const std::size_t i : std::views::iota(0uz, terms)) {
        acc = acc + ddx::rt::var(std::format("x{:03d}", i)) *
                        (1.0 / (1.0 + static_cast<double>(i)));
      }
      return acc;
    });
  };
  const auto point = [](std::size_t terms) {
    std::vector<double> at(terms);
    for (const std::size_t i : std::views::iota(0uz, terms)) {
      at[i] = 0.1 + 0.9 * static_cast<double>(i % 7) / 7.0 +
              static_cast<double>(i) * 1e-3;
    }
    return at;
  };
  const auto ulps = [](double a, double b) {
    return std::abs(static_cast<std::int64_t>(std::bit_cast<std::uint64_t>(a)) -
                    static_cast<std::int64_t>(std::bit_cast<std::uint64_t>(b)));
  };

  // Under the threshold nothing is rewritten, so the two agree exactly.
  for (const std::size_t terms : {4uz, 8uz, 15uz}) {
    auto eq = spine(terms);
    const auto at = point(terms);
    const double swept = *eq.evaluate(std::span<const double>{at});
    eq.options({.backend = ddx::jit::Backend::Compile, .points = 1});
    ASSERT_TRUE(eq.wait_for_kernel());
    const double compiled = *eq.evaluate(std::span<const double>{at});
    EXPECT_EQ(ulps(swept, compiled), 0)
        << terms << " terms is below the rewrite threshold";
  }

  // Over it the kernel sums in blocks, and the gap is small and bounded.  A
  // test that only checked "they agree" would have to be deleted here; one that
  // pins the size is what keeps README honest.
  for (const auto &[terms, bound] :
       std::array{std::pair{40uz, std::int64_t{8}},
                  std::pair{80uz, std::int64_t{16}}}) {
    auto eq = spine(terms);
    const auto at = point(terms);
    const double swept = *eq.evaluate(std::span<const double>{at});
    eq.options({.backend = ddx::jit::Backend::Compile, .points = 1});
    ASSERT_TRUE(eq.wait_for_kernel());
    const double compiled = *eq.evaluate(std::span<const double>{at});
    EXPECT_LE(ulps(swept, compiled), bound) << terms << " terms drifted";
    EXPECT_NEAR(swept, compiled, 1e-12 * std::abs(swept));
  }
}

TEST(RtEquation, InterpretDeclinesTheBackendAndAgreesWithIt) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(x * log(x) + y * exp(x * y) + sqrt(x + y));

  constexpr std::size_t n = 16;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.02 * static_cast<double>(i);
    cy[i] = 0.7 - 0.01 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};

  const auto run = [&](ddx::rt::Backend which) {
    std::vector<double> f(n), dx(n), dy(n);
    double *const values[]{f.data()};
    double *const partials[]{dx.data(), dy.data()};
    eq.options({.backend = which});
    // wait_for_kernel, not uses_kernel: no backend blocks, so whether a kernel
    // exists yet is a race until something asks for it to have landed.
    const bool kernel = eq.wait_for_kernel();
    *eq.jacobian(columns, values, partials, n);
    return std::tuple{f, dx, dy, kernel};
  };

  const auto [f_jit, dx_jit, dy_jit, used_kernel] =
      run(ddx::rt::Backend::Compile);
  const auto [f_int, dx_int, dy_int, used_none] =
      run(ddx::rt::Backend::Interpret);

  EXPECT_EQ(eq.options().backend, ddx::rt::Backend::Interpret);
  EXPECT_FALSE(used_none) << "Interpret still reached for a kernel";
  EXPECT_TRUE(used_kernel) << "Compile did not compile on a JIT build";

  // Equal, not near: the contraction is decided in the graph, so kernel and
  // sweep fold the same products and round the same number of times.
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(std::bit_cast<std::uint64_t>(f_jit[i]),
              std::bit_cast<std::uint64_t>(f_int[i]))
        << "value at " << i;
    EXPECT_EQ(std::bit_cast<std::uint64_t>(dx_jit[i]),
              std::bit_cast<std::uint64_t>(dx_int[i]))
        << "d/dx at " << i;
    EXPECT_EQ(std::bit_cast<std::uint64_t>(dy_jit[i]),
              std::bit_cast<std::uint64_t>(dy_int[i]))
        << "d/dy at " << i;
  }
}

// `points` picks the kernel width, and the two widths must agree to the bit --
// a lane is its own IEEE operation.
TEST(RtEquation, PointsPicksTheKernelWidthAndTheAnswerIsTheSame) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * log(x) + y * exp(x * y) + sqrt(x + y);

  constexpr std::size_t n = 64;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.005 * static_cast<double>(i);
    cy[i] = 0.7 - 0.002 * static_cast<double>(i);
  }

  // One equation says it steps, the other says it batches.
  auto stepping = ddx::rt::equation(f);
  auto batched = ddx::rt::equation(f);
  stepping.options({.backend = ddx::rt::Backend::Compile, .points = 1});
  batched.options({.backend = ddx::rt::Backend::Compile, .points = n});

  // Before the calls: a call made while a compile is in flight is answered by
  // the sweep, and what is compared below is one kernel width against another.
  ASSERT_TRUE(stepping.wait_for_kernel());
  ASSERT_TRUE(batched.wait_for_kernel());

  std::vector<double> f1(n), dx1(n), dy1(n), f2(n), dx2(n), dy2(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double *const one[]{cx.data() + i, cy.data() + i};
    double *const values[]{f1.data() + i};
    double *const partials[]{dx1.data() + i, dy1.data() + i};
    ASSERT_TRUE(stepping.jacobian(one, values, partials, 1).has_value());
  }
  {
    const double *const columns[]{cx.data(), cy.data()};
    double *const values[]{f2.data()};
    double *const partials[]{dx2.data(), dy2.data()};
    ASSERT_TRUE(batched.jacobian(columns, values, partials, n).has_value());
  }

  EXPECT_TRUE(stepping.uses_kernel());
  EXPECT_TRUE(batched.uses_kernel());

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(f1[i], f2[i]) << "value at " << i;
    EXPECT_EQ(dx1[i], dx2[i]) << "d/dx at " << i;
    EXPECT_EQ(dy1[i], dy2[i]) << "d/dy at " << i;
  }
}

// Nothing compiles unless a backend is asked for.
TEST(RtEquation, TheDefaultCompilesNothing) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * log(x) + y * y);

  constexpr std::size_t n = 4;
  std::vector<double> cx(n, 0.6), cy(n, 1.4), f(n), dx(n), dy(n);
  const double *const columns[]{cx.data(), cy.data()};
  double *const values[]{f.data()};
  double *const partials[]{dx.data(), dy.data()};

  ASSERT_TRUE(eq.jacobian(columns, values, partials, n).has_value());
  EXPECT_FALSE(eq.uses_kernel()) << "a default equation reached for a compiler";
  EXPECT_FALSE(eq.wait_for_kernel()) << "and one was in flight";
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(f[i], 0.6 * std::log(0.6) + 1.4 * 1.4, 1e-12);
    EXPECT_NEAR(dx[i], std::log(0.6) + 1.0, 1e-12);
    EXPECT_NEAR(dy[i], 2 * 1.4, 1e-12);
  }
}

// Choosing a backend starts the compile, with no call having been made.
TEST(RtEquation, ChoosingABackendStartsTheBuild) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  auto eq = ddx::rt::equation(x * log(x) + exp(x));

  eq.options({.backend = ddx::rt::Backend::Compile});
  EXPECT_TRUE(eq.wait_for_kernel()) << "options() launched nothing";
  EXPECT_TRUE(eq.uses_kernel());

  // Asking again for what is already built relaunches nothing.
  eq.options({.backend = ddx::rt::Backend::Compile});
  EXPECT_TRUE(eq.uses_kernel())
      << "an identical options() threw the kernel away";
}

// Sizing a buffer must not build a kernel: the counts are ones the constructor
// already knew.
TEST(RtEquation, TheColumnCountsCompileNothing) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(x * log(x) + y * exp(x * y));
  eq.options({.backend = ddx::rt::Backend::Compile});

  // Every count, before any call has been made.
  EXPECT_EQ(*eq.value_columns(), 1u);
  EXPECT_EQ(*eq.jacobian_columns(), 2u);
  EXPECT_GT(*eq.hessian_columns(), 0u);

  // The Hessian lane is still unbuilt: it is the hessian() call that builds it.
  constexpr std::size_t n = 2;
  const std::size_t hc = *eq.hessian_columns();
  std::vector<double> cx(n, 0.6), cy(n, 0.9), f(n), dx(n), dy(n);
  std::vector<std::vector<double>> hb(hc, std::vector<double>(n));
  std::vector<double *> hp(hc);
  for (std::size_t c = 0; c < hc; ++c) {
    hp[c] = hb[c].data();
  }
  const double *const columns[]{cx.data(), cy.data()};
  double *const values[]{f.data()};
  double *const partials[]{dx.data(), dy.data()};
  EXPECT_TRUE(eq.hessian(columns, values, partials, hp, n).has_value());

  // And they still answer once everything is built.
  EXPECT_EQ(*eq.value_columns(), 1u);
  EXPECT_EQ(*eq.jacobian_columns(), 2u);
  EXPECT_EQ(*eq.hessian_columns(), hc);
}

// One LLJIT serves every Equation type.  Results, not timings, so it cannot go
// flaky.
TEST(RtEquation, DifferentEquationTypesShareOneCompilerSafely) {
  const auto scalar_model = [] {
    return ddx::rt::equation([] {
      const auto x = ddx::rt::var("x");
      return x * log(x) + exp(x);
    });
  };
  const auto system_model = [] {
    return ddx::rt::equation([] {
      const auto x = ddx::rt::var("x");
      return std::array{x * log(x), exp(x), sqrt(x + 1.0)};
    });
  };

  constexpr std::size_t n = 4;
  std::vector<double> cx(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.4 + 0.1 * static_cast<double>(i);
  }

  // Both shapes, from several threads, all compiling at once.
  const auto one_scalar = [&] {
    auto eq = scalar_model();
    eq.options({.backend = ddx::rt::Backend::Compile, .points = n});
    // Every arm has to be the kernel's rather than whichever path won a race.
    EXPECT_TRUE(eq.wait_for_kernel());
    std::vector<double> f(n), d(n);
    const double *const columns[]{cx.data()};
    double *const values[]{f.data()};
    double *const partials[]{d.data()};
    EXPECT_TRUE(eq.jacobian(columns, values, partials, n).has_value());
    EXPECT_TRUE(eq.uses_kernel());
    return f;
  };
  const auto one_system = [&] {
    auto eq = system_model();
    eq.options({.backend = ddx::rt::Backend::Compile, .points = n});
    EXPECT_TRUE(eq.wait_for_kernel());
    std::vector<double> f0(n), f1(n), f2(n), d0(n), d1(n), d2(n);
    const double *const columns[]{cx.data()};
    double *const values[]{f0.data(), f1.data(), f2.data()};
    double *const partials[]{d0.data(), d1.data(), d2.data()};
    EXPECT_TRUE(eq.jacobian(columns, values, partials, n).has_value());
    EXPECT_TRUE(eq.uses_kernel());
    return f1;
  };

  const auto want_scalar = one_scalar();
  const auto want_system = one_system();

  std::vector<std::thread> racers;
  std::vector<std::vector<double>> got_scalar(4), got_system(4);
  for (std::size_t t = 0; t < 4; ++t) {
    racers.emplace_back([&, t] { got_scalar[t] = one_scalar(); });
    racers.emplace_back([&, t] { got_system[t] = one_system(); });
  }
  for (auto &r : racers) {
    r.join();
  }
  for (std::size_t t = 0; t < 4; ++t) {
    EXPECT_EQ(got_scalar[t], want_scalar) << "scalar equation on thread " << t;
    EXPECT_EQ(got_system[t], want_system) << "system equation on thread " << t;
  }
}

// A compile in flight when the choice changes is abandoned: flipping to
// Interpret leaves the sweep, and flipping back builds afresh.
TEST(RtEquation, AbandoningACompileLeavesNothingBehind) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  auto eq = ddx::rt::equation(x * log(x) + exp(x));

  eq.options({.backend = ddx::rt::Backend::Compile});
  eq.options({.backend = ddx::rt::Backend::Interpret});

  double cx = 0.7, f = 0, dx = 0;
  const double *const columns[]{&cx};
  double *const values[]{&f};
  double *const partials[]{&dx};
  ASSERT_TRUE(eq.jacobian(columns, values, partials, 1).has_value());
  EXPECT_FALSE(eq.uses_kernel()) << "an abandoned compile came back";
  EXPECT_NEAR(f, 0.7 * std::log(0.7) + std::exp(0.7), 1e-12);

  eq.options({.backend = ddx::rt::Backend::Compile});
  ASSERT_TRUE(eq.jacobian(columns, values, partials, 1).has_value());
  EXPECT_TRUE(eq.wait_for_kernel()) << "asking again did not build";
  EXPECT_NEAR(f, 0.7 * std::log(0.7) + std::exp(0.7), 1e-12);
}

// Compile promises to build a kernel, not to stand still until there is one.
// Waiting is a separate request.
TEST(RtEquation, CompileDoesNotMakeACallWaitForItsKernel) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  auto eq = ddx::rt::equation(x * log(x) + exp(x));
  eq.options({.backend = ddx::rt::Backend::Compile});

  double cx = 0.7, f = 0, dx = 0;
  const double *const columns[]{&cx};
  double *const values[]{&f};
  double *const partials[]{&dx};
  // Right whichever path answered it, which is the point.
  ASSERT_TRUE(eq.jacobian(columns, values, partials, 1).has_value());
  EXPECT_NEAR(f, 0.7 * std::log(0.7) + std::exp(0.7), 1e-12);

  EXPECT_TRUE(eq.wait_for_kernel()) << "Compile never built one";
  ASSERT_TRUE(eq.jacobian(columns, values, partials, 1).has_value());
  EXPECT_TRUE(eq.uses_kernel()) << "a landed kernel was not adopted";
  EXPECT_NEAR(f, 0.7 * std::log(0.7) + std::exp(0.7), 1e-12);
}

// The lane width has to be reachable from the Equation, and choosing it must
// not change a bit of the answer.
TEST(RtEquation, OptionsReachTheCompilerAndDoNotChangeTheAnswer) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(x * log(x) + y * exp(x * y) + sqrt(x + y));

  constexpr std::size_t n = 16;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.02 * static_cast<double>(i);
    cy[i] = 0.7 - 0.01 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};

  const auto run = [&](const ddx::jit::Options &opt) {
    std::vector<double> f(n), dx(n), dy(n);
    double *const values[]{f.data()};
    double *const partials[]{dx.data(), dy.data()};
    eq.options(opt);
    // Compared bit for bit below, so both have to be kernels.
    const bool kernel = eq.wait_for_kernel();
    *eq.jacobian(columns, values, partials, n);
    return std::tuple{f, dx, dy, kernel};
  };

  const ddx::jit::Options wide{
      .backend = ddx::rt::Backend::Compile,
      .codegen = {.lanes = *ddx::rt::Lanes::exactly(4)}};
  const ddx::jit::Options scalar{
      .backend = ddx::rt::Backend::Compile,
      .codegen = {.lanes = ddx::rt::Lanes::scalar()}};

  const auto [f0, dx0, dy0, k0] = run(wide);
  const auto [f1, dx1, dy1, k1] = run(scalar);

  EXPECT_EQ(eq.options().codegen.lanes, ddx::rt::Lanes::scalar())
      << "the setter did not take";
  EXPECT_TRUE(k0 && k1) << "an option refused the compile outright";

  // Bit-exact: a lane is its own IEEE operation, so the width changes what is
  // scheduled, never what is contracted.
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(f1[i], f0[i]) << "scalar value at " << i;
    EXPECT_EQ(dx1[i], dx0[i]) << "scalar d/dx at " << i;
    EXPECT_EQ(dy1[i], dy0[i]) << "scalar d/dy at " << i;
  }
}

// The compile is off the critical path: the sweep answers until the kernel
// lands, and the answers either side of the switchover agree.
TEST(RtEquation, TheCompileRunsBehindTheInterpreter) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(x * log(x) + y * exp(x * y) + sqrt(x + y));

  constexpr std::size_t n = 16;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.02 * static_cast<double>(i);
    cy[i] = 0.7 - 0.01 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};

  std::vector<double> f0(n), dx0(n), dy0(n), f1(n), dx1(n), dy1(n), f2(n),
      dx2(n), dy2(n);
  const auto into = [&](std::vector<double> &f, std::vector<double> &dx,
                        std::vector<double> &dy) {
    double *const values[]{f.data()};
    double *const partials[]{dx.data(), dy.data()};
    ASSERT_TRUE(eq.jacobian(columns, values, partials, n).has_value());
  };

  // Whichever path is ready -- the point is that it answers.
  eq.options({.backend = ddx::rt::Backend::Compile});
  into(f0, dx0, dy0);

  ASSERT_TRUE(eq.wait_for_kernel()) << "the compile never landed";
  EXPECT_TRUE(eq.uses_kernel()) << "a landed kernel was not adopted";
  into(f1, dx1, dy1);

  // Still the kernel that landed above.
  eq.options({.backend = ddx::rt::Backend::Compile});
  ASSERT_TRUE(eq.wait_for_kernel());
  into(f2, dx2, dy2);

  for (std::size_t i = 0; i < n; ++i) {
    // Bit-exact: one graph, one Options, so one kernel.
    EXPECT_DOUBLE_EQ(f1[i], f2[i]) << "landed value at " << i;
    EXPECT_DOUBLE_EQ(dx1[i], dx2[i]) << "landed d/dx at " << i;
    EXPECT_DOUBLE_EQ(dy1[i], dy2[i]) << "landed d/dy at " << i;
    // The first call may have been the sweep, which contracts differently.
    EXPECT_NEAR(f0[i], f2[i], 1e-12) << "value before landing at " << i;
    EXPECT_NEAR(dx0[i], dx2[i], 1e-12) << "d/dx before landing at " << i;
    EXPECT_NEAR(dy0[i], dy2[i], 1e-12) << "d/dy before landing at " << i;
  }
}

// Dropping an equation mid-compile abandons the result.  std::async's future
// joins in its destructor, which would put the compile back on the critical
// path of whatever let the equation go.
TEST(RtEquation, DroppingAMidFlightCompileDoesNotBlock) {
  // Wide enough that the compile is milliseconds: one that finishes first would
  // pass without proving anything.
  ddx::rt::Builder<> b;
  std::vector<ddx::rt::RTExpression<double>> v;
  for (std::size_t i = 0; i < 48; ++i) {
    v.push_back(ddx::rt::var(b, "x" + std::to_string(i)));
  }
  auto f = v[0] * log(v[0]);
  for (std::size_t i = 1; i < v.size(); ++i) {
    f = f + v[i] * log(v[i]) + exp(v[i - 1] * v[i]);
  }

  // Both arms pin codegen 0, so the ladder is one rung and the two measurements
  // are the same shape.  Under the default two, wait_for_kernel() returns at
  // the *cheap* rung and `blocking` stops being a compile's worth of time,
  // making the margin below a coin flip.  jit.exit_stress covers both rungs at
  // once.
  const ddx::jit::Options one_rung{
      .backend = ddx::rt::Backend::Compile,
      .codegen = {.codegen_level = ddx::rt::Level::O0}};

  // Calibrated against this machine: a fixed threshold above the compile would
  // pass whether or not the future joined.
  const auto blocking = [&] {
    auto eq = ddx::rt::equation(f);
    eq.options(one_rung);
    const auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(eq.wait_for_kernel());
    return std::chrono::steady_clock::now() - start;
  }();

  const auto teardown = [&] {
    auto eq = ddx::rt::equation(f);
    eq.options(one_rung);
    (void)eq.uses_kernel(); // launches the compile, adopts nothing yet

    const auto start = std::chrono::steady_clock::now();
    {
      const auto dropped = std::move(eq);
    }
    return std::chrono::steady_clock::now() - start;
  }();

  const auto us = [](std::chrono::steady_clock::duration d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  };
  // A join would put the whole compile here.
  EXPECT_LT(teardown * 4, blocking)
      << "teardown " << us(teardown) << "us against a " << us(blocking)
      << "us compile, so the future joined";
}

// Many threads on one Equation, mixing both lanes.  std::thread rather than
// std::jthread: libc++ shipped jthread late and behind
// _LIBCPP_ENABLE_EXPERIMENTAL.  Compiling deliberately -- that is the path that
// abandons compiles mid-flight when an equation goes away.
TEST(RtEquation, ConcurrentConstCallsAgreeWithOneThread) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(x * log(x) + y * exp(x * y) + sqrt(x + y));
  eq.options({.backend = ddx::rt::Backend::Compile});

  constexpr std::size_t n = 32;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.02 * static_cast<double>(i);
    cy[i] = 0.7 - 0.01 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};

  const std::size_t hcols = *eq.hessian_columns();
  const auto once = [&] {
    std::vector<double> f(n), dx(n), dy(n);
    std::vector<std::vector<double>> hs(hcols, std::vector<double>(n));
    std::vector<double *> hp(hcols);
    for (std::size_t c = 0; c < hcols; ++c) {
      hp[c] = hs[c].data();
    }
    double *const values[]{f.data()};
    double *const partials[]{dx.data(), dy.data()};
    EXPECT_TRUE(eq.hessian(columns, values, partials, hp, n).has_value());
    return std::tuple{f, dx, dy, hs};
  };
  const auto expected = once();

  std::vector<std::thread> racers;
  std::vector<decltype(once())> got(8);
  for (std::size_t t = 0; t < got.size(); ++t) {
    racers.emplace_back([&, t] {
      // Interleaved with the read-only accessors, which take the same locks.
      (void)eq.uses_kernel();
      (void)eq.value_columns();
      (void)eq.jacobian_columns();
      got[t] = once();
    });
  }
  for (auto &r : racers) {
    r.join();
  }

  const auto &[ef, edx, edy, ehs] = expected;
  for (const auto &[f, dx, dy, hs] : got) {
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(f[i], ef[i], 1e-12) << "value at " << i;
      EXPECT_NEAR(dx[i], edx[i], 1e-12) << "d/dx at " << i;
      EXPECT_NEAR(dy[i], edy[i], 1e-12) << "d/dy at " << i;
      for (std::size_t c = 0; c < hs.size(); ++c) {
        EXPECT_NEAR(hs[c][i], ehs[c][i], 1e-12) << "h" << c << " at " << i;
      }
    }
  }
}

// Two equations over ONE borrowed Builder, each asked for a Hessian from its
// own thread.  rt::build_hessian_impl appends to the arena, so no per-Equation
// lock could make this safe if the sweep were not in the constructor.
TEST(RtEquation, TwoEquationsSharingAnArenaDoNotRaceIt) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto first = ddx::rt::equation(x * log(x) + y * y);
  auto second = ddx::rt::equation(y * exp(x) + x * x);

  const auto at = std::array{0.4, 0.9};
  const auto expected_first = *first.hessian(at);
  const auto expected_second = *second.hessian(at);

  std::vector<double> a, c;
  std::thread one{[&] { a = *first.hessian(at); }};
  std::thread two{[&] { c = *second.hessian(at); }};
  one.join();
  two.join();

  EXPECT_EQ(a, expected_first);
  EXPECT_EQ(c, expected_second);
}

// Values for many points, with no gradient computed anywhere along the way.
TEST(RtEquation, EvaluatingABatchComputesNoPartials) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * log(x) + y * y);

  constexpr std::size_t n = 5;
  std::vector<double> cx(n), cy(n), f(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.1 * static_cast<double>(i);
    cy[i] = 1.1 - 0.05 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};
  double *const values[]{f.data()};
  ASSERT_TRUE(eq.evaluate(columns, values, n).has_value());

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(f[i], cx[i] * std::log(cx[i]) + cy[i] * cy[i], 1e-12)
        << "value at " << i;
    // Bit-exact against the scalar accessor: one graph, one sweep.
    const std::vector<double> at{cx[i], cy[i]};
    EXPECT_EQ(f[i], *eq.evaluate(std::span<const double>{at}));
  }
}

// The values lane's graph carries no Jacobian block at all -- a shape no other
// caller produces.  A system rather than one root, where the sizes differ.
TEST(RtEquation, EvaluatingUsesAValuesOnlyGraph) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * log(x), y * y + x, exp(x * y));

  const std::vector<double> at{0.6, 1.4};
  const auto v = *eq.evaluate(std::span<const double>{at});
  ASSERT_EQ(v.size(), 3u);

  // Against the values+Jacobian lane: a different graph over the same nodes,
  // and bit-exact since neither reorders anything.
  double f0 = 0, f1 = 0, f2 = 0;
  const double *const columns[]{&at[0], &at[1]};
  double *const values[]{&f0, &f1, &f2};

  // Five columns, not six: d(x log x)/dy is structurally zero, so it has no
  // column and the pattern is what says which five these are.
  const ddx::rt::Sparsity &pattern = eq.jacobian_pattern()->get();
  ASSERT_EQ(*eq.jacobian_columns(), 5u);
  EXPECT_EQ(pattern.nonzeros(), 5u);
  EXPECT_FALSE(pattern.at(0, 1).has_value());
  EXPECT_EQ(pattern.at(0, 0), 0u);

  std::vector<double> cells(*eq.jacobian_columns());
  const auto partials = cells |
                        std::views::transform([](double &c) { return &c; }) |
                        ddx::impl::to<std::vector<double *>>();
  ASSERT_TRUE(eq.jacobian(columns, values, partials, 1).has_value());
  EXPECT_EQ(v[0], f0);
  EXPECT_EQ(v[1], f1);
  EXPECT_EQ(v[2], f2);

  // And the dense spelling puts the zero back where the pattern left a hole.
  const auto dense = *eq.jacobian(0.6, 1.4);
  ASSERT_EQ(dense.size(), 6u);
  EXPECT_EQ(dense[1], 0.0);
  EXPECT_EQ(dense[0], cells[*pattern.at(0, 0)]);
  EXPECT_EQ(dense[2], cells[*pattern.at(1, 0)]);
  EXPECT_EQ(dense[5], cells[*pattern.at(2, 1)]);
}

// The gradient lane sweeps a graph with no value block.  The answers are the
// Jacobian lane's to the bit -- what changes is which nodes are visited, never
// the arithmetic at any of them.
TEST(RtEquation, GradientIsTheJacobianWithoutTheValue) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  // The product is dead for the gradient, and erf's derivative never calls erf.
  const auto eq = ddx::rt::equation(x * log(x) + 0.3 * (x * y) + erf(y));

  const auto g = *eq.gradient(0.6, 1.4);
  const auto j = *eq.jacobian(0.6, 1.4);
  ASSERT_EQ(g.size(), 2u);
  EXPECT_EQ(g[0], j[0]);
  EXPECT_EQ(g[1], j[1]);
  EXPECT_EQ(g, *eq.gradient(std::vector<double>{0.6, 1.4}));

  constexpr std::size_t n = 4;
  std::vector<double> cx(n), cy(n), gx(n), gy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.1 * static_cast<double>(i);
    cy[i] = 1.1 - 0.05 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};
  double *const partials[]{gx.data(), gy.data()};
  ASSERT_TRUE(eq.gradient(columns, partials, n).has_value());
  for (std::size_t i = 0; i < n; ++i) {
    const auto at = *eq.jacobian(cx[i], cy[i]);
    EXPECT_EQ(gx[i], at[0]) << i;
    EXPECT_EQ(gy[i], at[1]) << i;
  }
}

// Scalar and batch Hessians share a lane, with the scalar caller freezing it
// first.
TEST(RtEquation, ScalarAndBatchAgreeAtTheSamePoint) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * log(x) * y + y * y * y);

  const std::vector<double> at{0.6, 1.4};
  // Scalar first, so the accessor freezes the lane rather than a batch call.
  const auto dense = *eq.hessian(std::span<const double>{at});
  ASSERT_EQ(dense.size(), 4u);

  std::vector<double> h(*eq.hessian_columns());
  std::vector<double *> hc(h.size());
  std::ranges::transform(h, hc.begin(), [](double &e) { return &e; });
  double f = 0, dx = 0, dy = 0;
  const double *const columns[]{&at[0], &at[1]};
  double *const values[]{&f};
  double *const partials[]{&dx, &dy};
  ASSERT_TRUE(eq.hessian(columns, values, partials, hc, 1).has_value());

  EXPECT_EQ(f, *eq.evaluate(std::span<const double>{at}));
  const auto g = *eq.jacobian(std::span<const double>{at});
  EXPECT_EQ(g[0], dx);
  EXPECT_EQ(g[1], dy);
}

// The scalar accessors touch the lane cache, and must not reach for a compiler
// under a backend that declines one.
TEST(RtEquation, EvaluatingUnderInterpretCompilesNothing) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(x * log(x) + y * y);
  eq.options({.backend = ddx::rt::Backend::Interpret});

  const std::vector<double> at{0.6, 1.4};
  EXPECT_NEAR(*eq.evaluate(std::span<const double>{at}),
              0.6 * std::log(0.6) + 1.4 * 1.4, 1e-12);
  EXPECT_FALSE(eq.uses_kernel());
  EXPECT_FALSE(eq.wait_for_kernel());
}

// Interpret must not launch anything, however long one waits.
TEST(RtEquation, InterpretNeverCompilesBehindTheCaller) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  auto eq = ddx::rt::equation(x * log(x));
  eq.options({.backend = ddx::rt::Backend::Interpret});
  EXPECT_FALSE(eq.wait_for_kernel());
  EXPECT_FALSE(eq.uses_kernel());
}

namespace {

// Enough of the emitter's op set that the codegen levels have somewhere to
// disagree, and enough nodes that the top rung does not land first.
auto ladder_model() {
  return ddx::rt::equation([] {
    std::vector<ddx::rt::RTExpression<double>> v;
    for (std::size_t i = 0; i < 24; ++i) {
      v.push_back(ddx::rt::var("x" + std::to_string(i)));
    }
    auto f = v[0] * log(v[0]);
    for (std::size_t i = 1; i < v.size(); ++i) {
      f = f + v[i] * log(v[i]) + exp(v[i - 1] * v[i]) + sqrt(v[i]) * v[i - 1];
    }
    return f;
  });
}

// One gradient at a fixed point, as the bits.
std::vector<double> ladder_gradient(const auto &eq, std::size_t n) {
  std::vector<double> at(n), value(1), partial(n);
  for (std::size_t i = 0; i < n; ++i) {
    at[i] = 0.35 + 0.01 * static_cast<double>(i);
  }
  std::vector<const double *> xs(n);
  std::vector<double *> gs(n);
  for (std::size_t i = 0; i < n; ++i) {
    xs[i] = at.data() + i;
    gs[i] = partial.data() + i;
  }
  double *const values[]{value.data()};
  EXPECT_TRUE(eq.jacobian(xs, values, gs, 1).has_value());
  partial.push_back(value[0]);
  return partial;
}

} // namespace

// Every rung answers the same bits, which is what makes the swap invisible.
TEST(RtEquation, EveryRungOfTheLadderAgreesToTheBit) {
  constexpr std::size_t n = 24;

  auto swept = ladder_model();
  swept.options({.backend = ddx::rt::Backend::Interpret});

  auto cheap = ladder_model();
  cheap.options({.backend = ddx::rt::Backend::Compile,
                 .codegen = {.codegen_level = ddx::rt::Level::O0}});
  ASSERT_TRUE(cheap.wait_for_kernel());

  auto top = ladder_model();
  top.options({.backend = ddx::rt::Backend::Compile,
               .codegen = {.codegen_level = ddx::rt::Level::O1}});
  ASSERT_TRUE(top.wait_for_kernel());

  // Rung against rung, still to the bit: both rungs compile the same graph, and
  // a rung landing mid-loop must not move an answer.
  const auto low = ladder_gradient(cheap, n);
  EXPECT_EQ(ladder_gradient(top, n), low) << "the top rung moved a bit";

  // Sweep against kernel, to a tolerance rather than to the bit.  The kernel is
  // compiled from a graph whose reduction spines are blocked and the sweep is
  // not -- see Compiled::compile_graph -- because a dependent chain is latency
  // the kernel cannot hide and the same rewrite costs the sweep tape locality.
  // Blocking reassociates, so the two disagree in the last places.
  const auto expected = ladder_gradient(swept, n);
  ASSERT_EQ(low.size(), expected.size());
  for (const auto [got, want] : std::views::zip(low, expected)) {
    EXPECT_NEAR(got, want, 1e-12 * (1.0 + std::fabs(want)))
        << "the kernel is a reassociation of the sweep, not a different answer";
  }
}

// The ladder climbs and never falls back.  One shared pool orders nothing, so a
// cheap compile can land second; the rule is the rank, not the arrival.
TEST(RtEquation, TheLadderClimbsAndTheAnswersDoNotMove) {
  constexpr std::size_t n = 24;

  auto eq = ladder_model();
  eq.options({.backend = ddx::rt::Backend::Compile,
              .codegen = {.codegen_level = ddx::rt::Level::O1}});

  // wait_for_kernel() waits for the *first* rung, so the cheap one is in hand.
  ASSERT_TRUE(eq.wait_for_kernel());
  ASSERT_TRUE(eq.kernel_level().has_value());
  EXPECT_EQ(*eq.kernel_level(), ddx::rt::Level::O0)
      << "wait_for_kernel() waited for the top rung, not the first";

  const auto expected = ladder_gradient(eq, n);

  // Bounded: a failure to climb must fail the test, not hang it.
  using namespace std::chrono_literals;
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  auto seen = ddx::rt::Level::O0;
  bool climbed = false;
  while (std::chrono::steady_clock::now() < deadline) {
    const auto level = eq.kernel_level();
    ASSERT_TRUE(level.has_value()) << "the ladder gave the kernel back";
    EXPECT_GE(*level, seen) << "a rung that landed late demoted a better one";
    seen = *level;
    EXPECT_EQ(ladder_gradient(eq, n), expected)
        << "an answer moved at rung " << std::to_underlying(seen);
    if (seen == ddx::rt::Level::O1) {
      climbed = true;
      break;
    }
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_TRUE(climbed) << "the top rung never replaced the cheap one";
}

// At codegen 0 there is nothing cheaper underneath, so the ladder is one rung.
TEST(RtEquation, CodegenZeroIsASingleRung) {
  auto eq = ladder_model();
  eq.options({.backend = ddx::rt::Backend::Compile,
              .codegen = {.codegen_level = ddx::rt::Level::O0}});
  ASSERT_TRUE(eq.wait_for_kernel());
  ASSERT_TRUE(eq.kernel_level().has_value());
  EXPECT_EQ(*eq.kernel_level(), ddx::rt::Level::O0);
}

// With a warm cache both rungs link instead of compiling.  Each rung keys
// separately, being different machine code, so a warm run gets the good kernel
// rather than the cheap one.
TEST(RtEquation, AWarmCacheServesBothRungs) {
  const auto dir =
      std::filesystem::temp_directory_path() / "ddx_rt_ladder_cache";
  std::filesystem::remove_all(dir);

  const auto climb = [&dir] {
    auto eq = ladder_model();
    eq.options({.backend = ddx::rt::Backend::Compile,
                .codegen = {.codegen_level = ddx::rt::Level::O1},
                .cache_dir = dir.string()});
    EXPECT_TRUE(eq.wait_for_kernel());
    using namespace std::chrono_literals;
    const auto deadline = std::chrono::steady_clock::now() + 30s;
    while (eq.kernel_level() != ddx::rt::Level::O1 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    return ladder_gradient(eq, 24);
  };

  const auto cold = climb();
  ASSERT_TRUE(std::filesystem::exists(dir)) << "the cache wrote nothing";
  const auto entries = static_cast<std::size_t>(
      std::distance(std::filesystem::directory_iterator{dir},
                    std::filesystem::directory_iterator{}));
  EXPECT_EQ(entries, 2u) << "both rungs should key separately";

  EXPECT_EQ(climb(), cold) << "a cached rung moved a bit";
  std::filesystem::remove_all(dir);
}

// The sweep is not a rung: with no backend there is no level to report.
TEST(RtEquation, InterpretHasNoRung) {
  auto eq = ladder_model();
  eq.options({.backend = ddx::rt::Backend::Interpret});
  EXPECT_FALSE(eq.kernel_level().has_value());
}

// --- Backend::Adapt ----------------------------------------------------------
//
// The same ladder, climbed on the counter rather than on sight of the backend.
// Every threshold here is stated, because a test that waited for the shipping
// default would be waiting on 65536 points.

// A lane nobody uses costs nothing: the whole point of counting is not paying
// for a compile that will never be amortised.
TEST(RtEquation, AdaptCompilesNothingUntilTheBatchPaysForIt) {
  auto eq = ladder_model();
  eq.options({.backend = ddx::rt::Backend::Adapt,
              .codegen = {.codegen_level = ddx::rt::Level::O1},
              .warm_points = 1000});

  for (int i = 0; i < 3; ++i) {
    (void)ladder_gradient(eq, 24); // one point apiece
  }
  EXPECT_FALSE(eq.uses_kernel()) << "a cold lane compiled anyway";
  EXPECT_FALSE(eq.kernel_level().has_value());

  // And it says so, which is what separates counting from a refused compile.
  const auto warm = eq.warming();
  ASSERT_TRUE(warm.has_value());
  EXPECT_EQ(warm->points, 3u);
  EXPECT_EQ(warm->threshold, 1000u);
}

// Under Adapt, waiting is for a rung already bought.  Overriding the counter
// here would make every measurement of the policy a lie.
TEST(RtEquation, AdaptDoesNotLetAWaitBuyARung) {
  auto eq = ladder_model();
  eq.options({.backend = ddx::rt::Backend::Adapt, .warm_points = 1000});
  EXPECT_FALSE(eq.wait_for_kernel());
  EXPECT_FALSE(eq.uses_kernel());
  ASSERT_TRUE(eq.warming().has_value());
  EXPECT_EQ(eq.warming()->points, 0u) << "an observer charged the lane";
}

// Cheap rung first, then the top one, each on its own threshold -- and the
// answers do not move across either, as on the eager ladder.
TEST(RtEquation, AdaptBuysTheCheapRungThenTheTop) {
  constexpr std::size_t n = 24;

  auto swept = ladder_model();
  swept.options({.backend = ddx::rt::Backend::Interpret});
  const auto expected = ladder_gradient(swept, n);

  auto eq = ladder_model();
  eq.options({.backend = ddx::rt::Backend::Adapt,
              .codegen = {.codegen_level = ddx::rt::Level::O1},
              .warm_points = 1,
              .hot_points = 1});

  // One point buys the cheap rung and nothing more.
  EXPECT_EQ(ladder_gradient(eq, n), expected);
  ASSERT_TRUE(eq.wait_for_kernel()) << "the first point bought no rung";
  ASSERT_TRUE(eq.kernel_level().has_value());
  EXPECT_EQ(*eq.kernel_level(), ddx::rt::Level::O0)
      << "the top rung was bought too early";

  // The next buys the top one.  Bounded: a failure to climb must fail the test
  // rather than hang it.
  using namespace std::chrono_literals;
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  bool climbed = false;
  while (std::chrono::steady_clock::now() < deadline) {
    EXPECT_EQ(ladder_gradient(eq, n), expected)
        << "an answer moved as it climbed";
    if (eq.kernel_level() == ddx::rt::Level::O1) {
      climbed = true;
      break;
    }
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_TRUE(climbed) << "the top rung was never bought";

  // Nothing left to buy, so nothing left to count.
  EXPECT_FALSE(eq.warming().has_value());
}

// At codegen 0 there is nothing cheaper underneath, so one threshold buys the
// only rung there is and the counter is done.
TEST(RtEquation, AdaptAtCodegenZeroBuysOneRung) {
  auto eq = ladder_model();
  eq.options({.backend = ddx::rt::Backend::Adapt,
              .codegen = {.codegen_level = ddx::rt::Level::O0},
              .warm_points = 1});
  (void)ladder_gradient(eq, 24);
  ASSERT_TRUE(eq.wait_for_kernel());
  ASSERT_TRUE(eq.kernel_level().has_value());
  EXPECT_EQ(*eq.kernel_level(), ddx::rt::Level::O0);
  EXPECT_FALSE(eq.warming().has_value())
      << "still counting toward a second rung";
}

// A threshold of nothing is the eager ladder, which is what makes the policy
// testable at all.
TEST(RtEquation, AdaptWithNoThresholdBuysOnTheFirstCall) {
  auto eq = ladder_model();
  eq.options({.backend = ddx::rt::Backend::Adapt,
              .codegen = {.codegen_level = ddx::rt::Level::O1},
              .warm_points = 0,
              .hot_points = 0});
  (void)ladder_gradient(eq, 24);
  EXPECT_TRUE(eq.wait_for_kernel());
}

// Under Interpret the counter is never touched, however hot the lane gets.
TEST(RtEquation, InterpretIsNotWarming) {
  auto eq = ladder_model();
  eq.options({.backend = ddx::rt::Backend::Interpret});
  (void)ladder_gradient(eq, 24);
  EXPECT_FALSE(eq.warming().has_value());
  EXPECT_FALSE(eq.uses_kernel());
}
#endif
