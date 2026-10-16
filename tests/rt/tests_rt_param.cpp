#include "ddx.hpp"
#include "rt/equation.hpp"
#include "rt/text/equation.hpp"
#include "scratch.hpp"

#include <gtest/gtest.h>

#include <array>
#include <span>
#include <string>
#include <vector>

// Inputs in declaration order, and parameters: inputs read in every call and
// never differentiated.
namespace {

using namespace ddx::literals;

TEST(RtParam, SymbolsTakeTheOrderTheyAreNamedIn) {
  const auto eq = ddx::rt::equation([] {
    const auto b = ddx::rt::var("b");
    const auto a = ddx::rt::var("a");
    return a * a * b;
  });
  EXPECT_TRUE(std::ranges::equal(*eq.symbols(),
                                 std::array<std::string_view, 2>{"b", "a"}));
  // Positional (1, 2) is b = 1, a = 2; named points do not care.
  EXPECT_DOUBLE_EQ(*eq.evaluate(1.0, 2.0), 4.0);
  EXPECT_DOUBLE_EQ(*eq.evaluate("a"_s = 2.0, "b"_s = 1.0), 4.0);
  EXPECT_EQ(*eq.jacobian(1.0, 2.0), (std::vector{4.0, 4.0}));
  const auto &row = *eq.jacobian_pattern()->get().begin();
  EXPECT_EQ(std::vector(row.begin(), row.end()),
            (std::vector<std::uint32_t>{0, 1}));
}

TEST(RtParam, TextSymbolsTakeTheOrderTheyFirstAppearIn) {
  const auto eq = ddx::rt::equation("y * x");
  ASSERT_TRUE(eq.has_value());
  EXPECT_TRUE(std::ranges::equal(*eq->symbols(),
                                 std::array<std::string_view, 2>{"y", "x"}));
}

[[nodiscard]] auto scaled() {
  return ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto a = ddx::rt::param("a");
    return a * x * x;
  });
}

TEST(RtParam, AParamIsAnInputWithNoColumn) {
  const auto eq = scaled();
  ASSERT_FALSE(eq.poisoned());
  EXPECT_TRUE(std::ranges::equal(*eq.symbols(),
                                 std::array<std::string_view, 1>{"x"}));
  EXPECT_TRUE(std::ranges::equal(*eq.params(),
                                 std::array<std::string_view, 1>{"a"}));
  EXPECT_EQ(eq.arity(), 1u);
  EXPECT_EQ(eq.input_count(), 2u);
  EXPECT_EQ(*eq.inputs(), (std::vector<std::string>{"x", "a"}));
  EXPECT_EQ(eq.jacobian_columns(), 1u);

  // Two values of `a` at one x, without rebuilding: positional is x then a.
  EXPECT_DOUBLE_EQ(*eq.evaluate(3.0, 2.0), 18.0);
  EXPECT_DOUBLE_EQ(*eq.evaluate(3.0, 5.0), 45.0);
  EXPECT_EQ(*eq.gradient(3.0, 2.0), std::vector{12.0});
  EXPECT_EQ(*eq.gradient("a"_s = 5.0, "x"_s = 3.0), std::vector{30.0});
  EXPECT_EQ(*eq.hessian(3.0, 5.0), std::vector{10.0});
  EXPECT_EQ(*eq.hvp("x", 3.0, 5.0), std::vector{10.0});
  EXPECT_EQ(eq.hvp("a", 3.0, 5.0).error().code, ddx::errc::unknown_symbol)
      << "a parameter has no Hessian column";
}

TEST(RtParam, APointMustSupplyEveryParam) {
  const auto eq = scaled();
  EXPECT_EQ(eq.evaluate("x"_s = 3.0).error().code, ddx::errc::short_point);
  EXPECT_EQ(eq.evaluate(3.0).error().code, ddx::errc::wrong_arity);
}

TEST(RtParam, ABatchCarriesAParamColumn) {
  const auto eq = scaled();
  const std::array xs{1.0, 2.0, 3.0};
  const std::array as{2.0, 2.0, 5.0};
  std::array<double, 3> f{};
  std::array<double, 3> g{};
  const std::array<const double *, 2> in{xs.data(), as.data()};
  const std::array<double *, 1> fv{f.data()};
  const std::array<double *, 1> gv{g.data()};
  ASSERT_TRUE(eq.jacobian(in, fv, gv, 3).has_value());
  EXPECT_EQ(f, (std::array{2.0, 8.0, 45.0}));
  EXPECT_EQ(g, (std::array{4.0, 8.0, 30.0}));

  // One column short is a refusal, not a zero parameter.
  const std::array<const double *, 1> short_in{xs.data()};
  EXPECT_EQ(eq.jacobian(short_in, fv, gv, 3).error().code,
            ddx::errc::wrong_column_count);
}

TEST(RtParam, SeededProductsSpliceTheDirectionAfterTheParams) {
  const auto eq = scaled();
  const std::array v{1.0};
  EXPECT_EQ(*eq.jvp(std::span<const double>{v}, 3.0, 5.0), std::vector{30.0});
  EXPECT_EQ(*eq.hvp(std::span<const double>{v}, 3.0, 5.0), std::vector{10.0});
}

TEST(RtParam, SavesAndLoads) {
  const Scratch path{"param"};
  const auto eq = scaled();
  ASSERT_TRUE(eq.save(path).has_value());
  const auto loaded = ddx::rt::load(path);
  ASSERT_TRUE(loaded.has_value());
  EXPECT_TRUE(std::ranges::equal(*loaded->params(), *eq.params()));
  EXPECT_EQ(*loaded->gradient(3.0, 5.0), *eq.gradient(3.0, 5.0));
  EXPECT_EQ(*loaded->evaluate("a"_s = 5.0, "x"_s = 3.0), 45.0);
}

TEST(RtParam, ANameIsOneKindOfInput) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return x * ddx::rt::param("x");
  });
  ASSERT_TRUE(eq.poisoned());
  EXPECT_EQ(eq.status()->code, ddx::errc::name_clash);
}

} // namespace
