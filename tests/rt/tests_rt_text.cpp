#include "ddx.hpp"
#include "rt/equation.hpp"
#include "rt/opcode.hpp"
#include "rt/text/ast.hpp"
#include "rt/text/equation.hpp"
#include "rt/text/lower.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <ranges>
#include <string>
#include <string_view>

// An equation written down instead of built.  The property under all of it is
// that text is a second spelling and not a second implementation: the same
// model reaches the same arena, node for node, whichever way it was said.
namespace {

using ddx::errc;
using ddx::rt::text::lower;
using ddx::rt::text::parse;

// The value of a one-line model at a point, which is what most of the
// precedence questions below reduce to.
template <typename... Ts>
[[nodiscard]] double value_of(const std::string_view source, const Ts... at) {
  const auto eq = ddx::rt::equation(source);
  EXPECT_TRUE(eq.has_value()) << source;
  if (!eq) {
    return std::nan("");
  }
  const auto v = eq->evaluate(at...);
  EXPECT_TRUE(v.has_value()) << source;
  return v ? *v : std::nan("");
}

TEST(RtText, ReadsPythonsPrecedenceAndAssociativity) {
  // Sums and products fold to the left...
  EXPECT_DOUBLE_EQ(value_of("a - b - c", 10.0, 3.0, 2.0), 5.0);
  EXPECT_DOUBLE_EQ(value_of("a / b / c", 12.0, 3.0, 2.0), 2.0);
  // ...and multiplication binds tighter than addition.
  EXPECT_DOUBLE_EQ(value_of("a + b * c", 1.0, 2.0, 3.0), 7.0);
}

TEST(RtText, RaisesToAPowerTheWayPythonDoes) {
  // Right-associative, unlike every other binary operator here: left would
  // be 64.
  EXPECT_DOUBLE_EQ(value_of("x ** y ** z", 2.0, 3.0, 2.0), 512.0);
  // Tighter than unary minus on its left...
  EXPECT_DOUBLE_EQ(value_of("-x ** 2", 3.0), -9.0);
  // ...and looser on its right, which is what makes an exponent signable.
  EXPECT_DOUBLE_EQ(value_of("x ** -2", 2.0), 0.25);
  // A run of signs is a parity, as it is in Python and unlike C++, where the
  // same two characters are an operator.
  EXPECT_DOUBLE_EQ(value_of("--x", 4.0), 4.0);
  EXPECT_DOUBLE_EQ(value_of("-+-x", 4.0), 4.0);
}

// Past the range is what Python reads: the infinity or the zero.
TEST(RtText, ALiteralPastTheRangeRounds) {
  EXPECT_EQ(value_of("x + 1e999", 1.0),
            std::numeric_limits<double>::infinity());
  EXPECT_EQ(value_of("x + 1e-999", 1.0), 1.0);
  EXPECT_EQ(value_of("x * 1e+999", 1.0),
            std::numeric_limits<double>::infinity());
}

TEST(RtText, BuildsTheSameArenaAsTheCppSpelling) {
  // One subexpression per name, in the order the text reduces them: C++ leaves
  // the order of an operator's operands unspecified, and an arena records the
  // order it was built in, so `exp(x) * sin(y) + x * 2.0` written as one line
  // is the same graph under whichever numbering the compiler happened to pick.
  ddx::rt::Arena<> written;
  const auto x = var(written, "x");
  const auto y = var(written, "y");
  const auto left = exp(x);
  const auto right = sin(y);
  const auto product = left * right;
  const auto scaled = x * 2.0;
  const auto by_hand = product + scaled;

  ddx::rt::Arena<> parsed;
  const auto ast = parse("exp(x)*sin(y) + x*2");
  ASSERT_TRUE(ast.has_value());
  const auto lowered = lower(parsed, *ast);
  ASSERT_TRUE(lowered.has_value());

  // Node for node: interning makes the arena a function of the model, so equal
  // models are the same bytes and not merely the same answers.
  const auto same = [](const auto &l, const auto &r) {
    return l.op == r.op && l.in == r.in && l.payload == r.payload &&
           l.value == r.value;
  };
  EXPECT_TRUE(std::ranges::equal(written.nodes(), parsed.nodes(), same));
  EXPECT_EQ(by_hand.id(written), lowered->id(parsed));
  EXPECT_TRUE(std::ranges::equal(written.symbols(), parsed.symbols()));
}

// The four comparisons the graph does not carry are compositions, and the text
// makes the same ones the operators do -- node for node, `1` included.
TEST(RtText, ComposesEqualityAsTheOperatorsDo) {
  const auto same_arena = [](const std::string_view source, auto &&build) {
    ddx::rt::Arena<> written;
    const auto x = var(written, "x");
    const auto y = var(written, "y");
    const auto by_hand = build(x, y);

    ddx::rt::Arena<> parsed;
    const auto ast = parse(source);
    ASSERT_TRUE(ast.has_value()) << source;
    const auto lowered = lower(parsed, *ast);
    ASSERT_TRUE(lowered.has_value()) << source;
    const auto same = [](const auto &l, const auto &r) {
      return l.op == r.op && l.in == r.in && l.payload == r.payload &&
             l.value == r.value;
    };
    EXPECT_TRUE(std::ranges::equal(written.nodes(), parsed.nodes(), same))
        << source;
    EXPECT_EQ(by_hand.id(written), lowered->id(parsed)) << source;
  };
  same_arena("x == y", [](auto x, auto y) { return x == y; });
  same_arena("x != y", [](auto x, auto y) { return x != y; });
  same_arena("x >= y", [](auto x, auto y) { return x >= y; });
  same_arena("x > y + 1", [](auto x, auto y) { return x > y + 1.0; });

  EXPECT_DOUBLE_EQ(value_of("x == y", 2.0, 2.0), 1.0);
  EXPECT_DOUBLE_EQ(value_of("x == y", 2.0, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(value_of("x != y", 2.0, 1.0), 1.0);
  EXPECT_DOUBLE_EQ(value_of("x != y", std::nan(""), 1.0), 1.0);
  EXPECT_DOUBLE_EQ(value_of("x == y", std::nan(""), std::nan("")), 0.0);
}

TEST(RtText, ReadsEveryLiteralAtTheScalarType) {
  // The grammar has no types, so this is a quotient and never a truncation.
  EXPECT_DOUBLE_EQ(value_of("(1/2) * x", 1.0), 0.5);
  EXPECT_DOUBLE_EQ(value_of("1e-3 * x", 4.0), 0.004);
  EXPECT_DOUBLE_EQ(value_of(".5 + x", 1.0), 1.5);
  EXPECT_DOUBLE_EQ(value_of("2. * x", 3.0), 6.0);

  // Read at T rather than at double and narrowed: 0.1f is not float(0.1).
  ddx::rt::Arena<float> arena;
  const auto ast = parse("0.1");
  ASSERT_TRUE(ast.has_value());
  const auto lowered = lower(arena, *ast);
  ASSERT_TRUE(lowered.has_value());
  EXPECT_EQ(lowered->literal(), 0.1F);
}

TEST(RtText, FoldsWhatNeverNeededAGraph) {
  ddx::rt::Arena<> arena;
  const auto ast = parse("3 + 4");
  ASSERT_TRUE(ast.has_value());
  const auto lowered = lower(arena, *ast);
  ASSERT_TRUE(lowered.has_value());

  // A literal that has met no symbol stays pending, so a constant expression
  // costs no node at all -- exactly as `rt::Expression{3.0} + 4.0` does.
  EXPECT_EQ(arena.size(), 0U);
  EXPECT_TRUE(lowered->pending());
  EXPECT_DOUBLE_EQ(lowered->literal(), 7.0);
}

TEST(RtText, SpellsEveryOpcodeTheTablesName) {
  // The parser has no function table of its own; this is the assertion that
  // says so.  A transcendental added to the opcode tables becomes callable from
  // text with no edit to the grammar, and this test would notice if it did not.
  const auto callable = [](const std::string_view label) {
    return std::ranges::all_of(label, [](const char c) {
      return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    });
  };

  std::size_t named = 0;
  for (const auto &label : ddx::rt::opcode_labels()) {
    const auto op = ddx::rt::opcode_of(label);
    ASSERT_TRUE(op.has_value()) << label;
    // What the library says text may not name -- leaves, externs, tensor rows
    // -- and the operators, which have their own syntax.
    if (!ddx::rt::callable(*op) || !callable(label)) {
      continue;
    }
    const auto args = ddx::rt::arity_of(*op);
    const auto source = args == 1   ? std::format("{}(x)", label)
                        : args == 3 ? std::format("{}(x, y, z)", label)
                                    : std::format("{}(x, y)", label);
    const auto ast = parse(source);
    ASSERT_TRUE(ast.has_value()) << source;
    EXPECT_EQ(ast->terms[ast->root].op, *op) << source;
    ++named;
  }
  // The transcendentals alone are eighteen; a table that stopped being read
  // would leave this at zero.
  EXPECT_GT(named, 18U);
}

TEST(RtText, LowersADifferenceAsTheSumOfANegation) {
  // There is no Sub opcode -- the label "-" is unary Neg -- so this is the one
  // place the grammar and the opcode table do not line up one to one.
  const auto ast = parse("x - y");
  ASSERT_TRUE(ast.has_value());
  const auto &root = ast->terms[ast->root];
  EXPECT_EQ(root.op, ddx::rt::OpCode::Add);
  EXPECT_EQ(ast->terms[root.in[1]].op, ddx::rt::OpCode::Neg);
}

TEST(RtText, RefusesWhatItCannotMean) {
  // Python's spellings: `^` is not exponentiation, there is no ternary --
  // `select` is a call -- and a comparison is one per expression.
  EXPECT_EQ(parse("x ^ 2").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x > 0 ? x : -x").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x % y").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x < y < 2").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x == y == 2").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x = y").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x <> y").error().code, errc::bad_syntax);

  // Malformed rather than unmeanable.
  EXPECT_EQ(parse("x +").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("(x").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("2 x").error().code, errc::bad_syntax);

  // Named, but not by the opcode tables.  Its own code, not unknown_symbol:
  // that one is about a point's symbols and would describe the wrong thing.
  EXPECT_EQ(parse("nope(x)").error().code, errc::unknown_function);
  // Named by them, and not callable: a leaf has no argument list.
  EXPECT_EQ(parse("const(x)").error().code, errc::unknown_function);
  EXPECT_EQ(parse("var(x)").error().code, errc::unknown_function);
  // Callable, and asked for the wrong number.
  EXPECT_EQ(parse("sin(x, y)").error().code, errc::wrong_argument_count);
  EXPECT_EQ(parse("pow(x)").error().code, errc::wrong_argument_count);
}

TEST(RtText, TakesOneStringPerFunction) {
  const auto sys = ddx::rt::equation("x*x + y*y - 4", "x*y - 1");
  ASSERT_TRUE(sys.has_value());
  EXPECT_EQ(sys->arity(), 2U);

  // The same system tests/rt/tests_rt_newton.cpp builds by hand.
  ddx::rt::Arena<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto by_hand = ddx::rt::equation(x * x + y * y - 4.0, x * y - 1.0);

  const auto got = sys->jacobian(1.7, 0.6);
  const auto want = by_hand.jacobian(1.7, 0.6);
  ASSERT_TRUE(got.has_value());
  ASSERT_TRUE(want.has_value());
  EXPECT_TRUE(std::ranges::equal(*got, *want));
}

TEST(RtText, NamesSymbolsWhicheverFunctionFirstUsesThem) {
  // Every source is read before any of them is an equation, so a symbol the
  // second function introduces still gets a slot.
  const auto sys = ddx::rt::equation("x + 1", "y + 2");
  ASSERT_TRUE(sys.has_value());
  EXPECT_EQ(sys->arity(), 2U);
  const auto symbols = sys->symbols();
  ASSERT_TRUE(symbols.has_value());
  EXPECT_TRUE(
      std::ranges::equal(*symbols, std::array<std::string_view, 2>{"x", "y"}));
}

TEST(RtText, RefusesAModelThatNamesNoSymbol) {
  // A bare constant folds to a pending literal, which names no arena -- the
  // same refusal `ddx::rt::equation(rt::Expression{7.0})` earns.
  const auto eq = ddx::rt::equation("3 + 4");
  ASSERT_TRUE(eq.has_value());
  EXPECT_TRUE(eq->poisoned());
  EXPECT_EQ(eq->status()->code, errc::no_graph);
}

} // namespace
