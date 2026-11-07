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

// Text is a second spelling of a model, not a second implementation: the same
// model reaches the same arena, node for node, however it was written.
namespace {

using ddx::errc;
using ddx::rt::text::lower;
using ddx::rt::text::parse;

// The value of a one-line model at a point.
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
  // Sums and products fold to the left, and multiplication binds tighter.
  EXPECT_DOUBLE_EQ(value_of("a - b - c", 10.0, 3.0, 2.0), 5.0);
  EXPECT_DOUBLE_EQ(value_of("a / b / c", 12.0, 3.0, 2.0), 2.0);
  EXPECT_DOUBLE_EQ(value_of("a + b * c", 1.0, 2.0, 3.0), 7.0);
}

TEST(RtText, RaisesToAPowerTheWayPythonDoes) {
  // ** is right-associative, unlike every other binary operator; left
  // associativity would give 64.
  EXPECT_DOUBLE_EQ(value_of("x ** y ** z", 2.0, 3.0, 2.0), 512.0);
  // ** binds tighter than a unary minus on its left, and looser than one on its
  // right, which lets an exponent carry a sign.
  EXPECT_DOUBLE_EQ(value_of("-x ** 2", 3.0), -9.0);
  EXPECT_DOUBLE_EQ(value_of("x ** -2", 2.0), 0.25);
  // A run of signs is a parity, as in Python, whereas in C++ the same two
  // characters form an operator.
  EXPECT_DOUBLE_EQ(value_of("--x", 4.0), 4.0);
  EXPECT_DOUBLE_EQ(value_of("-+-x", 4.0), 4.0);
}

// A literal past the range reads as Python reads it, as infinity or zero.
TEST(RtText, ALiteralPastTheRangeRounds) {
  EXPECT_EQ(value_of("x + 1e999", 1.0),
            std::numeric_limits<double>::infinity());
  EXPECT_EQ(value_of("x + 1e-999", 1.0), 1.0);
  EXPECT_EQ(value_of("x * 1e+999", 1.0),
            std::numeric_limits<double>::infinity());
}

TEST(RtText, BuildsTheSameArenaAsTheCppSpelling) {
  // One statement per subexpression, since C++ leaves operand order unspecified
  // and the arena records build order; this fixes the numbering the text uses.
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

  // Interning makes the arena a function of the model, so equal models are the
  // same bytes and not merely the same answers.
  const auto same = [](const auto &l, const auto &r) {
    return l.op == r.op && l.in == r.in && l.payload == r.payload &&
           l.value == r.value;
  };
  EXPECT_TRUE(std::ranges::equal(written.nodes(), parsed.nodes(), same));
  EXPECT_EQ(by_hand.id(written), lowered->id(parsed));
  EXPECT_TRUE(std::ranges::equal(written.symbols(), parsed.symbols()));
}

// The four comparisons the graph does not carry are compositions, and the text
// builds the same ones the operators do, node for node.
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

  // Read at T rather than read as double and narrowed, since 0.1f is not
  // float(0.1).
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
  // costs no node, exactly as rt::Expression{3.0} + 4.0 does.
  EXPECT_EQ(arena.size(), 0U);
  EXPECT_TRUE(lowered->pending());
  EXPECT_DOUBLE_EQ(lowered->literal(), 7.0);
}

TEST(RtText, SpellsEveryOpcodeTheTablesName) {
  // The parser has no function table of its own, so every callable op in the
  // opcode tables must be callable from text with no edit to the grammar.
  const auto callable = [](const std::string_view label) {
    return std::ranges::all_of(label, [](const char c) {
      return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    });
  };

  std::size_t named = 0;
  for (const auto &label : ddx::rt::opcode_labels()) {
    const auto op = ddx::rt::opcode_of(label);
    ASSERT_TRUE(op.has_value()) << label;
    // Skips what text may not name (leaves, externs and tensor rows) and the
    // operators, which have their own syntax.
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
  // The transcendentals alone are eighteen, so a table that stopped being read
  // would leave this at zero.
  EXPECT_GT(named, 18U);
}

TEST(RtText, LowersADifferenceAsTheSumOfANegation) {
  // There is no Sub opcode (the label "-" is unary Neg), so this is the one
  // place where the grammar and the opcode table do not line up one to one.
  const auto ast = parse("x - y");
  ASSERT_TRUE(ast.has_value());
  const auto &root = ast->terms[ast->root];
  EXPECT_EQ(root.op, ddx::rt::OpCode::Add);
  EXPECT_EQ(ast->terms[root.in[1]].op, ddx::rt::OpCode::Neg);
}

TEST(RtText, RefusesWhatItCannotMean) {
  // Python's spellings: ^ is not exponentiation, select is a call rather than a
  // ternary, and an expression holds at most one comparison.
  EXPECT_EQ(parse("x ^ 2").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x > 0 ? x : -x").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x % y").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x < y < 2").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x == y == 2").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x = y").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("x <> y").error().code, errc::bad_syntax);

  EXPECT_EQ(parse("x +").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("(x").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("").error().code, errc::bad_syntax);
  EXPECT_EQ(parse("2 x").error().code, errc::bad_syntax);

  // Unknown to the opcode tables. This has its own code rather than
  // unknown_symbol, which is about a point's symbols.
  EXPECT_EQ(parse("nope(x)").error().code, errc::unknown_function);
  // Known to the opcode tables but not callable, since a leaf has no arguments.
  EXPECT_EQ(parse("const(x)").error().code, errc::unknown_function);
  EXPECT_EQ(parse("var(x)").error().code, errc::unknown_function);
  // Callable, but with the wrong number of arguments.
  EXPECT_EQ(parse("sin(x, y)").error().code, errc::wrong_argument_count);
  EXPECT_EQ(parse("pow(x)").error().code, errc::wrong_argument_count);
}

TEST(RtText, TakesOneStringPerFunction) {
  const auto sys = ddx::rt::equation("x*x + y*y - 4", "x*y - 1");
  ASSERT_TRUE(sys.has_value());
  EXPECT_EQ(sys->arity(), 2U);

  // The same system that tests_rt_newton.cpp builds by hand.
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
  // Every source is read before any becomes an equation, so a symbol that only
  // the second function introduces still gets a slot.
  const auto sys = ddx::rt::equation("x + 1", "y + 2");
  ASSERT_TRUE(sys.has_value());
  EXPECT_EQ(sys->arity(), 2U);
  const auto symbols = sys->symbols();
  ASSERT_TRUE(symbols.has_value());
  EXPECT_TRUE(
      std::ranges::equal(*symbols, std::array<std::string_view, 2>{"x", "y"}));
}

TEST(RtText, RefusesAModelThatNamesNoSymbol) {
  // A bare constant folds to a pending literal, which names no arena, and is
  // refused as rt::equation(rt::Expression{7.0}) is.
  const auto eq = ddx::rt::equation("3 + 4");
  ASSERT_TRUE(eq.has_value());
  EXPECT_TRUE(eq->poisoned());
  EXPECT_EQ(eq->status()->code, errc::no_graph);
}

TEST(RtText, AnOutOfRangeLiteralRoundsByItsMagnitude) {
  const std::string zeros(400, '0');
  const auto inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(value_of("x + 0." + zeros + "1", 1.0), 1.0);
  EXPECT_EQ(value_of("x + 0." + zeros + "1e5", 1.0), 1.0);
  EXPECT_EQ(value_of("x + 1e-999", 1.0), 1.0);
  EXPECT_EQ(value_of("x + 1e-99999999999999999999", 1.0), 1.0);
  EXPECT_EQ(value_of("x + 1" + zeros + "e-1", 1.0), inf);
  EXPECT_EQ(value_of("x + 1" + zeros, 1.0), inf);
  EXPECT_EQ(value_of("x + 1e999", 1.0), inf);
  EXPECT_EQ(value_of("x + 1e99999999999999999999", 1.0), inf);
}

[[nodiscard]] std::string repeated(std::string_view word, std::size_t n) {
  std::string out;
  out.reserve(word.size() * n);
  for (std::size_t i = 0; i < n; ++i) {
    out += word;
  }
  return out;
}

// Parentheses, calls and ** recurse, and a stack overflow cannot be refused,
// so nesting that is too deep is reported as a syntax error first.
TEST(RtText, TextNestedTooDeepIsRefused) {
  constexpr std::size_t deep = 200'000;
  const auto refused = [](const std::string &source) {
    const auto ast = parse(source);
    return !ast.has_value() && ast.error().code == errc::bad_syntax;
  };
  EXPECT_TRUE(refused(repeated("(", deep) + "x" + repeated(")", deep)));
  EXPECT_TRUE(refused(repeated("sin(", deep) + "x" + repeated(")", deep)));
  EXPECT_TRUE(refused("x" + repeated("**x", deep)));
  EXPECT_TRUE(refused("x" + repeated("**-x", deep)));
  EXPECT_TRUE(refused(repeated("(x**", deep) + "x" + repeated(")", deep)));
}

// A flat sum or a run of signs does not recurse with its length, and a model
// nested as deeply as anyone writes one still reads.
TEST(RtText, LongTextThatIsNotDeepIsRead) {
  constexpr std::size_t shallow = 100;
  const std::string nested =
      repeated("(", shallow) + "x" + repeated(")", shallow);
  EXPECT_DOUBLE_EQ(value_of(nested, 2.0), 2.0);
  EXPECT_TRUE(parse("x" + repeated("+x", 10'000)).has_value());
  EXPECT_TRUE(parse("x**2" + repeated("+x**2", 10'000)).has_value());
  EXPECT_TRUE(parse(repeated("-", 10'000) + "x").has_value());
  EXPECT_TRUE(parse(repeated("(x)+", 10'000) + "x").has_value());
}

} // namespace
