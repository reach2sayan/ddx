#include "rt/text/ast.hpp"

#include "rt/opcode.hpp"
#include "util/error.hpp"

#include <boost/parser/parser.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// A parser for Python's arithmetic grammar.
namespace ddx::rt::text {
namespace {

namespace bp = boost::parser;
static_assert(BOOST_PARSER_USE_STD_TUPLE,
              "rt::text: Boost.Parser must use std::tuple, not Hana");

// The name and arguments of a function call, as one aggregate attribute.
struct Call {
  std::string name;
  std::vector<std::uint32_t> args;
};

// The state the actions build. It is passed through with_globals because the
// rules live at namespace scope, where a lambda has nothing to capture.
struct Building {
  Ast ast;
  errc why = errc::bad_syntax; // the error a refusing action reported
  std::uint32_t depth = 0;     // the current nesting depth
};

// CPython's limit on nested parentheses, shallow enough for a thread's stack
// in a debug build.
inline constexpr std::uint32_t deepest = 200;

[[nodiscard]] constexpr std::uint32_t append(Ast &ast, const Term t) {
  ast.terms.push_back(t);
  return static_cast<std::uint32_t>(ast.terms.size() - 1);
}
// Appends an `Op` term over its operands, with the arity checked at compile
// time.
template <OpCode Op, std::same_as<std::uint32_t>... Ids>
  requires(sizeof...(Ids) == arity_of(Op))
[[nodiscard]] constexpr std::uint32_t
append(Ast &ast, std::integral_constant<OpCode, Op>, const Ids... in) {
  Term t{.op = Op};
  std::ranges::copy(std::array{in...}, t.in.begin());
  return append(ast, t);
}

// The index of `word` in `table`, appending it if it is not there yet.
[[nodiscard]] constexpr std::uint32_t intern(std::vector<std::string> &table,
                                             const std::string_view word) {
  const auto at = std::ranges::find(table, word);
  if (at == table.end()) {
    table.emplace_back(word);
    return static_cast<std::uint32_t>(table.size() - 1);
  }
  return static_cast<std::uint32_t>(std::ranges::distance(table.begin(), at));
}

// --- the actions ------------------------------------------------------------
namespace act {

// Folds the operand just read (_attr) into everything read so far (_val). The
// grammar, not this action, makes exponentiation right-associative.
template <OpCode Op>
constexpr auto fold = [](auto &ctx) {
  _val(ctx) = append(_globals(ctx).ast, opcode<Op>, _val(ctx), _attr(ctx));
};

// `a > b` is built as `b < a` and `a >= b` as `b <= a`, since only Lt and Le
// opcodes exist. rt::gt and rt::ge do the same.
template <OpCode Op>
constexpr auto fold_swapped = [](auto &ctx) {
  _val(ctx) = append(_globals(ctx).ast, opcode<Op>, _attr(ctx), _val(ctx));
};

// `a == b` is `le(a, b) * le(b, a)`, so a NaN operand gives 0, and `a != b` is
// `1 - (a == b)`, matching operator== and operator!= in expressions.hpp.
constexpr auto equal = [](auto &ctx) {
  Ast &ast = _globals(ctx).ast;
  const std::uint32_t l = _val(ctx);
  const std::uint32_t r = _attr(ctx);
  const auto le = append(ast, opcode<OpCode::Le>, l, r);
  const auto ge = append(ast, opcode<OpCode::Le>, r, l);
  _val(ctx) = append(ast, opcode<OpCode::Mul>, le, ge);
};
constexpr auto unequal = [](auto &ctx) {
  equal(ctx);
  Ast &ast = _globals(ctx).ast;
  const auto one =
      append(ast, {.op = OpCode::Const,
                   .leaf = LiteralIndex{intern(ast.literals, "1")}});
  const auto negated = append(ast, opcode<OpCode::Neg>, _val(ctx));
  _val(ctx) = append(ast, opcode<OpCode::Add>, one, negated);
};

// There is no Sub opcode, so `a - b` is built as `a + (-b)`, as the C++
// operator- does.
constexpr auto subtract = [](auto &ctx) {
  Ast &ast = _globals(ctx).ast;
  const auto negated = append(ast, opcode<OpCode::Neg>, _attr(ctx));
  _val(ctx) = append(ast, opcode<OpCode::Add>, _val(ctx), negated);
};

template <typename Index>
constexpr auto leaf = [](auto &ctx) {
  Ast &ast = _globals(ctx).ast;
  _val(ctx) = append(
      ast, {.op = Index::op,
            .leaf = Index{intern(ast.template table<Index>(), _attr(ctx))}});
};

constexpr auto assign = [](auto &ctx) { _val(ctx) = _attr(ctx); };

// Rejects nesting deeper than `deepest` before it can overflow the stack. A
// failed branch never ascends, which is harmless since the parse fails anyway.
constexpr auto descend = [](auto &ctx) {
  _pass(ctx) = ++_globals(ctx).depth <= deepest;
};
constexpr auto ascend = [](auto &ctx) { --_globals(ctx).depth; };

// Copies a raw[] span of the source into a string. raw[] avoids the attribute
// merge rules for a number's optional exponent.
constexpr auto capture = [](auto &ctx) {
  _val(ctx) = std::string(_attr(ctx).begin(), _attr(ctx).end());
};

// A run of signs reduces to one bit kept in the rule's local, so `- -x` is x.
constexpr auto flip = [](auto &ctx) { _locals(ctx) = !_locals(ctx); };
constexpr auto signed_operand = [](auto &ctx) {
  _val(ctx) = _locals(ctx)
                  ? append(_globals(ctx).ast, opcode<OpCode::Neg>, _attr(ctx))
                  : _attr(ctx);
};

// Function calls are resolved through the opcode tables, so a new opcode is
// callable from text with no edit here.
constexpr auto apply = [](auto &ctx) {
  Building &building = _globals(ctx);
  const Call &invocation = _attr(ctx);
  const auto op = opcode_of(invocation.name);
  const auto refuse = [&](const errc why) {
    building.why = why;
    _pass(ctx) = false;
  };
  if (!op || !callable(*op)) {
    refuse(errc::unknown_function);
  } else if (arity_of(*op) != invocation.args.size()) {
    refuse(errc::wrong_argument_count);
  } else {
    Term t{.op = *op};
    std::ranges::copy(invocation.args, t.in.begin());
    _val(ctx) = append(building.ast, t);
  }
};

} // namespace act

// --- the grammar ------------------------------------------------------------
// As in Python, `**` is exponentiation, `2**-1` is legal and `-x**2` is
// -(x**2).

constexpr bp::rule<struct identifier_tag, std::string> identifier =
    "identifier";
constexpr bp::rule<struct literal_tag, std::string> literal = "number";
constexpr bp::rule<struct number_tag, std::uint32_t> number = "number";
constexpr bp::rule<struct symbol_tag, std::uint32_t> symbol = "identifier";
constexpr bp::rule<struct call_text_tag, Call> call_text = "function call";
constexpr bp::rule<struct call_tag, std::uint32_t> call = "function call";
constexpr bp::rule<struct group_tag, std::uint32_t> group =
    "parenthesised expression";
constexpr bp::rule<struct operand_tag, std::uint32_t> operand = "operand";
constexpr bp::rule<struct power_tag, std::uint32_t> power = "power";
// The bool local counts the signs read so far, modulo two.
constexpr bp::rule<struct unary_tag, std::uint32_t, bool> unary =
    "unary expression";
constexpr bp::rule<struct product_tag, std::uint32_t> product = "term";
constexpr bp::rule<struct sum_tag, std::uint32_t> sum = "expression";
constexpr bp::rule<struct expression_tag, std::uint32_t> expression =
    "expression";

constexpr auto letter =
    bp::char_('a', 'z') | bp::char_('A', 'Z') | bp::char_('_');
constexpr auto digits = +bp::char_('0', '9');
// Decimal only, so from_chars reads back exactly what was matched. Without
// lexeme[] the skipper would accept "1 . 5".
constexpr auto exponent = (bp::char_('e') | bp::char_('E')) >>
                          -(bp::char_('+') | bp::char_('-')) >> digits;
constexpr auto mantissa =
    (digits >> -(bp::char_('.') >> *bp::char_('0', '9'))) |
    (bp::char_('.') >> digits);

constexpr auto identifier_def =
    bp::raw[bp::lexeme[letter >> *(letter | bp::char_('0', '9'))]]
           [act::capture];
constexpr auto literal_def =
    bp::raw[bp::lexeme[mantissa >> -exponent]][act::capture];

constexpr auto number_def = literal[act::leaf<LiteralIndex>];
constexpr auto symbol_def = identifier[act::leaf<NameIndex>];
// At least one argument; the apply action checks the count against arity_of.
constexpr auto call_text_def = identifier >> '(' >> bp::eps[act::descend] >>
                               (expression % ',') >> bp::eps[act::ascend] >>
                               ')';
constexpr auto call_def = call_text[act::apply];
constexpr auto group_def = '(' >> bp::eps[act::descend] >> expression >>
                           bp::eps[act::ascend] >> ')';
// call is tried before symbol because both start with an identifier.
constexpr auto operand_def = number | call | symbol | group;

// Each action is attached to the operand it consumes, not to the sequence,
// whose _attr would be empty.
constexpr auto power_def =
    operand[act::assign] >>
    -(bp::lit("**") >> bp::eps[act::descend] >> unary[act::fold<OpCode::Pow>] >>
      bp::eps[act::ascend]);
// Signs are iterated rather than recursed, because a self-recursive rule loses
// its attribute. The language is the same as Python's u_expr.
constexpr auto unary_def =
    *(bp::lit('-')[act::flip] | bp::lit('+')) >> power[act::signed_operand];
constexpr auto product_def = unary[act::assign] >>
                             *((bp::lit('*') >> unary[act::fold<OpCode::Mul>]) |
                               (bp::lit('/') >> unary[act::fold<OpCode::Div>]));
constexpr auto sum_def = product[act::assign] >>
                         *((bp::lit('+') >> product[act::fold<OpCode::Add>]) |
                           (bp::lit('-') >> product[act::subtract]));
// At most one comparison, since Python and C read `a < b < c` differently.
// Two-character tokens are tried first so that `<` does not shadow `<=`.
constexpr auto
    expression_def = sum[act::assign] >>
                     -((bp::lit("<=") >> sum[act::fold<OpCode::Le>]) |
                       (bp::lit(">=") >> sum[act::fold_swapped<OpCode::Le>]) |
                       (bp::lit("==") >> sum[act::equal]) |
                       (bp::lit("!=") >> sum[act::unequal]) |
                       (bp::lit('<') >> sum[act::fold<OpCode::Lt>]) |
                       (bp::lit('>') >> sum[act::fold_swapped<OpCode::Lt>]));

BOOST_PARSER_DEFINE_RULES(identifier, literal, number, symbol, call_text, call,
                          group, operand, power, unary, product, sum,
                          expression);

} // namespace

result<Ast> parse(const std::string_view source) {
  Building building;
  // The grammar has no expectation points, so nothing should throw; the catch
  // guarantees it. The silent handler leaves building.why to report the error.
  const bp::callback_error_handler quiet;
  const auto root = [&]() noexcept -> std::optional<std::uint32_t> {
    try {
      return bp::parse(
          source,
          bp::with_error_handler(bp::with_globals(expression, building), quiet),
          bp::ws);
    } catch (...) {
      return std::optional<std::uint32_t>{};
    }
  }();

  if (!root) {
    return fail(building.why);
  }
  building.ast.root = *root;
  return std::move(building.ast);
}

} // namespace ddx::rt::text
