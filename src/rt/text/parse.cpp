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

// Python's arithmetic grammar
namespace ddx::rt::text {
namespace {

namespace bp = boost::parser;
static_assert(BOOST_PARSER_USE_STD_TUPLE,
              "rt::text: Boost.Parser must use std::tuple, not Hana");

// The name and arguments of a call, which the grammar reads as one sequence and
// an aggregate attribute takes apart.
struct Call {
  std::string name;
  std::vector<std::uint32_t> args;
};

// What the actions build.  Carried through with_globals rather than captured:
// BOOST_PARSER_DEFINE_RULES wants its rules at namespace scope, where a lambda
// has nothing to capture.
struct Building {
  Ast ast;
  errc why = errc::bad_syntax; // what the refusing action meant, if one did
};

[[nodiscard]] constexpr std::uint32_t append(Ast &ast, const Term t) {
  ast.terms.push_back(t);
  return static_cast<std::uint32_t>(ast.terms.size() - 1);
}
// An `Op` term over its operands, the arity checked as it compiles.
template <OpCode Op, std::same_as<std::uint32_t>... Ids>
  requires(sizeof...(Ids) == arity_of(Op))
[[nodiscard]] constexpr std::uint32_t
append(Ast &ast, std::integral_constant<OpCode, Op>, const Ids... in) {
  Term t{.op = Op};
  std::ranges::copy(std::array{in...}, t.in.begin());
  return append(ast, t);
}

// Where `word` sits in `table`, putting it there if it does not yet.
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
//
// Generic where the grammar repeats itself: every binary operator folds the
// same way and differs only in its opcode, and the two leaves differ only in
// which of the Ast's tables names them.  What is left is the handful of places
// the grammar and the opcode tables genuinely do not line up.
namespace act {

// _val is everything read so far and _attr the operand just read, which is what
// makes a left-associative fold one line.  Exponentiation folds the same way;
// it is the grammar, not the action, that makes it right-associative.
template <OpCode Op>
constexpr auto fold = [](auto &ctx) {
  _val(ctx) = append(_globals(ctx).ast, opcode<Op>, _val(ctx), _attr(ctx));
};

// `a > b` is `b < a` and `a >= b` is `b <= a`: only two comparison opcodes
// exist, and the other two spellings are those read the other way round -- the
// same composition rt::gt and rt::ge make in expressions.hpp.
template <OpCode Op>
constexpr auto fold_swapped = [](auto &ctx) {
  _val(ctx) = append(_globals(ctx).ast, opcode<Op>, _attr(ctx), _val(ctx));
};

// `a == b` is `le(a,b) * le(b,a)` -- both ways round, so a NaN operand answers
// 0 `a != b` is `1 - (a == b)`: the compositions operator== and operator!= make
// in expressions.hpp
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

// There is no Sub opcode -- the label "-" is unary Neg -- so a difference is
// the sum of a negation, which is what the C++ operator- builds as well.
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

// A token is a span of the source, and raw[] says so: the attribute of a
// sequence of char parsers is a merge rule per shape, and a number's shape has
// an optional exponent in it.
constexpr auto capture = [](auto &ctx) {
  _val(ctx) = std::string(_attr(ctx).begin(), _attr(ctx).end());
};

// A run of signs is one bit -- `- -x` is x -- kept in the rule's local.
constexpr auto flip = [](auto &ctx) { _locals(ctx) = !_locals(ctx); };
constexpr auto signed_operand = [](auto &ctx) {
  _val(ctx) = _locals(ctx)
                  ? append(_globals(ctx).ast, opcode<OpCode::Neg>, _attr(ctx))
                  : _attr(ctx);
};

// The whole of the function table is opcode_of, callable and arity_of, so a
// transcendental added to the opcode tables becomes callable from text with no
// edit here.
//
// Its own two codes rather than unknown_symbol and wrong_arity: those speak
// about a *point* -- its symbols and how many values it carries -- and reading
// either one back for a misspelt call describes the wrong thing entirely.
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
//
// Python's readings and its spellings: `**` is exponentiation and `^` is not.
// power's right operand reaches over unary, which is what makes `2**-1` legal;
// unary sitting below power is what makes `-x**2` mean -(x**2).

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
// The bool is the rule's local: how many signs it has read, modulo two.
constexpr bp::rule<struct unary_tag, std::uint32_t, bool> unary =
    "unary expression";
constexpr bp::rule<struct product_tag, std::uint32_t> product = "term";
constexpr bp::rule<struct sum_tag, std::uint32_t> sum = "expression";
constexpr bp::rule<struct expression_tag, std::uint32_t> expression =
    "expression";

constexpr auto letter =
    bp::char_('a', 'z') | bp::char_('A', 'Z') | bp::char_('_');
constexpr auto digits = +bp::char_('0', '9');
// Decimal only: no hex, no suffix, no digit separator, so from_chars reads back
// exactly what was matched.  lexeme[] or the skipper would pass "1 . 5".
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
// One argument at least; the action below checks the count against
// arity_of, which is 1, 2 or -- for select -- 3.
constexpr auto call_text_def = identifier >> '(' >> (expression % ',') >> ')';
constexpr auto call_def = call_text[act::apply];
constexpr auto group_def = '(' >> expression >> ')';
// call before symbol: both open with an identifier.
constexpr auto operand_def = number | call | symbol | group;

// Each action rides the operand it consumes, never the sequence that reaches
// it: a sequence pairing a literal with one attribute answers _attr with none.
constexpr auto power_def = operand[act::assign] >>
                           -(bp::lit("**") >> unary[act::fold<OpCode::Pow>]);
// The signs iterate rather than recurse: a rule that names itself inside its
// own definition reports no attribute back to the reference, and ("-"|"+")*
// power is the same language as Python's right-recursive u_expr anyway.
constexpr auto unary_def =
    *(bp::lit('-')[act::flip] | bp::lit('+')) >> power[act::signed_operand];
constexpr auto product_def = unary[act::assign] >>
                             *((bp::lit('*') >> unary[act::fold<OpCode::Mul>]) |
                               (bp::lit('/') >> unary[act::fold<OpCode::Div>]));
constexpr auto sum_def = product[act::assign] >>
                         *((bp::lit('+') >> product[act::fold<OpCode::Add>]) |
                           (bp::lit('-') >> product[act::subtract]));
// One comparison and no more.  Python chains `a < b < c` into a conjunction and
// C folds it into `(a < b) < c`; refusing it says so, where either reading
// would have been a silent choice between two languages.  The two-character
// tokens before the one-character ones, or the shorter token wins and leaves
// an `=` nobody can parse.
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
  // No expectation points in the grammar, so nothing above throws; the catch is
  // what keeps that a property of this file rather than of every caller.
  // With no callbacks it says only that the parse failed, and leaves the errc
  // to say what about it did.
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
