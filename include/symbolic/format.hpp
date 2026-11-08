#pragma once

#include "ops/operations.hpp"
#include "symbolic/expressions.hpp"
#include "symbolic/traits.hpp" // CVariable
#include "util/fmt.hpp"        // detail::fmt_put
#include "util/meta.hpp"

#include <boost/hof/match.hpp>

#include <format>
#include <ostream>
#include <string_view>
#include <tuple>
#include <utility>

namespace ddx::impl {

namespace detail {

// A two-child sum whose right operand is a negation prints as a subtraction.
template <CExpression E> consteval bool renders_as_subtraction() {
  return boost::hof::match(
      []<Numeric T, CExpression A, CExpression B>(
          meta::Type<Expression<SumOp<T>, A, B>>) {
        return is_negation_expr_v<std::remove_cvref_t<B>>;
      },
      [](auto) { return false; })(meta::type_c<std::remove_cvref_t<E>>);
}

// The output context and value formatter shared by a whole print walk.
template <Numeric V> struct printer {
  std::format_context &ctx;
  const std::formatter<V, char> &value_fmt;

  void put(std::string_view s) const { fmt_put(ctx, s); }
  void put(char c) const { put(std::string_view{&c, 1}); }

  // Uses the nested formatter, so a spec such as "{::.3f}" reaches every leaf.
  void put_value(const Numeric auto &v) const {
    ctx.advance_to(value_fmt.format(v, ctx));
  }
};

// Parenthesises a node looser than min_prec. A right operand must bind one
// step tighter, giving x - (y + z) but x - y * z.
template <Numeric V>
void print_infix(const printer<V> &p, const CExpression auto &e, int min_prec) {
  using U = std::remove_cvref_t<decltype(e)>;
  if constexpr (CExpressionNode<U>) {
    using Op = typename U::op_type;
    // A call is atomic to its parent.
    constexpr int prec =
        Op::notation == Notation::Function ? precedence_atom : Op::precedence;
    const bool parens = prec < min_prec;
    if (parens) {
      p.put('(');
    }

    auto &&kids = e.expressions();

    if constexpr (renders_as_subtraction<U>()) {
      print_infix(p, std::get<0>(kids), prec);
      p.put(" - ");
      auto &&inner = std::get<1>(kids).expressions();
      print_infix(p, std::get<0>(inner), prec + 1);
    } else if constexpr (Op::notation == Notation::Infix) {
      static_assert(std::tuple_size_v<typename U::children_t> == 2,
                    "infix notation is only defined for binary nodes");
      print_infix(p, std::get<0>(kids), prec);
      p.put(' ');
      p.put(Op::label);
      p.put(' ');
      print_infix(p, std::get<1>(kids), prec + 1);
    } else if constexpr (Op::notation == Notation::Prefix) {
      static_assert(std::tuple_size_v<typename U::children_t> == 1,
                    "prefix notation is only defined for unary nodes");
      p.put(Op::label);
      print_infix(p, std::get<0>(kids), prec + 1); // -(-c), never --c
    } else { // function notation brings its own parentheses
      p.put(Op::label);
      p.put('(');
      bool first = true;
      meta::for_each(kids, [&](const auto &child) {
        if (!std::exchange(first, false)) {
          p.put(", ");
        }
        print_infix(p, child, 0);
      });
      p.put(')');
    }

    if (parens) {
      p.put(')');
    }
  } else if constexpr (CVariable<U>) {
    p.put(U::label.view());
    if constexpr (U::frozen) {
      p.put("_c");
    }
  } else if constexpr (CConstant<U>) {
    p.put_value(e.get());
  } else {
    p.put_value(e.eval());
  }
}

} // namespace detail

std::ostream &operator<<(std::ostream &out, const CExpression auto &e) {
  return out << std::format("{}", e);
}

} // namespace ddx::impl

// The whole spec applies to each value; a leading ':' is optional, as for
// ranges.
template <ddx::impl::CExpression E> struct std::formatter<E, char> {
  constexpr auto parse(std::format_parse_context &ctx) {
    auto it = ctx.begin();
    if (it != ctx.end() && *it == ':') {
      ++it;
    }
    ctx.advance_to(it);
    return value_fmt_.parse(ctx);
  }

  auto format(const E &e, std::format_context &ctx) const {
    const ddx::impl::detail::printer p{ctx, value_fmt_};
    ddx::impl::detail::print_infix(p, e, 0);
    return ctx.out();
  }

private:
  std::formatter<typename E::value_type, char> value_fmt_{};
};
