#pragma once

#include "rt/builder.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"
#include "rt/text/ast.hpp"
#include "util/error.hpp"

#include <algorithm>
#include <charconv>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

// An Ast into somebody's arena.  Every node goes through RTExpression::form,
// which interns, folds and carries poison, so nothing here re-implements what
// building an expression by hand already does.
namespace ddx::rt::text {

namespace detail {

// The literal as written, read at T rather than at double and then narrowed:
// the grammar has no types, so "1/2" is T(1)/T(2) whatever T is.  Whole or not
// at all -- a trailing character is a literal the grammar should not have
// matched.
template <std::floating_point T>
[[nodiscard]] result<T> scalar_of(const std::string_view text) noexcept {
  T v{};
  const auto *const end = text.data() + text.size();
  const auto [stopped, ec] = std::from_chars(text.data(), end, v);
  return ec == std::errc{} && stopped == end ? result<T>{v}
                                             : fail(errc::bad_syntax);
}

// One term, with its symbols and its operands already standing.
template <std::floating_point T>
[[nodiscard]] constexpr result<RTExpression<T>>
form(const Ast &ast, std::span<const RTExpression<T>> symbols,
     std::span<const RTExpression<T>> built, const Term &t) {
  using Expr = RTExpression<T>;
  const auto at = [](std::span<const Expr> from,
                     const std::uint32_t i) -> result<Expr> {
    return i < from.size() ? result<Expr>{from[i]} : fail(errc::bad_syntax);
  };

  if (is_leaf(t.op)) {
    return std::visit(
        [&]<typename Index>(const Index &leaf) -> result<Expr> {
          if constexpr (std::same_as<Index, NameIndex>) {
            return at(symbols, leaf.i);
          } else if constexpr (std::same_as<Index, LiteralIndex>) {
            // Pending, not lit(): a literal that has not met a symbol yet
            // stays out of the graph, so "3 + 4" folds without a node and
            // `x * 2` builds exactly what the same expression written in C++
            // builds.
            return (leaf.i < ast.literals.size()
                        ? scalar_of<T>(ast.literals[leaf.i])
                        : result<T>{fail(errc::bad_syntax)})
                .transform([](const T v) { return Expr{v}; });
          } else {
            return fail(errc::bad_syntax);
          }
        },
        t.leaf);
  }
  if (arity_of(t.op) == 1) {
    return at(built, t.a).transform([&t](const Expr &u) {
      return Expr::form(t.op, u);
    });
  }
  if (arity_of(t.op) == 3) {
    return at(built, t.a).and_then([&](const Expr &c) {
      return at(built, t.b).and_then([&](const Expr &x) {
        return at(built, t.c).transform([&](const Expr &y) {
          return Expr::form(t.op, c, x, y);
        });
      });
    });
  }
  return at(built, t.a).and_then([&](const Expr &l) {
    return at(built, t.b).transform([&](const Expr &r) {
      return Expr::form(t.op, l, r);
    });
  });
}

} // namespace detail

// The arena gains one symbol per free identifier and one node per term; the
// answer is the root.
template <std::floating_point T>
[[nodiscard]] result<RTExpression<T>> lower(Builder<T> &arena, const Ast &ast) {
  using Expr = RTExpression<T>;

  // Every symbol before any node, in the order the text first named them: the
  // arena is then a function of the model rather than of the order the parser
  // happened to reduce it in, which is what lets a model that was written down
  // be the same nodes as the same model built by hand.
  std::vector<Expr> symbols;
  symbols.reserve(ast.names.size());
  std::ranges::transform(
      ast.names, std::back_inserter(symbols),
      [&arena](const std::string &name) { return var(arena, name); });

  // Children before parents, so a term's operands are always already built and
  // the pass never looks forward.
  std::vector<Expr> built;
  built.reserve(ast.terms.size());
  result<void> ok{};
  for (const Term &t : ast.terms) {
    ok = ok.and_then([&] {
      return detail::form<T>(ast, symbols, built, t)
          .transform([&built](Expr e) { built.push_back(std::move(e)); });
    });
  }
  return ok.and_then([&]() -> result<Expr> {
    return ast.root < built.size() ? result<Expr>{built[ast.root]}
                                   : fail(errc::bad_syntax);
  });
}

} // namespace ddx::rt::text
