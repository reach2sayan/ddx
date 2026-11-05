#pragma once

#include "rt/arena.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"
#include "rt/text/ast.hpp"
#include "util/config.hpp" // DDX_TRY
#include "util/error.hpp"

#include <algorithm>
#include <charconv>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

// An Ast into somebody's arena.  Every node goes through Expression::form,
// which interns, folds and carries poison, so nothing here re-implements what
// building an expression by hand already does.
namespace ddx::rt::text {

namespace detail {

// The literal as written, read at T rather than at double and then narrowed:
// the grammar has no types, so "1/2" is T(1)/T(2) whatever T is.  Whole or not
// at all -- a trailing character is a literal the grammar should not have
// matched.  Past the range is the infinity or the zero it rounds to, as a
// number in source would be.
//
// Too small or too large, from_chars says only that; 0.d x 10^k is below one
// exactly when k <= 0, k being the exponent plus the leading digit's place.
[[nodiscard]] inline bool below_one(const std::string_view text) noexcept {
  const auto e = text.find_first_of("eE");
  const std::string_view mantissa = text.substr(0, e);
  std::int32_t exponent = 0;
  if (e != text.npos) {
    const char *first = text.data() + e + 1;
    first += *first == '+' ? 1 : 0;
    const auto [_, ec] =
        std::from_chars(first, text.data() + text.size(), exponent);
    if (ec == std::errc::result_out_of_range) {
      return *first == '-';
    }
  }
  const auto point =
      static_cast<std::int64_t>(std::min(mantissa.find('.'), mantissa.size()));
  const auto lead = static_cast<std::int64_t>(mantissa.find_first_not_of("0."));
  return exponent + (lead < point ? point - lead : point + 1 - lead) <= 0;
}

template <std::floating_point T>
[[nodiscard]] result<T> scalar_of(const std::string_view text) noexcept {
  T v{};
  const auto *const end = text.data() + text.size();
  const auto [stopped, ec] = std::from_chars(text.data(), end, v);
  if (stopped != end) {
    return fail(errc::bad_syntax);
  } else if (ec == std::errc::result_out_of_range) {
    return below_one(text) ? T{0} : std::numeric_limits<T>::infinity();
  }
  return ec == std::errc{} ? result<T>{v} : fail(errc::bad_syntax);
}

// One term, with its symbols and its operands already standing.
template <std::floating_point T>
[[nodiscard]] constexpr result<Expression<T>>
form(const Ast &ast, std::span<const Expression<T>> symbols,
     std::span<const Expression<T>> built, const Term &t) {
  using Expr = Expression<T>;
  const auto at = [](std::span<const Expr> from,
                     const std::uint32_t i) -> result<Expr> {
    return i < from.size() ? result<Expr>{from[i]} : fail(errc::bad_syntax);
  };
  // The op over its first sizeof...(I) operands, every one already built.
  const auto over =
      [&]<std::size_t... I>(std::index_sequence<I...>) -> result<Expr> {
    const auto stands = [&built](std::uint32_t i) { return i < built.size(); };
    return std::ranges::all_of(std::span{t.in}.first(sizeof...(I)), stands)
               ? result<Expr>{Expr::form(t.op, built[t.in[I]]...)}
               : fail(errc::bad_syntax);
  };

  switch (kind_of(t.op)) {
  case Kind::Const:
  case Kind::Input:
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
  case Kind::Unary:
    return over(std::make_index_sequence<1>{});
  case Kind::Binary:
    return over(std::make_index_sequence<2>{});
  case Kind::Ternary:
    return over(std::make_index_sequence<3>{});
  case Kind::Extern:
  case Kind::Nary:
  case Kind::Tensor:
    break; // the parser refuses the names
  }
  return fail(errc::bad_syntax);
}

} // namespace detail

// The arena gains one symbol per free identifier and one node per term; the
// answer is the root.
template <std::floating_point T>
[[nodiscard]] result<Expression<T>> lower(Arena<T> &arena, const Ast &ast) {
  using Expr = Expression<T>;

  // Every symbol before any node, in the order the text first named them: the
  // arena is then a function of the model rather than of the order the parser
  // happened to reduce it in, which is what lets a model that was written down
  // be the same nodes as the same model built by hand.
  arena.reserve(arena.size() + ast.names.size() + ast.terms.size(),
                arena.symbols().size() + ast.names.size());
  std::vector<Expr> symbols;
  symbols.reserve(ast.names.size());
  std::ranges::transform(
      ast.names, std::back_inserter(symbols),
      [&arena](const std::string &name) { return var(arena, name); });

  // Children before parents, so a term's operands are always already built and
  // the pass never looks forward.
  std::vector<Expr> built;
  built.reserve(ast.terms.size());
  for (const Term &t : ast.terms) {
    DDX_TRY(e, detail::form<T>(ast, symbols, built, t));
    built.push_back(*std::move(e));
  }
  if (ast.root >= built.size()) {
    return fail(errc::bad_syntax);
  }
  return built[ast.root];
}

} // namespace ddx::rt::text
