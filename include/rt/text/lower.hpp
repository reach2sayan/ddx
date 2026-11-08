#pragma once

#include "rt/arena.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"
#include "rt/text/ast.hpp"
#include "util/config.hpp" // DDX_TRY
#include "util/error.hpp"
#include "util/meta.hpp"

#include <boost/hof/match.hpp>

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

// Lowers an Ast into an arena through Expression::form, which interns, folds
// and propagates poison exactly as hand-built expressions do.
namespace ddx::rt::text {

namespace detail {

// Whether an out-of-range literal underflows rather than overflows, since
// from_chars does not say. 0.d x 10^k is below one exactly when k <= 0.
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

// Reads a literal directly at T. An out-of-range value becomes zero or
// infinity, and a partially consumed literal is a syntax error.
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

// Builds one term whose symbols and operands are already built.
template <std::floating_point T>
[[nodiscard]] constexpr result<Expression<T>>
form(const Ast &ast, std::span<const Expression<T>> symbols,
     std::span<const Expression<T>> built, const Term &t) {
  using Expr = Expression<T>;
  // Applies the op to its first `arity` operands, checking each exists.
  const auto over = [&](auto arity) -> result<Expr> {
    const auto stands = [&built](std::uint32_t i) { return i < built.size(); };
    if (!std::ranges::all_of(std::span{t.in}.first(arity), stands)) {
      return fail(errc::bad_syntax);
    }
    return meta::unpack(meta::iota<arity>, [&](auto... i) {
      return Expr::form(t.op, built[t.in[i]]...);
    });
  };

  switch (kind_of(t.op)) {
  case Kind::Const:
  case Kind::Input:
    return std::visit(
        boost::hof::match(
            [&](const NameIndex &leaf) -> result<Expr> {
              return leaf.i < symbols.size() ? result<Expr>{symbols[leaf.i]}
                                             : fail(errc::bad_syntax);
            },
            // A pending literal rather than lit(), so "3 + 4" folds without a
            // node and the result matches the same expression built in C++.
            [&](const LiteralIndex &leaf) -> result<Expr> {
              return (leaf.i < ast.literals.size()
                          ? scalar_of<T>(ast.literals[leaf.i])
                          : result<T>{fail(errc::bad_syntax)})
                  .transform([](const T v) { return Expr{v}; });
            },
            [](std::monostate) -> result<Expr> {
              return fail(errc::bad_syntax);
            }),
        t.leaf);
  case Kind::Unary:
    return over(meta::index_t<1>{});
  case Kind::Binary:
    return over(meta::index_t<2>{});
  case Kind::Ternary:
    return over(meta::index_t<3>{});
  case Kind::Extern:
  case Kind::Nary:
  case Kind::Tensor:
    break; // the parser refuses the names
  }
  return fail(errc::bad_syntax);
}

} // namespace detail

// Adds one symbol per free identifier and one node per term, returning the
// root.
template <std::floating_point T>
[[nodiscard]] result<Expression<T>> lower(Arena<T> &arena, const Ast &ast) {
  using Expr = Expression<T>;

  // Symbols are declared first, in order of first use, so the arena matches
  // the same model built by hand regardless of parse order.
  arena.reserve(arena.size() + ast.names.size() + ast.terms.size(),
                arena.symbols().size() + ast.names.size());
  std::vector<Expr> symbols;
  symbols.reserve(ast.names.size());
  std::ranges::transform(
      ast.names, std::back_inserter(symbols),
      [&arena](const std::string &name) { return var(arena, name); });

  // Terms are stored children first, so operands are always already built.
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
