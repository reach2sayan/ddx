#pragma once

#include "rt/equation.hpp"
#include "rt/expressions.hpp"
#include "rt/text/ast.hpp"
#include "rt/text/lower.hpp"
#include "util/error.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

// An equation written down rather than built:
//
//   const auto eq  = ddx::rt::equation("sin(x)*y + 3");
//   const auto sys = ddx::rt::equation("x*x + y*y - 4", "x*y - 1");
//
// Beside the other spellings of equation(), in a header rt/equation.hpp does
// not include: text knows what an arena is and the arena does not know that
// text exists.  One string per function, so the output count is the pack's
// size -- which is what Equation needs it to be, output_dim being part of the
// type.
namespace ddx::rt {

// A cache leads, as it does over an expression pack and for the same reason.
template <typename C, impl::Numeric T = double,
          std::convertible_to<std::string_view>... Ss>
  requires std::floating_point<T> && CValueCache<C, T>
[[nodiscard]] auto equation(C memo, const std::string_view first,
                            const Ss &...rest) {
  return impl::index_apply<sizeof...(Ss)>([&]<std::size_t... I>() {
    using Eq = impl::Equation<RTExpression<T>, detail::Repeat<I, T>..., C>;
    const std::array sources{first, std::string_view{rest}...};

    auto arena = std::make_unique<Builder<T>>();
    std::vector<RTExpression<T>> roots;
    roots.reserve(sources.size());

    // Every source read before any of them is an equation: create() seals the
    // arena, and a symbol first named by the second function still wants a
    // slot.
    result<void> ok{};
    for (const std::string_view source : sources) {
      ok = ok.and_then([&] {
        return text::parse(source)
            .and_then([&arena](const text::Ast &ast) {
              return text::lower(*arena, ast);
            })
            .transform(
                [&roots](RTExpression<T> e) { roots.push_back(std::move(e)); });
      });
    }
    return ok.transform([&] {
      return Eq::create(std::move(memo), std::move(arena), roots[0],
                        roots[I + 1]...);
    });
  });
}

template <impl::Numeric T = double>
  requires std::floating_point<T>
[[nodiscard]] auto
equation(const std::string_view first,
         const std::convertible_to<std::string_view> auto &...rest) {
  return equation(detail::NoCache<T>{}, first, rest...);
}

} // namespace ddx::rt
