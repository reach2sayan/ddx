#pragma once

#include "rt/equation.hpp"
#include "rt/expressions.hpp"
#include "rt/text/ast.hpp"
#include "rt/text/lower.hpp"
#include "util/config.hpp" // DDX_TRY
#include "util/error.hpp"
#include "util/meta.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

// Parses an equation from text, one string per function, such as
// equation("x*x + y*y - 4", "x*y - 1").
namespace ddx::rt {

template <impl::Numeric T = double, std::convertible_to<std::string_view>... Ss>
  requires std::floating_point<T>
[[nodiscard]] auto equation(const std::string_view first, const Ss &...rest)
    -> result<
        meta::repeat_t<impl::Equation, Expression<T>, 1 + sizeof...(Ss)>> {
  using Eq = meta::repeat_t<impl::Equation, Expression<T>, 1 + sizeof...(Ss)>;
  const std::array sources{first, std::string_view{rest}...};

  auto arena = std::make_unique<Arena<T>>();
  std::vector<Expression<T>> roots;
  roots.reserve(sources.size());

  // Every source is parsed before create() seals the arena, so symbols
  // first named in a later function still get a slot.
  for (const std::string_view source : sources) {
    DDX_TRY(ast, text::parse(source));
    DDX_TRY(root, text::lower(*arena, *ast));
    roots.push_back(*std::move(root));
  }
  return meta::unpack(meta::iota<1 + sizeof...(Ss)>, [&](auto... i) {
    return detail::Creating::create<Eq>(std::move(arena), roots[i]...);
  });
}

} // namespace ddx::rt
