#pragma once

#include "jit/kernel.hpp"
#include "util/fmt.hpp" // fmt_put

#include <format>
#include <ostream>
#include <string_view>

template <typename T> struct std::formatter<ddx::jit::Ir<T>, char> {
  constexpr auto parse(std::format_parse_context &ctx) const {
    return ctx.begin();
  }

  auto format(const ddx::jit::Ir<T> &ir, std::format_context &ctx) const {
    const auto text = ir.str();
    ddx::impl::detail::fmt_put(ctx,
                               text ? std::string_view{*text}
                                    : std::string_view{text.error().detail});
    return ctx.out();
  }
};

namespace ddx::jit {
template <typename T>
std::ostream &operator<<(std::ostream &out, const Ir<T> &ir) {
  return out << std::format("{}", ir);
}
} // namespace ddx::jit
