#pragma once

// Shared pieces of every std::formatter in the library.

#include <algorithm>
#include <format>
#include <ostream>
#include <ranges>
#include <string_view>
#include <type_traits>

namespace ddx::impl::detail {

// U+03B5, as UTF-8 bytes so it does not depend on the source encoding.
inline constexpr std::string_view eps = "\xce\xb5";

inline void fmt_put(std::format_context &ctx, std::string_view s) {
  ctx.advance_to(std::ranges::copy(s, ctx.out()).out);
}

// S unconstrained: std::formattable<S, char> would recurse through the very
// std::formatter specialisation that derives from this.
template <typename S> struct dual_formatter_base {
  constexpr auto parse(std::format_parse_context &ctx) {
    return part_.parse(ctx);
  }

protected:
  static void put(std::format_context &ctx, std::string_view s) {
    fmt_put(ctx, s);
  }

  void num(std::format_context &ctx, const S &v) const {
    ctx.advance_to(part_.format(v, ctx));
  }

  // Nested series need brackets, or Dual<Dual<double>> prints 1+2ε+3+4εε.
  void term(std::format_context &ctx, const S &v) const {
    if constexpr (std::is_arithmetic_v<S>) {
      num(ctx, v);
    } else {
      put(ctx, "(");
      num(ctx, v);
      put(ctx, ")");
    }
  }

  // c0+c1ε+c2ε^2+..., zeros included, so the order stays legible.
  void series(std::format_context &ctx, const auto &coeffs) const {
    for (const auto [k, c] : coeffs | std::views::enumerate) {
      if (k > 0) {
        put(ctx, "+");
      }
      term(ctx, c);
      if (k == 1) {
        put(ctx, eps);
      } else if (k > 1) {
        ctx.advance_to(std::format_to(ctx.out(), "{}^{}", eps, k));
      }
    }
  }

private:
  std::formatter<S, char> part_{};
};

// Opt-in list for the operator<< below; each dual header adds itself.
template <typename T> inline constexpr bool is_dual_family_v = false;

} // namespace ddx::impl::detail

namespace ddx::impl {

template <typename D>
  requires detail::is_dual_family_v<std::remove_cvref_t<D>>
std::ostream &operator<<(std::ostream &out, const D &d) {
  return out << std::format("{}", d);
}

} // namespace ddx::impl
