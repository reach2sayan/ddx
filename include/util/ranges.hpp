#pragma once

#include "util/config.hpp"

#include <algorithm>
#include <concepts>
#include <iterator>
#include <ranges>
#include <utility>

namespace ddx::impl {

// r | to<C>(), piping to std::ranges::to ourselves because clang rejects the
// libstdc++ 14.2 pipe adaptor during constraint checking.
template <typename C> struct to_closure {
  [[nodiscard]] friend constexpr C operator|(std::ranges::input_range auto &&r,
                                             to_closure)
    requires requires { std::ranges::to<C>(DDX_FWD(r)); }
  {
    return std::ranges::to<C>(DDX_FWD(r));
  }
};

template <typename C> [[nodiscard]] constexpr to_closure<C> to() noexcept {
  return {};
}

// Stands in for c.append_range(r), which libstdc++ 14 lacks.
constexpr void append(auto &c, std::ranges::input_range auto &&r)
  requires std::convertible_to<std::ranges::range_reference_t<decltype(r)>,
                               std::ranges::range_value_t<decltype(c)>>
{
  if constexpr (std::ranges::sized_range<decltype(r)>) {
    // resize grows geometrically, where an exact reserve copies c every time.
    const auto at = std::ranges::ssize(c);
    c.resize(c.size() + std::ranges::size(r));
    std::ranges::copy(DDX_FWD(r), std::ranges::begin(c) + at);
  } else {
    std::ranges::copy(DDX_FWD(r), std::back_inserter(c));
  }
}

} // namespace ddx::impl
