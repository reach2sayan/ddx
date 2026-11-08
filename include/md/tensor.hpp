#pragma once

#include "md/layouts.hpp"
#include "md/md.hpp"
#include "symbolic/expressions.hpp"
#include "util/config.hpp"
#include "util/meta.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>

namespace ddx::impl {

// The shape of a derivative tensor of order Order over N variables, and the
// same with a leading output axis, as a vector-valued Equation produces.
// The pack is expanded through decltype(n)::value, never a parameter's value,
// which MSVC refuses as a template argument.
template <std::size_t N, std::size_t Order>
using uniform_extents_t = decltype(meta::unpack(
    meta::repeat_t<meta::list, meta::index_t<N>, Order>{}, [](auto... n) {
      return md::extents<std::size_t, decltype(n)::value...>{};
    }));
template <std::size_t Lead, std::size_t N, std::size_t Order>
using stacked_extents_t = decltype(meta::unpack(
    meta::repeat_t<meta::list, meta::index_t<N>, Order>{}, [](auto... n) {
      return md::extents<std::size_t, Lead, decltype(n)::value...>{};
    }));

// An owning, constexpr-usable tensor with compile-time extents and a layout.
// There is no rank-reducing view, since packed layouts have no cheap one.
template <Numeric S, md::CStaticExtents Ext, typename Layout = md::layout_right>
class md_tensor {
public:
  using value_type = S;
  using extents_type = Ext;
  using layout_type = Layout;
  using mapping_type = typename Layout::template mapping<Ext>;
  using index_type = typename Ext::index_type;
  using reference = S &;

  static constexpr std::size_t rank() noexcept { return Ext::rank(); }

private:
  static_assert(std::constructible_from<mapping_type, Ext>,
                "md_tensor needs a layout whose mapping is constructible from "
                "bare extents; layouts carrying extra state (layout_stride) "
                "cannot be owned this way");
  static_assert(rank() > 0, "md_tensor: rank 0 is just S");

  static constexpr mapping_type kMapping{Ext{}};
  static constexpr std::size_t kSize =
      static_cast<std::size_t>(kMapping.required_span_size());

  std::array<S, kSize> data_{};

public:
  constexpr md_tensor() noexcept = default;

  // The number of stored cells, fewer than the addressable ones when packed.
  [[nodiscard]] static constexpr std::size_t size() noexcept { return kSize; }
  [[nodiscard]] static constexpr index_type extent(std::size_t r) noexcept {
    return Ext{}.extent(r);
  }

  [[nodiscard]] constexpr auto *data(this auto &&self) noexcept {
    return DDX_FWD(self).data_.data();
  }

  template <std::integral... I>
    requires(sizeof...(I) == Ext::rank())
  [[nodiscard]] constexpr auto &operator[](this auto &&self,
                                           I... idx) noexcept {
    return DDX_FWD(self).data_[static_cast<std::size_t>(
        kMapping(static_cast<index_type>(idx)...))];
  }

  [[nodiscard]] constexpr decltype(auto)
  at_index(this auto &&self,
           const std::array<index_type, Ext::rank()> &idx) noexcept {
    return meta::unpack(
        meta::iota<Ext::rank()>, [&](auto... k) -> decltype(auto) {
          return DDX_FWD(self)
              .data_[static_cast<std::size_t>(kMapping(idx[k]...))];
        });
  }

  [[nodiscard]] friend constexpr bool operator==(const md_tensor &a,
                                                 const md_tensor &b) noexcept {
    return a.data_ == b.data_;
  }
};

// A derivative tensor, packed by symmetry by default.
template <Numeric S, std::size_t N, std::size_t Order,
          typename Layout = layout_simplex_packed>
using nd_tensor_t = md_tensor<S, uniform_extents_t<N, Order>, Layout>;

// One derivative tensor per output, stacked along a dense leading axis.
template <Numeric S, std::size_t Lead, std::size_t N, std::size_t Order,
          typename Layout = layout_leading_simplex<1>>
using nd_stack_t = md_tensor<S, stacked_extents_t<Lead, N, Order>, Layout>;

} // namespace ddx::impl
