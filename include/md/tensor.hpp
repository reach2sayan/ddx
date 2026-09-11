#pragma once

#include "md/layouts.hpp"
#include "md/md.hpp"
#include "symbolic/expressions.hpp"
#include "util/config.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/integral.hpp>
#include <boost/mp11/list.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>

namespace ddx::impl {

// An owning, constant-evaluable tensor with compile-time extents and a layout
// policy.  t[i, j, k] is one mapping call; there is no rank-reducing view,
// since the packed and sparse layouts have no cheap submdspan_mapping.

namespace detail {

template <typename... I>
using extents_of = md::extents<std::size_t, I::value...>;
template <std::size_t N, std::size_t Order>
using repeated =
    boost::mp11::mp_repeat_c<boost::mp11::mp_list<boost::mp11::mp_size_t<N>>,
                             Order>;

} // namespace detail

// The shape of a derivative tensor of order Order over N variables.
template <std::size_t N, std::size_t Order>
using uniform_extents_t =
    boost::mp11::mp_apply<detail::extents_of, detail::repeated<N, Order>>;

// The same with one leading axis: a per-output stack, as a vector-valued
// Equation produces.
template <std::size_t Lead, std::size_t N, std::size_t Order>
using stacked_extents_t = boost::mp11::mp_apply<
    detail::extents_of,
    boost::mp11::mp_push_front<detail::repeated<N, Order>,
                               boost::mp11::mp_size_t<Lead>>>;

template <Numeric S, md::CStaticExtents Ext, md::CLayoutFor<Ext> Layout>
class md_tensor;

template <Numeric S, md::CStaticExtents Ext,
          md::CLayoutFor<Ext> Layout = md::layout_right>
class md_tensor {
public:
  using element_type = S;
  using value_type = S;
  using extents_type = Ext;
  using layout_type = Layout;
  using mapping_type = typename Layout::template mapping<Ext>;
  using index_type = typename Ext::index_type;
  using reference = S &;
  using const_reference = const S &;

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

  // Stored, which under a packed layout is fewer than addressable.
  [[nodiscard]] static constexpr std::size_t size() noexcept { return kSize; }
  [[nodiscard]] static constexpr index_type extent(std::size_t r) noexcept {
    return Ext{}.extent(r);
  }

  // Const-ness rides on the qualifier, so one body serves every category.
#define DDX_MD_TENSOR_ACCESSORS(Q, SELF)                                       \
  [[nodiscard]] constexpr auto *data() Q noexcept {                            \
    return SELF.data_.data();                                                  \
  }                                                                            \
                                                                               \
  template <std::integral... I>                                                \
    requires(sizeof...(I) == Ext::rank())                                      \
  [[nodiscard]] constexpr auto &operator[](I... idx) Q noexcept {              \
    return SELF.data_[static_cast<std::size_t>(                                \
        kMapping(static_cast<index_type>(idx)...))];                           \
  }                                                                            \
                                                                               \
  [[nodiscard]] constexpr decltype(auto) at_index(                             \
      const std::array<index_type, Ext::rank()> &idx) Q noexcept {             \
    return index_apply<Ext::rank()>(                                           \
        [&]<std::size_t... K>() -> decltype(auto) {                            \
          return SELF.data_[static_cast<std::size_t>(kMapping(idx[K]...))];    \
        });                                                                    \
  }
  DDX_VALUE_CATEGORIES(DDX_MD_TENSOR_ACCESSORS)
#undef DDX_MD_TENSOR_ACCESSORS

  [[nodiscard]] friend constexpr bool operator==(const md_tensor &a,
                                                 const md_tensor &b) noexcept {
    return a.data_ == b.data_;
  }
};

// The default layout is the symmetric packing, not layout_right:
template <Numeric S, std::size_t N, std::size_t Order,
          typename Layout = layout_simplex_packed>
using nd_tensor_t = md_tensor<S, uniform_extents_t<N, Order>, Layout>;

// A per-output stack, as one rank-(Order + 1) object.  The output axis is
// dense: it indexes different functions.
template <Numeric S, std::size_t Lead, std::size_t N, std::size_t Order,
          typename Layout = layout_leading_simplex<1>>
using nd_stack_t = md_tensor<S, stacked_extents_t<Lead, N, Order>, Layout>;

// Index-based, not a pointer walk into data(): under a packed mapping a row is
// not contiguous.  The sweeps fill a plain array first, since a rank-1 mdspan
// is not a range.
template <Numeric S, md::CStaticExtents Ext, md::CLayoutFor<Ext> Layout,
          std::ranges::input_range R>
  requires(Ext::rank() == 2)
constexpr void assign_row(md_tensor<S, Ext, Layout> &t,
                          typename Ext::index_type i, R &&src) noexcept {
  for (auto &&[k, v] : std::views::zip(
           std::views::iota(typename Ext::index_type{0}, Ext{}.extent(1)),
           src)) {
    t[i, k] = v;
  }
}

} // namespace ddx::impl
