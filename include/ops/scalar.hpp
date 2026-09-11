#pragma once

#include "ops/numeric.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ddx::impl {

template <Numeric T> class Dual;

template <typename T> inline constexpr bool is_dual_v = false;
template <Numeric T> inline constexpr bool is_dual_v<Dual<T>> = true;

template <typename X>
concept DualLike = is_dual_v<std::remove_cvref_t<X>>;

template <Numeric T> struct dual_scalar_type {
  using type = T;
};
template <Numeric T> struct dual_scalar_type<Dual<T>> {
  using type = T;
};
template <Numeric T> using dual_scalar_t = typename dual_scalar_type<T>::type;

// Unconstrained on purpose: through dual_scalar_type it would ask Numeric<Dual>
// of the very operators Numeric<Dual> is being decided by.
template <DualLike X> struct dual_value_type;
template <Numeric T> struct dual_value_type<Dual<T>> {
  using type = T;
};
template <DualLike X>
using dual_value_t = typename dual_value_type<std::remove_cvref_t<X>>::type;

template <typename A, typename B>
concept DualCompatible = DualLike<A> && DualLike<B> &&
                         std::same_as<dual_value_t<A>, dual_value_t<B>>;

// N of them nested: 2^N components, one per subset of the ε's, and the all-ones
// component is the Nth derivative -- what extract_nth reads and
// make_mixed_seed seeds for.  Ref: Fike & Alonso, AIAA 2011-886;
// docs/hyperdual_nth_order_by_example.md draws the lattice.
template <typename Inner, typename> using wrap_dual = Dual<Inner>;
template <Numeric T, std::size_t N>
using nth_dual_t = boost::mp11::mp_fold<
    boost::mp11::mp_repeat_c<boost::mp11::mp_list<void>, N>, T, wrap_dual>;

template <Numeric T> inline constexpr std::size_t dual_depth_v = 0;
template <Numeric T>
inline constexpr std::size_t dual_depth_v<Dual<T>> = 1 + dual_depth_v<T>;

template <Numeric T> auto scalar_base_impl(std::type_identity<T>) -> T;
template <Numeric T>
auto scalar_base_impl(std::type_identity<Dual<T>>)
    -> decltype(scalar_base_impl(std::type_identity<T>{}));

template <Numeric T>
using scalar_base_t = decltype(scalar_base_impl(std::type_identity<T>{}));

template <Numeric T, std::size_t N>
constexpr nth_dual_t<T, N> embed_constant(T val) noexcept {
  if constexpr (N == 0) {
    return val;
  } else {
    return nth_dual_t<T, N>{embed_constant<T, N - 1>(val),
                            nth_dual_t<T, N - 1>{}};
  }
}

template <Numeric U> struct ConstantEmbedder {
  static constexpr U embed(scalar_base_t<U> val) noexcept {
    return embed_constant<scalar_base_t<U>, dual_depth_v<U>>(val);
  }
};

// N levels down along component I: 0 is the value chain, 1 the tangent chain.
template <std::size_t N, std::size_t I>
constexpr auto component(const Numeric auto &x) noexcept {
  if constexpr (N == 0) {
    return x;
  } else {
    return component<N - 1, I>(x.template get<I>());
  }
}
template <std::size_t N>
constexpr auto get_real_part(const Numeric auto &x) noexcept {
  return component<N, 0>(x);
}

// `a op= b` spelled as `a = a op b`: the binary operators are the rule, and the
// gate is the body itself, so whatever they refuse is refused here.  CRTP,
// which also gives every level of Dual<Dual<T>> its own empty base: the outer
// and its val_ cannot share one, and a shared base costs 8 bytes of padding.
template <typename Derived> struct compound_from_binary {
#define DDX_COMPOUND_FROM_BINARY(OP)                                           \
  template <typename B>                                                        \
    requires requires(Derived &d, const B &o) { d = d OP o; }                  \
  constexpr Derived &operator OP##=(const B & o) & noexcept {                  \
    auto &self = static_cast<Derived &>(*this);                                \
    return self = self OP o;                                                   \
  }
  DDX_COMPOUND_FROM_BINARY(+)
  DDX_COMPOUND_FROM_BINARY(-)
  DDX_COMPOUND_FROM_BINARY(*)
  DDX_COMPOUND_FROM_BINARY(/)
#undef DDX_COMPOUND_FROM_BINARY
};

template <typename X>
concept DualOrArithmetic = DualLike<X> || CArithmetic<X>;

// The scalar ends of the recursions dual/dual.hpp continues for a Dual.
constexpr auto val(CArithmetic auto x) noexcept { return x; }

// Zero in every component; operator== compares val() alone and cannot say this.
constexpr bool all_zero(CArithmetic auto x) noexcept {
  return x == std::remove_cvref_t<decltype(x)>{};
}

constexpr double to_double(const Numeric auto &x) noexcept {
  return static_cast<double>(val(x));
}

} // namespace ddx::impl

namespace std {
template <ddx::impl::Numeric T>
struct tuple_size<ddx::impl::Dual<T>> : integral_constant<std::size_t, 2> {};
template <ddx::impl::Numeric T, std::size_t N>
struct tuple_element<N, ddx::impl::Dual<T>> {
  using type = T;
};
} // namespace std
