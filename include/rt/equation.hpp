#pragma once

#include "rt/core.hpp"
#include "symbolic/equation.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <concepts>
#include <memory>
#include <span>
#include <utility>
#include <vector>

// Keyed on the RTExpression<T> *pattern*: a `requires` over <TFirst, TRest...>
// would be ambiguous with the compile-time specialisation.
namespace ddx::impl {

namespace rt_detail {

template <typename T> struct is_output {
  template <typename U> using fn = std::is_same<U, rt::RTExpression<T>>;
};

// Outputs, or rt::detail::Dynamic alone -- a count the type does not fix --
// and nothing else.
template <typename T, typename... Rest>
inline constexpr bool dynamic_outputs =
    std::same_as<boost::mp11::mp_list<Rest...>,
                 boost::mp11::mp_list<rt::detail::Dynamic>>;
template <typename T, typename... Rest>
concept COutputPack =
    dynamic_outputs<T, Rest...> ||
    boost::mp11::mp_all_of_q<boost::mp11::mp_list<Rest...>,
                             is_output<T>>::value;

// The count's policy: fixed by the pack, or Dynamic as written.
template <typename T, typename... Rest>
using policy_of =
    boost::mp11::mp_if_c<dynamic_outputs<T, Rest...>, rt::detail::Dynamic,
                         rt::detail::Fixed<1 + sizeof...(Rest)>>;

} // namespace rt_detail

// rt::detail::Core with the output count in the type -- evaluate() answers a
// T for one function, and a file holding another count is refused -- or, with
// Dynamic in the pack, a count only known at run time: one function per
// species read from a database, every value block a vector, and the
// one-function members refusing with errc::not_single_output.
template <Numeric T, typename... Rest>
  requires rt_detail::COutputPack<T, Rest...>
class Equation<rt::RTExpression<T>, Rest...>
    : public rt::detail::Facade<Equation<rt::RTExpression<T>, Rest...>, T,
                                rt_detail::policy_of<T, Rest...>> {
  using Outputs = rt_detail::policy_of<T, Rest...>;
  using Base = rt::detail::Facade<Equation, T, Outputs>;
  friend rt::detail::Core<T>;

public:
  using Base::create;
  // What the type fixes: 0 where it fixes nothing (a System).
  static constexpr std::size_t output_dim = Outputs::expected.value_or(0);

  template <typename... Es>
  static constexpr bool are_outputs =
      Outputs::expected == sizeof...(Es) + 1 &&
      (std::same_as<Es, rt::RTExpression<T>> && ...);

  template <typename... Es>
    requires are_outputs<Es...>
  [[nodiscard]] static constexpr Equation
  create(std::unique_ptr<rt::Builder<T>> owned, rt::RTExpression<T> first,
         Es... rest) {
    const std::array outputs{first, rest...};
    return Base::create(std::move(owned), outputs);
  }

private:
  using Base::Base;
};

} // namespace ddx::impl

namespace ddx::rt {

using jit::Backend;
using jit::Lanes;
using jit::Level;

// An equation whose output count is a run-time number.
template <impl::Numeric T = double>
using System = impl::Equation<RTExpression<T>, detail::Dynamic>;

// Over expressions already built in a caller's own arena, which the equation
// copies: the caller may go on building in it.  Partial
// specialisations contribute no deduction guides, so this is the whole of CTAD.
template <impl::Numeric T, typename... Ts>
  requires(std::same_as<Ts, RTExpression<T>> && ...)
[[nodiscard]] constexpr auto equation(RTExpression<T> first, Ts... rest) {
  return impl::Equation<RTExpression<T>, Ts...>::create(nullptr, first,
                                                        rest...);
}

// Build an Equation:
//
//   const auto eq = ddx::rt::equation([] {
//     const auto x = var("x");
//     const auto y = var("y");
//     return exp(x) * sin(y);
//   });
//
// A callback returning an array is a system with that many functions in its
// type; one returning any other range -- a vector built over a species list
// -- is a System, with the count it held.

namespace detail {

// `make(type_identity<Eq>, roots...)` over the Equation type the build's shape
// names: one root, a tuple-like of them -- the count lives in the type, so the
// size must too -- or any other range, whose count is whatever it holds.
template <impl::Numeric T, typename Built>
[[nodiscard]] auto over(const Built &built, auto &&make) {
  if constexpr (std::same_as<Built, RTExpression<T>>) {
    return make(std::type_identity<impl::Equation<RTExpression<T>>>{}, built);
  } else if constexpr (requires { std::tuple_size<Built>::value; }) {
    constexpr std::size_t outputs = std::tuple_size_v<Built>;
    static_assert(outputs > 0,
                  "equation: a system needs at least one function");
    return impl::index_apply<outputs - 1>([&]<std::size_t... Rest>() {
      return make(std::type_identity<
                      impl::Equation<RTExpression<T>, Repeat<Rest, T>...>>{},
                  built[0], built[Rest + 1]...);
    });
  } else {
    const auto outputs = built | impl::to<std::vector<RTExpression<T>>>();
    return make(std::type_identity<System<T>>{},
                std::span<const RTExpression<T>>{outputs});
  }
}

} // namespace detail

template <impl::Numeric T = double>
[[nodiscard]] auto equation(std::invocable auto &&assemble) {
  auto [arena, built] = detail::assemble<T>(DDX_FWD(assemble));
  return detail::over<T>(
      built, [&]<typename Eq>(std::type_identity<Eq>, auto... roots) {
        return Eq::create(std::move(arena), roots...);
      });
}

} // namespace ddx::rt
