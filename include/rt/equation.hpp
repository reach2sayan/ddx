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

namespace ddx::rt::detail {

template <typename T> struct is_output {
  template <typename U> using fn = std::is_same<U, Expression<T>>;
};

// Outputs, or Dynamic alone -- a count the type does not fix -- and nothing
// else.
template <typename T, typename... Rest>
inline constexpr bool dynamic_outputs =
    std::same_as<boost::mp11::mp_list<Rest...>, boost::mp11::mp_list<Dynamic>>;
template <typename T, typename... Rest>
concept COutputPack = dynamic_outputs<T, Rest...> ||
                      boost::mp11::mp_all_of_q<boost::mp11::mp_list<Rest...>,
                                               is_output<T>>::value;

// The count's policy: fixed by the pack, or Dynamic as written.
template <typename T, typename... Rest>
using policy_of = boost::mp11::mp_if_c<dynamic_outputs<T, Rest...>, Dynamic,
                                       Fixed<1 + sizeof...(Rest)>>;

} // namespace ddx::rt::detail

// Keyed on the Expression<T> *pattern*: a `requires` over <TFirst, TRest...>
// would be ambiguous with the compile-time specialisation.
namespace ddx::impl {

// rt::detail::Core with the output count in the type -- evaluate() answers a
// T for one function -- or, with Dynamic in the pack, a count only known at
// run time: one function per species read from a database, every value block
// a vector, and the one-function members refusing with
// errc::not_single_output.
template <Numeric T, typename... Rest>
  requires rt::detail::COutputPack<T, Rest...>
class Equation<rt::Expression<T>, Rest...> : public rt::detail::Core<T> {
  using Outputs = rt::detail::policy_of<T, Rest...>;
  using Base = rt::detail::Core<T>;
  friend Base;

public:
  // A refusal rides on the equation rather than a result at every call site;
  // see poisoned().  `owned` is null where the outputs' arena is the caller's,
  // which is then copied.
  [[nodiscard]] static constexpr Equation
  create(std::unique_ptr<rt::Builder<T>> owned,
         std::span<const rt::Expression<T>> outputs) {
    return Base::template make<Equation>(outputs, std::move(owned));
  }
  // What the type fixes: 0 where it fixes nothing (a System).
  static constexpr std::size_t output_dim = Outputs::expected.value_or(0);

  template <typename... Es>
  static constexpr bool are_outputs =
      Outputs::expected == sizeof...(Es) + 1 &&
      (std::same_as<Es, rt::Expression<T>> && ...);

  template <typename... Es>
    requires are_outputs<Es...>
  [[nodiscard]] static constexpr Equation
  create(std::unique_ptr<rt::Builder<T>> owned, rt::Expression<T> first,
         Es... rest) {
    const std::array outputs{first, rest...};
    return create(std::move(owned),
                  std::span<const rt::Expression<T>>{outputs});
  }

  // `Outputs` is the one place the count's policy acts.
  [[nodiscard]] constexpr auto
  evaluate(const rt::detail::CPointArg<T> auto &...args) const {
    return this->values(args...).transform([](std::vector<T> f) {
      if constexpr (Outputs::scalar) {
        return f.front();
      } else {
        return f;
      }
    });
  }
  using Base::evaluate;

private:
  using Base::Base;
};

} // namespace ddx::impl

namespace ddx::rt {

using jit::Backend;
using jit::Lanes;
using jit::OptLevel;

// An equation whose output count is a run-time number.
template <impl::Numeric T = double>
using System = impl::Equation<Expression<T>, detail::Dynamic>;

// Over expressions already built in a caller's own arena, which the equation
// copies: the caller may go on building in it.  Partial
// specialisations contribute no deduction guides, so this is the whole of CTAD.
template <impl::Numeric T, typename... Ts>
  requires(std::same_as<Ts, Expression<T>> && ...)
[[nodiscard]] constexpr auto equation(Expression<T> first, Ts... rest) {
  return impl::Equation<Expression<T>, Ts...>::create(nullptr, first, rest...);
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
  if constexpr (std::same_as<Built, Expression<T>>) {
    return make(std::type_identity<impl::Equation<Expression<T>>>{}, built);
  } else if constexpr (requires { std::tuple_size<Built>::value; }) {
    constexpr std::size_t outputs = std::tuple_size_v<Built>;
    static_assert(outputs > 0,
                  "equation: a system needs at least one function");
    return impl::index_apply<outputs - 1>([&]<std::size_t... Rest>() {
      return make(std::type_identity<
                      impl::Equation<Expression<T>, Repeat<Rest, T>...>>{},
                  built[0], built[Rest + 1]...);
    });
  } else {
    const auto outputs = built | impl::to<std::vector<Expression<T>>>();
    return make(std::type_identity<System<T>>{},
                std::span<const Expression<T>>{outputs});
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
