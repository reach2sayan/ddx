#pragma once

#include "rt/core.hpp"
#include "symbolic/equation.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <concepts>
#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <vector>

// Keyed on the RTExpression<T> *pattern*: a `requires` over <TFirst, TRest...>
// would be ambiguous with the compile-time specialisation.
namespace ddx::impl {

namespace rt_detail {

// An Equation is spelled over its outputs and, behind them, the cache a factory
// deduced.  Which a trailing type is is asked of the type system rather than
// assumed by position.
template <typename... Rest>
using tail_of =
    boost::mp11::mp_eval_if_c<sizeof...(Rest) == 0, void, boost::mp11::mp_back,
                              boost::mp11::mp_list<Rest...>>;

template <typename T, typename... Rest>
inline constexpr bool packs_cache = rt::CValueCache<tail_of<Rest...>, T>;

template <typename T, typename... Rest>
using cache_of = boost::mp11::mp_if_c<packs_cache<T, Rest...>, tail_of<Rest...>,
                                      rt::detail::NoCache<T>>;

template <typename T, typename... Rest>
using outputs_of = boost::mp11::mp_eval_if_c<
    !packs_cache<T, Rest...>, boost::mp11::mp_list<Rest...>,
    boost::mp11::mp_pop_back, boost::mp11::mp_list<Rest...>>;

template <typename T> struct is_output {
  template <typename U> using fn = std::is_same<U, rt::RTExpression<T>>;
};

// Outputs -- or rt::detail::Dynamic alone, a count the type does not fix --
// then at most one cache, and nothing else.
template <typename T, typename... Rest>
inline constexpr bool dynamic_outputs =
    std::same_as<outputs_of<T, Rest...>,
                 boost::mp11::mp_list<rt::detail::Dynamic>>;
template <typename T, typename... Rest>
concept COutputPack =
    dynamic_outputs<T, Rest...> ||
    boost::mp11::mp_all_of_q<outputs_of<T, Rest...>, is_output<T>>::value;

// The count's policy: fixed by the pack, or Dynamic as written.
template <typename T, typename... Rest>
using policy_of = boost::mp11::mp_if_c<
    dynamic_outputs<T, Rest...>, rt::detail::Dynamic,
    rt::detail::Fixed<1 + boost::mp11::mp_size<outputs_of<T, Rest...>>::value>>;

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
                                rt_detail::cache_of<T, Rest...>,
                                rt_detail::policy_of<T, Rest...>> {
  using Cache = rt_detail::cache_of<T, Rest...>;
  using Outputs = rt_detail::policy_of<T, Rest...>;
  using Base = rt::detail::Facade<Equation, T, Cache, Outputs>;
  friend rt::detail::Core<T, Cache>;

public:
  using Base::cached;
  using Base::create;
  // What the type fixes: rt::dynamic, 0, where it fixes nothing.
  static constexpr std::size_t output_dim = Outputs::expected.value_or(0);

  // The outputs as their own pack: the class's holds the cache too.
  template <typename... Es>
  static constexpr bool are_outputs =
      Outputs::expected == sizeof...(Es) + 1 &&
      (std::same_as<Es, rt::RTExpression<T>> && ...);

  // The cache leads, because the functions are a pack and nothing can be
  // deduced behind one.
  template <typename... Es>
    requires are_outputs<Es...>
  [[nodiscard]] static constexpr Equation
  create(Cache memo, std::unique_ptr<rt::Builder<T>> owned,
         rt::RTExpression<T> first, Es... rest) {
    const std::array outputs{first, rest...};
    return Base::create(std::move(memo), std::move(owned), outputs);
  }
  template <typename... Es>
    requires are_outputs<Es...> && std::floating_point<T>
  [[nodiscard]] static Equation cached(Cache memo,
                                       const std::filesystem::path &path,
                                       std::unique_ptr<rt::Builder<T>> owned,
                                       rt::RTExpression<T> first, Es... rest) {
    const std::array outputs{first, rest...};
    return Base::cached(std::move(memo), path, std::move(owned), outputs);
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
template <impl::Numeric T = double, CValueCache<T> C = detail::NoCache<T>>
using System = impl::Equation<RTExpression<T>, detail::Dynamic, C>;

// Over expressions already built in a caller's own arena, which the equation
// copies: the caller may go on building in it.  Partial
// specialisations contribute no deduction guides, so this is the whole of CTAD.
//
// A cache leads here and trails everywhere else: nothing can be deduced behind
// a pack, and the functions are the pack.
template <impl::Numeric T, typename... Ts>
  requires(std::same_as<Ts, RTExpression<T>> && ...)
[[nodiscard]] constexpr auto equation(RTExpression<T> first, Ts... rest) {
  return equation(detail::NoCache<T>{}, first, rest...);
}

template <typename C, impl::Numeric T, typename... Ts>
  requires(CValueCache<C, T> && (std::same_as<Ts, RTExpression<T>> && ...))
[[nodiscard]] constexpr auto equation(C memo, RTExpression<T> first,
                                      Ts... rest) {
  return impl::Equation<RTExpression<T>, Ts..., C>::create(
      std::move(memo), nullptr, first, rest...);
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
template <impl::Numeric T, typename Cache, typename Built>
[[nodiscard]] auto over(const Built &built, auto &&make) {
  if constexpr (std::same_as<Built, RTExpression<T>>) {
    return make(std::type_identity<impl::Equation<RTExpression<T>, Cache>>{},
                built);
  } else if constexpr (requires { std::tuple_size<Built>::value; }) {
    constexpr std::size_t outputs = std::tuple_size_v<Built>;
    static_assert(outputs > 0,
                  "equation: a system needs at least one function");
    return impl::index_apply<outputs - 1>([&]<std::size_t... Rest>() {
      return make(
          std::type_identity<
              impl::Equation<RTExpression<T>, Repeat<Rest, T>..., Cache>>{},
          built[0], built[Rest + 1]...);
    });
  } else {
    const auto outputs = built | impl::to<std::vector<RTExpression<T>>>();
    return make(std::type_identity<System<T, Cache>>{},
                std::span<const RTExpression<T>>{outputs});
  }
}

} // namespace detail

template <impl::Numeric T = double, CValueCache<T> C = detail::NoCache<T>>
[[nodiscard]] auto equation(std::invocable auto &&assemble, C memo = {}) {
  auto [arena, built] = detail::assemble<T>(DDX_FWD(assemble));
  return detail::over<T, C>(
      built, [&]<typename Eq>(std::type_identity<Eq>, auto... roots) {
        return Eq::create(std::move(memo), std::move(arena), roots...);
      });
}

// An equation from a file and nothing else: symbols, arena, Jacobian and
// colouring all come off disk, and no sweep runs.  The output count is part of
// the type, so a file holding some other number is refused, not adapted to --
// or `dynamic`, a System, which takes whatever the file holds.  An extern
// function the file names is bound from `fns` by name.
//
//   const auto eq = ddx::rt::load("f.ddx");        // one function
//   const auto eq = ddx::rt::load<double, 3>("f.ddx");
//   const auto sys = ddx::rt::load<double, ddx::rt::dynamic>("s.ddx");
inline constexpr std::size_t dynamic = 0;
template <impl::Numeric T = double, std::size_t Outputs = 1,
          CValueCache<T> C = detail::NoCache<T>>
  requires std::floating_point<T>
[[nodiscard]] auto load(const std::filesystem::path &path, C memo = {},
                        const ExternFns<T> &fns = {}) {
  if constexpr (Outputs == dynamic) {
    return System<T, C>::load(path, std::move(memo), fns);
  } else {
    return impl::index_apply<Outputs - 1>([&]<std::size_t... Rest>() {
      return impl::Equation<RTExpression<T>, detail::Repeat<Rest, T>...,
                            C>::load(path, std::move(memo), fns);
    });
  }
}

// The same model, with a file to keep the work in:
//
//   const auto eq = ddx::rt::equation("f.ddx", [] {
//     const auto x = var("x");
//     return exp(x) * x;
//   });
//
// The first run builds, sweeps and writes; a later one rebuilds the arena --
// interning, which is cheap -- and takes the sweeps and the colouring off disk.
// Editing the model changes the key, so the file is rebuilt, not trusted.  It
// never refuses; eq.loaded() says which happened.
template <impl::Numeric T = double, CValueCache<T> C = detail::NoCache<T>>
  requires std::floating_point<T>
[[nodiscard]] auto equation(const std::filesystem::path &path,
                            std::invocable auto &&assemble, C memo = {}) {
  auto [arena, built] = detail::assemble<T>(DDX_FWD(assemble));
  return detail::over<T, C>(
      built, [&]<typename Eq>(std::type_identity<Eq>, auto... roots) {
        return Eq::cached(std::move(memo), path, std::move(arena), roots...);
      });
}

} // namespace ddx::rt
