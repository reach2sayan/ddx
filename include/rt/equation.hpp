#pragma once

#include "rt/core.hpp"
#include "symbolic/equation.hpp"
#include "util/meta.hpp"

#include <concepts>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ddx::rt::detail {

// True when the pack is exactly Dynamic, so the output count is not fixed.
template <typename T, typename... Rest>
inline constexpr bool dynamic_outputs =
    sizeof...(Rest) == 1 && (std::same_as<Rest, Dynamic> && ...);

// The output-count policy: Fixed by the pack size, or Dynamic.
template <typename T, typename... Rest>
using policy_of = std::conditional_t<dynamic_outputs<T, Rest...>, Dynamic,
                                     Fixed<1 + sizeof...(Rest)>>;

// A friend of Equation, so that the equation() factories here and in rt/text
// are the only callers of create().
struct Creating {
  template <typename Eq>
  [[nodiscard]] static constexpr Eq create(auto &&...args) {
    return Eq::create(DDX_FWD(args)...);
  }
};

} // namespace ddx::rt::detail

// Specialised on the Expression<T> pattern, since a requires clause would be
// ambiguous with the compile-time specialisation.
namespace ddx::impl {

// A runtime equation with its output count in the type, or with Dynamic a
// count known only at run time, where single-output members return an error.
template <Numeric T, typename... Rest>
  requires(rt::detail::dynamic_outputs<T, Rest...> ||
           (std::same_as<Rest, rt::Expression<T>> && ...))
class Equation<rt::Expression<T>, Rest...> : public rt::detail::Core<T> {
  using Outputs = rt::detail::policy_of<T, Rest...>;
  using Base = rt::detail::Core<T>;
  friend Base;
  friend struct rt::detail::Creating;

  // A null owned arena means the outputs live in the caller's arena, which is
  // then copied. Build failures are carried by the equation; see poisoned().
  [[nodiscard]] static constexpr Equation
  create(std::unique_ptr<rt::Arena<T>> owned,
         std::span<const rt::Expression<T>> outputs) {
    return Base::template make<Equation>(outputs, std::move(owned));
  }

  template <typename... Es>
  static constexpr bool are_outputs =
      Outputs::expected == sizeof...(Es) + 1 &&
      (std::same_as<Es, rt::Expression<T>> && ...);

  template <typename... Es>
    requires are_outputs<Es...>
  [[nodiscard]] static constexpr Equation
  create(std::unique_ptr<rt::Arena<T>> owned, rt::Expression<T> first,
         Es... rest) {
    const std::array outputs{first, rest...};
    return create(std::move(owned),
                  std::span<const rt::Expression<T>>{outputs});
  }

public:
  // The output count fixed by the type, or 0 for a System.
  static constexpr std::size_t output_dim = Outputs::expected.value_or(0);

  // Returns a single T for one output and a vector otherwise.
  [[nodiscard]] constexpr auto
  evaluate(const rt::detail::CPointArg auto &...args) const {
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

// A single-output equation, as returned by equation() for one expression.
template <impl::Numeric T = double>
using Equation = impl::Equation<Expression<T>>;

template <impl::Numeric T = double>
using System = impl::Equation<Expression<T>, detail::Dynamic>;

// Builds an equation from expressions in the caller's arena, which is copied
// so the caller may keep building in it.
template <impl::Numeric T, typename... Ts>
  requires(std::same_as<Ts, Expression<T>> && ...)
[[nodiscard]] constexpr auto equation(Expression<T> first, Ts... rest) {
  return detail::Creating::create<impl::Equation<Expression<T>, Ts...>>(
      nullptr, first, rest...);
}

namespace detail {

// Picks the Equation type from what the callback built: one root, a fixed-size
// tuple-like of roots, or any other range, which becomes a System.
template <impl::Numeric T, typename Built>
[[nodiscard]] auto over(const Built &built, auto &&make) {
  if constexpr (std::same_as<Built, Expression<T>>) {
    return make(std::type_identity<impl::Equation<Expression<T>>>{}, built);
  } else if constexpr (requires { std::tuple_size<Built>::value; }) {
    constexpr std::size_t outputs = std::tuple_size_v<Built>;
    static_assert(outputs > 0,
                  "equation: a system needs at least one function");
    return meta::unpack(meta::iota<outputs>, [&](auto... i) {
      return make(std::type_identity<
                      meta::repeat_t<impl::Equation, Expression<T>, outputs>>{},
                  built[i]...);
    });
  } else {
    const auto outputs = built | impl::to<std::vector<Expression<T>>>();
    return make(std::type_identity<System<T>>{},
                std::span<const Expression<T>>{outputs});
  }
}

} // namespace detail

// Builds an equation from a callback such as [] { return exp(var("x")); }.
// An array of roots gives a fixed-size system and any other range a System.
template <impl::Numeric T = double>
[[nodiscard]] auto equation(std::invocable auto &&assemble) {
  auto [arena, built] = detail::assemble<T>(DDX_FWD(assemble));
  return detail::over<T>(
      built, [&]<typename Eq>(std::type_identity<Eq>, auto... roots) {
        return detail::Creating::create<Eq>(std::move(arena), roots...);
      });
}

} // namespace ddx::rt
