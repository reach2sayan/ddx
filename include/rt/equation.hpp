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
  // The output-count policy: Fixed by the pack size, or Dynamic.
  using Outputs = std::conditional_t<rt::detail::dynamic_outputs<T, Rest...>,
                                     rt::detail::Dynamic,
                                     rt::detail::Fixed<1 + sizeof...(Rest)>>;
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
    requires(Outputs::expected == sizeof...(Es) + 1 &&
             (std::same_as<Es, rt::Expression<T>> && ...))
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

// Builds an equation from a callback such as [] { return exp(var("x")); }.
// One root gives an Equation, a fixed-size tuple-like of roots a fixed-size
// system, and any other range a System.
template <impl::Numeric T = double>
[[nodiscard]] auto equation(std::invocable auto &&assemble) {
  auto [arena, built] = detail::assemble<T>(DDX_FWD(assemble));
  using Built = std::remove_cvref_t<decltype(built)>;
  using detail::Creating;
  if constexpr (std::same_as<Built, Expression<T>>) {
    return Creating::create<Equation<T>>(std::move(arena), built);
  } else if constexpr (meta::CTupleLike<Built>) {
    constexpr std::size_t outputs = std::tuple_size_v<Built>;
    static_assert(outputs > 0,
                  "equation: a system needs at least one function");
    return meta::unpack(meta::iota<outputs>, [&](auto... i) {
      return Creating::create<
          meta::repeat_t<impl::Equation, Expression<T>, outputs>>(
          std::move(arena), built[i]...);
    });
  } else {
    const auto outputs = built | impl::to<std::vector<Expression<T>>>();
    return Creating::create<System<T>>(std::move(arena),
                                       std::span<const Expression<T>>{outputs});
  }
}

} // namespace ddx::rt
