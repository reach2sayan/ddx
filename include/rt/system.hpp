#pragma once

#include "rt/core.hpp"

#include <filesystem>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

namespace ddx::rt {

// An equation whose output count is a run-time number: one output per species
// is a count no template parameter knows.  Everything Equation has, with every
// value block a vector and the one-output members refusing a system with
// errc::not_single_output.
//
//   const auto sys = ddx::rt::system([&] {
//     std::vector<ddx::rt::RTExpression<>> out;
//     for (...) out.push_back(...);
//     return out;
//   });
template <impl::Numeric T = double, CValueCache<T> Cache = detail::NoCache<T>>
class System : public detail::Core<T, Cache> {
  using Base = detail::Core<T, Cache>;
  friend Base;

public:
  // Over expressions already built in a caller's own arena, which is copied.
  [[nodiscard]] static constexpr System
  create(std::span<const RTExpression<T>> outputs, Cache memo = {}) {
    return Base::template make<System>(std::move(memo), outputs, nullptr);
  }

  // `assemble` returns any range of expressions; the count is however many it
  // holds, and none is errc::no_graph.
  template <std::invocable F>
    requires std::ranges::input_range<std::invoke_result_t<F>>
  [[nodiscard]] static System build(F &&assemble, Cache memo = {}) {
    auto arena = std::make_unique<Builder<T>>();
    const auto outputs = [&] {
      const auto scope = detail::scoped_arena(*arena);
      return std::invoke(DDX_FWD(assemble)) |
             impl::to<std::vector<RTExpression<T>>>();
    }();
    return Base::template make<System>(std::move(memo), outputs,
                                       std::move(arena));
  }

  // Nothing built and nothing swept: any output count the file holds.
  [[nodiscard]] static result<System> load(const std::filesystem::path &path,
                                           Cache memo = {})
    requires std::floating_point<T>
  {
    return Base::template load<System>(path, std::move(memo), std::nullopt);
  }

  // m long, always a vector.
  [[nodiscard]] constexpr result<std::vector<T>>
  evaluate(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return this->values(args...);
  }
  using Base::evaluate;

private:
  using Base::Base;
};

template <impl::Numeric T = double, CValueCache<T> C = detail::NoCache<T>>
[[nodiscard]] System<T, C> system(std::invocable auto &&assemble, C memo = {}) {
  return System<T, C>::build(DDX_FWD(assemble), std::move(memo));
}

template <impl::Numeric T = double, CValueCache<T> C = detail::NoCache<T>>
  requires std::floating_point<T>
[[nodiscard]] result<System<T, C>>
load_system(const std::filesystem::path &path, C memo = {}) {
  return System<T, C>::load(path, std::move(memo));
}

} // namespace ddx::rt
