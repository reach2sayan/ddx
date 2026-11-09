#pragma once

// Hand-made graphs for tests that freeze or compile one directly, for example
// GraphBuilder{b}.value(f).build_jacobian().finish().

#include "rt/derivative.hpp"
#include "rt/graph.hpp"

#include <initializer_list>
#include <span>
#include <vector>

namespace ddx::test {

template <impl::Numeric T = double> class GraphBuilder {
public:
  explicit constexpr GraphBuilder(rt::Arena<T> &b) noexcept : arena_{&b} {}

  constexpr GraphBuilder &value(const rt::Expression<T> &root) {
    return values({root});
  }

  constexpr GraphBuilder &
  values(std::initializer_list<rt::Expression<T>> roots) {
    return values_from(roots | std::views::transform([&](const auto &e) {
                         return e.id(*arena_);
                       }) |
                       impl::to<std::vector>());
  }

  // Takes the functions to differentiate only, so the graph has no value
  // block.
  constexpr GraphBuilder &values_from(std::span<const rt::NodeId> roots,
                                      bool output = true) {
    roots_.assign(roots.begin(), roots.end());
    blocks_.values = output ? roots_ : std::vector<rt::NodeId>{};
    return *this;
  }

  constexpr GraphBuilder &build_jacobian() {
    const rt::Jacobian j = rt::derive_jacobian(*arena_, roots_);
    jacobian_pattern_ = j.pattern;
    blocks_.jacobian = j.partial;
    return *this;
  }

  GraphBuilder &build_hessian() {
    const rt::Hessian h = rt::derive_hessian(*arena_, roots_.front());
    colouring_ = h.colouring;
    blocks_.hessian = h.compressed;
    return *this;
  }

  [[nodiscard]] rt::Graph<T> finish() const {
    return rt::Graph<T>::freeze(*arena_, blocks_, colouring_,
                                jacobian_pattern_);
  }

private:
  rt::Arena<T> *arena_;
  std::vector<rt::NodeId> roots_;
  rt::OutputBlocks blocks_;
  rt::Colouring colouring_;
  rt::Sparsity jacobian_pattern_;
};

template <impl::Numeric T> GraphBuilder(rt::Arena<T> &) -> GraphBuilder<T>;

// Builds a graph in which every output is a value.
template <impl::Numeric T>
[[nodiscard]] rt::Graph<T> freeze_values(const rt::Arena<T> &b,
                                         std::span<const rt::NodeId> values,
                                         bool contract = true) {
  rt::OutputBlocks blocks;
  blocks.values.assign(values.begin(), values.end());
  return rt::Graph<T>::freeze(b, std::move(blocks), {}, {}, contract);
}

} // namespace ddx::test
