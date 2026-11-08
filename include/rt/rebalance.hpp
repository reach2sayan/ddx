#pragma once

#include "rt/arena.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <ranges>
#include <span>
#include <stack>
#include <vector>

namespace ddx::rt {

// Blocked summation of long Add chains, which a kernel otherwise pays as
// latency. Only the kernel graph uses it, so it may differ by a few ULP.

namespace detail {

// The terms of a maximal single-use Add spine, in left-to-right order.
template <impl::Numeric T>
[[nodiscard]] auto spine_terms(const Arena<T> &b, NodeId top,
                               std::span<const std::uint32_t> uses,
                               const std::vector<bool> &root) {
  // Roots end a spine even when used once, so they are never absorbed into it.
  const auto links = [&](NodeId u) {
    return b[u].op == OpCode::Add && uses[u] == 1 && !root[u];
  };
  boost::container::small_vector<NodeId, 64> terms;
  std::stack<NodeId, boost::container::small_vector<NodeId, 32>> stack;
  stack.push(top);
  while (!stack.empty()) {
    const NodeId u = stack.top();
    stack.pop();
    if (b[u].op != OpCode::Add || (u != top && !links(u))) {
      terms.push_back(u);
      continue;
    }
    const auto [a, c, absent] = b.operands(u); // Add is binary
    stack.push(c); // pushed first so that the left operand pops first
    stack.push(a);
  }
  return terms;
}

// Sums terms into `blocks` interleaved partial sums, then combines pairwise.
template <impl::Numeric T>
[[nodiscard]] constexpr NodeId
blocked_sum(Arena<T> &b, std::span<const NodeId> terms, std::size_t blocks) {
  // Sized to the default block count so a round never allocates.
  using Partials = boost::container::small_vector<Expression<T>, 16>;
  const auto as_expr = [&b](NodeId t) { return Expression<T>{b, t}; };
  Partials partials;
  for (const std::size_t j : std::views::iota(0uz, std::min(blocks, terms.size()))) {
    partials.push_back(*std::ranges::fold_left_first(
        terms | std::views::drop(j) | std::views::stride(blocks) |
            std::views::transform(as_expr),
        std::plus<>{}));
  }

  // Combine pairwise so the final reduction is not a chain either.
  while (partials.size() > 1) {
    partials =
        partials | std::views::chunk(2) | std::views::transform([](auto pair) {
          return std::ranges::fold_left_first(pair, std::plus<>{}).value();
        }) |
        impl::to<Partials>();
  }
  return partials.front().id(b);
}

// Returns the rewritten roots. Nodes are immutable, so the arena is rebuilt
// bottom-up and interning maps unchanged subtrees back to themselves.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
rebalance(Arena<T> &b, std::span<const NodeId> roots, std::size_t blocks = 16,
          std::size_t least = 16) {
  const auto size = static_cast<NodeId>(b.size());
  const std::vector live = detail::reachable(b, roots);

  std::vector<std::uint32_t> uses(size, 0);
  for (const NodeId v : detail::live_ids(live)) {
    detail::for_each_operand(b, v, [&uses](NodeId u) { ++uses[u]; });
  }
  std::vector is_root(size, false);
  for (const NodeId r : roots) {
    is_root[r] = true;
  }

  std::vector remap(size, no_node);
  for (const NodeId v : detail::live_ids(live)) {

    // A copy, since building may reallocate the arena.
    const Node node = b[v];
    if (is_leaf(node.op)) {
      remap[v] = v;
      continue;
    }
    // A SumN is an already flattened spine; an Add spine is collected from
    // its single-use interior.
    const bool spine_top =
        node.op == OpCode::SumN ||
        (node.op == OpCode::Add && (uses[v] != 1 || is_root[v]));
    if (spine_top) {
      auto terms =
          node.op == OpCode::SumN
              ? arguments(b, v) |
                    impl::to<boost::container::small_vector<NodeId, 64>>()
              : detail::spine_terms(b, v, uses, is_root);
      if (terms.size() >= least) {
        std::ranges::transform(terms, terms.begin(),
                               [&remap](NodeId t) { return remap[t]; });
        remap[v] = detail::blocked_sum(b, terms, blocks);
        continue;
      }
    }
    remap[v] = b.rebuild(v, [&remap](NodeId u) { return remap[u]; });
  }

  return roots |
         std::views::transform([&remap](NodeId r) { return remap[r]; }) |
         impl::to<std::vector<NodeId>>();
}

} // namespace detail
} // namespace ddx::rt
