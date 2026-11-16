#pragma once

#include "rt/arena.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace ddx::rt::detail {

// Each single-use chain of a binary op that has an n-ary form becomes one
// node over the chain's terms in its order, so it rounds as the chain does.
// Only the running operand is followed, since a + (b + c) would reassociate;
// a commutative op may hold it on either side. Implicit closures are roots
// too, and are rewritten with them.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
flatten(Arena<T> &b, std::span<const NodeId> roots) {
  std::vector<NodeId> &closures = b.model_.closures;
  const auto size = static_cast<NodeId>(b.size());
  auto tops = roots | impl::to<std::vector>();
  std::ranges::copy_if(closures, std::back_inserter(tops),
                       [](NodeId f) { return f != no_node; });
  const std::vector live = detail::reachable(b, tops);

  std::vector uses(size, std::uint32_t{0});
  for (const NodeId v : detail::live_ids(live)) {
    detail::for_each_operand(b, v, [&uses](NodeId u) { ++uses[u]; });
  }
  std::vector is_root(size, false);
  for (const NodeId r : tops) {
    is_root[r] = true;
  }

  // The operand slot the chain through v continues in, if any.
  const auto running = [&](NodeId v) -> std::optional<std::size_t> {
    const OpCode step = b.op_of(v);
    const auto in = b.operands(v);
    const auto links = [&](NodeId u) {
      return b.op_of(u) == step && uses[u] == 1 && !is_root[u];
    };
    if (links(in[0])) {
      return 0;
    }
    if (is_commutative<T>(step) && links(in[1])) {
      return 1;
    }
    return std::nullopt;
  };
  const auto chains = [&](NodeId v) {
    return nary_of(b.op_of(v)).has_value() && running(v).has_value();
  };
  // A product's last step stays a Mul, the product an add contracts into an
  // fma; the chain below it is flattened.
  std::vector below_product(size, false);
  const auto keeps_step = [&](NodeId v) {
    return b.op_of(v) == OpCode::Mul && !below_product[v];
  };

  // A chain's interior is read by its top alone, so it is never rebuilt.
  std::vector absorbed(size, false);
  for (const NodeId v : detail::live_ids(live) | std::views::reverse) {
    if (absorbed[v] || !chains(v)) {
      continue;
    }
    if (keeps_step(v)) {
      below_product[b.operands(v)[*running(v)]] = true;
      continue;
    }
    for (NodeId u = v; const auto side = running(u);) {
      u = b.operands(u)[*side];
      absorbed[u] = true;
    }
  }

  std::vector remap(size, no_node);
  const auto mapped = [&remap](NodeId u) { return remap[u]; };
  for (const NodeId v : detail::live_ids(live)) {
    if (absorbed[v]) {
      continue;
    }
    if (is_leaf(b.op_of(v))) {
      remap[v] = v;
      continue;
    }
    if (!chains(v) || keeps_step(v)) {
      remap[v] = b.rebuild(v, mapped);
      continue;
    }
    // Last term first, down to the chain's first two.
    std::vector<NodeId> terms;
    NodeId u = v;
    for (auto side = running(u); side; side = running(u)) {
      terms.push_back(remap[b.operands(u)[1 - *side]]);
      u = b.operands(u)[*side];
    }
    const auto first = b.operands(u);
    terms.push_back(remap[first[1]]);
    terms.push_back(remap[first[0]]);
    std::ranges::reverse(terms);
    remap[v] = b.folded(*nary_of(b.op_of(v)), terms);
  }

  for (NodeId &f : closures) {
    f = f == no_node ? f : remap[f];
  }
  return roots | std::views::transform(mapped) | impl::to<std::vector>();
}

// Blocked summation of long sums, which a kernel otherwise pays as latency.
// Only the kernel graph uses it, so it may differ by a few ULP.

// Sums terms into `blocks` interleaved partial sums, then combines pairwise.
template <impl::Numeric T>
[[nodiscard]] constexpr NodeId
blocked_sum(Arena<T> &b, std::span<const NodeId> terms, std::size_t blocks) {
  // Sized to the default block count so a round never allocates.
  using Partials = boost::container::small_vector<Expression<T>, 16>;
  const auto as_expr = [&b](NodeId t) { return Expression{b, t}; };
  Partials partials;
  for (std::size_t j = 0; j < std::min(blocks, terms.size()); ++j) {
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
rebalance(Arena<T> &b, std::span<const NodeId> roots) {
  // Sums of at least `least` terms are split into `blocks` partial sums.
  constexpr std::size_t blocks = 16;
  constexpr std::size_t least = 16;
  const std::vector live = detail::reachable(b, roots);

  std::vector remap(b.size(), no_node);
  const auto mapped = [&remap](NodeId u) { return remap[u]; };
  for (const NodeId v : detail::live_ids(live)) {
    if (is_leaf(b.op_of(v))) {
      remap[v] = v;
    } else if (b.op_of(v) == OpCode::SumN &&
               arguments(b, v).size() >= least) {
      const auto terms = arguments(b, v) | std::views::transform(mapped) |
                         impl::to<std::vector>();
      remap[v] = detail::blocked_sum(b, terms, blocks);
    } else {
      remap[v] = b.rebuild(v, mapped);
    }
  }

  return roots | std::views::transform(mapped) | impl::to<std::vector>();
}

} // namespace ddx::rt::detail
