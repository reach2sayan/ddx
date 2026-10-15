#pragma once

#include "rt/archive/codec.hpp"
#include "rt/archive/snapshot.hpp" // the described member list this walks
#include "rt/graph.hpp"

#include <boost/describe.hpp>
#include <boost/hash2/fnv1a.hpp>
#include <boost/hash2/hash_append.hpp>
#include <boost/mp11/algorithm.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

// Two keys over a graph: what a file is stale against, and what a compiled
// lane was compiled from.
namespace ddx::rt {

namespace detail {
// to_bits, never the member itself: hash2 appends a float as
// bit_cast<uint64_t>(v + 0), which maps -0.0 onto +0.0.  These digests key the
// object cache, and a graph holding -0.0 must not load +0.0's kernel.  A
// described type folds member by member, so its field list is its coverage; a
// range its length then its elements, field by field, never a memcpy --
// Node<double> has interior padding.
constexpr void fold(boost::hash2::fnv1a_64 &h, const auto &v) {
  using U = std::remove_cvref_t<decltype(v)>;
  if constexpr (std::same_as<U, std::string>) {
    boost::hash2::hash_append(h, wire_flavor{}, v);
  } else if constexpr (boost::describe::has_describe_members<U>::value) {
    boost::mp11::mp_for_each<described_members<U>>(
        [&](auto D) { fold(h, v.*D.pointer); });
  } else if constexpr (std::ranges::range<U>) {
    boost::hash2::hash_append(h, wire_flavor{},
                              static_cast<std::uint64_t>(std::ranges::size(v)));
    std::ranges::for_each(v, [&h](const auto &e) { fold(h, e); });
  } else {
    boost::hash2::hash_append(h, wire_flavor{}, to_bits(v));
  }
}
} // namespace detail

// The model as a file keys on it: every described field, with the arena cut
// at `upto`.  Off the one list, so a field added to Model is keyed on.
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(const Model<T> &m, std::size_t upto) {
  boost::hash2::fnv1a_64 h;
  boost::mp11::mp_for_each<detail::described_members<Model<T>>>([&](auto D) {
    if constexpr (std::string_view{decltype(D)::name} == "nodes") {
      detail::fold(h, upto);
      detail::fold(h, std::span{m.nodes}.first(std::min(upto, m.nodes.size())));
    } else {
      detail::fold(h, m.*D.pointer);
    }
  });
  return h.result();
}

// The same key off a saved snapshot and off a live arena.
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(const Snapshot<T> &s) {
  return digest(s.model, s.model_nodes);
}
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(const Builder<T> &b, std::size_t upto) {
  return digest(b.model(), upto);
}

// Keyed on what codegen reads of a lane, so a consumer sees a compile is done
// *before* emitting a module: the layout and every node's property, member by
// member off their described lists, and every operand.  Ids are
// construction-ordered, making this a within-one-binary key, so a stale one
// must be a miss.
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(const Graph<T> &g) {
  boost::hash2::fnv1a_64 h;
  const Inputs &in = g.inputs();
  for (const std::size_t count : in.by_kind) {
    detail::fold(h, count);
  }
  detail::fold(h, in.seeds);
  detail::fold(h, g.layout());
  const auto schedule = g.schedule();
  detail::fold(h, schedule.size());
  for (const auto &[v, fma] : schedule) {
    detail::fold(h, v);
    detail::fold(h, g[v]);
    std::ranges::for_each(g.operands(v),
                          [&h](NodeId u) { detail::fold(h, u); });
  }
  std::ranges::for_each(g.outputs(), [&h](NodeId o) { detail::fold(h, o); });
  return h.result();
}

} // namespace ddx::rt
