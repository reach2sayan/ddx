#pragma once

#include "rt/archive/codec.hpp"
#include "rt/archive/snapshot.hpp" // the described member list this walks
#include "rt/graph.hpp"

#include <boost/describe.hpp>
#include <boost/hash2/fnv1a.hpp>
#include <boost/hash2/hash_append.hpp>
#include <boost/mp11/algorithm.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>

// Two keys over a graph: what a file is stale against, and what a compiled
// lane was compiled from.
namespace ddx::rt {

namespace detail {
// to_bits, never the member itself: hash2 appends a float as
// bit_cast<uint64_t>(v + 0), which maps -0.0 onto +0.0.  These digests key the
// object cache, and a graph holding -0.0 must not load +0.0's kernel.  A
// described type folds member by member, so its field list is its coverage.
constexpr void fold(boost::hash2::fnv1a_64 &h, const auto &v) {
  using U = std::remove_cvref_t<decltype(v)>;
  if constexpr (boost::describe::has_describe_members<U>::value) {
    boost::mp11::mp_for_each<
        boost::describe::describe_members<U, boost::describe::mod_any_access>>(
        [&](auto D) { fold(h, v.*D.pointer); });
  } else {
    boost::hash2::hash_append(h, wire_flavor{}, to_bits(v));
  }
}
} // namespace detail

// The model as the file keys on it: every input's name, kind by kind, then the
// arena up to `upto`.  Field by field, never a memcpy -- Node<double> has
// interior padding.
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(std::span<const std::string> symbols,
                                   std::span<const std::string> params,
                                   std::span<const std::string> implicits,
                                   std::span<const NodeId> closures,
                                   std::span<const Node<T>> nodes,
                                   std::size_t upto) {
  boost::hash2::fnv1a_64 h;
  // Length-framed by hash_append itself, for each list and each name.
  for (const auto names : {symbols, params, implicits}) {
    boost::hash2::hash_append(h, detail::wire_flavor{}, names);
  }
  std::ranges::for_each(closures, [&h](NodeId f) { detail::fold(h, f); });
  detail::fold(h, upto);
  for (const auto &n : nodes.first(std::min(upto, nodes.size()))) {
    boost::mp11::mp_for_each<boost::describe::describe_members<
        Node<T>, boost::describe::mod_public>>(
        [&](auto D) { detail::fold(h, n.*D.pointer); });
  }
  return h.result();
}

// The same key off a saved snapshot and off a live arena.
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(const Snapshot<T> &s) {
  return digest<T>(s.symbols, s.params, s.implicits, s.closures, s.nodes,
                   s.model_nodes);
}
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(const Builder<T> &b, std::size_t upto) {
  return digest<T>(b.symbols(), b.params(), b.implicits(), b.closures(),
                   b.nodes(), upto);
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
  for (const std::size_t count : {in.vars, in.params, in.implicits, in.seeds}) {
    detail::fold(h, count);
  }
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
