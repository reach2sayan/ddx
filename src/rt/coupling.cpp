// The Hessian column colouring. It is not templated on T, so it lives here to
// keep Boost.Graph out of every consumer's includes.

#include "rt/coupling.hpp"

#include <boost/graph/compressed_sparse_row_graph.hpp>
// Must come first: smallest_last_ordering.hpp uses this header without
// including it, as of Boost 1.92.
#include <boost/property_map/shared_array_property_map.hpp>

#include <boost/graph/sequential_vertex_coloring.hpp>
#include <boost/graph/smallest_last_ordering.hpp>

#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

namespace ddx::rt {
namespace {

// CSR has no undirected form, so each conflict is inserted both ways. The
// adjacency stays ascending, which keeps the colouring deterministic.
using ConflictGraph = boost::compressed_sparse_row_graph<boost::directedS>;

} // namespace

Colouring colour_columns(const CouplingRows &rows) {
  if (rows.empty()) {
    return {};
  }
  const std::size_t n = rows.size();

  // Two columns conflict when any row couples both.
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = j + 1; k < n; ++k) {
      if (rows[j].intersects(rows[k])) {
        edges.emplace_back(j, k);
        edges.emplace_back(k, j);
      }
    }
  }
  const ConflictGraph conflicts{boost::edges_are_unsorted_multi_pass,
                                edges.begin(), edges.end(), n};

  // Greedy colouring quality depends on the vertex order, so use smallest-last.
  Colouring out{
      .colour = std::vector<std::size_t>(n, 0), .scatter = {}, .cell = {}};
  const auto order = boost::smallest_last_vertex_ordering(conflicts);
  // clang-format off
  const auto count = static_cast<std::size_t>(
    boost::sequential_vertex_coloring(
      conflicts,
      boost::make_iterator_property_map(order.begin(), boost::identity_property_map()),
      boost::make_iterator_property_map(out.colour.begin(), boost::get(boost::vertex_index, conflicts))));
  // clang-format on

  // scatter[colour][row] is the column that row's sweep result belongs to. The
  // colouring guarantees there is at most one.
  out.scatter.assign(count * n, no_column);
  const auto scatter = by_colour(out.scatter, count, n);
  for (const auto [index, row] : rows | std::views::enumerate) {
    const auto j = static_cast<std::size_t>(index);
    for (const std::size_t i : set_bits(row)) {
      scatter[out.colour[j], i] = j;
    }
  }

  // Number the owned cells in (colour, row) order, so that the compressed block
  // is deterministic.
  out.cell.assign(count * n, no_column);
  const auto cell = by_colour(out.cell, count, n);
  std::size_t next = 0;
  for (const auto [c, i] : std::views::cartesian_product(
           std::views::iota(0uz, count), std::views::iota(0uz, n))) {
    if (scatter[c, i] != no_column) {
      cell[c, i] = next++;
    }
  }
  return out;
}

} // namespace ddx::rt
