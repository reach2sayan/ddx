// The Hessian column colouring.  Here rather than in rt/coupling.hpp because it
// names no `T`: a header would emit one copy of the Boost.Graph machinery per
// scalar, on every consumer's include path.

#include "rt/coupling.hpp"

#include <boost/graph/compressed_sparse_row_graph.hpp>
// First, and not sortable into the block below: smallest_last_ordering.hpp
// uses make_shared_array_property_map without including it, as of Boost 1.92.
#include <boost/property_map/shared_array_property_map.hpp>

#include <boost/graph/sequential_vertex_coloring.hpp>
#include <boost/graph/smallest_last_ordering.hpp>

#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

namespace ddx::rt {
namespace {

// The conflict relation is symmetric and CSR has no undirected flavour, so
// both directions are inserted; the ordering and the colouring read only
// out_degree and adjacent_vertices, which then answer as the undirected list
// did -- ascending, which keeps the colours identical.  One-shot from an edge
// range, as graph.hpp already builds its.
using ConflictGraph = boost::compressed_sparse_row_graph<boost::directedS>;

} // namespace

Colouring colour_columns(const CouplingRows &rows) {
  if (rows.empty()) {
    return {};
  }
  const std::size_t n = rows.size();

  // Every pair of columns a row couples conflicts.
  const auto conflicting = [&](const std::size_t j) {
    return std::views::iota(j + 1, n) |
           std::views::filter([&, j](const std::size_t k) {
             return rows[j].intersects(rows[k]);
           }) |
           std::views::transform(
               [j](const std::size_t k) { return std::pair{j, k}; });
  };
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (const auto [j, k] : std::views::iota(0uz, n) |
                               std::views::transform(conflicting) |
                               std::views::join) {
    edges.emplace_back(j, k);
    edges.emplace_back(k, j);
  }
  const ConflictGraph conflicts{boost::edges_are_unsorted_multi_pass,
                                edges.begin(), edges.end(), n};

  // Greedy colouring is only as good as the order it walks.
  Colouring out{.colour = std::vector<std::size_t>(n, 0), .scatter = {}, .cell = {}};
  const auto order = boost::smallest_last_vertex_ordering(conflicts);
  // clang-format off
  const auto count = static_cast<std::size_t>(
    boost::sequential_vertex_coloring(
      conflicts,
      boost::make_iterator_property_map(order.begin(), boost::identity_property_map()),
      boost::make_iterator_property_map(out.colour.begin(), boost::get(boost::vertex_index, conflicts))));
  // clang-format on

  // scatter[colour][row] is the column that row's sweep result belongs to; the
  // colouring guarantees at most one, so the harvest needs no search.
  out.scatter.assign(count * n, no_column);
  const auto scatter = by_colour(out.scatter, count, n);
  for (const auto [index, row] : rows | std::views::enumerate) {
    const auto j = static_cast<std::size_t>(index);
    for (const std::size_t i : set_bits(row)) {
      scatter[out.colour[j], i] = j;
    }
  }

  // The owned cells numbered in (colour, row) order, so the compressed block
  // stays deterministic.
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
