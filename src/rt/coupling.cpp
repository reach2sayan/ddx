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

// Four stages, each feeding the next off shared state; a step per member keeps
// that state out of the parameter lists and the order in one place.
class ColumnColouring {
public:
  explicit ColumnColouring(const CouplingRows &rows)
      : rows_{rows}, n_{rows.size()},
        out_{.colour = std::vector<std::size_t>(n_, 0),
             .scatter = {},
             .cell = {}} {}

  [[nodiscard]] Colouring run() && {
    link_conflicts();
    pick_colours();
    map_scatter();
    number_cells();
    return std::move(out_);
  }

private:
  void link_conflicts() {
    const auto conflicting = [&](const std::size_t j) {
      return std::views::iota(j + 1, n_) |
             std::views::filter([&, j](const std::size_t k) {
               return rows_[j].intersects(rows_[k]);
             }) |
             std::views::transform(
                 [j](const std::size_t k) { return std::pair{j, k}; });
    };
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    for (const auto [j, k] : std::views::iota(0uz, n_) |
                                 std::views::transform(conflicting) |
                                 std::views::join) {
      edges.emplace_back(j, k);
      edges.emplace_back(k, j);
    }
    conflicts_ = ConflictGraph{boost::edges_are_unsorted_multi_pass,
                               edges.begin(), edges.end(), n_};
  }

  void pick_colours() {
    // Greedy colouring is only as good as the order it walks.
    const auto order = boost::smallest_last_vertex_ordering(conflicts_);
    // clang-format off
    count_ = static_cast<std::size_t>(
      boost::sequential_vertex_coloring(
        conflicts_,
        boost::make_iterator_property_map(order.begin(), boost::identity_property_map()),
        boost::make_iterator_property_map(out_.colour.begin(), boost::get(boost::vertex_index, conflicts_))));
    // clang-format on
  }

  // scatter[colour][row] is the column that row's sweep result belongs to; the
  // colouring guarantees at most one, so the harvest needs no search.
  void map_scatter() {
    out_.scatter.assign(count_ * n_, no_column);
    const auto scatter = by_colour(out_.scatter, count_, n_);
    for (const auto [index, row] : rows_ | std::views::enumerate) {
      const auto j = static_cast<std::size_t>(index);
      const std::size_t colour = out_.colour[j];
      for (const std::size_t i : set_bits(row)) {
        scatter[colour, i] = j;
      }
    }
  }

  // Number the owned cells in (colour, row) order, so the compressed block
  // stays deterministic.  Reads `scatter` directly rather than through the
  // finished Colouring: the tables are still half-built here.
  void number_cells() {
    out_.cell.assign(count_ * n_, no_column);
    const auto scatter = by_colour(out_.scatter, count_, n_);
    const auto cell = by_colour(out_.cell, count_, n_);
    std::size_t next = 0;
    for (const auto [c, i] : std::views::cartesian_product(
             std::views::iota(0uz, count_), std::views::iota(0uz, n_))) {
      if (scatter[c, i] != no_column) {
        cell[c, i] = next++;
      }
    }
  }

  const CouplingRows &rows_;
  std::size_t n_;
  ConflictGraph conflicts_;
  std::size_t count_ = 0;
  Colouring out_;
};

} // namespace

Colouring colour_columns(const CouplingRows &rows) {
  if (rows.empty()) {
    return {};
  }
  return ColumnColouring{rows}.run();
}

} // namespace ddx::rt
