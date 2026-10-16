#pragma once

#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <boost/dynamic_bitset.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/range/iterator_range_core.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stack>
#include <utility>
#include <vector>

namespace ddx::rt {

// Which steps a moved input column reaches.  Vertices are positions in the
// schedule rather than node ids, so the walk hands its answer straight to the
// loop that runs the steps.  NOTES.md, "Remembering the last call", says why
// the schedule is read upward instead of propagated forward.
class Cone {
public:
  using Stack = std::stack<std::uint32_t, std::vector<std::uint32_t>>;

  Cone() = default;

  template <impl::Numeric T>
  Cone(const Builder<T> &b, std::span<const Step> schedule,
       std::size_t columns) {
    const Inputs inputs = b.inputs();
    steps_ = schedule.size();

    std::vector<std::uint32_t> at(b.size(), unscheduled);
    for (const auto [i, s] : schedule | std::views::enumerate) {
      at[s.node] = static_cast<std::uint32_t>(i);
    }

    std::vector<std::pair<std::size_t, std::size_t>> edges;
    edges.reserve(schedule.size() * 3);
    seeds_.assign(columns + 1, 0);
    for (const auto &[i, s] : schedule | std::views::enumerate) {
      const Node<T> &n = b[s.node];
      if (const auto column = read_by(n, s, inputs)) {
        ++seeds_[*column + 1];
        continue;
      }
      for (const NodeId u : operands_of(n, s)) {
        if (const std::uint32_t from = at[u]; from != unscheduled) {
          edges.emplace_back(from, static_cast<std::size_t>(i));
        }
      }
    }
    readers_ = Readers{boost::edges_are_unsorted_multi_pass, edges.begin(),
                       edges.end(), schedule.size()};

    // One flat run per column, so a walk's seeds are a subrange rather than a
    // vector of vectors chased per call.
    std::inclusive_scan(seeds_.begin(), seeds_.end(), seeds_.begin());
    seeded_.resize(seeds_.back());
    std::vector<std::uint32_t> cursor = seeds_;
    for (const auto &[i, s] : schedule | std::views::enumerate) {
      if (const auto column = read_by(b[s.node], s, inputs)) {
        seeded_[cursor[*column]++] = static_cast<std::uint32_t>(i);
      }
    }
  }

  [[nodiscard]] std::size_t steps() const noexcept { return steps_; }

  // The steps `changed` reaches, set in `dirty`.  Both scratches are the
  // caller's, so a walk allocates nothing.
  void reach(std::span<const bool> changed, boost::dynamic_bitset<> &dirty,
             Stack &pending) const {
    dirty.resize(steps_);
    dirty.reset();
    for (const auto [column, moved] : changed | std::views::enumerate) {
      if (!moved) {
        continue;
      }
      for (const std::uint32_t step :
           seeds_of(static_cast<std::size_t>(column))) {
        if (!dirty.test_set(step)) {
          pending.push(step);
        }
      }
    }
    while (!pending.empty()) {
      const auto step = static_cast<Reader>(pending.top());
      pending.pop();
      for (const auto &edge :
           boost::make_iterator_range(boost::out_edges(step, readers_))) {
        const auto reader =
            static_cast<std::uint32_t>(boost::target(edge, readers_));
        if (!dirty.test_set(reader)) {
          pending.push(reader);
        }
      }
    }
  }

  [[nodiscard]] std::span<const std::uint32_t>
  seeds_of(std::size_t column) const {
    return std::span{seeded_}.subspan(seeds_[column],
                                      seeds_[column + 1] - seeds_[column]);
  }

private:
  using Readers = boost::compressed_sparse_row_graph<boost::directedS>;
  using Reader = boost::graph_traits<Readers>::vertex_descriptor;

  static constexpr std::uint32_t unscheduled =
      std::numeric_limits<std::uint32_t>::max();

  // The column a step reads its value from, where it reads one at all.  A
  // literal is a leaf that no point moves.
  template <impl::Numeric T>
  [[nodiscard]] static constexpr std::optional<std::size_t>
  read_by(const Node<T> &n, const Step &s, const Inputs &inputs) {
    if (s.fma || kind_of(n.op) != Kind::Input) {
      return std::nullopt;
    }
    return input_column(inputs, n.op, n.slot);
  }

  // An absent operand is no_node; a contraction's three are all real.
  template <impl::Numeric T>
  [[nodiscard]] static constexpr auto operands_of(const Node<T> &n,
                                                  const Step &s) {
    return (s.fma ? std::array{s.fma.x, s.fma.y, s.fma.z}
                  : std::array{n.a, n.b, n.c}) |
           std::views::filter([](NodeId v) { return v != no_node; });
  }

  Readers readers_;
  std::vector<std::uint32_t> seeded_;
  std::vector<std::uint32_t> seeds_;
  std::size_t steps_ = 0;
};

// The steps a moved point reaches, over a tape that still holds the last
// point's values: what the move does not reach keeps the bits it had, which is
// what makes this equal a cold sweep exactly rather than nearly.
template <impl::Numeric T, std::ranges::random_access_range R, impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>>
void evaluate_reached(const Builder<T> &b, const R &point,
                      std::span<const Step> schedule, const Cone &cone,
                      std::span<const bool> changed,
                      boost::dynamic_bitset<> &dirty, Cone::Stack &pending,
                      std::span<U> tape) {
  cone.reach(changed, dirty, pending);
  const auto lane = detail::lanes_of<1>(tape);
  const auto nodes = b.nodes();
  for (auto step = dirty.find_first(); step != boost::dynamic_bitset<>::npos;
       step = dirty.find_next(step)) {
    detail::sweep_step<1>(b, nodes, schedule[step], std::ranges::begin(point),
                          lane);
  }
}

} // namespace ddx::rt
