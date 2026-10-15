#pragma once

#include "rt/graph.hpp"
#include "rt/opcode.hpp"
#include "util/export.hpp"
#include "util/fmt.hpp" // detail::fmt_put
#include "util/ranges.hpp"

#include <boost/dynamic_bitset.hpp>

#include <concepts>
#include <cstdint>
#include <format>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// The frozen graph as DOT, for looking at it:
//
//   std::cout << ddx::rt::Dot{g};          // or `... | dot -Tsvg`
//
// Its own header: boost/graph/graphviz.hpp carries the reader as well, and the
// writer lives in src/rt/dot.cpp, so graphviz reaches one TU only.
namespace ddx::rt {

// Live is what codegen emits; All draws what the freeze pruned, dashed.
enum class Scope : std::uint8_t { Live, All };

struct DotNode {
  std::string label;
  std::string_view shape; // always one of three literals
  bool live = true;
  bool show_slots = false;
};

[[nodiscard]] DDX_API std::string
to_dot(const Adjacency &adj, std::span<const DotNode> nodes, Scope scope);

// Borrows and renders on demand, as jit::Ir does; the deleted overload checks
// that the graph outlives the handle.  Edges point from a node to its operands
// and carry the operand slot where the two are not interchangeable.
template <impl::Numeric T = double> class Dot {
public:
  explicit Dot(const Graph<T> &g, Scope scope = Scope::Live) noexcept
      : graph_{g}, scope_{scope} {}
  Dot(const Graph<T> &&, Scope = Scope::Live) = delete;

  [[nodiscard]] std::string str() const {
    return to_dot(graph_.children(), describe(), scope_);
  }

private:
  [[nodiscard]] std::vector<DotNode> describe() const {
    boost::dynamic_bitset<> is_output(graph_.size());
    std::ranges::for_each(graph_.outputs(),
                          [&](NodeId o) { is_output.set(o); });

    return std::views::iota(NodeId{0}, static_cast<NodeId>(graph_.size())) |
           std::views::transform([&](NodeId v) {
             const auto op = graph_[v].op;
             return DotNode{.label = text(v),
                            .shape = is_output[v]  ? "doubleoctagon"
                                     : is_leaf(op) ? "box"
                                                   : "ellipse",
                            .live = graph_.live(v),
                            // Only where reading it back matters: nothing to
                            // tell apart on a commutative or unary node.
                            // `>= 2`, not `== 2`: a select's three operands
                            // are the one place the slot matters most, the
                            // condition reading nothing like its arms.
                            .show_slots =
                                arity_of(op) >= 2 && !is_commutative<T>(op)};
           }) |
           impl::to<std::vector<DotNode>>();
  }

  // `id: what it is` -- codegen, the interpreter and errors all name nodes by
  // the id.
  [[nodiscard]] std::string text(NodeId v) const {
    const auto &p = graph_[v];
    switch (p.op) {
    case OpCode::Var:
    case OpCode::Param:
    case OpCode::Implicit:
      return std::format("{}: {}", v,
                         graph_.names()[input_column(graph_.inputs(), p.op,
                                                     p.slot)]);
    case OpCode::Seed:
      return std::format("{}: v[{}]", v, p.slot);
    case OpCode::Const:
      if constexpr (std::formattable<T, char>) {
        return std::format("{}: {}", v, p.value);
      } else {
        return std::format("{}: const", v);
      }
    default:
      return std::format("{}: {}", v, label_of(p.op));
    }
  }

  friend std::ostream &operator<<(std::ostream &out, const Dot &d) {
    return out << d.str();
  }

  const Graph<T> &graph_;
  Scope scope_;
};

template <impl::Numeric T> Dot(const Graph<T> &, Scope = Scope::Live) -> Dot<T>;

} // namespace ddx::rt

template <ddx::impl::Numeric T> struct std::formatter<ddx::rt::Dot<T>, char> {
  constexpr auto parse(std::format_parse_context &ctx) const {
    return ctx.begin();
  }
  auto format(const ddx::rt::Dot<T> &d, std::format_context &ctx) const {
    ddx::impl::detail::fmt_put(ctx, d.str());
    return ctx.out();
  }
};
