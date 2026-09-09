// Here rather than in rt/dot.hpp so boost/graph/graphviz.hpp -- which carries
// the DOT *reader* too -- reaches this translation unit and no other.

#include "rt/dot.hpp"

#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/function_property_map.hpp>

#include <sstream>
#include <string>

namespace ddx::rt {
namespace {

// Only vertices are filtered: reaching a node is what made its operands live.
struct IsLive {
  std::span<const DotNode> nodes;
  bool operator()(Vertex v) const { return nodes[v].live; }
};

using Edge = boost::graph_traits<Adjacency>::edge_descriptor;

auto vertex_map(std::invocable<NodeId> auto f) {
  return boost::make_function_property_map<Vertex, std::string>(
      [f = std::move(f)](Vertex v) { return f(static_cast<NodeId>(v)); });
}

// dynamic_properties rather than a hand-written label writer, so the names are
// the ones graphviz knows and BGL does the quoting.
boost::dynamic_properties properties(const Adjacency &adj,
                                     std::span<const DotNode> nodes) {
  boost::dynamic_properties dp;
  dp.property("node_id", boost::get(boost::vertex_index, adj));
  dp.property("label",
              vertex_map([nodes](NodeId v) { return nodes[v].label; }));
  dp.property("shape", vertex_map([nodes](NodeId v) {
                return std::string{nodes[v].shape};
              }));
  dp.property("style", vertex_map([nodes](NodeId v) {
                return std::string(nodes[v].live ? "solid" : "dashed");
              }));
  dp.property("label",
              boost::make_function_property_map<Edge, std::string>(
                  [&adj, nodes, slot = boost::get(boost::edge_bundle, adj)](
                      const Edge &e) -> std::string {
                    const auto v = static_cast<NodeId>(boost::source(e, adj));
                    return nodes[v].show_slots
                               ? std::to_string(boost::get(slot, e))
                               : std::string{};
                  }));
  return dp;
}

} // namespace

std::string to_dot(const Adjacency &adj, std::span<const DotNode> nodes,
                   Scope scope) {
  std::ostringstream out;
  auto dp = properties(adj, nodes);
  if (scope == Scope::All) {
    boost::write_graphviz_dp(out, adj, dp);
  } else {
    // An adaptor over the liveness bits the freeze already computed: a
    // predicate, no traversal, and the vertices keep their ids.
    boost::write_graphviz_dp(
        out, boost::filtered_graph{adj, boost::keep_all{}, IsLive{nodes}}, dp);
  }
  return out.str();
}

} // namespace ddx::rt
