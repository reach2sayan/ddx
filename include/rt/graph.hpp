#pragma once

#include "rt/builder.hpp"
#include "rt/coupling.hpp"
#include "rt/dense.hpp"
#include "rt/derivative.hpp"
#include "rt/opcode.hpp"
#include "util/ranges.hpp"

#include <boost/describe/class.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ddx::rt {

// Befriended by Graph: the CSR itself is the writer's business.
template <impl::Numeric T> class Dot;

// At namespace scope rather than inside Graph because it carries no `T`: every
// scalar freezes to the same CSR, so the graphviz writer compiles once.
using Adjacency =
    boost::compressed_sparse_row_graph<boost::directedS, boost::no_property,
                                       std::uint32_t>;
using Vertex = boost::graph_traits<Adjacency>::vertex_descriptor;

// What a lane's graph carries; each level is the one below plus a block.
// Appended rather than inserted, so a saved Object's `want` byte still names
// the same lane it did.  The columns: whether the lane is one function's (a
// system refuses it), whether it is second order (a function supplying first
// partials only refuses it), whether its kernel may be built over the
// rebalanced graph, and what its seeds are counted in.  The seeded and
// second-order lanes may not rebalance: their block is swept from the
// unblocked roots, so a rebalanced copy would answer in different last bits
// than the sweep it is checked against.
#define DDX_RT_WANT_TABLE(X)                                                   \
  X(Value, false, false, true, None)                                           \
  X(Jacobian, false, false, true, None)                                        \
  X(Hessian, true, true, false, None)                                          \
  X(Hvp, true, true, false, Symbols)                                           \
  X(Vjp, false, false, false, Functions)                                       \
  X(Jvp, false, false, false, Symbols)                                         \
  X(Gradient, false, false, true, None)                                        \
  X(JacobianTangent, false, true, false, Symbols)                              \
  X(DenseHessian, true, true, false, None)

enum class Want : std::uint8_t {
#define DDX_RT_WANT(Name, ...) Name,
  DDX_RT_WANT_TABLE(DDX_RT_WANT)
#undef DDX_RT_WANT
};
enum class Seeds : std::uint8_t { None, Symbols, Functions };
struct WantInfo {
  std::string_view name;
  bool single;
  bool second_order;
  bool rebalanceable;
  Seeds seeds;
};
inline constexpr auto want_table = [] {
  std::array<WantInfo, 0
#define DDX_RT_WANT(...) +1
                               DDX_RT_WANT_TABLE(DDX_RT_WANT)
#undef DDX_RT_WANT
             >
      t{};
  std::size_t i = 0;
#define DDX_RT_WANT(Name, single, second, rebalance, seeds)                    \
  t[i++] = {#Name, single, second, rebalance, Seeds::seeds};
  DDX_RT_WANT_TABLE(DDX_RT_WANT)
#undef DDX_RT_WANT
  return t;
}();
inline constexpr std::size_t want_count = want_table.size();

// The lanes in Cache order, so a lane zipped against the cache needs no cast
// back from an index.
inline constexpr auto want_values = [] {
  std::array<Want, want_count> out{};
  for (const std::size_t i : std::views::iota(0uz, want_count)) {
    out[i] = static_cast<Want>(i);
  }
  return out;
}();

[[nodiscard]] constexpr const WantInfo &want_info(Want w) noexcept {
  return want_table[std::to_underlying(w)];
}
[[nodiscard]] constexpr std::string_view name_of(Want w) noexcept {
  return want_info(w).name;
}
[[nodiscard]] constexpr bool rebalanceable(Want w) noexcept {
  return want_info(w).rebalanceable;
}

// One of a thing per output block, in the order the kernel writes them.  The
// ids a builder collects, the spans a frozen graph lends, the counts read off
// them and the columns codegen loads are the same three names, said once.
// (jit::KernelShape cannot share it: the core may not include rt/.)
template <std::semiregular Per> struct Blocks {
  Per values{};
  Per jacobian{};
  Per hessian{};
  BOOST_DESCRIBE_CLASS(Blocks, (), (values, jacobian, hessian), (), ())
};

using OutputBlocks = Blocks<std::vector<NodeId>>;    // what a freeze is handed
using OutputSpans = Blocks<std::span<const NodeId>>; // frozen graph view
// The column counts, read off the blocks so no column can be miscounted:
// `values` is m, `jacobian` is the pattern's nonzeros row-major by function,
// and `hessian` is colours * n, compressed.
using Layout = Blocks<std::size_t>;
// What a sweep writes into: a triple rather than three parameters, so a block
// and the columns it fills cannot be paired one way here and another there.
template <impl::Numeric T> using Columns = Blocks<std::span<T *const>>;

// The three, in the order the ABI writes them.  Said here and nowhere else.
template <std::semiregular Per>
[[nodiscard]] constexpr auto in_order(const Blocks<Per> &b) {
  return std::array{b.values, b.jacobian, b.hessian};
}

// One triple against another, block for block.
template <std::semiregular A, std::semiregular B>
[[nodiscard]] constexpr auto zip_blocks(const Blocks<A> &a,
                                        const Blocks<B> &b) {
  return std::views::zip(in_order(a), in_order(b));
}

// Where each output's value sits: a scalar's tape row, or a vector's tensor
// rows, which fill `width` consecutive output columns -- the dense Hessian is
// one such output, n^2 columns wide.
using OutputRows = Blocks<std::span<const TensorRow>>;

// `stride` is the tape's lanes per node, `width` how many this sweep filled,
// `i` where they start in the caller's columns.  A scalar output is one copy
// off the tape; a vector output is one column per element off the tensor tape,
// whose lanes are element-minor.
template <impl::Numeric T>
constexpr void scatter_blocks(const OutputRows &rows, std::span<const T> tape,
                              std::span<const T> vtape, const Columns<T> &out,
                              std::size_t i, std::size_t stride,
                              std::size_t width) {
  for (const auto [columns, block] : zip_blocks(out, rows)) {
    auto column = columns.begin();
    for (const TensorRow &r : block) {
      if (r.width == 0) {
        std::ranges::copy_n(tape.data() + std::size_t{r.at} * stride,
                            static_cast<std::ptrdiff_t>(width), *column + i);
        ++column;
        continue;
      }
      const T *const base = vtape.data() + std::size_t{r.at} * stride;
      for (std::size_t e = 0; e < r.width; ++e, ++column) {
        for (std::size_t k = 0; k < width; ++k) {
          (*column)[i + k] = base[k * r.width + e];
        }
      }
    }
  }
}

// A builder frozen into CSR, the form codegen walks.  Operand position rides
// along as an edge attribute, because a CSR row is a set.
template <impl::Numeric T = double> class Graph {
public:
  using value_type = T;
  struct Property {
    OpCode op = OpCode::Const;
    T value{};
    std::uint32_t slot = 0;
    BOOST_DESCRIBE_CLASS(Property, (), (op, value, slot), (), ())
  };

  // `contract` is settled here, not per sweep: it decides the arithmetic, so
  // changing it afterwards means freezing again.
  [[nodiscard]] static Graph freeze(const Builder<T> &b, OutputBlocks blocks,
                                    Coloring coloring = {},
                                    Sparsity jacobian = {},
                                    bool contract = true) {
    Graph g;
    // Node counts per block, and the column counts: a vector output is one
    // column per element.
    const auto columns = [&b](std::span<const NodeId> ids) {
      return std::ranges::fold_left(ids, 0uz, [&b](std::size_t n, NodeId v) {
        return n + (tensor_valued(b.op_of(v)) ? width_of(b, v) : 1uz);
      });
    };
    g.counts_ = {.values = blocks.values.size(),
                 .jacobian = blocks.jacobian.size(),
                 .hessian = blocks.hessian.size()};
    g.layout_ = {.values = columns(blocks.values),
                 .jacobian = columns(blocks.jacobian),
                 .hessian = columns(blocks.hessian)};
    g.outputs_ = std::move(blocks.values);
    g.outputs_.reserve(g.counts_.values + g.counts_.jacobian +
                       g.counts_.hessian);
    impl::append(g.outputs_, blocks.jacobian);
    impl::append(g.outputs_, blocks.hessian);
    g.coloring_ = std::move(coloring);
    g.jacobian_ = std::move(jacobian);
    g.inputs_ = b.inputs();
    g.externs_.assign(b.externs().begin(), b.externs().end());
    g.names_ = b.names();
    g.properties_.reserve(b.size());

    std::vector<std::pair<std::size_t, std::size_t>> edges;
    std::vector<std::uint32_t> slots;
    edges.reserve(b.size() * 3);
    slots.reserve(b.size() * 3);

    for (const auto [v, n] : b.nodes() | std::views::enumerate) {
      g.properties_.push_back({.op = n.op, .value = n.value, .slot = n.slot});
      const auto id = static_cast<std::size_t>(v);
      for (const auto [slot, u] : std::array{n.a, n.b, n.c} |
                                      std::views::enumerate |
                                      std::views::take(arity_of(n.op))) {
        edges.emplace_back(id, u);
        slots.push_back(static_cast<std::uint32_t>(slot));
      }
    }

    g.children_ =
        adjacency_type(boost::edges_are_unsorted_multi_pass, edges.begin(),
                       edges.end(), slots.begin(), b.size());
    g.mark_live();
    g.contract(contract);
    // Read off what survived rather than taken from the builder, so a lane
    // that kept no seed keeps the kernel and the digest it had.
    for (const NodeId v : g.schedule_ | std::views::transform(&Step::node)) {
      if (g.properties_[v].op == OpCode::Seed) {
        g.inputs_.seeds =
            std::max(g.inputs_.seeds, std::size_t{g.properties_[v].slot} + 1);
      }
      g.calls_extern_ |= kind_of(g.properties_[v].op) == Kind::Extern;
      g.has_tensors_ |= kind_of(g.properties_[v].op) == Kind::Tensor;
    }
    // The tape packed by liveness, scalar rows and tensor rows alike; off the
    // arena, which holds the tensor table the widths come from.
    auto packed = detail::pack_slots(b, g.schedule_, g.outputs_);
    g.slots_ = std::move(packed.slot);
    g.tape_slots_ = packed.rows;
    g.output_slots_ = g.outputs_ | std::views::transform([&g](NodeId o) {
                        return g.slots_[o];
                      }) |
                      impl::to<std::vector<std::uint32_t>>();
    g.tensors_ = detail::tensor_layout(b, g.schedule_, g.outputs_);
    g.output_rows_ =
        g.outputs_ | std::views::transform([&g, &b](NodeId o) {
          return tensor_valued(b.op_of(o)) ? g.tensors_.rows[o]
                                           : TensorRow{.at = g.slots_[o]};
        }) |
        impl::to<std::vector<TensorRow>>();
    return g;
  }

  // Every output a value.
  [[nodiscard]] static Graph freeze(const Builder<T> &b,
                                    std::span<const NodeId> values,
                                    bool contract = true) {
    OutputBlocks blocks;
    blocks.values.assign(values.begin(), values.end());
    return freeze(b, std::move(blocks), {}, {}, contract);
  }

  [[nodiscard]] std::size_t size() const { return properties_.size(); }
  [[nodiscard]] const Property &operator[](NodeId v) const {
    return properties_[v];
  }
  [[nodiscard]] std::span<const NodeId> outputs() const { return outputs_; }
  [[nodiscard]] const Layout &layout() const { return layout_; }
  [[nodiscard]] const Coloring &coloring() const { return coloring_; }
  // Which (function, symbol) cells the jacobian block holds, in its order.
  [[nodiscard]] const Sparsity &jacobian_pattern() const { return jacobian_; }
  // How many input columns of each kind `xs` carries, and the names of all
  // but the seeds, in column order.
  [[nodiscard]] const Inputs &inputs() const { return inputs_; }
  [[nodiscard]] const std::vector<std::string> &names() const {
    return names_;
  }

  // The functions extern nodes call, shared with the arena: a kernel lowered
  // from this graph calls them by index into this table.
  [[nodiscard]] std::span<const ExternEntry<T>> externs() const {
    return externs_;
  }
  // Whether a live node calls one: what a device, which cannot call the host,
  // refuses -- a function the arena registered and nothing reaches is not.
  [[nodiscard]] bool calls_extern() const noexcept { return calls_extern_; }
  // Whether a live node is a tensor op: what a kernel, which computes scalars
  // only, refuses, and what makes the sweep carry a second tape.
  [[nodiscard]] bool has_tensors() const noexcept { return has_tensors_; }
  // Where each tensor-valued node's rows sit in that tape.
  [[nodiscard]] const TensorLayout &tensors() const noexcept {
    return tensors_;
  }

  // How many input columns `xs` carries: the symbol count is not the width.
  [[nodiscard]] std::size_t arity() const { return inputs_.total(); }

  [[nodiscard]] bool live(NodeId v) const { return live_[v]; }

  // How many ids the freeze kept, which is what a compile reports having
  // emitted.
  [[nodiscard]] std::size_t live_count() const { return live_order_.size(); }

  // The ids left to compute, in topological order so a consumer emits them in
  // one pass, each with its fma.  A multiply every reader swallowed into an fma
  // is computed by nobody; contraction_at() decides that arithmetic, off the
  // nodes, and the freeze keeps the answer so no sweep re-derives it per point.
  // The span must not outlive this graph.
  [[nodiscard]] std::span<const Step> schedule() const { return schedule_; }

  // Derived once, for codegen, the interpreter and the ABI size checks.
  [[nodiscard]] OutputSpans output_blocks() const {
    return blocks_of(std::span<const NodeId>{outputs_});
  }
  // The same blocks as tape rows: what a sweep reads its answers off.
  [[nodiscard]] OutputSpans output_slots() const {
    return blocks_of(std::span<const std::uint32_t>{output_slots_});
  }
  [[nodiscard]] OutputRows output_rows() const {
    return blocks_of(std::span<const TensorRow>{output_rows_});
  }
  // By id, the tape row each scheduled node's value sits in; how many rows the
  // tape has.  Packed by liveness, so the tape is a fraction of the arena.
  [[nodiscard]] std::span<const std::uint32_t> slots() const noexcept {
    return slots_;
  }
  [[nodiscard]] std::size_t tape_slots() const noexcept { return tape_slots_; }

  [[nodiscard]] OpCode op_of(NodeId v) const { return properties_[v].op; }

  // Operands in slot order; arity is at most three, a select being the only op
  // that reaches it.
  [[nodiscard]] std::array<NodeId, 3> operands(NodeId v) const {
    std::array out{no_node, no_node, no_node};
    for (const auto &e : operand_edges(v)) {
      out[children_[e]] = static_cast<NodeId>(boost::target(e, children_));
    }
    return out;
  }

private:
  using adjacency_type = Adjacency;
  using vertex_type = Vertex;

  template <typename E>
  [[nodiscard]] Blocks<std::span<const E>> blocks_of(std::span<const E> all) const {
    return {.values = all.first(counts_.values),
            .jacobian = all.subspan(counts_.values, counts_.jacobian),
            .hessian = all.subspan(counts_.values + counts_.jacobian,
                                   counts_.hessian)};
  }

  // Only the DOT writer wants these; everything else reads `operands`.
  friend class Dot<T>;
  [[nodiscard]] const adjacency_type &children() const { return children_; }

  [[nodiscard]] auto operand_edges(NodeId v) const {
    const auto [first, last] =
        boost::out_edges(static_cast<vertex_type>(v), children_);
    return std::ranges::subrange(first, last);
  }

  // Nothing but the outputs and what they reach needs to be emitted.
  void mark_live() {
    live_ = detail::reachable(
        properties_.size(), outputs_, [this](NodeId v, auto &&mark) {
          for (const auto &edge : operand_edges(v)) {
            mark(static_cast<NodeId>(boost::target(edge, children_)));
          }
        });
    live_order_.reserve(
        static_cast<std::size_t>(std::ranges::count(live_, true)));
    impl::append(live_order_, detail::live_ids(live_));
  }

  // Liveness under contraction, which is all this freeze settles -- an add
  // reaches the multiply's operands rather than the multiply, so a multiply
  // nothing else reads drops out with the Neg of a subtraction behind it.
  void contract(bool on) {
    if (!on) {
      schedule_ = detail::schedule_of(*this, live_order_, false);
      return;
    }
    const std::vector<bool> live = detail::reachable(
        properties_.size(), outputs_, [this](NodeId v, auto &&mark) {
          if (const Contraction c = contraction_at(*this, v)) {
            mark(c.x);
            mark(c.y);
            mark(c.z);
            return;
          }
          for (const auto &edge : operand_edges(v)) {
            mark(static_cast<NodeId>(boost::target(edge, children_)));
          }
        });
    schedule_ = detail::schedule_of(
        *this, live_order_ |
                   std::views::filter([&live](NodeId v) { return live[v]; }));
  }

  std::vector<Property> properties_;
  adjacency_type children_;
  std::vector<NodeId> outputs_;
  Layout layout_;          // columns per block
  Blocks<std::size_t> counts_; // nodes per block
  std::vector<TensorRow> output_rows_;
  Coloring coloring_;
  Sparsity jacobian_;
  Inputs inputs_;
  std::vector<std::string> names_;
  std::vector<ExternEntry<T>> externs_;
  bool calls_extern_ = false;
  bool has_tensors_ = false;
  TensorLayout tensors_;
  std::vector<std::uint32_t> slots_;
  std::vector<std::uint32_t> output_slots_;
  std::size_t tape_slots_ = 0;
  std::vector<bool> live_;
  std::vector<NodeId> live_order_;
  std::vector<Step> schedule_;
};

// Each step names one block of output columns; `finish` is the only thing that
// produces a Graph.
//
//   GraphBuilder{b}.value(f).build_jacobian().finish()
//   GraphBuilder{b}.values({f0, f1}).build_jacobian().finish()
//   GraphBuilder{b}.value(f).build_jacobian().build_hessian().finish()
//
// The class template argument is deduced from the builder.
// What a freeze may ask a lane for.  Each is fetched only on the branch that
// wants it, so a caller whose sweeps are lazy stays lazy.
template <typename S>
concept CSweeps = requires(S &s) {
  { s.roots() } -> std::convertible_to<std::span<const NodeId>>;
  { s.jacobian() } -> std::convertible_to<const Jacobian &>;
  { s.hessian() } -> std::convertible_to<const Hessian &>;
  { s.hessian_vector() } -> std::convertible_to<const HessianVector &>;
  { s.vector_jacobian() } -> std::convertible_to<const VectorJacobian &>;
  { s.tangent() } -> std::convertible_to<const Tangent &>;
  { s.jacobian_tangent() } -> std::convertible_to<std::span<const NodeId>>;
  { s.dense_hessian() } -> std::convertible_to<const DenseHessian &>;
};

// Defined below GraphBuilder, which befriends it: the only caller of the
// block-taking members it keeps private.
template <impl::Numeric T>
[[nodiscard]] Graph<T> freeze_for(Want want, Builder<T> &arena,
                                  CSweeps auto &sweeps, bool contract);

template <impl::Numeric T = double> class GraphBuilder {
public:
  explicit constexpr GraphBuilder(Builder<T> &b) noexcept : builder_{&b} {}

  // The function the kernel computes.
  constexpr GraphBuilder &value(const RTExpression<T> &root) {
    return values({root});
  }

  // A system: m functions over the same symbols.
  constexpr GraphBuilder &values(std::initializer_list<RTExpression<T>> roots) {
    roots_ =
        roots |
        std::views::transform([&](const auto &e) { return e.id(*builder_); }) |
        impl::to<std::vector<NodeId>>();
    blocks_.values = roots_;
    return *this;
  }

  // Nodes a caller already has: Equation builds its derivative in the
  // constructor and freezes only on a batch call.
  constexpr GraphBuilder &values_from(std::span<const NodeId> roots) {
    roots_.assign(roots.begin(), roots.end());
    blocks_.values = roots_;
    return *this;
  }

  // The functions to differentiate, and not outputs: the graph then carries
  // no value block, and whatever only the value needs is not live.
  constexpr GraphBuilder &roots_from(std::span<const NodeId> roots) {
    roots_.assign(roots.begin(), roots.end());
    blocks_.values.clear();
    return *this;
  }

  // The whole sweep rather than its nodes: the pattern is what makes the block
  // readable, so the two must not travel separately.
  constexpr GraphBuilder &jacobian_from(const Jacobian &j) {
    jacobian_ = j.pattern;
    return block(&OutputBlocks::jacobian, j.partial);
  }

  // Every partial, in symbol order.  One reverse sweep per function.
  constexpr GraphBuilder &build_jacobian() {
    return jacobian_from(rt::build_jacobian_impl(*builder_, roots_));
  }

  // Compressed by colour, not n x n: a handful of columns when banded.
  GraphBuilder &build_hessian() {
    return hessian_from(rt::build_hessian_impl(*builder_, roots_.front()));
  }

  [[nodiscard]] Graph<T> finish(bool contract = true) const {
    return Graph<T>::freeze(*builder_, blocks_, coloring_, jacobian_, contract);
  }

private:
  // The Hessian an Equation already holds, so the arena is not swept again:
  // rt::build_hessian_impl appends to the builder, and freeze() must not.
  template <impl::Numeric U>
  friend Graph<U> freeze_for(Want, Builder<U> &, CSweeps auto &, bool);

  constexpr GraphBuilder &block(std::vector<NodeId> OutputBlocks::*which,
                                std::span<const NodeId> ids) {
    (blocks_.*which).assign(ids.begin(), ids.end());
    return *this;
  }
  constexpr GraphBuilder &hessian_from(const Hessian &h) {
    coloring_ = h.coloring;
    return block(&OutputBlocks::hessian, h.compressed);
  }
  // H v is a second-order block of one column, so the colouring stays empty
  // and coloring() is not always a way to read that block back.
  constexpr GraphBuilder &hessian_vector_from(const HessianVector &h) {
    return block(&OutputBlocks::hessian, h.product);
  }
  // J v is a first-order block of m columns, one per function.
  constexpr GraphBuilder &tangent_from(const Tangent &t) {
    return block(&OutputBlocks::jacobian, t.product);
  }
  // dJ/dv has the Jacobian's layout: one column per cell, read with its
  // pattern.
  constexpr GraphBuilder &
  jacobian_tangent_from(std::span<const NodeId> product) {
    return block(&OutputBlocks::hessian, product);
  }
  // w'J is a first-order block of n columns and dense, so jacobian_pattern()
  // stays empty.
  constexpr GraphBuilder &vector_jacobian_from(const VectorJacobian &j) {
    return block(&OutputBlocks::jacobian, j.product);
  }
  // The dense Hessian is one vector output, n x n wide; no colouring.
  constexpr GraphBuilder &dense_hessian_from(const DenseHessian &h) {
    return block(&OutputBlocks::hessian, std::span{&h.block, 1});
  }

  Builder<T> *builder_;
  std::vector<NodeId> roots_;
  OutputBlocks blocks_;
  Coloring coloring_;
  Sparsity jacobian_;
};

template <impl::Numeric T> GraphBuilder(Builder<T> &) -> GraphBuilder<T>;

// The graph one lane computes: the value block unless the want is Gradient,
// which differentiates the roots without computing them, and the one derivative
// block that want names.  Vjp and Jvp replace the Jacobian block rather than
// adding to it -- n columns instead of the pattern's nonzeros is the whole
// point of asking.
template <impl::Numeric T>
[[nodiscard]] Graph<T> freeze_for(Want want, Builder<T> &arena,
                                  CSweeps auto &sweeps, bool contract) {
  GraphBuilder<T> gb{arena};
  if (want == Want::Gradient) {
    gb.roots_from(sweeps.roots());
  } else {
    gb.values_from(sweeps.roots());
  }
  switch (want) {
  case Want::Value:
    break;
  case Want::Vjp:
    gb.vector_jacobian_from(sweeps.vector_jacobian());
    break;
  case Want::Jvp:
    gb.tangent_from(sweeps.tangent());
    break;
  case Want::Jacobian:
  case Want::Gradient:
    gb.jacobian_from(sweeps.jacobian());
    break;
  case Want::Hessian:
    gb.jacobian_from(sweeps.jacobian()).hessian_from(sweeps.hessian());
    break;
  case Want::Hvp:
    gb.jacobian_from(sweeps.jacobian())
        .hessian_vector_from(sweeps.hessian_vector());
    break;
  case Want::JacobianTangent:
    gb.jacobian_from(sweeps.jacobian())
        .jacobian_tangent_from(sweeps.jacobian_tangent());
    break;
  case Want::DenseHessian:
    gb.jacobian_from(sweeps.jacobian()).dense_hessian_from(sweeps.dense_hessian());
    break;
  }
  return gb.finish(contract);
}

} // namespace ddx::rt
