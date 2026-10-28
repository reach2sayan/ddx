#pragma once

#include "rt/arena.hpp"
#include "rt/coupling.hpp"
#include "rt/dense.hpp"
#include "rt/derivative.hpp"
#include "rt/opcode.hpp"
#include "util/export.hpp"
#include "util/ranges.hpp"

#include <boost/describe/class.hpp>

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

// What a Want's graph carries; each level is the one below plus a block.  The
// columns: whether the Want is one function's (a system refuses it), whether
// it is second order (a function supplying first partials only refuses it),
// whether its kernel may be built over the rebalanced graph, what its
// direction is counted in, whether it fills the values block, and the shape of
// its Jacobian and Hessian blocks -- the pattern's cells, one per symbol or per
// function, the colouring's compressed columns, or n x n dense.  The seeded
// and second-order Wants may not rebalance: their block is swept from the
// unblocked roots, so a rebalanced copy would answer in different last bits
// than the sweep it is checked against.
#define DDX_RT_WANT_TABLE(X)                                                   \
  X(Value, false, false, true, None, true, None, None)                         \
  X(Jacobian, false, false, true, None, true, Cells, None)                     \
  X(Hessian, true, true, false, None, true, Cells, Colours)                    \
  X(Hvp, true, true, false, Symbols, true, Cells, Symbols)                     \
  X(Vjp, false, false, false, Functions, true, Symbols, None)                  \
  X(Jvp, false, false, false, Symbols, true, Functions, None)                  \
  X(Gradient, false, false, true, None, false, Cells, None)                    \
  X(JacobianJvp, false, true, false, Symbols, true, Cells, Cells)              \
  X(DenseHessian, true, true, false, None, true, Cells, Dense)

enum class Want : std::uint8_t {
#define DDX_RT_WANT(Name, ...) Name,
  DDX_RT_WANT_TABLE(DDX_RT_WANT)
#undef DDX_RT_WANT
};
enum class Direction : std::uint8_t { None, Symbols, Functions };
enum class BlockShape : std::uint8_t {
  None,
  Cells,
  Symbols,
  Functions,
  Colours,
  Dense
};
struct WantInfo {
  std::string_view name;
  bool single;
  bool second_order;
  bool rebalanceable;
  Direction direction;
  bool values;
  BlockShape jacobian;
  BlockShape hessian;
};
inline constexpr std::array want_table{
#define DDX_RT_WANT(Name, single, second, rebalance, direction, values, jac,   \
                    hes)                                                       \
  WantInfo{#Name,  single, second,          rebalance, Direction::direction, \
           values, BlockShape::jac, BlockShape::hes},
    DDX_RT_WANT_TABLE(DDX_RT_WANT)
#undef DDX_RT_WANT
};
inline constexpr std::size_t want_count = want_table.size();

// The Wants in table order, so a pipeline zipped against them needs no cast
// back from an index.
inline constexpr std::array want_values{
#define DDX_RT_WANT(Name, ...) Want::Name,
    DDX_RT_WANT_TABLE(DDX_RT_WANT)
#undef DDX_RT_WANT
};

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
// ids an arena collects, the spans a frozen graph lends, the counts read off
// them and the columns codegen loads are the same three names, said once.
// (jit::KernelShape cannot share it: the core may not include rt/.)
template <std::semiregular Per> struct PerBlock {
  Per values{};
  Per jacobian{};
  Per hessian{};
  BOOST_DESCRIBE_CLASS(PerBlock, (), (values, jacobian, hessian), (), ())
};

using OutputBlocks = PerBlock<std::vector<NodeId>>; // what a freeze is handed
using OutputSpans = PerBlock<std::span<const NodeId>>; // frozen graph view
// The column counts, read off the blocks so no column can be miscounted:
// `values` is m, `jacobian` is the pattern's nonzeros row-major by function,
// and `hessian` is colours * n, compressed.
using Layout = PerBlock<std::size_t>;
// What a sweep writes into: a triple rather than three parameters, so a block
// and the columns it fills cannot be paired one way here and another there.
template <impl::Numeric T> using Columns = PerBlock<std::span<T *const>>;

// The three, in the order the ABI writes them.  Said here and nowhere else.
template <std::semiregular Per>
[[nodiscard]] constexpr auto in_order(const PerBlock<Per> &b) {
  return std::array{b.values, b.jacobian, b.hessian};
}

// One triple against another, block for block.
template <std::semiregular A, std::semiregular B>
[[nodiscard]] constexpr auto zip_blocks(const PerBlock<A> &a,
                                        const PerBlock<B> &b) {
  return std::views::zip(in_order(a), in_order(b));
}

// Where each output's value sits: a scalar's tape row, or a vector's tensor
// rows, which fill `width` consecutive output columns -- the dense Hessian is
// one such output, n^2 columns wide.
using OutputRows = PerBlock<std::span<const TensorRow>>;

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
      // Column c of a column-major rows x cols matrix, read row-major.
      const std::size_t cols = r.width / r.rows;
      const T *const base = vtape.data() + std::size_t{r.at} * stride;
      for (std::size_t c = 0; c < r.width; ++c, ++column) {
        const std::size_t e = c / cols + (c % cols) * r.rows;
        for (std::size_t k = 0; k < width; ++k) {
          (*column)[i + k] = base[k * r.width + e];
        }
      }
    }
  }
}

// An arena frozen into per-node arrays, the form codegen walks: three operand
// slots each, so position is the index, and lists for the n-ary nodes.
template <impl::Numeric T = double> class DDX_TEMPLATE_API Graph {
public:
  using value_type = T;
  struct NodeRecord {
    OpCode op = OpCode::Const;
    T value{};
    std::uint32_t payload = 0; // read through detail::get
    BOOST_DESCRIBE_CLASS(NodeRecord, (), (op, value, payload), (), ())
  };

  // `contract` is settled here, not per sweep: it decides the arithmetic, so
  // changing it afterwards means freezing again.
  // `directions` is how many direction columns the call hands, read or not.
  [[nodiscard]] static Graph freeze(const Arena<T> &b, OutputBlocks blocks,
                                    Colouring colouring = {},
                                    Sparsity jacobian = {},
                                    bool contract = true,
                                    std::size_t directions = 0) {
    Graph g;
    // Node counts per block, and the column counts: a vector output is one
    // column per element.
    const auto columns = [&b](std::span<const NodeId> ids) {
      return std::ranges::fold_left(ids, 0uz, [&b](std::size_t n, NodeId v) {
        return n + (detail::is_tensor(b[v]) ? width_of(b, v) : 1uz);
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
    g.colouring_ = std::move(colouring);
    g.jacobian_pattern_ = std::move(jacobian);
    g.inputs_ = b.inputs();
    g.inputs_.directions = directions;
    g.externs_.assign(b.externs().begin(), b.externs().end());
    g.lists_.assign(b.lists().begin(), b.lists().end());
    g.names_ = b.names();
    g.properties_.reserve(b.size());
    g.operands_.reserve(b.size());
    for (const Node<T> &n : b.nodes()) {
      g.properties_.push_back(
          {.op = n.op, .value = n.value, .payload = n.payload});
      g.operands_.push_back(n.in);
    }
    g.mark_live();
    g.contract(contract);
    // Read off what survived rather than taken from the arena, so a
    // pipeline that kept no seed keeps the kernel it had.
    for (const NodeId v : g.schedule_ | std::views::transform(&Step::node)) {
      if (g.properties_[v].op == OpCode::Seed) {
        g.inputs_.directions = std::max(
            g.inputs_.directions,
            std::size_t{detail::get<Slot>(g.properties_[v]).value} + 1);
      }
      g.has_tensors_ |= detail::on_tensors(g.properties_[v]);
    }
    if (g.has_tensors_) {
      g.pair_products();
    }
    // The tape packed by liveness, scalar rows and tensor rows alike; off the
    // arena, which holds the tensor table the widths come from.
    auto packed = detail::pack_rows(b, g.schedule_, g.outputs_);
    g.rows_ = std::move(packed.by_id);
    g.tape_rows_ = packed.rows;
    g.fused_ = detail::fused_into(g, g.schedule_, g.outputs_);
    g.tensors_ =
        detail::tensor_layout(b, g.schedule_, g.outputs_, g.fused_);
    if (g.has_tensors_) {
      g.shapes_.resize(b.size());
      for (const Step &step : g.schedule_) {
        const NodeId v = step.node;
        if (b.op_of(v) == OpCode::MatConst) {
          const auto &m = b.tensors()[detail::get<TensorRef>(b[v]).value];
          g.shapes_[v] = {.rows = m.rows, .cols = m.cols};
        } else if (detail::is_tensor(b[v])) {
          g.shapes_[v] = shape_of(b, v);
        }
      }
    }
    g.output_rows_ = g.outputs_ | std::views::transform([&g, &b](NodeId o) {
                       if (!detail::is_tensor(b[o])) {
                         return TensorRow{.at = g.rows_[o]};
                       }
                       TensorRow r = g.tensors_.rows[o];
                       r.rows = shape_of(b, o).rows;
                       return r;
                     }) |
                     impl::to<std::vector<TensorRow>>();
    return g;
  }

  // Every output a value.
  [[nodiscard]] static Graph freeze(const Arena<T> &b,
                                    std::span<const NodeId> values,
                                    bool contract = true) {
    OutputBlocks blocks;
    blocks.values.assign(values.begin(), values.end());
    return freeze(b, std::move(blocks), {}, {}, contract);
  }

  [[nodiscard]] std::size_t size() const { return properties_.size(); }
  [[nodiscard]] const NodeRecord &operator[](NodeId v) const {
    return properties_[v];
  }
  [[nodiscard]] std::span<const NodeId> outputs() const { return outputs_; }
  [[nodiscard]] const Layout &layout() const { return layout_; }
  [[nodiscard]] const Colouring &colouring() const { return colouring_; }
  // Which (function, symbol) cells the jacobian block holds, in its order.
  [[nodiscard]] const Sparsity &jacobian_pattern() const {
    return jacobian_pattern_;
  }
  // How many input columns of each kind `xs` carries, and the names of all
  // but the seeds, in column order.
  [[nodiscard]] const Inputs &inputs() const { return inputs_; }
  [[nodiscard]] const std::vector<std::string> &names() const { return names_; }

  // The functions extern nodes call, shared with the arena: a kernel lowered
  // from this graph calls them by index into this table.
  [[nodiscard]] std::span<const ExternEntry<T>> externs() const {
    return externs_;
  }
  // Every has_list node's ids, shared with the arena: what arguments() reads.
  [[nodiscard]] std::span<const NodeId> lists() const { return lists_; }
  // Whether a live node is a tensor op: what makes the sweep carry a second
  // tape, and a kernel call the interpreter's step for that node.
  [[nodiscard]] bool has_tensors() const noexcept { return has_tensors_; }
  // Where each tensor-valued node's rows sit in that tape.
  [[nodiscard]] const TensorLayout &tensors() const noexcept {
    return tensors_;
  }
  // By id, a tensor node's rows x cols, and a table's; what a kernel's loops
  // run over.
  [[nodiscard]] Shape shape(NodeId v) const { return shapes_[v]; }
  // By id, the step a kernel computes a node inside, or no_node where the
  // node is a row of its own.  The tensor layout keeps such a node's operands
  // live until that step.
  [[nodiscard]] NodeId fused_into(NodeId v) const {
    return fused_.empty() ? no_node : fused_[v];
  }
  // By id, the other product of a pair -- one matrix times one vector, both
  // ways round -- which the schedule runs next to it, else no_node.  Empty
  // for a graph with no tensor.
  [[nodiscard]] std::span<const NodeId> pairs() const { return paired_; }

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
  [[nodiscard]] OutputRows output_rows() const {
    return blocks_of(std::span<const TensorRow>{output_rows_});
  }
  // By id, the tape row each scheduled node's value sits in; how many rows the
  // tape has.  Packed by liveness, so the tape is a fraction of the arena.
  [[nodiscard]] std::span<const std::uint32_t> rows() const noexcept {
    return rows_;
  }
  [[nodiscard]] std::size_t tape_rows() const noexcept { return tape_rows_; }

  [[nodiscard]] OpCode op_of(NodeId v) const { return properties_[v].op; }

  // Operands in slot order; arity is at most three, a select being the only op
  // that reaches it.
  [[nodiscard]] std::array<NodeId, 3> operands(NodeId v) const {
    return operands_[v];
  }

private:
  template <typename E>
  [[nodiscard]] PerBlock<std::span<const E>>
  blocks_of(std::span<const E> all) const {
    return {.values = all.first(counts_.values),
            .jacobian = all.subspan(counts_.values, counts_.jacobian),
            .hessian = all.subspan(counts_.values + counts_.jacobian,
                                   counts_.hessian)};
  }

  // M v and M^T v, which a quadratic form's value and gradient both ask for,
  // are a pair: the later step moves to just after the earlier, where both
  // its operands already are, so a kernel reads the matrix once for the two.
  void pair_products() {
    paired_.assign(properties_.size(), no_node);
    std::vector<bool> later(properties_.size(), false);
    std::vector<NodeId> products;
    for (const NodeId v : schedule_ | std::views::transform(&Step::node)) {
      if (op_of(v) != OpCode::MatVec) {
        continue;
      }
      const auto mate = std::ranges::find_if(products, [&](NodeId u) {
        return paired_[u] == no_node && operands(u)[0] == operands(v)[0] &&
               operands(u)[1] == operands(v)[1];
      });
      if (mate == products.end()) {
        products.push_back(v);
        continue;
      }
      paired_[*mate] = v;
      paired_[v] = *mate;
      later[v] = true;
    }
    std::vector<Step> moved(properties_.size());
    for (const Step &step : schedule_) {
      if (later[step.node]) {
        moved[step.node] = step;
      }
    }
    std::vector<Step> out;
    out.reserve(schedule_.size());
    for (const Step &step : schedule_) {
      if (later[step.node]) {
        continue;
      }
      out.push_back(step);
      if (paired_[step.node] != no_node) {
        out.push_back(moved[paired_[step.node]]);
      }
    }
    schedule_ = std::move(out);
  }

  // Nothing but the outputs and what they reach needs to be emitted.
  void mark_live() {
    live_ = detail::reachable(*this, outputs_);
    live_order_.reserve(
        static_cast<std::size_t>(std::ranges::count(live_, true)));
    impl::append(live_order_, detail::live_ids(live_));
  }

  // Liveness under contraction, which is all this freeze settles -- an add,
  // or a sum, reaches a multiply's operands rather than the multiply, so a
  // multiply nothing else reads drops out with the Neg of a subtraction
  // behind it.
  void contract(bool on) {
    if (!on) {
      schedule_ = detail::schedule_of(*this, live_order_, false);
      return;
    }
    const std::vector<bool> live = detail::reachable(
        *this, outputs_, [](const Graph &g, NodeId v, auto &&mark) {
          if (const Contraction c = contraction_at(g, v)) {
            mark(c.x);
            mark(c.y);
            mark(c.z);
            return;
          }
          if (g.op_of(v) == OpCode::SumN) {
            for (const NodeId t : arguments(g, v)) {
              if (const Contraction c = product_of(g, t)) {
                mark(c.x);
                mark(c.y);
              } else {
                mark(t);
              }
            }
            return;
          }
          detail::for_each_operand(g, v, mark);
        });
    schedule_ = detail::schedule_of(
        *this, live_order_ |
                   std::views::filter([&live](NodeId v) { return live[v]; }));
  }

  std::vector<NodeRecord> properties_;
  std::vector<std::array<NodeId, 3>> operands_;
  std::vector<NodeId> outputs_;
  Layout layout_;                // columns per block
  PerBlock<std::size_t> counts_; // nodes per block
  std::vector<TensorRow> output_rows_;
  Colouring colouring_;
  Sparsity jacobian_pattern_;
  Inputs inputs_;
  std::vector<std::string> names_;
  std::vector<ExternEntry<T>> externs_;
  std::vector<NodeId> lists_;
  bool has_tensors_ = false;
  TensorLayout tensors_;
  std::vector<Shape> shapes_;
  std::vector<NodeId> fused_;
  std::vector<NodeId> paired_;
  std::vector<std::uint32_t> rows_;
  std::size_t tape_rows_ = 0;
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
// The class template argument is deduced from the arena.
template <impl::Numeric T = double> class GraphBuilder {
public:
  explicit constexpr GraphBuilder(Arena<T> &b) noexcept : arena_{&b} {}

  // The function the kernel computes.
  constexpr GraphBuilder &value(const Expression<T> &root) {
    return values({root});
  }

  // A system: m functions over the same symbols.
  constexpr GraphBuilder &values(std::initializer_list<Expression<T>> roots) {
    return values_from(roots | std::views::transform([&](const auto &e) {
                         return e.id(*arena_);
                       }) |
                       impl::to<std::vector<NodeId>>());
  }

  // Nodes a caller already has.  Not `output`: the functions to differentiate
  // only, so the graph carries no value block and whatever only the value
  // needs is not live.
  constexpr GraphBuilder &values_from(std::span<const NodeId> roots,
                                      bool output = true) {
    roots_.assign(roots.begin(), roots.end());
    blocks_.values = output ? roots_ : std::vector<NodeId>{};
    return *this;
  }

  // Every partial, in symbol order, with the pattern that makes the block
  // readable.  One reverse sweep per function.
  constexpr GraphBuilder &build_jacobian() {
    const Jacobian j = rt::derive_jacobian(*arena_, roots_);
    jacobian_pattern_ = j.pattern;
    blocks_.jacobian = j.partial;
    return *this;
  }

  // Compressed by colour, not n x n: a handful of columns when banded.
  GraphBuilder &build_hessian() {
    const Hessian h = rt::derive_hessian(*arena_, roots_.front());
    colouring_ = h.colouring;
    blocks_.hessian = h.compressed;
    return *this;
  }

  [[nodiscard]] Graph<T> finish(bool contract = true) const {
    return Graph<T>::freeze(*arena_, blocks_, colouring_, jacobian_pattern_,
                            contract);
  }

private:
  Arena<T> *arena_;
  std::vector<NodeId> roots_;
  OutputBlocks blocks_;
  Colouring colouring_;
  Sparsity jacobian_pattern_;
};

template <impl::Numeric T> GraphBuilder(Arena<T> &) -> GraphBuilder<T>;

} // namespace ddx::rt

DDX_EXTERN_INSTANCE(class DDX_INSTANCE_API ::ddx::rt::Graph<double>)
DDX_EXTERN_INSTANCE(class DDX_INSTANCE_API ::ddx::rt::Graph<float>)
