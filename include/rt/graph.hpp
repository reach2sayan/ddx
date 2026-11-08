#pragma once

#include "rt/arena.hpp"
#include "rt/coupling.hpp"
#include "rt/dense.hpp"
#include "rt/derivative.hpp"
#include "rt/opcode.hpp"
#include "util/export.hpp"
#include "util/ranges.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ddx::rt {

// What each Want's graph computes; the columns are WantInfo's fields in order.
// Seeded and second-order Wants are swept from unblocked roots, so they cannot
// use the rebalanced graph without changing their last bits.
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
  WantInfo{#Name,                                                              \
           single,                                                             \
           second,                                                             \
           rebalance,                                                          \
           Direction::direction,                                               \
           values,                                                             \
           BlockShape::jac,                                                    \
           BlockShape::hes},
    DDX_RT_WANT_TABLE(DDX_RT_WANT)
#undef DDX_RT_WANT
};
inline constexpr std::size_t want_count = want_table.size();

// Every Want in table order, for zipping against per-Want arrays.
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

// One Per for each output block, in the order the kernel writes them.
// jit::KernelShape cannot reuse it because the JIT core may not include rt/.
template <std::semiregular Per> struct PerBlock {
  Per values{};
  Per jacobian{};
  Per hessian{};
};

using OutputBlocks = PerBlock<std::vector<NodeId>>; // the input to freeze()
// Column counts: m values, the Jacobian pattern's nonzeros, and colours * n
// compressed Hessian columns.
using Layout = PerBlock<std::size_t>;
// The output columns a sweep writes into, one span per block.
template <impl::Numeric T> using Columns = PerBlock<std::span<T *const>>;

// The three blocks as an array, in the order the ABI writes them.
template <std::semiregular Per>
[[nodiscard]] constexpr auto in_order(const PerBlock<Per> &b) {
  return std::array{b.values, b.jacobian, b.hessian};
}

template <std::semiregular A, std::semiregular B>
[[nodiscard]] constexpr auto zip_blocks(const PerBlock<A> &a,
                                        const PerBlock<B> &b) {
  return std::views::zip(in_order(a), in_order(b));
}

// Where each output's value sits: a scalar's tape row, or a tensor's rows,
// which fill `width` consecutive output columns.
using OutputRows = PerBlock<std::span<const TensorRow>>;

// Copies a sweep's `width` lanes into the output columns starting at point i.
// stride is the tape's lanes per node; tensor-tape lanes are element-minor.
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
      // Output column c reads a column-major matrix in row-major order.
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

// An arena frozen into flat per-node arrays, the form that codegen and the
// interpreter walk.
template <impl::Numeric T = double> class DDX_TEMPLATE_API Graph {
public:
  using value_type = T;
  struct NodeRecord {
    OpCode op = OpCode::Const;
    T value{};
    std::uint32_t payload = 0; // read through detail::get
    ListOffset list = no_list;
  };

  // contract fixes FMA contraction for this graph. fuse lets a kernel compute
  // elementwise rows inside their reader; a sweep gains nothing from it.
  [[nodiscard]] static Graph
  freeze(const Arena<T> &b, OutputBlocks blocks, Colouring colouring = {},
         Sparsity jacobian = {}, bool contract = true,
         std::size_t directions = 0, bool fuse = true) {
    Graph g;
    // A tensor output takes one column per element.
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
          {.op = n.op, .value = n.value, .payload = n.payload, .list = n.list});
      g.operands_.push_back(n.in);
    }
    g.mark_live();
    g.contract(contract ? Contract::on : Contract::off);
    // Directions are counted from the surviving seeds, not from the arena.
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
    // Packs the tape rows by liveness, for scalar and tensor rows alike.
    auto packed = detail::pack_rows(b, g.schedule_, g.outputs_);
    g.rows_ = std::move(packed.by_id);
    g.tape_rows_ = packed.rows;
    if (fuse) {
      g.fused_ = detail::fused_into(g, g.schedule_, g.outputs_);
    }
    g.tensors_ = detail::tensor_layout(b, g.schedule_, g.outputs_, g.fused_);
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

  [[nodiscard]] std::size_t size() const { return properties_.size(); }
  [[nodiscard]] const NodeRecord &operator[](NodeId v) const {
    return properties_[v];
  }
  [[nodiscard]] std::span<const NodeId> outputs() const { return outputs_; }
  [[nodiscard]] const Layout &layout() const { return layout_; }
  [[nodiscard]] const Colouring &colouring() const { return colouring_; }
  // The (function, symbol) cells of the jacobian block, in block order.
  [[nodiscard]] const Sparsity &jacobian_pattern() const {
    return jacobian_pattern_;
  }
  // The input column count of each kind, and the input names in column order.
  [[nodiscard]] const Inputs &inputs() const { return inputs_; }
  [[nodiscard]] const std::vector<std::string> &names() const { return names_; }

  // The extern functions, shared with the arena and called by index.
  [[nodiscard]] std::span<const ExternEntry<T>> externs() const {
    return externs_;
  }
  // The operand lists of n-ary nodes, shared with the arena.
  [[nodiscard]] std::span<const NodeId> lists() const { return lists_; }
  // Whether any live node is a tensor op, which needs a second tape.
  [[nodiscard]] bool has_tensors() const noexcept { return has_tensors_; }
  // Where each tensor node's rows sit in the tensor tape.
  [[nodiscard]] const TensorLayout &tensors() const noexcept {
    return tensors_;
  }
  // A node's rows x cols; scalars, and every node of a graph with no tensor,
  // report the default shape.
  [[nodiscard]] Shape shape(NodeId v) const {
    return shapes_.empty() ? Shape{} : shapes_[v];
  }
  // The step a kernel computes node v inside, or no_node when v is its own row.
  [[nodiscard]] NodeId fused_into(NodeId v) const {
    return fused_.empty() ? no_node : fused_[v];
  }
  // For each M v or M^T v product, its scheduled partner, else no_node.
  [[nodiscard]] std::span<const NodeId> pairs() const { return paired_; }

  // The total number of input columns, which can exceed the symbol count.
  [[nodiscard]] std::size_t arity() const { return inputs_.total(); }

  [[nodiscard]] bool live(NodeId v) const { return live_[v]; }

  [[nodiscard]] std::size_t live_count() const { return live_order_.size(); }

  // The steps to compute in topological order, with FMA contraction already
  // decided. Multiplies absorbed into an FMA are omitted.
  [[nodiscard]] std::span<const Step> schedule() const { return schedule_; }

  [[nodiscard]] PerBlock<std::span<const NodeId>> output_blocks() const {
    return blocks_of(std::span<const NodeId>{outputs_});
  }
  [[nodiscard]] OutputRows output_rows() const {
    return blocks_of(std::span<const TensorRow>{output_rows_});
  }
  // Each scheduled node's tape row, and the row count. Rows are packed by
  // liveness, so the tape is much smaller than the arena.
  [[nodiscard]] std::span<const TapeRow> rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t tape_rows() const noexcept { return tape_rows_; }

  [[nodiscard]] OpCode op_of(NodeId v) const { return properties_[v].op; }

  // Operands in slot order; only a select uses all three.
  [[nodiscard]] std::array<NodeId, 3> operands(NodeId v) const {
    return operands_[v];
  }
  [[nodiscard]] ListOffset list_of(NodeId v) const {
    return properties_[v].list;
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

  // Schedules M v and M^T v next to each other so that a kernel reads the
  // matrix once for both, as a quadratic form's value and gradient need.
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

  // Marks the nodes reachable from the outputs.
  void mark_live() {
    live_ = detail::reachable(*this, outputs_);
    live_order_.reserve(
        static_cast<std::size_t>(std::ranges::count(live_, true)));
    impl::append(live_order_, detail::live_ids(live_));
  }

  // Recomputes liveness with FMA contraction, so a multiply read only by
  // contracted adds is dropped from the schedule.
  void contract(Contract contract) {
    if (contract == Contract::off) {
      schedule_ = detail::schedule_of(*this, live_order_, Contract::off);
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
  std::vector<TapeRow> rows_;
  std::size_t tape_rows_ = 0;
  std::vector<bool> live_;
  std::vector<NodeId> live_order_;
  std::vector<Step> schedule_;
};

// Static counts of the steps a call executes and the values they write,
// taken from a frozen graph. A call's cost is proportional to values.
struct Census {
  struct Row {
    OpCode op = OpCode::Const;
    std::size_t steps = 0;
    std::size_t values = 0;
    friend bool operator==(const Row &, const Row &) = default;
  };
  std::size_t steps = 0;
  std::size_t values = 0;
  // Per-opcode counts, sorted by values written, largest first.
  std::vector<Row> by_op;
};

namespace detail {
template <impl::Numeric T> [[nodiscard]] Census census_of(const Graph<T> &g) {
  std::array<Census::Row, op_count> rows{};
  Census out{.steps = g.schedule().size(), .values = 0, .by_op = {}};
  for (const Step &s : g.schedule()) {
    Census::Row &r = rows[std::to_underlying(s.op)];
    const std::size_t values = g.shape(s.node).size();
    r.op = s.op;
    ++r.steps;
    r.values += values;
    out.values += values;
  }
  out.by_op = rows |
              std::views::filter([](const Census::Row &r) { return r.steps; }) |
              impl::to<std::vector>();
  std::ranges::stable_sort(out.by_op, std::ranges::greater{},
                           &Census::Row::values);
  return out;
}
} // namespace detail

} // namespace ddx::rt

DDX_EXTERN_INSTANCE(class DDX_INSTANCE_API ::ddx::rt::Graph<double>)
DDX_EXTERN_INSTANCE(class DDX_INSTANCE_API ::ddx::rt::Graph<float>)
