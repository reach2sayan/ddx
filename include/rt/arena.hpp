#pragma once

#include "ops/algebra.hpp"
#include "ops/scalar.hpp"
#include "rt/apply.hpp"
#include "rt/kernels.hpp"
#include "rt/opcode.hpp"
#include "util/config.hpp"
#include "util/error.hpp"
#include "util/export.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstdint>
#include <format>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ddx::rt {

// A typed index that reads as a plain index but is only made explicitly, so
// one kind of index, or a count, never becomes another by accident.
template <typename Tag> struct Id {
  using difference_type = std::int64_t;
  std::uint32_t value = 0;

  constexpr Id() = default;
  // Ids are built from sizes and enumerate() indices; narrow here, once.
  constexpr explicit Id(std::convertible_to<std::uint32_t> auto v) noexcept
      : value{static_cast<std::uint32_t>(v)} {}
  constexpr operator std::uint32_t() const noexcept { return value; }
  constexpr Id &operator++() noexcept {
    ++value;
    return *this;
  }
  constexpr Id operator++(int) noexcept {
    const Id was = *this;
    ++value;
    return was;
  }
  constexpr Id &operator--() noexcept {
    --value;
    return *this;
  }
  constexpr Id operator--(int) noexcept {
    const Id was = *this;
    --value;
    return was;
  }
  friend constexpr bool operator==(Id, Id) = default;
  friend constexpr auto operator<=>(Id, Id) = default;
};
using NodeId = Id<struct NodeTag>;
// A scalar's row in a sweep's tape: the node id in an arena walk, or a row
// packed by liveness in a frozen graph.
using TapeRow = Id<struct TapeRowTag>;
// Where a node's list starts in Model::lists.
using ListOffset = Id<struct ListTag>;
static_assert(sizeof(NodeId) == 4 && std::is_trivially_copyable_v<NodeId>);
static_assert(std::weakly_incrementable<NodeId>);
inline constexpr NodeId no_node{~std::uint32_t{0}};
inline constexpr TapeRow no_row{~std::uint32_t{0}};
inline constexpr ListOffset no_list{~std::uint32_t{0}};

} // namespace ddx::rt

template <typename Tag>
struct std::formatter<ddx::rt::Id<Tag>> : std::formatter<std::uint32_t> {
  auto format(ddx::rt::Id<Tag> id, auto &ctx) const {
    return std::formatter<std::uint32_t>::format(id.value, ctx);
  }
};

namespace ddx::rt {
namespace detail {
// A tensor op's or a select's operands under their table names, in slot
// order, as make() takes them and operands() returns them.
template <OpCode Op> struct Operands;
// Keyed on arity, not __VA_OPT__, which MSVC's default preprocessor lacks.
#define DDX_RT_MEMBERS_0(...)
#define DDX_RT_MEMBERS_1(...) NodeId __VA_ARGS__;
#define DDX_RT_MEMBERS_2 DDX_RT_MEMBERS_1
#define DDX_RT_MEMBERS_3 DDX_RT_MEMBERS_1
#define DDX_RT_OPERANDS(Op, arity, names)                                      \
  template <> struct Operands<OpCode::Op> {                                    \
    DDX_RT_MEMBERS_##arity names                                               \
  };                                                                           \
  static_assert((arity) == 0 ||                                                \
                sizeof(Operands<OpCode::Op>) == (arity) * sizeof(NodeId));
#define DDX_RT_TENSOR_OPERANDS(fn, Op, label, arity, kind, payload, yields,    \
                               names)                                          \
  DDX_RT_OPERANDS(Op, arity, names)
#define DDX_RT_TERNARY_OPERANDS(fn, Op, label, functor, yields, names)         \
  DDX_RT_OPERANDS(Op, 3, names)
DDX_RT_TENSOR_TABLE(DDX_RT_TENSOR_OPERANDS)
DDX_RT_TERNARY_TABLE(DDX_RT_TERNARY_OPERANDS)
#undef DDX_RT_TERNARY_OPERANDS
#undef DDX_RT_TENSOR_OPERANDS
#undef DDX_RT_OPERANDS
#undef DDX_RT_MEMBERS_3
#undef DDX_RT_MEMBERS_2
#undef DDX_RT_MEMBERS_1
#undef DDX_RT_MEMBERS_0

template <OpCode Op>
[[nodiscard]] constexpr Operands<Op>
operands(const std::array<NodeId, 3> &in) noexcept {
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return Operands<Op>{in[I]...};
  }(std::make_index_sequence<arity_of(Op)>{});
}
// The operands of a node, which must be an `Op`.
template <OpCode Op>
[[nodiscard]] constexpr Operands<Op> operands(const auto &node) noexcept
  requires requires { node.in; }
{
  assert(node.op == Op);
  return operands<Op>(node.in);
}

// One term an `op=` fold has taken, indexing the arena's pending list.
using PendingEntry = Id<struct PendingTag>;
inline constexpr PendingEntry no_entry{~std::uint32_t{0}};
// A handle's fold in progress: its last entry, typed by the Nary op that its
// entries are read under.
template <OpCode Fold> struct Folding {
  PendingEntry entry;
};
} // namespace detail

template <impl::Numeric T> class Expression;
template <impl::Numeric T = double> class DDX_TEMPLATE_API Arena;

// The one primitive that names an input; var(), param() and implicit() all
// call it, and Arena befriends it.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> named(Arena<T> &b, OpCode op,
                                            std::string_view name);
template <impl::Numeric T, std::invocable<const Expression<T> &> F>
[[nodiscard]] constexpr Expression<T>
implicit(Arena<T> &b, std::string_view name, F &&residual);
namespace detail {
struct Implicits;
}

// A caller-supplied function: `fn(in, out)` writes its answer in ExternRecord's
// layout. It may evaluate another ddx equation; see TapeLease in rt/core.hpp.
template <impl::Numeric T>
using ExternFn = std::function<void(std::span<const T>, std::span<T>)>;

// What extern_fn() registers. The answer holds the value, the first partials,
// then the row-major upper triangle of the second partials when order is 2.
struct ExternRecord {
  std::string name;
  std::uint32_t arity = 1;
  std::uint32_t order = 2;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return arity >= 1 && order >= 1 && order <= 2;
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept {
    const std::size_t n = arity;
    return 1 + n + (order >= 2 ? n * (n + 1) / 2 : 0);
  }
  [[nodiscard]] constexpr std::uint32_t first(std::size_t l) const noexcept {
    return static_cast<std::uint32_t>(1 + l);
  }
  [[nodiscard]] constexpr std::uint32_t second(std::size_t l,
                                               std::size_t m) const noexcept {
    if (m < l) {
      std::swap(l, m);
    }
    return static_cast<std::uint32_t>(1 + arity + l * arity - l * (l - 1) / 2 +
                                      (m - l));
  }
  // The component holding d/dx_l of `component`: a first partial of the value
  // or a second partial of a first. Nothing is defined past second order.
  [[nodiscard]] constexpr std::optional<std::uint32_t>
  up(std::uint32_t component, std::size_t l) const noexcept {
    if (component == 0) {
      return first(l);
    }
    if (component <= arity && order >= 2) {
      return second(component - 1, l);
    }
    return std::nullopt;
  }
};

template <impl::Numeric T> struct ExternEntry : ExternRecord {
  std::shared_ptr<const ExternFn<T>> fn;
};

template <impl::Numeric T> class ExternFunction;
template <impl::Numeric T>
[[nodiscard]] ExternFunction<T> extern_fn(Arena<T> &b, ExternRecord record,
                                          std::type_identity_t<ExternFn<T>> fn);
namespace detail {
template <impl::Numeric T, typename Sink>
constexpr void extern_partials(Arena<T> &b, NodeId v, Sink &&sink);
}

// A named input: its kind of leaf and its slot among that kind.
struct Input {
  OpCode op = OpCode::Var;
  Slot index{};
  constexpr bool operator==(const Input &) const = default;
};

// Befriended by Arena for symmetry with var(); a seed slot moves nothing.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> seed(Arena<T> &b, std::uint32_t slot);

// A multiply and add rounded once as x * y + z, with x negated when the product
// reached the add through a Neg, as a subtraction builds it.
struct Contraction {
  NodeId x = no_node;
  NodeId y = no_node;
  NodeId z = no_node;
  bool negated = false;
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return x != no_node;
  }
};

// One node of a sweep with its fma, if any. `out` and `in` are its tape rows,
// and `op` sits in padding so the sweep dispatches without loading the node.
struct Step {
  NodeId node = no_node;
  Contraction fma{};
  TapeRow out = no_row;
  std::array<TapeRow, 3> in{no_row, no_row, no_row};
  bool fused = false; // Set on a SumN whose product terms contract to fma.
  OpCode op = OpCode::Const;
};
static_assert(sizeof(Step) == 40);

namespace detail {

template <typename P> [[nodiscard]] constexpr P get(const auto &node) noexcept {
  assert(info(node.op).payload == payload_index<P>);
  return from_word<P>(node.payload);
}

[[nodiscard]] constexpr OpCode effective_op(const auto &node) noexcept {
  return node.op == OpCode::Map || node.op == OpCode::Zip ? get<OpCode>(node)
                                                          : node.op;
}

[[nodiscard]] constexpr bool is_tensor(const auto &node) noexcept {
  return yields_of(node.op) == Yields::Row;
}

[[nodiscard]] constexpr bool is_scalar(const auto &node) noexcept {
  return yields_of(node.op) == Yields::Scalar;
}

[[nodiscard]] constexpr bool on_tensors(const auto &node) noexcept {
  return kind_of(node.op) == Kind::Tensor || is_tensor(node);
}

[[nodiscard]] constexpr auto operands_of(const auto &nodes, NodeId v) {
  return nodes.operands(v) |
         std::views::filter([](NodeId u) { return u != no_node; });
}

} // namespace detail

// Unused operands are no_node; a node with a list uses `list` instead. The
// payload is a word that detail::get reads as the op's payload type.
template <impl::Numeric T> struct Node {
  OpCode op = OpCode::Const;
  std::array<NodeId, 3> in{no_node, no_node, no_node};
  T value{}; // Only a Const uses this.
  std::uint32_t payload = 0;
  ListOffset list = no_list;
};

// Pinned, since a new member would silently widen every node in every arena.
static_assert(sizeof(Node<double>) == 32);

template <impl::Numeric T> struct TensorConst {
  std::uint32_t rows = 0;
  std::uint32_t cols = 1;
  std::vector<T> data;
  [[nodiscard]] constexpr bool valid() const noexcept {
    return rows >= 1 && cols >= 1 && data.size() == std::size_t{rows} * cols;
  }
};

// Everything that makes up a model, as the Arena holds it.
template <impl::Numeric T> struct Model {
  // Each named kind's names in declaration order, which is slot order.
  PerKind<std::vector<std::string>> names;
  std::vector<NodeId> closures; // The residual each implicit is a root of.
  std::vector<ExternEntry<T>> externs;
  std::vector<TensorConst<T>> tensors;
  std::vector<NodeId> lists; // Every list node's ids, end to end.
  std::vector<Node<T>> nodes;

  // The CNodeSource interface, which the frozen Graph also provides.
  [[nodiscard]] constexpr const Node<T> &operator[](NodeId v) const {
    return nodes[v];
  }
  [[nodiscard]] constexpr OpCode op_of(NodeId v) const { return nodes[v].op; }
  [[nodiscard]] constexpr std::array<NodeId, 3> operands(NodeId v) const {
    return nodes[v].in;
  }
  [[nodiscard]] constexpr ListOffset list_of(NodeId v) const {
    return nodes[v].list;
  }
  [[nodiscard]] constexpr std::size_t size() const { return nodes.size(); }
};

// The ids a list node reads, in order: an extern's arguments, or a Pack's,
// StackRows' or sum's items, as many as the payload's count.
[[nodiscard]] constexpr std::span<const NodeId> arguments(const auto &s,
                                                          NodeId v) {
  const OpCode op = s.op_of(v);
  const std::size_t count =
      is_extern(op) ? s.externs()[detail::get<ExternRef>(s[v]).function].arity
                    : detail::get<Count>(s[v]).value;
  return s.lists().subspan(s.list_of(v), count);
}

namespace detail {
// Calls `f` on every id a node reads, its list or its operands. Every
// reachability walk uses this one definition of a node's children.
constexpr void for_each_operand(const auto &s, NodeId v, auto &&f) {
  if (has_list(s.op_of(v))) {
    std::ranges::for_each(arguments(s, v), f);
  } else {
    std::ranges::for_each(operands_of(s, v), f);
  }
}

inline constexpr auto every_operand = [](const auto &s, NodeId v, auto &&mark) {
  for_each_operand(s, v, mark);
};

// Marks the roots' cone, walking ids downward; `reaches` marks what a node
// reads.
template <typename Reach = decltype(every_operand)>
[[nodiscard]] constexpr std::vector<bool>
reachable(const auto &nodes, std::span<const NodeId> roots,
          Reach reaches = every_operand) {
  std::vector live(nodes.size(), false);
  std::ranges::for_each(roots, [&](NodeId r) { live[r] = true; });
  for (NodeId v = static_cast<NodeId>(nodes.size()); v-- > 0;) {
    if (live[v]) {
      reaches(nodes, v, [&](NodeId u) { live[u] = true; });
    }
  }
  return live;
}
} // namespace detail

// A row's shape as rows x cols, column-major, with cols 1 for a vector.
struct Shape {
  std::uint32_t rows = 1;
  std::uint32_t cols = 1;
  [[nodiscard]] constexpr std::uint32_t size() const noexcept {
    return rows * cols;
  }
  constexpr bool operator==(const Shape &) const = default;
};
template <impl::Numeric T>
[[nodiscard]] constexpr Shape shape_of(const Arena<T> &b, NodeId v) {
  return b.shape(v);
}
// A tensor node's arithmetic through a kernel table, writing a row to `to` or
// a reduction to `reduced`. Returns false for non-arithmetic nodes.
template <impl::Numeric U>
DDX_ALWAYS_INLINE constexpr bool tensor_row(const auto &s, NodeId v, auto in,
                                            auto scalar, std::span<U> to,
                                            U &reduced, const auto &kt) {
  const auto &n = s[v];
  switch (n.op) {
  case OpCode::Map:
    kt.map(detail::get<OpCode>(n), in(0), to);
    break;
  case OpCode::Zip:
    kt.zip(detail::get<OpCode>(n), in(0), in(1), to);
    break;
  case OpCode::Scale:
    kt.scale(scalar(0), in(1), to);
    break;
  case OpCode::Sum:
    reduced = kt.sum(in(0));
    break;
  case OpCode::Dot:
    reduced = kt.dot(in(0), in(1));
    break;
  case OpCode::Gather:
    reduced = in(0)[detail::get<Element>(n).value];
    break;
  case OpCode::AddAt:
    std::ranges::copy(in(0), to.begin());
    to[detail::get<Element>(n).value] =
        to[detail::get<Element>(n).value] + scalar(1);
    break;
  // The matrix ops below; a matrix is column-major.
  case OpCode::Diag: {
    const auto d = in(0);
    std::ranges::fill(to, U{});
    for (std::size_t i = 0; i < d.size(); ++i) {
      to[i + i * d.size()] = d[i];
    }
    break;
  }
  case OpCode::Outer: {
    const auto u = in(0);
    const auto g = in(1);
    for (std::size_t j = 0; j < g.size(); ++j) {
      kt.scale(g[j], u, to.subspan(j * u.size(), u.size()));
    }
    break;
  }
  case OpCode::RowScale: { // diag(a) M scales every column by a.
    const auto d = in(0);
    const auto m = in(1);
    for (std::size_t j = 0; j < m.size() / d.size(); ++j) {
      kt.zip(OpCode::Mul, d, m.subspan(j * d.size(), d.size()),
             to.subspan(j * d.size(), d.size()));
    }
    break;
  }
  case OpCode::ColScale: { // M diag(b) scales column j by b[j].
    const auto m = in(0);
    const auto d = in(1);
    const std::size_t rows = m.size() / d.size();
    for (std::size_t j = 0; j < d.size(); ++j) {
      kt.scale(d[j], m.subspan(j * rows, rows), to.subspan(j * rows, rows));
    }
    break;
  }
  case OpCode::MatVec: {
    const Shape m = shape_of(s, n.in[0]);
    kt.gemv(m.rows, m.cols, in(0), in(1), to,
            detail::get<Orientation>(n) == Orientation::transposed);
    break;
  }
  case OpCode::MatMul: {
    const Shape l = shape_of(s, n.in[0]);
    kt.gemm(l.rows, l.cols, shape_of(s, n.in[1]).cols, in(0), in(1), to);
    break;
  }
  case OpCode::RowOf: {
    const auto from = in(0);
    const std::size_t rows = shape_of(s, n.in[0]).rows;
    for (std::size_t j = 0; j < to.size(); ++j) {
      to[j] = from[detail::get<Element>(n).value + j * rows];
    }
    break;
  }
  default:
    return false;
  }
  return true;
}
// Whether a node is a MatRow over a constant table, in either orientation.
[[nodiscard]] constexpr bool is_table(const auto &s, NodeId v) {
  return s.op_of(v) == OpCode::MatRow &&
         s.op_of(s.operands(v)[0]) == OpCode::MatConst;
}
// How many scalars a node's value holds: one for a scalar or a reduction,
// rows x cols for a row, and zero for a MatConst table.
[[nodiscard]] constexpr std::uint32_t width_of(const auto &s, NodeId v) {
  return shape_of(s, v).size();
}

// Nodes are interned as they are formed, so equal subexpressions share an id.
template <impl::Numeric T> class DDX_TEMPLATE_API Arena {
public:
  using value_type = T;
  [[nodiscard]] constexpr NodeId constant(const T &v) {
    return intern({.op = OpCode::Const, .value = v});
  }

  // Reserves room for `nodes` nodes in total, in the arena and its hash table.
  constexpr void reserve(std::size_t nodes, std::size_t symbols = 0) {
    model_.nodes.reserve(nodes);
    shapes_.reserve(nodes);
    if (2 * nodes > table_.size()) {
      rehash(nodes);
    }
    const auto k = *named_index(OpCode::Var);
    by_name_.reserve(by_name_.size() + symbols -
                     std::min(symbols, model_.names[k].size()));
    model_.names[k].reserve(symbols);
    ids_[k].reserve(symbols);
  }

  // A VecConst for one column of `data`, a MatConst otherwise. Interned by
  // content where T compares, so a matrix given twice is kept once.
  [[nodiscard]] constexpr NodeId tensor_const(std::uint32_t rows,
                                              std::uint32_t cols,
                                              std::span<const T> data) {
    assert(data.size() == std::size_t{rows} * cols);
    auto at = model_.tensors.end();
    if constexpr (std::equality_comparable<T>) {
      at = std::ranges::find_if(model_.tensors, [&](const TensorConst<T> &t) {
        return t.rows == rows && t.cols == cols &&
               std::ranges::equal(t.data, data, identical);
      });
    }
    if (at == model_.tensors.end()) {
      model_.tensors.push_back(
          {.rows = rows, .cols = cols, .data = {data.begin(), data.end()}});
      at = std::prev(model_.tensors.end());
    }
    const TensorRef ref{
        static_cast<std::uint32_t>(std::distance(model_.tensors.begin(), at))};
    return cols == 1 ? make(opcode<OpCode::VecConst>, ref)
                     : make(opcode<OpCode::MatConst>, ref);
  }
  // A row of `shape` holding v everywhere; a matrix is its table's MatRow.
  [[nodiscard]] constexpr NodeId filled(Shape shape, const T &v) {
    const NodeId t =
        tensor_const(shape.rows, shape.cols, std::vector<T>(shape.size(), v));
    return shape.cols == 1
               ? t
               : make(opcode<OpCode::MatRow>, Orientation::plain, t);
  }
  [[nodiscard]] constexpr NodeId zeros(Shape shape) {
    return filled(shape, T{0});
  }
  [[nodiscard]] constexpr NodeId ones(Shape shape) {
    return filled(shape, T{1});
  }
  // The vector of `items`, or with StackRows the matrix whose rows they are.
  [[nodiscard]] constexpr NodeId pack(std::span<const NodeId> items,
                                      OpCode op = OpCode::Pack) {
    assert(op == OpCode::Pack || op == OpCode::StackRows);
    return listed(op, items);
  }

  // An `Op` node, with its arity and payload type checked at compile time.
  template <OpCode Op, std::same_as<NodeId>... Ids>
    requires(sizeof...(Ids) == arity_of(Op) && !named_operands(Op))
  [[nodiscard]] constexpr NodeId make(std::integral_constant<OpCode, Op>,
                                      payload_of_t<Op> payload, Ids... in) {
    return formed({.op = Op, .in = padded(in...), .payload = to_word(payload)});
  }
  // An op with named operands, given as `{.matrix = m, .vec = v}`, which only
  // compiles in slot order.
  template <OpCode Op>
    requires(named_operands(Op))
  [[nodiscard]] constexpr NodeId make(std::integral_constant<OpCode, Op>,
                                      payload_of_t<Op> payload,
                                      const detail::Operands<Op> &in) {
    const auto ids = std::bit_cast<std::array<NodeId, arity_of(Op)>>(in);
    std::array padded_in{no_node, no_node, no_node};
    std::ranges::copy(ids, padded_in.begin());
    return formed({.op = Op, .in = padded_in, .payload = to_word(payload)});
  }
  template <OpCode Op>
    requires(named_operands(Op) && std::same_as<payload_of_t<Op>, NoPayload>)
  [[nodiscard]] constexpr NodeId make(std::integral_constant<OpCode, Op> op,
                                      const detail::Operands<Op> &in) {
    return make(op, NoPayload{}, in);
  }
  // A list-carrying `Op` node over the list at offset `list`.
  template <OpCode Op>
    requires(has_list(Op))
  [[nodiscard]] constexpr NodeId make(std::integral_constant<OpCode, Op>,
                                      payload_of_t<Op> payload,
                                      ListOffset list) {
    return formed({.op = Op, .payload = to_word(payload), .list = list});
  }
  // An op chosen at run time, which carries no payload.
  [[nodiscard]] constexpr NodeId make(OpCode op, NodeId a, NodeId b = no_node,
                                      NodeId c = no_node) {
    assert(detail::info(op).payload == payload_index<NoPayload>);
    assert(!named_operands(op) && "named operands are made by name");
    return formed({.op = op, .in = {a, b, c}});
  }

  // The same node over remapped operands or a remapped list.
  [[nodiscard]] constexpr NodeId rebuild(NodeId v, auto &&remap) {
    Node<T> n = model_[v]; // A copy, since list() and formed() may reallocate.
    if (has_list(n.op)) {
      n.list = list(arguments(*this, v) | std::views::transform(remap) |
                    impl::to<std::vector<NodeId>>());
    } else {
      std::ranges::transform(n.in, n.in.begin(), [&](NodeId u) {
        return u == no_node ? no_node : remap(u);
      });
    }
    return formed(n);
  }
  // The same node with operand `slot` replaced. rebuild() maps by value, so it
  // would replace both operands of a Dot(x, x).
  [[nodiscard]] constexpr NodeId replaced(NodeId v, std::size_t slot,
                                          NodeId u) {
    Node<T> n = model_[v];
    n.in[slot] = u;
    return formed(n);
  }

  [[nodiscard]] constexpr const Node<T> &operator[](NodeId id) const {
    return model_[id];
  }
  [[nodiscard]] constexpr OpCode op_of(NodeId id) const {
    return model_.op_of(id);
  }
  [[nodiscard]] constexpr std::array<NodeId, 3> operands(NodeId id) const {
    return model_.operands(id);
  }
  [[nodiscard]] constexpr ListOffset list_of(NodeId id) const {
    return model_.list_of(id);
  }
  [[nodiscard]] constexpr std::size_t size() const { return model_.size(); }
  [[nodiscard]] constexpr Shape shape(NodeId v) const {
    assert(v < shapes_.size());
    return shapes_[v];
  }
  [[nodiscard]] constexpr std::span<const Node<T>> nodes() const {
    return model_.nodes;
  }
  // A kind's names in declaration order, which is the order of a positional
  // point and of every result.
  [[nodiscard]] constexpr const std::vector<std::string> &
  names(OpCode op) const {
    return model_.names[*named_index(op)];
  }
  [[nodiscard]] constexpr const std::vector<std::string> &symbols() const {
    return names(OpCode::Var);
  }
  [[nodiscard]] constexpr const std::vector<std::string> &params() const {
    return names(OpCode::Param);
  }
  [[nodiscard]] constexpr const std::vector<std::string> &implicits() const {
    return names(OpCode::Implicit);
  }
  // Every named input in column order: kinds in table order, each in
  // declaration order.
  [[nodiscard]] constexpr std::vector<std::string> names() const {
    return model_.names | std::views::join |
           impl::to<std::vector<std::string>>();
  }
  // Each implicit's residual by slot, or no_node if never closed. broken() is
  // the first misuse of implicit(), which the equation then refuses with.
  [[nodiscard]] constexpr std::span<const NodeId> closures() const {
    return model_.closures;
  }
  [[nodiscard]] constexpr std::optional<errc> broken() const { return broken_; }
  [[nodiscard]] constexpr std::span<const ExternEntry<T>> externs() const {
    return model_.externs;
  }
  [[nodiscard]] constexpr std::span<const TensorConst<T>> tensors() const {
    return model_.tensors;
  }
  [[nodiscard]] constexpr std::span<const NodeId> lists() const {
    return model_.lists;
  }
  [[nodiscard]] constexpr Inputs inputs() const {
    Inputs in;
    std::ranges::transform(model_.names, in.by_kind.begin(),
                           &std::vector<std::string>::size);
    return in;
  }

  [[nodiscard]] constexpr NodeId leaf(OpCode op, Slot slot) const {
    return ids_[*named_index(op)][slot.value];
  }

  [[nodiscard]] constexpr std::optional<Input>
  lookup(std::string_view name) const {
    const auto at = std::ranges::lower_bound(by_name_, name, {}, name_of());
    return at != by_name_.end() && name_of()(*at) == name ? std::optional{*at}
                                                          : std::nullopt;
  }
  // The slot of the symbol named `name`; params and implicits are not symbols.
  [[nodiscard]] constexpr result<std::uint32_t>
  symbol(std::string_view name) const {
    const auto in = lookup(name);
    return in && in->op == OpCode::Var ? result<std::uint32_t>{in->index.value}
                                       : fail(errc::unknown_symbol);
  }

  // Whether a node is a Const of `v` or a VecConst holding only `v`.
  [[nodiscard]] constexpr bool is_constant(NodeId id, const T &v) const {
    const Node<T> &n = model_.nodes[id];
    if constexpr (std::equality_comparable<T>) {
      using impl::same_value;
      if (n.op == OpCode::Const) {
        return same_value(n.value, v);
      }
      return n.op == OpCode::VecConst &&
             std::ranges::all_of(
                 model_.tensors[detail::get<TensorRef>(n).value].data,
                 [&v](const T &x) { return same_value(x, v); });
    } else {
      return false;
    }
  }

private:
  // All named inputs share one namespace: renaming returns the same leaf, a new
  // name takes its kind's next slot, and a name of another kind gives nullopt.
  template <impl::Numeric U>
  friend constexpr Expression<U> named(Arena<U> &, OpCode, std::string_view);
  [[nodiscard]] constexpr std::optional<NodeId> input(OpCode op,
                                                      std::string_view name) {
    const auto at = std::ranges::lower_bound(by_name_, name, {}, name_of());
    if (at != by_name_.end() && name_of()(*at) == name) {
      return at->op == op ? std::optional{leaf(op, at->index)} : std::nullopt;
    }
    const auto k = *named_index(op);
    const Slot index{static_cast<std::uint32_t>(model_.names[k].size())};
    model_.names[k].emplace_back(name);
    ids_[k].push_back(intern({.op = op, .payload = to_word(index)}));
    by_name_.insert(at, {.op = op, .index = index});
    if (op == OpCode::Implicit) {
      model_.closures.push_back(no_node);
      closure_rows_.emplace_back();
    }
    assert(ids_[k].size() == model_.names[k].size());
    assert(model_.closures.size() == closure_rows_.size());
    return ids_[k].back();
  }

  // Records `f` as implicit z's residual. A misuse is kept in broken_ rather
  // than thrown, so the equation over this arena refuses with it.
  template <impl::Numeric U, std::invocable<const Expression<U> &> F>
  friend constexpr Expression<U> implicit(Arena<U> &, std::string_view, F &&);
  constexpr void close(NodeId z, std::expected<NodeId, errc> f) {
    NodeId &closure = model_.closures[detail::get<Slot>(model_.nodes[z]).value];
    if (!f || closure != no_node) {
      broken_ = broken_.value_or(f ? errc::closed_twice : f.error());
    } else {
      closure = *f;
    }
  }

  // Only extern_fn() registers an extern; the returned index names it in calls.
  template <impl::Numeric U>
  friend ExternFunction<U> extern_fn(Arena<U> &, ExternRecord,
                                     std::type_identity_t<ExternFn<U>>);
  constexpr std::optional<std::uint32_t> add_extern(ExternEntry<T> e) {
    if (!ExternRef::fits(model_.externs.size(), e.size())) {
      return std::nullopt;
    }
    model_.externs.push_back(std::move(e));
    return static_cast<std::uint32_t>(model_.externs.size() - 1);
  }

  // The call node for the function's value; sweeps ask for other components.
  template <impl::Numeric U> friend class ExternFunction;
  constexpr NodeId extern_call(std::uint32_t function,
                               std::span<const NodeId> args) {
    return make(opcode<OpCode::Extern>, ExternRef{.function = function},
                list(args));
  }

  // The pool offset of `ids`, appending them if new. Lists are interned by
  // content like nodes, so an offset identifies a list.
  [[nodiscard]] constexpr ListOffset list(std::span<const NodeId> ids) {
    if ((listed_.size() + 1) * 2 > list_table_.size()) {
      rehash_lists(listed_.size() + 1);
    }
    const auto spells = [this, ids](std::uint32_t k) {
      const ListKey key = listed_[k];
      return key.size == ids.size() &&
             std::ranges::equal(
                 std::span{model_.lists}.subspan(key.at, key.size), ids);
    };
    auto run = probe(hash_ids(ids), list_table_.size());
    const std::size_t i = *std::ranges::find_if(run, [&](std::size_t k) {
      return list_table_[k] == kFreeList || spells(list_table_[k]);
    });
    if (list_table_[i] != kFreeList) {
      return listed_[list_table_[i]].at;
    }
    const ListOffset at{static_cast<std::uint32_t>(model_.lists.size())};
    impl::append(model_.lists, ids);
    list_table_[i] = static_cast<std::uint32_t>(listed_.size());
    listed_.push_back(
        {.at = at, .size = static_cast<std::uint32_t>(ids.size())});
    return at;
  }
  constexpr void rehash_lists(std::size_t lists) {
    list_table_.assign(std::bit_ceil(std::max<std::size_t>(64, 2 * lists)),
                       kFreeList);
    for (const auto [k, key] : listed_ | std::views::enumerate) {
      auto run =
          probe(hash_ids(std::span{model_.lists}.subspan(key.at, key.size)),
                list_table_.size());
      list_table_[*std::ranges::find_if(run, [this](std::size_t j) {
        return list_table_[j] == kFreeList;
      })] = static_cast<std::uint32_t>(k);
    }
  }

  // An `op=` fold's terms, as a list sharing its prefix with copied handles so
  // each copy keeps its value. The fold becomes a node only when first read.
  template <impl::Numeric U> friend class Expression;
  struct Pending {
    NodeId term = no_node;
    detail::PendingEntry prev = detail::no_entry; // None for the first term.
    NodeId node = no_node; // The node this entry materialised as.
    // Whether `node` is bit-exactly the fold's running value here, so that a
    // later read may resume from it.
    bool running = false;
  };
  std::vector<Pending> pending_;
  [[nodiscard]] constexpr detail::PendingEntry
  push_term(detail::PendingEntry prev, NodeId term) {
    pending_.push_back({.term = term, .prev = prev});
    return detail::PendingEntry{
        static_cast<std::uint32_t>(pending_.size() - 1)};
  }
  template <OpCode Fold>
  [[nodiscard]] constexpr NodeId fold_of(detail::Folding<Fold> folding) {
    const detail::PendingEntry entry = folding.entry;
    if (pending_[entry].node != no_node) {
      return pending_[entry].node;
    }
    // Walk back to the start or to an earlier running value, so a loop reading
    // its accumulator every step lists each term only once.
    std::vector<NodeId> terms;
    bool resumed = false;
    for (auto i = entry; i != detail::no_entry; i = pending_[i].prev) {
      if (i != entry && pending_[i].running) {
        terms.push_back(pending_[i].node);
        resumed = true;
        break;
      }
      terms.push_back(pending_[i].term);
    }
    std::ranges::reverse(terms);
    const NodeId node = terms.size() == 1 ? terms[0]
                        : terms.size() == 2
                            ? make(fold_op(Fold), terms[0], terms[1])
                            : listed(Fold, terms);
    // An Add contracts whichever operand is a product, so it is the running
    // value only when its first operand is not one, and a product never is.
    const auto product = [this](NodeId u) {
      return static_cast<bool>(product_of(*this, u));
    };
    pending_[entry].node = node;
    pending_[entry].running =
        Fold != OpCode::SumN ||
        (!product(node) &&
         (terms.size() != 2 || resumed || !product(terms[0])));
    return node;
  }
  // Another component of the extern call `v`: the same function and list.
  template <impl::Numeric U, typename Sink>
  friend constexpr void detail::extern_partials(Arena<U> &, NodeId, Sink &&);
  constexpr NodeId extern_component(NodeId v, std::uint32_t component) {
    const Node<T> n = model_[v]; // A copy, since intern() may reallocate.
    return make(opcode<OpCode::Extern>,
                ExternRef{.function = detail::get<ExternRef>(n).function,
                          .component = component},
                n.list);
  }

  // dz/ds for each implicit by the implicit-function theorem, built on first
  // reach by derivative.hpp and kept.
  friend struct detail::Implicits;
  std::vector<std::optional<std::vector<NodeId>>> closure_rows_;

  // A seed slot is given rather than looked up; interning keeps one node each.
  template <impl::Numeric U>
  friend constexpr Expression<U> seed(Arena<U> &, std::uint32_t);
  [[nodiscard]] constexpr NodeId seed_node(Slot slot) {
    return intern({.op = OpCode::Seed, .payload = to_word(slot)});
  }

  static constexpr std::uint64_t payload_of(const Node<T> &n) {
    if (n.op != OpCode::Const) {
      return std::uint64_t{n.payload};
    }
    // Anything wider than a word hashes to one bucket and is told apart by
    // same().
    if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 8) {
      return std::bit_cast<std::uint64_t>(n.value);
    } else if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 4) {
      return std::bit_cast<std::uint32_t>(n.value);
    } else {
      return 0;
    }
  }

  static constexpr bool same(const Node<T> &l, const Node<T> &r) {
    if (l.op != r.op || l.in != r.in || l.list != r.list) {
      return false;
    } else if (l.op != OpCode::Const) {
      return l.payload == r.payload;
    }
    return identical(l.value, r.value);
  }
  // Compares by bits where T is a word, as the hash does, so -0.0 and +0.0 stay
  // distinct and a NaN equals itself.
  static constexpr bool identical(const T &l, const T &r) {
    if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 8) {
      return std::bit_cast<std::uint64_t>(l) == std::bit_cast<std::uint64_t>(r);
    } else if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 4) {
      return std::bit_cast<std::uint32_t>(l) == std::bit_cast<std::uint32_t>(r);
    } else if constexpr (std::equality_comparable<T>) {
      using impl::same_value;
      return same_value(l, r);
    } else {
      return false;
    }
  }

  static constexpr void mix(std::uint64_t &h, std::uint64_t v) {
    constexpr std::uint64_t golden_ratio_64 = 0x9e3779b97f4a7c15ULL;
    constexpr std::uint64_t murmur_mix_64 = 0xff51afd7ed558ccdULL;
    h ^= v + golden_ratio_64 + (h << 6) + (h >> 2);
    h *= murmur_mix_64;
    h ^= h >> 33;
  }
  static constexpr std::size_t hash_of(const Node<T> &n) {
    std::uint64_t h = static_cast<std::uint64_t>(n.op);
    for (const NodeId u : n.in) {
      mix(h, u);
    }
    mix(h, payload_of(n));
    mix(h, n.list);
    return static_cast<std::size_t>(h);
  }
  static constexpr std::size_t hash_ids(std::span<const NodeId> ids) {
    std::uint64_t h = ids.size();
    for (const NodeId u : ids) {
      mix(h, u);
    }
    return static_cast<std::size_t>(h);
  }

  // Unbounded, since the table is at most half full and every run reaches an
  // empty slot.
  static constexpr auto probe(std::size_t h, std::size_t size) {
    return std::views::iota(h) |
           std::views::transform(
               [mask = size - 1](std::size_t k) { return k & mask; });
  }

  // Rebuilds the table for `nodes` nodes, at a power-of-two size (as the mask
  // needs) at most half full, where linear probing starts to degrade.
  constexpr void rehash(std::size_t nodes) {
    table_.assign(std::bit_ceil(std::max<std::size_t>(64, 2 * nodes)), no_node);
    for (const auto [id, n] : model_.nodes | std::views::enumerate) {
      auto run = probe(hash_of(n), table_.size());
      table_[*std::ranges::find_if(run, [this](std::size_t i) {
        return table_[i] == no_node;
      })] = static_cast<NodeId>(id);
    }
  }

  constexpr NodeId intern(const Node<T> &n) {
    if ((model_.nodes.size() + 1) * 2 > table_.size()) {
      rehash(model_.nodes.size() + 1);
    }
    // The probe stops at the node itself or at the empty slot it would take.
    auto run = probe(hash_of(n), table_.size());
    const std::size_t i = *std::ranges::find_if(run, [this, &n](std::size_t k) {
      return table_[k] == no_node || same(model_.nodes[table_[k]], n);
    });
    if (table_[i] != no_node) {
      return table_[i];
    }
    const auto id = static_cast<NodeId>(model_.nodes.size());
    model_.nodes.push_back(n);
    table_[i] = id;
    shapes_.push_back(shape_from(id));
    assert(shapes_.size() == model_.nodes.size());
    return id;
  }

  // A new node's shape from its operands' shapes. Elementwise ops take the
  // first operand's, except Scale, RowScale and VSelect, which take the second.
  [[nodiscard]] constexpr Shape shape_from(NodeId v) const {
    const auto &n = model_.nodes[v];
    const auto [a, b, c] = n.in;
    const auto table = [this](NodeId t) {
      const auto &m =
          model_.tensors[detail::get<TensorRef>(model_.nodes[t]).value];
      return Shape{.rows = m.rows, .cols = m.cols};
    };
    switch (n.op) {
    case OpCode::VecConst:
      return {.rows = table(v).rows, .cols = 1};
    case OpCode::MatConst:
      return {.rows = 0, .cols = 0};
    case OpCode::MatRow: {
      // A table, or a computed matrix such as the dense sweep's transpose.
      const Shape m =
          model_.nodes[a].op == OpCode::MatConst ? table(a) : shapes_[a];
      return detail::get<Orientation>(n) == Orientation::transposed
                 ? Shape{.rows = m.cols, .cols = m.rows}
                 : m;
    }
    case OpCode::Pack:
      return {.rows = detail::get<Count>(n).value, .cols = 1};
    case OpCode::Diag:
      return {.rows = shapes_[a].rows, .cols = shapes_[a].rows};
    case OpCode::Outer:
      return {.rows = shapes_[a].rows, .cols = shapes_[b].rows};
    case OpCode::MatVec:
      return {.rows = detail::get<Orientation>(n) == Orientation::transposed
                          ? shapes_[a].cols
                          : shapes_[a].rows,
              .cols = 1};
    case OpCode::MatMul:
      return {.rows = shapes_[a].rows, .cols = shapes_[b].cols};
    case OpCode::StackRows:
      return {.rows = detail::get<Count>(n).value,
              .cols = shapes_[arguments(*this, v)[0]].rows};
    case OpCode::RowOf:
      return {.rows = shapes_[a].cols, .cols = 1};
    case OpCode::Map:
    case OpCode::Zip:
    case OpCode::AddAt:
    case OpCode::ColScale:
      return shapes_[a];
    case OpCode::Scale:
    case OpCode::RowScale:
    case OpCode::VSelect:
      return shapes_[b];
    default:
      return {};
    }
  }

  // A scalar with no equality never matches, so ops/algebra.hpp's identity
  // rewrites never fire for it.
  constexpr bool holds(NodeId id, int k) const {
    return id != no_node && is_constant(id, T(k));
  }

  // The RuleOp of an opcode, or nullopt for one carrying no identities.
  static constexpr std::optional<impl::algebra::RuleOp>
  rule_op_of(OpCode op) noexcept {
    switch (op) {
    case OpCode::Add:
      return impl::algebra::RuleOp::Add;
    case OpCode::Mul:
      return impl::algebra::RuleOp::Mul;
    case OpCode::Div:
      return impl::algebra::RuleOp::Div;
    case OpCode::Pow:
      return impl::algebra::RuleOp::Pow;
    case OpCode::Neg:
      return impl::algebra::RuleOp::Neg;
    default:
      return std::nullopt;
    }
  }

  // The algebra rules' predicates, answered by id comparison, since interning
  // makes structural identity an id equality.
  constexpr bool holds_pred(impl::algebra::Pred p, NodeId a,
                            NodeId b) const noexcept {
    using impl::algebra::Pred;
    switch (p) {
    case Pred::ZeroA:
      return holds(a, 0);
    case Pred::ZeroB:
      return holds(b, 0);
    case Pred::OneA:
      return holds(a, 1);
    case Pred::OneB:
      return holds(b, 1);
    case Pred::Same:
      return a == b;
    case Pred::TwoB:
      return holds(b, 2);
    case Pred::AOverB:
      return cancel_quotient(a, b).has_value();
    case Pred::BOverA:
      return cancel_quotient(b, a).has_value();
    case Pred::NegatedA:
      return a != no_node && model_.nodes[a].op == OpCode::Neg;
    }
    std::unreachable();
  }

  constexpr std::optional<NodeId> apply_rule(const impl::algebra::Rule &r,
                                             NodeId a, NodeId b) {
    using impl::algebra::Take;
    switch (r.then) {
    case Take::LitZero:
      return constant(T{0});
    case Take::LitOne:
      return constant(T{1});
    case Take::OperandA:
      return a;
    case Take::OperandB:
      return b;
    case Take::NumeratorOfA:
      return cancel_quotient(a, b);
    case Take::NumeratorOfB:
      return cancel_quotient(b, a);
    case Take::OperandOfA:
      return model_.nodes[a].in[0];
    case Take::SquareOfA:
      // This cannot recurse into the rule, since Mul carries no Same identity.
      return make(OpCode::Mul, a, a);
    }
    return std::nullopt;
  }

  constexpr std::optional<NodeId> fold(OpCode op, NodeId a, NodeId b) {
    const bool ca = a != no_node && model_.nodes[a].op == OpCode::Const;
    const bool cb = b != no_node && model_.nodes[b].op == OpCode::Const;
    const bool unary = arity_of(op) == 1;

    // Constant arithmetic first, through the same *_impl functor that the
    // compile-time evaluator uses.
    if (unary && ca) {
      return constant(apply<T>(op, model_.nodes[a].value));
    } else if (!unary && ca && cb) {
      return constant(
          apply<T>(op, model_.nodes[a].value, model_.nodes[b].value));
    }

    const auto kind = rule_op_of(op);
    if (!kind) {
      return std::nullopt;
    }
    // The first match wins, in ops/algebra.hpp's order.
    const auto r = std::ranges::find_if(
        impl::algebra::kRules, [&](const impl::algebra::Rule &rule) {
          return rule.op == *kind &&
                 impl::algebra::arity_of(rule.op) == (unary ? 1uz : 2uz) &&
                 (!rule.needs_commutative_multiply ||
                  impl::CCommutativeMultiply<T>) &&
                 holds_pred(rule.when, a, b);
        });
    return r == std::ranges::cend(impl::algebra::kRules) ? std::nullopt
                                                         : apply_rule(*r, a, b);
  }

  [[nodiscard]] static constexpr std::array<NodeId, 3>
  padded(std::same_as<NodeId> auto... ids) {
    std::array<NodeId, 3> in{no_node, no_node, no_node};
    std::ranges::copy(std::array<NodeId, sizeof...(ids)>{ids...}, in.begin());
    return in;
  }

  // List and tensor nodes skip fold() but still get the commutative swap and
  // fold_tensor(). Where * commutes, M (s v) becomes s (M v) to share gemvs.
  [[nodiscard]] constexpr NodeId formed(Node<T> n) {
    auto &[a, b, c] = n.in;
    if (has_list(n.op) || kind_of(n.op) == Kind::Tensor) {
      const bool commutes =
          (n.op == OpCode::Zip && is_commutative<T>(detail::get<OpCode>(n))) ||
          (n.op == OpCode::Dot && is_commutative<T>(OpCode::Mul));
      if (commutes && b < a) {
        std::swap(a, b);
      }
      if (n.op == OpCode::Scale && holds(a, 1)) {
        return b;
      }
      if constexpr (impl::CCommutativeMultiply<T>) {
        if (n.op == OpCode::MatVec && model_.nodes[b].op == OpCode::Scale) {
          const auto [scalar, vec] =
              detail::operands<OpCode::Scale>(model_.nodes[b]);
          return make(
              opcode<OpCode::Scale>,
              {.scalar = scalar,
               .vec = make(opcode<OpCode::MatVec>, detail::get<Orientation>(n),
                           {.matrix = a, .vec = vec})});
        }
      }
      const NodeId id = intern(n);
      return fold_tensor(id).value_or(id);
    }
    if (kind_of(n.op) == Kind::Ternary) {
      if (const auto chosen = fold_select(a, b, c)) {
        return *chosen;
      }
    } else {
      if (const auto folded = fold(n.op, a, b)) {
        return *folded;
      }
      if (is_commutative<T>(n.op) && b != no_node && b < a) {
        std::swap(a, b);
      }
    }
    return intern(n);
  }

  // Folds a tensor node over constants into a table or a Const, using the
  // reference kernels. Skipped in constant evaluation.
  constexpr std::optional<NodeId> fold_tensor(NodeId v) {
    if consteval {
      return std::nullopt;
    }
    using K = reference::Kernels<T>;
    const Node<T> n = model_.nodes[v];
    const auto [a, b, c] = n.in;
    const auto scalar = [this](NodeId u) -> std::optional<T> {
      if (u == no_node || model_.nodes[u].op != OpCode::Const) {
        return std::nullopt;
      }
      return model_.nodes[u].value;
    };
    // A constant's elements in column-major order, transposing if needed.
    const auto elements = [this](NodeId u) -> std::optional<std::vector<T>> {
      if (u == no_node) {
        return std::nullopt;
      }
      const Node<T> &x = model_.nodes[u];
      if (x.op == OpCode::VecConst) {
        return model_.tensors[detail::get<TensorRef>(x).value].data;
      }
      if (x.op != OpCode::MatRow) {
        return std::nullopt;
      }
      const Node<T> &source =
          model_.nodes[detail::operands<OpCode::MatRow>(x).source];
      if (source.op != OpCode::MatConst) {
        return std::nullopt;
      }
      const TensorConst<T> &m =
          model_.tensors[detail::get<TensorRef>(source).value];
      if (detail::get<Orientation>(x) == Orientation::plain) {
        return m.data;
      }
      std::vector<T> t(m.data.size());
      for (std::size_t i = 0; i < m.rows; ++i) {
        for (std::size_t j = 0; j < m.cols; ++j) {
          t[j + i * m.cols] = m.data[i + j * m.rows];
        }
      }
      return t;
    };
    const Shape s = shape_of(*this, v);
    std::vector<T> out(s.size());
    const auto table = [&]() -> NodeId {
      const NodeId t = tensor_const(s.rows, s.cols, out);
      return s.cols == 1 ? t
                         : make(opcode<OpCode::MatRow>, Orientation::plain, t);
    };
    if (n.op == OpCode::Pack) {
      for (const auto [e, item] : arguments(*this, v) | std::views::enumerate) {
        const auto x = scalar(item);
        if (!x) {
          return std::nullopt;
        }
        out[static_cast<std::size_t>(e)] = *x;
      }
      return table();
    }
    if (has_list(n.op)) {
      return std::nullopt;
    }
    // Fold only when every operand is a constant.
    const auto u = elements(a);
    const auto w = elements(b);
    const auto su = scalar(a);
    const auto sw = scalar(b);
    if ((a != no_node && !u && !su) || (b != no_node && !w && !sw) ||
        c != no_node) {
      return std::nullopt;
    }
    T reduced{};
    const bool computed = tensor_row(
        *this, v, [&](int i) { return std::span<const T>{i == 0 ? *u : *w}; },
        [&](int i) { return i == 0 ? *su : *sw; }, std::span<T>{out}, reduced,
        K{});
    if (!computed) {
      return std::nullopt;
    }
    return detail::is_scalar(n) ? constant(reduced) : table();
  }

  // A Count-carrying node (a Pack, StackRows or fold) over `items`.
  [[nodiscard]] constexpr NodeId listed(OpCode op,
                                        std::span<const NodeId> items) {
    assert(!items.empty() && detail::info(op).payload == payload_index<Count>);
    return formed(
        {.op = op,
         .payload = to_word(Count{static_cast<std::uint32_t>(items.size())}),
         .list = list(items)});
  }

  // Folds a select with equal arms or a literal condition (nonzero, NaN too, is
  // true), so PowOpFn's 0 * inf guard costs nothing for a literal exponent.
  constexpr std::optional<NodeId> fold_select(NodeId cond, NodeId t,
                                              NodeId f) const {
    if (t == f) {
      return t;
    }
    if (model_.nodes[cond].op == OpCode::Const) {
      if constexpr (std::equality_comparable<T>) {
        return model_.nodes[cond].value != T{0} ? t : f;
      }
    }
    return std::nullopt;
  }

  // (n/d) * d -> n, where the denominator match is an id comparison.
  constexpr std::optional<NodeId> cancel_quotient(NodeId quotient,
                                                  NodeId factor) const {
    const Node<T> &q = model_.nodes[quotient];
    return q.op == OpCode::Div && q.in[1] == factor ? std::optional{q.in[0]}
                                                    : std::nullopt;
  }

  Model<T> model_;
  // Everything below indexes model_. The hash tables are hand-written because
  // interning runs in constant evaluation, where no library one is constexpr.
  std::vector<NodeId> table_; // Power-of-two size; no_node marks a free slot.
  // Each node's shape by id, recorded when intern() appended it.
  std::vector<Shape> shapes_;
  // The lists in the pool, and a table over them by content like table_.
  struct ListKey {
    ListOffset at;
    std::uint32_t size;
  };
  static constexpr std::uint32_t kFreeList = ~std::uint32_t{0};
  std::vector<ListKey> listed_;
  std::vector<std::uint32_t> list_table_;
  // Per named kind, the leaf node of each slot.
  PerKind<std::vector<NodeId>> ids_;
  // Every named input sorted by name, a hand-written flat map since no library
  // flat container is constexpr before C++26.
  std::vector<Input> by_name_;
  std::optional<errc> broken_;

  // The key by_name_ is sorted by.
  [[nodiscard]] constexpr auto name_of() const {
    return [this](const Input &in) -> std::string_view {
      return names(in.op)[in.index.value];
    };
  }
};

// An id-addressed node source, satisfied by both Arena and the frozen Graph so
// that walkers such as contraction_at() are written once.
template <typename S>
concept CNodeSource = requires(const S &s, NodeId v) {
  { s.op_of(v) } -> std::same_as<OpCode>;
  { s.operands(v) } -> std::same_as<std::array<NodeId, 3>>;
  { s.size() } -> std::convertible_to<std::size_t>;
};

// The product a term is, if any: a Mul or a negated Mul. `z` is left for the
// add it is fused into.
[[nodiscard]] constexpr Contraction product_of(const CNodeSource auto &nodes,
                                               NodeId u) {
  const bool negated = nodes.op_of(u) == OpCode::Neg;
  const NodeId m = negated ? nodes.operands(u)[0] : u;
  if (nodes.op_of(m) != OpCode::Mul) {
    return {};
  }
  const auto factors = nodes.operands(m);
  return {.x = factors[0], .y = factors[1], .negated = negated};
}

// fadd(fmul(x, y), z) -> fma(x, y, z), decided structurally so every backend
// agrees to the bit. With two products the lower-id operand wins.
[[nodiscard]] constexpr Contraction
contraction_at(const CNodeSource auto &nodes, NodeId v) {
  if (nodes.op_of(v) != OpCode::Add) {
    return {};
  }
  const auto ops = nodes.operands(v);
  for (const auto side : {0uz, 1uz}) {
    if (Contraction c = product_of(nodes, ops[side])) {
      c.z = ops[1 - side];
      return c;
    }
  }
  return {};
}

// Whether a sweep fuses a product into the add that reads it, decided once per
// equation by jit::Codegen::contract.
enum class Contract : bool { off, on };

// The ids a step's rows hold: a contraction's three, else the operands. A list
// is read off the arena rather than the Step.
[[nodiscard]] constexpr std::array<NodeId, 3>
step_inputs(const CNodeSource auto &nodes, NodeId v, const Contraction &c) {
  if (c) {
    return {c.x, c.y, c.z};
  }
  return nodes.operands(v);
}
// The tape rows of `ids` in a tape indexed by id.
[[nodiscard]] constexpr std::array<TapeRow, 3>
rows_of(const std::array<NodeId, 3> &ids) {
  std::array<TapeRow, 3> rows{no_row, no_row, no_row};
  std::ranges::transform(ids, rows.begin(), [](NodeId u) {
    return u == no_node ? no_row : TapeRow{u};
  });
  return rows;
}

// The steps of `order` with each node's contraction resolved once, rather than
// re-derived per point inside the sweep.
namespace detail {
[[nodiscard]] constexpr std::vector<Step>
schedule_of(const CNodeSource auto &nodes,
            std::ranges::input_range auto &&order,
            Contract contract = Contract::on)
  requires std::convertible_to<std::ranges::range_value_t<decltype(order)>,
                               NodeId>
{
  const bool on = contract == Contract::on;
  return DDX_FWD(order) | std::views::transform([&nodes, on](NodeId v) {
           const OpCode op = nodes.op_of(v);
           const Contraction c = on ? contraction_at(nodes, v) : Contraction{};
           return Step{.node = v,
                       .fma = c,
                       .out = TapeRow{v},
                       .in = rows_of(step_inputs(nodes, v, c)),
                       .fused = on && op == OpCode::SumN,
                       .op = op};
         }) |
         impl::to<std::vector<Step>>();
}

// The ids a step reads, which liveness is computed over: an fma's three, a
// list (a fused sum's products as their factors), or the operands.
[[nodiscard]] constexpr boost::container::small_vector<NodeId, 8>
reads_of(const auto &s, const Step &step) {
  if (step.fma) {
    return {step.fma.x, step.fma.y, step.fma.z};
  }
  boost::container::small_vector<NodeId, 8> out;
  if (step.fused) {
    for (const NodeId t : arguments(s, step.node)) {
      if (const Contraction c = product_of(s, t)) {
        out.insert(out.end(), {c.x, c.y});
      } else {
        out.push_back(t);
      }
    }
    return out;
  }
  for_each_operand(s, step.node, [&out](NodeId u) { out.push_back(u); });
  return out;
}

// Whether `op` reads each element of the operand at `slot` exactly once; a
// gemv's vector or an outer's operands are read repeatedly.
[[nodiscard]] constexpr bool reads_once(OpCode op, std::size_t slot) noexcept {
  switch (op) {
  case OpCode::Map:
  case OpCode::Sum:
  case OpCode::MatVec:
  case OpCode::ColScale:
  case OpCode::Diag:
  case OpCode::MatRow:
  case OpCode::Gather:
  case OpCode::RowOf:
  case OpCode::AddAt:
    return slot == 0;
  case OpCode::Scale:
  case OpCode::RowScale:
    return slot == 1;
  case OpCode::Zip:
  case OpCode::Dot:
  case OpCode::VSelect:
    return true;
  default:
    return false;
  }
}
// The step each elementwise tensor node read once is fused into, else no_node.
// Only a kernel fuses; the interpreter forms every row.
[[nodiscard]] constexpr std::vector<NodeId>
fused_into(const auto &s, std::span<const Step> schedule,
           std::span<const NodeId> outputs) {
  std::vector<NodeId> into(s.size(), no_node);
  std::vector<std::uint32_t> uses(s.size(), 0);
  for (const Step &step : schedule) {
    for_each_operand(s, step.node, [&](NodeId u) { ++uses[u]; });
  }
  for (const NodeId o : outputs) {
    uses[o] += 2;
  }
  const auto elementwise = [&s](NodeId u) {
    switch (s.op_of(u)) {
    case OpCode::Map:
    case OpCode::Zip:
    case OpCode::Scale:
    case OpCode::Outer:
    case OpCode::RowScale:
    case OpCode::ColScale:
    case OpCode::VSelect:
    case OpCode::Diag:
      return is_tensor(s[u]);
    case OpCode::MatRow:
      return get<Orientation>(s[u]) == Orientation::transposed;
    default:
      return false;
    }
  };
  for (const Step &step : schedule) {
    for (const auto [slot, u] : s.operands(step.node) | std::views::enumerate) {
      if (u != no_node && uses[u] == 1 && elementwise(u) &&
          reads_once(s.op_of(step.node), static_cast<std::size_t>(slot))) {
        into[u] = step.node;
      }
    }
  }
  return into;
}

// The last schedule position reading each node (its own if unread, past the
// end for an output). A fused node reads its operands inside its reader.
[[nodiscard]] constexpr std::vector<std::uint32_t>
last_uses(const auto &s, std::span<const Step> schedule,
          std::span<const NodeId> outputs, std::span<const NodeId> fused = {}) {
  std::vector<std::uint32_t> last(s.size(), 0);
  // Ascending, and a node is formed before it is read, so the last write wins.
  for (const auto [p, step] : schedule | std::views::enumerate) {
    last[step.node] = static_cast<std::uint32_t>(p);
    for (const NodeId u : reads_of(s, step)) {
      last[u] = static_cast<std::uint32_t>(p);
    }
  }
  // Descending, so a chain of fused nodes reaches the step that forms it.
  for (const Step &step : schedule | std::views::reverse) {
    if (!fused.empty() && fused[step.node] != no_node) {
      for_each_operand(s, step.node, [&](NodeId u) {
        last[u] = std::max(last[u], last[step.node]);
      });
    }
  }
  for (const NodeId o : outputs) {
    last[o] = static_cast<std::uint32_t>(schedule.size());
  }
  return last;
}

// Packs the tape by liveness and writes every Step's rows. Frees happen after
// the take, so a step never writes a row it reads and lane loops never alias.
struct Packed {
  std::vector<TapeRow> by_id; // no_row for a node that is not scheduled.
  std::size_t rows = 0;
};
[[nodiscard]] constexpr Packed pack_rows(const auto &s,
                                         std::span<Step> schedule,
                                         std::span<const NodeId> outputs) {
  const auto last = last_uses(s, schedule, outputs);
  Packed out{.by_id = std::vector<TapeRow>(s.size(), no_row)};
  std::vector<TapeRow> taken(s.size(), no_row); // A node's row while live.
  std::vector<TapeRow> free;
  TapeRow next{0};
  for (const auto [p, step] : schedule | std::views::enumerate) {
    const auto at = static_cast<std::uint32_t>(p);
    auto &row = taken[step.node];
    if (free.empty()) {
      row = next++;
    } else {
      row = free.back();
      free.pop_back();
    }
    out.by_id[step.node] = row;
    step.out = row;
    std::ranges::transform(
        step_inputs(s, step.node, step.fma), step.in.begin(),
        [&taken](NodeId u) { return u == no_node ? no_row : taken[u]; });
    // Free after the take; a row read twice in one step is freed once.
    auto reads = reads_of(s, step);
    reads.push_back(step.node);
    for (const NodeId u : reads) {
      if (last[u] == at && taken[u] != no_row) {
        free.push_back(taken[u]);
        taken[u] = no_row;
      }
    }
  }
  out.rows = next;
  return out;
}
} // namespace detail

// A tensor node's place in the tensor tape, per lane; width 0 means no row.
// An output matrix records `rows` too, since it leaves row-major.
struct TensorRow {
  std::uint32_t at = 0;
  std::uint32_t width = 0;
  std::uint32_t rows = 0;
};
// Every tensor row of a schedule by id, and one lane's total element count.
// Both are empty for a schedule without tensors.
struct TensorLayout {
  std::vector<TensorRow> rows;
  std::size_t extent = 0;
};

// Packs the tensor tape by liveness like the scalar rows, reusing a freed row
// for the next node of the same width.
namespace detail {
[[nodiscard]] constexpr TensorLayout
tensor_layout(const auto &s, std::span<const Step> schedule,
              std::span<const NodeId> outputs = {},
              std::span<const NodeId> fused = {}) {
  TensorLayout out;
  if (std::ranges::none_of(schedule, [&s](const Step &step) {
        return is_tensor(s[step.node]);
      })) {
    return out;
  }
  const auto last = last_uses(s, schedule, outputs, fused);
  out.rows.resize(s.size());
  std::vector<TensorRow> free; // Runs that no live row holds.
  for (const auto [p, step] : schedule | std::views::enumerate) {
    const auto at = static_cast<std::uint32_t>(p);
    // A table's MatRow is read in place and gets no row, unless it is an
    // output, which leaves through a row.
    if (is_tensor(s[step.node]) &&
        !(is_table(s, step.node) &&
          get<Orientation>(s[step.node]) == Orientation::plain &&
          !std::ranges::contains(outputs, step.node))) {
      const std::uint32_t w = width_of(s, step.node);
      const auto fit = std::ranges::find(free, w, &TensorRow::width);
      if (fit == free.end()) {
        out.rows[step.node] = {.at = static_cast<std::uint32_t>(out.extent),
                               .width = w};
        out.extent += w;
      } else {
        out.rows[step.node] = *fit;
        free.erase(fit);
      }
    }
    // A freed row keeps its address for this sweep; reads are deduplicated so
    // it is freed only once.
    auto reads = reads_of(s, step);
    reads.push_back(step.node);
    // The reads of nodes fused into this step count as this step's reads.
    for (std::size_t r = 0; !fused.empty() && r < reads.size(); ++r) {
      if (fused[reads[r]] == step.node ||
          (fused[reads[r]] != no_node &&
           std::ranges::contains(reads, fused[reads[r]]))) {
        for_each_operand(s, reads[r], [&](NodeId u) { reads.push_back(u); });
      }
    }
    std::ranges::sort(reads);
    reads.erase(std::ranges::unique(reads).begin(), reads.end());
    for (const NodeId u : reads) {
      if (last[u] == at && out.rows[u].width != 0) {
        free.push_back(out.rows[u]);
      }
    }
  }
  return out;
}

// The ids a liveness pass kept, in ascending order.
[[nodiscard]] constexpr auto live_ids(const std::vector<bool> &live) {
  return std::views::iota(NodeId{0}, static_cast<NodeId>(live.size())) |
         std::views::filter([&live](NodeId v) { return live[v]; });
}

} // namespace detail

} // namespace ddx::rt

DDX_EXTERN_INSTANCE(class DDX_INSTANCE_API ::ddx::rt::Arena<double>)
DDX_EXTERN_INSTANCE(class DDX_INSTANCE_API ::ddx::rt::Arena<float>)
