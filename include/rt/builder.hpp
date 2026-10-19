#pragma once

#include "ops/algebra.hpp"
#include "rt/apply.hpp"
#include "rt/opcode.hpp"
#include "util/config.hpp"
#include "util/error.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>
#include <boost/describe/class.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ddx::rt {

using NodeId = std::uint32_t;
inline constexpr NodeId no_node = ~NodeId{0};

template <impl::Numeric T> class Expression;
template <impl::Numeric T = double> class Builder;

// Befriended by Builder: naming an input is this one primitive's alone, and
// var(), param() and implicit() are its spellings.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> named(Builder<T> &b, OpCode op,
                                            std::string_view name);
template <impl::Numeric T, std::invocable<const Expression<T> &> F>
[[nodiscard]] constexpr Expression<T>
implicit(Builder<T> &b, std::string_view name, F &&residual);
namespace detail {
struct Implicits;
}

// A function the caller supplies: `fn(in, out)` reads the arguments and writes
// the answer in ExternRecord's layout.  Shared, so a frozen graph and a
// compiled kernel keep it alive as long as they can call it.  It must not
// evaluate a ddx equation of the same T: the sweep's tapes are per thread, and
// a nested sweep would resize them under the outer one.
template <impl::Numeric T>
using ExternFn = std::function<void(std::span<const T>, std::span<T>)>;

// What extern_fn() registers: a name, how many
// arguments, and how many orders of partials the function fills -- 1 or 2,
// since no call ddx answers asks for a third.  The answer's layout is this
// type's alone: 0 the value, 1..k the first partials, then the upper triangle
// of the second, row-major.
struct ExternRecord {
  std::string name;
  std::uint32_t arity = 1;
  std::uint32_t order = 2;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return arity >= 1 && order >= 1 && order <= 2;
  }
  // The answer's length.
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
  // Which component is d/dx_l of `component`: a first partial off the value,
  // a second off a first.  Nothing past that: an order-1 function is refused
  // before a second-order sweep reaches it.
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
  BOOST_DESCRIBE_CLASS(ExternRecord, (), (name, arity, order), (), ())
};

// The record and the function itself.
template <impl::Numeric T> struct ExternEntry : ExternRecord {
  std::shared_ptr<const ExternFn<T>> fn;
  BOOST_DESCRIBE_CLASS(ExternEntry, (ExternRecord), (), (), ())
};

template <impl::Numeric T> class ExternFunction;
template <impl::Numeric T>
[[nodiscard]] ExternFunction<T> extern_fn(Builder<T> &b, ExternRecord record,
                                          std::type_identity_t<ExternFn<T>> fn);
namespace detail {
template <impl::Numeric T, typename Sink>
constexpr void extern_partials(Builder<T> &b, NodeId v, Sink &&sink);
}

// A named input: which kind of leaf, and its place among that kind.
struct Input {
  OpCode op = OpCode::Var;
  std::uint32_t index = 0;
  constexpr bool operator==(const Input &) const = default;
};

// Befriended for symmetry with var(); a seed slot moves nothing.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> seed(Builder<T> &b, std::uint32_t slot);

// A multiply and an add taken as one rounding: x * y + z, with x negated where
// the multiply reached the add through a Neg, which is what a subtraction
// builds.  Named by the add.
struct Contraction {
  NodeId x = no_node;
  NodeId y = no_node;
  NodeId z = no_node;
  bool negated = false;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return x != no_node;
  }
};

// One node of a sweep and the fma it takes, if any.  Together, so a walker is
// never handed an order without the contractions resolved over it.  `out` and
// `in` are where the sweep writes and reads: tape rows, which an arena walk
// leaves as the ids and a freeze packs by liveness -- the ids above stay what
// codegen and the cone read.
struct Step {
  NodeId node = no_node;
  Contraction fma{};
  std::uint32_t out = no_node;
  std::array<std::uint32_t, 3> in{no_node, no_node, no_node};
  bool fused = false; // a SumN under contraction: its product terms are fma
};

namespace detail {

// What a node's `aux` word holds, by op -- the table in rt/opcode.hpp.  Reads
// go through these; only a node being formed or copied spells `aux`.
[[nodiscard]] constexpr std::uint32_t index_of(const auto &node) noexcept {
  return node.aux;
}
[[nodiscard]] constexpr std::uint32_t count_of(const auto &node) noexcept {
  return node.aux;
}
[[nodiscard]] constexpr bool is_transposed(const auto &node) noexcept {
  return node.aux != 0;
}
[[nodiscard]] constexpr ExternRef extern_of(const auto &node) noexcept {
  return ExternRef::of(node.aux);
}
// The inner opcode a Map or Zip applies elementwise.
[[nodiscard]] constexpr OpCode inner_of(const auto &node) noexcept {
  return static_cast<OpCode>(node.aux);
}
// The opcode whose rule a node follows: a Map or Zip is its inner one.
[[nodiscard]] constexpr OpCode effective_op(const auto &node) noexcept {
  return node.op == OpCode::Map || node.op == OpCode::Zip ? inner_of(node)
                                                          : node.op;
}
// Whether a node's value is a vector row rather than a scalar: its op's
// answer, but for a Select, which is one over tensor arms.
[[nodiscard]] constexpr bool is_tensor(const auto &node) noexcept {
  return tensor_valued(node.op) || (node.op == OpCode::Select && node.aux != 0);
}

// Ids are topological, so one descending pass settles it.  Not a filtered view:
// the body marks entries the pass has not reached.
[[nodiscard]] constexpr std::vector<bool>
reachable(std::size_t n, std::span<const NodeId> roots, auto &&operands_of) {
  std::vector<bool> live(n, false);
  std::ranges::for_each(roots, [&](NodeId r) { live[r] = true; });
  for (NodeId v = static_cast<NodeId>(n); v-- > 0;) {
    if (live[v]) {
      operands_of(v, [&](NodeId u) { live[u] = true; });
    }
  }
  return live;
}

// The operands a node actually has.  Arity is at most three and the absent ones
// are no_node, so every walker would otherwise repeat the same skip.  Not for
// a node with a list (has_list), whose `a` is no operand: for_each_operand
// below is what a walk over every node takes.
[[nodiscard]] constexpr auto operands_of(const auto &nodes, NodeId v) {
  return nodes.operands(v) |
         std::views::filter([](NodeId u) { return u != no_node; });
}

} // namespace detail

template <impl::Numeric T> struct Node {
  OpCode op = OpCode::Const;
  NodeId a = no_node;
  NodeId b = no_node;
  NodeId c = no_node;    // the third operand, which only a select has
  T value{};             // Const
  std::uint32_t aux = 0; // per op: see rt/opcode.hpp and index_of() et al.
  BOOST_DESCRIBE_CLASS(Node, (), (op, a, b, c, value, aux), (), ())
};

namespace detail {
// The layout before a select needed a third operand, kept for the assertion.
template <impl::Numeric T> struct TwoOperandNode {
  OpCode op;
  NodeId a;
  NodeId b;
  T value;
  std::uint32_t aux;
};
} // namespace detail

// The third operand lands in padding the two-operand layout already carried.
// Pinned rather than assumed: a member above `value` would silently widen every
// node in every arena.
static_assert(sizeof(Node<double>) == sizeof(detail::TwoOperandNode<double>));

// The data behind a VecConst or MatConst leaf, column-major; a vector is one
// column.  Owned by the model, so a file carries it and a Gemv's matrix is
// read in place rather than copied into nodes.
template <impl::Numeric T> struct TensorConst {
  std::uint32_t rows = 0;
  std::uint32_t cols = 1;
  std::vector<T> data;
  [[nodiscard]] constexpr bool valid() const noexcept {
    return rows >= 1 && cols >= 1 && data.size() == std::size_t{rows} * cols;
  }
  BOOST_DESCRIBE_CLASS(TensorConst, (), (rows, cols, data), (), ())
};

// Everything a model is: what the Builder holds, as one described list.
template <impl::Numeric T> struct Model {
  // Each named kind's names in declaration order, which is slot order.
  PerKind<std::vector<std::string>> names;
  std::vector<NodeId> closures; // the residual each implicit is a root of
  std::vector<ExternEntry<T>> externs;
  std::vector<TensorConst<T>> tensors;
  std::vector<NodeId> lists; // every has_list node's ids, end to end
  std::vector<Node<T>> nodes;
  BOOST_DESCRIBE_CLASS(Model, (),
                       (names, closures, externs, tensors, lists, nodes), (),
                       ())

  // The CNodeSource half, which the Builder lends and the frozen Graph
  // answers as well.
  [[nodiscard]] constexpr const Node<T> &operator[](NodeId v) const {
    return nodes[v];
  }
  [[nodiscard]] constexpr OpCode op_of(NodeId v) const { return nodes[v].op; }
  [[nodiscard]] constexpr std::array<NodeId, 3> operands(NodeId v) const {
    return {nodes[v].a, nodes[v].b, nodes[v].c};
  }
  [[nodiscard]] constexpr std::size_t size() const { return nodes.size(); }
};

// The function table and the list pool of a source: the Model holds them, the
// Builder and the frozen Graph lend them.
[[nodiscard]] constexpr auto externs_of(const auto &s) {
  if constexpr (requires { s.externs(); }) {
    return s.externs();
  } else {
    return std::span{s.externs};
  }
}
[[nodiscard]] constexpr std::span<const NodeId> lists_of(const auto &s) {
  if constexpr (requires { s.lists(); }) {
    return s.lists();
  } else {
    return std::span{s.lists};
  }
}
// The tensor table of a source: the Model holds it, the Builder lends it.
[[nodiscard]] constexpr auto tensors_of(const auto &s) {
  if constexpr (requires { s.tensors(); }) {
    return s.tensors();
  } else {
    return std::span{s.tensors};
  }
}

// The ids a has_list node is over, in order: an extern's arguments, the
// function's arity long; a Pack's or StackRows' items or a sum's terms, the
// slot long.
[[nodiscard]] constexpr std::span<const NodeId> arguments(const auto &s,
                                                          NodeId v) {
  const OpCode op = s.op_of(v);
  const std::size_t count =
      is_extern(op) ? externs_of(s)[detail::extern_of(s[v]).function].arity
                    : detail::count_of(s[v]);
  return lists_of(s).subspan(s.operands(v)[0], count); // `a` is the offset
}

namespace detail {
// Every id a node reads: its list, or the operands its arity names.  The one
// spelling of a node's children, which every reachability walk takes.
constexpr void for_each_operand(const auto &s, NodeId v, auto &&f) {
  if (has_list(s.op_of(v))) {
    std::ranges::for_each(arguments(s, v), f);
  } else {
    std::ranges::for_each(operands_of(s, v), f);
  }
}
} // namespace detail

// A row's shape: rows x cols, column-major, cols 1 for a vector.  No node
// carries it; it is a function of the operands -- a walk down the first (or,
// for a Scale, second) operand through the elementwise ops, and a step into
// the matrix ops' own operands.  Asked of a sound source, whose ids are
// topological, so the walk ends.
struct Shape {
  std::uint32_t rows = 1;
  std::uint32_t cols = 1;
  [[nodiscard]] constexpr std::uint32_t size() const noexcept {
    return rows * cols;
  }
  constexpr bool operator==(const Shape &) const = default;
};
[[nodiscard]] constexpr Shape shape_of(const auto &s, NodeId v) {
  for (;;) {
    const auto [a, b, c] = s.operands(v);
    switch (s.op_of(v)) {
    case OpCode::VecConst:
      return {.rows = tensors_of(s)[detail::index_of(s[v])].rows, .cols = 1};
    case OpCode::MatConst:
      return {.rows = 0, .cols = 0};
    case OpCode::MatRow: {
      // A table, or -- the dense sweep's transpose -- a computed matrix.
      const auto table = [&] {
        const auto &m = tensors_of(s)[detail::index_of(s[a])];
        return Shape{.rows = m.rows, .cols = m.cols};
      };
      const Shape m = s.op_of(a) == OpCode::MatConst ? table() : shape_of(s, a);
      return detail::is_transposed(s[v]) ? Shape{.rows = m.cols, .cols = m.rows}
                                         : m;
    }
    case OpCode::Pack:
      return {.rows = detail::count_of(s[v]), .cols = 1};
    case OpCode::Gemv: {
      const auto &m = tensors_of(s)[detail::index_of(s[a])];
      return {.rows = detail::is_transposed(s[v]) ? m.cols : m.rows, .cols = 1};
    }
    case OpCode::Diag: {
      const std::uint32_t w = shape_of(s, a).rows;
      return {.rows = w, .cols = w};
    }
    case OpCode::Outer:
      return {.rows = shape_of(s, a).rows, .cols = shape_of(s, b).rows};
    case OpCode::MatVec: {
      const Shape m = shape_of(s, a);
      return {.rows = detail::is_transposed(s[v]) ? m.cols : m.rows, .cols = 1};
    }
    case OpCode::MatMul:
      return {.rows = shape_of(s, a).rows, .cols = shape_of(s, b).cols};
    case OpCode::StackRows:
      return {.rows = detail::count_of(s[v]),
              .cols = shape_of(s, arguments(s, v)[0]).rows};
    case OpCode::RowOf:
      return {.rows = shape_of(s, a).cols, .cols = 1};
    case OpCode::Map:
    case OpCode::Zip:
    case OpCode::AddAt:
    case OpCode::ColScale:
      v = a;
      break;
    case OpCode::Scale:
    case OpCode::RowScale:
    case OpCode::Select:
      v = b;
      break;
    default:
      return {};
    }
  }
}
// How many scalars a node's value is: one for a scalar node and for the tensor
// ops that read a vector down to one, else the row's rows x cols.  Zero for a
// MatConst, which is a table and never a row.
[[nodiscard]] constexpr std::uint32_t width_of(const auto &s, NodeId v) {
  return shape_of(s, v).size();
}

// Nodes are interned as they are formed, so an id *is* the identity of a
// subexpression -- where the compile-time side compares types.
template <impl::Numeric T> class Builder {
public:
  using value_type = T;
  using node_type = Node<T>;

  [[nodiscard]] constexpr NodeId constant(const T &v) {
    return intern({.op = OpCode::Const, .value = v});
  }

  // Room for `nodes` in all, arena and table at once, instead of growing both
  // through every doubling on the way: what a pass that knows the size of
  // what it is about to build asks for.
  constexpr void reserve(std::size_t nodes) {
    model_.nodes.reserve(nodes);
    if (2 * nodes > table_.size()) {
      rehash(nodes);
    }
  }

  // The leaf over `data`: a VecConst for one column, a MatConst otherwise.
  // Interned by content where T compares, so zeros(w) is one node per width
  // and a matrix given twice is kept once.
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
    return intern({.op = cols == 1 ? OpCode::VecConst : OpCode::MatConst,
                   .aux = static_cast<std::uint32_t>(
                       std::distance(model_.tensors.begin(), at))});
  }
  // A row of `shape` holding v everywhere: a matrix is the table's MatRow.
  [[nodiscard]] constexpr NodeId filled(Shape shape, const T &v) {
    const NodeId t =
        tensor_const(shape.rows, shape.cols, std::vector<T>(shape.size(), v));
    return shape.cols == 1 ? t : make(OpCode::MatRow, t);
  }
  [[nodiscard]] constexpr NodeId zeros(Shape shape) {
    return filled(shape, T{0});
  }
  [[nodiscard]] constexpr NodeId ones(Shape shape) {
    return filled(shape, T{1});
  }
  // The vector of `scalars` -- or, as StackRows, the matrix whose rows are
  // `items`.
  [[nodiscard]] constexpr NodeId pack(std::span<const NodeId> items,
                                      OpCode op = OpCode::Pack) {
    assert(!items.empty());
    return make(op, list(items), no_node, no_node,
                static_cast<std::uint32_t>(items.size()));
  }

  // A node with a list, a tensor node or an extern call is formed with its
  // aux word and past fold() -- no identity below reads a vector or a list --
  // and past the commutative swap, which is a Zip's inner op's business and not
  // Zip's.
  [[nodiscard]] constexpr NodeId make(OpCode op, NodeId a, NodeId b = no_node,
                                      NodeId c = no_node,
                                      std::uint32_t aux = 0) {
    if (has_list(op) || kind_of(op) == Kind::Tensor) {
      return intern({.op = op, .a = a, .b = b, .c = c, .aux = aux});
    }
    assert(aux == 0 || op == OpCode::Select);
    // fold() and the commutative swap are both binary questions; a third
    // operand means neither applies, and asking would hand a binary functor
    // three arguments.
    if (c == no_node) {
      if (const auto folded = fold(op, a, b)) {
        return *folded;
      }
      if (is_commutative<T>(op) && b != no_node && b < a) {
        std::swap(a, b);
      }
    } else if (const auto chosen = fold_select(a, b, c)) {
      return *chosen;
    }
    return intern({.op = op, .a = a, .b = b, .c = c, .aux = aux});
  }

  // The same node over remapped operands -- or over its remapped list --
  // what a rewrite that keeps every operation forms.
  [[nodiscard]] constexpr NodeId rebuild(NodeId v, auto &&remap) {
    Node<T> n = model_[v]; // by value: list() and make() may reallocate
    if (has_list(n.op)) {
      n.a = list(arguments(*this, v) | std::views::transform(remap) |
                 impl::to<std::vector<NodeId>>());
      return make(n.op, n.a, no_node, no_node, n.aux);
    }
    for (NodeId *const u : {&n.a, &n.b, &n.c}) {
      *u = *u == no_node ? no_node : remap(*u);
    }
    return make(n.op, n.a, n.b, n.c, n.aux);
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
  [[nodiscard]] constexpr std::size_t size() const { return model_.size(); }
  [[nodiscard]] constexpr std::span<const Node<T>> nodes() const {
    return model_.nodes;
  }
  // Each kind's names in declaration order, which is slot order: the order a
  // positional point is read in and every result is laid out in.
  [[nodiscard]] constexpr const Model<T> &model() const { return model_; }
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
  // Every named input in column order: the kinds in table order, each in
  // declaration order.
  [[nodiscard]] constexpr std::vector<std::string> names() const {
    return model_.names | std::views::join |
           impl::to<std::vector<std::string>>();
  }
  // The residual whose root each implicit input is, by slot; no_node only
  // beside a broken() arena.  And the first misuse of implicit(), which no
  // expression can carry, for the equation to refuse with.
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

  // The leaf a named input is.
  [[nodiscard]] constexpr NodeId leaf(OpCode op, std::uint32_t index) const {
    return ids_[*named_index(op)][index];
  }

  // Which input `name` is, if any.
  [[nodiscard]] constexpr std::optional<Input>
  lookup(std::string_view name) const {
    const auto at = std::ranges::lower_bound(by_name_, name, {}, name_of());
    return at != by_name_.end() && name_of()(*at) == name ? std::optional{*at}
                                                          : std::nullopt;
  }

  // A Const of `v`, or a VecConst of nothing but `v`: what a cancelled term
  // is in either shape.
  [[nodiscard]] constexpr bool is_constant(NodeId id, const T &v) const {
    const Node<T> &n = model_.nodes[id];
    if constexpr (std::equality_comparable<T>) {
      if (n.op == OpCode::Const) {
        return n.value == v;
      }
      return n.op == OpCode::VecConst &&
             std::ranges::all_of(model_.tensors[detail::index_of(n)].data,
                                 [&v](const T &x) { return x == v; });
    } else {
      return false;
    }
  }

private:
  // One namespace for every named input: a name is one kind of input, never
  // two, and naming it again hands back the same leaf.  A new name takes the
  // next slot of its kind and moves nothing already built.  no_node is a name
  // already taken by another kind, which named() poisons.
  template <impl::Numeric U>
  friend constexpr Expression<U> named(Builder<U> &, OpCode, std::string_view);
  [[nodiscard]] constexpr NodeId input(OpCode op, std::string_view name) {
    const auto at = std::ranges::lower_bound(by_name_, name, {}, name_of());
    if (at != by_name_.end() && name_of()(*at) == name) {
      return at->op == op ? leaf(op, at->index) : no_node;
    }
    const auto k = *named_index(op);
    const auto index = static_cast<std::uint32_t>(model_.names[k].size());
    model_.names[k].emplace_back(name);
    ids_[k].push_back(intern({.op = op, .aux = index}));
    by_name_.insert(at, {.op = op, .index = index});
    if (op == OpCode::Implicit) {
      model_.closures.push_back(no_node);
      closure_rows_.emplace_back();
    }
    return ids_[k].back();
  }

  // Implicit `z` is a root of `f`.  A misuse -- a poisoned residual, or the
  // same implicit named twice -- is kept rather than thrown: the equation
  // built over this arena refuses with it.
  template <impl::Numeric U, std::invocable<const Expression<U> &> F>
  friend constexpr Expression<U> implicit(Builder<U> &, std::string_view, F &&);
  constexpr void close(NodeId z, NodeId f, std::optional<errc> why) {
    NodeId &closure = model_.closures[detail::index_of(model_.nodes[z])];
    if (why || closure != no_node) {
      broken_ = broken_.value_or(why.value_or(errc::closed_twice));
    } else {
      closure = f;
    }
  }

  // Registering an extern is extern_fn()'s alone; its index is what the call
  // names.
  template <impl::Numeric U>
  friend ExternFunction<U> extern_fn(Builder<U> &, ExternRecord,
                                     std::type_identity_t<ExternFn<U>>);
  constexpr std::optional<std::uint32_t> add_extern(ExternEntry<T> e) {
    if (!ExternRef::fits(model_.externs.size(), e.size())) {
      return std::nullopt;
    }
    model_.externs.push_back(std::move(e));
    return static_cast<std::uint32_t>(model_.externs.size() - 1);
  }

  // The call, the function's value; the sweeps ask for the other components.
  template <impl::Numeric U> friend class ExternFunction;
  constexpr NodeId extern_call(std::uint32_t function,
                               std::span<const NodeId> args) {
    return intern({.op = OpCode::Extern,
                   .a = list(args),
                   .aux = ExternRef{.function = function}.packed()});
  }

  // Where `ids` sit in the pool: the run already there that spells them, else
  // appended.  Content-keyed, so two nodes over the same list intern to one.
  [[nodiscard]] constexpr NodeId list(std::span<const NodeId> ids) {
    const auto found = std::ranges::search(model_.lists, ids);
    if (!found.empty()) {
      return static_cast<NodeId>(found.begin() - model_.lists.begin());
    }
    const auto at = static_cast<NodeId>(model_.lists.size());
    impl::append(model_.lists, ids);
    return at;
  }

  // A sum still being accumulated by `+=`: the terms so far as a list that
  // shares its prefix with every handle copied along the way, so a copy of an
  // accumulator keeps the value it had.  Nothing is a node until the sum is
  // read, and then once: one term is itself, two are an Add -- the chain's
  // own folds and fma -- and more a SumN over the terms in order.
  template <impl::Numeric U> friend class Expression;
  struct Pending {
    NodeId term = no_node;
    std::uint32_t prev = no_node; // the entry before, no_node for the first
    NodeId node = no_node;        // what this entry materialised as
  };
  std::vector<Pending> pending_;
  [[nodiscard]] constexpr std::uint32_t push_term(std::uint32_t prev,
                                                  NodeId term) {
    pending_.push_back({.term = term, .prev = prev});
    return static_cast<std::uint32_t>(pending_.size() - 1);
  }
  [[nodiscard]] constexpr NodeId sum_of(std::uint32_t entry) {
    if (pending_[entry].node != no_node) {
      return pending_[entry].node;
    }
    std::vector<NodeId> terms;
    for (auto i = entry; i != no_node; i = pending_[i].prev) {
      terms.push_back(pending_[i].term);
    }
    std::ranges::reverse(terms);
    const NodeId node = terms.size() == 1 ? terms[0]
                        : terms.size() == 2
                            ? make(OpCode::Add, terms[0], terms[1])
                            : make(OpCode::SumN, list(terms), no_node, no_node,
                                   static_cast<std::uint32_t>(terms.size()));
    pending_[entry].node = node;
    return node;
  }
  // Another component of the call `v` is: the same function and list,
  // another slot.
  template <impl::Numeric U, typename Sink>
  friend constexpr void detail::extern_partials(Builder<U> &, NodeId, Sink &&);
  constexpr NodeId extern_component(NodeId v, std::uint32_t component) {
    const Node<T> n = model_[v]; // by value: intern() may reallocate
    return intern({.op = OpCode::ExternD,
                   .a = n.a,
                   .aux = ExternRef{.function = detail::extern_of(n).function,
                                    .component = component}
                              .packed()});
  }

  // dz/ds for each implicit, by the implicit-function theorem, built the first
  // time a sweep reaches the leaf and kept: derivative.hpp's alone.
  friend struct detail::Implicits;
  std::vector<std::optional<std::vector<NodeId>>> closure_rows_;

  // A seed slot is given rather than looked up.  Interning leaves one node per
  // slot.
  template <impl::Numeric U>
  friend constexpr Expression<U> seed(Builder<U> &, std::uint32_t);
  [[nodiscard]] constexpr NodeId seed_node(std::uint32_t index) {
    return intern({.op = OpCode::Seed, .aux = index});
  }

  static constexpr std::uint64_t payload_of(const Node<T> &n) {
    if (n.op != OpCode::Const) {
      return std::uint64_t{n.aux};
    }
    // Anything wider than a word lands in one bucket and is separated by
    // `same` below.
    if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 8) {
      return std::bit_cast<std::uint64_t>(n.value);
    } else {
      return 0;
    }
  }

  static constexpr bool same(const Node<T> &l, const Node<T> &r) {
    if (l.op != r.op || l.a != r.a || l.b != r.b || l.c != r.c) {
      return false;
    } else if (l.op != OpCode::Const) {
      return l.aux == r.aux;
    }
    return identical(l.value, r.value);
  }
  // By the bits where T is a word, as the hash is: -0.0 and +0.0 stay two
  // constants and a NaN is one.  CFieldLike promises arithmetic, not equality.
  static constexpr bool identical(const T &l, const T &r) {
    if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 8) {
      return std::bit_cast<std::uint64_t>(l) == std::bit_cast<std::uint64_t>(r);
    } else if constexpr (std::equality_comparable<T>) {
      return l == r;
    } else {
      return false;
    }
  }

  static constexpr std::size_t hash_of(const Node<T> &n) {
    std::uint64_t h = static_cast<std::uint64_t>(n.op);
    const auto mix = [&h](std::uint64_t v) {
      constexpr std::uint64_t golden_ratio_64 = 0x9e3779b97f4a7c15ULL;
      constexpr std::uint64_t murmur_mix_64 = 0xff51afd7ed558ccdULL;
      h ^= v + golden_ratio_64 + (h << 6) + (h >> 2);
      h *= murmur_mix_64;
      h ^= h >> 33;
    };
    mix(n.a);
    mix(n.b);
    mix(n.c);
    mix(payload_of(n));
    return static_cast<std::size_t>(h);
  }

  // Unbounded because the table is kept at most half full: every run reaches an
  // empty slot.
  constexpr auto probe(std::size_t h) const {
    return std::views::iota(h) |
           std::views::transform(
               [mask = table_.size() - 1](std::size_t k) { return k & mask; });
  }

  // The table for `nodes` in all.  bit_ceil, not doubling: the size is the
  // invariant the mask needs, and half full is where linear probing starts to
  // degrade.
  constexpr void rehash(std::size_t nodes) {
    table_.assign(std::bit_ceil(std::max<std::size_t>(64, 2 * nodes)), no_node);
    for (const auto [id, n] : model_.nodes | std::views::enumerate) {
      auto run = probe(hash_of(n));
      table_[*std::ranges::find_if(run, [this](std::size_t i) {
        return table_[i] == no_node;
      })] = static_cast<NodeId>(id);
    }
  }

  constexpr NodeId intern(const Node<T> &n) {
    if ((model_.nodes.size() + 1) * 2 > table_.size()) {
      rehash(model_.nodes.size() + 1);
    }
    // The run stops at whichever comes first, the node itself or the empty
    // slot it would take.
    auto run = probe(hash_of(n));
    const std::size_t i = *std::ranges::find_if(run, [this, &n](std::size_t k) {
      return table_[k] == no_node || same(model_.nodes[table_[k]], n);
    });
    if (table_[i] != no_node) {
      return table_[i];
    }
    const auto id = static_cast<NodeId>(model_.nodes.size());
    model_.nodes.push_back(n);
    table_[i] = id;
    return id;
  }

  // A scalar with no equality never matches, so ops/algebra.hpp's identity
  // rewrites simply stop firing for it.
  constexpr bool holds(NodeId id, int k) const {
    return id != no_node && is_constant(id, T(k));
  }

  // Which RuleOp an opcode is, or none for one carrying no identities.
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

  // simplify.hpp's predicates, answered by id compare: interning makes
  // structural identity an id equality.
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
      return model_.nodes[a].a;
    case Take::SquareOfA:
      // No recursion back into this rule: Mul carries no Same identity.
      return make(OpCode::Mul, a, a);
    }
    return std::nullopt;
  }

  constexpr std::optional<NodeId> fold(OpCode op, NodeId a, NodeId b) {
    const bool ca = a != no_node && model_.nodes[a].op == OpCode::Const;
    const bool cb = b != no_node && model_.nodes[b].op == OpCode::Const;
    const bool unary = arity_of(op) == 1;

    // Constant arithmetic first, through the same *_impl functor the
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
    // First match wins, as ops/algebra.hpp is ordered.
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

  // A literal condition has already chosen, read as select_impl reads it (any
  // nonzero, NaN included, is true), and two equal arms leave nothing to
  // choose.  What lets PowOpFn guard its 0 * inf at no cost to the graphs that
  // never reach it: a literal exponent folds the guard to the bare partial.
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

  // (n/d) * d -> n.  On a DAG the denominator match is an id compare.
  constexpr std::optional<NodeId> cancel_quotient(NodeId quotient,
                                                  NodeId factor) const {
    const Node<T> &q = model_.nodes[quotient];
    return q.op == OpCode::Div && q.b == factor ? std::optional{q.a}
                                                : std::nullopt;
  }

  Model<T> model_;
  // Everything below is an index over model_.
  // Open-addressed by hand: interning runs inside constant evaluation and no
  // library hash container is constexpr.
  std::vector<NodeId> table_; // power-of-two capacity; no_node == free slot
  // Per named kind, the leaf interning made for each slot.
  PerKind<std::vector<NodeId>> ids_;
  // Every named input, sorted by name: a flat map by hand, since interning runs
  // inside constant evaluation and no library flat container is constexpr
  // before C++26.
  std::vector<Input> by_name_;
  std::optional<errc> broken_;

  // The key by_name_ is sorted on.
  [[nodiscard]] constexpr auto name_of() const {
    return [this](const Input &in) -> std::string_view {
      return names(in.op)[in.index];
    };
  }
};

// What a walker reads: an id-addressed source that answers a node's operation
// and its operands.  Builder and the frozen Graph both do, which is what lets
// contraction_at() be written once over the two.
template <typename S>
concept CNodeSource = requires(const S &s, NodeId v) {
  { s.op_of(v) } -> std::same_as<OpCode>;
  { s.operands(v) } -> std::same_as<std::array<NodeId, 3>>;
  { s.size() } -> std::convertible_to<std::size_t>;
};

// The product a term is, if it is one: a Mul, or a Mul behind a Neg (what a
// subtraction builds); `z` left for the add it is fused into.
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

// fadd(fmul(x, y), z) -> fma(x, y, z),
// The rule is structural -- it asks the add what its operands are and nothing
// about how many readers they have -- which is what lets the arena walk, the
// frozen graph and the kernel agree to the bit.  The lower-id operand wins
// where an add has two products, the order Builder::make already sorted
// commutative operands into.  A SumN fuses by the same rule, term by term
// into the running sum (the `fused` Step).
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

// The contraction at every node of `order`, resolved once.  Asking
// contraction_at() inside a sweep re-derives per point what is a property of
// the nodes, at two dependent loads a time.  `on` false contracts nothing.
namespace detail {
[[nodiscard]] constexpr std::vector<Step>
schedule_of(const CNodeSource auto &nodes,
            std::ranges::input_range auto &&order, bool on = true)
  requires std::convertible_to<std::ranges::range_value_t<decltype(order)>,
                               NodeId>
{
  return DDX_FWD(order) | std::views::transform([&nodes, on](NodeId v) {
           const OpCode op = nodes.op_of(v);
           const Contraction c = on ? contraction_at(nodes, v) : Contraction{};
           // A list is read off the arena, not the Step.
           const auto ops = has_list(op) ? std::array{no_node, no_node, no_node}
                                         : nodes.operands(v);
           return Step{.node = v,
                       .fma = c,
                       .out = v,
                       .in = c ? std::array{c.x, c.y, c.z} : ops,
                       .fused = on && op == OpCode::SumN};
         }) |
         impl::to<std::vector<Step>>();
}

// The rows a step reads: a contraction's three, a list -- a fused sum's
// product terms as their factors -- else the operands.  What liveness is
// computed over, so the one spelling of it.
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

// The last schedule position at which each node is read -- its own, if
// nothing reads it -- and past the end for an output.  Indexed by id over the
// source.
[[nodiscard]] constexpr std::vector<std::uint32_t>
last_uses(const auto &s, std::span<const Step> schedule,
          std::span<const NodeId> outputs) {
  std::vector<std::uint32_t> last(s.size(), 0);
  // Ascending, and a node is formed before anything reads it, so the last
  // write is the last use.
  for (const auto [p, step] : schedule | std::views::enumerate) {
    last[step.node] = static_cast<std::uint32_t>(p);
    for (const NodeId u : reads_of(s, step)) {
      last[u] = static_cast<std::uint32_t>(p);
    }
  }
  for (const NodeId o : outputs) {
    last[o] = static_cast<std::uint32_t>(schedule.size());
  }
  return last;
}

// Pack the tape: a row is freed after the step that reads it last and taken by
// the next node formed, so the tape is the live set rather than the arena.
// Writes every Step's `out` and `in`; answers the row count and, by id, the row
// each node took -- for the reads a Step cannot carry, an extern's or a Pack's
// list.  A node's own row is never one it reads this step: the frees happen
// after the take, which is what lets the lane loops promise no aliasing.
struct Packed {
  std::vector<std::uint32_t> by_id; // no_node for a node not scheduled
  std::size_t rows = 0;
};
[[nodiscard]] constexpr Packed pack_rows(const auto &s,
                                         std::span<Step> schedule,
                                         std::span<const NodeId> outputs) {
  const auto last = last_uses(s, schedule, outputs);
  Packed out{.by_id = std::vector<std::uint32_t>(s.size(), no_node)};
  std::vector<std::uint32_t> taken(s.size(), no_node); // the row, while live
  std::vector<std::uint32_t> free;
  std::uint32_t next = 0;
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
    for (const auto [k, u] : step.in | std::views::enumerate) {
      step.in[static_cast<std::size_t>(k)] = u == no_node ? no_node : taken[u];
    }
    // After the take.  A row read twice in one step is freed once.
    auto reads = reads_of(s, step);
    reads.push_back(step.node);
    for (const NodeId u : reads) {
      if (last[u] == at && taken[u] != no_node) {
        free.push_back(taken[u]);
        taken[u] = no_node;
      }
    }
  }
  out.rows = next;
  return out;
}
} // namespace detail

// Where a tensor-valued node's rows sit in the tensor tape: `at` elements in,
// per lane, `width` wide.  Width 0 is a node with no row.
struct TensorRow {
  std::uint32_t at = 0;
  std::uint32_t width = 0;
};
// Every row of a schedule, indexed by id, and the elements one lane of all of
// them takes.  Empty rows and a zero extent is a schedule with no tensor in
// it, which is what every sweep without one costs.
struct TensorLayout {
  std::vector<TensorRow> rows;
  std::size_t extent = 0;
};

// Packed as the scalar rows are: a row is freed after its last reader and
// taken again by the next node of the same width, so the tensor tape is the
// live vectors, not every one the schedule forms.
namespace detail {
[[nodiscard]] constexpr TensorLayout
tensor_layout(const auto &s, std::span<const Step> schedule,
              std::span<const NodeId> outputs = {}) {
  TensorLayout out;
  if (std::ranges::none_of(schedule, [&s](const Step &step) {
        return is_tensor(s[step.node]);
      })) {
    return out;
  }
  const auto last = last_uses(s, schedule, outputs);
  out.rows.resize(s.size());
  std::vector<TensorRow> free; // (at, width) runs no live row holds
  for (const auto [p, step] : schedule | std::views::enumerate) {
    const auto at = static_cast<std::uint32_t>(p);
    if (is_tensor(s[step.node])) {
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
    // The row keeps its address for the sweep; only the free list forgets it,
    // once per node however often this step reads it.
    auto reads = reads_of(s, step);
    reads.push_back(step.node);
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
} // namespace detail

namespace detail {

// The roots' cone in an arena, which every walker below asks for first.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<bool>
reachable(const Builder<T> &b, std::span<const NodeId> roots) {
  return reachable(b.size(), roots, [&b](NodeId v, auto &&mark) {
    for_each_operand(b, v, mark);
  });
}

// The ids a liveness pass kept, ascending.
[[nodiscard]] constexpr auto live_ids(const std::vector<bool> &live) {
  return std::views::iota(NodeId{0}, static_cast<NodeId>(live.size())) |
         std::views::filter([&live](NodeId v) { return live[v]; });
}

} // namespace detail

} // namespace ddx::rt
