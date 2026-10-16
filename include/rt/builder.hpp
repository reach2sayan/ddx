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
#include <iterator>
#include <map>
#include <functional>
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

template <impl::Numeric T> class RTExpression;
template <impl::Numeric T = double> class Builder;
template <impl::Numeric T> class Verified;

// Befriended by Builder: naming an input is this one primitive's alone, and
// var(), param() and implicit() are its spellings.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> named(Builder<T> &b, OpCode op,
                                              std::string_view name);
template <impl::Numeric T>
constexpr void closes(const RTExpression<T> &z, const RTExpression<T> &f);
namespace detail {
struct Implicits;
}

// A function the caller supplies: `fn(in, out)` reads the arguments and writes
// the answer in ExternRecord's layout.  Shared, so a frozen graph and a
// compiled kernel keep it alive as long as they can call it.
template <impl::Numeric T>
using ExternFn = std::function<void(std::span<const T>, std::span<T>)>;

// Functions to bind by name when a saved equation is read back.
template <impl::Numeric T>
using ExternFns = std::map<std::string, ExternFn<T>, std::less<>>;

// What extern_fn() registers and a file keeps: a name to bind by, how many
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
    return 1 + arity + (order >= 2 ? arity * (arity + 1) / 2 : 0);
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

// The record and the function itself.  Null in an arena restored from a file
// until the caller binds it by name.
template <impl::Numeric T> struct ExternEntry : ExternRecord {
  std::shared_ptr<const ExternFn<T>> fn;
  // The record alone: the function is the one thing a file cannot carry.
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
  std::uint32_t slot = 0;
  constexpr bool operator==(const Input &) const = default;
};

// Befriended for symmetry with var(); a seed slot moves nothing.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> seed(Builder<T> &b, std::uint32_t slot);

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
// never handed an order without the contractions resolved over it.
struct Step {
  NodeId node = no_node;
  Contraction fma{};
};

namespace detail {

// Befriended by Builder: installing a saved node stream verbatim is the
// loader's alone, every other way in interning and folding.  Defined in
// rt/archive/snapshot.hpp, its only user.

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
// are no_node, so every walker would otherwise repeat the same skip.
[[nodiscard]] constexpr auto operands_of(const auto &nodes, NodeId v) {
  return nodes.operands(v) |
         std::views::filter([](NodeId u) { return u != no_node; });
}

} // namespace detail

template <impl::Numeric T> struct Node {
  OpCode op = OpCode::Const;
  NodeId a = no_node;
  NodeId b = no_node;
  NodeId c = no_node;     // the third operand, which only a select has
  T value{};              // Const
  std::uint32_t slot = 0; // Var
  BOOST_DESCRIBE_CLASS(Node, (), (op, a, b, c, value, slot), (), ())
};

namespace detail {
// The layout before a select needed a third operand, kept for the assertion.
template <impl::Numeric T> struct TwoOperandNode {
  OpCode op;
  NodeId a;
  NodeId b;
  T value;
  std::uint32_t slot;
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

// Everything a model is: what the Builder holds, a file keeps, and the model
// digest keys on -- one described list, so adding to it reaches all three.
template <impl::Numeric T> struct Model {
  // Each named kind's names in declaration order, which is slot order.
  PerKind<std::vector<std::string>> names;
  std::vector<NodeId> closures; // the residual each implicit is a root of
  std::vector<ExternEntry<T>> externs;
  std::vector<TensorConst<T>> tensors;
  std::vector<Node<T>> nodes;
  BOOST_DESCRIBE_CLASS(Model, (), (names, closures, externs, tensors, nodes),
                       (), ())

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

// An extern node's arguments, off any source that answers operands and
// externs: the list ExternArgs(first, rest) spells, the function's arity long,
// a list of one being the argument itself.  Nothing where the slot names no
// function or a hop is not a list -- what a file's checker asks, and what
// every other caller dereferences.
[[nodiscard]] std::optional<boost::container::small_vector<NodeId, 8>>
extern_arguments(const auto &s, NodeId v) {
  // The function table: the Model holds it, the Builder and the Graph lend it.
  const auto externs = [&s] {
    if constexpr (requires { s.externs(); }) {
      return s.externs();
    } else {
      return std::span{s.externs};
    }
  }();
  const auto [fn, component] = ExternSlot::of(s[v].slot);
  if (fn >= externs.size()) {
    return std::nullopt;
  }
  boost::container::small_vector<NodeId, 8> out;
  NodeId args = s.operands(v)[0];
  for (auto left = externs[fn].arity; left > 1; --left) {
    if (kind_of(s.op_of(args)) != Kind::Args) {
      return std::nullopt;
    }
    const auto [first, rest, unused] = s.operands(args);
    out.push_back(first);
    args = rest;
  }
  out.push_back(args);
  return out;
}

// The tensor table of a source: the Model holds it, the Builder lends it.
[[nodiscard]] constexpr auto tensors_of(const auto &s) {
  if constexpr (requires { s.tensors(); }) {
    return s.tensors();
  } else {
    return std::span{s.tensors};
  }
}

// A Pack's scalars, the list PackArgs(first, rest) spells, `slot` long, a list
// of one being the scalar itself.  Nothing where a hop is not a list.
[[nodiscard]] std::optional<boost::container::small_vector<NodeId, 8>>
pack_arguments(const auto &s, NodeId v) {
  boost::container::small_vector<NodeId, 8> out;
  NodeId args = s.operands(v)[0];
  for (auto left = s[v].slot; left > 1; --left) {
    if (s.op_of(args) != OpCode::PackArgs) {
      return std::nullopt;
    }
    const auto [first, rest, unused] = s.operands(args);
    out.push_back(first);
    args = rest;
  }
  out.push_back(args);
  return out;
}

// How many scalars a node's value is: one for a scalar node and for the tensor
// ops that read a vector down to one; a VecConst's rows; what a Gemv's matrix
// and flag say; and otherwise the operand's -- a walk, since no node carries
// its width.  Zero for a MatConst, which is a table and never a row.  Asked of
// a sound source: ids are topological, so the walk ends.
[[nodiscard]] constexpr std::uint32_t width_of(const auto &s, NodeId v) {
  for (;;) {
    const auto [a, b, c] = s.operands(v);
    switch (s.op_of(v)) {
    case OpCode::VecConst:
      return tensors_of(s)[s[v].slot].rows;
    case OpCode::MatConst:
      return 0;
    case OpCode::Pack:
      return s[v].slot;
    case OpCode::Gemv: {
      const auto &m = tensors_of(s)[s[a].slot];
      return s[v].slot ? m.cols : m.rows;
    }
    case OpCode::Map:
    case OpCode::Zip:
    case OpCode::AddAt:
      v = a;
      break;
    case OpCode::Scale:
      v = b;
      break;
    default:
      return 1;
    }
  }
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

  // A tensor node: formed with its slot, past fold() -- no identity below
  // reads a vector -- and past the commutative swap, which is a Zip's inner
  // op's business and not Zip's.
  [[nodiscard]] constexpr NodeId make_tensor(OpCode op, NodeId a,
                                             NodeId b = no_node,
                                             std::uint32_t slot = 0) {
    assert(kind_of(op) == Kind::Tensor);
    return intern({.op = op, .a = a, .b = b, .slot = slot});
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
               std::ranges::equal(t.data, data);
      });
    }
    if (at == model_.tensors.end()) {
      model_.tensors.push_back(
          {.rows = rows, .cols = cols, .data = {data.begin(), data.end()}});
      at = std::prev(model_.tensors.end());
    }
    return intern({.op = cols == 1 ? OpCode::VecConst : OpCode::MatConst,
                   .slot = static_cast<std::uint32_t>(
                       std::distance(model_.tensors.begin(), at))});
  }
  [[nodiscard]] constexpr NodeId filled(std::uint32_t width, const T &v) {
    return tensor_const(width, 1, std::vector<T>(width, v));
  }
  [[nodiscard]] constexpr NodeId zeros(std::uint32_t width) {
    return filled(width, T{0});
  }
  [[nodiscard]] constexpr NodeId ones(std::uint32_t width) {
    return filled(width, T{1});
  }
  // What the tensor twin of a cancelled scalar is: a VecConst of nothing but
  // zero.
  [[nodiscard]] constexpr bool is_zero_tensor(NodeId id) const {
    if (model_.nodes[id].op != OpCode::VecConst) {
      return false;
    }
    if constexpr (std::equality_comparable<T>) {
      return std::ranges::all_of(model_.tensors[model_.nodes[id].slot].data,
                                 [](const T &x) { return x == T{0}; });
    } else {
      return false;
    }
  }

  // The vector of `scalars`: the list formed from the back as extern_call
  // forms its, a list of one being the scalar itself.
  [[nodiscard]] constexpr NodeId pack(std::span<const NodeId> scalars) {
    assert(!scalars.empty());
    NodeId list = scalars.back();
    for (const NodeId s : scalars | std::views::reverse | std::views::drop(1)) {
      list = intern({.op = OpCode::PackArgs, .a = s, .b = list});
    }
    return make_tensor(OpCode::Pack, list, no_node,
                       static_cast<std::uint32_t>(scalars.size()));
  }

  [[nodiscard]] constexpr NodeId make(OpCode op, NodeId a, NodeId b = no_node,
                                      NodeId c = no_node) {
    // formed with its slot, through rebuild() or a friend
    assert(!is_extern(op) && kind_of(op) != Kind::Tensor);
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
    return intern({.op = op, .a = a, .b = b, .c = c});
  }

  // The same node over remapped operands: what a rewrite that keeps every
  // operation forms.  An extern and a tensor node keep their slot, which
  // make() is never told, and no rule folds either.
  [[nodiscard]] constexpr NodeId rebuild(Node<T> n, auto &&remap) {
    for (NodeId *const u : {&n.a, &n.b, &n.c}) {
      *u = *u == no_node ? no_node : remap(*u);
    }
    return is_extern(n.op) || kind_of(n.op) == Kind::Tensor
               ? intern(n)
               : make(n.op, n.a, n.b, n.c);
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
  // The residual whose root each implicit input is, by slot; no_node for one
  // nothing closed.  And the first misuse of closes(), which no expression
  // can carry, for the equation to refuse with.
  [[nodiscard]] constexpr std::span<const NodeId> closures() const {
    return model_.closures;
  }
  [[nodiscard]] constexpr std::optional<errc> broken() const {
    return broken_;
  }
  [[nodiscard]] constexpr std::span<const ExternEntry<T>> externs() const {
    return model_.externs;
  }
  [[nodiscard]] constexpr std::span<const TensorConst<T>> tensors() const {
    return model_.tensors;
  }
  [[nodiscard]] constexpr Inputs inputs() const {
    Inputs in;
    std::ranges::transform(model_.names, in.by_kind.begin(),
                           &std::vector<std::string>::size);
    return in;
  }

  // The leaf a named input is.
  [[nodiscard]] constexpr NodeId leaf(OpCode op, std::uint32_t slot) const {
    return ids_[*named_index(op)][slot];
  }

  // Which input `name` is, if any.
  [[nodiscard]] constexpr std::optional<Input>
  lookup(std::string_view name) const {
    const auto at = std::ranges::lower_bound(by_name_, name, {}, name_of());
    return at != by_name_.end() && name_of()(*at) == name
               ? std::optional{*at}
               : std::nullopt;
  }

  [[nodiscard]] constexpr bool is_constant(NodeId id, const T &v) const {
    if (model_.nodes[id].op != OpCode::Const) {
      return false;
    }
    if constexpr (std::equality_comparable<T>) {
      return model_.nodes[id].value == v;
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
  friend constexpr RTExpression<U> named(Builder<U> &, OpCode,
                                         std::string_view);
  [[nodiscard]] constexpr NodeId input(OpCode op, std::string_view name) {
    const auto at = std::ranges::lower_bound(by_name_, name, {}, name_of());
    if (at != by_name_.end() && name_of()(*at) == name) {
      return at->op == op ? leaf(op, at->slot) : no_node;
    }
    const auto k = *named_index(op);
    const auto slot = static_cast<std::uint32_t>(model_.names[k].size());
    model_.names[k].emplace_back(name);
    ids_[k].push_back(intern({.op = op, .slot = slot}));
    by_name_.insert(at, {.op = op, .slot = slot});
    if (op == OpCode::Implicit) {
      model_.closures.push_back(no_node);
      closure_rows_.emplace_back();
    }
    return ids_[k].back();
  }

  // Implicit `z` is a root of `f`.  A misuse is kept rather than thrown: the
  // equation built over this arena refuses with it.
  template <impl::Numeric U>
  friend constexpr void closes(const RTExpression<U> &, const RTExpression<U> &);
  constexpr void close(NodeId z, NodeId f) {
    if (z == no_node || model_.nodes[z].op != OpCode::Implicit) {
      broken_ = broken_.value_or(errc::not_implicit);
    } else if (model_.closures[model_.nodes[z].slot] != no_node) {
      broken_ = broken_.value_or(errc::closed_twice);
    } else {
      model_.closures[model_.nodes[z].slot] = f;
    }
  }

  // Registering an extern is extern_fn()'s alone; its index is what the call
  // names.
  template <impl::Numeric U>
  friend ExternFunction<U> extern_fn(Builder<U> &, ExternRecord,
                                     std::type_identity_t<ExternFn<U>>);
  constexpr std::optional<std::uint32_t> add_extern(ExternEntry<T> e) {
    if (!ExternSlot::fits(model_.externs.size(), e.size())) {
      return std::nullopt;
    }
    model_.externs.push_back(std::move(e));
    return static_cast<std::uint32_t>(model_.externs.size() - 1);
  }

  // The call: the one place the list is formed, from the back, a list of one
  // being the argument itself.  The function's value; the sweeps ask for the
  // other components.
  template <impl::Numeric U> friend class ExternFunction;
  constexpr NodeId extern_call(std::uint32_t function,
                               std::span<const NodeId> args) {
    NodeId list = args.back();
    for (const NodeId a : args | std::views::reverse | std::views::drop(1)) {
      list = intern({.op = OpCode::ExternArgs, .a = a, .b = list});
    }
    return intern({.op = OpCode::Extern,
                   .a = list,
                   .slot = ExternSlot{.function = function}.packed()});
  }
  // Another component of the call `v` is: the same function and list,
  // another slot.
  template <impl::Numeric U, typename Sink>
  friend constexpr void detail::extern_partials(Builder<U> &, NodeId, Sink &&);
  constexpr NodeId extern_component(NodeId v, std::uint32_t component) {
    const Node<T> n = model_[v]; // by value: intern() may reallocate
    return intern({.op = OpCode::ExternD,
                   .a = n.a,
                   .slot = ExternSlot{.function = ExternSlot::of(n.slot).function,
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
  friend constexpr RTExpression<U> seed(Builder<U> &, std::uint32_t);
  [[nodiscard]] constexpr NodeId seed_node(std::uint32_t slot) {
    return intern({.op = OpCode::Seed, .slot = slot});
  }

  // An arena as it was, not as make() would form it again: make() folds and
  // swaps commutative operands, and the saved sweeps name subexpressions by id.
  // The one caller holds a snapshot Sound has passed, which is what checks
  // what this cannot.
  friend class Verified<T>;
  constexpr void restore(Model<T> model) {
    model_ = std::move(model);
    reindex();
  }
  // The functions a file could not carry, by name; one the lookup lacks stays
  // null, for the equation to refuse.
  constexpr void bind(auto &&lookup) {
    for (ExternEntry<T> &e : model_.externs) {
      e.fn = lookup(std::string_view{e.name});
    }
  }

  // The leaf ids and the name index, off the names and the nodes.  Interning
  // leaves exactly one leaf per name, so one walk fills every entry.
  constexpr void reindex() {
    by_name_.clear();
    closure_rows_.assign(model_.closures.size(), std::nullopt);
    for (const auto [names, ids] : std::views::zip(model_.names, ids_)) {
      ids.assign(names.size(), no_node);
    }
    for (const auto [id, n] : model_.nodes | std::views::enumerate) {
      if (const auto k = named_index(n.op)) {
        ids_[*k][n.slot] = static_cast<NodeId>(id);
        by_name_.push_back({.op = n.op, .slot = n.slot});
      }
    }
    std::ranges::sort(by_name_, {}, name_of());
    rehash();
  }

  static constexpr std::uint64_t payload_of(const Node<T> &n) {
    if (n.op != OpCode::Const) {
      return std::uint64_t{n.slot};
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
      return l.slot == r.slot;
    }
    // CFieldLike promises arithmetic, not equality.
    if constexpr (std::equality_comparable<T>) {
      return l.value == r.value;
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

  constexpr void rehash() {
    // bit_ceil, not doubling: the size is the invariant the mask needs, and
    // half full is where linear probing starts to degrade.
    table_.assign(
        std::bit_ceil(std::max<std::size_t>(64, 2 * (model_.nodes.size() + 1))),
        no_node);
    for (const auto [id, n] : model_.nodes | std::views::enumerate) {
      auto run = probe(hash_of(n));
      table_[*std::ranges::find_if(run, [this](std::size_t i) {
        return table_[i] == no_node;
      })] = static_cast<NodeId>(id);
    }
  }

  constexpr NodeId intern(const Node<T> &n) {
    if ((model_.nodes.size() + 1) * 2 > table_.size()) {
      rehash();
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
      return constant(apply<T>(op, model_.nodes[a].value, model_.nodes[b].value));
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
  // Everything below is an index over model_, rebuilt by reindex().
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
      return names(in.op)[in.slot];
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

// fadd(fmul(x, y), z) -> fma(x, y, z),
// The rule is structural -- it asks the add what its operands are and nothing
// about how many readers they have -- which is what lets the arena walk, the
// frozen graph and the kernel agree to the bit.  The lower-id operand wins
// where an add has two products, the order Builder::make already sorted
// commutative operands into.
[[nodiscard]] constexpr Contraction
contraction_at(const CNodeSource auto &nodes, NodeId v) {
  if (nodes.op_of(v) != OpCode::Add) {
    return {};
  }
  const auto ops = nodes.operands(v);
  for (const auto side : {0uz, 1uz}) {
    const NodeId u = ops[side];
    if (u == no_node) {
      continue;
    }
    const bool negated = nodes.op_of(u) == OpCode::Neg;
    const NodeId m = negated ? nodes.operands(u)[0] : u;
    if (m == no_node || nodes.op_of(m) != OpCode::Mul) {
      continue;
    }
    const auto factors = nodes.operands(m);
    return {.x = factors[0],
            .y = factors[1],
            .z = ops[1 - side],
            .negated = negated};
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
           return Step{.node = v,
                       .fma = on ? contraction_at(nodes, v) : Contraction{}};
         }) |
         impl::to<std::vector<Step>>();
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

namespace detail {
[[nodiscard]] constexpr TensorLayout
tensor_layout(const auto &s, std::span<const Step> schedule) {
  TensorLayout out;
  for (const Step &step : schedule) {
    if (!tensor_valued(s.op_of(step.node))) {
      continue;
    }
    if (out.rows.empty()) {
      out.rows.resize(s.size());
    }
    const std::uint32_t w = width_of(s, step.node);
    out.rows[step.node] = {.at = static_cast<std::uint32_t>(out.extent),
                           .width = w};
    out.extent += w;
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
    std::ranges::for_each(operands_of(b, v), mark);
  });
}

// The ids a liveness pass kept, ascending.
[[nodiscard]] constexpr auto live_ids(const std::vector<bool> &live) {
  return std::views::iota(NodeId{0}, static_cast<NodeId>(live.size())) |
         std::views::filter([&live](NodeId v) { return live[v]; });
}

} // namespace detail

} // namespace ddx::rt
