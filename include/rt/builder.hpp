#pragma once

#include "ops/algebra.hpp"
#include "rt/apply.hpp"
#include "rt/opcode.hpp"
#include "util/config.hpp"
#include "util/error.hpp"
#include "util/ranges.hpp"

#include <boost/describe/class.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstdint>
#include <iterator>
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

// Befriended by Builder: naming an input is var()'s and param()'s alone.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> var(Builder<T> &b,
                                            std::string_view name);
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> param(Builder<T> &b,
                                              std::string_view name);
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> implicit(Builder<T> &b,
                                                 std::string_view name);
template <impl::Numeric T>
constexpr void closes(const RTExpression<T> &z, const RTExpression<T> &f);
namespace detail {
struct Implicits;
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

// Nodes are interned as they are formed, so an id *is* the identity of a
// subexpression -- where the compile-time side compares types.
template <impl::Numeric T> class Builder {
public:
  using value_type = T;
  using node_type = Node<T>;

  [[nodiscard]] constexpr NodeId constant(const T &v) {
    return intern({.op = OpCode::Const, .value = v});
  }

  [[nodiscard]] constexpr NodeId make(OpCode op, NodeId a, NodeId b = no_node,
                                      NodeId c = no_node) {
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

  [[nodiscard]] constexpr const Node<T> &operator[](NodeId id) const {
    return nodes_[id];
  }

  // The CNodeSource half, which the frozen Graph answers as well.
  [[nodiscard]] constexpr OpCode op_of(NodeId id) const {
    return nodes_[id].op;
  }
  [[nodiscard]] constexpr std::array<NodeId, 3> operands(NodeId id) const {
    return {nodes_[id].a, nodes_[id].b, nodes_[id].c};
  }
  [[nodiscard]] constexpr std::size_t size() const { return nodes_.size(); }
  [[nodiscard]] constexpr std::span<const Node<T>> nodes() const {
    return nodes_;
  }
  // Each kind's names in declaration order, which is slot order: the order a
  // positional point is read in and every result is laid out in.
  [[nodiscard]] constexpr const std::vector<std::string> &symbols() const {
    return vars_.names;
  }
  [[nodiscard]] constexpr const std::vector<std::string> &params() const {
    return params_.names;
  }
  [[nodiscard]] constexpr const std::vector<std::string> &implicits() const {
    return implicits_.names;
  }
  // The residual whose root each implicit input is, by slot; no_node for one
  // nothing closed.  And the first misuse of closes(), which no expression
  // can carry, for the equation to refuse with.
  [[nodiscard]] constexpr std::span<const NodeId> closures() const {
    return closures_;
  }
  [[nodiscard]] constexpr std::optional<errc> broken() const {
    return broken_;
  }
  [[nodiscard]] constexpr Inputs inputs() const {
    return {.vars = vars_.names.size(),
            .params = params_.names.size(),
            .implicits = implicits_.names.size()};
  }

  // The leaf a named input is.
  [[nodiscard]] constexpr NodeId leaf(OpCode op, std::uint32_t slot) const {
    return named(op).ids[slot];
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
    if (nodes_[id].op != OpCode::Const) {
      return false;
    }
    if constexpr (std::equality_comparable<T>) {
      return nodes_[id].value == v;
    } else {
      return false;
    }
  }

private:
  // One namespace for every named input: a name is a symbol or a parameter,
  // never both, and naming it again hands back the same leaf.  A new name takes
  // the next slot of its kind and moves nothing already built.  no_node is a
  // name already taken by the other kind, which var() and param() poison.
  template <impl::Numeric U>
  friend constexpr RTExpression<U> var(Builder<U> &, std::string_view);
  template <impl::Numeric U>
  friend constexpr RTExpression<U> param(Builder<U> &, std::string_view);
  template <impl::Numeric U>
  friend constexpr RTExpression<U> implicit(Builder<U> &, std::string_view);
  [[nodiscard]] constexpr NodeId input(OpCode op, std::string_view name) {
    const auto at = std::ranges::lower_bound(by_name_, name, {}, name_of());
    if (at != by_name_.end() && name_of()(*at) == name) {
      return at->op == op ? named(op).ids[at->slot] : no_node;
    }
    Named &kind = named(op);
    const auto slot = static_cast<std::uint32_t>(kind.names.size());
    kind.names.emplace_back(name);
    kind.ids.push_back(intern({.op = op, .slot = slot}));
    by_name_.insert(at, {.op = op, .slot = slot});
    if (op == OpCode::Implicit) {
      closures_.push_back(no_node);
    }
    return kind.ids.back();
  }

  // Implicit `z` is a root of `f`.  A misuse is kept rather than thrown: the
  // equation built over this arena refuses with it.
  template <impl::Numeric U>
  friend constexpr void closes(const RTExpression<U> &, const RTExpression<U> &);
  constexpr void close(NodeId z, NodeId f) {
    if (z == no_node || nodes_[z].op != OpCode::Implicit) {
      broken_ = broken_.value_or(errc::not_implicit);
    } else if (closures_[nodes_[z].slot] != no_node) {
      broken_ = broken_.value_or(errc::closed_twice);
    } else {
      closures_[nodes_[z].slot] = f;
    }
  }

  // dz/ds for each implicit, by the implicit-function theorem, built the first
  // time a sweep reaches the leaf and kept: derivative.hpp's alone.
  friend struct detail::Implicits;
  std::vector<std::vector<NodeId>> closure_rows_;

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
  constexpr void restore(std::vector<Node<T>> nodes,
                         std::vector<std::string> symbols,
                         std::vector<std::string> params,
                         std::vector<std::string> implicits,
                         std::vector<NodeId> closures) {
    nodes_ = std::move(nodes);
    closures_ = std::move(closures);
    vars_.names = std::move(symbols);
    params_.names = std::move(params);
    implicits_.names = std::move(implicits);
    by_name_.clear();
    for (const OpCode op : {OpCode::Var, OpCode::Param, OpCode::Implicit}) {
      Named &kind = named(op);
      // Interning leaves exactly one leaf per name, so this walk fills every
      // entry.
      kind.ids.assign(kind.names.size(), no_node);
      for (const auto [id, n] : nodes_ | std::views::enumerate) {
        if (n.op == op) {
          kind.ids[n.slot] = static_cast<NodeId>(id);
        }
      }
      for (const std::uint32_t slot :
           std::views::iota(0u, static_cast<std::uint32_t>(kind.ids.size()))) {
        by_name_.push_back({.op = op, .slot = slot});
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
        std::bit_ceil(std::max<std::size_t>(64, 2 * (nodes_.size() + 1))),
        no_node);
    for (const auto [id, n] : nodes_ | std::views::enumerate) {
      auto run = probe(hash_of(n));
      table_[*std::ranges::find_if(run, [this](std::size_t i) {
        return table_[i] == no_node;
      })] = static_cast<NodeId>(id);
    }
  }

  constexpr NodeId intern(const Node<T> &n) {
    if ((nodes_.size() + 1) * 2 > table_.size()) {
      rehash();
    }
    // The run stops at whichever comes first, the node itself or the empty
    // slot it would take.
    auto run = probe(hash_of(n));
    const std::size_t i = *std::ranges::find_if(run, [this, &n](std::size_t k) {
      return table_[k] == no_node || same(nodes_[table_[k]], n);
    });
    if (table_[i] != no_node) {
      return table_[i];
    }
    const auto id = static_cast<NodeId>(nodes_.size());
    nodes_.push_back(n);
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
      return a != no_node && nodes_[a].op == OpCode::Neg;
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
      return nodes_[a].a;
    case Take::SquareOfA:
      // No recursion back into this rule: Mul carries no Same identity.
      return make(OpCode::Mul, a, a);
    }
    return std::nullopt;
  }

  constexpr std::optional<NodeId> fold(OpCode op, NodeId a, NodeId b) {
    const bool ca = a != no_node && nodes_[a].op == OpCode::Const;
    const bool cb = b != no_node && nodes_[b].op == OpCode::Const;
    const bool unary = arity_of(op) == 1;

    // Constant arithmetic first, through the same *_impl functor the
    // compile-time evaluator uses.
    if (unary && ca) {
      return constant(apply<T>(op, nodes_[a].value));
    } else if (!unary && ca && cb) {
      return constant(apply<T>(op, nodes_[a].value, nodes_[b].value));
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
    if (nodes_[cond].op == OpCode::Const) {
      if constexpr (std::equality_comparable<T>) {
        return nodes_[cond].value != T{0} ? t : f;
      }
    }
    return std::nullopt;
  }

  // (n/d) * d -> n.  On a DAG the denominator match is an id compare.
  constexpr std::optional<NodeId> cancel_quotient(NodeId quotient,
                                                  NodeId factor) const {
    const Node<T> &q = nodes_[quotient];
    return q.op == OpCode::Div && q.b == factor ? std::optional{q.a}
                                                : std::nullopt;
  }

  std::vector<Node<T>> nodes_;
  // Open-addressed by hand: interning runs inside constant evaluation and no
  // library hash container is constexpr.
  std::vector<NodeId> table_; // power-of-two capacity; no_node == free slot
  // One per kind of named input, slot-addressed: names[slot] and the leaf
  // ids[slot] interning made for it.
  struct Named {
    std::vector<std::string> names;
    std::vector<NodeId> ids;
  };
  Named vars_;
  Named params_;
  Named implicits_;
  // Every named input, sorted by name: a flat map by hand, since interning runs
  // inside constant evaluation and no library flat container is constexpr
  // before C++26.
  std::vector<Input> by_name_;
  std::vector<NodeId> closures_; // by implicit slot
  std::optional<errc> broken_;

  [[nodiscard]] constexpr Named &named(OpCode op) {
    return op == OpCode::Var ? vars_ : op == OpCode::Param ? params_ : implicits_;
  }
  [[nodiscard]] constexpr const Named &named(OpCode op) const {
    return op == OpCode::Var ? vars_ : op == OpCode::Param ? params_ : implicits_;
  }
  // The key by_name_ is sorted on.
  [[nodiscard]] constexpr auto name_of() const {
    return [this](const Input &in) -> std::string_view {
      return named(in.op).names[in.slot];
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
