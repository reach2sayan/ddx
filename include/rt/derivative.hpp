#pragma once

#include "ops/mode.hpp" // DiffMode
#include "rt/coupling.hpp"
#include "rt/expressions.hpp"
#include "rt/tensor.hpp"
#include "util/ranges.hpp"

#include <boost/describe/class.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <ranges>
#include <utility>
#include <vector>

namespace ddx::rt {

namespace detail {

// The rules below are written over the one handle, which reads elementwise
// where the node is a vector: a Map or Zip is its inner opcode's rule, built
// through the handle, which emits a Map or Zip again.
template <impl::Numeric T> using H = Expression<T>;

// A template, so `if constexpr` actually discards: deriv_from_value reuses the
// primal node, and d(exp)/du becomes the exp node itself.
template <typename Fn, impl::Numeric T>
[[nodiscard]] constexpr H<T> rule(const H<T> &u, const H<T> &fu) {
  if constexpr (impl::detail::has_deriv_from_value_v<Fn, H<T>>) {
    return Fn::deriv_from_value(u, fu);
  } else {
    return Fn::deriv(u);
  }
}

// Nothing here passes `f` to a binary rule: the descriptors recompute it, which
// is free on a graph since Arena::make interns it back onto the primal.
template <impl::Numeric T>
[[nodiscard]] constexpr H<T> contribution(OpCode op, const H<T> &adj,
                                          const H<T> &u, const H<T> &f) {
  switch (op) {
#define DDX_RT_CONTRIB(fn, Op, label, functor, Desc)                           \
  case OpCode::Op:                                                             \
    return impl::detail::adjoint_of<impl::detail::Desc, 0>(adj, u);
    DDX_RT_UNARY_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
    // `adj * f'(u)` is already the association DDX_UNARY_MATH_OP's generated
    // adjoints() uses.
#define DDX_RT_CONTRIB(fn, Op, label)                                          \
  case OpCode::Op:                                                             \
    return adj * rule<impl::detail::Op##Fn<H<T>>>(u, f);
    DDX_UNARY_MATH_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
  default:
    return H<T>{0};
  }
}

// What reaches operand I of a two-argument op: one operand at a time, so a
// sweep builds the rule only for an operand a derivative moves.  With a unit
// adjoint it is that operand's partial.
template <std::size_t I, impl::Numeric T>
[[nodiscard]] constexpr H<T> contribution_to(OpCode op, const H<T> &adj,
                                             const H<T> &l, const H<T> &r) {
  switch (op) {
#define DDX_RT_CONTRIB(fn, Op, label, functor, Desc)                           \
  case OpCode::Op:                                                             \
    return impl::detail::adjoint_of<impl::detail::Desc, I>(adj, l, r);
    DDX_RT_BINARY_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
  default:
    return H<T>{0};
  }
}

// Both operands' rules, `absent` for an operand that does not move; the left
// built first, always: `make` orders a commutative node's operands by id, so
// the order nodes are built in is part of the answer.
template <impl::Numeric T>
[[nodiscard]] constexpr std::pair<H<T>, H<T>>
contributions(OpCode op, const H<T> &adj, const H<T> &l, const H<T> &r,
              bool left, bool right, const H<T> &absent) {
  H<T> cl = left ? contribution_to<0>(op, adj, l, r) : absent;
  H<T> cr = right ? contribution_to<1>(op, adj, l, r) : absent;
  return {std::move(cl), std::move(cr)};
}

// The same rule seeded with a unit adjoint: Arena::make fires the x*1 rule
// before a node exists, so this is the partial and not a multiply by one.
template <impl::Numeric T>
[[nodiscard]] constexpr H<T> partial(OpCode op, const H<T> &u, const H<T> &f) {
  return contribution(op, H<T>{1}, u, f);
}

// A fold as the chain it replaces: one (running value, term) pair per binary
// step, the running value the node that chain would have had.  What a
// product's or a quotient's rules read, one step at a time; a sum's never do.
// The terms are copied out first: forming a node may reallocate the arena.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<std::pair<NodeId, NodeId>>
fold_steps(Arena<T> &b, NodeId v) {
  const OpCode op = fold_op(b.op_of(v));
  const auto terms = arguments(b, v) | impl::to<std::vector<NodeId>>();
  std::vector<std::pair<NodeId, NodeId>> steps{{terms[0], terms[1]}};
  for (const NodeId t : terms | std::views::drop(2)) {
    const auto [l, r] = steps.back();
    steps.emplace_back(H<T>::form(op, H<T>{b, l}, H<T>{b, r}).id(b), t);
  }
  return steps;
}

} // namespace detail

// One root's row of the Jacobian.  derive_jacobian() is an overload set:
// the number of roots picks the shape, as output_dim does.
struct JacobianRow {
  NodeId value = no_node;       // the node for f itself
  std::vector<NodeId> partial;  // one per symbol, in Arena::symbols() order
  std::vector<NodeId> implicit; // df/dz with every z held, one per implicit
};

template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow
build_reverse_jacobian(Arena<T> &b, NodeId root, bool ift = true);

namespace detail {
// The implicit-function theorem's half of a sweep: for implicit z closed by
// F(z, x) = 0, dz/dx_s = -F_s / F_z.  Built once per arena from one reverse
// sweep of F that holds z fixed, the first time anything differentiates
// through z, and kept on the Arena.
struct Implicits {
  template <impl::Numeric T>
  [[nodiscard]] static constexpr const std::vector<NodeId> &
  row(Arena<T> &b, std::uint32_t slot) {
    auto &row = b.closure_rows_[slot];
    if (!row) {
      const auto f = build_reverse_jacobian(b, b.closures()[slot], false);
      const Expression<T> fz{b, f.implicit[slot]};
      row = f.partial | std::views::transform([&](NodeId fs) {
              return b.is_constant(fs, T{0})
                         ? fs
                         : (-(Expression<T>{b, fs} / fz)).id(b);
            }) |
            impl::to<std::vector<NodeId>>();
    }
    return *row;
  }
};

// The chain rule's one walk through a function that differentiated itself:
// each argument of the extern node `v` with the component one order up, in
// argument order, as far as the function's order reaches.  Both sweeps take
// it, and differ only in what they do with the pair.
template <impl::Numeric T, typename Sink>
constexpr void extern_partials(Arena<T> &b, NodeId v, Sink &&sink) {
  const auto [fn, at] = detail::get<ExternRef>(b[v]);
  const ExternRecord &e = b.externs()[fn];
  for (const auto [l, arg] : arguments(b, v) | std::views::enumerate) {
    if (const auto up = e.up(at, static_cast<std::size_t>(l))) {
      sink(arg, Expression<T>{b, b.extern_component(v, *up)});
    }
  }
}

// Whether a function the roots -- or an implicit's residual -- reach supplies
// first partials only, so a second-order call has nothing to read.
template <impl::Numeric T>
[[nodiscard]] constexpr bool first_order_only(const Arena<T> &b,
                                              std::span<const NodeId> roots) {
  if (std::ranges::all_of(
          b.externs(), [](const ExternEntry<T> &e) { return e.order >= 2; })) {
    return false;
  }
  std::vector<NodeId> wanted(roots.begin(), roots.end());
  impl::append(wanted, b.closures());
  const auto live = reachable(b, wanted);
  return std::ranges::any_of(live_ids(live), [&b](NodeId v) {
    return kind_of(b[v].op) == Kind::Extern &&
           b.externs()[detail::get<ExternRef>(b[v]).function].order < 2;
  });
}

// Which of the first live.size() nodes a derivative moves: every symbol and
// implicit -- the IFT step reaches symbols outside the cone -- and whatever in
// the cone reads one.  A param, a seed, a constant or a table holds still, and
// no adjoint is built for it.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<bool>
moving(const Arena<T> &b, const std::vector<bool> &live) {
  std::vector<bool> moves(live.size(), false);
  for (const NodeId v :
       std::views::iota(NodeId{0}, static_cast<NodeId>(live.size()))) {
    const OpCode op = b.op_of(v);
    if (op == OpCode::Var || op == OpCode::Implicit) {
      moves[v] = true;
    } else if (live[v]) {
      for_each_operand(b, v,
                       [&](NodeId u) { moves[v] = moves[v] || moves[u]; });
    }
  }
  return moves;
}

// Why the implicits cannot be differentiated, if they cannot: a misuse of
// implicit(), or a residual reaching another implicit -- which would need a
// linear solve, not a quotient.
template <impl::Numeric T>
[[nodiscard]] constexpr std::optional<errc> unclosed(const Arena<T> &b) {
  if (b.broken()) {
    return b.broken();
  }
  const auto implicits =
      std::views::iota(0u, static_cast<std::uint32_t>(b.closures().size()));
  for (const std::uint32_t slot : implicits) {
    const NodeId f = b.closures()[slot];
    if (f == no_node) {
      return errc::unclosed_implicit;
    }
    if (!is_scalar(b[f])) {
      return errc::tensor_shape; // F_z is a quotient only for a scalar F
    }
    const auto live = reachable(b, std::span{&f, 1});
    if (std::ranges::any_of(implicits, [&](std::uint32_t other) {
          return other != slot && live[b.leaf(OpCode::Implicit, other)];
        })) {
      return errc::nested_implicit;
    }
  }
  return std::nullopt;
}
} // namespace detail

// The reverse sweep, accumulating *nodes*.  It appends to the arena it reads:
// new nodes land above the snapshot, so reverse id order stays topological.
//
// Beside each adjoint rides a guard: the condition under which the node lies on
// a chosen path (no_node: always).  Below a select's unchosen arm the adjoint
// is a runtime zero, and 0 * f'(u) is NaN wherever f' is infinite -- log or
// sqrt at 0 -- so every contribution out of a guarded node is itself chosen by
// the guard.  Guards are selects, read as select_impl reads a condition, and an
// unguarded graph builds exactly the nodes it did without them.  Below a Select
// over a tensor condition the guard is a tensor too: it guards children of
// its own shape and, through a Pack, each element its own; past an op that
// mixes elements it guards nothing.
//
// `ift` false holds every implicit fixed and only harvests df/dz; true -- the
// default -- folds dz/dx in at the end, adj[z] * dz/dx_s into each symbol.
// Second order needs nothing more: the partial then contains z, and the next
// sweep over it reaches z and applies the same rule.
template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow
build_reverse_jacobian(Arena<T> &b, NodeId root, bool ift) {
  const auto n = static_cast<NodeId>(b.size());
  const auto live = detail::reachable(b, std::span{&root, 1});
  // Who a contribution is built for: a node a derivative moves, and a
  // constant.  A constant's adjoint is read by nothing either, but a later
  // sweep interns against the nodes its contribution builds, and making them
  // there instead moves which product an fma takes -- the bit gate's jvp.
  const auto moving = detail::moving(b, live);
  const auto moves = [&](NodeId u) {
    return moving[u] || kind_of(b.op_of(u)) == Kind::Const;
  };
  // Two nodes per node of the cone is what a reverse pass comes to.
  b.reserve(n + 2 * static_cast<std::size_t>(std::ranges::count(live, true)));
  std::vector<NodeId> adj(n, no_node);
  std::vector<NodeId> guard(n, no_node);
  adj[root] = b.constant(T{1});

  const Expression<T> zero{0};
  const Expression<T> one{1};
  const auto expr = [&b](NodeId id) { return Expression<T>{b, id}; };
  const auto both = [&](NodeId g, const Expression<T> &cond) {
    return g == no_node ? cond : select(expr(g), cond, zero);
  };

  // A contribution under its guard, as a node of the child's width: a
  // select, which over a tensor copies the chosen row.
  const auto guarded = [&](NodeId child, NodeId g, const Expression<T> &c) {
    return both(g, c).id(b, shape_of(b, child));
  };
  const auto fits = [&b](NodeId g, NodeId child) {
    return g == no_node || !detail::is_tensor(b[g]) ||
           shape_of(b, g) == shape_of(b, child);
  };

  // `g` is the edge's guard; a child reached by an unguarded edge is always
  // on a chosen path, by any number of guarded ones on the OR of theirs.  The
  // handle's `+` is the sum: a Zip(add) where the child is a vector.  The
  // contribution is built only for a child `moves` names: a param's, a seed's
  // or a table's adjoint is read by nothing.
  const auto add_to = [&](NodeId child, NodeId g, auto &&contribution) {
    if (!moves(child)) {
      return;
    }
    g = fits(g, child) ? g : no_node;
    const NodeId c = guarded(child, g, contribution());
    // What sign(), and so abs(), hands back.  Storing it would mark the child
    // live and carry a zero adjoint down its whole cone.
    if (b.is_constant(c, T{0})) {
      return;
    }
    if (adj[child] == no_node) {
      adj[child] = c;
      guard[child] = g;
      return;
    }
    adj[child] = (expr(adj[child]) + expr(c)).id(b);
    if (guard[child] != no_node) {
      guard[child] = g == no_node
                         ? no_node
                         : select(expr(guard[child]), one, expr(g)).id(b);
    }
  };

  for (const NodeId v : detail::live_ids(live) | std::views::reverse) {
    // no_node is "nothing reached it", a folded Const 0 is "what reached it
    // cancelled"; neither contributes below.
    if (adj[v] == no_node || b.is_constant(adj[v], T{0})) {
      continue;
    }
    const auto node = b[v]; // by value: building below may reallocate
    // The adjoint has the node's shape: a vector's is a vector of its width.
    const Expression<T> a = expr(adj[v]);
    const NodeId g = guard[v];
    const OpCode op = detail::effective_op(node);
    switch (kind_of(op)) {
    case Kind::Const:
    case Kind::Input:
      break;
    case Kind::Extern:
      detail::extern_partials(b, v, [&](NodeId arg, const Expression<T> &c) {
        add_to(arg, g, [&] { return a * c; });
      });
      break;
    case Kind::Nary: {
      if (op == OpCode::SumN) { // linear: every term takes the adjoint whole
        for (const NodeId t : arguments(b, v)) {
          add_to(t, g, [&] { return a; });
        }
        break;
      }
      // The chain's rule, last step first: the adjoint reaching each running
      // value is what its step hands its left operand.
      const OpCode step = fold_op(op);
      const auto steps = detail::fold_steps(b, v);
      Expression<T> carried = a;
      for (const auto &[left, term] : steps | std::views::reverse) {
        const Expression<T> l = expr(left);
        const Expression<T> r = expr(term);
        add_to(term, g,
               [&] { return detail::contribution_to<1>(step, carried, l, r); });
        carried = detail::contribution_to<0>(step, carried, l, r);
      }
      add_to(steps.front().first, g, [&] { return carried; });
      break;
    }
    case Kind::Unary:
      add_to(node.in[0], g, [&] {
        return detail::contribution(op, a, expr(node.in[0]), expr(v));
      });
      break;
    case Kind::Ternary: {
      // Here rather than in ops/adjoints.hpp: two arms of different types is
      // what a type-encoded tree cannot hold, so there is no compile-time rule
      // to share with.  Nothing reaches the condition -- it chooses.
      const Expression<T> cond = expr(node.in[0]);
      if (moves(node.in[1])) {
        add_to(node.in[1], both(g, cond).id(b), [&] { return a; });
      }
      if (moves(node.in[2])) {
        add_to(node.in[2], both(g, select(cond, zero, one)).id(b),
               [&] { return a; });
      }
      break;
    }
    case Kind::Binary: {
      // Both rules before either sum.
      const Expression<T> l = expr(node.in[0]);
      const Expression<T> r = expr(node.in[1]);
      // A pair, not a structured binding: clang will not constant-evaluate a
      // binding captured by reference.
      const auto c = detail::contributions(op, a, l, r, moves(node.in[0]),
                                           moves(node.in[1]), zero);
      add_to(node.in[0], g, [&] { return c.first; });
      add_to(node.in[1], g, [&] { return c.second; });
      break;
    }
    case Kind::Tensor: {
      // The ops with a rule of their own; a Map or Zip took the arm above.
      switch (node.op) {
      case OpCode::VecConst:
      case OpCode::MatConst:
        break;
      case OpCode::Pack: {
        const auto args = arguments(b, v);
        const bool each = g != no_node && detail::is_tensor(b[g]);
        for (const auto [i, s] : args | std::views::enumerate) {
          const auto at = static_cast<std::size_t>(i);
          if (moves(s)) {
            add_to(s, each ? expr(g)[at].id(b) : g, [&] { return a[at]; });
          }
        }
        break;
      }
      case OpCode::Gather:
        add_to(node.in[0], g, [&] {
          return detail::add_at(expr(b.zeros(shape_of(b, node.in[0]))), a,
                                detail::get<Element>(node).value);
        });
        break;
      case OpCode::Scale:
        add_to(node.in[0], g, [&] { return dot(a, expr(node.in[1])); });
        add_to(node.in[1], g, [&] { return expr(node.in[0]) * a; });
        break;
      case OpCode::Sum:
        add_to(node.in[0], g,
               [&] { return a * expr(b.ones(shape_of(b, node.in[0]))); });
        break;
      case OpCode::Dot:
        add_to(node.in[0], g, [&] { return a * expr(node.in[1]); });
        add_to(node.in[1], g, [&] { return a * expr(node.in[0]); });
        break;
      case OpCode::AddAt:
        add_to(node.in[0], g, [&] { return a; });
        add_to(node.in[1], g,
               [&] { return a[detail::get<Element>(node).value]; });
        break;
      case OpCode::MatRow: // a table: nothing moves it
        break;
      case OpCode::MatVec: {
        // y = M v moves v by M^T a and M by a v^T; y = M^T v, by M a and v a^T.
        const bool t =
            detail::get<Orientation>(node) == Orientation::transposed;
        add_to(node.in[1], g, [&] {
          return expr(b.make(opcode<OpCode::MatVec>,
                             t ? Orientation::plain : Orientation::transposed,
                             node.in[0], adj[v]));
        });
        add_to(node.in[0], g, [&] {
          return expr(t ? b.make(OpCode::Outer, node.in[1], adj[v])
                        : b.make(OpCode::Outer, adj[v], node.in[1]));
        });
        break;
      }
      case OpCode::Outer: // u w^T: u moves by A w, w by A^T u
        add_to(node.in[0], g, [&] {
          return expr(b.make(opcode<OpCode::MatVec>, Orientation::plain, adj[v],
                             node.in[1]));
        });
        add_to(node.in[1], g, [&] {
          return expr(b.make(opcode<OpCode::MatVec>, Orientation::transposed,
                             adj[v], node.in[0]));
        });
        break;
      default: // the dense sweep's own rows, built under no root
        std::unreachable();
      }
      break;
    }
    }
  }

  const auto implicits = b.inputs().count(OpCode::Implicit);
  if (ift) {
    for (const auto slot :
         std::views::iota(0u, static_cast<std::uint32_t>(implicits))) {
      const NodeId z = b.leaf(OpCode::Implicit, slot);
      if (z >= n || adj[z] == no_node || b.is_constant(adj[z], T{0})) {
        continue;
      }
      for (const auto [s, dz] :
           detail::Implicits::row(b, slot) | std::views::enumerate) {
        add_to(b.leaf(OpCode::Var, static_cast<std::uint32_t>(s)), guard[z],
               [&] { return expr(adj[z]) * expr(dz); });
      }
    }
  }

  JacobianRow g{.value = root,
                .partial = std::vector<NodeId>(b.symbols().size(), no_node),
                .implicit = std::vector<NodeId>(implicits, no_node)};
  for (const auto [v, node] : std::views::enumerate(b.nodes().first(n))) {
    if (node.op == OpCode::Var) {
      g.partial[detail::get<Slot>(node).value] =
          adj[static_cast<std::size_t>(v)];
    } else if (node.op == OpCode::Implicit) {
      g.implicit[detail::get<Slot>(node).value] =
          adj[static_cast<std::size_t>(v)];
    }
  }
  // A symbol the expression never mentions has no leaf, so no adjoint reached
  // it; those partials are the literal zero.
  // Interned only then: a zero nothing reads would still shift every later id.
  if (std::ranges::contains(g.partial, no_node) ||
      std::ranges::contains(g.implicit, no_node)) {
    const NodeId literal = b.constant(T{0});
    std::ranges::replace(g.partial, no_node, literal);
    std::ranges::replace(g.implicit, no_node, literal);
  }
  return g;
}

// dv/ds carried up the graph in one pass, where `tangent[j]` is ds for symbol
// j.  The roots' cone only, the arena by then holding derivative blocks whose
// tangents nobody asked for.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
tangent_sweep(Arena<T> &b, std::span<const NodeId> roots,
              std::span<const NodeId> tangent) {
  const auto n = static_cast<NodeId>(b.size());
  const auto live = detail::reachable(b, roots);
  const auto moves = detail::moving(b, live);
  b.reserve(n + 2 * static_cast<std::size_t>(std::ranges::count(live, true)));
  std::vector<NodeId> d(n, no_node);
  // A node nothing moves has the zero of its shape for a tangent, made only
  // where an arm reads it; a table is never read as a value.
  const auto at = [&](NodeId u) {
    if (d[u] == no_node && !moves[u]) {
      d[u] =
          detail::is_tensor(b[u]) ? b.zeros(shape_of(b, u)) : b.constant(T{0});
    }
    return d[u];
  };
  const auto expr = [&b](NodeId id) { return Expression<T>{b, id}; };
  const auto dv = [&](NodeId u) { return Expression<T>{b, at(u)}; };
  // The terms of a product rule whose operand moves, and only those: a zero
  // term is no node on a scalar, but a tensor op is never folded, so it would
  // be computed at every point.  At least one moves, or `v` would not.
  const auto moved = [&](NodeId v, NodeId l, auto &&dl, NodeId r, auto &&dr) {
    const Shape shape = shape_of(b, v);
    if (!moves[r]) {
      return dl().id(b, shape);
    }
    return (moves[l] ? dl() + dr() : dr()).id(b, shape);
  };

  for (const NodeId v : detail::live_ids(live)) {
    const auto node = b[v]; // by value: building below may reallocate
    if (!moves[v]) {
      continue;
    }
    const OpCode op = detail::effective_op(node);
    switch (kind_of(op)) {
    case Kind::Extern: {
      Expression<T> sum{0};
      detail::extern_partials(b, v, [&](NodeId arg, const Expression<T> &c) {
        sum = sum + c * Expression<T>{b, at(arg)};
      });
      d[v] = sum.id(b);
      break;
    }
    case Kind::Nary: {
      if (op == OpCode::SumN) { // the sum of the terms' tangents, a sum again
        Expression<T> sum{0};
        for (const NodeId t : arguments(b, v)) {
          sum += Expression<T>{b, at(t)};
        }
        d[v] = sum.id(b);
        break;
      }
      // The chain's tangent, one step at a time: the running value's and,
      // where it moves, the term's, each by its partial.
      const OpCode step = fold_op(op);
      const auto steps = detail::fold_steps(b, v);
      const Expression<T> one{1};
      Expression<T> carried{b, at(steps.front().first)};
      for (const auto &[left, term] : steps) {
        const Expression<T> l{b, left};
        const Expression<T> r{b, term};
        const Expression<T> pl = detail::contribution_to<0>(step, one, l, r);
        if (!moves[term]) {
          carried = pl * carried;
          continue;
        }
        const Expression<T> pr = detail::contribution_to<1>(step, one, l, r);
        carried = pl * carried + pr * Expression<T>{b, at(term)};
      }
      d[v] = carried.id(b);
      break;
    }
    case Kind::Const: // holds still, above
      std::unreachable();
    case Kind::Input: // a symbol or an implicit: a param or a seed holds still
      if (node.op == OpCode::Implicit) {
        // dz = sum_s dz/dx_s dx_s: z moves with every symbol its residual does.
        const auto dz =
            detail::Implicits::row(b, detail::get<Slot>(node).value);
        d[v] = std::ranges::fold_left(
                   std::views::zip(dz, tangent) |
                       std::views::transform([&b](const auto &pair) {
                         const auto [zs, ts] = pair;
                         return Expression<T>{b, zs} * Expression<T>{b, ts};
                       }),
                   Expression<T>{0}, std::plus<>{})
                   .id(b);
      } else {
        d[v] = tangent[detail::get<Slot>(node).value];
      }
      break;
    case Kind::Unary: {
      const Expression<T> u{b, node.in[0]};
      const Expression<T> self{b, v};
      d[v] = (detail::partial(op, u, self) * Expression<T>{b, at(node.in[0])})
                 .id(b);
      break;
    }
    case Kind::Ternary:
      // select(c, dt, df): the condition picks the derivative as it picks
      // the value.
      d[v] =
          select(Expression<T>{b, node.in[0]}, Expression<T>{b, at(node.in[1])},
                 Expression<T>{b, at(node.in[2])})
              .id(b);
      break;
    case Kind::Binary: {
      // Both partials before either product, as in the reverse sweep.
      const Expression<T> l{b, node.in[0]};
      const Expression<T> r{b, node.in[1]};
      const Expression<T> one{1};
      const auto p = detail::contributions(op, one, l, r, moves[node.in[0]],
                                           moves[node.in[1]], one);
      d[v] = moved(
          v, node.in[0], [&] { return p.first * dv(node.in[0]); }, node.in[1],
          [&] { return p.second * dv(node.in[1]); });
      break;
    }
    case Kind::Tensor: {
      // The forward twin of each reverse rule above; a vector's tangent is a
      // vector of its width.  A constant or a table holds still, above.
      switch (node.op) {
      case OpCode::Pack: {
        const auto args = arguments(b, v);
        const auto tangents =
            args | std::views::transform([&](NodeId s) { return at(s); }) |
            impl::to<boost::container::small_vector<NodeId, 8>>();
        d[v] = b.pack(tangents);
        break;
      }
      case OpCode::Gather:
        d[v] = dv(node.in[0])[detail::get<Element>(node).value].id(b);
        break;
      case OpCode::Sum:
        d[v] = sum(dv(node.in[0])).id(b);
        break;
      case OpCode::AddAt:
        d[v] = b.make(opcode<OpCode::AddAt>, detail::get<Element>(node),
                      at(node.in[0]), at(node.in[1]));
        break;
      case OpCode::Scale: // the bilinear ops: the node with one operand's
      case OpCode::Dot:   // tangent in that operand's slot
      case OpCode::MatVec:
      case OpCode::Outer: {
        const auto with = [&](std::size_t slot) {
          return [&, slot] {
            return expr(b.replaced(v, slot, at(node.in[slot])));
          };
        };
        d[v] = moved(v, node.in[0], with(0), node.in[1], with(1));
        break;
      }
      default: // the dense sweep's own rows, built under no root
        std::unreachable();
      }
      break;
    }
    }
  }

  return roots | std::views::transform([&](NodeId r) { return at(r); }) |
         impl::to<std::vector<NodeId>>();
}

// One sweep per symbol, with a unit tangent in each.  Here to check Reverse
// against, which produces fewer nodes on everything but a single variable.
template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow build_symbolic_jacobian(Arena<T> &b,
                                                            NodeId root) {
  const auto nsym = b.symbols().size();
  JacobianRow g{.value = root,
                .partial = std::vector<NodeId>(nsym, no_node),
                .implicit = {}};
  const std::array<NodeId, 1> roots{root};

  // One vector, the unit moved through it: interning makes both constants
  // stable ids, created in the order the per-sweep spelling did.
  const NodeId one = b.constant(T{1});
  const NodeId zero = b.constant(T{0});
  std::vector<NodeId> unit(nsym, zero);
  for (const std::size_t s : std::views::iota(0uz, nsym)) {
    unit[s] = one;
    g.partial[s] = tangent_sweep(b, roots, unit).front();
    unit[s] = zero;
  }
  return g;
}

// m functions over the same symbols: m sweeps sharing one graph, rows stacked
// in function order to match Equation::jacobian.  A partial that folded to the
// literal zero node costs no output column; `pattern` puts the rest at (i, j).
struct Jacobian {
  std::vector<NodeId> value;   // m
  std::vector<NodeId> partial; // pattern.nonzeros(), row-major by function
  Sparsity pattern;
  NodeId zero = no_node; // what a cell outside the pattern reads

  [[nodiscard]] constexpr NodeId at(std::size_t i, std::size_t j) const {
    return pattern.at(i, j)
        .transform([this](std::size_t k) { return partial[k]; })
        .value_or(zero);
  }
  BOOST_DESCRIBE_CLASS(Jacobian, (), (value, partial, pattern, zero), (), ())
};

template <impl::Numeric T>
[[nodiscard]] constexpr Jacobian
derive_jacobian(Arena<T> &b, std::span<const NodeId> roots) {
  Jacobian j{
      .value = {roots.begin(), roots.end()},
      .partial = {},
      .pattern = {.rowptr = {0}, .col = {}, .columns = b.symbols().size()},
      .zero = b.constant(T{0})};
  j.pattern.rowptr.reserve(roots.size() + 1);
  // The previous rows' nodes are unreachable from this root, so the sweep
  // skips them; what they provide is subexpressions to share.
  for (const NodeId r : roots) {
    const auto g = build_reverse_jacobian(b, r);
    // The predicate the sweep itself refuses a contribution by, so the two
    // cannot drift -- and it reads -0.0 the same way.
    auto nonzero = g.partial | std::views::enumerate |
                   std::views::filter([&b](const auto &cell) {
                     return !b.is_constant(std::get<1>(cell), T{0});
                   });
    for (const auto [symbol, node] : nonzero) {
      j.partial.push_back(node);
      j.pattern.col.push_back(static_cast<std::uint32_t>(symbol));
    }
    j.pattern.rowptr.push_back(j.partial.size());
  }
  return j;
}

// One root rather than a span: a span is not constructible from a NodeId, so
// this and the m-root overload never compete.
template <impl::DiffMode Mode = impl::DiffMode::Reverse, impl::Numeric T>
[[nodiscard]] constexpr JacobianRow derive_jacobian(Arena<T> &b, NodeId root) {
  if constexpr (Mode == impl::DiffMode::Symbolic) {
    return build_symbolic_jacobian(b, root);
  } else {
    return build_reverse_jacobian(b, root);
  }
}

namespace detail {
// The arena f is differentiated in, or why it cannot be.
template <impl::Numeric T>
[[nodiscard]] result<Arena<T> *> differentiable(const Expression<T> &f) {
  Arena<T> *const b = arena_of<T>({&f, 1});
  const auto why = b == nullptr        ? std::optional{errc::outside_equation}
                   : Router::scalar(f) ? Router::why(f)
                                       : errc::tensor_shape;
  return why ? fail(*why) : result<Arena<T> *>{b};
}
} // namespace detail

// df/dx for every symbol, in symbols() order, as nodes to build on: sound in
// an arena still being built, the sweep's nodes landing above f's.  Each one
// poisoned, as f is, where f cannot be differentiated.
template <impl::Numeric T>
[[nodiscard]] std::vector<Expression<T>> gradient(const Expression<T> &f) {
  const auto b = detail::differentiable(f);
  if (!b) {
    const Arena<T> *const at = detail::arena_of<T>({&f, 1});
    return std::vector(at ? at->symbols().size() : 1uz,
                       detail::Router::poison<T>(b.error().code));
  }
  return derive_jacobian(**b, f.id(**b)).partial |
         std::views::transform([&b](NodeId p) {
           return p == no_node ? Expression<T>{T{0}} : Expression<T>{**b, p};
         }) |
         impl::to<std::vector<Expression<T>>>();
}
// One symbol's, by name: unknown_symbol for anything but a var.
template <impl::Numeric T>
[[nodiscard]] Expression<T> derivative(const Expression<T> &f,
                                       std::string_view symbol) {
  const auto in = detail::differentiable(f).and_then(
      [&](const Arena<T> *b) { return b->symbol(symbol); });
  return in ? gradient(f)[*in] : detail::Router::poison<T>(in.error().code);
}

// One sweep per colour.  Sweeping from the sum of a colour's partials gives,
// for row i, the sum over j in the colour of d2f/dxi dxj -- of which the
// colouring guarantees at most one is not structurally zero.
struct Hessian {
  NodeId value = no_node;
  std::vector<NodeId> partial;    // n
  std::vector<NodeId> compressed; // colours * n
  Colouring colouring;
  NodeId zero = no_node;

  // Scattered on read, so a caller never sees the compressed form: a cell no
  // column owns has no storage and answers the colouring's structural zero.
  [[nodiscard]] constexpr NodeId at(std::size_t i, std::size_t j) const {
    return colouring.cell_of(i, j)
        .transform([this](std::size_t k) { return compressed[k]; })
        .value_or(zero);
  }
  [[nodiscard]] constexpr std::size_t colours() const {
    return colouring.count();
  }
  BOOST_DESCRIBE_CLASS(Hessian, (),
                       (value, partial, compressed, colouring, zero), (), ())
};

template <impl::Numeric T>
[[nodiscard]] Hessian derive_hessian(Arena<T> &b, NodeId root) {
  const auto g = build_reverse_jacobian(b, root);
  const auto rows = coupling_pattern(b, root);
  Hessian h{.value = g.value,
            .partial = g.partial,
            .compressed = {},
            .colouring = colour_columns(rows),
            .zero = b.constant(T{0})};

  // Only the cells a column owns: the rest cost an output column each and a
  // whole cone of nodes behind it, which on an arrow Hessian is most of it.
  h.compressed.assign(h.colouring.width(), h.zero);
  for (const std::size_t c : h.colouring.colours()) {
    // Summing a colour's partials before the sweep is what makes one sweep do
    // the work of |colour| of them.
    auto make_partial_expression = [&](std::size_t j) {
      return Expression<T>{b, h.partial[j]};
    };
    const auto seed = std::ranges::fold_left(
        h.colouring.columns_of(c) |
            std::views::transform(std::move(make_partial_expression)),
        Expression<T>{0}, std::plus<>{});
    const auto row = build_reverse_jacobian(b, seed.id(b));
    for (const Cell &owned : h.colouring.owned_cells(c)) {
      h.compressed[owned.slot] = row.partial[owned.row];
    }
  }
  return h;
}

namespace detail {
// Σ_j seed_j * ids[j], the re-rooting every seeded product sweeps from.  A
// structurally-zero partial is the literal zero node, so Mul/ZeroB and Add/Zero
// drop its term before either node exists.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> seeded_sum(Arena<T> &b,
                                                 std::span<const NodeId> ids) {
  auto term = [&b, ids](std::size_t j) {
    return seed(b, static_cast<std::uint32_t>(j)) * Expression<T>{b, ids[j]};
  };
  return std::ranges::fold_left(std::views::iota(0uz, ids.size()) |
                                    std::views::transform(std::move(term)),
                                Expression<T>{0}, std::plus<>{});
}
} // namespace detail

// H(x)v without ever forming H.  Sweeping from s = Sum_j v_j df/dx_j gives
// ds/dx_i = Sum_j v_j d2f/dx_i dx_j = (Hv)_i, so the direction goes into the
// same fold the colouring above sums and one sweep answers.
// Pearlmutter, "Fast Exact Multiplication by the Hessian", Neural Computation
// 6(1) (1994) 147-160.
struct HessianVector {
  NodeId value = no_node;
  std::vector<NodeId> partial; // n -- the gradient, which comes free
  std::vector<NodeId> product; // n -- H v
  BOOST_DESCRIBE_CLASS(HessianVector, (), (value, partial, product), (), ())
};

template <impl::Numeric T>
[[nodiscard]] constexpr HessianVector derive_hvp(Arena<T> &b, NodeId root) {
  const auto g = build_reverse_jacobian(b, root);
  const auto along = detail::seeded_sum(b, g.partial);
  return {.value = g.value,
          .partial = g.partial,
          .product = build_reverse_jacobian(b, along.id(b)).partial};
}

// w'J, the same re-rooting one order down: sweeping from s = Sum_i w_i f_i
// gives ds/dx_j = (w'J)_j.  One sweep and n columns, where jacobian() is m
// sweeps and pattern.nonzeros() columns; the seed slots here are counted in
// functions, not symbols.  n nodes, w'J.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
derive_vjp(Arena<T> &b, std::span<const NodeId> roots) {
  const auto along = detail::seeded_sum(b, roots);
  return build_reverse_jacobian(b, along.id(b)).partial;
}

namespace detail {
// One seed per symbol: the direction a forward sweep carries.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId> seeds_along(Arena<T> &b) {
  return std::views::iota(0u, static_cast<std::uint32_t>(b.symbols().size())) |
         std::views::transform(
             [&b](std::uint32_t j) { return seed(b, j).id(b); }) |
         impl::to<std::vector<NodeId>>();
}
} // namespace detail

// J v, the one product that is a forward sweep: one pass, a node per root,
// whatever n is.  Rooted at the Jacobian's cells it is dJ/dv, every
// d2 f_i / dx_j dv in a single sweep, with the Jacobian's layout.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
derive_jvp(Arena<T> &b, std::span<const NodeId> roots) {
  return tangent_sweep(b, roots, detail::seeds_along(b));
}

} // namespace ddx::rt
