#pragma once

#include "ops/mode.hpp"
#include "rt/coupling.hpp"
#include "rt/expressions.hpp"
#include "rt/tensor.hpp"
#include "util/ranges.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

namespace ddx::rt {

namespace detail {

// The rules are written over Expression, which works elementwise on vectors,
// so a Map or Zip uses its inner opcode's rule and yields a Map or Zip again.
template <impl::Numeric T> using H = Expression<T>;

// A template so that `if constexpr` discards; deriv_from_value reuses the
// primal node, making d(exp)/du the exp node itself.
template <typename Fn, impl::Numeric T>
[[nodiscard]] constexpr H<T> rule(const H<T> &u, const H<T> &fu) {
  if constexpr (impl::detail::has_deriv_from_value_v<Fn, H<T>>) {
    return Fn::deriv_from_value(u, fu);
  } else {
    return Fn::deriv(u);
  }
}

// Binary rules recompute `f` instead of taking it, which costs nothing here
// because Arena::make interns it back onto the primal node.
template <impl::Numeric T>
[[nodiscard]] constexpr H<T> contribution(OpCode op, const H<T> &adj,
                                          const H<T> &u, const H<T> &f) {
  switch (op) {
#define DDX_RT_CONTRIB(fn, Op, label, functor, Desc)                           \
  case OpCode::Op:                                                             \
    return impl::detail::adjoint_of<impl::detail::Desc, 0>(adj, u);
    DDX_RT_UNARY_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
    // `adj * f'(u)` matches the association of DDX_UNARY_MATH_OP's adjoints().
#define DDX_RT_CONTRIB(fn, Op, label)                                          \
  case OpCode::Op:                                                             \
    return adj * rule<impl::detail::Op##Fn<H<T>>>(u, f);
    DDX_UNARY_MATH_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
  default:
    return H<T>{0};
  }
}

// The adjoint reaching operand I of a binary op, built per operand so a sweep
// skips operands that do not move. With a unit adjoint it is the partial.
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

// Which operands of a binary node move.
struct Moving {
  bool left = false;
  bool right = false;
};

// Both operands' rules, `absent` for one that does not move. Left is always
// built first, since node build order affects ids and so the bits.
template <impl::Numeric T>
[[nodiscard]] constexpr std::pair<H<T>, H<T>>
contributions(OpCode op, const H<T> &adj, const H<T> &l, const H<T> &r,
              Moving moving, const H<T> &absent) {
  H<T> cl = moving.left ? contribution_to<0>(op, adj, l, r) : absent;
  H<T> cr = moving.right ? contribution_to<1>(op, adj, l, r) : absent;
  return {std::move(cl), std::move(cr)};
}

// The rule with a unit adjoint; Arena::make folds the x*1, so this yields the
// partial itself rather than a multiply by one.
template <impl::Numeric T>
[[nodiscard]] constexpr H<T> partial(OpCode op, const H<T> &u, const H<T> &f) {
  return contribution(op, H<T>{1}, u, f);
}

// A product or quotient fold as its binary chain of (running value, term)
// pairs. Terms are copied first, since forming a node may reallocate.
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

// One root's row of the Jacobian.
struct JacobianRow {
  NodeId value = no_node;       // The node for f itself.
  std::vector<NodeId> partial;  // One per symbol, in Arena::symbols() order.
  std::vector<NodeId> implicit; // df/dz with every z held, one per implicit.
};

namespace detail {
// At an implicit z a reverse sweep either solves through it, folding
// adj[z] * dz/dx into each symbol, or holds z and only collects df/dz.
enum class AtImplicits : bool { solved, held };
template <AtImplicits At, impl::Numeric T>
[[nodiscard]] constexpr JacobianRow reverse_sweep(Arena<T> &b, NodeId root);
} // namespace detail

template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow build_reverse_jacobian(Arena<T> &b,
                                                           NodeId root) {
  return detail::reverse_sweep<detail::AtImplicits::solved>(b, root);
}

namespace detail {
// For implicit z with F(z, x) = 0, dz/dx_s = -F_s / F_z, built once per arena
// from a reverse sweep of F holding z fixed, and kept on the Arena.
struct Implicits {
  template <impl::Numeric T>
  [[nodiscard]] static constexpr const std::vector<NodeId> &row(Arena<T> &b,
                                                                Slot slot) {
    auto &row = b.closure_rows_[slot.value];
    if (!row) {
      const auto f =
          reverse_sweep<AtImplicits::held>(b, b.closures()[slot.value]);
      const Expression<T> fz{b, f.implicit[slot.value]};
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

// Calls `sink` with each argument of extern node `v` and the component one
// order up, as far as the function's order reaches. Both sweeps use it.
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

// Whether an extern reachable from the roots or a residual supplies only
// first partials, leaving a second-order call nothing to read.
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

// Which nodes a derivative moves: every symbol and implicit, and whatever in
// the cone reads one. No adjoint is built for a node that holds still.
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

// Why the implicits cannot be differentiated, if so: a misuse of implicit(),
// or a residual reaching another implicit, which would need a linear solve.
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
      return errc::tensor_shape; // F_z is a quotient only for a scalar F.
    }
    const auto live = reachable(b, std::span{&f, 1});
    if (std::ranges::any_of(implicits, [&](std::uint32_t other) {
          return other != slot && live[b.leaf(OpCode::Implicit, Slot{other})];
        })) {
      return errc::nested_implicit;
    }
  }
  return std::nullopt;
}
} // namespace detail

namespace detail {
// The condition under which a node lies on a chosen select path, so that
// 0 * inf below an unchosen arm cannot produce NaN. no_node means always.
struct Guard {
  NodeId cond = no_node;
  [[nodiscard]] constexpr bool always() const noexcept {
    return cond == no_node;
  }
};
} // namespace detail

// The reverse sweep, appending adjoint nodes above the cone so reverse id order
// stays topological. Solved implicits fold adj[z] * dz/dx_s in at the end.
template <detail::AtImplicits At, impl::Numeric T>
[[nodiscard]] constexpr JacobianRow detail::reverse_sweep(Arena<T> &b,
                                                          NodeId root) {
  const auto n = static_cast<NodeId>(b.size());
  const auto live = detail::reachable(b, std::span{&root, 1});
  // Contributions are built for moving nodes and also constants, whose nodes
  // later sweeps intern against; skipping them changes fma choices and bits.
  const auto moving = detail::moving(b, live);
  const auto moves = [&](NodeId u) {
    return moving[u] || kind_of(b.op_of(u)) == Kind::Const;
  };
  // A reverse pass adds about two nodes per node of the cone.
  b.reserve(n + 2 * static_cast<std::size_t>(std::ranges::count(live, true)));
  std::vector adj(n, no_node);
  std::vector<Guard> guard(n);
  adj[root] = b.constant(T{1});

  const Expression<T> zero{0};
  const Expression<T> one{1};
  const auto expr = [&b](NodeId id) { return Expression<T>{b, id}; };
  const auto both = [&](Guard g, const Expression<T> &cond) {
    return g.always() ? cond : select(expr(g.cond), cond, zero);
  };

  // A contribution under its guard, as a node of the child's width.
  const auto guarded = [&](NodeId child, Guard g, const Expression<T> &c) {
    return both(g, c).id(b, shape_of(b, child));
  };
  const auto fits = [&b](Guard g, NodeId child) {
    return g.always() || !detail::is_tensor(b[g.cond]) ||
           (detail::is_tensor(b[child]) &&
            shape_of(b, g.cond) == shape_of(b, child));
  };

  // The OR of two guards: a child reached by any unguarded edge is always on a
  // chosen path.
  const auto joined = [&](Guard held, Guard g) {
    return held.always() || g.always()
               ? Guard{}
               : Guard{select(expr(held.cond), one, expr(g.cond)).id(b)};
  };
  // Per vector, element reads summed as scalars and packed once at the flush,
  // since an AddAt per read costs n^2 on a model indexing every element.
  struct Gathered {
    NodeId base = no_node;
    Guard guard{};    // The vector's guard when the reads began.
    Guard reads{};    // The OR of the reads' guards.
    bool open = true; // False once a read or the base drops the guard.
    std::vector<NodeId> element;
  };
  // The guard add_to would have left: the vector's own, if it had an adjoint,
  // ORed with every read's.
  const auto guard_of = [&](const Gathered &row) {
    if (!row.open) {
      return Guard{};
    }
    return row.base == no_node ? row.reads : joined(row.guard, row.reads);
  };
  std::vector<Gathered> gathered(n);
  const auto flush = [&](NodeId v) {
    if (v >= gathered.size() || gathered[v].element.empty()) {
      return;
    }
    Gathered &row = gathered[v];
    for (const auto [k, e] : row.element | std::views::enumerate) {
      if (e == no_node) {
        e = row.base == no_node
                ? b.constant(T{0})
                : expr(row.base)[static_cast<std::size_t>(k)].id(b);
      }
    }
    adj[v] = b.pack(row.element);
    guard[v] = guard_of(row);
    row = {};
  };
  // Records one element read, guarding it and joining the guard as add_to
  // would.
  const auto gather = [&](NodeId from, std::size_t k, const Expression<T> &a,
                          std::size_t rows, Guard g) {
    Gathered &row = gathered[from];
    const bool first = row.element.empty();
    if (first) {
      row.base = adj[from];
      row.guard = guard[from];
      row.open = row.base == no_node || !row.guard.always();
      row.element.assign(rows, no_node);
    }
    const Expression<T> c = g.always() ? a : select(expr(g.cond), a, zero);
    NodeId &e = row.element[k];
    e = e != no_node          ? (expr(e) + c).id(b)
        : row.base != no_node ? (expr(row.base)[k] + c).id(b)
                              : c.id(b);
    row.open = row.open && !g.always();
    if (row.open) {
      row.reads = first ? g : joined(row.reads, g);
    }
  };

  const auto add_to = [&](NodeId child, Guard g, auto &&contribution) {
    if (!moves(child)) {
      return;
    }
    flush(child);
    g = fits(g, child) ? g : Guard{};
    const NodeId c = guarded(child, g, contribution());
    // A literal zero, as sign() and abs() yield; storing it would carry a zero
    // adjoint down the child's whole cone.
    if (b.is_constant(c, T{0})) {
      return;
    }
    if (adj[child] == no_node) {
      adj[child] = c;
      guard[child] = g;
      return;
    }
    adj[child] = (expr(adj[child]) + expr(c)).id(b);
    guard[child] = joined(guard[child], g);
  };

  // dot(A, M) for A = N (.) u w^T is u . (M (.) N) w, never forming A. Whether
  // a node contracts is decided first from ids, so failures build nothing.
  std::vector<bool> contracts;
  const auto contractible = [&](NodeId id) {
    for (auto u = static_cast<NodeId>(contracts.size()); u <= id; ++u) {
      const Node<T> &at = b[u];
      const OpCode op =
          at.op == OpCode::Zip ? detail::get<OpCode>(at) : OpCode::Const;
      contracts.push_back(
          (at.op == OpCode::Outer && shape_of(b, u).cols > 1) ||
          (op == OpCode::Add && contracts[at.in[0]] && contracts[at.in[1]]) ||
          (op == OpCode::Mul && (contracts[at.in[0]] || contracts[at.in[1]])));
    }
    return contracts[id];
  };
  const auto contracted =
      [&](this const auto &self, NodeId id,
          const Expression<T> &m) -> std::optional<Expression<T>> {
    if (!contractible(id)) {
      return std::nullopt;
    }
    // A copy, since building below may reallocate.
    const Node<T> at = b[id];
    if (at.op == OpCode::Outer) {
      return dot(expr(at.in[0]), gemv(m, expr(at.in[1])));
    }
    if (detail::get<OpCode>(at) == OpCode::Add) {
      const auto l = *self(at.in[0], m); // Left first: build order sets bits.
      return l + *self(at.in[1], m);
    }
    return contractible(at.in[1]) ? self(at.in[1], m * expr(at.in[0]))
                                  : self(at.in[0], m * expr(at.in[1]));
  };

  for (const NodeId v : detail::live_ids(live) | std::views::reverse) {
    // A Pack whose adjoint is only element reads passes each read straight to
    // its argument, rather than building a vector to take apart.
    if (v < gathered.size() && !gathered[v].element.empty() &&
        gathered[v].base == no_node && b.op_of(v) == OpCode::Pack) {
      const auto row = std::move(gathered[v].element);
      const Guard g = guard_of(gathered[v]);
      gathered[v] = {};
      for (const auto [k, s] : arguments(b, v) | std::views::enumerate) {
        if (const NodeId e = row[static_cast<std::size_t>(k)]; e != no_node) {
          add_to(s, g, [&] { return expr(e); });
        }
      }
      continue;
    }
    flush(v);
    // Skip a node that nothing reached (no_node) or whose adjoint cancelled.
    if (adj[v] == no_node || b.is_constant(adj[v], T{0})) {
      continue;
    }
    const auto node = b[v]; // A copy, since building below may reallocate.
    // The adjoint has the node's shape.
    const Expression<T> a = expr(adj[v]);
    const Guard g = guard[v];
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
      if (op == OpCode::SumN) { // Linear: every term takes the whole adjoint.
        for (const NodeId t : arguments(b, v)) {
          add_to(t, g, [&] { return a; });
        }
        break;
      }
      // The chain rule, last step first: each running value's adjoint is what
      // its step hands its left operand.
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
      // Not in ops/adjoints.hpp, since a type-encoded tree cannot hold arms of
      // different types. The condition receives no adjoint.
      const auto [c, then, otherwise] =
          detail::operands<OpCode::Select>(node.in);
      const Expression<T> cond = expr(c);
      if (moves(then)) {
        add_to(then, Guard{both(g, cond).id(b)}, [&] { return a; });
      }
      if (moves(otherwise)) {
        add_to(otherwise, Guard{both(g, select(cond, zero, one)).id(b)},
               [&] { return a; });
      }
      break;
    }
    case Kind::Binary: {
      // Build both rules before either sum.
      const Expression<T> l = expr(node.in[0]);
      const Expression<T> r = expr(node.in[1]);
      // A pair, not a structured binding, because clang will not
      // constant-evaluate a binding captured by reference.
      const auto c = detail::contributions(
          op, a, l, r, {.left = moves(node.in[0]), .right = moves(node.in[1])},
          zero);
      add_to(node.in[0], g, [&] { return c.first; });
      add_to(node.in[1], g, [&] { return c.second; });
      break;
    }
    case Kind::Tensor: {
      // Ops with their own rule; a Map or Zip used its inner op's rule above.
      switch (node.op) {
      case OpCode::VecConst:
      case OpCode::MatConst:
        break;
      case OpCode::Pack: {
        const auto args = arguments(b, v);
        const bool each = !g.always() && detail::is_tensor(b[g.cond]);
        for (const auto [i, s] : args | std::views::enumerate) {
          const auto at = static_cast<std::size_t>(i);
          if (moves(s)) {
            add_to(s, each ? Guard{expr(g.cond)[at].id(b)} : g,
                   [&] { return a[at]; });
          }
        }
        break;
      }
      case OpCode::Gather: {
        const NodeId from = node.in[0];
        const Shape shape = shape_of(b, from);
        if ((!g.always() && detail::is_tensor(b[g.cond])) || shape.cols != 1 ||
            !moves(from)) {
          add_to(from, g, [&] {
            return detail::add_at(expr(b.zeros(shape)), a,
                                  detail::get<Element>(node).value);
          });
          break;
        }
        gather(from, detail::get<Element>(node).value, a, shape.rows, g);
        break;
      }
      case OpCode::Scale: {
        const auto [scalar, vec] = detail::operands<OpCode::Scale>(node);
        add_to(scalar, g, [&] {
          return contracted(adj[v], expr(vec)).value_or(dot(a, expr(vec)));
        });
        add_to(vec, g, [&] { return expr(scalar) * a; });
        break;
      }
      case OpCode::Sum:
        add_to(node.in[0], g,
               [&] { return a * expr(b.ones(shape_of(b, node.in[0]))); });
        break;
      case OpCode::Dot:
        add_to(node.in[0], g, [&] { return a * expr(node.in[1]); });
        add_to(node.in[1], g, [&] { return a * expr(node.in[0]); });
        break;
      case OpCode::AddAt: {
        const auto [vec, scalar] = detail::operands<OpCode::AddAt>(node);
        add_to(vec, g, [&] { return a; });
        add_to(scalar, g, [&] { return a[detail::get<Element>(node).value]; });
        break;
      }
      case OpCode::MatRow: // A table, which nothing moves.
        break;
      case OpCode::MatVec: {
        // y = M v moves v by M^T a and M by a v^T; y = M^T v, by M a and v a^T.
        const bool t =
            detail::get<Orientation>(node) == Orientation::transposed;
        const auto [matrix, vec] = detail::operands<OpCode::MatVec>(node);
        add_to(vec, g, [&] {
          return expr(b.make(opcode<OpCode::MatVec>,
                             t ? Orientation::plain : Orientation::transposed,
                             {.matrix = matrix, .vec = adj[v]}));
        });
        add_to(matrix, g, [&] {
          return expr(
              t ? b.make(opcode<OpCode::Outer>, {.left = vec, .right = adj[v]})
                : b.make(opcode<OpCode::Outer>,
                         {.left = adj[v], .right = vec}));
        });
        break;
      }
      case OpCode::Outer: { // u w^T: u moves by A w, and w by A^T u.
        const auto [u, w] = detail::operands<OpCode::Outer>(node);
        add_to(u, g, [&] {
          return expr(b.make(opcode<OpCode::MatVec>, Orientation::plain,
                             {.matrix = adj[v], .vec = w}));
        });
        add_to(w, g, [&] {
          return expr(b.make(opcode<OpCode::MatVec>, Orientation::transposed,
                             {.matrix = adj[v], .vec = u}));
        });
        break;
      }
      default: // Only the dense sweep builds these, never under a root.
        std::unreachable();
      }
      break;
    }
    }
  }

  const auto implicits = b.inputs().count(OpCode::Implicit);
  if constexpr (At == detail::AtImplicits::solved) {
    for (const auto slot :
         std::views::iota(0u, static_cast<std::uint32_t>(implicits))) {
      const NodeId z = b.leaf(OpCode::Implicit, Slot{slot});
      if (z >= n || adj[z] == no_node || b.is_constant(adj[z], T{0})) {
        continue;
      }
      for (const auto [s, dz] :
           detail::Implicits::row(b, Slot{slot}) | std::views::enumerate) {
        add_to(b.leaf(OpCode::Var, Slot{static_cast<std::uint32_t>(s)}),
               guard[z], [&] { return expr(adj[z]) * expr(dz); });
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
  // Unreached symbols get the literal zero, interned only when needed, since an
  // unused node would still shift every later id.
  if (std::ranges::contains(g.partial, no_node) ||
      std::ranges::contains(g.implicit, no_node)) {
    const NodeId literal = b.constant(T{0});
    std::ranges::replace(g.partial, no_node, literal);
    std::ranges::replace(g.implicit, no_node, literal);
  }
  return g;
}

// Forward mode: dv/ds carried up the roots' cone in one pass, where
// `tangent[j]` is ds for symbol j.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
tangent_sweep(Arena<T> &b, std::span<const NodeId> roots,
              std::span<const NodeId> tangent) {
  const auto n = static_cast<NodeId>(b.size());
  const auto live = detail::reachable(b, roots);
  const auto moves = detail::moving(b, live);
  b.reserve(n + 2 * static_cast<std::size_t>(std::ranges::count(live, true)));
  std::vector<NodeId> d(n, no_node);
  // A node that holds still has a zero tangent of its shape, made only when an
  // arm reads it.
  const auto at = [&](NodeId u) {
    if (d[u] == no_node && !moves[u]) {
      d[u] =
          detail::is_tensor(b[u]) ? b.zeros(shape_of(b, u)) : b.constant(T{0});
    }
    return d[u];
  };
  const auto expr = [&b](NodeId id) { return Expression<T>{b, id}; };
  const auto dv = [&](NodeId u) { return Expression<T>{b, at(u)}; };
  // Only the product-rule terms whose operand moves, since a zero tensor term
  // would not fold away. At least one operand moves, or `v` would not.
  const auto moved = [&](NodeId v, NodeId l, auto &&dl, NodeId r, auto &&dr) {
    const Shape shape = shape_of(b, v);
    if (!moves[r]) {
      return dl().id(b, shape);
    }
    return (moves[l] ? dl() + dr() : dr()).id(b, shape);
  };

  for (const NodeId v : detail::live_ids(live)) {
    const auto node = b[v]; // A copy, since building below may reallocate.
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
      if (op == OpCode::SumN) { // The sum of the terms' tangents.
        Expression<T> sum{0};
        for (const NodeId t : arguments(b, v)) {
          sum += Expression<T>{b, at(t)};
        }
        d[v] = sum.id(b);
        break;
      }
      // The chain's tangent one step at a time, from the running value and,
      // where it moves, the term, each times its partial.
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
    case Kind::Const: // Constants hold still and were skipped above.
      std::unreachable();
    case Kind::Input: // Only a symbol or an implicit moves.
      if (node.op == OpCode::Implicit) {
        // dz = sum_s dz/dx_s dx_s.
        const auto dz = detail::Implicits::row(b, detail::get<Slot>(node));
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
      {
        const auto [cond, then, otherwise] =
            detail::operands<OpCode::Select>(node.in);
        d[v] = select(Expression<T>{b, cond}, Expression<T>{b, at(then)},
                      Expression<T>{b, at(otherwise)})
                   .id(b);
      }
      break;
    case Kind::Binary: {
      // Build both partials before either product, as in the reverse sweep.
      const Expression<T> l{b, node.in[0]};
      const Expression<T> r{b, node.in[1]};
      const Expression<T> one{1};
      const auto p = detail::contributions(
          op, one, l, r,
          {.left = moves[node.in[0]], .right = moves[node.in[1]]}, one);
      d[v] = moved(
          v, node.in[0], [&] { return p.first * dv(node.in[0]); }, node.in[1],
          [&] { return p.second * dv(node.in[1]); });
      break;
    }
    case Kind::Tensor: {
      // The forward twin of each reverse rule; a vector's tangent is a vector
      // of the same width.
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
      case OpCode::AddAt: {
        const auto [vec, scalar] = detail::operands<OpCode::AddAt>(node);
        d[v] = b.make(opcode<OpCode::AddAt>, detail::get<Element>(node),
                      {.vec = at(vec), .scalar = at(scalar)});
        break;
      }
      // Bilinear ops: the node with one operand replaced by its tangent.
      case OpCode::Scale:
      case OpCode::Dot:
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
      default: // Only the dense sweep builds these, never under a root.
        std::unreachable();
      }
      break;
    }
    }
  }

  return roots | std::views::transform([&](NodeId r) { return at(r); }) |
         impl::to<std::vector<NodeId>>();
}

// One forward sweep per symbol, kept as a check on the reverse Jacobian, which
// builds fewer nodes everywhere except at a single variable.
template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow build_symbolic_jacobian(Arena<T> &b,
                                                            NodeId root) {
  const auto nsym = b.symbols().size();
  JacobianRow g{.value = root,
                .partial = std::vector<NodeId>(nsym, no_node),
                .implicit = {}};
  const std::array<NodeId, 1> roots{root};

  // One unit vector, reused by moving the 1 through it.
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

// An m-function Jacobian from m sweeps sharing one graph, rows in function
// order. Literal-zero partials are left out of `pattern` and cost no column.
struct Jacobian {
  std::vector<NodeId> value;   // One per function.
  std::vector<NodeId> partial; // pattern.nonzeros(), row-major by function.
  Sparsity pattern;
  NodeId zero = no_node; // What a cell outside the pattern reads.

  [[nodiscard]] constexpr NodeId at(std::size_t i, std::size_t j) const {
    return pattern.at(i, j)
        .transform([this](std::size_t k) { return partial[k]; })
        .value_or(zero);
  }
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
  // Earlier rows' nodes are not in this root's cone but still share
  // subexpressions through interning.
  for (const NodeId r : roots) {
    const auto g = build_reverse_jacobian(b, r);
    // The same zero test the sweep uses, so the two cannot drift, -0.0
    // included.
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

// The single-root overload; a span is not constructible from a NodeId, so it
// never competes with the m-root one.
template <impl::DiffMode Mode = impl::DiffMode::Reverse, impl::Numeric T>
[[nodiscard]] constexpr JacobianRow derive_jacobian(Arena<T> &b, NodeId root) {
  if constexpr (Mode == impl::DiffMode::Symbolic) {
    return build_symbolic_jacobian(b, root);
  } else {
    return build_reverse_jacobian(b, root);
  }
}

namespace detail {
// The arena f is differentiated in, or the reason it cannot be.
template <impl::Numeric T>
[[nodiscard]] result<Arena<T> *> differentiable(const Expression<T> &f) {
  Arena<T> *const b = arena_of<T>({&f, 1});
  const auto why = b == nullptr        ? std::optional{errc::outside_equation}
                   : Router::scalar(f) ? Router::why(f)
                                       : errc::tensor_shape;
  return why ? fail(*why) : result<Arena<T> *>{b};
}
} // namespace detail

// df/dx for every symbol in symbols() order, usable in an arena still being
// built. Each is poisoned when f cannot be differentiated.
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
// df/dx for one symbol by name, or unknown_symbol for anything but a var.
template <impl::Numeric T>
[[nodiscard]] Expression<T> derivative(const Expression<T> &f,
                                       std::string_view symbol) {
  const auto in = detail::differentiable(f).and_then(
      [&](const Arena<T> *b) { return b->symbol(symbol); });
  return in ? gradient(f)[*in] : detail::Router::poison<T>(in.error().code);
}

// One sweep per colour: from the sum of a colour's partials, row i gets the sum
// of d2f/dxi dxj over the colour, at most one of which is nonzero.
struct Hessian {
  NodeId value = no_node;
  std::vector<NodeId> partial;    // n nodes.
  std::vector<NodeId> compressed; // colours * n nodes.
  Colouring colouring;
  NodeId zero = no_node;

  // Reads H(i, j) through the colouring; a cell no column owns is zero.
  [[nodiscard]] constexpr NodeId at(std::size_t i, std::size_t j) const {
    return colouring.cell_of(i, j)
        .transform([this](std::size_t k) { return compressed[k]; })
        .value_or(zero);
  }
  [[nodiscard]] constexpr std::size_t colours() const {
    return colouring.count();
  }
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

  // Only the cells a column owns are kept, since each other cell would cost an
  // output column and a whole cone of nodes.
  h.compressed.assign(h.colouring.width(), h.zero);
  for (const std::size_t c : h.colouring.colours()) {
    // Summing a colour's partials first lets one sweep do |colour| sweeps'
    // work.
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
// Sum_j seed_j * ids[j], the root every seeded product sweeps from. Literal
// zero partials drop out through the algebra rules before any node exists.
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

// H(x)v without forming H: sweeping from s = Sum_j v_j df/dx_j gives
// ds/dx_i = (Hv)_i in one sweep.
// Ref: Pearlmutter, "Fast Exact Multiplication by the Hessian", 1994.
struct HessianVector {
  NodeId value = no_node;
  std::vector<NodeId> partial; // The gradient, which comes free.
  std::vector<NodeId> product; // H v.
};

template <impl::Numeric T>
[[nodiscard]] constexpr HessianVector derive_hvp(Arena<T> &b, NodeId root) {
  const auto g = build_reverse_jacobian(b, root);
  const auto along = detail::seeded_sum(b, g.partial);
  return {.value = g.value,
          .partial = g.partial,
          .product = build_reverse_jacobian(b, along.id(b)).partial};
}

// w'J by the same re-rooting: sweeping from s = Sum_i w_i f_i gives
// ds/dx_j = (w'J)_j. Here the seed slots count functions, not symbols.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
derive_vjp(Arena<T> &b, std::span<const NodeId> roots) {
  const auto along = detail::seeded_sum(b, roots);
  return build_reverse_jacobian(b, along.id(b)).partial;
}

namespace detail {
// One seed per symbol, the direction a forward sweep carries.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId> seeds_along(Arena<T> &b) {
  return std::views::iota(0u, static_cast<std::uint32_t>(b.symbols().size())) |
         std::views::transform(
             [&b](std::uint32_t j) { return seed(b, j).id(b); }) |
         impl::to<std::vector<NodeId>>();
}
} // namespace detail

// J v by one forward sweep, a node per root. Rooted at the Jacobian's cells
// it gives dJ/dv in the Jacobian's layout.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
derive_jvp(Arena<T> &b, std::span<const NodeId> roots) {
  return tangent_sweep(b, roots, detail::seeds_along(b));
}

} // namespace ddx::rt
