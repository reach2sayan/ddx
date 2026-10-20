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
// is free on a graph since Builder::make interns it back onto the primal.
template <impl::Numeric T>
[[nodiscard]] constexpr H<T> contribution(OpCode op, const H<T> &adj,
                                          const H<T> &u, const H<T> &f) {
  switch (op) {
#define DDX_RT_CONTRIB(fn, Op, label, functor, Desc)                           \
  case OpCode::Op:                                                             \
    return impl::detail::adjoints_of<impl::detail::Desc>(adj, u)[0];
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

// The pair reaching a two-argument op's operands.
template <impl::Numeric T>
[[nodiscard]] constexpr std::pair<H<T>, H<T>>
contributions(OpCode op, const H<T> &adj, const H<T> &l, const H<T> &r) {
  switch (op) {
#define DDX_RT_CONTRIB(fn, Op, label, functor, Desc)                           \
  case OpCode::Op: {                                                           \
    const auto c = impl::detail::adjoints_of<impl::detail::Desc>(adj, l, r);   \
    return {c[0], c[1]};                                                       \
  }
    DDX_RT_BINARY_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
  default:
    return {H<T>{0}, H<T>{0}};
  }
}

// The same rule seeded with a unit adjoint: Builder::make fires the x*1 rule
// before a node exists, so this is the partial and not a multiply by one.
template <impl::Numeric T>
[[nodiscard]] constexpr H<T> partial(OpCode op, const H<T> &u, const H<T> &f) {
  return contribution(op, H<T>{1}, u, f);
}

template <impl::Numeric T>
[[nodiscard]] constexpr std::pair<H<T>, H<T>> partials(OpCode op, const H<T> &l,
                                                       const H<T> &r) {
  return contributions(op, H<T>{1}, l, r);
}

} // namespace detail

// One root's row of the Jacobian.  derive_jacobian() is an overload set:
// the number of roots picks the shape, as output_dim does.
struct JacobianRow {
  NodeId value = no_node;       // the node for f itself
  std::vector<NodeId> partial;  // one per symbol, in Builder::symbols() order
  std::vector<NodeId> implicit; // df/dz with every z held, one per implicit
};

template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow
build_reverse_jacobian(Builder<T> &b, NodeId root, bool ift = true);

namespace detail {
// The implicit-function theorem's half of a sweep: for implicit z closed by
// F(z, x) = 0, dz/dx_s = -F_s / F_z.  Built once per arena from one reverse
// sweep of F that holds z fixed, the first time anything differentiates
// through z, and kept on the Builder.
struct Implicits {
  template <impl::Numeric T>
  [[nodiscard]] static constexpr const std::vector<NodeId> &
  row(Builder<T> &b, std::uint32_t slot) {
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
constexpr void extern_partials(Builder<T> &b, NodeId v, Sink &&sink) {
  const auto [fn, at] = detail::extern_of(b[v]);
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
[[nodiscard]] constexpr bool first_order_only(const Builder<T> &b,
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
           b.externs()[detail::extern_of(b[v]).function].order < 2;
  });
}

// Why the implicits cannot be differentiated, if they cannot: a misuse of
// implicit(), or a residual reaching another implicit -- which would need a
// linear solve, not a quotient.
template <impl::Numeric T>
[[nodiscard]] constexpr std::optional<errc> unclosed(const Builder<T> &b) {
  if (b.broken()) {
    return b.broken();
  }
  const auto implicits =
      std::views::iota(0u, static_cast<std::uint32_t>(b.closures().size()));
  for (const std::uint32_t slot : implicits) {
    const NodeId f = b.closures()[slot];
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

// The reverse sweep, accumulating *nodes*.  It appends to the builder it reads:
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
build_reverse_jacobian(Builder<T> &b, NodeId root, bool ift) {
  const auto n = static_cast<NodeId>(b.size());
  const auto live = detail::reachable(b, std::span{&root, 1});
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
  // handle's `+` is the sum: a Zip(add) where the child is a vector.
  const auto add_to = [&](NodeId child, NodeId g,
                          const Expression<T> &contribution) {
    g = fits(g, child) ? g : no_node;
    const NodeId c = guarded(child, g, contribution);
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
        add_to(arg, g, a * c);
      });
      break;
    case Kind::Nary: // linear: every term takes the adjoint whole
      for (const NodeId t : arguments(b, v)) {
        add_to(t, g, a);
      }
      break;
    case Kind::Unary:
      add_to(node.a, g, detail::contribution(op, a, expr(node.a), expr(v)));
      break;
    case Kind::Ternary: {
      // Here rather than in ops/adjoints.hpp: two arms of different types is
      // what a type-encoded tree cannot hold, so there is no compile-time rule
      // to share with.  Nothing reaches the condition -- it chooses.
      const Expression<T> cond = expr(node.a);
      add_to(node.b, both(g, cond).id(b), a);
      add_to(node.c, both(g, select(cond, zero, one)).id(b), a);
      break;
    }
    case Kind::Binary: {
      const auto [cl, cr] =
          detail::contributions(op, a, expr(node.a), expr(node.b));
      add_to(node.a, g, cl);
      add_to(node.b, g, cr);
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
          add_to(s, each ? expr(g)[at].id(b) : g, a[at]);
        }
        break;
      }
      case OpCode::Gather:
        add_to(node.a, g,
               detail::add_at(expr(b.zeros(shape_of(b, node.a))), a,
                              detail::index_of(node)));
        break;
      case OpCode::Scale:
        add_to(node.a, g, dot(a, expr(node.b)));
        add_to(node.b, g, expr(node.a) * a);
        break;
      case OpCode::Sum:
        add_to(node.a, g, a * expr(b.ones(shape_of(b, node.a))));
        break;
      case OpCode::Dot:
        add_to(node.a, g, a * expr(node.b));
        add_to(node.b, g, a * expr(node.a));
        break;
      case OpCode::AddAt:
        add_to(node.a, g, a);
        add_to(node.b, g, a[detail::index_of(node)]);
        break;
      case OpCode::MatRow: // a table: nothing moves it
        break;
      case OpCode::MatVec: {
        // y = M v moves v by M^T a and M by a v^T; y = M^T v, by M a and v a^T.
        const bool t = detail::is_transposed(node);
        add_to(node.b, g,
               expr(b.make(OpCode::MatVec, node.a, adj[v], no_node, !t)));
        if (!is_table(b, node.a)) { // a table: nothing moves it
          add_to(node.a, g,
                 expr(t ? b.make(OpCode::Outer, node.b, adj[v])
                        : b.make(OpCode::Outer, adj[v], node.b)));
        }
        break;
      }
      case OpCode::Outer: // u w^T: u moves by A w, w by A^T u
        add_to(node.a, g, expr(b.make(OpCode::MatVec, adj[v], node.b)));
        add_to(node.b, g,
               expr(b.make(OpCode::MatVec, adj[v], node.a, no_node, 1)));
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
               expr(adj[z]) * expr(dz));
      }
    }
  }

  JacobianRow g{.value = root,
                .partial = std::vector<NodeId>(b.symbols().size(), no_node),
                .implicit = std::vector<NodeId>(implicits, no_node)};
  for (const auto [v, node] : std::views::enumerate(b.nodes().first(n))) {
    if (node.op == OpCode::Var) {
      g.partial[detail::index_of(node)] = adj[static_cast<std::size_t>(v)];
    } else if (node.op == OpCode::Implicit) {
      g.implicit[detail::index_of(node)] = adj[static_cast<std::size_t>(v)];
    }
  }
  // A symbol the expression never mentions has no leaf, so no adjoint reached
  // it; those partials are the literal zero.
  std::ranges::replace(g.partial, no_node, b.constant(T{0}));
  std::ranges::replace(g.implicit, no_node, b.constant(T{0}));
  return g;
}

// dv/ds carried up the graph in one pass, where `tangent[j]` is ds for symbol
// j.  The roots' cone only, the arena by then holding derivative blocks whose
// tangents nobody asked for.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
tangent_sweep(Builder<T> &b, std::span<const NodeId> roots,
              std::span<const NodeId> tangent) {
  const auto n = static_cast<NodeId>(b.size());
  const auto live = detail::reachable(b, roots);
  b.reserve(n + 2 * static_cast<std::size_t>(std::ranges::count(live, true)));
  std::vector<NodeId> d(n, no_node);

  for (const NodeId v : detail::live_ids(live)) {
    const auto node = b[v]; // by value: building below may reallocate
    const OpCode op = detail::effective_op(node);
    switch (kind_of(op)) {
    case Kind::Extern: {
      Expression<T> sum{0};
      detail::extern_partials(b, v, [&](NodeId arg, const Expression<T> &c) {
        sum = sum + c * Expression<T>{b, d[arg]};
      });
      d[v] = sum.id(b);
      break;
    }
    case Kind::Nary: { // the sum of the terms' tangents, as a sum again
      Expression<T> sum{0};
      for (const NodeId t : arguments(b, v)) {
        sum += Expression<T>{b, d[t]};
      }
      d[v] = sum.id(b);
      break;
    }
    case Kind::Const:
      d[v] = b.constant(T{0});
      break;
    case Kind::Input:
      if (node.op == OpCode::Implicit) {
        // dz = sum_s dz/dx_s dx_s: z moves with every symbol its residual does.
        const auto dz = detail::Implicits::row(b, detail::index_of(node));
        d[v] = std::ranges::fold_left(
                   std::views::zip(dz, tangent) |
                       std::views::transform([&b](const auto &pair) {
                         const auto [zs, ts] = pair;
                         return Expression<T>{b, zs} * Expression<T>{b, ts};
                       }),
                   Expression<T>{0}, std::plus<>{})
                   .id(b);
      } else {
        // A Seed is a direction, not a variable, so it carries no tangent.
        d[v] = node.op == OpCode::Var ? tangent[detail::index_of(node)]
                                      : b.constant(T{0});
      }
      break;
    case Kind::Unary: {
      const Expression<T> u{b, node.a};
      const Expression<T> self{b, v};
      d[v] = (detail::partial(op, u, self) * Expression<T>{b, d[node.a]}).id(b);
      break;
    }
    case Kind::Ternary:
      // select(c, dt, df): the condition picks the derivative as it picks
      // the value.
      d[v] = select(Expression<T>{b, node.a}, Expression<T>{b, d[node.b]},
                    Expression<T>{b, d[node.c]})
                 .id(b);
      break;
    case Kind::Binary: {
      const Expression<T> l{b, node.a};
      const Expression<T> r{b, node.b};
      const auto [dl, dr] = detail::partials(op, l, r);
      d[v] =
          (dl * Expression<T>{b, d[node.a]} + dr * Expression<T>{b, d[node.b]})
              .id(b, shape_of(b, v));
      break;
    }
    case Kind::Tensor: {
      // The forward twin of each reverse rule above; a vector's tangent is a
      // vector of its width.
      const auto expr = [&b](NodeId id) { return Expression<T>{b, id}; };
      const auto dv = [&](NodeId u) { return Expression<T>{b, d[u]}; };
      switch (node.op) {
      case OpCode::VecConst:
      case OpCode::MatRow:
        d[v] = b.zeros(shape_of(b, v));
        break;
      case OpCode::MatConst: // a table, never read as a value
        break;
      case OpCode::Pack: {
        const auto args = arguments(b, v);
        const auto tangents =
            args | std::views::transform([&d](NodeId s) { return d[s]; }) |
            impl::to<boost::container::small_vector<NodeId, 8>>();
        d[v] = b.pack(tangents);
        break;
      }
      case OpCode::Gather:
        d[v] = dv(node.a)[detail::index_of(node)].id(b);
        break;
      case OpCode::Scale:
        d[v] = (dv(node.a) * expr(node.b) + expr(node.a) * dv(node.b))
                   .id(b, shape_of(b, v));
        break;
      case OpCode::Sum:
        d[v] = sum(dv(node.a)).id(b);
        break;
      case OpCode::Dot:
        d[v] = (dot(dv(node.a), expr(node.b)) + dot(expr(node.a), dv(node.b)))
                   .id(b);
        break;
      case OpCode::AddAt:
        d[v] = b.make(OpCode::AddAt, d[node.a], d[node.b], no_node, node.aux);
        break;
      case OpCode::MatVec: {
        const Expression<T> moved{
            b, b.make(OpCode::MatVec, node.a, d[node.b], no_node, node.aux)};
        d[v] = is_table(b, node.a)
                   ? moved.id(b)
                   : (expr(b.make(OpCode::MatVec, d[node.a], node.b, no_node,
                                  node.aux)) +
                      moved)
                         .id(b);
        break;
      }
      case OpCode::Outer:
        d[v] = (expr(b.make(OpCode::Outer, d[node.a], node.b)) +
                expr(b.make(OpCode::Outer, node.a, d[node.b])))
                   .id(b);
        break;
      default: // the dense sweep's own rows, built under no root
        std::unreachable();
      }
      break;
    }
    }
  }

  return roots | std::views::transform([&d](NodeId r) { return d[r]; }) |
         impl::to<std::vector<NodeId>>();
}

// One sweep per symbol, with a unit tangent in each.  Here to check Reverse
// against, which produces fewer nodes on everything but a single variable.
template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow build_symbolic_jacobian(Builder<T> &b,
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
derive_jacobian(Builder<T> &b, std::span<const NodeId> roots) {
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
[[nodiscard]] constexpr JacobianRow derive_jacobian(Builder<T> &b,
                                                    NodeId root) {
  if constexpr (Mode == impl::DiffMode::Symbolic) {
    return build_symbolic_jacobian(b, root);
  } else {
    return build_reverse_jacobian(b, root);
  }
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
[[nodiscard]] Hessian derive_hessian(Builder<T> &b, NodeId root) {
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
[[nodiscard]] constexpr Expression<T> seeded_sum(Builder<T> &b,
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
[[nodiscard]] constexpr HessianVector derive_hvp(Builder<T> &b, NodeId root) {
  const auto g = build_reverse_jacobian(b, root);
  const auto along = detail::seeded_sum(b, g.partial);
  return {.value = g.value,
          .partial = g.partial,
          .product = build_reverse_jacobian(b, along.id(b)).partial};
}

// w'J, the same re-rooting one order down: sweeping from s = Sum_i w_i f_i
// gives ds/dx_j = (w'J)_j.  One sweep and n columns, where jacobian() is m
// sweeps and pattern.nonzeros() columns; the seed slots here are counted in
// functions, not symbols.
struct VectorJacobian {
  std::vector<NodeId> value;   // m
  std::vector<NodeId> product; // n -- w'J
  BOOST_DESCRIBE_CLASS(VectorJacobian, (), (value, product), (), ())
};

template <impl::Numeric T>
[[nodiscard]] constexpr VectorJacobian
derive_vjp(Builder<T> &b, std::span<const NodeId> roots) {
  const auto along = detail::seeded_sum(b, roots);
  return {.value = {roots.begin(), roots.end()},
          .product = build_reverse_jacobian(b, along.id(b)).partial};
}

// J v, the one product that is a forward sweep: one pass, m outputs, whatever
// n is.
struct Tangent {
  std::vector<NodeId> value;   // m
  std::vector<NodeId> product; // m -- J v
  BOOST_DESCRIBE_CLASS(Tangent, (), (value, product), (), ())
};

namespace detail {
// One seed per symbol: the direction a forward sweep carries.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId> seeds_along(Builder<T> &b) {
  return std::views::iota(0u, static_cast<std::uint32_t>(b.symbols().size())) |
         std::views::transform(
             [&b](std::uint32_t j) { return seed(b, j).id(b); }) |
         impl::to<std::vector<NodeId>>();
}
} // namespace detail

template <impl::Numeric T>
[[nodiscard]] constexpr Tangent derive_jvp(Builder<T> &b,
                                           std::span<const NodeId> roots) {
  return {.value = {roots.begin(), roots.end()},
          .product = tangent_sweep(b, roots, detail::seeds_along(b))};
}

// The Jacobian's derivative along v, dJ/dv: its own cells carried forward
// along the seeds in one pass over their cone.  One node per Jacobian cell, so
// the block has the Jacobian's layout and reads with its Sparsity -- for m
// functions, every d2 f_i / dx_j dv in a single forward sweep.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
derive_jacobian_tangent(Builder<T> &b, const Jacobian &j) {
  return tangent_sweep(b, j.partial, detail::seeds_along(b));
}

} // namespace ddx::rt
