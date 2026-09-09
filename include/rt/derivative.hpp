#pragma once

#include "ops/mode.hpp" // DiffMode
#include "rt/coupling.hpp"
#include "rt/expressions.hpp"
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

// The sweeps' side of Builder::seal(): a struct, not a friended function
// template per sweep, so builder.hpp forward-declares an empty type.
struct Sealing {
  template <impl::Numeric T>
  static constexpr void seal(Builder<T> &b) noexcept {
    b.seal();
  }
};

// A template, so `if constexpr` actually discards: deriv_from_value reuses the
// primal node, and d(exp)/du becomes the exp node itself.
template <typename Fn, impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> rule(const RTExpression<T> &u,
                                             const RTExpression<T> &fu) {
  if constexpr (impl::detail::has_deriv_from_value_v<Fn, RTExpression<T>>) {
    return Fn::deriv_from_value(u, fu);
  } else {
    return Fn::deriv(u);
  }
}

// Nothing here passes `f` to a binary rule: the descriptors recompute it, which
// is free on a graph since Builder::make interns it back onto the primal.
template <impl::Numeric S>
[[nodiscard]] constexpr RTExpression<S>
contribution(OpCode op, const RTExpression<S> &adj, const RTExpression<S> &u,
             const RTExpression<S> &f) {
  using T = RTExpression<S>;
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
    return adj * rule<impl::detail::Op##Fn<T>>(u, f);
    DDX_UNARY_MATH_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
  default:
    return T{0};
  }
}

// The pair reaching a two-argument op's operands.
template <impl::Numeric S>
[[nodiscard]] constexpr std::pair<RTExpression<S>, RTExpression<S>>
contributions(OpCode op, const RTExpression<S> &adj, const RTExpression<S> &l,
              const RTExpression<S> &r) {
  using T = RTExpression<S>;
  switch (op) {
#define DDX_RT_CONTRIB(fn, Op, label, functor, Desc)                           \
  case OpCode::Op: {                                                           \
    const auto c = impl::detail::adjoints_of<impl::detail::Desc>(adj, l, r);   \
    return {c[0], c[1]};                                                       \
  }
    DDX_RT_BINARY_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
  default:
    return {T{0}, T{0}};
  }
}

// The same rule seeded with a unit adjoint: Builder::make fires the x*1 rule
// before a node exists, so this is the partial and not a multiply by one.
template <impl::Numeric S>
[[nodiscard]] constexpr RTExpression<S>
partial(OpCode op, const RTExpression<S> &u, const RTExpression<S> &f) {
  return contribution(op, RTExpression<S>{1}, u, f);
}

template <impl::Numeric S>
[[nodiscard]] constexpr std::pair<RTExpression<S>, RTExpression<S>>
partials(OpCode op, const RTExpression<S> &l, const RTExpression<S> &r) {
  return contributions(op, RTExpression<S>{1}, l, r);
}

} // namespace detail

// One root's row of the Jacobian.  build_jacobian_impl() is an overload set:
// the number of roots picks the shape, as output_dim does.
struct JacobianRow {
  NodeId value = no_node;      // the node for f itself
  std::vector<NodeId> partial; // one per symbol, in Builder::symbols() order
};

// reverse_sweep, accumulating *nodes*.  It appends to the builder it reads: new
// nodes land above the snapshot, so reverse id order stays topological.
template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow build_reverse_jacobian(Builder<T> &b,
                                                           NodeId root) {
  detail::Sealing::seal(b);
  const auto n = static_cast<NodeId>(b.size());
  std::vector<NodeId> adj(n, no_node);
  adj[root] = b.constant(T{1});

  const auto add_to = [&](NodeId child, const RTExpression<T> &contribution) {
    const NodeId c = contribution.id(b);
    // What sign(), and so abs(), hands back.  Storing it would mark the child
    // live and carry a zero adjoint down its whole cone.
    if (b.is_constant(c, T{0})) {
      return;
    }
    adj[child] = adj[child] == no_node
                     ? c
                     : (RTExpression{b, adj[child]} + RTExpression{b, c}).id(b);
  };

  for (NodeId v = n; v-- > 0;) {
    // no_node is "nothing reached it", a folded Const 0 is "what reached it
    // cancelled"; neither contributes below.
    if (adj[v] == no_node || b.is_constant(adj[v], T{0})) {
      continue;
    }
    const auto node = b[v]; // by value: building below may reallocate
    if (is_leaf(node.op)) {
      continue;
    }
    const RTExpression<T> a{b, adj[v]};
    if (arity_of(node.op) == 1) {
      const RTExpression<T> u{b, node.a};
      const RTExpression<T> self{b, v};
      add_to(node.a, detail::contribution(node.op, a, u, self));
    } else if (arity_of(node.op) == 3) {
      // Here rather than in ops/adjoints.hpp: two arms of different types is
      // what a type-encoded tree cannot hold, so there is no compile-time rule
      // to share with.  Nothing reaches the condition -- it chooses.
      const RTExpression<T> cond{b, node.a};
      add_to(node.b, select(cond, a, RTExpression<T>{0}));
      add_to(node.c, select(cond, RTExpression<T>{0}, a));
    } else {
      const RTExpression<T> l{b, node.a};
      const RTExpression<T> r{b, node.b};
      const auto [cl, cr] = detail::contributions(node.op, a, l, r);
      add_to(node.a, cl);
      add_to(node.b, cr);
    }
  }

  JacobianRow g{.value = root,
                .partial = std::vector<NodeId>(b.symbols().size(), no_node)};
  for (const auto [v, node] : std::views::enumerate(b.nodes().first(n)) |
                                  std::views::filter([](const auto &entry) {
                                    return std::get<1>(entry).op == OpCode::Var;
                                  })) {
    g.partial[node.slot] = adj[static_cast<std::size_t>(v)];
  }
  // A symbol the expression never mentions has no leaf, so no adjoint reached
  // it; those partials are the literal zero.
  std::ranges::replace(g.partial, no_node, b.constant(T{0}));
  return g;
}

// dv/ds carried up the graph in one pass, where `tangent[j]` is ds for symbol
// j.  The roots' cone only, the arena by then holding derivative blocks whose
// tangents nobody asked for.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
tangent_sweep(Builder<T> &b, std::span<const NodeId> roots,
              std::span<const NodeId> tangent) {
  detail::Sealing::seal(b);
  const auto n = static_cast<NodeId>(b.size());
  const auto live = detail::reachable(b, roots);
  std::vector<NodeId> d(n, no_node);

  for (const NodeId v : detail::live_ids(live)) {
    const auto node = b[v]; // by value: building below may reallocate
    if (is_leaf(node.op)) {
      // A Seed is a direction, not a variable, so it carries no tangent.
      d[v] = node.op == OpCode::Var ? tangent[node.slot] : b.constant(T{0});
      continue;
    }
    if (arity_of(node.op) == 1) {
      const RTExpression<T> u{b, node.a};
      const RTExpression<T> self{b, v};
      d[v] = (detail::partial(node.op, u, self) * RTExpression<T>{b, d[node.a]})
                 .id(b);
    } else if (arity_of(node.op) == 3) {
      // select(c, dt, df): the condition picks the derivative as it picks
      // the value.
      d[v] = select(RTExpression<T>{b, node.a}, RTExpression<T>{b, d[node.b]},
                    RTExpression<T>{b, d[node.c]})
                 .id(b);
    } else {
      const RTExpression<T> l{b, node.a};
      const RTExpression<T> r{b, node.b};
      const auto [dl, dr] = detail::partials(node.op, l, r);
      d[v] = (dl * RTExpression<T>{b, d[node.a]} +
              dr * RTExpression<T>{b, d[node.b]})
                 .id(b);
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
  detail::Sealing::seal(b);
  const auto nsym = b.symbols().size();
  JacobianRow g{.value = root, .partial = std::vector<NodeId>(nsym, no_node)};
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
build_jacobian_impl(Builder<T> &b, std::span<const NodeId> roots) {
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
// this and the m-root overload never compete, and DiffMode does not satisfy
// Numeric, which keeps it clear of the plain spelling below.
template <impl::DiffMode Mode, impl::Numeric T>
[[nodiscard]] constexpr JacobianRow build_jacobian_impl(Builder<T> &b,
                                                        NodeId root) {
  if constexpr (Mode == impl::DiffMode::Symbolic) {
    return build_symbolic_jacobian(b, root);
  } else {
    return build_reverse_jacobian(b, root);
  }
}

template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow build_jacobian_impl(Builder<T> &b,
                                                        NodeId root) {
  return build_reverse_jacobian(b, root);
}

// One sweep per colour.  Sweeping from the sum of a colour's partials gives,
// for row i, the sum over j in the colour of d2f/dxi dxj -- of which the
// colouring guarantees at most one is not structurally zero.
struct Hessian {
  NodeId value = no_node;
  std::vector<NodeId> partial;    // n
  std::vector<NodeId> compressed; // colours * n
  Coloring coloring;
  NodeId zero = no_node;

  // Scattered on read, so a caller never sees the compressed form: a cell no
  // column owns has no storage and answers the colouring's structural zero.
  [[nodiscard]] constexpr NodeId at(std::size_t i, std::size_t j) const {
    return coloring.cell_of(i, j)
        .transform([this](std::size_t k) { return compressed[k]; })
        .value_or(zero);
  }
  [[nodiscard]] constexpr std::size_t colors() const {
    return coloring.count();
  }
  BOOST_DESCRIBE_CLASS(Hessian, (),
                       (value, partial, compressed, coloring, zero), (), ())
};

template <impl::Numeric T>
[[nodiscard]] Hessian build_hessian_impl(Builder<T> &b, NodeId root) {
  const auto g = build_reverse_jacobian(b, root);
  const auto rows = coupling_pattern(b, root);
  Hessian h{.value = g.value,
            .partial = g.partial,
            .compressed = {},
            .coloring = color_columns(rows),
            .zero = b.constant(T{0})};

  // Only the cells a column owns: the rest cost an output column each and a
  // whole cone of nodes behind it, which on an arrow Hessian is most of it.
  h.compressed.assign(h.coloring.width(), h.zero);
  for (const std::size_t c : h.coloring.colors()) {
    // Summing a colour's partials before the sweep is what makes one sweep do
    // the work of |colour| of them.
    auto make_partial_expression = [&](std::size_t j) {
      return RTExpression<T>{b, h.partial[j]};
    };
    const auto seed = std::ranges::fold_left(
        h.coloring.columns_of(c) |
            std::views::transform(std::move(make_partial_expression)),
        RTExpression<T>{0}, std::plus<>{});
    const auto row = build_reverse_jacobian(b, seed.id(b));
    for (const Cell &owned : h.coloring.owned_cells(c)) {
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
[[nodiscard]] constexpr RTExpression<T>
seeded_sum(Builder<T> &b, std::span<const NodeId> ids) {
  detail::Sealing::seal(b);
  auto term = [&b, ids](std::size_t j) {
    return seed(b, static_cast<std::uint32_t>(j)) * RTExpression<T>{b, ids[j]};
  };
  return std::ranges::fold_left(std::views::iota(0uz, ids.size()) |
                                    std::views::transform(std::move(term)),
                                RTExpression<T>{0}, std::plus<>{});
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
[[nodiscard]] constexpr HessianVector build_hvp_impl(Builder<T> &b,
                                                     NodeId root) {
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
build_vjp_impl(Builder<T> &b, std::span<const NodeId> roots) {
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

template <impl::Numeric T>
[[nodiscard]] constexpr Tangent build_jvp_impl(Builder<T> &b,
                                               std::span<const NodeId> roots) {
  std::vector<NodeId> along(b.symbols().size());
  for (const auto [j, node] : along | std::views::enumerate) {
    node = seed(b, static_cast<std::uint32_t>(j)).id(b);
  }
  return {.value = {roots.begin(), roots.end()},
          .product = tangent_sweep(b, roots, along)};
}

} // namespace ddx::rt
