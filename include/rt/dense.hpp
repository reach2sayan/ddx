#pragma once

#include "rt/derivative.hpp"
#include "rt/tensor.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>
#include <boost/describe/class.hpp>

#include <algorithm>
#include <cstdint>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

// The dense Hessian of one function in O(n^2), as one matrix-valued node.
//
// hessian() colours the symbols and sweeps once per colour; a dense term --
// Dot(x, Gemv(C, x)) -- couples every symbol to every other, so that is n + 1
// reverse sweeps and n GEMVs per point: n^3.  The hand formula is cheap
// because d/dx [Gemv(C, x)] is C, a copy, and d/dx [g(s) grad s] is an outer
// product: the Hessian of such a model is a short sum of shaped terms.
//
// So this is a forward sweep over the gradient's cone carrying matrix-valued
// tangents symbolically.  D[x] for the symbol vector is the identity; every
// vector node's tangent is a sum of terms, each Identity, Diag(d), Outer(u, g)
// or Dense(M), all w x n; a scalar node's tangent is its gradient row, a vector
// of width n.  Each rule is the matrix calculus of a linear map over a sum, and
// only Gemv over a Dense term needs a product of two dense matrices -- the one
// n^3 case, counted so a caller can see when a model's structure defeated the
// algebra.  The rows the partials end with are stacked into the block.
namespace ddx::rt {

struct DenseHessian {
  NodeId value = no_node;
  std::vector<NodeId> partial; // n -- the gradient, which comes free
  // The n x n block, one node: column-major, H(i, j) = d partial_i / d x_j at
  // i + j n.
  NodeId block = no_node;
  std::uint32_t n2_terms = 0; // matrix terms the algebra kept quadratic
  std::uint32_t n3_terms = 0; // MatMul fallbacks: right, and not O(n^2)
  BOOST_DESCRIBE_CLASS(DenseHessian, (),
                       (value, partial, block, n2_terms, n3_terms), (), ())
};

namespace detail {

// One term of a vector node's tangent, w x n over the n symbols.
struct Term {
  enum class Kind : std::uint8_t { Identity, Diag, Outer, Dense };
  Kind kind = Kind::Identity;
  NodeId a = no_node; // Diag: d (n); Outer: u (w); Dense: the w x n row
  NodeId b = no_node; // Outer: g (n)
};
using Terms = boost::container::small_vector<Term, 4>;

template <impl::Numeric T> class DenseSweep {
public:
  DenseSweep(Builder<T> &b, std::span<const NodeId> partials)
      : b_{b}, n_{static_cast<std::uint32_t>(b.symbols().size())},
        row_(b.size(), no_node), mat_(b.size()) {
    const auto live = reachable(b_, partials);
    for (const NodeId v : live_ids(live)) {
      visit(v);
    }
  }

  // The row each partial ends with, stacked: a partial nothing moves is a
  // zero row.
  [[nodiscard]] NodeId block(std::span<const NodeId> partials) {
    return stacked(partials);
  }
  [[nodiscard]] std::uint32_t n2() const noexcept { return n2_; }
  [[nodiscard]] std::uint32_t n3() const noexcept { return n3_; }

private:
  using V = Expression<T>; // a handle known to be a vector
  using E = Expression<T>;

  [[nodiscard]] V vec(NodeId id) const { return V{b_, id}; }
  [[nodiscard]] E expr(NodeId id) const { return E{b_, id}; }
  [[nodiscard]] NodeId zeros() { return b_.zeros(n_); }
  [[nodiscard]] NodeId ones() { return b_.ones(n_); }
  // e_i, interned once per i and width.
  [[nodiscard]] NodeId unit(std::uint32_t i) { return unit(i, n_); }
  [[nodiscard]] NodeId unit(std::uint32_t i, std::uint32_t width) {
    return b_.make(OpCode::AddAt, b_.zeros(width), b_.constant(T{1}), no_node,
                   i);
  }
  // The rows of `ids` as a matrix, a zero row for an id nothing moves.
  [[nodiscard]] NodeId stacked(const auto &ids) {
    const auto rows = ids | std::views::transform([this](NodeId g) {
                        return row_[g] == no_node ? zeros() : row_[g];
                      }) |
                      impl::to<std::vector<NodeId>>();
    return b_.pack(rows, OpCode::StackRows);
  }

  // Row arithmetic, with no_node the zero row.
  [[nodiscard]] NodeId add(NodeId l, NodeId r) {
    if (l == no_node) {
      return r;
    }
    if (r == no_node) {
      return l;
    }
    return (vec(l) + vec(r)).id(b_, n_);
  }
  [[nodiscard]] NodeId scaled(const E &s, NodeId r) {
    if (r == no_node || s.poisoned()) {
      return no_node;
    }
    const NodeId sid = s.id(b_);
    if (b_.is_constant(sid, T{0})) {
      return no_node;
    }
    return (expr(sid) * vec(r)).id(b_, n_);
  }
  [[nodiscard]] NodeId scaled(NodeId s, NodeId r) { return scaled(expr(s), r); }

  // Elementwise d * term, the step every Map and Zip takes: diag(d) applied
  // on the left of a w x n term.
  [[nodiscard]] Term diag_scale(const V &d, const Term &t) {
    switch (t.kind) {
    case Term::Kind::Identity:
      ++n2_;
      return {.kind = Term::Kind::Diag, .a = d.id(b_, n_)};
    case Term::Kind::Diag:
      return {.kind = Term::Kind::Diag, .a = (d * vec(t.a)).id(b_, n_)};
    case Term::Kind::Outer:
      return {.kind = Term::Kind::Outer,
              .a = (d * vec(t.a)).id(b_, d.size()),
              .b = t.b};
    case Term::Kind::Dense:
      return {.kind = Term::Kind::Dense,
              .a = b_.make(OpCode::RowScale, d.id(b_, n_), t.a)};
    }
    std::unreachable();
  }
  [[nodiscard]] Terms diag_scale(const V &d, const Terms &ts) {
    return ts | std::views::transform([&](const Term &t) {
             return diag_scale(d, t);
           }) |
           impl::to<Terms>();
  }
  // s * term for a scalar s.
  [[nodiscard]] Term scale(const E &s, const Term &t) {
    switch (t.kind) {
    case Term::Kind::Identity:
      ++n2_;
      return {.kind = Term::Kind::Diag, .a = (s * vec(ones())).id(b_, n_)};
    case Term::Kind::Diag:
      return {.kind = Term::Kind::Diag, .a = (s * vec(t.a)).id(b_, n_)};
    case Term::Kind::Outer:
      return {.kind = Term::Kind::Outer,
              .a = (s * vec(t.a)).id(b_, width_of(b_, t.a)),
              .b = t.b};
    case Term::Kind::Dense:
      return {.kind = Term::Kind::Dense,
              .a = (s * vec(t.a)).id(b_, width_of(b_, t.a))};
    }
    std::unreachable();
  }
  // Row i of a term: a vector of width n.
  [[nodiscard]] NodeId row_of(const Term &t, std::uint32_t i) {
    switch (t.kind) {
    case Term::Kind::Identity:
      return unit(i);
    case Term::Kind::Diag:
      return b_.make(OpCode::AddAt, zeros(), vec(t.a)[i].id(b_), no_node, i);
    case Term::Kind::Outer:
      return scaled(vec(t.a)[i], t.b);
    case Term::Kind::Dense:
      return b_.make(OpCode::RowOf, t.a, no_node, no_node, i);
    }
    std::unreachable();
  }
  // u^T applied to a term: a row.
  [[nodiscard]] NodeId left(const V &u, const Term &t) {
    switch (t.kind) {
    case Term::Kind::Identity:
      return u.id(b_, n_);
    case Term::Kind::Diag:
      return (u * vec(t.a)).id(b_, n_);
    case Term::Kind::Outer:
      return scaled(dot(u, vec(t.a)), t.b);
    case Term::Kind::Dense:
      return b_.make(OpCode::MatVec, t.a, u.id(b_, width_of(b_, t.a) / n_),
                     no_node, 1);
    }
    std::unreachable();
  }
  [[nodiscard]] NodeId left(const V &u, const Terms &ts) {
    return std::ranges::fold_left(ts, no_node, [&](NodeId r, const Term &t) {
      return add(r, left(u, t));
    });
  }
  // The constant matrix C (or C^T) applied on the left of a term.
  [[nodiscard]] Term gemv(NodeId matrix, bool transposed, const Term &t) {
    const NodeId c =
        b_.make(OpCode::MatRow, matrix, no_node, no_node, transposed);
    switch (t.kind) {
    case Term::Kind::Identity:
      ++n2_;
      return {.kind = Term::Kind::Dense, .a = c};
    case Term::Kind::Diag:
      ++n2_;
      return {.kind = Term::Kind::Dense,
              .a = b_.make(OpCode::ColScale, c, t.a)};
    case Term::Kind::Outer:
      return {.kind = Term::Kind::Outer,
              .a = b_.make(OpCode::Gemv, matrix, t.a, no_node, transposed),
              .b = t.b};
    case Term::Kind::Dense:
      ++n3_;
      return {.kind = Term::Kind::Dense, .a = b_.make(OpCode::MatMul, c, t.a)};
    }
    std::unreachable();
  }

  // Whether a Pack is the symbol vector itself, in order: the identity.
  [[nodiscard]] bool is_symbols(NodeId v) const {
    const auto args = arguments(b_, v);
    return args.size() == n_ &&
           std::ranges::all_of(std::views::iota(0u, n_), [&](std::uint32_t j) {
             return args[j] == b_.leaf(OpCode::Var, j);
           });
  }

  void visit(NodeId v) {
    const Node<T> node = b_[v]; // by value: building below may reallocate
    if (tensor_valued(node.op)) {
      mat_[v] = matrix_tangent(v, node);
    } else {
      row_[v] = row_tangent(v, node);
    }
  }

  [[nodiscard]] NodeId row_tangent(NodeId v, const Node<T> &node) {
    switch (kind_of(node.op)) {
    case Kind::Const:
      return no_node;
    case Kind::Input:
      if (node.op == OpCode::Var) {
        return unit(index_of(node));
      }
      if (node.op == OpCode::Implicit) {
        return b_.pack(Implicits::row(b_, index_of(node)));
      }
      return no_node;
    case Kind::Nary:
      return std::ranges::fold_left(
          arguments(b_, v), no_node,
          [this](NodeId r, NodeId t) { return add(r, row_[t]); });
    case Kind::Extern: {
      NodeId r = no_node;
      extern_partials(b_, v, [&](NodeId arg, const E &c) {
        r = add(r, scaled(c, row_[arg]));
      });
      return r;
    }
    case Kind::Unary:
      return scaled(partial(node.op, expr(node.a), expr(v)), row_[node.a]);
    case Kind::Binary: {
      const auto [dl, dr] = partials(node.op, expr(node.a), expr(node.b));
      return add(scaled(dl, row_[node.a]), scaled(dr, row_[node.b]));
    }
    case Kind::Ternary: {
      const E c = expr(node.a);
      return add(scaled(select(c, E{1}, E{0}), row_[node.b]),
                 scaled(select(c, E{0}, E{1}), row_[node.c]));
    }
    case Kind::Tensor:
      switch (node.op) {
      case OpCode::MatConst: // a table, which a Gemv reads: nothing moves it
        return no_node;
      case OpCode::Gather:
        return std::ranges::fold_left(
            mat_[node.a], no_node, [&](NodeId r, const Term &t) {
              return add(r, row_of(t, index_of(node)));
            });
      case OpCode::Sum:
        return left(vec(ones_of(node.a)), mat_[node.a]);
      case OpCode::Dot:
        return add(left(vec(node.a), mat_[node.b]),
                   left(vec(node.b), mat_[node.a]));
      default:
        std::unreachable();
      }
    }
    std::unreachable();
  }
  [[nodiscard]] NodeId ones_of(NodeId v) { return b_.ones(width_of(b_, v)); }

  [[nodiscard]] Terms matrix_tangent(NodeId v, const Node<T> &node) {
    Terms out;
    switch (node.op) {
    case OpCode::VecConst:
      break;
    case OpCode::Pack:
      if (is_symbols(v)) {
        out.push_back({.kind = Term::Kind::Identity});
      } else {
        ++n2_;
        out.push_back(
            {.kind = Term::Kind::Dense, .a = stacked(arguments(b_, v))});
      }
      break;
    case OpCode::Map: {
      const V p = partial(inner_of(node), vec(node.a), vec(v));
      out = diag_scale(p, mat_[node.a]);
      break;
    }
    case OpCode::Zip: {
      const auto [pu, pv] = partials(inner_of(node), vec(node.a), vec(node.b));
      out = diag_scale(pu, mat_[node.a]);
      impl::append(out, diag_scale(pv, mat_[node.b]));
      break;
    }
    case OpCode::Scale: {
      const E s = expr(node.a);
      out = mat_[node.b] |
            std::views::transform([&](const Term &t) { return scale(s, t); }) |
            impl::to<Terms>();
      if (row_[node.a] != no_node) {
        ++n2_;
        out.push_back(
            {.kind = Term::Kind::Outer, .a = node.b, .b = row_[node.a]});
      }
      break;
    }
    case OpCode::Gemv:
      out = mat_[node.b] | std::views::transform([&](const Term &t) {
              return gemv(node.a, is_transposed(node), t);
            }) |
            impl::to<Terms>();
      break;
    case OpCode::AddAt:
      out = mat_[node.a];
      if (row_[node.b] != no_node) {
        ++n2_;
        out.push_back({.kind = Term::Kind::Outer,
                       .a = unit(index_of(node), width_of(b_, node.a)),
                       .b = row_[node.b]});
      }
      break;
    default:
      std::unreachable(); // a matrix row: formed here, never read here
    }
    return out;
  }
  Builder<T> &b_;
  std::uint32_t n_;
  std::vector<NodeId> row_;
  std::vector<Terms> mat_;
  std::uint32_t n2_ = 0;
  std::uint32_t n3_ = 0;
};

} // namespace detail

template <impl::Numeric T>
[[nodiscard]] DenseHessian derive_dense_hessian(Builder<T> &b, NodeId root) {
  const auto g = build_reverse_jacobian(b, root);
  detail::DenseSweep<T> sweep{b, g.partial};
  return {.value = g.value,
          .partial = g.partial,
          .block = sweep.block(g.partial),
          .n2_terms = sweep.n2(),
          .n3_terms = sweep.n3()};
}

} // namespace ddx::rt
