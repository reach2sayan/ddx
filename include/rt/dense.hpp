#pragma once

#include "rt/derivative.hpp"
#include "rt/tensor.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>
#include <boost/describe/class.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

// The dense Hessian of one function in O(n^2), as one matrix-valued node.
//
// hessian() colours the symbols and sweeps once per colour; a dense term --
// Dot(x, C x) -- couples every symbol to every other, so that is n + 1
// reverse sweeps and n GEMVs per point: n^3.  The hand formula is cheap
// because d/dx [C x] is C, a copy, and d/dx [g(s) grad s] is an outer
// product: the Hessian of such a model is a short sum of shaped terms.
//
// So this is a forward sweep over the gradient's cone carrying matrix-valued
// tangents symbolically.  D[x] for the symbol vector is the identity; every
// vector node's tangent is a sum of terms, each Identity, Diag(d), Outer(u, g)
// or Dense(M), all w x n; a scalar node's tangent is its gradient row, a vector
// of width n.  Each rule is the matrix calculus of a linear map over a sum, and
// only a MatVec over a Dense term needs a product of two dense matrices -- the one
// n^3 case, counted so a caller can see when a model's structure defeated the
// algebra.  Where the partials are the gradient vector's elements, the block
// is that vector's terms summed; otherwise the partials' rows are stacked.
//
// A vector beside a few scalar symbols is swept alone, the scalars held: its
// block is the Hessian at fixed scalars, and each scalar's row is one reverse
// sweep of its partial, placed beside it.
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

// The symbols the sweep differentiates by: `width` of them, `at` first.
struct Swept {
  std::uint32_t at = 0;
  std::uint32_t width = 0;
};

// The widest vector of consecutive symbols the model packs, where it is most
// of them; else every symbol.
template <impl::Numeric T>
[[nodiscard]] Swept swept_of(const Builder<T> &b, NodeId root) {
  const auto all = static_cast<std::uint32_t>(b.symbols().size());
  const auto run = [&b](NodeId v) -> Swept {
    const auto args = arguments(b, v);
    if (b.op_of(args[0]) != OpCode::Var) {
      return {};
    }
    const std::uint32_t first = index_of(b[args[0]]);
    const bool consecutive =
        std::ranges::all_of(std::views::iota(0uz, args.size()), [&](auto j) {
          return b.op_of(args[j]) == OpCode::Var &&
                 index_of(b[args[j]]) == first + j;
        });
    return consecutive ? Swept{.at = first,
                               .width = static_cast<std::uint32_t>(args.size())}
                       : Swept{};
  };
  const auto live = reachable(b, std::span{&root, 1});
  const Swept widest = std::ranges::fold_left(
      live_ids(live) | std::views::filter([&b](NodeId v) {
        return b.op_of(v) == OpCode::Pack;
      }) | std::views::transform(run),
      Swept{},
      [](const Swept &l, const Swept &r) { return r.width > l.width ? r : l; });
  return 2 * widest.width > all ? widest : Swept{.at = 0, .width = all};
}

template <impl::Numeric T> class DenseSweep {
public:
  DenseSweep(Builder<T> &b, std::span<const NodeId> partials, Swept swept)
      : b_{b}, at_{swept.at}, n_{swept.width}, row_(b.size(), no_node),
        mat_(b.size()), source_{gathered(partials.subspan(at_, n_))} {
    // Partials gathered from one vector are read through its tangent, so the
    // walk stops at the vector and builds no row per partial.
    const auto live =
        reachable(b_, source_ ? std::span<const NodeId>{&*source_, 1}
                              : partials.subspan(at_, n_));
    for (const NodeId v : live_ids(live)) {
      visit(v);
    }
  }

  // No block where a matrix the symbols move was multiplied: its tangent is
  // rows x cols x n, which no term holds.  Where the partials are the
  // elements of one vector in order -- what the reverse sweep hands back
  // through the symbols' Pack -- the block is that vector's tangent, its
  // terms summed as matrices; otherwise the row each partial ends with,
  // stacked, a partial nothing moves a zero row.
  [[nodiscard]] NodeId block(std::span<const NodeId> partials) {
    if (refused_) {
      return no_node;
    }
    const NodeId block =
        source_ ? summed(mat_[*source_]) : stacked(partials.subspan(at_, n_));
    return partials.size() == n_ ? block : bordered(partials, block);
  }
  [[nodiscard]] std::uint32_t n2() const noexcept { return n2_; }
  [[nodiscard]] std::uint32_t n3() const noexcept { return n3_; }

private:
  using E = Expression<T>;

  [[nodiscard]] E expr(NodeId id) const { return E{b_, id}; }
  [[nodiscard]] NodeId zeros() { return b_.zeros({n_}); }
  [[nodiscard]] NodeId ones() { return b_.ones({n_}); }
  // e_i, interned once per i and width.
  [[nodiscard]] NodeId unit(std::uint32_t i) { return unit(i, n_); }
  [[nodiscard]] NodeId unit(std::uint32_t i, std::uint32_t width) {
    return b_.make(OpCode::AddAt, b_.zeros({width}), b_.constant(T{1}), no_node,
                   i);
  }
  // The rows of `ids` as a matrix, a zero row for an id nothing moves.
  [[nodiscard]] NodeId stacked(const auto &ids) {
    const auto rows = ids | std::views::transform([this](NodeId g) {
                        return row(g) == no_node ? zeros() : row(g);
                      }) |
                      impl::to<std::vector<NodeId>>();
    return b_.pack(rows, OpCode::StackRows);
  }

  // The vector whose element i partial i reads, if there is one.
  [[nodiscard]] std::optional<NodeId>
  gathered(std::span<const NodeId> partials) const {
    if (partials.size() != n_ || b_.op_of(partials.front()) != OpCode::Gather) {
      return std::nullopt;
    }
    const NodeId source = b_[partials.front()].a;
    const bool in_order =
        width_of(b_, source) == n_ &&
        std::ranges::all_of(std::views::iota(0u, n_), [&](std::uint32_t i) {
          const Node<T> &g = b_[partials[i]];
          return g.op == OpCode::Gather && g.a == source && index_of(g) == i;
        });
    return in_order ? std::optional{source} : std::nullopt;
  }
  // A tangent as one n x n matrix row: the diagonal terms summed as a vector
  // and the outer products as one low-rank product, then every matrix written
  // once, contiguously, where reading a term back by rows is a strided pass
  // per row.
  [[nodiscard]] NodeId summed(const Terms &ts) {
    const auto of = [&ts](auto... kinds) {
      return ts | std::views::filter([=](const Term &t) {
               return ((t.kind == kinds) || ...);
             });
    };
    const NodeId d = std::ranges::fold_left(
        of(Term::Kind::Identity, Term::Kind::Diag), no_node,
        [this](NodeId r, const Term &t) {
          return add(r, t.kind == Term::Kind::Identity ? ones() : t.a);
        });
    using Ids = boost::container::small_vector<NodeId, 4>;
    auto matrices = of(Term::Kind::Dense) | std::views::transform(&Term::a) |
                    impl::to<Ids>();
    // sum_k u_k g_k^T is U G^T: an n x K by K x n product, written once, where
    // K outer products would each be written and then added.
    const auto outers = of(Term::Kind::Outer) | impl::to<Terms>();
    if (outers.size() == 1) {
      matrices.push_back(
          b_.make(OpCode::Outer, outers.front().a, outers.front().b));
    } else if (outers.size() > 1) {
      const auto rows = [&](NodeId Term::*side) {
        return b_.pack(outers | std::views::transform(side) | impl::to<Ids>(),
                       OpCode::StackRows);
      };
      matrices.push_back(
          b_.make(OpCode::MatMul,
                  b_.make(OpCode::MatRow, rows(&Term::a), no_node, no_node, 1),
                  rows(&Term::b)));
    }
    // The block is an output row; a table's MatRow has none (interpret.hpp
    // reads it in place), so a block that would be one alone gets a diagonal
    // to add it to.
    if (d != no_node || matrices.empty() ||
        (matrices.size() == 1 &&
         b_.op_of(matrices.front()) == OpCode::MatRow)) {
      matrices.push_back(b_.make(OpCode::Diag, d == no_node ? zeros() : d));
    }
    return std::ranges::fold_left(
               matrices | std::views::drop(1), expr(matrices.front()),
               [this](const E &sum, NodeId m) { return sum + expr(m); })
        .id(b_);
  }
  // The block over every symbol from the swept one: that at (at_, at_), and
  // each held symbol's row as a row and a column, its cells on held symbols
  // once.
  [[nodiscard]] NodeId bordered(std::span<const NodeId> partials,
                                NodeId swept) {
    const auto all = static_cast<std::uint32_t>(partials.size());
    const auto held = [this](std::uint32_t s) {
      return s < at_ || s >= at_ + n_;
    };
    // m's rows at at_ of `all`, zero rows around them.
    const auto spread_rows = [&](NodeId m) {
      const NodeId zero = b_.zeros({shape_of(b_, m).cols});
      return b_.pack(std::views::iota(0u, all) |
                         std::views::transform([&](std::uint32_t s) {
                           return held(s) ? zero
                                          : b_.make(OpCode::RowOf, m, no_node,
                                                    no_node, s - at_);
                         }) |
                         impl::to<std::vector<NodeId>>(),
                     OpCode::StackRows);
    };
    const auto transposed = [this](NodeId m) {
      return b_.make(OpCode::MatRow, m, no_node, no_node, 1);
    };
    // The swept block at (at_, at_): its columns spread by way of the
    // transpose, then its rows.
    const auto placed = [&](NodeId m) {
      return spread_rows(transposed(spread_rows(transposed(m))));
    };
    Terms ts{{.kind = Term::Kind::Dense, .a = placed(swept)}};
    // A held symbol's reverse sweep is its row; its column is the same less
    // the cells on held symbols, which the rows already hold.
    for (const std::uint32_t s :
         std::views::iota(0u, all) | std::views::filter(held)) {
      auto row = build_reverse_jacobian(b_, partials[s]).partial;
      const NodeId e = unit(s, all);
      ts.push_back({.kind = Term::Kind::Outer, .a = e, .b = b_.pack(row)});
      for (const std::uint32_t t :
           std::views::iota(0u, all) | std::views::filter(held)) {
        row[t] = b_.constant(T{0});
      }
      ts.push_back({.kind = Term::Kind::Outer, .a = b_.pack(row), .b = e});
    }
    return summed(ts);
  }

  // Row arithmetic, with no_node the zero row.
  [[nodiscard]] NodeId add(NodeId l, NodeId r) {
    if (l == no_node) {
      return r;
    }
    if (r == no_node) {
      return l;
    }
    return (expr(l) + expr(r)).id(b_, {n_});
  }
  [[nodiscard]] NodeId scaled(const E &s, NodeId r) {
    if (r == no_node || s.poisoned()) {
      return no_node;
    }
    const NodeId sid = s.id(b_);
    if (b_.is_constant(sid, T{0})) {
      return no_node;
    }
    return (expr(sid) * expr(r)).id(b_, {n_});
  }

  // m * term.  A vector m (or a literal of the row's width) is diag(m) on the
  // left of a w x n term, the step every Map and Zip takes; a scalar node is a
  // multiple.  A diagonal folds into the matrix's own diag(e) where it has
  // one: a vector scaled where the matrix would be copied.
  [[nodiscard]] Term scale(const E &m, const Term &t) {
    const bool vector = m.size() != 0 || m.pending();
    switch (t.kind) {
    case Term::Kind::Identity:
      ++n2_;
      return {.kind = Term::Kind::Diag,
              .a = (vector ? m : m * expr(ones())).id(b_, {n_})};
    case Term::Kind::Diag:
      return {.kind = Term::Kind::Diag, .a = (m * expr(t.a)).id(b_, {n_})};
    case Term::Kind::Outer:
      return {.kind = Term::Kind::Outer,
              .a = (m * expr(t.a)).id(b_, shape_of(b_, t.a)),
              .b = t.b};
    case Term::Kind::Dense: {
      const Node<T> inner = b_[t.a];
      const std::uint32_t rows = shape_of(b_, t.a).rows;
      const auto rescaled = [&](NodeId e) {
        return (m * expr(e)).id(b_, {width_of(b_, e)});
      };
      NodeId a = no_node;
      if (inner.op == OpCode::RowScale) {
        a = b_.make(OpCode::RowScale, rescaled(inner.a), inner.b);
      } else if (!vector && inner.op == OpCode::ColScale) {
        a = b_.make(OpCode::ColScale, inner.a, rescaled(inner.b));
      } else { // a RowScale, so the next diagonal folds into it
        a = b_.make(OpCode::RowScale,
                    vector ? m.id(b_, {rows})
                           : (m * expr(b_.ones({rows}))).id(b_),
                    t.a);
      }
      return {.kind = Term::Kind::Dense, .a = a};
    }
    }
    std::unreachable();
  }
  [[nodiscard]] Terms scale(const E &m, const Terms &ts) {
    return ts |
           std::views::transform([&](const Term &t) { return scale(m, t); }) |
           impl::to<Terms>();
  }
  // Row i of a term: a vector of width n.
  [[nodiscard]] NodeId row_of(const Term &t, std::uint32_t i) {
    switch (t.kind) {
    case Term::Kind::Identity:
      return unit(i);
    case Term::Kind::Diag:
      return b_.make(OpCode::AddAt, zeros(), expr(t.a)[i].id(b_), no_node, i);
    case Term::Kind::Outer:
      return scaled(expr(t.a)[i], t.b);
    case Term::Kind::Dense:
      return b_.make(OpCode::RowOf, t.a, no_node, no_node, i);
    }
    std::unreachable();
  }
  // u^T applied to a term: a row.
  [[nodiscard]] NodeId left(const E &u, const Term &t) {
    switch (t.kind) {
    case Term::Kind::Identity:
      return u.id(b_, {n_});
    case Term::Kind::Diag:
      return (u * expr(t.a)).id(b_, {n_});
    case Term::Kind::Outer:
      return scaled(dot(u, expr(t.a)), t.b);
    case Term::Kind::Dense:
      return b_.make(OpCode::MatVec, t.a, u.id(b_, {shape_of(b_, t.a).rows}),
                     no_node, 1);
    }
    std::unreachable();
  }
  [[nodiscard]] NodeId left(const E &u, const Terms &ts) {
    return std::ranges::fold_left(ts, no_node, [&](NodeId r, const Term &t) {
      return add(r, left(u, t));
    });
  }
  // A matrix M (or M^T) that no symbol moves -- a table, or one computed from
  // parameters -- applied on the left of a term.
  [[nodiscard]] Term gemv(NodeId matrix, bool transposed, const Term &t) {
    const bool table = is_table(b_, matrix);
    const NodeId c =
        table || transposed
            ? b_.make(OpCode::MatRow, table ? b_[matrix].a : matrix, no_node,
                      no_node, transposed)
            : matrix;
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
              .a = b_.make(OpCode::MatVec, matrix, t.a, no_node, transposed),
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
             return args[j] == b_.leaf(OpCode::Var, at_ + j);
           });
  }

  // A swept symbol's row is e_j, made where something reads it: a symbol read
  // only through the symbol vector's Pack is the identity term, and no row.
  [[nodiscard]] NodeId row(NodeId u) {
    if (row_[u] == no_node && b_.op_of(u) == OpCode::Var) {
      const std::uint32_t j = index_of(b_[u]);
      if (j >= at_ && j < at_ + n_) {
        row_[u] = unit(j - at_);
      }
    }
    return row_[u];
  }

  void visit(NodeId v) {
    const Node<T> node = b_[v]; // by value: building below may reallocate
    if (is_tensor(node)) {
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
      if (node.op == OpCode::Implicit) {
        return b_.pack(
            std::span{Implicits::row(b_, index_of(node))}.subspan(at_, n_));
      }
      return no_node;
    case Kind::Nary:
      return std::ranges::fold_left(
          arguments(b_, v), no_node,
          [this](NodeId r, NodeId t) { return add(r, row(t)); });
    case Kind::Extern: {
      NodeId r = no_node;
      extern_partials(b_, v, [&](NodeId arg, const E &c) {
        r = add(r, scaled(c, row(arg)));
      });
      return r;
    }
    // A partial is built only to scale a row there is.
    case Kind::Unary:
      return row(node.a) == no_node
                 ? no_node
                 : scaled(partial(node.op, expr(node.a), expr(v)), row(node.a));
    case Kind::Binary: {
      // Both partials before either product: `make` orders a commutative
      // node's operands by id.
      const E l = expr(node.a);
      const E r = expr(node.b);
      const E pl = row(node.a) == no_node
                       ? E{0}
                       : contribution_to<0>(node.op, E{1}, l, r);
      const E pr = row(node.b) == no_node
                       ? E{0}
                       : contribution_to<1>(node.op, E{1}, l, r);
      return add(scaled(pl, row(node.a)), scaled(pr, row(node.b)));
    }
    case Kind::Ternary: { // picked, so the row not taken is never scaled by 0
      if (row(node.b) == no_node && row(node.c) == no_node) {
        return no_node;
      }
      const auto or_zeros = [this](NodeId r) {
        return expr(r == no_node ? zeros() : r);
      };
      return select(expr(node.a), or_zeros(row(node.b)), or_zeros(row(node.c)))
          .id(b_, {n_});
    }
    case Kind::Tensor:
      switch (node.op) {
      case OpCode::MatConst: // a table, read through its MatRow: nothing moves it
        return no_node;
      case OpCode::Gather:
        return std::ranges::fold_left(
            mat_[node.a], no_node, [&](NodeId r, const Term &t) {
              return add(r, row_of(t, index_of(node)));
            });
      case OpCode::Sum:
        return left(expr(b_.ones(shape_of(b_, node.a))), mat_[node.a]);
      case OpCode::Dot:
        return add(left(expr(node.a), mat_[node.b]),
                   left(expr(node.b), mat_[node.a]));
      default:
        std::unreachable();
      }
    }
    std::unreachable();
  }

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
    // A partial is built only to scale terms there are.
    case OpCode::Map:
      if (!mat_[node.a].empty()) {
        out =
            scale(partial(inner_of(node), expr(node.a), expr(v)), mat_[node.a]);
      }
      break;
    case OpCode::Zip: {
      const E u = expr(node.a);
      const E w = expr(node.b);
      const E pu = mat_[node.a].empty()
                       ? E{0}
                       : contribution_to<0>(inner_of(node), E{1}, u, w);
      const E pw = mat_[node.b].empty()
                       ? E{0}
                       : contribution_to<1>(inner_of(node), E{1}, u, w);
      out = scale(pu, mat_[node.a]);
      impl::append(out, scale(pw, mat_[node.b]));
      break;
    }
    case OpCode::Scale: {
      const E s = expr(node.a);
      out = scale(s, mat_[node.b]);
      if (row(node.a) != no_node) {
        ++n2_;
        out.push_back(
            {.kind = Term::Kind::Outer, .a = node.b, .b = row(node.a)});
      }
      break;
    }
    case OpCode::MatVec:
      refused_ = refused_ || !mat_[node.a].empty();
      out = mat_[node.b] | std::views::transform([&](const Term &t) {
              return gemv(node.a, is_transposed(node), t);
            }) |
            impl::to<Terms>();
      break;
    case OpCode::MatRow: // a table
      break;
    case OpCode::Outer:
      refused_ = refused_ || !mat_[node.a].empty() || !mat_[node.b].empty();
      break;
    case OpCode::Select: {
      if (mat_[node.b].empty() && mat_[node.c].empty()) {
        break;
      }
      // A vector condition spread along each row of the w x n terms.
      const std::uint32_t w = width_of(b_, v);
      const auto block = [&](const Terms &ts) {
        return expr(ts.empty() ? b_.zeros({w, n_}) : summed(ts));
      };
      const E c = expr(node.a);
      const E cond = c.size() ? expr(b_.make(OpCode::Outer, node.a, ones())) : c;
      ++n2_;
      out.push_back({.kind = Term::Kind::Dense,
                     .a = select(cond, block(mat_[node.b]), block(mat_[node.c]))
                              .id(b_, {w, n_})});
      break;
    }
    case OpCode::AddAt:
      out = mat_[node.a];
      if (row(node.b) != no_node) {
        ++n2_;
        out.push_back({.kind = Term::Kind::Outer,
                       .a = unit(index_of(node), width_of(b_, node.a)),
                       .b = row(node.b)});
      }
      break;
    default:
      std::unreachable(); // a matrix row: formed here, never read here
    }
    return out;
  }
  Builder<T> &b_;
  std::uint32_t at_; // the first swept symbol
  std::uint32_t n_;
  std::vector<NodeId> row_;
  std::vector<Terms> mat_;
  std::uint32_t n2_ = 0;
  std::uint32_t n3_ = 0;
  bool refused_ = false;
  std::optional<NodeId> source_; // the vector the partials are gathered from
};

} // namespace detail

template <impl::Numeric T>
[[nodiscard]] DenseHessian derive_dense_hessian(Builder<T> &b, NodeId root) {
  const auto g = build_reverse_jacobian(b, root);
  detail::DenseSweep<T> sweep{b, g.partial, detail::swept_of(b, root)};
  return {.value = g.value,
          .partial = g.partial,
          .block = sweep.block(g.partial),
          .n2_terms = sweep.n2(),
          .n3_terms = sweep.n3()};
}

} // namespace ddx::rt
