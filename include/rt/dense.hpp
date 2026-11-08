#pragma once

#include "rt/derivative.hpp"
#include "rt/tensor.hpp"
#include "util/meta.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>
#include <boost/hof/match.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <variant>
#include <vector>

// The dense Hessian as one O(n^2) matrix node, from a forward sweep carrying
// tangents as sums of Identity, Diag, Outer and Dense terms.
namespace ddx::rt {

struct DenseHessian {
  NodeId value = no_node;
  std::vector<NodeId> partial; // the n gradient entries
  // The n x n Hessian as one column-major node: H(i, j) is at i + j * n.
  NodeId block = no_node;
  std::uint32_t n2_terms = 0; // matrix terms that stayed O(n^2)
  std::uint32_t n3_terms = 0; // O(n^3) MatMul fallbacks
};

namespace detail {

// One w x n term of a vector node's tangent with respect to the n symbols.
namespace term {
struct Identity {};
struct Diag {
  NodeId d; // n long: diag(d)
};
struct Outer {
  NodeId u; // w long
  NodeId g; // n long: u g^T
};
struct Dense {
  NodeId block; // the w x n row
};
} // namespace term
using Term = std::variant<term::Identity, term::Diag, term::Outer, term::Dense>;
static_assert(sizeof(Term) <= 12);
using Terms = boost::container::small_vector<Term, 4>;

// The part that two like terms add: the left factor of an outer product or a
// dense block's matrix.
[[nodiscard]] constexpr NodeId &primary(Term &t) {
  return std::holds_alternative<term::Outer>(t)
             ? std::get<term::Outer>(t).u
             : std::get<term::Dense>(t).block;
}
[[nodiscard]] constexpr bool diagonal(const Term &t) {
  return std::holds_alternative<term::Identity>(t) ||
         std::holds_alternative<term::Diag>(t);
}

// The contiguous symbols the sweep differentiates by, starting at `at`.
struct Swept {
  std::uint32_t at = 0;
  std::uint32_t width = 0;
};

// The widest packed vector of consecutive symbols when it covers most of
// them, else every symbol. Held scalar symbols get one reverse sweep each.
template <impl::Numeric T>
[[nodiscard]] Swept swept_of(const Arena<T> &b, NodeId root) {
  const auto all = static_cast<std::uint32_t>(b.symbols().size());
  const auto run = [&b](NodeId v) -> Swept {
    const auto args = arguments(b, v);
    if (b.op_of(args[0]) != OpCode::Var) {
      return {};
    }
    const std::uint32_t first = get<Slot>(b[args[0]]).value;
    const bool consecutive =
        std::ranges::all_of(std::views::iota(0uz, args.size()), [&](auto j) {
          return b.op_of(args[j]) == OpCode::Var &&
                 get<Slot>(b[args[j]]).value == first + j;
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
  DenseSweep(Arena<T> &b, std::span<const NodeId> partials, Swept swept)
      : b_{b}, at_{swept.at}, n_{swept.width}, row_(b.size(), no_node),
        mat_(b.size()), source_{gathered(partials.subspan(at_, n_))} {
    // Partials gathered from one vector are read through that vector's
    // tangent, so the walk stops there.
    const auto live =
        reachable(b_, source_ ? std::span<const NodeId>{&*source_, 1}
                              : partials.subspan(at_, n_));
    for (const NodeId v : live_ids(live)) {
      visit(v);
    }
  }

  // The block is the gradient vector's summed tangent when the partials are its
  // elements in order, otherwise the partials' rows stacked.
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
  // The unit vector e_i, interned once per index and width.
  [[nodiscard]] NodeId unit(std::uint32_t i) { return unit(i, n_); }
  [[nodiscard]] NodeId unit(std::uint32_t i, std::uint32_t width) {
    return b_.make(opcode<OpCode::AddAt>, Element{i},
                   {.vec = b_.zeros({width}), .scalar = b_.constant(T{1})});
  }
  // The rows of ids stacked as a matrix, with zero rows for constant ids.
  [[nodiscard]] NodeId stacked(const auto &ids) {
    const auto rows = ids | std::views::transform([this](NodeId g) {
                        return row(g) == no_node ? zeros() : row(g);
                      }) |
                      impl::to<std::vector<NodeId>>();
    return b_.pack(rows, OpCode::StackRows);
  }

  // The vector whose element i is partial i, if one exists.
  [[nodiscard]] std::optional<NodeId>
  gathered(std::span<const NodeId> partials) const {
    if (partials.size() != n_ || b_.op_of(partials.front()) != OpCode::Gather) {
      return std::nullopt;
    }
    const NodeId source = b_[partials.front()].in[0];
    const bool in_order =
        width_of(b_, source) == n_ &&
        std::ranges::all_of(std::views::iota(0u, n_), [&](std::uint32_t i) {
          const Node<T> &g = b_[partials[i]];
          return g.op == OpCode::Gather && g.in[0] == source &&
                 get<Element>(g).value == i;
        });
    return in_order ? std::optional{source} : std::nullopt;
  }
  // A tangent summed into one n x n matrix: diagonals as a vector, outer
  // products as one low-rank product, and each matrix written contiguously.
  [[nodiscard]] NodeId summed(const Terms &ts) {
    const auto of = [&ts]<typename K>(meta::Type<K>) {
      return ts | std::views::filter([](const Term &t) {
               return std::holds_alternative<K>(t);
             }) |
             std::views::transform(
                 [](const Term &t) -> const K & { return std::get<K>(t); });
    };
    const NodeId d =
        std::ranges::fold_left(ts | std::views::filter(diagonal), no_node,
                               [this](NodeId r, const Term &t) {
                                 const auto *g = std::get_if<term::Diag>(&t);
                                 return add(r, g ? g->d : ones());
                               });
    using Ids = boost::container::small_vector<NodeId, 4>;
    auto matrices = of(meta::type_c<term::Dense>) |
                    std::views::transform(&term::Dense::block) |
                    impl::to<Ids>();
    // sum_k u_k g_k^T is computed as one product U G^T instead of K outer
    // products added together.
    const auto outers =
        of(meta::type_c<term::Outer>) |
        impl::to<boost::container::small_vector<term::Outer, 4>>();
    if (outers.size() == 1) {
      matrices.push_back(
          b_.make(opcode<OpCode::Outer>,
                  {.left = outers.front().u, .right = outers.front().g}));
    } else if (outers.size() > 1) {
      const auto rows = [&](NodeId term::Outer::*side) {
        return b_.pack(outers | std::views::transform(side) | impl::to<Ids>(),
                       OpCode::StackRows);
      };
      matrices.push_back(b_.make(
          opcode<OpCode::MatMul>,
          {.left = b_.make(opcode<OpCode::MatRow>, Orientation::transposed,
                           rows(&term::Outer::u)),
           .right = rows(&term::Outer::g)}));
    }
    // A block with no terms is the zero diagonal. A lone table needs no node,
    // since tensor_layout gives an output table its own row.
    if (d != no_node || matrices.empty()) {
      matrices.push_back(b_.make(OpCode::Diag, d == no_node ? zeros() : d));
    }
    return std::ranges::fold_left(
               matrices | std::views::drop(1), expr(matrices.front()),
               [this](const E &sum, NodeId m) { return sum + expr(m); })
        .id(b_);
  }
  // The full block: the swept block placed at (at_, at_) and each held
  // symbol's reverse-sweep row added as both a row and a column.
  [[nodiscard]] NodeId bordered(std::span<const NodeId> partials,
                                NodeId swept) {
    const auto all = static_cast<std::uint32_t>(partials.size());
    const auto held = [this](std::uint32_t s) {
      return s < at_ || s >= at_ + n_;
    };
    // Places m's rows starting at row at_, with zero rows around them.
    const auto spread_rows = [&](NodeId m) {
      const NodeId zero = b_.zeros({shape_of(b_, m).cols});
      return b_.pack(std::views::iota(0u, all) |
                         std::views::transform([&](std::uint32_t s) {
                           return held(s) ? zero
                                          : b_.make(opcode<OpCode::RowOf>,
                                                    Element{s - at_}, m);
                         }) |
                         impl::to<std::vector<NodeId>>(),
                     OpCode::StackRows);
    };
    const auto transposed = [this](NodeId m) {
      return b_.make(opcode<OpCode::MatRow>, Orientation::transposed, m);
    };
    // Spreads the swept block's columns through the transpose, then its rows.
    const auto placed = [&](NodeId m) {
      return spread_rows(transposed(spread_rows(transposed(m))));
    };
    Terms ts{term::Dense{placed(swept)}};
    // A held symbol's column is its row minus the cells on held symbols, which
    // the rows already hold.
    SweepScratch scratch;
    for (const std::uint32_t s :
         std::views::iota(0u, all) | std::views::filter(held)) {
      auto row = build_reverse_jacobian(b_, partials[s], scratch).partial;
      const NodeId e = unit(s, all);
      ts.push_back(term::Outer{.u = e, .g = b_.pack(row)});
      for (const std::uint32_t t :
           std::views::iota(0u, all) | std::views::filter(held)) {
        row[t] = b_.constant(T{0});
      }
      ts.push_back(term::Outer{.u = b_.pack(row), .g = e});
    }
    return summed(ts);
  }

  // Row arithmetic, where no_node stands for the zero row.
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

  // m * term, where a vector m means diag(m) on the left and a scalar m is a
  // multiple. A diagonal folds into an existing diag so no matrix is copied.
  [[nodiscard]] Term scale(const E &m, const Term &t) {
    const bool vector = m.size() != 0 || m.pending();
    return std::visit(
        boost::hof::match(
            [&](term::Identity) -> Term {
              ++n2_;
              return term::Diag{(vector ? m : m * expr(ones())).id(b_, {n_})};
            },
            [&](const term::Diag &k) -> Term {
              return term::Diag{(m * expr(k.d)).id(b_, {n_})};
            },
            [&](const term::Outer &k) -> Term {
              return term::Outer{.u = (m * expr(k.u)).id(b_, shape_of(b_, k.u)),
                                 .g = k.g};
            },
            [&](const term::Dense &k) -> Term {
              return dense_scale(m, vector, k.block);
            }),
        t);
  }
  [[nodiscard]] Term dense_scale(const E &m, bool vector, NodeId block) {
    const Node<T> inner = b_[block];
    const std::uint32_t rows = shape_of(b_, block).rows;
    const auto rescaled = [&](NodeId e) {
      return (m * expr(e)).id(b_, {width_of(b_, e)});
    };
    NodeId a = no_node;
    if (inner.op == OpCode::RowScale) {
      const auto [scale, matrix] = operands<OpCode::RowScale>(inner);
      a = b_.make(opcode<OpCode::RowScale>,
                  {.scale = rescaled(scale), .matrix = matrix});
    } else if (!vector && inner.op == OpCode::ColScale) {
      const auto [matrix, scale] = operands<OpCode::ColScale>(inner);
      a = b_.make(opcode<OpCode::ColScale>,
                  {.matrix = matrix, .scale = rescaled(scale)});
    } else { // a RowScale, so that the next diagonal folds into it
      a = b_.make(opcode<OpCode::RowScale>,
                  {.scale = vector ? m.id(b_, {rows})
                                   : (m * expr(b_.ones({rows}))).id(b_),
                   .matrix = block});
    }
    return term::Dense{a};
  }
  // Merges like terms: diagonals, dense blocks, and outer products sharing a
  // right factor. Without it, f(x) + g(x) doubles the term list at each level.
  [[nodiscard]] Terms merged(const Terms &ts) {
    const auto diag = [this](const Term &d) {
      const auto *g = std::get_if<term::Diag>(&d);
      return g ? g->d : ones();
    };
    const auto right = [](const Term &d) {
      const auto *o = std::get_if<term::Outer>(&d);
      return o ? o->g : no_node;
    };
    Terms out;
    for (Term t : ts) {
      const auto like = std::ranges::find_if(out, [&](Term &o) {
        return diagonal(o)
                   ? diagonal(t)
                   : t.index() == o.index() && right(t) == right(o) &&
                         shape_of(b_, primary(t)) == shape_of(b_, primary(o));
      });
      if (like == out.end()) {
        out.push_back(t);
        continue;
      }
      if (diagonal(t)) {
        *like = term::Diag{add(diag(*like), diag(t))};
      } else {
        primary(*like) = (expr(primary(*like)) + expr(primary(t)))
                             .id(b_, shape_of(b_, primary(t)));
      }
    }
    return out;
  }
  [[nodiscard]] Terms scale(const E &m, const Terms &ts) {
    return ts |
           std::views::transform([&](const Term &t) { return scale(m, t); }) |
           impl::to<Terms>();
  }
  [[nodiscard]] NodeId row_of(const Term &t, std::uint32_t i) {
    return std::visit(
        boost::hof::match(
            [&](term::Identity) { return unit(i); },
            [&](const term::Diag &k) {
              return b_.make(opcode<OpCode::AddAt>, Element{i},
                             {.vec = zeros(), .scalar = expr(k.d)[i].id(b_)});
            },
            [&](const term::Outer &k) { return scaled(expr(k.u)[i], k.g); },
            [&](const term::Dense &k) {
              return b_.make(opcode<OpCode::RowOf>, Element{i}, k.block);
            }),
        t);
  }
  // u^T times a term, which is a row.
  [[nodiscard]] NodeId left(const E &u, const Term &t) {
    return std::visit(
        boost::hof::match(
            [&](term::Identity) { return u.id(b_, {n_}); },
            [&](const term::Diag &k) { return (u * expr(k.d)).id(b_, {n_}); },
            [&](const term::Outer &k) { return scaled(dot(u, expr(k.u)), k.g); },
            [&](const term::Dense &k) {
              return b_.make(opcode<OpCode::MatVec>, Orientation::transposed,
                             {.matrix = k.block,
                              .vec = u.id(b_, {shape_of(b_, k.block).rows})});
            }),
        t);
  }
  [[nodiscard]] NodeId left(const E &u, const Terms &ts) {
    return std::ranges::fold_left(ts, no_node, [&](NodeId r, const Term &t) {
      return add(r, left(u, t));
    });
  }
  // A constant matrix M, or its transpose, applied on the left of a term.
  [[nodiscard]] Term gemv(NodeId matrix, Orientation orientation,
                          const Term &t) {
    const bool table = is_table(b_, matrix);
    const NodeId c =
        table || orientation == Orientation::transposed
            ? b_.make(opcode<OpCode::MatRow>, orientation,
                      table ? operands<OpCode::MatRow>(b_[matrix]).source
                            : matrix)
            : matrix;
    return std::visit(
        boost::hof::match(
            [&](term::Identity) -> Term {
              ++n2_;
              return term::Dense{c};
            },
            [&](const term::Diag &k) -> Term {
              ++n2_;
              return term::Dense{
                  b_.make(opcode<OpCode::ColScale>, {.matrix = c, .scale = k.d})};
            },
            [&](const term::Outer &k) -> Term {
              return term::Outer{.u = b_.make(opcode<OpCode::MatVec>, orientation,
                                              {.matrix = matrix, .vec = k.u}),
                                 .g = k.g};
            },
            [&](const term::Dense &k) -> Term {
              ++n3_;
              return term::Dense{
                  b_.make(opcode<OpCode::MatMul>, {.left = c, .right = k.block})};
            }),
        t);
  }

  // Whether a Pack is the symbol vector in order, whose tangent is the
  // identity.
  [[nodiscard]] bool is_symbols(NodeId v) const {
    const auto args = arguments(b_, v);
    return args.size() == n_ &&
           std::ranges::all_of(std::views::iota(0u, n_), [&](std::uint32_t j) {
             return args[j] == b_.leaf(OpCode::Var, Slot{at_ + j});
           });
  }

  // A swept symbol's row is e_j, built only when something reads the symbol
  // outside the symbol vector's Pack.
  [[nodiscard]] NodeId row(NodeId u) {
    if (row_[u] == no_node && b_.op_of(u) == OpCode::Var) {
      const std::uint32_t j = get<Slot>(b_[u]).value;
      if (j >= at_ && j < at_ + n_) {
        row_[u] = unit(j - at_);
      }
    }
    return row_[u];
  }

  void visit(NodeId v) {
    const Node<T> node = b_[v]; // a copy, since building may reallocate
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
            std::span{Implicits::row(b_, get<Slot>(node))}.subspan(at_, n_));
      }
      return no_node;
    case Kind::Nary: {
      if (node.op == OpCode::SumN) {
        return std::ranges::fold_left(
            arguments(b_, v), no_node,
            [this](NodeId r, NodeId t) { return add(r, row(t)); });
      }
      // Folds the chain step by step, as the Binary arm handles each step.
      const OpCode step = fold_op(node.op);
      const auto steps = fold_steps(b_, v);
      NodeId r = row(steps.front().first);
      for (const auto &[left, term] : steps) {
        const E l = expr(left);
        const E t = expr(term);
        const auto [pl, pr] = contributions(
            step, E{1}, l, t,
            {.left = r != no_node, .right = row(term) != no_node}, E{0});
        r = add(scaled(pl, r), scaled(pr, row(term)));
      }
      return r;
    }
    case Kind::Extern: {
      NodeId r = no_node;
      extern_partials(b_, v, [&](NodeId arg, const E &c) {
        r = add(r, scaled(c, row(arg)));
      });
      return r;
    }
    // Partials are built only when there is a row to scale.
    case Kind::Unary:
      return row(node.in[0]) == no_node
                 ? no_node
                 : scaled(partial(node.op, expr(node.in[0]), expr(v)),
                          row(node.in[0]));
    case Kind::Binary: {
      // Both partials are built before either product, because make() orders
      // a commutative node's operands by id.
      const E l = expr(node.in[0]);
      const E r = expr(node.in[1]);
      const auto [pl, pr] = contributions(node.op, E{1}, l, r,
                                          {.left = row(node.in[0]) != no_node,
                                           .right = row(node.in[1]) != no_node},
                                          E{0});
      return add(scaled(pl, row(node.in[0])), scaled(pr, row(node.in[1])));
    }
    case Kind::Ternary: { // a select never scales the untaken row by 0
      const auto [cond, then, otherwise] = operands<OpCode::Select>(node.in);
      if (row(then) == no_node && row(otherwise) == no_node) {
        return no_node;
      }
      const auto or_zeros = [this](NodeId r) {
        return expr(r == no_node ? zeros() : r);
      };
      return select(expr(cond), or_zeros(row(then)), or_zeros(row(otherwise)))
          .id(b_, {n_});
    }
    case Kind::Tensor:
      switch (node.op) {
      case OpCode::MatConst: // a constant table
        return no_node;
      case OpCode::Gather:
        return std::ranges::fold_left(
            mat_[node.in[0]], no_node, [&](NodeId r, const Term &t) {
              return add(r, row_of(t, get<Element>(node).value));
            });
      case OpCode::Sum:
        return left(expr(b_.ones(shape_of(b_, node.in[0]))), mat_[node.in[0]]);
      case OpCode::Dot: {
        const auto [l, r] = operands<OpCode::Dot>(node);
        return add(left(expr(l), mat_[r]), left(expr(r), mat_[l]));
      }
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
        out.push_back(term::Identity{});
      } else if (std::ranges::any_of(arguments(b_, v), [this](NodeId g) {
                   return row(g) != no_node;
                 })) {
        ++n2_;
        out.push_back(term::Dense{stacked(arguments(b_, v))});
      }
      break;
    // Partials are built only when there are terms to scale.
    case OpCode::Map:
      if (!mat_[node.in[0]].empty()) {
        out = scale(partial(get<OpCode>(node), expr(node.in[0]), expr(v)),
                    mat_[node.in[0]]);
      }
      break;
    case OpCode::Zip: {
      const auto [l, r] = operands<OpCode::Zip>(node);
      const auto [pu, pw] = contributions(
          get<OpCode>(node), E{1}, expr(l), expr(r),
          {.left = !mat_[l].empty(), .right = !mat_[r].empty()}, E{0});
      out = scale(pu, mat_[l]);
      impl::append(out, scale(pw, mat_[r]));
      out = merged(out);
      break;
    }
    case OpCode::Scale: {
      const auto [scalar, vec] = operands<OpCode::Scale>(node);
      out = scale(expr(scalar), mat_[vec]);
      if (row(scalar) != no_node) {
        ++n2_;
        out.push_back(term::Outer{.u = vec, .g = row(scalar)});
      }
      break;
    }
    case OpCode::MatVec: {
      const auto [matrix, vec] = operands<OpCode::MatVec>(node);
      refused_ = refused_ || !mat_[matrix].empty();
      out = mat_[vec] | std::views::transform([&](const Term &t) {
              return gemv(matrix, get<Orientation>(node), t);
            }) |
            impl::to<Terms>();
      break;
    }
    case OpCode::MatRow: // a constant table
      break;
    case OpCode::Outer:
      refused_ =
          refused_ || !mat_[node.in[0]].empty() || !mat_[node.in[1]].empty();
      break;
    case OpCode::VSelect: {
      const auto [cond_of, then, otherwise] = operands<OpCode::VSelect>(node);
      if (mat_[then].empty() && mat_[otherwise].empty()) {
        break;
      }
      // Spreads a vector condition along each row of the w x n terms.
      const std::uint32_t w = width_of(b_, v);
      const auto block = [&](const Terms &ts) {
        return expr(ts.empty() ? b_.zeros({w, n_}) : summed(ts));
      };
      const E c = expr(cond_of);
      const E cond = c.size()
                         ? expr(b_.make(opcode<OpCode::Outer>,
                                        {.left = cond_of, .right = ones()}))
                         : c;
      ++n2_;
      out.push_back(
          term::Dense{select(cond, block(mat_[then]), block(mat_[otherwise]))
                          .id(b_, {w, n_})});
      break;
    }
    case OpCode::AddAt: {
      const auto [vec, scalar] = operands<OpCode::AddAt>(node);
      out = mat_[vec];
      if (row(scalar) != no_node) {
        ++n2_;
        out.push_back(
            term::Outer{.u = unit(get<Element>(node).value, width_of(b_, vec)),
                        .g = row(scalar)});
      }
      break;
    }
    default:
      std::unreachable(); // matrix rows are built here but never visited
    }
    return out;
  }
  Arena<T> &b_;
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
[[nodiscard]] DenseHessian derive_dense_hessian(Arena<T> &b, NodeId root) {
  const auto g = build_reverse_jacobian(b, root);
  if (g.partial.empty()) {
    return {.value = g.value, .partial = g.partial};
  }
  detail::DenseSweep<T> sweep{b, g.partial, detail::swept_of(b, root)};
  return {.value = g.value,
          .partial = g.partial,
          .block = sweep.block(g.partial),
          .n2_terms = sweep.n2(),
          .n3_terms = sweep.n3()};
}

} // namespace ddx::rt
