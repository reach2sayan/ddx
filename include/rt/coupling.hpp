#pragma once

#include "md/md.hpp"
#include "ops/operations.hpp"
#include "rt/arena.hpp"
#include "rt/opcode.hpp"
#include "util/config.hpp" // DDX_FWD
#include "util/export.hpp"
#include "util/ranges.hpp" // impl::to

#include <boost/container/small_vector.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/stl_interfaces/iterator_interface.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace ddx::rt {

inline constexpr std::size_t no_column = static_cast<std::size_t>(-1);

// A dynamic_bitset because the colouring's hot loop is the pairwise
// intersects() test, which ANDs a word at a time.
using SymbolSet = boost::dynamic_bitset<>;
using CouplingRows = std::vector<SymbolSet>;

class SetBits : public boost::stl_interfaces::v2::proxy_iterator_interface<
                    SetBits, std::forward_iterator_tag, std::size_t> {
  using Interface = boost::stl_interfaces::v2::proxy_iterator_interface<
      SetBits, std::forward_iterator_tag, std::size_t>;

public:
  // Re-exposes the derived postfix ++, which the prefix declaration hides.
  using Interface::operator++;

  constexpr SetBits() noexcept = default;
  explicit SetBits(const SymbolSet &s) : bits_{&s}, at_{s.find_first()} {}

  [[nodiscard]] std::size_t operator*() const noexcept { return at_; }
  SetBits &operator++() {
    at_ = bits_->find_next(at_);
    return *this;
  }
  // An exhausted iterator holds npos, which equals the default-constructed end.
  [[nodiscard]] bool operator==(const SetBits &other) const noexcept {
    return at_ == other.at_;
  }

private:
  const SymbolSet *bits_ = nullptr;
  SymbolSet::size_type at_ = SymbolSet::npos;
};
static_assert(std::forward_iterator<SetBits>);

// The indices of the set bits, in ascending order.
[[nodiscard]] inline auto set_bits(const SymbolSet &s) {
  return std::ranges::subrange{SetBits{s}, SetBits{}};
}

// Views a flat colour-major block as a colours x n matrix.
[[nodiscard]] constexpr auto by_colour(auto &&flat, std::size_t colours,
                                       std::size_t n) {
  return impl::md::mdspan{std::ranges::data(flat),
                          impl::md::dextents<std::size_t, 2>{colours, n}};
}

// A stored cell of a derivative block: its dense (row, column) position and
// the slot where its value is kept in the compressed block.
struct Cell {
  std::size_t row;
  std::size_t column;
  std::size_t slot;
};

// The nonzero cells of a derivative block in CSR form, read off a swept row.
// It is exact, whereas coupling_pattern is conservative.
struct Sparsity {
  std::vector<std::size_t> rowptr; // rows() + 1, or empty for no rows
  std::vector<std::uint32_t> col;  // nonzeros(), ascending within a row
  std::size_t columns = 0;

  // One row's columns and the offset of its first value in the value block.
  struct Row : std::ranges::view_interface<Row> {
    std::span<const std::uint32_t> cols;
    std::size_t offset = 0; // this row's first cell in the value block
    std::size_t index = 0;  // i
    [[nodiscard]] constexpr auto begin() const noexcept { return cols.begin(); }
    [[nodiscard]] constexpr auto end() const noexcept { return cols.end(); }
  };

  [[nodiscard]] constexpr auto rows() const {
    return std::views::iota(0uz, size()) |
           std::views::transform([this](std::size_t i) {
             return Row{.cols = std::span{col}.subspan(
                            rowptr[i], rowptr[i + 1] - rowptr[i]),
                        .offset = rowptr[i],
                        .index = i};
           });
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return rowptr.empty() ? 0 : rowptr.size() - 1;
  }
  [[nodiscard]] constexpr std::size_t nonzeros() const noexcept {
    return col.size();
  }

  // Every stored cell, in row-major order.
  [[nodiscard]] constexpr auto entries() const {
    const auto stored = [](const Row &r) {
      return std::views::zip(std::views::iota(r.offset), r.cols) |
             std::views::transform([i = r.index](const auto &at) {
               return Cell{.row = i,
                           .column = std::get<1>(at),
                           .slot = std::get<0>(at)};
             });
    };
    return rows() | std::views::transform(stored) | std::views::join;
  }
  // Every stored cell as a (row, column, value) tuple read from a value block.
  [[nodiscard]] constexpr auto
  cells(std::ranges::random_access_range auto &&values) const {
    // Captured through views::all so that a view is copied and a container
    // is referenced, never dangling either way. Called rather than piped:
    // an owning_view makes the lambda move-only, which libstdc++ 14.2's pipe
    // adaptor refuses.
    return std::views::transform(
        entries(), [block = std::views::all(DDX_FWD(values))](const Cell &c) {
          return std::tuple{c.row, c.column, block[c.slot]};
        });
  }

  // The slot of (i, j) in the compressed block, or nullopt for a structural
  // zero.
  [[nodiscard]] constexpr std::optional<std::size_t>
  at(std::size_t i, std::size_t j) const noexcept {
    const auto present = rows()[i];
    const auto found =
        std::ranges::lower_bound(present, static_cast<std::uint32_t>(j));
    return found != present.end() && *found == j
               ? std::optional{present.offset + static_cast<std::size_t>(
                                                    found - present.begin())}
               : std::nullopt;
  }
};

// Columns share a colour only when no row couples them, so one sweep can seed
// a whole colour and the results still separate.
struct Colouring {
  std::vector<std::size_t> colour;  // per symbol
  std::vector<std::size_t> scatter; // count() * n; column a (colour, row) owns
  // count() * n; the storage slot of (colour, row), or no_column when no
  // column of that colour owns the cell, so it gets no storage.
  std::vector<std::size_t> cell;

  [[nodiscard]] constexpr std::size_t count() const noexcept {
    return colour.empty() ? 0 : scatter.size() / colour.size();
  }
  // The compressed block's width, which is the number of owned cells.
  [[nodiscard]] constexpr std::size_t width() const noexcept {
    return static_cast<std::size_t>(std::ranges::count_if(
        cell, [](std::size_t k) { return k != no_column; }));
  }
  [[nodiscard]] constexpr auto colours() const noexcept {
    return std::views::iota(0uz, count());
  }
  [[nodiscard]] constexpr auto columns_of(std::size_t c) const {
    return std::views::iota(0uz, colour.size()) |
           std::views::filter(
               [this, c](std::size_t j) { return colour[j] == c; });
  }
  // The cells that one sweep of colour c fills; rows no column owns are
  // skipped.
  [[nodiscard]] constexpr auto owned_cells(std::size_t c) const {
    const auto n = colour.size();
    const auto owner = by_colour(scatter, count(), n);
    const auto slot = by_colour(cell, count(), n);
    return std::views::iota(0uz, n) |
           std::views::transform([c, owner, slot](std::size_t i) {
             return Cell{.row = i, .column = owner[c, i], .slot = slot[c, i]};
           }) |
           std::views::filter(
               [](const Cell &owned) { return owned.column != no_column; });
  }
  // Every stored cell, in colour-major order.
  [[nodiscard]] constexpr auto entries() const {
    return colours() | std::views::transform([this](std::size_t c) {
             return owned_cells(c);
           }) |
           std::views::join;
  }
  // The slot where H(i, j) is stored, or nullopt for a cell known to be zero.
  [[nodiscard]] constexpr std::optional<std::size_t>
  cell_of(std::size_t i, std::size_t j) const {
    const auto n = colour.size();
    const auto colours = count();
    const std::size_t c = colour[j];
    return by_colour(scatter, colours, n)[c, i] == j
               ? std::optional{by_colour(cell, colours, n)[c, i]}
               : std::nullopt;
  }
};

namespace detail {

// Nested loops, as views::cartesian_product measured 7-21% slower here. The
// size_t constraint keeps a dynamic_bitset, a range of bools, out of this one.
template <std::ranges::forward_range A, std::ranges::forward_range B>
  requires std::same_as<std::ranges::range_value_t<A>, std::size_t> &&
           std::same_as<std::ranges::range_value_t<B>, std::size_t>
void couple(CouplingRows &rows, A &&a, B &&b) {
  for (const std::size_t i : a) {
    for (const std::size_t j : b) {
      rows[i].set(j);
      rows[j].set(i);
    }
  }
}
inline void couple(CouplingRows &rows, const SymbolSet &a, const SymbolSet &b) {
  couple(rows, set_bits(a), set_bits(b));
}
static_assert(![]<typename S = SymbolSet>() {
  return requires(CouplingRows &r, S &s) { couple<S &, S &>(r, s, s); };
}());

// One vector element's sorted support. Usually a few symbols, so this is
// smaller than an n-bit SymbolSet per element.
using Indices = boost::container::small_vector<std::size_t, 2>;

[[nodiscard]] inline Indices indices_of(const SymbolSet &s) {
  return std::ranges::to<Indices>(set_bits(s));
}
[[nodiscard]] inline Indices united(const Indices &a, const Indices &b) {
  Indices out;
  std::ranges::set_union(a, b, std::back_inserter(out));
  return out;
}

} // namespace detail

// rows[i][j] is true when d2f/dxi dxj may be nonzero. The pattern is
// symmetric and conservative: a superset costs sweeps, never correctness.
template <impl::Numeric T>
[[nodiscard]] CouplingRows coupling_pattern(const Arena<T> &b, NodeId root) {
  const std::size_t n = b.symbols().size();
  CouplingRows rows(n, SymbolSet(n));

  // Restrict to nodes the root reaches, ignoring other expressions in the
  // arena.
  const auto live = detail::reachable(b, std::span{&root, 1});

  // One support set per reachable node, indexed by rank. Operands precede
  // their users, so they are always ranked first.
  std::vector<NodeId> rank(b.size(), no_node);
  std::vector<SymbolSet> support;
  support.reserve(static_cast<std::size_t>(std::ranges::count(live, true)));
  const auto of = [&](NodeId u) -> const SymbolSet & {
    return support[rank[u]];
  };
  // Per-element supports of a vector where they are known, or empty. They let
  // x[i] * x[j] couple only i and j rather than all of x.
  using Elements = std::vector<detail::Indices>;
  std::vector<Elements> elements;
  elements.reserve(support.capacity());
  const auto each = [&](NodeId u) -> const Elements & {
    return elements[rank[u]];
  };
  // Couples the operands of a field op and returns true. Returns false for any
  // other op, whose caller then couples the union with itself.
  const auto step = [&rows](OpCode op, const auto &l, const auto &r) {
    if (op == OpCode::Mul) {
      detail::couple(rows, l, r);
    } else if (op == OpCode::Div) {
      detail::couple(rows, l, r);
      detail::couple(rows, r, r);
    }
    return op == OpCode::Add || op == OpCode::Mul || op == OpCode::Div;
  };
  // Applies a binary op element by element, coupling like the Binary arm.
  const auto zipped = [&](OpCode op, const Elements &l,
                          const Elements &r) -> std::optional<Elements> {
    if (l.empty() || r.size() != l.size()) {
      return std::nullopt;
    }
    if (op == OpCode::Lt || op == OpCode::Le) {
      return Elements(l.size());
    }
    Elements out(l.size());
    for (const auto k : std::views::iota(0uz, l.size())) {
      out[k] = detail::united(l[k], r[k]);
      if (!step(op, l[k], r[k])) {
        detail::couple(rows, out[k], out[k]);
      }
    }
    return out;
  };
  // Per-element supports for node v, or nullopt to use the whole-node arms.
  const auto elementwise = [&](NodeId v, const Node<T> &node,
                               OpCode op) -> std::optional<Elements> {
    switch (node.op) {
    case OpCode::Pack:
      return arguments(b, v) | std::views::transform([&](NodeId u) {
               return detail::indices_of(of(u));
             }) |
             impl::to<Elements>();
    case OpCode::VecConst:
      return Elements(shape_of(b, v).rows);
    case OpCode::Map:
      if (op != OpCode::Neg) {
        std::ranges::for_each(each(node.in[0]), [&](const auto &e) {
          detail::couple(rows, e, e);
        });
      }
      return each(node.in[0]).empty() ? std::nullopt
                                      : std::optional{each(node.in[0])};
    case OpCode::Zip: {
      const auto [left, right] = detail::operands<OpCode::Zip>(node);
      return zipped(op, each(left), each(right));
    }
    case OpCode::Scale: { // s * v: s against each element
      const auto [scalar, vec] = detail::operands<OpCode::Scale>(node);
      return zipped(OpCode::Mul,
                    Elements(each(vec).size(), detail::indices_of(of(scalar))),
                    each(vec));
    }
    case OpCode::VSelect: { // a selection couples like an Add
      const auto [cond, then, otherwise] =
          detail::operands<OpCode::VSelect>(node);
      return zipped(OpCode::Add, each(then), each(otherwise));
    }
    case OpCode::AddAt: { // linear: v with element i plus a scalar
      const auto [vec, scalar] = detail::operands<OpCode::AddAt>(node);
      if (each(vec).empty()) {
        return std::nullopt;
      }
      Elements out = each(vec);
      auto &e = out[detail::get<Element>(node).value];
      e = detail::united(e, detail::indices_of(of(scalar)));
      return out;
    }
    default:
      return std::nullopt;
    }
  };
  const auto mark = [](SymbolSet &s, auto &&indices) {
    std::ranges::for_each(indices, [&s](std::size_t i) { s.set(i); });
  };
  for (const NodeId v : detail::live_ids(live)) {
    rank[v] = static_cast<NodeId>(support.size());
    SymbolSet &here = support.emplace_back(n);
    Elements &mine = elements.emplace_back();
    const auto unite = [&](const auto &ids) {
      std::ranges::for_each(ids, [&](NodeId u) { here |= of(u); });
    };
    const auto &node = b[v];
    // A Map or Zip couples as its inner op does.
    const OpCode op = detail::effective_op(node);
    // u . w couples each element of u with the same element of w only.
    if (node.op == OpCode::Dot && !each(node.in[0]).empty() &&
        each(node.in[0]).size() == each(node.in[1]).size()) {
      for (const auto k : std::views::iota(0uz, each(node.in[0]).size())) {
        detail::couple(rows, each(node.in[0])[k], each(node.in[1])[k]);
      }
      here = of(node.in[0]);
      here |= of(node.in[1]);
      continue;
    }
    if (node.op == OpCode::Gather && !each(node.in[0]).empty()) {
      mark(here, each(node.in[0])[detail::get<Element>(node).value]);
      continue;
    }
    if (auto known = elementwise(v, node, op)) {
      mine = *std::move(known);
      mark(here, mine | std::views::join);
      continue;
    }
    switch (kind_of(op)) {
    case Kind::Const:
      break;
    case Kind::Input:
      if (node.op == OpCode::Var) {
        here.set(detail::get<Slot>(node).value);
      } else if (node.op == OpCode::Implicit) {
        // An implicit root depends nonlinearly on every symbol its residual
        // reads, so that whole set is coupled with itself.
        const NodeId f = b.closures()[detail::get<Slot>(node).value];
        const auto through = detail::reachable(b, std::span{&f, 1});
        for (const auto [u, leaf] : b.nodes() | std::views::enumerate) {
          if (leaf.op == OpCode::Var && through[static_cast<NodeId>(u)]) {
            here.set(detail::get<Slot>(leaf).value);
          }
        }
        detail::couple(rows, here, here);
      }
      break;
    case Kind::Unary:
      here = of(node.in[0]);
      if (op != OpCode::Neg) {
        detail::couple(rows, here, here);
      }
      break;
    case Kind::Extern:
      unite(arguments(b, v));
      detail::couple(rows, here, here);
      break;
    case Kind::Nary:
      // Each step couples the running support with the next term's.
      for (const auto [k, t] : arguments(b, v) | std::views::enumerate) {
        if (k > 0) {
          step(fold_op(op), here, of(t));
        }
        here |= of(t);
      }
      break;
    case Kind::Ternary:
      // A select couples like an Add, since it only chooses between branches.
      {
        const auto [cond, then, otherwise] =
            detail::operands<OpCode::Select>(node.in);
        here = of(then);
        here |= of(otherwise);
      }
      break;
    case Kind::Binary:
      // Comparisons have zero derivative, so they get no support or coupling.
      if (op == OpCode::Lt || op == OpCode::Le) {
        break;
      }
      here = of(node.in[0]);
      here |= of(node.in[1]);
      if (!step(op, of(node.in[0]), of(node.in[1]))) {
        detail::couple(rows, here, here);
      }
      break;
    case Kind::Tensor:
      // Conservatively the union of the operands' supports. Products couple
      // their two operands; the remaining tensor ops are linear.
      detail::for_each_operand(b, v, [&](NodeId u) { here |= of(u); });
      if (op == OpCode::Scale || op == OpCode::Dot || op == OpCode::MatVec ||
          op == OpCode::Outer) {
        detail::couple(rows, of(node.in[0]), of(node.in[1]));
      }
      break;
    }
  }
  return rows;
}

// Copies each stored cell of a Sparsity or Colouring, points values long, into
// a dense row-major n-wide block. The caller zero-fills the block first.
constexpr void scatter(const auto &layout, auto &&cell, auto &&dense,
                       std::size_t n, std::size_t points) {
  for (const Cell &c : layout.entries()) {
    std::ranges::copy_n(cell(c.slot), static_cast<std::ptrdiff_t>(points),
                        dense(c.row * n + c.column));
  }
}

// CPR column colouring in smallest-last order; columns conflict when their
// coupling rows overlap.
// Ref: Curtis, Powell & Reid, J. Inst. Math. Appl. 13(1) (1974) 117.
// Ref: Coleman & Moré, SIAM J. Numer. Anal. 20(1) (1983) 187.
[[nodiscard]] DDX_API Colouring colour_columns(const CouplingRows &rows);

} // namespace ddx::rt
