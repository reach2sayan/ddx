#pragma once

#include "md/md.hpp"
#include "ops/operations.hpp"
#include "rt/builder.hpp"
#include "rt/opcode.hpp"
#include "util/config.hpp" // DDX_FWD
#include "util/export.hpp"

#include <boost/describe/class.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/stl_interfaces/iterator_interface.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

// The runtime analogue of drivers/coupling.hpp; `n` is run-time here, so its
// bitset rows become words.
namespace ddx::rt {

inline constexpr std::size_t no_column = static_cast<std::size_t>(-1);

// coupling_pattern's opcode arms mirror what the ops declare of themselves;
// pinned, so the two walks cannot answer different sparsities.
static_assert(impl::SumOp<double>::curvature == impl::Curvature::None &&
              impl::NegateOp<double>::curvature == impl::Curvature::None &&
              impl::MultiplyOp<double>::curvature ==
                  impl::Curvature::Bilinear &&
              impl::DivideOp<double>::curvature == impl::Curvature::Quotient);

// boost::dynamic_bitset, not vector<bool>: the colouring's pairwise overlap
// test is the hot loop, and `intersects` is a word-at-a-time AND.
using SymbolSet = boost::dynamic_bitset<>;
using CouplingRows = std::vector<SymbolSet>;

// `find_first`/`find_next`/`npos` is `begin`/`++`/`end` and dynamic_bitset
// ships no iterator for it, so this is the one walk in the tree that has to be
// written out.  Not `iota | filter`: that tests every clear bit the word skip
// exists to skip, which is what SymbolSet was chosen over vector<bool> for.
// Forward only -- there is no find_prev, and nothing walks backwards.
//
// `v2` -- the CRTP flavour -- and never the unqualified name or `v3`
class SetBits : public boost::stl_interfaces::v2::proxy_iterator_interface<
                    SetBits, std::forward_iterator_tag, std::size_t> {
  using Interface = boost::stl_interfaces::v2::proxy_iterator_interface<
      SetBits, std::forward_iterator_tag, std::size_t>;

public:
  // Declaring the prefix form hides the postfix one the interface derives from
  // it, and without that this is not even an input_iterator.
  using Interface::operator++;

  constexpr SetBits() noexcept = default;
  explicit SetBits(const SymbolSet &s) : bits_{&s}, at_{s.find_first()} {}

  [[nodiscard]] std::size_t operator*() const noexcept { return at_; }
  SetBits &operator++() {
    at_ = bits_->find_next(at_);
    return *this;
  }
  // A spent iterator is npos, which is what a default-constructed one holds.
  [[nodiscard]] bool operator==(const SetBits &other) const noexcept {
    return at_ == other.at_;
  }

private:
  const SymbolSet *bits_ = nullptr;
  SymbolSet::size_type at_ = SymbolSet::npos;
};
static_assert(std::forward_iterator<SetBits>);

// The symbols a row couples, ascending.
[[nodiscard]] inline auto set_bits(const SymbolSet &s) {
  return std::ranges::subrange{SetBits{s}, SetBits{}};
}

// Always colours-major and n wide, said once so `c * n + i` is not respelled.
[[nodiscard]] constexpr auto by_color(auto &&flat, std::size_t colors,
                                      std::size_t n) {
  return impl::md::mdspan{std::ranges::data(flat),
                          impl::md::dextents<std::size_t, 2>{colors, n}};
}

// A stored cell of a derivative block: where it belongs in the dense matrix,
// and where it is actually kept.  CSR by row and colour-major are two ways of
// leaving cells out, and both answer this -- `slot` as the compile-time twin
// names it (symbolic/coupling.hpp's `compressed_entry`).
struct Cell {
  std::size_t row;
  std::size_t column;
  std::size_t slot;
};

// Which cells of a derivative block exist, CSR by row.  Read off a swept row
// rather than propagated: a partial the sweep folded to the zero constant *is*
// structurally zero, so this is exact where `coupling_pattern` is conservative.
//
// The block is its rows: `for (const Row &r : pattern)`, and `cells(values)`
// zips those rows against a value block.  Nothing outside reads `rowptr`.
struct Sparsity {
  std::vector<std::size_t> rowptr; // rows() + 1, or empty for no rows
  std::vector<std::uint32_t> col;  // nonzeros(), ascending within a row
  std::size_t columns = 0;

  // One row: the columns it holds, and where their values sit in a block laid
  // out the way `col` is.
  struct Row : std::ranges::view_interface<Row> {
    std::span<const std::uint32_t> cols;
    std::size_t offset = 0; // this row's first cell in the value block
    std::size_t index = 0;  // i
    [[nodiscard]] constexpr auto begin() const noexcept { return cols.begin(); }
    [[nodiscard]] constexpr auto end() const noexcept { return cols.end(); }
  };

  // Every positional operator is `rowptr`'s, forwarded through
  // `base_reference`; the dereference is the only one that is ours.  `v2` for
  // the reason given on SetBits below.
  class Iterator : public boost::stl_interfaces::v2::proxy_iterator_interface<
                       Iterator, std::random_access_iterator_tag, Row> {
  public:
    constexpr Iterator() noexcept = default;
    [[nodiscard]] constexpr Row operator*() const;

  private:
    friend struct Sparsity;
    friend struct boost::stl_interfaces::access;
    using Base = std::vector<std::size_t>::const_iterator;
    constexpr Iterator(const Sparsity &s, Base at) noexcept
        : owner_{&s}, at_{at} {}
    [[nodiscard]] constexpr Base &base_reference() noexcept { return at_; }
    [[nodiscard]] constexpr const Base &base_reference() const noexcept {
      return at_;
    }
    const Sparsity *owner_ = nullptr;
    Base at_{};
  };

  [[nodiscard]] constexpr Iterator begin() const noexcept {
    return {*this, rowptr.begin()};
  }
  // One short of the end: a row needs rowptr[i + 1] as well as rowptr[i].
  [[nodiscard]] constexpr Iterator end() const noexcept {
    return {*this, rowptr.end() - static_cast<std::ptrdiff_t>(!rowptr.empty())};
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return rowptr.empty() ? 0 : rowptr.size() - 1;
  }
  [[nodiscard]] constexpr std::size_t nonzeros() const noexcept {
    return col.size();
  }

  // Every stored cell, row-major -- the walk the call sites used to rebuild
  // from a count and a subscript.
  [[nodiscard]] constexpr auto entries() const {
    const auto stored = [](const Row &r) {
      return std::views::zip(std::views::iota(r.offset), r.cols) |
             std::views::transform([i = r.index](const auto &at) {
               return Cell{.row = i,
                           .column = std::get<1>(at),
                           .slot = std::get<0>(at)};
             });
    };
    return *this | std::views::transform(stored) | std::views::join;
  }
  // The same walk against a block laid out the way `col` is: (row, column,
  // value), which is what a caller scattering one back to a dense matrix wants.
  [[nodiscard]] constexpr auto
  cells(std::ranges::random_access_range auto &&values) const {
    // Into the closure by value, never by reference.  `views::all` copies a
    // view and binds a container, so both spellings own their way back to the
    // block.
    return entries() |
           std::views::transform(
               [block = std::views::all(DDX_FWD(values))](const Cell &c) {
                 return std::tuple{c.row, c.column, block[c.slot]};
               });
  }

  // Where (i, j) sits in the compressed block, or nullopt for a cell the
  // structure says cannot be nonzero.  A search, not a walk: the one thing the
  // rows above do not answer.
  [[nodiscard]] constexpr std::optional<std::size_t>
  at(std::size_t i, std::size_t j) const noexcept {
    const auto present = begin()[static_cast<std::ptrdiff_t>(i)];
    const auto found =
        std::ranges::lower_bound(present, static_cast<std::uint32_t>(j));
    return found != present.end() && *found == j
               ? std::optional{present.offset + static_cast<std::size_t>(
                                                    found - present.begin())}
               : std::nullopt;
  }

  BOOST_DESCRIBE_CLASS(Sparsity, (), (rowptr, col, columns), (), ())
};

constexpr Sparsity::Row Sparsity::Iterator::operator*() const {
  const std::size_t first = *at_;
  return {.cols = std::span{owner_->col}.subspan(first, *(at_ + 1) - first),
          .offset = first,
          .index = static_cast<std::size_t>(at_ - owner_->rowptr.begin())};
}
static_assert(std::random_access_iterator<Sparsity::Iterator>);

// Two columns share a colour only when no row couples them, so one sweep can
// seed all of a colour's columns and the results still separate.
struct Coloring {
  std::vector<std::size_t> color;   // per symbol
  std::vector<std::size_t> scatter; // count() * n; column a (colour, row) owns
  // count() * n again, but the other question: *where* (colour, row) is stored,
  // or no_column for a cell no column of that colour owns.  Those are the ones
  // a sweep computes for nobody, so they get no storage and no output column.
  std::vector<std::size_t> cell;

  [[nodiscard]] constexpr std::size_t count() const noexcept {
    return color.empty() ? 0 : scatter.size() / color.size();
  }
  // The compressed block's width: how many cells any column owns.
  [[nodiscard]] constexpr std::size_t width() const noexcept {
    return static_cast<std::size_t>(std::ranges::count_if(
        cell, [](std::size_t k) { return k != no_column; }));
  }
  [[nodiscard]] constexpr auto colors() const noexcept {
    return std::views::iota(0uz, count());
  }
  // The symbols one sweep of colour `c` seeds together.
  [[nodiscard]] constexpr auto columns_of(std::size_t c) const {
    return std::views::iota(0uz, color.size()) |
           std::views::filter(
               [this, c](std::size_t j) { return color[j] == c; });
  }
  // The cells colour `c` owns -- what one sweep's result goes into.  A row of
  // that colour that no column owns is computed for nobody, and is absent.
  [[nodiscard]] constexpr auto owned_cells(std::size_t c) const {
    const auto n = color.size();
    const auto owner = by_color(scatter, count(), n);
    const auto slot = by_color(cell, count(), n);
    return std::views::iota(0uz, n) |
           std::views::transform([c, owner, slot](std::size_t i) {
             return Cell{.row = i, .column = owner[c, i], .slot = slot[c, i]};
           }) |
           std::views::filter(
               [](const Cell &owned) { return owned.column != no_column; });
  }
  // Every stored cell, colour-major.  What a caller scatters a compressed block
  // back to (i, j) with, rather than asking `cell_of` about every pair.
  [[nodiscard]] constexpr auto entries() const {
    return colors() | std::views::transform([this](std::size_t c) {
             return owned_cells(c);
           }) |
           std::views::join;
  }
  // Where H(i, j) is stored, or nullopt for a cell the colouring calls zero.
  [[nodiscard]] constexpr std::optional<std::size_t>
  cell_of(std::size_t i, std::size_t j) const {
    const auto n = color.size();
    const auto colors = count();
    const std::size_t c = color[j];
    return by_color(scatter, colors, n)[c, i] == j
               ? std::optional{by_color(cell, colors, n)[c, i]}
               : std::nullopt;
  }
  BOOST_DESCRIBE_CLASS(Coloring, (), (color, scatter, cell), (), ())
};

namespace detail {

// `rows` is never `a` or `b` -- the supports live in their own vector -- so
// setting bits as the sets are walked cannot disturb the walk.
//
// Nested, not `cartesian_product(set_bits(a), set_bits(b))`: measured on this
// box, the product view costs 7-21% over 64 to 1024 symbols at every density,
// where the nested walk is a wash against the callback it replaced.  The
// composed spelling reads better and this is `coupling_pattern`'s hot loop.
inline void couple(CouplingRows &rows, const SymbolSet &a, const SymbolSet &b) {
  for (const std::size_t i : set_bits(a)) {
    for (const std::size_t j : set_bits(b)) {
      rows[i].set(j);
      rows[j].set(i);
    }
  }
}

} // namespace detail

// rows[i][j] is true when d2f/dxi dxj may be nonzero.  Conservative and
// symmetric, as hessian_pattern is: a superset costs sweeps, never correctness.
template <impl::Numeric T>
[[nodiscard]] CouplingRows coupling_pattern(const Builder<T> &b, NodeId root) {
  const std::size_t n = b.symbols().size();
  CouplingRows rows(n, SymbolSet(n));

  // Only what the root reaches: another expression sharing the builder would
  // otherwise contribute couplings this one does not have.
  const auto live = detail::reachable(b, std::span{&root, 1});

  std::vector<SymbolSet> support(b.size());
  for (const NodeId v : detail::live_ids(live)) {
    const auto &node = b[v];
    switch (arity_of(node.op)) {
    case 0:
      support[v] = SymbolSet(n);
      if (node.op == OpCode::Var) {
        support[v].set(node.slot);
      }
      break;
    case 1:
      support[v] = support[node.a];
      // Negation is linear; anything else couples its argument with itself.
      if (node.op != OpCode::Neg) {
        detail::couple(rows, support[v], support[v]);
      }
      break;
    case 3:
      // As linear as an Add: it chooses between two derivatives rather than
      // combining them.
      support[v] = support[node.b];
      support[v] |= support[node.c];
      break;
    default:
      // Zero derivative everywhere it exists, so no support and no coupling --
      // the conservative arm below cost a colour or two per tested quantity.
      if (node.op == OpCode::Lt || node.op == OpCode::Le) {
        support[v] = SymbolSet(n);
        break;
      }
      support[v] = support[node.a];
      support[v] |= support[node.b];
      if (node.op == OpCode::Add) {
        break; // linear: contributes nothing
      } else if (node.op == OpCode::Mul) {
        detail::couple(rows, support[node.a], support[node.b]);
      } else if (node.op == OpCode::Div) {
        detail::couple(rows, support[node.a], support[node.b]);
        detail::couple(rows, support[node.b], support[node.b]);
      } else {
        detail::couple(rows, support[v], support[v]);
      }
      break;
    }
  }
  return rows;
}

// Columns j and k conflict iff their coupling rows overlap -- the CPR colouring
// symbolic/coupling.hpp runs at compile time, and carries the citations.
[[nodiscard]] DDX_API Coloring color_columns(const CouplingRows &rows);

} // namespace ddx::rt
