#pragma once

#include "rt/archive/snapshot.hpp"
#include "rt/coupling.hpp"
#include "rt/opcode.hpp"
#include "util/error.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <span>
#include <string_view>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

// The trust boundary.  A loaded snapshot has cleared a checksum, which detects
// accidents and confers nothing: what the interpreter, the liveness walk and
// codegen rely on and none re-tests is checked here, and until a file nothing
// could build a graph that broke one.
namespace ddx::rt {
template <impl::Numeric T> class Verified;
} // namespace ddx::rt

namespace ddx::rt::detail {

template <impl::Numeric T> class Sound {
public:
  explicit Sound(Snapshot<T> s) noexcept : s_{std::move(s)} {}

  // Every refusal is the same one: the file parsed, and does not hold.  The
  // order is load-bearing -- each check is what the next one indexes with.
  [[nodiscard]] result<Verified<T>> operator()() && {
    return symbols() && nodes() && vars() && externs() && reachable() &&
                   jacobian() &&
                   hessians() && seeded() && objects()
               ? result<Verified<T>>{Verified<T>{std::move(s_)}}
               : fail(errc::archive_corrupt);
  }

private:
  [[nodiscard]] constexpr auto within() const noexcept {
    return [n = s_.model.nodes.size()](NodeId v) { return v < n; };
  }

  // One namespace for every named input, as Builder::input keeps it: no name
  // twice, within a kind or across two.
  [[nodiscard]] bool symbols() const {
    auto names = s_.model.names | std::views::join |
                 impl::to<std::vector<std::string_view>>();
    std::ranges::sort(names);
    return std::ranges::adjacent_find(names) == names.end();
  }
  [[nodiscard]] std::size_t count(OpCode op) const {
    return s_.model.names[*named_index(op)].size();
  }

  // Topological: every operand strictly below its reader, which is what the
  // runtime single-passes on.  Also what puts every named leaf's slot in range,
  // which vars() then indexes by.
  [[nodiscard]] bool nodes() const {
    // A seed slot is an xs column the kernel loads, and the widest block any
    // lane can carry is one per symbol or one per function.
    const auto nseed = std::max(count(OpCode::Var), s_.roots.size());
    return std::ranges::all_of(
        s_.model.nodes | std::views::enumerate, [this, nseed](const auto &at) {
          const auto &[i, node] = at;
          const auto arity = arity_of(node.op);
          const auto k = named_index(node.op);
          return (arity >= 1 ? node.a < static_cast<NodeId>(i)
                             : node.a == no_node) &&
                 (arity >= 2 ? node.b < static_cast<NodeId>(i)
                             : node.b == no_node) &&
                 (arity == 3 ? node.c < static_cast<NodeId>(i)
                             : node.c == no_node) &&
                 (!k || node.slot < s_.model.names[*k].size()) &&
                 (node.op != OpCode::Seed || node.slot < nseed) &&
                 // As make() sorts them: contraction_at's tie-break reads it.
                 (!is_commutative<T>(node.op) || node.a <= node.b);
        });
  }

  // Exactly one leaf per named input: Builder::restore walks for them, so a
  // name no leaf carries would leave a no_node for var() to hand back.
  // Interning is what makes it exactly one rather than at least one.
  [[nodiscard]] bool vars() const {
    std::array<boost::container::small_vector<std::uint32_t, 32>, named_count>
        seen;
    for (const auto [names, counts] : std::views::zip(s_.model.names, seen)) {
      counts.assign(names.size(), 0);
    }
    for (const auto &node : s_.model.nodes) {
      if (const auto k = named_index(node.op)) {
        ++seen[*k][node.slot]; // nodes() pinned the slot
      }
    }
    return std::ranges::all_of(seen | std::views::join,
                               [](std::uint32_t k) { return k == 1; });
  }

  // Every extern node names a registered function and one of its components,
  // and its argument list is as long as the function's arity: the interpreter
  // and the sweeps walk it without a bound.
  [[nodiscard]] bool externs() const {
    const auto &m = s_.model;
    return std::ranges::all_of(
               m.externs | std::views::enumerate,
               [](const auto &at) {
                 const auto &[i, e] = at;
                 return e.valid() &&
                        ExternSlot::fits(static_cast<std::size_t>(i), e.size());
               }) &&
           std::ranges::all_of(
               std::views::iota(NodeId{0}, static_cast<NodeId>(m.size())),
               [&m](NodeId v) {
                 return kind_of(m.op_of(v)) != Kind::Extern ||
                        (extern_arguments(m, v).has_value() &&
                         ExternSlot::of(m[v].slot).component <
                             m.externs[ExternSlot::of(m[v].slot).function]
                                 .size());
               });
  }

  [[nodiscard]] bool reachable() const {
    const auto in = within();
    return s_.model.closures.size() == count(OpCode::Implicit) &&
           std::ranges::all_of(s_.model.closures, in) &&
           std::ranges::all_of(s_.roots, in) &&
           std::ranges::all_of(s_.jacobian.value, in) &&
           std::ranges::all_of(s_.jacobian.partial, in) && in(s_.jacobian.zero);
  }

  // The shape is pinned against the roots and the symbols first; `rowptr` is
  // then walked whole before a single row is dereferenced off it, and the
  // columns are checked last -- each is an unchecked payload scalar until it
  // is.  The `&&` order is the guarantee: sorted, starting at zero and ending
  // at col.size() is what makes every row's subspan in range.
  [[nodiscard]] bool jacobian() const {
    const auto &j = s_.jacobian;
    const auto &p = j.pattern;
    // Strictly ascending inside a row and inside the symbol count: at() binary
    // searches, so a duplicate or an unsorted row makes a cell unreachable.
    const auto sound_row = [&p](const Sparsity::Row &row) {
      return std::ranges::all_of(
                 row, [&p](std::uint32_t c) { return c < p.columns; }) &&
             std::ranges::is_sorted(row) &&
             std::ranges::adjacent_find(row) == row.end();
    };
    // The row count is `rowptr.size() - 1` rather than a field of its own, so
    // a file cannot claim one thing and carry another.
    return !p.rowptr.empty() && p.size() == s_.roots.size() &&
           p.columns == count(OpCode::Var) && p.col.size() == j.partial.size() &&
           p.rowptr.front() == 0 && p.rowptr.back() == p.col.size() &&
           std::ranges::is_sorted(p.rowptr) &&
           std::ranges::all_of(p, sound_row);
  }

  // One per root, or none: a constant-evaluated equation never swept them.
  [[nodiscard]] bool hessians() const {
    return (s_.hessians.empty() || s_.hessians.size() == s_.roots.size()) &&
           std::ranges::all_of(
               s_.hessians, [this](const Hessian &h) { return coloured(h); });
  }

  // Each product is either whole or absent -- a constant-evaluated equation
  // swept none, and a half-written one is corrupt.
  [[nodiscard]] bool seeded() const {
    const auto in = within();
    const auto nsym = count(OpCode::Var);
    const auto &h = s_.hvp;
    const auto &j = s_.vjp;
    const auto &t = s_.jvp;
    const bool hvp_absent =
        h.value == no_node && h.partial.empty() && h.product.empty();
    const bool vjp_absent = j.value.empty() && j.product.empty();
    const bool jvp_absent = t.value.empty() && t.product.empty();
    return (hvp_absent ||
            (in(h.value) && h.partial.size() == nsym &&
             h.product.size() == nsym && std::ranges::all_of(h.partial, in) &&
             std::ranges::all_of(h.product, in))) &&
           (vjp_absent ||
            (j.value.size() == s_.roots.size() && j.product.size() == nsym &&
             std::ranges::all_of(j.value, in) &&
             std::ranges::all_of(j.product, in))) &&
           (jvp_absent || (t.value.size() == s_.roots.size() &&
                           t.product.size() == s_.roots.size() &&
                           std::ranges::all_of(t.value, in) &&
                           std::ranges::all_of(t.product, in)));
  }

  [[nodiscard]] bool coloured(const Hessian &h) const {
    const auto &c = h.coloring;
    const auto nsym = count(OpCode::Var);
    const auto in = within();
    // The colour count is `scatter.size() / nsym`, not a field, so the only
    // thing left to check is that the division is exact and that both tables
    // are the same grid.  A forged count that wraps into agreeing with its own
    // product is no longer expressible.
    const auto gridded = [nsym](std::span<const std::size_t> v) {
      return nsym == 0 ? v.empty() : v.size() % nsym == 0;
    };
    // Every cell must name a slot inside the compressed block: two sharing one
    // would sum two second derivatives into a single column.  `width()` counts
    // them, so it agrees with `cell` by construction; what it still has to pin
    // is the block that is stored against it.
    const auto width = c.width();
    const auto colors = c.count();
    return in(h.value) && in(h.zero) && std::ranges::all_of(h.partial, in) &&
           std::ranges::all_of(h.compressed, in) && c.color.size() == nsym &&
           gridded(c.scatter) && c.cell.size() == c.scatter.size() &&
           h.partial.size() == nsym &&
           std::ranges::all_of(
               c.color, [colors](std::size_t k) { return k < colors; }) &&
           h.compressed.size() == width &&
           std::ranges::all_of(c.cell,
                               [width](std::size_t k) {
                                 return k == no_column || k < width;
                               }) &&
           std::ranges::all_of(c.scatter, [nsym](std::size_t k) {
             return k == no_column || k < nsym;
           });
  }

  // A lane the enum names; the byte is what a file carries.
  [[nodiscard]] bool objects() const {
    return std::ranges::all_of(s_.objects, [](const Object &o) {
      return std::to_underlying(o.want) < want_count;
    });
  }

  Snapshot<T> s_;
};

} // namespace ddx::rt::detail

namespace ddx::rt {

// An arena rebuilt from a verified snapshot, and what the snapshot still holds
// once its node array has become the arena's.
template <impl::Numeric T> struct Rebuilt {
  std::unique_ptr<Builder<T>> arena;
  Snapshot<T> rest;
};

// A snapshot that holds: every invariant the interpreter, the liveness walk
// and codegen rely on without re-testing.  Only Sound mints one, so an arena
// is never restored from anything less; releasing the snapshot drops the
// proof with it.
template <impl::Numeric T> class Verified {
public:
  [[nodiscard]] const Snapshot<T> &operator*() const noexcept { return s_; }
  [[nodiscard]] const Snapshot<T> *operator->() const noexcept { return &s_; }

  // Owning, and it consumes the model: it *is* the arena's.  `lookup` is the
  // one way a file's extern functions are bound: by name, a function it lacks
  // staying unbound for the equation to refuse.
  [[nodiscard]] Rebuilt<T> rebuild(auto &&lookup) && {
    auto arena = std::make_unique<Builder<T>>();
    arena->restore(std::move(s_.model));
    arena->bind(DDX_FWD(lookup));
    return {.arena = std::move(arena), .rest = std::move(s_)};
  }
  [[nodiscard]] Rebuilt<T> rebuild() && {
    return std::move(*this).rebuild(
        [](std::string_view) { return std::shared_ptr<const ExternFn<T>>{}; });
  }
  [[nodiscard]] Snapshot<T> release() && noexcept { return std::move(s_); }

private:
  friend class detail::Sound<T>;
  explicit Verified(Snapshot<T> s) noexcept : s_(std::move(s)) {}
  Snapshot<T> s_;
};

template <impl::Numeric T>
[[nodiscard]] result<Verified<T>> verified(Snapshot<T> snap) {
  return detail::Sound<T>{std::move(snap)}();
}

} // namespace ddx::rt
