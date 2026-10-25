#pragma once

#include "rt/apply.hpp"
#include "rt/arena.hpp"
#include "rt/kernels.hpp"
#include "util/config.hpp"

#include <boost/container/small_vector.hpp>

#include <array>
#include <cassert>
#include <cmath> // std::fma
#include <concepts>
#include <functional>
#include <ranges>
#include <span>
#include <type_traits> // std::bool_constant
#include <utility>     // std::unreachable
#include <vector>

namespace ddx::rt {

// Points per block sweep: two AVX2 registers of doubles.
inline constexpr std::size_t block_lanes = 8;

// W points per node instead of one
namespace detail {

// One counted loop for every arity: the operand columns are the pack, and the
// body is apply()'s per element.  `supported` is what keeps the loop compiling
// at a scalar the op has no meaning for.
template <std::size_t W, typename Fn, impl::Numeric T, bool Ok,
          std::same_as<T>... Ts>
constexpr void lanes(T *DDX_RESTRICT out,
                     const Ts *DDX_RESTRICT... in) noexcept {
  for (std::size_t k = 0; k < W; ++k) {
    out[k] = supported<Fn, T, Ok>(in[k]...);
  }
}

// x * y + z rounded once, which is what the kernel's llvm.fma lowers to.
template <impl::Numeric T>
[[nodiscard]] constexpr T fused_multiply_add(const T &x, const T &y,
                                             const T &z) noexcept {
  if constexpr (std::floating_point<T>) {
    // <cmath> is constexpr only in C++26 and folding a libm call before that is
    // a GCC extension, so clang takes the arithmetic unfused here.  The probe
    // asks the compiler rather than the version macros.
    if consteval {
      if constexpr (!requires {
                      std::bool_constant<(std::fma(T{2}, T{3}, T{4}) ==
                                          T{10})>{};
                    }) {
        return T{x * y + z};
      }
    }
    return std::fma(x, y, z);
  } else {
    return T{x * y + z};
  }
}

template <typename Sign> struct fma_fn {
  template <impl::Numeric T>
  constexpr T operator()(const T &x, const T &y, const T &z) const noexcept {
    return fused_multiply_add<T>(Sign{}(x), y, z);
  }
};

// The sign is hoisted out of the lane loop rather than tested in it.
template <std::size_t W, impl::Numeric T>
constexpr void
lanes_fma_signed(bool negated, const T *DDX_RESTRICT x, const T *DDX_RESTRICT y,
                 const T *DDX_RESTRICT z, T *DDX_RESTRICT out) noexcept {
  if (negated) {
    lanes<W, fma_fn<std::negate<>>, T, true>(out, x, y, z);
  } else {
    lanes<W, fma_fn<std::identity>, T, true>(out, x, y, z);
  }
}

// Out of line past one lane.  Inlined into the sweep's switch, GCC hoists the
// two arms' shared loads above the branch and then packs only half of each.
template <std::size_t W, impl::Numeric T>
DDX_NOINLINE constexpr void
lanes_fma_block(bool negated, const T *DDX_RESTRICT x, const T *DDX_RESTRICT y,
                const T *DDX_RESTRICT z, T *DDX_RESTRICT out) noexcept {
  lanes_fma_signed<W>(negated, x, y, z, out);
}

template <std::size_t W, impl::Numeric T>
DDX_ALWAYS_INLINE constexpr void
lanes_fma(bool negated, const T *DDX_RESTRICT x, const T *DDX_RESTRICT y,
          const T *DDX_RESTRICT z, T *DDX_RESTRICT out) noexcept {
  if constexpr (W == 1) {
    lanes_fma_signed<W>(negated, x, y, z, out);
  } else {
    lanes_fma_block<W>(negated, x, y, z, out);
  }
}


// The tape row an id names, W points wide.
template <std::size_t W, impl::Numeric U>
[[nodiscard]] constexpr auto lanes_of(std::span<U> tape) {
  return [tape](NodeId v) { return tape.data() + std::size_t{v} * W; };
}

// Where a sweep's rows are and what computes them: the scalar tape's layout,
// the tensor tape -- `extent` elements per lane, laid out lane-major so each
// lane's vector is contiguous -- and the kernel table, at the sweep's scalar
// and null where that is not the arena's.  Empty tensor rows is a sweep with
// no tensor in it.
template <typename U> struct TapeView {
  // By id, the tape row a node's value sits in; empty is the ids themselves,
  // which is what an arena walk's schedule says.  The hot path reads rows off
  // the Step; this is for the reads a Step cannot carry -- an extern's or a
  // Pack's argument list.
  std::span<const std::uint32_t> scalar_rows{};
  std::span<const TensorRow> tensor_rows{};
  U *vtape = nullptr;
  const TensorKernelTable<U> *kernels = nullptr;

  [[nodiscard]] constexpr std::uint32_t row(NodeId v) const noexcept {
    return scalar_rows.empty() ? v : scalar_rows[v];
  }
};

// An extern node, W lanes of it: the function is called once per lane and
// the component this node names is kept.  Opaque to every scalar but the
// function's own, which computes_at() tells a sweep before it gets here.
// `lane` by value here and in the two below, for the reason lanes_apply
// gives.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_NOINLINE constexpr void lanes_extern(const Arena<T> &b, NodeId v, U *out,
                                         auto lane, const TapeView<U> &t) {
  if constexpr (std::same_as<U, T>) {
    const auto [fn, component] = get<ExternRef>(b[v]);
    const ExternEntry<T> &e = b.externs()[fn];
    const auto args = arguments(b, v);
    boost::container::small_vector<T, 8> in(args.size());
    boost::container::small_vector<T, 16> answer(e.size());
    for (std::size_t k = 0; k < W; ++k) {
      for (const auto [j, arg] : args | std::views::enumerate) {
        in[static_cast<std::size_t>(j)] = lane(t.row(arg))[k];
      }
      (*e.fn)(in, answer);
      out[k] = answer[component];
    }
  } else {
    std::ranges::fill_n(out, W, U{});
  }
}

// A tensor node, W lanes of it: each lane's vector is one contiguous span
// handed to the kernel table, except the copies -- Pack, Gather, AddAt, a
// VecConst -- which the interpreter does itself.  At any scalar but the
// arena's the rows are zero: computes_at() keeps such a sweep away.  So are
// they at a field-like arena, which has no table to compute them.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_NOINLINE constexpr void lanes_tensor(const Arena<T> &b, NodeId v,
                                         const Node<T> &n, U *out, auto lane,
                                         const TapeView<U> &t) {
  if constexpr (!std::same_as<U, T> || !std::floating_point<T>) {
    std::ranges::fill_n(out, W, U{});
  } else {
    assert(!t.tensor_rows.empty());
    const auto row = [&t](NodeId u) {
      return std::pair{t.vtape + std::size_t{t.tensor_rows[u].at} * W,
                       std::size_t{t.tensor_rows[u].width}};
    };
    // A table's MatRow is the table, read in place rather than copied out
    // per sweep; only its transpose is a row of its own.
    const auto table = [&b](NodeId u) {
      return is_table(b, u) && get<Orientation>(b[u]) == Orientation::plain;
    };
    // Lane k of node u, as the kernels read it.
    const auto in = [&](NodeId u, std::size_t k) {
      if (table(u)) {
        return std::span<const U>{
            b.tensors()[get<TensorRef>(b[b[u].in[0]]).value].data};
      }
      const auto [base, w] = row(u);
      return std::span<const U>{base + k * w, w};
    };
    const auto [self, w] = row(v);
    const auto mine = [self, w](std::size_t k) {
      return std::span<U>{self + k * w, w};
    };
    const TensorKernelTable<U> &kt = *t.kernels;
    // Lane k of a scalar operand.
    const auto at = [&](NodeId s, std::size_t k) { return lane(t.row(s))[k]; };
    for (std::size_t k = 0; k < W; ++k) {
      const auto to = mine(k);
      switch (n.op) {
      case OpCode::VecConst:
        std::ranges::copy(b.tensors()[get<TensorRef>(n).value].data,
                          to.begin());
        break;
      case OpCode::Pack:
        std::ranges::transform(arguments(b, v), to.begin(),
                               [&](NodeId s) { return at(s, k); });
        break;
      case OpCode::Gather:
        out[k] = in(n.in[0], k)[get<Element>(n).value];
        break;
      case OpCode::Map:
        kt.map(get<OpCode>(n), in(n.in[0], k), to);
        break;
      case OpCode::Zip:
        kt.zip(get<OpCode>(n), in(n.in[0], k), in(n.in[1], k), to);
        break;
      case OpCode::Scale:
        kt.scale(at(n.in[0], k), in(n.in[1], k), to);
        break;
      case OpCode::Sum:
        out[k] = kt.sum(in(n.in[0], k));
        break;
      case OpCode::Dot:
        out[k] = kt.dot(in(n.in[0], k), in(n.in[1], k));
        break;
      case OpCode::AddAt:
        std::ranges::copy(in(n.in[0], k), to.begin());
        to[get<Element>(n).value] += at(n.in[1], k);
        break;
      // The matrix rows.  Shapes come off the arena, as the widths do; a
      // matrix is column-major.
      case OpCode::MatRow: { // a table transposed, or a computed matrix
        if (table(v)) {
          break; // read in place by in()
        }
        // Every other MatRow is a transpose: of the table, or of a row.
        const Shape m = shape_of(b, v); // cols x rows of the source
        const std::span<const U> from =
            b.op_of(n.in[0]) == OpCode::MatConst
                ? std::span<
                      const U>{b.tensors()[get<TensorRef>(b[n.in[0]]).value]
                                   .data}
                : in(n.in[0], k);
        for (std::size_t i = 0; i < m.cols; ++i) {
          for (std::size_t j = 0; j < m.rows; ++j) {
            to[j + i * m.rows] = from[i + j * m.cols];
          }
        }
        break;
      }
      case OpCode::Diag: {
        const auto d = in(n.in[0], k);
        std::ranges::fill(to, U{});
        for (std::size_t i = 0; i < d.size(); ++i) {
          to[i + i * d.size()] = d[i];
        }
        break;
      }
      case OpCode::Outer: {
        const auto u = in(n.in[0], k);
        const auto g = in(n.in[1], k);
        for (std::size_t j = 0; j < g.size(); ++j) {
          kt.scale(g[j], u, to.subspan(j * u.size(), u.size()));
        }
        break;
      }
      case OpCode::RowScale: { // diag(a) M: every column scaled by a
        const auto d = in(n.in[0], k);
        const auto m = in(n.in[1], k);
        for (std::size_t j = 0; j < m.size() / d.size(); ++j) {
          kt.zip(OpCode::Mul, d, m.subspan(j * d.size(), d.size()),
                 to.subspan(j * d.size(), d.size()));
        }
        break;
      }
      case OpCode::ColScale: { // M diag(b): column j scaled by b[j]
        const auto m = in(n.in[0], k);
        const auto d = in(n.in[1], k);
        const std::size_t rows = m.size() / d.size();
        for (std::size_t j = 0; j < d.size(); ++j) {
          kt.scale(d[j], m.subspan(j * rows, rows), to.subspan(j * rows, rows));
        }
        break;
      }
      case OpCode::MatVec: {
        const Shape m = shape_of(b, n.in[0]);
        kt.gemv(m.rows, m.cols, in(n.in[0], k), in(n.in[1], k), to,
                get<Orientation>(n) == Orientation::transposed);
        break;
      }
      case OpCode::MatMul: {
        const Shape l = shape_of(b, n.in[0]);
        kt.gemm(l.rows, l.cols, shape_of(b, n.in[1]).cols, in(n.in[0], k),
                in(n.in[1], k), to);
        break;
      }
      case OpCode::StackRows: { // item i is row i: (i, j) at i + j rows
        const auto items = arguments(b, v);
        for (const auto [i, item] : items | std::views::enumerate) {
          const auto r = in(item, k);
          for (std::size_t j = 0; j < r.size(); ++j) {
            to[static_cast<std::size_t>(i) + j * items.size()] = r[j];
          }
        }
        break;
      }
      case OpCode::RowOf: {
        const auto from = in(n.in[0], k);
        const std::size_t rows = shape_of(b, n.in[0]).rows;
        for (std::size_t j = 0; j < to.size(); ++j) {
          to[j] = from[get<Element>(n).value + j * rows];
        }
        break;
      }
      case OpCode::VSelect: { // copied, so the arm not taken never meets a 0
        const auto chosen = [](U c) { return c != U{0}; }; // as select_impl
        const auto on = in(n.in[1], k);
        const auto off = in(n.in[2], k);
        if (!is_tensor(b[n.in[0]])) {
          std::ranges::copy(chosen(at(n.in[0], k)) ? on : off, to.begin());
          break;
        }
        const auto c = in(n.in[0], k);
        for (std::size_t j = 0; j < to.size(); ++j) {
          to[j] = chosen(c[j]) ? on[j] : off[j];
        }
        break;
      }
      case OpCode::MatConst: // a table its MatRow reads in place
        break;
      default:
        std::unreachable();
      }
    }
  }
}

// A fold, W lanes of it: the terms in list order into the running value --
// the same arithmetic the chain it replaces took, one node and one dispatch
// for every term.  In a sum a product term is one fma where the step is
// fused.  The first term's product is formed, not fused: fma(x, y, 0) would
// turn its -0.0 into +0.0.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_NOINLINE constexpr void lanes_fold(const Arena<T> &b, NodeId v, OpCode op,
                                       bool fused, U *DDX_RESTRICT out,
                                       auto lane, const TapeView<U> &t) {
  const auto row = [&](NodeId u) { return lane(t.row(u)); };
  if (op != OpCode::SumN) {
    dispatch<U>(fold_op(op), [&]<typename R>(R) {
      if constexpr (R::arity == 2) {
        for (const auto [i, term] : arguments(b, v) | std::views::enumerate) {
          const U *const r = row(term);
          for (std::size_t k = 0; k < W; ++k) {
            out[k] = i == 0 ? r[k]
                            : supported<typename R::functor, U, R::ok>(out[k],
                                                                      r[k]);
          }
        }
      }
    });
    return;
  }
  for (const auto [i, term] : arguments(b, v) | std::views::enumerate) {
    const Contraction c = fused ? product_of(b, term) : Contraction{};
    if (!c) {
      const U *const r = row(term);
      for (std::size_t k = 0; k < W; ++k) {
        out[k] = i == 0 ? r[k] : out[k] + r[k];
      }
      continue;
    }
    const U *const x = row(c.x);
    const U *const y = row(c.y);
    for (std::size_t k = 0; k < W; ++k) {
      if (i == 0) {
        out[k] = c.negated ? -(x[k] * y[k]) : x[k] * y[k];
      } else if (c.negated) {
        out[k] = fma_fn<std::negate<>>{}(x[k], y[k], out[k]);
      } else {
        out[k] = fma_fn<std::identity>{}(x[k], y[k], out[k]);
      }
    }
  }
}

// The rows apply.hpp hands the interpreter whole: an extern's, a fold's and a
// tensor's.  The kind test sits here, past the opcode switch, where it costs
// a sweep of scalars nothing.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_ALWAYS_INLINE constexpr void
lanes_opaque(const Arena<T> &b, NodeId v, const Node<T> &n, bool fused,
             U *out, auto &&lane, const TapeView<U> &t) {
  if (on_tensors(n)) {
    lanes_tensor<W>(b, v, n, out, lane, t);
  } else if (kind_of(n.op) == Kind::Nary) {
    lanes_fold<W>(b, v, n.op, fused, out, lane, t);
  } else {
    lanes_extern<W>(b, v, out, lane, t);
  }
}

// Every op's lane loop, off apply.hpp's one dispatch.  A row with nothing to
// call answers true and is called past the switch: called from inside the
// visitor, GCC keeps those instances out of line, the closure goes to memory,
// and the tape base with it -- reloaded on every step of the sweep.
template <std::size_t W, impl::Numeric T, typename U>
DDX_ALWAYS_INLINE constexpr void
lanes_apply(const Arena<T> &b, NodeId i, const Node<T> &n,
            U *DDX_RESTRICT out, auto &&lane, const Step &s,
            const TapeView<U> &t) {
  const auto &in = s.in;
  const bool opaque = dispatch<U>(s.op, [&]<typename R>(R) {
    using Fn = typename R::functor;
    if constexpr (std::is_void_v<Fn>) {
      return true;
    } else if (std::same_as<Fn, impl::detail::select_impl> && is_tensor(n)) {
      return true; // a Select over tensor rows: the tensor arm's
    } else {
      // The arities the tables spell.  A `std::unreachable()` in a trailing
      // `else` would leave the `return false` below it unreachable on the
      // instance that took it, which MSVC reports as C4702; the assert says
      // the same thing where it costs no code.
      static_assert(R::arity >= 1 && R::arity <= 3);
      if constexpr (R::arity == 1) {
        lanes<W, Fn, U, R::ok>(out, lane(in[0]));
      } else if constexpr (R::arity == 2) {
        lanes<W, Fn, U, R::ok>(out, lane(in[0]), lane(in[1]));
      } else {
        lanes<W, Fn, U, R::ok>(out, lane(in[0]), lane(in[1]), lane(in[2]));
      }
      return false;
    }
  });
  if (opaque) {
    lanes_opaque<W>(b, i, n, s.fused, out, lane, t);
  }
}

// The only place that turns a Step into numbers: rows by the Step's slots, the
// op and a leaf's column by the node.  `nodes` is b.nodes(), taken
// once by the caller: read through `b`, every opaque call -- a libm one, an
// extern -- makes the compiler reload the arena's base on the next step.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_ALWAYS_INLINE constexpr void
sweep_step(const Arena<T> &b, std::span<const Node<T>> nodes,
           const Inputs &inputs, const Step &s, auto at, auto &&lane,
           const TapeView<U> &t) {
  const NodeId i = s.node;
  const Node<T> &n = nodes[i];
  U *const out = lane(s.out);
  if (s.fma) {
    detail::lanes_fma<W>(s.fma.negated, lane(s.in[0]), lane(s.in[1]),
                         lane(s.in[2]), out);
    return;
  }
  // Not a switch on kind_of(): the shape below is 1.3.6's, and on a
  // libm-heavy graph (compare/'s swap) a kind test ahead of lanes_apply's own
  // switch costs 1.4x in mispredicted dispatch.
  if (s.op == OpCode::Const) {
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = static_cast<U>(n.value);
    }
  } else if (is_leaf(s.op)) {
    const auto src =
        at + static_cast<std::ptrdiff_t>(
                 input_column(inputs, s.op, get<Slot>(n).value) * W);
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = src[static_cast<std::ptrdiff_t>(k)];
    }
  } else {
    detail::lanes_apply<W>(b, i, n, out, lane, s, t);
  }
}

} // namespace detail

// The nodes of `schedule`, for W points at once.  `point_lanes` is
// symbol-major (symbol s, lane k at s * W + k) and `tape` node-major, so every
// lane loop is contiguous in both.  Same constraints on the schedule as
// evaluate_into; a batch's tail repeats a point, and those lanes are not read.
// `tensors` is the second tape, sized for this schedule; a schedule with no
// tensor op needs none.
template <std::size_t W, impl::Numeric T, std::ranges::random_access_range R,
          impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>>
constexpr void evaluate_block(const Arena<T> &b, const R &point_lanes,
                              std::span<const Step> schedule, std::span<U> tape,
                              const detail::TapeView<U> &tensors = {}) {
  const auto lane = detail::lanes_of<W>(tape);
  const Inputs inputs = b.inputs();
  const auto nodes = b.nodes();
  for (const Step &s : schedule) {
    detail::sweep_step<W>(b, nodes, inputs, s, std::ranges::begin(point_lanes),
                          lane, tensors);
  }
}

// Whether every op the roots reach computes at U.  What a sweep at a scalar
// other than the graph's asks first: apply() answers U{} for an op the scalar
// lacks, and a zero node is a wrong number with nothing to say so.
template <impl::Numeric U, impl::Numeric T>
[[nodiscard]] constexpr bool computes_at(const Arena<T> &b,
                                         std::span<const NodeId> roots) {
  const auto live = detail::reachable(b, roots);
  return std::ranges::all_of(detail::live_ids(live),
                             [&b](NodeId v) { return supports<U>(b[v].op); });
}

template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto evaluate(const Arena<T> &b, NodeId root,
                                      const R &point) {
  return evaluate_all(b, point)[root];
}

// The nodes of `schedule`, one point at a time, into caller-owned scratch
// indexed by node id.  The schedule must be topological and closed under
// operands; detail::schedule_of over id order and Graph::schedule() are both.
// Entries outside it are left alone.  The width-one block sweep, not a second
// implementation of it.
template <impl::Numeric T, std::ranges::random_access_range R, impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>>
constexpr void evaluate_into(const Arena<T> &b, const R &point,
                             std::span<const Step> schedule, std::span<U> v,
                             const detail::TapeView<U> &tensors = {}) {
  evaluate_block<1>(b, point, schedule, v, tensors);
}

namespace detail {
// What a walk over `ids` takes from the graph alone -- the nodes in order, as
// evaluate_into leaves them, and where their tensor rows sit -- kept once and
// swept at one point after another.  The tapes are the caller's: `values` by
// id and arena-sized, `vtape` extent() long.  The reference table unless told
// otherwise, which is what a walk outside any equation computes with.
class ArenaWalk {
public:
  constexpr ArenaWalk() = default;
  template <impl::Numeric T>
  constexpr ArenaWalk(const Arena<T> &b, auto ids)
      : schedule_{schedule_of(b, ids)}, layout_{tensor_layout(b, schedule_)} {}
  // Every node, in id order: a child always precedes its parent.
  template <impl::Numeric T>
  constexpr explicit ArenaWalk(const Arena<T> &b)
      : ArenaWalk{b,
                  std::views::iota(NodeId{0}, static_cast<NodeId>(b.size()))} {}

  [[nodiscard]] constexpr std::size_t extent() const noexcept {
    return layout_.extent;
  }

  template <impl::Numeric T, impl::Numeric U>
  constexpr void
  operator()(const Arena<T> &b, const auto &point, std::span<U> values,
             std::span<U> vtape,
             const TensorKernelTable<T> &kernels = reference_table<T>) const {
    // At a scalar other than the arena's, an op U lacks answers U{}: the
    // caller asks computes_at() first, as univariate_derivative() does.
    assert((std::same_as<U, T> ||
            std::ranges::all_of(schedule_, [&b](const Step &s) {
              return supports<U>(b[s.node].op);
            })));
    // No table at a scalar other than the arena's: no tensor row computes.
    TapeView<U> tensors{.tensor_rows = layout_.rows, .vtape = vtape.data()};
    if constexpr (std::same_as<U, T>) {
      tensors.kernels = &kernels;
    }
    evaluate_into(b, point, schedule_, values, tensors);
  }
  // Into tapes of its own; the values come back.
  template <impl::Numeric T>
  [[nodiscard]] constexpr auto
  operator()(const Arena<T> &b,
             const std::ranges::random_access_range auto &point,
             const TensorKernelTable<T> &kernels = reference_table<T>) const {
    using U = std::ranges::range_value_t<decltype(point)>;
    std::vector<U> values(b.size()), vtape(extent());
    (*this)(b, point, std::span<U>{values}, std::span<U>{vtape}, kernels);
    return values;
  }

private:
  std::vector<Step> schedule_;
  TensorLayout layout_;
};
} // namespace detail

// evaluate_all narrowed to what `roots` reach.  Still id-indexed and
// arena-sized, with the entries the roots miss left value-initialised and not
// to be read.  Worth the extra pass: a derivative sweep leaves a whole Hessian
// in the arena that one root's walk has no use for.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto
evaluate_reachable(const Arena<T> &b, std::span<const NodeId> roots,
                   const R &point,
                   const TensorKernelTable<T> &kernels = reference_table<T>) {
  return detail::ArenaWalk{b, detail::live_ids(detail::reachable(b, roots))}(
      b, point, kernels);
}

// Every node once, in id order: a child always precedes its parent, and the
// reference the JIT is checked against.  The point's element type chooses the
// arithmetic, so Dual<double> carries derivatives through the same walk.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto
evaluate_all(const Arena<T> &b, const R &point,
             const TensorKernelTable<T> &kernels = reference_table<T>) {
  return detail::ArenaWalk{b}(b, point, kernels);
}

} // namespace ddx::rt
