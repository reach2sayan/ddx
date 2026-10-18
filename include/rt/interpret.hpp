#pragma once

#include "rt/apply.hpp"
#include "rt/builder.hpp"
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

} // namespace detail

namespace detail {

// The tape row an id names, W points wide.
template <std::size_t W, impl::Numeric U>
[[nodiscard]] constexpr auto lanes_of(std::span<U> tape) {
  return [tape](NodeId v) { return tape.data() + std::size_t{v} * W; };
}

// Where a sweep's rows are and what computes them: the scalar tape's layout,
// the tensor tape -- `extent` elements per lane, laid out lane-major so each
// lane's vector is contiguous -- and the kernel table.  Empty tensor rows is a
// sweep with no tensor in it.
template <typename U> struct TapeView {
  // By id, the tape row a node's value sits in; empty is the ids themselves,
  // which is what an arena walk's schedule says.  The hot path reads rows off
  // the Step; this is for the reads a Step cannot carry -- an extern's or a
  // Pack's argument list.
  std::span<const std::uint32_t> scalar_rows{};
  std::span<const TensorRow> tensor_rows{};
  U *vtape = nullptr;
  const TensorKernelTable *kernels = nullptr;

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
DDX_NOINLINE constexpr void lanes_extern(const Builder<T> &b, NodeId v, U *out,
                                         auto lane, const TapeView<U> &t) {
  if constexpr (std::same_as<U, T>) {
    const auto [fn, component] = extern_of(b[v]);
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
// VecConst -- which the interpreter does itself.  At any scalar but double
// the rows are zero: computes_at() keeps such a sweep away.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_NOINLINE constexpr void lanes_tensor(const Builder<T> &b, NodeId v,
                                         const Node<T> &n, U *out, auto lane,
                                         const TapeView<U> &t) {
  if constexpr (!std::same_as<U, double> || !std::same_as<T, double>) {
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
      return b.op_of(u) == OpCode::MatRow && !is_transposed(b[u]) &&
             b.op_of(b[u].a) == OpCode::MatConst;
    };
    // Lane k of node u, as the kernels read it.
    const auto in = [&](NodeId u, std::size_t k) {
      if (table(u)) {
        return std::span<const U>{b.tensors()[index_of(b[b[u].a])].data};
      }
      const auto [base, w] = row(u);
      return std::span<const U>{base + k * w, w};
    };
    const auto [self, w] = row(v);
    const auto mine = [self, w](std::size_t k) {
      return std::span<U>{self + k * w, w};
    };
    const TensorKernelTable &kt = *t.kernels;
    switch (n.op) {
    case OpCode::VecConst: {
      const auto &data = b.tensors()[index_of(n)].data;
      for (std::size_t k = 0; k < W; ++k) {
        std::ranges::copy(data, mine(k).begin());
      }
      break;
    }
    case OpCode::Pack: {
      const auto args = arguments(b, v);
      for (std::size_t k = 0; k < W; ++k) {
        std::ranges::transform(args, mine(k).begin(),
                               [&](NodeId s) { return lane(t.row(s))[k]; });
      }
      break;
    }
    case OpCode::Gather:
      for (std::size_t k = 0; k < W; ++k) {
        out[k] = in(n.a, k)[index_of(n)];
      }
      break;
    case OpCode::Map:
      for (std::size_t k = 0; k < W; ++k) {
        kt.map(inner_of(n), in(n.a, k), mine(k));
      }
      break;
    case OpCode::Zip:
      for (std::size_t k = 0; k < W; ++k) {
        kt.zip(inner_of(n), in(n.a, k), in(n.b, k), mine(k));
      }
      break;
    case OpCode::Scale:
      for (std::size_t k = 0; k < W; ++k) {
        kt.scale(lane(t.row(n.a))[k], in(n.b, k), mine(k));
      }
      break;
    case OpCode::Sum:
      for (std::size_t k = 0; k < W; ++k) {
        out[k] = kt.sum(in(n.a, k));
      }
      break;
    case OpCode::Dot:
      for (std::size_t k = 0; k < W; ++k) {
        out[k] = kt.dot(in(n.a, k), in(n.b, k));
      }
      break;
    case OpCode::Gemv: {
      const TensorConst<T> &m = b.tensors()[index_of(b[n.a])];
      for (std::size_t k = 0; k < W; ++k) {
        kt.gemv(m.rows, m.cols, m.data, in(n.b, k), mine(k), is_transposed(n));
      }
      break;
    }
    case OpCode::AddAt:
      for (std::size_t k = 0; k < W; ++k) {
        const auto to = mine(k);
        std::ranges::copy(in(n.a, k), to.begin());
        to[index_of(n)] += lane(t.row(n.b))[k];
      }
      break;
    // The matrix rows.  Shapes come off the arena, as the widths do; a
    // matrix is column-major.
    case OpCode::MatRow: { // a table transposed, or a computed matrix
      if (table(v)) {
        break; // read in place by in()
      }
      // Every other MatRow is a transpose: of the table, or of a row.
      const Shape to = shape_of(b, v); // cols x rows of the source
      for (std::size_t k = 0; k < W; ++k) {
        const std::span<const U> from =
            b.op_of(n.a) == OpCode::MatConst
                ? std::span<const U>{b.tensors()[index_of(b[n.a])].data}
                : in(n.a, k);
        const auto dst = mine(k);
        for (std::size_t i = 0; i < to.cols; ++i) {
          for (std::size_t j = 0; j < to.rows; ++j) {
            dst[j + i * to.rows] = from[i + j * to.cols];
          }
        }
      }
      break;
    }
    case OpCode::Diag:
      for (std::size_t k = 0; k < W; ++k) {
        const auto d = in(n.a, k);
        const auto to = mine(k);
        std::ranges::fill(to, U{});
        for (std::size_t i = 0; i < d.size(); ++i) {
          to[i + i * d.size()] = d[i];
        }
      }
      break;
    case OpCode::Outer:
      for (std::size_t k = 0; k < W; ++k) {
        const auto u = in(n.a, k);
        const auto g = in(n.b, k);
        const auto to = mine(k);
        for (std::size_t j = 0; j < g.size(); ++j) {
          kt.scale(g[j], u, to.subspan(j * u.size(), u.size()));
        }
      }
      break;
    case OpCode::RowScale: // diag(a) M: every column scaled elementwise by a
      for (std::size_t k = 0; k < W; ++k) {
        const auto d = in(n.a, k);
        const auto m = in(n.b, k);
        const auto to = mine(k);
        for (std::size_t j = 0; j < m.size() / d.size(); ++j) {
          kt.zip(OpCode::Mul, d, m.subspan(j * d.size(), d.size()),
                 to.subspan(j * d.size(), d.size()));
        }
      }
      break;
    case OpCode::ColScale: // M diag(b): column j scaled by b[j]
      for (std::size_t k = 0; k < W; ++k) {
        const auto m = in(n.a, k);
        const auto d = in(n.b, k);
        const auto to = mine(k);
        const std::size_t rows = m.size() / d.size();
        for (std::size_t j = 0; j < d.size(); ++j) {
          kt.scale(d[j], m.subspan(j * rows, rows), to.subspan(j * rows, rows));
        }
      }
      break;
    case OpCode::MatVec: {
      const Shape m = shape_of(b, n.a);
      for (std::size_t k = 0; k < W; ++k) {
        kt.gemv(m.rows, m.cols, in(n.a, k), in(n.b, k), mine(k),
                is_transposed(n));
      }
      break;
    }
    case OpCode::MatMul: {
      const Shape l = shape_of(b, n.a);
      const Shape r = shape_of(b, n.b);
      for (std::size_t k = 0; k < W; ++k) {
        kt.gemm(l.rows, l.cols, r.cols, in(n.a, k), in(n.b, k), mine(k));
      }
      break;
    }
    case OpCode::StackRows: { // row i of the matrix is item i: (i, j) at i + j
                              // w
      const auto items = arguments(b, v);
      const std::size_t rows = items.size();
      for (std::size_t k = 0; k < W; ++k) {
        const auto to = mine(k);
        for (const auto [i, item] : items | std::views::enumerate) {
          const auto r = in(item, k);
          for (std::size_t j = 0; j < r.size(); ++j) {
            to[static_cast<std::size_t>(i) + j * rows] = r[j];
          }
        }
      }
      break;
    }
    case OpCode::RowOf: {
      const Shape m = shape_of(b, n.a);
      for (std::size_t k = 0; k < W; ++k) {
        const auto from = in(n.a, k);
        const auto to = mine(k);
        for (std::size_t j = 0; j < m.cols; ++j) {
          to[j] = from[index_of(n) + j * m.rows];
        }
      }
      break;
    }
    case OpCode::MatConst: // a table a Gemv reads in place
      break;
    default:
      std::unreachable();
    }
  }
}

// A sum, W lanes of it: the terms in list order into the running sum, a
// product term as one fma where the step is fused -- the same arithmetic the
// Add chain it replaces took, one node and no dispatch per term.  The first
// term's product is formed, not fused: fma(x, y, 0) would turn its -0.0 into
// +0.0.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_NOINLINE constexpr void lanes_sum(const Builder<T> &b, NodeId v, bool fused,
                                      U *DDX_RESTRICT out, auto lane,
                                      const TapeView<U> &t) {
  const auto row = [&](NodeId u) { return lane(t.row(u)); };
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

// The rows apply.hpp hands the interpreter whole: an extern's, a sum's and a
// tensor's.  The kind test sits here, past the opcode switch, where it costs
// a sweep of scalars nothing.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_ALWAYS_INLINE constexpr void
lanes_opaque(const Builder<T> &b, NodeId v, const Node<T> &n, bool fused,
             U *out, auto &&lane, const TapeView<U> &t) {
  switch (kind_of(n.op)) {
  case Kind::Tensor:
    lanes_tensor<W>(b, v, n, out, lane, t);
    break;
  case Kind::Nary:
    lanes_sum<W>(b, v, fused, out, lane, t);
    break;
  default:
    lanes_extern<W>(b, v, out, lane, t);
  }
}

// Every op's lane loop, off apply.hpp's one dispatch.  A row with nothing to
// call answers true and is called past the switch: called from inside the
// visitor, GCC keeps those instances out of line, the closure goes to memory,
// and the tape base with it -- reloaded on every step of the sweep.
template <std::size_t W, impl::Numeric T, typename U>
DDX_ALWAYS_INLINE constexpr void
lanes_apply(const Builder<T> &b, NodeId i, const Node<T> &n,
            U *DDX_RESTRICT out, auto &&lane, const Step &s,
            const TapeView<U> &t) {
  const auto &in = s.in;
  const bool opaque = dispatch<U>(n.op, [&]<typename R>(R) {
    using Fn = typename R::functor;
    if constexpr (std::is_void_v<Fn>) {
      return true;
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
sweep_step(const Builder<T> &b, std::span<const Node<T>> nodes,
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
  if (n.op == OpCode::Const) {
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = static_cast<U>(n.value);
    }
  } else if (is_leaf(n.op)) {
    const auto src = at + static_cast<std::ptrdiff_t>(
                              input_column(inputs, n.op, index_of(n)) * W);
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = src[static_cast<std::ptrdiff_t>(k)];
    }
  } else {
    detail::lanes_apply<W>(b, i, n, out, lane, s, t);
  }
}

// The scratch an arena walk owns for its tensor rows: the layout of its
// schedule and the tape, handed to the sweep as a view.  The reference table
// unless told otherwise, which is what a walk outside any equation computes
// with; an equation passes the table it was given.
template <impl::Numeric U> class TensorTape {
public:
  constexpr TensorTape() = default;
  constexpr TensorTape(const auto &s, std::span<const Step> schedule,
                       const TensorKernelTable &kernels)
      : layout_{tensor_layout(s, schedule)}, tape_(layout_.extent),
        kernels_{&kernels} {}
  [[nodiscard]] constexpr TapeView<U> view() noexcept {
    return {.tensor_rows = layout_.rows,
            .vtape = tape_.data(),
            .kernels = kernels_};
  }

private:
  TensorLayout layout_;
  std::vector<U> tape_;
  const TensorKernelTable *kernels_ = nullptr;
};

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
constexpr void evaluate_block(const Builder<T> &b, const R &point_lanes,
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
[[nodiscard]] constexpr bool computes_at(const Builder<T> &b,
                                         std::span<const NodeId> roots) {
  const auto live = detail::reachable(b, roots);
  return std::ranges::all_of(detail::live_ids(live),
                             [&b](NodeId v) { return supports<U>(b[v].op); });
}

template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto evaluate(const Builder<T> &b, NodeId root,
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
constexpr void evaluate_into(const Builder<T> &b, const R &point,
                             std::span<const Step> schedule, std::span<U> v,
                             const detail::TapeView<U> &tensors = {}) {
  evaluate_block<1>(b, point, schedule, v, tensors);
}

namespace detail {
// The nodes `ids` names, in order, as evaluate_into leaves them.
template <impl::Numeric T, std::ranges::random_access_range R>
[[nodiscard]] constexpr auto evaluate_over(const Builder<T> &b, auto ids,
                                           const R &point,
                                           const TensorKernelTable &kernels) {
  using U = std::ranges::range_value_t<R>;
  const auto schedule = schedule_of(b, ids);
  std::vector<U> v(b.size());
  TensorTape<U> scratch{b, schedule, kernels};
  evaluate_into(b, point, schedule, std::span<U>{v}, scratch.view());
  return v;
}
} // namespace detail

// evaluate_all narrowed to what `roots` reach.  Still id-indexed and
// arena-sized, with the entries the roots miss left value-initialised and not
// to be read.  Worth the extra pass: a derivative sweep leaves a whole Hessian
// in the arena that one root's walk has no use for.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto
evaluate_reachable(const Builder<T> &b, std::span<const NodeId> roots,
                   const R &point,
                   const TensorKernelTable &kernels = reference_table) {
  const auto live = detail::reachable(b, roots);
  return detail::evaluate_over(b, detail::live_ids(live), point, kernels);
}

// Every node once, in id order: a child always precedes its parent, and the
// reference the JIT is checked against.  The point's element type chooses the
// arithmetic, so Dual<double> carries derivatives through the same walk.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto
evaluate_all(const Builder<T> &b, const R &point,
             const TensorKernelTable &kernels = reference_table) {
  return detail::evaluate_over(
      b, std::views::iota(NodeId{0}, static_cast<NodeId>(b.size())), point,
      kernels);
}

} // namespace ddx::rt
