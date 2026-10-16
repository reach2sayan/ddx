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
constexpr void lanes_fma_signed(bool negated, const T *DDX_RESTRICT x,
                                const T *DDX_RESTRICT y,
                                const T *DDX_RESTRICT z,
                                T *DDX_RESTRICT out) noexcept {
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

// An extern node, W lanes of it: the function is called once per lane and
// the component this node names is kept.  Opaque to every scalar but the
// function's own, which computes_at() tells a sweep before it gets here.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_NOINLINE constexpr void lanes_extern(const Builder<T> &b, NodeId v, U *out,
                                         auto &&lane) {
  if constexpr (std::same_as<U, T>) {
    const auto [fn, component] = ExternSlot::of(b[v].slot);
    const ExternEntry<T> &e = b.externs()[fn];
    const auto args = *extern_arguments(b, v);
    boost::container::small_vector<T, 8> in(args.size());
    boost::container::small_vector<T, 16> answer(e.size());
    for (std::size_t k = 0; k < W; ++k) {
      for (const auto [j, arg] : args | std::views::enumerate) {
        in[static_cast<std::size_t>(j)] = lane(arg)[k];
      }
      (*e.fn)(in, answer);
      out[k] = answer[component];
    }
  } else {
    std::ranges::fill_n(out, W, U{});
  }
}

// The tensor rows a sweep writes: a second tape, `extent` elements per lane,
// laid out lane-major so each lane's vector is contiguous, and the table that
// computes them.  Empty rows is a sweep with no tensor in it.
template <typename U> struct Tensors {
  std::span<const TensorRow> rows{};
  U *vtape = nullptr;
  const KernelTable *kernels = nullptr;
};

// A tensor node, W lanes of it: each lane's vector is one contiguous span
// handed to the kernel table, except the copies -- Pack, Gather, AddAt, a
// VecConst -- which the interpreter does itself.  At any scalar but double
// the rows are zero: computes_at() keeps such a sweep away.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_NOINLINE constexpr void lanes_tensor(const Builder<T> &b, NodeId v,
                                         const Node<T> &n, U *out, auto &&lane,
                                         const Tensors<U> &t) {
  if constexpr (!std::same_as<U, double> || !std::same_as<T, double>) {
    std::ranges::fill_n(out, W, U{});
  } else {
    assert(!t.rows.empty());
    const auto row = [&t](NodeId u) {
      return std::pair{t.vtape + std::size_t{t.rows[u].at} * W,
                       std::size_t{t.rows[u].width}};
    };
    // Lane k of node u, as the kernels read it.
    const auto in = [&row](NodeId u, std::size_t k) {
      const auto [base, w] = row(u);
      return std::span<const U>{base + k * w, w};
    };
    const auto [self, w] = row(v);
    const auto mine = [self, w](std::size_t k) {
      return std::span<U>{self + k * w, w};
    };
    const KernelTable &kt = *t.kernels;
    switch (n.op) {
    case OpCode::VecConst: {
      const auto &data = b.tensors()[n.slot].data;
      for (std::size_t k = 0; k < W; ++k) {
        std::ranges::copy(data, mine(k).begin());
      }
      break;
    }
    case OpCode::Pack: {
      const auto args = *pack_arguments(b, v);
      for (std::size_t k = 0; k < W; ++k) {
        std::ranges::transform(args, mine(k).begin(),
                               [&lane, k](NodeId s) { return lane(s)[k]; });
      }
      break;
    }
    case OpCode::Gather:
      for (std::size_t k = 0; k < W; ++k) {
        out[k] = in(n.a, k)[n.slot];
      }
      break;
    case OpCode::Map:
      for (std::size_t k = 0; k < W; ++k) {
        kt.map(static_cast<OpCode>(n.slot), in(n.a, k), mine(k));
      }
      break;
    case OpCode::Zip:
      for (std::size_t k = 0; k < W; ++k) {
        kt.zip(static_cast<OpCode>(n.slot), in(n.a, k), in(n.b, k), mine(k));
      }
      break;
    case OpCode::Scale:
      for (std::size_t k = 0; k < W; ++k) {
        kt.scale(lane(n.a)[k], in(n.b, k), mine(k));
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
      const TensorConst<T> &m = b.tensors()[b[n.a].slot];
      for (std::size_t k = 0; k < W; ++k) {
        kt.gemv(m.rows, m.cols, m.data, in(n.b, k), mine(k), n.slot != 0);
      }
      break;
    }
    case OpCode::AddAt:
      for (std::size_t k = 0; k < W; ++k) {
        const auto to = mine(k);
        std::ranges::copy(in(n.a, k), to.begin());
        to[n.slot] += lane(n.b)[k];
      }
      break;
    case OpCode::MatConst:  // a table a Gemv reads in place
    case OpCode::PackArgs: // a list; its Pack reads the scalars
      break;
    default:
      std::unreachable();
    }
  }
}

// The rows apply.hpp hands the interpreter whole: an extern's and a tensor's.
// The kind test sits here, past the opcode switch, where it costs a sweep of
// scalars nothing.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_ALWAYS_INLINE constexpr void lanes_opaque(const Builder<T> &b, NodeId v,
                                              const Node<T> &n, U *out,
                                              auto &&lane,
                                              const Tensors<U> &t) {
  if (kind_of(n.op) == Kind::Tensor) {
    lanes_tensor<W>(b, v, n, out, lane, t);
  } else {
    lanes_extern<W>(b, v, out, lane);
  }
}

// Every op's lane loop, off apply.hpp's one dispatch.  An extern's row has
// nothing to call, and is the function's case of the same switch.
template <std::size_t W, impl::Numeric T, typename U>
DDX_ALWAYS_INLINE constexpr void lanes_apply(const Builder<T> &b, NodeId i,
                                             const Node<T> &n,
                                             U *DDX_RESTRICT out, auto &&lane,
                                             const Tensors<U> &t) {
  dispatch<U>(n.op, [&]<typename R>(R) {
    using Fn = typename R::functor;
    if constexpr (std::is_void_v<Fn>) {
      lanes_opaque<W>(b, i, n, out, lane, t);
    } else if constexpr (R::arity == 1) {
      lanes<W, Fn, U, R::ok>(out, lane(n.a));
    } else if constexpr (R::arity == 2) {
      lanes<W, Fn, U, R::ok>(out, lane(n.a), lane(n.b));
    } else if constexpr (R::arity == 3) {
      lanes<W, Fn, U, R::ok>(out, lane(n.a), lane(n.b), lane(n.c));
    } else {
      std::unreachable();
    }
  });
}

// The only place that turns a Step into numbers.  `nodes` is b.nodes(), taken
// once by the caller: read through `b`, every opaque call -- a libm one, an
// extern -- makes the compiler reload the arena's base on the next step.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_ALWAYS_INLINE constexpr void
sweep_step(const Builder<T> &b, std::span<const Node<T>> nodes,
           const Inputs &inputs, const Step &s, auto at, auto &&lane,
           const Tensors<U> &t) {
  const auto &[i, fma] = s;
  const Node<T> &n = nodes[i];
  U *const out = lane(i);
  if (fma) {
    detail::lanes_fma<W>(fma.negated, lane(fma.x), lane(fma.y), lane(fma.z),
                         out);
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
                              input_column(inputs, n.op, n.slot) * W);
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = src[static_cast<std::ptrdiff_t>(k)];
    }
  } else {
    detail::lanes_apply<W>(b, i, n, out, lane, t);
  }
}

// The scratch an arena walk owns for its tensor rows: the layout of its
// schedule and the tape, handed to the sweep as a view.  The reference table
// unless told otherwise, which is what a walk outside any equation computes
// with; an equation passes the table it was given.
template <impl::Numeric U> class TensorScratch {
public:
  constexpr TensorScratch() = default;
  constexpr TensorScratch(const auto &s, std::span<const Step> schedule,
                          const KernelTable &kernels)
      : layout_{tensor_layout(s, schedule)}, tape_(layout_.extent),
        kernels_{&kernels} {}
  [[nodiscard]] constexpr Tensors<U> view() noexcept {
    return {.rows = layout_.rows, .vtape = tape_.data(), .kernels = kernels_};
  }

private:
  TensorLayout layout_;
  std::vector<U> tape_;
  const KernelTable *kernels_ = nullptr;
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
                              const detail::Tensors<U> &tensors = {}) {
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
  return std::ranges::all_of(
      std::views::iota(NodeId{0}, static_cast<NodeId>(b.size())),
      [&](NodeId v) { return !live[v] || supports<U>(b[v].op); });
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
                             const detail::Tensors<U> &tensors = {}) {
  evaluate_block<1>(b, point, schedule, v, tensors);
}

// evaluate_all narrowed to what `roots` reach.  Still id-indexed and
// arena-sized, with the entries the roots miss left value-initialised and not
// to be read.  Worth the extra pass: a derivative sweep leaves a whole Hessian
// in the arena that one root's walk has no use for.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto
evaluate_reachable(const Builder<T> &b, std::span<const NodeId> roots,
                   const R &point,
                   const KernelTable &kernels = reference_table) {
  using U = std::ranges::range_value_t<R>;
  const auto live = detail::reachable(b, roots);
  const auto schedule = detail::schedule_of(b, detail::live_ids(live));
  std::vector<U> v(b.size());
  detail::TensorScratch<U> scratch{b, schedule, kernels};
  evaluate_into(b, point, schedule, std::span<U>{v}, scratch.view());
  return v;
}

// Every node once, in id order: a child always precedes its parent, and the
// reference the JIT is checked against.  The point's element type chooses the
// arithmetic, so Dual<double> carries derivatives through the same walk.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto
evaluate_all(const Builder<T> &b, const R &point,
             const KernelTable &kernels = reference_table) {
  using U = std::ranges::range_value_t<R>;
  const auto schedule = detail::schedule_of(
      b, std::views::iota(NodeId{0}, static_cast<NodeId>(b.size())));
  std::vector<U> v(b.size());
  detail::TensorScratch<U> scratch{b, schedule, kernels};
  evaluate_into(b, point, schedule, std::span<U>{v}, scratch.view());
  return v;
}

} // namespace ddx::rt
