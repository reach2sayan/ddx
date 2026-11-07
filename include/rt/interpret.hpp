#pragma once

#include "rt/apply.hpp"
#include "rt/arena.hpp"
#include "rt/kernels.hpp"
#include "util/config.hpp"

#include <boost/container/small_vector.hpp>

#include <cassert>
#include <cmath>
#include <concepts>
#include <functional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace ddx::rt {

// Points per block sweep, which is two AVX2 registers of doubles.
inline constexpr std::size_t block_lanes = 8;

namespace detail {

// Applies one op across W lanes for any arity. supported() keeps the loop
// compiling at scalars where the op is undefined.
template <std::size_t W, typename Fn, impl::Numeric T, bool Ok,
          std::same_as<T>... Ts>
constexpr void lanes(T *DDX_RESTRICT out,
                     const Ts *DDX_RESTRICT... in) noexcept {
  for (std::size_t k = 0; k < W; ++k) {
    out[k] = supported<Fn, T, Ok>(in[k]...);
  }
}

// x * y + z rounded once, matching what the kernel's llvm.fma lowers to.
template <impl::Numeric T>
[[nodiscard]] constexpr T fused_multiply_add(const T &x, const T &y,
                                             const T &z) noexcept {
  if constexpr (std::floating_point<T>) {
    // Constant-evaluating std::fma before C++26 is a GCC extension, so Clang
    // falls back to unfused arithmetic. The probe asks the compiler directly.
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

// Hoists the sign test out of the lane loop.
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

// Kept out of line for W > 1: inlined, GCC hoists the shared loads above the
// branch and then vectorizes only half of each arm.
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

// Maps a tape row to a pointer to its W lanes.
template <std::size_t W, impl::Numeric U>
[[nodiscard]] constexpr auto lanes_of(std::span<U> tape) {
  return [tape](TapeRow r) { return tape.data() + std::size_t{r} * W; };
}

// The tape layout of a sweep and the tensor kernel table. The tensor tape is
// lane-major, and kernels is null when the sweep scalar is not the arena's.
template <typename U> struct TapeView {
  // The tape row of each node id, or empty when rows equal ids. Only list
  // operands use it; the hot path reads rows from the Step.
  std::span<const TapeRow> scalar_rows{};
  std::span<const TensorRow> tensor_rows{};
  U *vtape = nullptr;
  const TensorKernelTable<U> *kernels = nullptr;
  // Each product's paired partner from Graph::pairs, or empty for no pairing.
  std::span<const NodeId> pairs{};

  [[nodiscard]] constexpr TapeRow row(NodeId v) const noexcept {
    return scalar_rows.empty() ? TapeRow{v} : scalar_rows[v];
  }

  // A view of a frozen graph over a tensor tape the caller owns.
  [[nodiscard]] static constexpr TapeView
  of(const auto &graph, U *vtape, const TensorKernelTable<U> *kernels) {
    return {.scalar_rows = graph.rows(),
            .tensor_rows = graph.tensors().rows,
            .vtape = vtape,
            .kernels = kernels,
            .pairs = graph.pairs()};
  }
};

// Calls an extern function once per lane and keeps this node's component.
// lane is taken by value here and below, for the reason lanes_apply gives.
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

// Computes a tensor node for W lanes through the kernel table, except copies
// done here. Other scalars, or a field-like arena, produce zeros.
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
    // A plain MatRow of a table is read in place rather than copied.
    const auto table = [&b](NodeId u) {
      return is_table(b, u) && get<Orientation>(b[u]) == Orientation::plain;
    };
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
    const auto at = [&](NodeId s, std::size_t k) { return lane(t.row(s))[k]; };
    const bool paired =
        n.op == OpCode::MatVec && !t.pairs.empty() && t.pairs[v] != no_node;
    for (std::size_t k = 0; k < W; ++k) {
      const auto to = mine(k);
      // Arithmetic rows go first through one dispatch, since a second switch
      // is costly on short rows.
      if (!paired &&
          tensor_row(
              b, v, [&](std::size_t i) { return in(n.in[i], k); },
              [&](std::size_t i) { return at(n.in[i], k); }, to, out[k], kt)) {
        continue;
      }
      switch (n.op) {
      case OpCode::VecConst:
        std::ranges::copy(b.tensors()[get<TensorRef>(n).value].data,
                          to.begin());
        break;
      case OpCode::Pack:
        std::ranges::transform(arguments(b, v), to.begin(),
                               [&](NodeId s) { return at(s, k); });
        break;
      // Matrices are column-major.
      case OpCode::MatRow: {
        if (table(v)) { // read in place by in(), copied only for an output
          if (w != 0) {
            std::ranges::copy(in(v, k), to.begin());
          }
          break;
        }
        // Every other MatRow is a transpose of a table or of a row.
        const Shape m = shape_of(b, v); // cols x rows of the source
        const NodeId source = operands<OpCode::MatRow>(n).source;
        const std::span<const U> from =
            b.op_of(source) == OpCode::MatConst
                ? std::span<
                      const U>{b.tensors()[get<TensorRef>(b[source]).value]
                                   .data}
                : in(source, k);
        for (std::size_t i = 0; i < m.cols; ++i) {
          for (std::size_t j = 0; j < m.rows; ++j) {
            to[j + i * m.rows] = from[i + j * m.cols];
          }
        }
        break;
      }
      case OpCode::StackRows: { // item i is row i, so (i, j) is at i + j * rows
        const auto items = arguments(b, v);
        for (const auto [i, item] : items | std::views::enumerate) {
          const auto r = in(item, k);
          for (std::size_t j = 0; j < r.size(); ++j) {
            to[static_cast<std::size_t>(i) + j * items.size()] = r[j];
          }
        }
        break;
      }
      case OpCode::VSelect: { // copied, so the untaken arm is never scaled
        const auto chosen = [](U c) { return c != U{0}; }; // as select_impl
        const auto [cond, then, otherwise] = operands<OpCode::VSelect>(n);
        const auto on = in(then, k);
        const auto off = in(otherwise, k);
        if (!is_tensor(b[cond])) {
          std::ranges::copy(chosen(at(cond, k)) ? on : off, to.begin());
          break;
        }
        const auto c = in(cond, k);
        for (std::size_t j = 0; j < to.size(); ++j) {
          to[j] = chosen(c[j]) ? on[j] : off[j];
        }
        break;
      }
      case OpCode::MatConst: // read in place through its MatRow
        break;
      case OpCode::MatVec: // only paired products reach here
        if (const NodeId mate = t.pairs[v]; mate > v) {
          // The first of a pair computes both in one pass over the matrix.
          const auto [other_at, other_w] = row(mate);
          const std::span<U> other{other_at + k * other_w, other_w};
          const bool transposed =
              get<Orientation>(n) == Orientation::transposed;
          const auto [matrix, vec] = operands<OpCode::MatVec>(n);
          if (kt.gemv_pair != nullptr) {
            kt.gemv_pair(shape_of(b, matrix).rows, in(matrix, k), in(vec, k),
                         transposed ? other : to, transposed ? to : other);
          } else {
            const Shape m = shape_of(b, matrix);
            kt.gemv(m.rows, m.cols, in(matrix, k), in(vec, k), to, transposed);
            kt.gemv(m.rows, m.cols, in(matrix, k), in(vec, k), other,
                    !transposed);
          }
        }
        break;
      default:
        std::unreachable();
      }
    }
  }
}

// Folds the terms in list order, matching the binary chain it replaces. The
// first product is not fused, since fma(x, y, 0) would turn -0.0 into +0.0.
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
            out[k] =
                i == 0 ? r[k]
                       : supported<typename R::functor, U, R::ok>(out[k], r[k]);
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

// Extern, fold and tensor rows, which apply.hpp leaves to the interpreter.
// Testing the kind after the opcode switch costs scalar sweeps nothing.
template <std::size_t W, impl::Numeric T, impl::Numeric U>
DDX_ALWAYS_INLINE constexpr void
lanes_opaque(const Arena<T> &b, NodeId v, const Node<T> &n, bool fused, U *out,
             auto &&lane, const TapeView<U> &t) {
  if (on_tensors(n)) {
    lanes_tensor<W>(b, v, n, out, lane, t);
  } else if (kind_of(n.op) == Kind::Nary) {
    lanes_fold<W>(b, v, n.op, fused, out, lane, t);
  } else {
    lanes_extern<W>(b, v, out, lane, t);
  }
}

// Opaque rows return true and run after the dispatch: called inside it, GCC
// spills the closure and reloads the tape base on every step.
template <std::size_t W, impl::Numeric T, typename U>
DDX_ALWAYS_INLINE constexpr void
lanes_apply(const Arena<T> &b, NodeId i, const Node<T> &n, U *DDX_RESTRICT out,
            auto &&lane, const Step &s, const TapeView<U> &t) {
  const auto &in = s.in;
  const bool opaque = dispatch<U>(s.op, [&]<typename R>(R) {
    using Fn = typename R::functor;
    if constexpr (std::is_void_v<Fn>) {
      return true;
    } else {
      if constexpr (std::same_as<Fn, impl::detail::select_impl>) {
        if (is_tensor(n)) {
          return true; // a tensor Select is handled by lanes_tensor
        }
      }
      // A static_assert rather than an else with std::unreachable(), which
      // would trigger MSVC warning C4702 on the return below.
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

// Executes one Step. nodes is b.nodes() taken once by the caller, because
// reading through b reloads the arena base after every opaque call.
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
  // Not a switch on kind_of(): a kind test before lanes_apply's own switch
  // measured 1.4x slower on libm-heavy graphs from branch mispredictions.
  if (s.op == OpCode::Const) {
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = static_cast<U>(n.value);
    }
  } else if (is_leaf(s.op)) {
    const auto src = at + static_cast<std::ptrdiff_t>(
                              input_column(inputs, s.op, get<Slot>(n)) * W);
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = src[static_cast<std::ptrdiff_t>(k)];
    }
  } else {
    detail::lanes_apply<W>(b, i, n, out, lane, s, t);
  }
}

} // namespace detail

// Evaluates schedule at W points at once. point_lanes is symbol-major (s * W +
// k) and tape is node-major, so every lane loop is contiguous.
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

// Whether every op the roots reach is supported at U. Check this first, since
// apply() silently returns U{} for an unsupported op.
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

// Evaluates schedule at one point into scratch indexed by node id. The schedule
// must be topological and closed under operands.
template <impl::Numeric T, std::ranges::random_access_range R, impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>>
constexpr void evaluate_into(const Arena<T> &b, const R &point,
                             std::span<const Step> schedule, std::span<U> v,
                             const detail::TapeView<U> &tensors = {}) {
  evaluate_block<1>(b, point, schedule, v, tensors);
}

namespace detail {
// A cached schedule and tensor layout for walking ids at many points. values
// is id-indexed and arena-sized, and vtape is extent() long.
class ArenaWalk {
public:
  constexpr ArenaWalk() = default;
  template <impl::Numeric T>
  constexpr ArenaWalk(const Arena<T> &b, auto ids,
                      Contract contract = Contract::on)
      : schedule_{schedule_of(b, ids, contract)},
        layout_{tensor_layout(b, schedule_)} {}
  // Walks every node in id order, where children precede parents.
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
    // At another scalar, callers must have checked computes_at() first.
    assert((std::same_as<U, T> ||
            std::ranges::all_of(schedule_, [&b](const Step &s) {
              return supports<U>(b[s.node].op);
            })));
    // Only the arena's own scalar has a kernel table for tensor rows.
    TapeView<U> tensors{.tensor_rows = layout_.rows, .vtape = vtape.data()};
    if constexpr (std::same_as<U, T>) {
      tensors.kernels = &kernels;
    }
    evaluate_into(b, point, schedule_, values, tensors);
  }
  // Allocates its own tapes and returns the values.
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

// evaluate_all restricted to the nodes the roots reach. The result is still
// id-indexed; entries the roots do not reach must not be read.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto
evaluate_reachable(const Arena<T> &b, std::span<const NodeId> roots,
                   const R &point,
                   const TensorKernelTable<T> &kernels = reference_table<T>,
                   Contract contract = Contract::on) {
  return detail::ArenaWalk{b, detail::live_ids(detail::reachable(b, roots)),
                           contract}(b, point, kernels);
}

// Evaluates every node in id order; this is the reference the JIT is checked
// against. The point's element type picks the arithmetic, so duals work too.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto
evaluate_all(const Arena<T> &b, const R &point,
             const TensorKernelTable<T> &kernels = reference_table<T>) {
  return detail::ArenaWalk{b}(b, point, kernels);
}

} // namespace ddx::rt
