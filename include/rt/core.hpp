#pragma once

#include "dual/taylor_dual.hpp" // the univariate sweep's scalar
#include "md/md.hpp"
#include "ops/numeric.hpp" // compile_time_factorial
#include "rt/coupling.hpp"
#include "rt/dense.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "rt/pipeline.hpp"
#include "rt/rebalance.hpp"
#include "symbolic/entry.hpp" // CEntry
#include "util/config.hpp"
#include "util/export.hpp"
#include "util/ranges.hpp" // to<C>() and append(), ours

#include <boost/container/small_vector.hpp>

#include "jit/options.hpp"

#include <mutex> // call_once, unique_lock
#include <shared_mutex>

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace ddx::rt::detail {

// A point keyed by name at run time: any range of (name, value), which is the
// map spelling of the names rt::var(name) reads.  Entry keys off a FixedString
// and so cannot say a name the program only learns while it runs.
template <typename R, typename T>
concept CNamedRange =
    std::ranges::input_range<R> &&
    requires(const std::ranges::range_value_t<R> &e) {
      { e.first } -> std::convertible_to<std::string_view>;
      requires impl::Numeric<std::remove_cvref_t<decltype(e.second)>>;
    };

template <typename A, typename T>
concept CPointArg =
    impl::Numeric<std::remove_cvref_t<A>> || impl::CEntry<A> ||
    CNamedRange<A, T> ||
    (std::ranges::input_range<A> &&
     impl::Numeric<std::remove_cvref_t<std::ranges::range_value_t<A>>>);

// CPointArg admits any input_range, so without this a span of columns would try
// to become a scalar.
template <typename R, typename Ptr>
concept CColumns = std::ranges::contiguous_range<R> &&
                   std::convertible_to<std::ranges::range_value_t<R>, Ptr>;

#ifdef DDX_HAS_JIT
// One LLJIT for the process -- a static in a class template would be per
// specialisation -- reached only from a compiling backend, so an interpret-only
// program never brings LLVM up.  Defined in libddx, so a program and the
// library share it.
[[nodiscard]] DDX_API jit::Compiler *shared_compiler();
#endif

// What a sweep gives up around itself where the caller holds nothing.  Visible,
// since an instantiation is no more visible than its arguments: Core<T>.
struct DDX_API Nothing {};

template <typename C> class DDX_TEMPLATE_API Restricted;

// The Jacobian spellings an equation and its restrict() view share.  `D`
// supplies at<W>(), dispatch(), unit() and jacobian_sparsity(), the pattern
// its cells are laid out by.
template <typename D, impl::Numeric T> class DDX_TEMPLATE_API JacobianSurface {
  using Want = rt::Want;

public:
  // Dense row-major m x n; at m == 1 the leading axis goes and this is n long.
  // The graph holds only the cells the pattern names, so the zeros go back in
  // here and this spelling never meets the compression.
  [[nodiscard]] constexpr result<std::vector<T>>
  jacobian(const rt::detail::CPointArg<T> auto &...args) const {
    return self()
        .template at<Want::Jacobian>({}, args...)
        .transform([this](const auto &b) {
          return dense(self().jacobian_sparsity(), b.jacobian);
        });
  }

  // The Jacobian alone, from a graph with no value block: whatever only the
  // value needs is not swept, which on a model whose derivative is simpler
  // than its function is most of the graph.
  [[nodiscard]] constexpr result<std::vector<T>>
  gradient(const rt::detail::CPointArg<T> auto &...args) const {
    return self()
        .template at<Want::Gradient>({}, args...)
        .transform([this](const auto &b) {
          return dense(self().jacobian_sparsity(), b.jacobian);
        });
  }

  // dJ/dv, dense row-major m x n: the derivative of every Jacobian cell along
  // `direction`, one forward sweep however many functions there are.
  [[nodiscard]] result<std::vector<T>>
  jacobian_jvp(std::span<const T> direction,
               const rt::detail::CPointArg<T> auto &...args) const {
    return self()
        .template at<Want::JacobianJvp>(direction, args...)
        .transform([this](const auto &b) {
          return dense(self().jacobian_sparsity(), b.hessian);
        });
  }
  // Along one symbol's basis vector: d/dx_j of the whole Jacobian.
  [[nodiscard]] result<std::vector<T>>
  jacobian_jvp(std::string_view along,
               const rt::detail::CPointArg<T> auto &...args) const {
    return self().unit(along).and_then([&](const auto &e) {
      return jacobian_jvp(std::span<const T>{e}, args...);
    });
  }

  // --- batch ---------------------------------------------------------------
  // `g` and `dg` have jacobian_columns() columns and read with
  // jacobian_pattern().
  [[nodiscard]] result<void>
  jacobian(const rt::detail::CColumns<const T *> auto &xs,
           const rt::detail::CColumns<T *> auto &f,
           const rt::detail::CColumns<T *> auto &g, std::size_t n) const {
    return self().dispatch(Want::Jacobian, as_columns(xs),
                           {.values = as_columns(f), .jacobian = as_columns(g)},
                           n);
  }

  // Its own pipeline, not jacobian()'s with the partials dropped: on a coupled
  // model the gradient is most of the graph.
  [[nodiscard]] result<void>
  gradient(const rt::detail::CColumns<const T *> auto &xs,
           const rt::detail::CColumns<T *> auto &g, std::size_t n) const {
    return self().dispatch(Want::Gradient, as_columns(xs),
                           {.jacobian = as_columns(g)}, n);
  }

  // The Jacobian, and its derivative along `v`.
  [[nodiscard]] result<void>
  jacobian_jvp(const rt::detail::CColumns<const T *> auto &xs,
               const rt::detail::CColumns<const T *> auto &v,
               const rt::detail::CColumns<T *> auto &f,
               const rt::detail::CColumns<T *> auto &g,
               const rt::detail::CColumns<T *> auto &dg, std::size_t n) const {
    return self().dispatch(Want::JacobianJvp, as_columns(widen(xs, v)),
                           {.values = as_columns(f),
                            .jacobian = as_columns(g),
                            .hessian = as_columns(dg)},
                           n);
  }

protected:
  [[nodiscard]] static auto
  as_columns(const std::ranges::contiguous_range auto &r) {
    using Ptr = std::ranges::range_value_t<std::remove_cvref_t<decltype(r)>>;
    return std::span<Ptr const>{std::ranges::data(r), std::ranges::size(r)};
  }

  // The point and the direction, in one array -- values or columns: the
  // named inputs first, then the seeds, which is the order the ABI takes and
  // input_column() states.
  [[nodiscard]] static auto widen(const auto &xs, const auto &seeds) {
    boost::container::small_vector<std::ranges::range_value_t<decltype(xs)>, 64>
        out;
    out.reserve(std::ranges::size(xs) + std::ranges::size(seeds));
    std::ranges::copy(xs, std::back_inserter(out));
    std::ranges::copy(seeds, std::back_inserter(out));
    return out;
  }

  // Row-major rows x columns from the cells the pattern names, the zeros put
  // back.
  [[nodiscard]] static constexpr std::vector<T>
  dense(const Sparsity &pattern, std::span<const T> cells) {
    std::vector<T> out(pattern.size() * pattern.columns, T{0});
    scatter(
        pattern, [&cells](std::size_t k) { return &cells[k]; },
        [&out](std::size_t c) { return &out[c]; }, pattern.columns, 1);
    return out;
  }

private:
  [[nodiscard]] constexpr const D &self() const {
    return static_cast<const D &>(*this);
  }
};

// A sweep built on first ask and kept: the flag beside its value, so neither
// travels without the other.
template <typename V> class Lazy {
public:
  const V &get(auto &&build) {
    std::call_once(flag_, [&] { value_ = build(); });
    return value_;
  }
  // Read only after get(): what a freeze reads of the sweeps.
  [[nodiscard]] const V &peek() const noexcept { return value_; }

private:
  std::once_flag flag_;
  V value_;
};

#ifdef DDX_HAS_JIT
// The roots and Jacobian a kernel is lowered from; see rt/rebalance.hpp.
struct Rebalanced {
  std::vector<NodeId> roots;
  Jacobian derivative;
};
#endif

// The arena's lock and every sweep built on first ask.  On the heap: neither
// moves, and constant evaluation never reaches them.
//
// Lock order: a pipeline's lock, then the arena's.  A sweep appends to the
// arena, so it writes; an interpreter reads.  Nothing under the arena's lock
// touches a pipeline, and nothing under either reacquires what a Yield gave up.
struct Sweeps {
  std::shared_mutex arena_mutex;
  Lazy<std::vector<Hessian>> hessians;
  Lazy<HessianVector> hvp;
  Lazy<VectorJacobian> vjp;
  Lazy<Tangent> jvp;
  Lazy<DenseHessian> dense;
  Lazy<std::vector<NodeId>> tangent; // one forward sweep
  Lazy<ArenaWalk> walk;              // a system's hessian() at a point
#ifdef DDX_HAS_JIT
  Lazy<Rebalanced> rebalanced;
#endif
};

// What a thread sweeps into, per scalar type: kept between calls, since two
// threads may be inside one equation at once and resize() on a kept vector
// allocates only when a sweep outgrows the last.  One set per call site.
// release_tapes() gives them back.
template <impl::Numeric T> struct ThreadTapes {
  std::vector<T> values, value_vtape; // a system's hessian() at a point
  std::vector<T> at, tape, vtape;     // a batch, one point at a time
  std::vector<T> lanes, lane_tape, lane_vtape; // a batch, block_lanes at once
  std::vector<T> kernel_tapes;                 // a kernel's tensor steps
  // A point call's output addresses: a dense block is n^2 of them, and a fresh
  // array that size is a page fault per page on every call.
  std::vector<T *> addresses;
};
// Block scope: GCC 15 never registers the destructor of a thread_local variable
// template named from a template, so every thread exit leaked its tapes.
template <impl::Numeric T>
[[nodiscard]] DDX_TEMPLATE_API ThreadTapes<T> &thread_tapes() {
  thread_local ThreadTapes<T> kept;
  return kept;
}

// An equation: an arena of its own, its roots, the Jacobian swept up front and
// every other sweep on first ask, a pipeline per Want.  Equation, System and
// the Python Equation are this plus the spellings their callers want.  `Yield`
// is what a sweep gives up while it runs: the GIL, on the Python side.
template <impl::Numeric T, std::default_initializable Yield = Nothing>
class DDX_TEMPLATE_API Core : public JacobianSurface<Core<T, Yield>, T> {
  // Freezes its own pipelines through checked(), over cells of this Jacobian.
  friend class Restricted<Core>;
  using Surface = JacobianSurface<Core, T>;
  friend Surface;

public:
  using value_type = T;
  using Want = rt::Want;

  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return bad_.has_value();
  }
  [[nodiscard]] constexpr std::optional<error> status() const noexcept {
    return bad_;
  }

  // Optional, not a degenerate 0: a literal-only graph legitimately has none.
  [[nodiscard]] constexpr std::optional<std::size_t> arity() const {
    return unless_poisoned([this] { return symbol_count(); });
  }
  [[nodiscard]] constexpr std::optional<std::size_t> outputs() const {
    return unless_poisoned([this] { return roots_.size(); });
  }
  // Declaration order, which is the order of a positional point and of every
  // derivative's columns.
  [[nodiscard]] constexpr std::optional<std::span<const std::string>>
  symbols() const {
    return unless_poisoned(
        [this] { return std::span<const std::string>{arena_->symbols()}; });
  }
  [[nodiscard]] constexpr std::optional<std::span<const std::string>>
  params() const {
    return unless_poisoned(
        [this] { return std::span<const std::string>{arena_->params()}; });
  }
  // Everything a point supplies, in the order a positional one is read: the
  // symbols, then the parameters.
  [[nodiscard]] constexpr std::optional<std::size_t> input_count() const {
    return unless_poisoned([this] { return arena_->inputs().named(); });
  }
  [[nodiscard]] std::optional<std::vector<std::string>> inputs() const {
    return unless_poisoned([this] { return arena_->names(); });
  }

  // The input list exists only at run time, so every spelling can refuse.
  template <rt::detail::CPointArg<T>... Args>
  [[nodiscard]] constexpr result<std::vector<T>>
  point(const Args &...args) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    std::vector<T> at(arena_->inputs().named(), T{});
    result<void> ok{};
    if constexpr ((impl::CEntry<Args> && ...) && sizeof...(Args) > 0) {
      std::vector<bool> seen(at.size(), false);
      ((ok = ok.and_then([&] { return assign_named(at, seen, args); })), ...);
      ok = ok.and_then([&] { return whole(seen); });
    } else if constexpr (sizeof...(Args) == 1 &&
                         (rt::detail::CNamedRange<Args, T> && ...)) {
      // Before the plain range: a map is an input_range too, and its elements
      // are pairs rather than the numbers assign_range would read.
      ok = assign_named_range(at, args...);
    } else if constexpr (sizeof...(Args) == 1 &&
                         (std::ranges::input_range<Args> && ...)) {
      ok = assign_range(at, args...);
    } else {
      const std::array<T, sizeof...(Args)> positional{static_cast<T>(args)...};
      ok = assign_range(at, positional);
    }
    return ok.transform([&at] { return std::move(at); });
  }

  // H(x)v, n long.  The direction leads and is a span, so it cannot be read as
  // a point value however the point itself is spelled.
  [[nodiscard]] result<std::vector<T>>
  hvp(std::span<const T> direction,
      const rt::detail::CPointArg<T> auto &...args) const {
    return at<Want::Hvp>(direction, args...).transform([](auto b) {
      return std::move(b.hessian);
    });
  }

  // The j-th Hessian column, named and not indexed: the symbol list exists
  // only at run time.
  [[nodiscard]] result<std::vector<T>>
  hvp(std::string_view along,
      const rt::detail::CPointArg<T> auto &...args) const {
    return unit(along).and_then(
        [&](const auto &e) { return hvp(std::span<const T>{e}, args...); });
  }

  // w'J, n long: one weight per function.
  [[nodiscard]] result<std::vector<T>>
  vjp(std::span<const T> weights,
      const rt::detail::CPointArg<T> auto &...args) const {
    return at<Want::Vjp>(weights, args...).transform([](auto b) {
      return std::move(b.jacobian);
    });
  }

  // J v, m long: the mirror of vjp().
  [[nodiscard]] result<std::vector<T>>
  jvp(std::span<const T> direction,
      const rt::detail::CPointArg<T> auto &...args) const {
    return at<Want::Jvp>(direction, args...).transform([](auto b) {
      return std::move(b.jacobian);
    });
  }

  // A view with only the named symbols' Jacobian columns, in symbol order: the
  // cells of the others are neither frozen nor swept.  It reads this
  // equation's arena and sweeps, so it must not outlive it.
  template <std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_reference_t<R>,
                                 std::string_view>
  [[nodiscard]] result<Restricted<Core>> restrict(const R &names) const {
    return alive().and_then([&]() -> result<Restricted<Core>> {
      std::vector<bool> chosen(symbol_count(), false);
      return std::ranges::fold_left(
                 names, result<void>{},
                 [&](result<void> ok, std::string_view name) {
                   return ok.and_then([&] {
                     return symbol(name).transform(
                         [&](std::uint32_t slot) { chosen[slot] = true; });
                   });
                 })
          .transform([&] { return Restricted<Core>{*this, chosen}; });
    });
  }
  [[nodiscard]] result<Restricted<Core>> restrict(
      std::initializer_list<std::string_view> names) const {
    return restrict(std::span{names});
  }

  // F(z, x) for each implicit's closure, at a point: zero, to rounding, where
  // the caller's root is one.  ddx never solves for z; this is the check.
  [[nodiscard]] result<std::vector<T>>
  residual(const rt::detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      [[maybe_unused]] const auto held = reading();
      const auto closures = arena_->closures();
      const auto values =
          evaluate_reachable(*arena_, closures, at, tensor_kernels());
      return closures |
             std::views::transform([&values](NodeId f) { return values[f]; }) |
             impl::to<std::vector<T>>();
    });
  }

  // Dense row-major m x n x n; the graph holds it compressed by colour.
  [[nodiscard]] result<std::vector<T>>
  hessian(const rt::detail::CPointArg<T> auto &...args) const {
    if (!poisoned() && is_system()) {
      // Off the arena, one output's Hessian at a time: no pipeline holds m
      // colourings.  The gate is the pipeline's, asked without one.
      return point(args...).and_then([this](const auto &at) {
        return orders(Want::Hessian).transform([&] {
          const std::size_t m = roots_.size();
          const std::size_t n = symbol_count();
          std::vector<T> f(m);
          std::vector<T> g(m * n);
          std::vector<T> out(m * n * n);
          walk(at, f, g, out);
          return out;
        });
      });
    }
    return at<Want::Hessian>({}, args...).transform([this](const auto &b) {
      // Only the cells the colouring stores; `out` is already zero, which is
      // what every other column of that colour is.
      const std::size_t n = symbol_count();
      std::vector<T> out(n * n);
      scatter(
          hessians().front().colouring,
          [&b](std::size_t k) { return &b.hessian[k]; },
          [&out](std::size_t c) { return &out[c]; }, n, 1);
      return out;
    });
  }

  // The same n x n, dense and row-major, from one O(n^2) sweep: the gradient's
  // cone walked with matrix tangents, where hessian() colours and sweeps n + 1
  // times over a dense term.  One output only.
  [[nodiscard]] result<std::vector<T>>
  dense_hessian(const rt::detail::CPointArg<T> auto &...args) const {
    return at<Want::DenseHessian>({}, args...).transform([](auto b) {
      return std::move(b.hessian);
    });
  }
  // How many terms the sweep kept quadratic, and how many fell back to a dense
  // product: a model's structure defeated the algebra where the second is not
  // zero -- still right, not O(n^2).  Nothing for a system, or for a block the
  // sweep refused, which dense_hessian() answers with tensor_shape.
  struct DenseCost {
    std::size_t n2_terms = 0;
    std::size_t n3_terms = 0;
  };
  [[nodiscard]] std::optional<DenseCost> dense_hessian_cost() const {
    if (poisoned() || is_system()) {
      return std::nullopt;
    }
    const DenseHessian &d = dense_sweep();
    return d.block == no_node
               ? std::nullopt
               : std::optional{
                     DenseCost{.n2_terms = d.n2_terms, .n3_terms = d.n3_terms}};
  }

  // Whether to compile at all, and how; discards anything already compiled.
  // The one non-const member: a call overlapping it is the caller's race.
  auto &options(this auto &self, const jit::Options &opt) {
    if (opt == self.options_) {
      return self;
    }
    self.options_ = opt;
    self.pipelines_ = std::make_unique<Pipelines>();
    return self;
  }
  [[nodiscard]] const jit::Options &options() const noexcept {
    return options_;
  }

  // What computes the tensor nodes: the process default
  // (rt::use_tensor_kernels) unless this equation is handed a table of its own.
  // A table decides the last bits, so the pipelines go with the old one.
  auto &tensor_kernels(this auto &self, const TensorKernelTable<T> &table) {
    if (&table == self.kernels_) {
      return self;
    }
    self.kernels_ = &table;
    self.pipelines_ = std::make_unique<Pipelines>();
    return self;
  }
  [[nodiscard]] constexpr const TensorKernelTable<T> &
  tensor_kernels() const noexcept {
    if (kernels_ != nullptr) {
      return *kernels_;
    }
    // The process default lives in libddx, which constant evaluation cannot
    // reach; a constant-evaluated equation computes no tensor row anyway.
    if consteval {
      return reference_table<T>;
    }
    return rt::tensor_kernels<T>();
  }

  // Sweep and freeze `wants` now, so the calls that follow -- from any number
  // of threads -- find their pipelines built and never take a write lock.
  [[nodiscard]] result<void> prepare(std::initializer_list<Want> wants) const {
    return std::ranges::fold_left(
        wants, result<void>{}, [this](result<void> ok, Want want) {
          return ok.and_then([&] { return ready(want); }).transform([&] {
            (void)pipeline(want);
          });
        });
  }

  // Whether a batch call goes through compiled code: false with no backend,
  // on a refused compile, and under Interpret.
  [[nodiscard]] bool uses_kernel() const {
    return !poisoned() && static_cast<bool>(pipeline(Want::Jacobian)->kernel);
  }

  // The codegen level of the kernel answering, and nothing where the sweep is.
  [[nodiscard]] std::optional<jit::OptLevel> kernel_level() const {
    if (poisoned()) {
      return std::nullopt;
    }
    if (const auto l = pipeline(Want::Jacobian); l->kernel) {
      return l->level;
    } else {
      return std::nullopt;
    }
  }

  // Build `want`'s pipeline now and answer whether a kernel answers it.
  [[nodiscard]] bool wait_for_kernel(Want want = Want::Jacobian) const {
    return ready(want)
        .transform([&] { return static_cast<bool>(pipeline(want)->kernel); })
        .value_or(false);
  }

  // One Taylor sweep: seed c[0] = x0, c[1] = 1, then un-normalise c[Order].
  template <std::size_t Order>
    requires(Order > 0)
  [[nodiscard]] result<T> univariate_derivative(T x0) const {
    return single().and_then([&]() -> result<T> {
      if (arena_->inputs().named() != 1 || symbol_count() != 1) {
        return fail(errc::not_univariate);
      }
      using Taylor = impl::TaylorDual<T, Order>;
      const std::span<const NodeId> root{roots_.data(), 1};
      [[maybe_unused]] const auto held = reading();
      // Refuse rather than sweep: an op the Taylor scalar lacks evaluates to
      // a zero node, which is a wrong number with nothing to say so.
      if (!computes_at<Taylor>(*arena_, root)) {
        return fail(errc::unsupported_scalar);
      }
      const std::array<Taylor, 1> at{Taylor::variable(x0)};
      // One root out of an arena that may hold a gradient and a Hessian too,
      // and a Taylor node costs (Order + 1) coefficients.
      const auto values = evaluate_reachable(*arena_, root, at);
      return values[roots_[0]].c[Order] *
             static_cast<T>(impl::detail::compile_time_factorial(Order));
    });
  }

  // --- batch ---------------------------------------------------------------
  // `xs[j]` is the column for symbol j; each output span holds one pointer per
  // output column, every column n long.

  [[nodiscard]] result<void>
  evaluate(const rt::detail::CColumns<const T *> auto &xs,
           const rt::detail::CColumns<T *> auto &f, std::size_t n) const {
    return dispatch(Want::Value, as_columns(xs), {.values = as_columns(f)}, n);
  }

  [[nodiscard]] result<void>
  hessian(const rt::detail::CColumns<const T *> auto &xs,
          const rt::detail::CColumns<T *> auto &f,
          const rt::detail::CColumns<T *> auto &g,
          const rt::detail::CColumns<T *> auto &h, std::size_t n) const {
    return dispatch(Want::Hessian, as_columns(xs),
                    {.values = as_columns(f),
                     .jacobian = as_columns(g),
                     .hessian = as_columns(h)},
                    n);
  }

  // `v` is a second block of input columns, not a second point: one column per
  // symbol, each n long, spliced behind `xs`.  `g` comes out of the same sweep.
  [[nodiscard]] result<void> hvp(const rt::detail::CColumns<const T *> auto &xs,
                                 const rt::detail::CColumns<const T *> auto &v,
                                 const rt::detail::CColumns<T *> auto &f,
                                 const rt::detail::CColumns<T *> auto &g,
                                 const rt::detail::CColumns<T *> auto &hv,
                                 std::size_t n) const {
    return dispatch(Want::Hvp, as_columns(widen(xs, v)),
                    {.values = as_columns(f),
                     .jacobian = as_columns(g),
                     .hessian = as_columns(hv)},
                    n);
  }

  // `h` is n^2 columns, row-major: H(i, j) is column i n + j.
  [[nodiscard]] result<void>
  dense_hessian(const rt::detail::CColumns<const T *> auto &xs,
                const rt::detail::CColumns<T *> auto &f,
                const rt::detail::CColumns<T *> auto &g,
                const rt::detail::CColumns<T *> auto &h, std::size_t n) const {
    return dispatch(Want::DenseHessian, as_columns(xs),
                    {.values = as_columns(f),
                     .jacobian = as_columns(g),
                     .hessian = as_columns(h)},
                    n);
  }

  [[nodiscard]] result<void> vjp(const rt::detail::CColumns<const T *> auto &xs,
                                 const rt::detail::CColumns<const T *> auto &w,
                                 const rt::detail::CColumns<T *> auto &f,
                                 const rt::detail::CColumns<T *> auto &out,
                                 std::size_t n) const {
    return dispatch(Want::Vjp, as_columns(widen(xs, w)),
                    {.values = as_columns(f), .jacobian = as_columns(out)}, n);
  }

  [[nodiscard]] result<void> jvp(const rt::detail::CColumns<const T *> auto &xs,
                                 const rt::detail::CColumns<const T *> auto &v,
                                 const rt::detail::CColumns<T *> auto &f,
                                 const rt::detail::CColumns<T *> auto &out,
                                 std::size_t n) const {
    return dispatch(Want::Jvp, as_columns(widen(xs, v)),
                    {.values = as_columns(f), .jacobian = as_columns(out)}, n);
  }

  // What a caller sizes buffers by.  None builds a pipeline, and only the
  // Hessian's sweeps anything: its colouring is what fixes the count.
  [[nodiscard]] constexpr std::optional<std::size_t> value_columns() const {
    return outputs();
  }
  [[nodiscard]] constexpr std::optional<std::size_t> jacobian_columns() const {
    return unless_poisoned([this] { return jacobian_.partial.size(); });
  }
  [[nodiscard]] std::optional<std::size_t> hessian_columns() const {
    return unless_system(
        [this] { return hessians().front().compressed.size(); });
  }
  [[nodiscard]] std::optional<std::size_t> hessian_colours() const {
    return unless_system([this] { return hessians().front().colours(); });
  }
  [[nodiscard]] std::optional<std::size_t> hvp_columns() const {
    return unless_system([this] { return symbol_count(); });
  }
  [[nodiscard]] std::optional<std::size_t> dense_hessian_columns() const {
    return unless_system([this] { return symbol_count() * symbol_count(); });
  }
  [[nodiscard]] std::optional<std::size_t> vjp_columns() const {
    return arity();
  }
  [[nodiscard]] std::optional<std::size_t> jvp_columns() const {
    return outputs();
  }

  // What a batch caller reads the two compressed blocks by: the dense
  // spellings undo the compression, the batch ones cannot, and without these
  // the column counts describe a block nobody can index.

  // `at(i, j)` is nullopt where d f[i] / d x[j] is structurally zero.
  [[nodiscard]] constexpr std::optional<std::reference_wrapper<const Sparsity>>
  jacobian_pattern() const {
    return unless_poisoned([this] { return std::cref(jacobian_.pattern); });
  }

  // Which (row, column) cell each Hessian column holds: Colouring::entries().
  // One output only, and it sweeps the Hessian on first ask.
  [[nodiscard]] std::optional<std::reference_wrapper<const Colouring>>
  hessian_pattern() const {
    return unless_system(
        [this] { return std::cref(hessians().front().colouring); });
  }

  // Where H(i, j) sits, or nullopt for a cell the colouring calls zero.
  [[nodiscard]] std::optional<std::size_t> hessian_cell(std::size_t i,
                                                        std::size_t j) const {
    if (poisoned() || is_system() || i >= symbol_count() ||
        j >= symbol_count()) {
      return std::nullopt;
    }
    return hessians().front().colouring.cell_of(i, j);
  }

protected:
  using Surface::as_columns;
  using Surface::dense;
  using Surface::widen;
  using Compiled = detail::Compiled<T>;
  using Pipeline = detail::Pipeline<T>;
  using Pipelines = std::array<Pipeline, want_count>;

  // Poisoned: arena_ null, roots_ empty, no pipelines or sweeps, and every
  // accessor short-circuiting first.
  constexpr explicit Core(error why) : bad_(why) { settle(); }

  // The arena is the equation's own from here: one it was lent is copied, so
  // nothing another equation or the caller builds can move a node under it.
  constexpr Core(std::unique_ptr<Arena<T>> arena, std::vector<NodeId> roots)
      : arena_(std::move(arena)), roots_(std::move(roots)) {
    // An implicit no derivative could carry refuses the whole equation, as a
    // poisoned output would; so does a vector where an output must be a
    // scalar -- spell a system over its elements.
    if (const auto why = unclosed(*arena_)) {
      poison(error{*why});
    } else if (std::ranges::any_of(roots_, [this](NodeId r) {
                 return !detail::is_scalar((*arena_)[r]);
               })) {
      poison(error{errc::tensor_shape});
    } else {
      // One reverse sweep per output is cheap, and it is what the column
      // counts, the pattern and constant evaluation read.
      jacobian_ = derive_jacobian(*arena_, roots_);
    }
    settle();
  }

  // Refused: by the code it carries, with nothing left to read.  poisoned()
  // <=> arena_ == nullptr from here.
  constexpr void poison(error why) {
    bad_ = why;
    arena_.reset();
    roots_.clear();
  }
  // Every equation's tail: what the gates read of the arena, and the pipelines
  // and sweeps, which constant evaluation and a poisoned equation never reach.
  constexpr void settle() {
    if (!arena_) {
      return;
    }
    first_order_ = first_order_only(*arena_, roots_);
    if !consteval {
      pipelines_ = std::make_unique<Pipelines>();
      sweeps_ = std::make_unique<Sweeps>();
    }
  }

  // Poison first, and by the code it carries: a clashing name or an absent
  // arena is a different mistake from a literal that reached no graph.
  [[nodiscard]] static constexpr std::optional<error>
  why_not(std::span<const Expression<T>> outputs) noexcept {
    for (const Expression<T> &e : outputs) {
      if (const auto why = Rooting::why(e)) {
        return error{*why};
      }
    }
    if (arena_of(outputs) == nullptr) {
      return error{errc::no_graph};
    }
    return std::nullopt;
  }
  // The arena the outputs were built in: the first that names one, since a
  // literal output names none.
  [[nodiscard]] static constexpr Arena<T> *
  arena_of(std::span<const Expression<T>> outputs) noexcept {
    const auto named = std::ranges::find_if(
        outputs, [](const auto &e) { return Rooting::arena(e) != nullptr; });
    return named == outputs.end() ? nullptr : Rooting::arena(*named);
  }

  // The arena the outputs will be read in -- `owned`, or a copy of the one
  // they were built in -- and their roots there.
  [[nodiscard]] static constexpr std::pair<std::unique_ptr<Arena<T>>,
                                           std::vector<NodeId>>
  rooted(std::span<const Expression<T>> outputs,
         std::unique_ptr<Arena<T>> owned) {
    auto arena = owned ? std::move(owned)
                       : std::make_unique<Arena<T>>(*arena_of(outputs));
    auto roots = outputs | std::views::transform([&arena](const auto &e) {
                   return e.id(*arena);
                 }) |
                 impl::to<std::vector<NodeId>>();
    return {std::move(arena), std::move(roots)};
  }

  // The three ways an equation comes to be, for whichever class wraps this.
  template <typename D>
  [[nodiscard]] static constexpr D make(std::span<const Expression<T>> outputs,
                                        std::unique_ptr<Arena<T>> owned) {
    if (const auto why = why_not(outputs)) {
      return D{*why};
    }
    auto [arena, roots] = rooted(outputs, std::move(owned));
    return D{std::move(arena), std::move(roots)};
  }

  // Past the poison guard, so the arena is known good; arity() is the checked
  // spelling.
  [[nodiscard]] constexpr std::size_t symbol_count() const {
    return arena_->symbols().size();
  }
  [[nodiscard]] constexpr std::span<const NodeId> roots() const {
    return roots_;
  }
  [[nodiscard]] constexpr const Jacobian &derivative() const {
    return jacobian_;
  }

  // Which Jacobian cells a set of pipelines is frozen over: every one, for the
  // equation's own, or the cells a restrict() kept, as indices into them.
  struct Cells {
    const Jacobian &jacobian;
    // Unset is all of them; a restriction can keep none.
    std::optional<std::span<const std::size_t>> kept{};
  };
  [[nodiscard]] constexpr Cells every_cell() const {
    return {.jacobian = jacobian_};
  }

  // The pipeline for `want`, frozen and compiled on first ask.
  [[nodiscard]] std::shared_ptr<const Compiled> pipeline(Want want) const {
    return pipeline_in(*pipelines_, every_cell(), want);
  }
  [[nodiscard]] std::shared_ptr<const Compiled>
  pipeline_in(Pipelines &pipelines, const Cells &cells, Want want) const {
    return pipelines[std::to_underlying(want)].template snapshot<Yield>(
        setting(), [&] { return frozen_for(want, cells); });
  }

  [[nodiscard]] result<void> dispatch(Want want, std::span<const T *const> xs,
                                      const Columns<T> &out,
                                      std::size_t n) const {
    return dispatch_in(*pipelines_, every_cell(), want, xs, out, n);
  }
  // Through a restrict() view's own pipelines and cells, or the equation's.
  [[nodiscard]] result<void> dispatch_in(Pipelines &pipelines,
                                         const Cells &cells, Want want,
                                         std::span<const T *const> xs,
                                         const Columns<T> &out,
                                         std::size_t n) const {
    return checked(pipelines, cells, want, xs, out)
        .transform([&](const std::shared_ptr<const Compiled> &c) {
          run(*c, xs, out, n);
        });
  }

  // The pipeline a call runs on, once the call is known to fit it.
  [[nodiscard]] result<std::shared_ptr<const Compiled>>
  checked(Pipelines &pipelines, const Cells &cells, Want want,
          std::span<const T *const> xs, const Columns<T> &out) const {
    return ready(want).and_then(
        [&]() -> result<std::shared_ptr<const Compiled>> {
          auto c = pipeline_in(pipelines, cells, want);
          // Columns, not nodes: a vector output is one column per element.
          const bool fits =
              xs.size() == c->graph->arity() &&
              std::ranges::none_of(zip_blocks(out, c->graph->layout()),
                                   [](const auto &pair) {
                                     const auto &[columns, count] = pair;
                                     return columns.size() != count;
                                   });
          return fits ? result<std::shared_ptr<const Compiled>>{std::move(c)}
                      : fail(errc::wrong_column_count);
        });
  }

  // A system's Hessians, read off the arena a point at a time: a frozen graph
  // carries one colouring, so no pipeline holds m of them.  `f` is m long, `g`
  // the dense m x n and `h` the dense m x n x n.
  void walk(std::span<const T> at, std::span<T> f, std::span<T> g,
            std::span<T> h) const {
    const auto &blocks = hessians();
    // What the three blocks read -- values, partials and Hessian::at's
    // compressed cells -- settled on first ask, since it depends only on the
    // sweeps.  Appended block by block rather than through views::join, whose
    // inner iterator GCC 15 calls maybe-uninitialized under -Wall; reachable()
    // reads a set, so the order is free.
    const auto &walk = sweeps_->walk.get([&] {
      std::vector<NodeId> wanted{roots_.begin(), roots_.end()};
      std::ranges::for_each(blocks, [&wanted](const Hessian &b) {
        impl::append(wanted, b.compressed);
        impl::append(wanted, b.partial);
        wanted.push_back(b.zero);
      });
      [[maybe_unused]] const auto held = reading();
      return ArenaWalk{*arena_, live_ids(reachable(*arena_, wanted))};
    });
    const std::size_t n = symbol_count();
    // The tape is thread-kept; every id read below is scheduled, so it is
    // written this call.
    ThreadTapes<T> &kept = thread_tapes<T>();
    std::vector<T> &values = kept.values;
    [[maybe_unused]] const auto held = sweeping();
    values.resize(arena_->size());
    kept.value_vtape.resize(walk.extent());
    walk(*arena_, at, std::span<T>{values}, std::span<T>{kept.value_vtape},
         tensor_kernels());
    const impl::md::mdspan dense{
        h.data(), impl::md::dextents<std::size_t, 3>{roots_.size(), n, n}};
    const auto dims = std::views::iota(0uz, n);
    for (const auto [k, block] : std::views::enumerate(blocks)) {
      const auto root = static_cast<std::size_t>(k);
      f[root] = values[roots_[root]];
      for (const std::size_t i : dims) {
        g[root * n + i] = values[block.partial[i]];
      }
      for (const auto [i, j] : std::views::cartesian_product(dims, dims)) {
        dense[root, i, j] = values[block.at(i, j)];
      }
    }
  }

  // nullopt on a poisoned equation, `value()` otherwise.
  [[nodiscard]] constexpr auto unless_poisoned(auto &&value) const {
    using V = std::remove_cvref_t<std::invoke_result_t<decltype(value)>>;
    return poisoned() ? std::optional<V>{} : std::optional<V>{value()};
  }

  // As unless_poisoned, and nullopt too for a system: the answer is about one
  // output's Hessian.
  [[nodiscard]] auto unless_system(auto &&value) const {
    using V = std::remove_cvref_t<std::invoke_result_t<decltype(value)>>;
    return poisoned() || is_system() ? std::optional<V>{}
                                     : std::optional<V>{value()};
  }

  // m long, always a vector: what System answers and Equation narrows.
  [[nodiscard]] constexpr result<std::vector<T>>
  values(const rt::detail::CPointArg<T> auto &...args) const {
    return at<Want::Value>({}, args...).transform([](auto b) {
      return std::move(b.values);
    });
  }

private:
  // Run `build` once, under the arena's write lock: every sweep appends.
  template <typename V> const V &sweep(Lazy<V> &slot, auto &&build) const {
    return slot.get([&] {
      [[maybe_unused]] const auto held = writing();
      return build();
    });
  }

  [[nodiscard]] const std::vector<Hessian> &hessians() const {
    return sweep(sweeps_->hessians, [this] {
      return roots_ | std::views::transform([this](NodeId r) {
               return derive_hessian(*arena_, r);
             }) |
             impl::to<std::vector<Hessian>>();
    });
  }
  // Root 0's only: a system refuses hvp before it gets here.
  [[nodiscard]] const HessianVector &hvp_sweep() const {
    return sweep(sweeps_->hvp,
                 [this] { return derive_hvp(*arena_, roots_.front()); });
  }
  [[nodiscard]] const DenseHessian &dense_sweep() const {
    return sweep(sweeps_->dense, [this] {
      return derive_dense_hessian(*arena_, roots_.front());
    });
  }
  [[nodiscard]] const VectorJacobian &vjp_sweep() const {
    return sweep(sweeps_->vjp, [this] { return derive_vjp(*arena_, roots_); });
  }
  [[nodiscard]] const Tangent &jvp_sweep() const {
    return sweep(sweeps_->jvp, [this] { return derive_jvp(*arena_, roots_); });
  }
  [[nodiscard]] const std::vector<NodeId> &tangent_sweep() const {
    return sweep(sweeps_->tangent, [this] {
      return derive_jacobian_tangent(*arena_, jacobian_);
    });
  }

  // The symbol named `name`, by slot: a parameter or an implicit is an input,
  // not a symbol.
  [[nodiscard]] result<std::uint32_t> symbol(std::string_view name) const {
    const auto input = arena_->lookup(name);
    return input && input->op == OpCode::Var
               ? result<std::uint32_t>{input->index}
               : fail(errc::unknown_symbol);
  }
  [[nodiscard]] constexpr const Sparsity &jacobian_sparsity() const {
    return jacobian_.pattern;
  }
  // e_j for a symbol named `along`, or unknown_symbol.
  [[nodiscard]] result<boost::container::small_vector<T, 64>>
  unit(std::string_view along) const {
    return alive()
        .and_then([&] { return symbol(along); })
        .transform([this](std::uint32_t slot) {
          boost::container::small_vector<T, 64> e(symbol_count(), T{0});
          e[slot] = T{1};
          return e;
        });
  }

protected:
  // The gates, in the order a call meets them.  alive(): not poisoned.
  // single(): one function's, which a system refuses.  orders(): second order
  // is refused where a reachable extern supplies first partials only, rather
  // than answered with the zeros a missing component would read.  blocked():
  // the dense Hessian has no block where a matrix the symbols move is
  // multiplied (rt/dense.hpp).  ready() is the one gate every batch and point
  // call passes, off the want's row.
  [[nodiscard]] constexpr bool is_system() const noexcept {
    return roots_.size() != 1;
  }
  [[nodiscard]] constexpr result<void> alive() const {
    return bad_ ? result<void>{std::unexpected{*bad_}} : result<void>{};
  }
  [[nodiscard]] constexpr result<void> single() const {
    return alive().and_then([this] {
      return is_system() ? fail(errc::not_single_output) : result<void>{};
    });
  }
  [[nodiscard]] constexpr result<void> orders(Want want) const {
    return want_info(want).second_order && first_order_
               ? fail(errc::extern_order)
               : result<void>{};
  }
  [[nodiscard]] constexpr result<void> blocked(Want want) const {
    return want == Want::DenseHessian && dense_sweep().block == no_node
               ? fail(errc::tensor_shape)
               : result<void>{};
  }
  [[nodiscard]] constexpr result<void> ready(Want want) const {
    return (want_info(want).single ? single() : alive())
        .and_then([&] { return orders(want); })
        .and_then([&] { return blocked(want); });
  }

private:
  [[nodiscard]] auto reading() const {
    return std::shared_lock{sweeps_->arena_mutex};
  }
  [[nodiscard]] auto writing() const {
    return std::unique_lock{sweeps_->arena_mutex};
  }
  // What a sweep holds and gives up: Yield first, so the lock is never waited
  // for -- or held -- while reacquiring what Yield let go.
  struct Sweeping {
    [[no_unique_address]] Yield yielded{};
    std::shared_lock<std::shared_mutex> held;
  };
  [[nodiscard]] Sweeping sweeping() const {
    return Sweeping{.held = reading()};
  }

  // The three blocks' column counts at one point of `want`: the table's shape
  // columns in this equation's numbers, the Jacobian cells being the view's
  // where a restrict() kept some.
  [[nodiscard]] constexpr Layout sizes(Want want, const Cells &cells) const {
    const std::size_t m = roots_.size();
    const std::size_t n = symbol_count();
    const auto count = [&](BlockShape shape) -> std::size_t {
      switch (shape) {
      case BlockShape::None:
        return 0;
      case BlockShape::Cells:
        return cells.kept ? cells.kept->size() : cells.jacobian.partial.size();
      case BlockShape::Symbols:
        return n;
      case BlockShape::Functions:
        return m;
      case BlockShape::Colours:
        return hessians().front().compressed.size();
      case BlockShape::Dense:
        return n * n;
      }
      std::unreachable();
    };
    const WantInfo &w = want_info(want);
    return {.values = w.values ? m : 0,
            .jacobian = count(w.jacobian),
            .hessian = count(w.hessian)};
  }

  // One point of `W`: the blocks sized by the table, and where the pipeline
  // takes a direction -- one value per symbol or per function, as the table
  // says -- the point widened by it.  `pipelines` and `cells` are the
  // equation's or a restrict() view's; constant evaluation takes gather()'s
  // arena walk.
  template <Want W>
  [[nodiscard]] constexpr result<PerBlock<std::vector<T>>>
  at_in(Pipelines *pipelines, const Cells &cells, std::span<const T> along,
        const rt::detail::CPointArg<T> auto &...args) const {
    return point(args...).and_then(
        [&](const std::vector<T> &pt) -> result<PerBlock<std::vector<T>>> {
          constexpr Direction direction = want_info(W).direction;
          if constexpr (direction != Direction::None) {
            if (along.size() != (direction == Direction::Functions
                                     ? roots_.size()
                                     : symbol_count())) {
              return fail(errc::wrong_direction);
            }
          }
          const Layout size = sizes(W, cells);
          PerBlock<std::vector<T>> out{.values = std::vector<T>(size.values),
                                       .jacobian =
                                           std::vector<T>(size.jacobian),
                                       .hessian = std::vector<T>(size.hessian)};
          const auto swept = [&] {
            if constexpr (direction == Direction::None) {
              return gather(pipelines, cells, W, pt, out.values, out.jacobian,
                            out.hessian);
            } else {
              return gather(pipelines, cells, W, widen(pt, along), out.values,
                            out.jacobian, out.hessian);
            }
          }();
          return swept.transform([&] { return std::move(out); });
        });
  }
  template <Want W>
  [[nodiscard]] constexpr result<PerBlock<std::vector<T>>>
  at(std::span<const T> along,
     const rt::detail::CPointArg<T> auto &...args) const {
    return at_in<W>(pipelines_.get(), every_cell(), along, args...);
  }

  // Naming a point is not the same as giving one: a symbol no argument reaches
  // would otherwise be read as zero, which is a value and not a refusal.
  [[nodiscard]] constexpr result<void>
  whole(const std::vector<bool> &seen) const {
    return std::ranges::contains(seen, false) ? fail(errc::short_point)
                                              : result<void>{};
  }

  [[nodiscard]] constexpr result<void> assign_one(std::vector<T> &at,
                                                  std::vector<bool> &seen,
                                                  std::string_view name,
                                                  const auto &value) const {
    const auto input = arena_->lookup(name);
    if (!input) {
      return fail(errc::unknown_symbol);
    }
    const auto column = input_column(arena_->inputs(), input->op, input->index);
    at[column] = static_cast<T>(value);
    seen[column] = true;
    return {};
  }

  template <impl::CEntry V>
  [[nodiscard]] constexpr result<void>
  assign_named(std::vector<T> &at, std::vector<bool> &seen, const V &nv) const {
    return assign_one(at, seen, std::remove_cvref_t<V>::symbol.view(),
                      nv.value);
  }

  [[nodiscard]] constexpr result<void>
  assign_named_range(std::vector<T> &at,
                     const rt::detail::CNamedRange<T> auto &vm) const {
    std::vector<bool> seen(at.size(), false);
    for (const auto &entry : vm) {
      if (const auto ok =
              assign_one(at, seen, std::string_view{entry.first}, entry.second);
          !ok) {
        return ok;
      }
    }
    return whole(seen);
  }

  [[nodiscard]] constexpr result<void>
  assign_range(std::vector<T> &at,
               const std::ranges::input_range auto &r) const {
    if (std::ranges::size(r) != at.size()) {
      return fail(errc::wrong_arity);
    }
    std::ranges::transform(r, at.begin(),
                           [](const auto &v) { return static_cast<T>(v); });
    return {};
  }

  // One point through the batch path at n = 1, not the arena walk, so the
  // batch gate is the point gate.  Constant evaluation keeps it, having no
  // pipeline and nothing to refuse.
  //
  // The arena walk is the tail, not an `else` on the `if !consteval`: that
  // branch would be an immediate function context, and what GCC then escalates
  // under P2564 is not what either path says.  MSVC sees the tail as dead code
  // instead -- it is, at run time -- hence the pragma, which has to sit outside
  // the body: a code-generation warning takes the state at the opening brace.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702) // unreachable code
#endif
  [[nodiscard]] constexpr result<void> gather(Pipelines *pipelines,
                                              const Cells &cells, Want want,
                                              std::span<const T> at,
                                              std::span<T> f, std::span<T> g,
                                              std::span<T> h) const {
    if !consteval {
      // At one point every column is one value, its address.
      const auto address = [](auto &v) { return std::addressof(v); };
      const auto xs = at | std::views::transform(address) |
                      impl::to<boost::container::small_vector<const T *, 32>>();
      std::vector<T *> &out = thread_tapes<T>().addresses;
      out.clear();
      impl::append(out, std::array{f, g, h} | std::views::join |
                            std::views::transform(address));
      const std::span<T *const> all{out};
      return dispatch_in(*pipelines, cells, want, as_columns(xs),
                         {.values = all.first(f.size()),
                          .jacobian = all.subspan(f.size(), g.size()),
                          .hessian = all.last(h.size())},
                         1);
    }
    auto wanted = roots_;
    impl::append(wanted, jacobian_.partial);
    const auto values =
        evaluate_reachable(*arena_, wanted, at, tensor_kernels());
    const auto pick = [&values](std::span<T> out, const auto &nodes) {
      std::ranges::transform(nodes | std::views::take(out.size()), out.begin(),
                             [&values](NodeId v) { return values[v]; });
    };
    pick(f, roots_);
    pick(g, jacobian_.partial);
    return {};
  }
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  // What a pipeline needs of the equation.  No compiler is a pipeline that will
  // never compile: an interpreting backend arrives here as one, and so does a
  // host with no JIT.
  [[nodiscard]] Setting setting() const {
    Setting out;
    if constexpr (jit::CKernelScalar<T>) {
      out.options = effective_options();
#ifdef DDX_HAS_JIT
      if (options_.backend == jit::Backend::Compile) {
        out.compiler = compiler();
      }
#endif
    }
    return out;
  }

  // The graphs a pipeline computes.
  [[nodiscard]] Compiled frozen_for(Want want, const Cells &cells) const {
    // What the want adds past the values and the Jacobian, swept before the
    // freeze takes the lock.  Vjp and Jvp replace the Jacobian block rather
    // than add to it -- n columns instead of the pattern's nonzeros is the
    // whole point of asking.  A restriction keeps the cells it chose, of dJ/dv
    // too.
    OutputBlocks added;
    Colouring colouring;
    switch (want) {
    case Want::Vjp:
      added.jacobian = vjp_sweep().product;
      break;
    case Want::Jvp:
      added.jacobian = jvp_sweep().product;
      break;
    case Want::Hessian:
      added.hessian = hessians().front().compressed;
      colouring = hessians().front().colouring;
      break;
    case Want::Hvp:
      added.hessian = hvp_sweep().product;
      break;
    case Want::JacobianJvp: {
      const auto &tangent = tangent_sweep();
      added.hessian = cells.kept
                          ? *cells.kept | std::views::transform([&](auto k) {
                              return tangent[k];
                            }) | impl::to<std::vector<NodeId>>()
                          : tangent;
      break;
    }
    case Want::DenseHessian:
      added.hessian = {dense_sweep().block};
      break;
    case Want::Value:
    case Want::Jacobian:
    case Want::Gradient:
      break;
    }
    // `roots` and `partials` differ between the swept graph and the kernel's
    // rebalanced one.  Gradient differentiates the roots without computing
    // them; every direction column the call hands counts, whether or not the
    // product reads it: y's never reaches H(x*x + y) v.
    const WantInfo &info = want_info(want);
    const auto freeze = [&](std::span<const NodeId> roots,
                            const Jacobian &partials) {
      OutputBlocks blocks = added;
      if (info.values) {
        blocks.values.assign(roots.begin(), roots.end());
      }
      const bool pattern = info.jacobian == BlockShape::Cells;
      if (pattern) {
        blocks.jacobian = partials.partial;
      }
      const std::size_t directions =
          info.direction == Direction::Symbols     ? symbol_count()
          : info.direction == Direction::Functions ? roots.size()
                                                   : 0;
      return std::make_shared<const Graph<T>>(Graph<T>::freeze(
          *arena_, std::move(blocks), colouring,
          pattern ? partials.pattern : Sparsity{}, contracts(), directions));
    };
    [[maybe_unused]] const bool rebalance = !cells.kept && rebalances(want);
    [[maybe_unused]] const auto held = reading();
    auto swept = freeze(roots_, cells.jacobian);
    auto compiled = swept;
#ifdef DDX_HAS_JIT
    if (rebalance) {
      const Rebalanced &over = sweeps_->rebalanced.peek();
      compiled = freeze(over.roots, over.derivative);
    }
#endif
    return {.graph = std::move(swept),
            .compile_graph = std::move(compiled),
            .kernels = &tensor_kernels()};
  }

  // Whether the kernel for `want` is lowered from rebalanced roots, which are
  // swept here on first ask.
  [[nodiscard]] bool rebalances([[maybe_unused]] Want want) const {
#ifdef DDX_HAS_JIT
    if constexpr (jit::CKernelScalar<T>) {
      // A tensor step is the sweep's own, so its graph is never rebalanced.
      // The roots' cone, not the arena: a dense Hessian swept earlier leaves
      // tensor nodes no scalar root reads.  Read under the lock, as a sweep on
      // another thread may be appending.
      const auto tensors = [this] {
        [[maybe_unused]] const auto held = reading();
        const auto live = detail::reachable(*arena_, roots_);
        return std::ranges::any_of(detail::live_ids(live), [this](NodeId v) {
          return detail::on_tensors((*arena_)[v]);
        });
      };
      if (!rebalanceable(want) || options_.backend != jit::Backend::Compile ||
          setting().compiler == nullptr || tensors()) {
        return false;
      }
      (void)sweep(sweeps_->rebalanced, [this] {
        Rebalanced over{.roots = detail::rebalance(*arena_, roots_),
                        .derivative = {}};
        over.derivative = derive_jacobian(*arena_, over.roots);
        return over;
      });
      return true;
    }
#endif
    return false;
  }

  // How wide to emit, from the batch the caller stated: a wide kernel computes
  // `w` points to answer for one, so a short batch emits scalar -- the same
  // threshold the sweep uses.  A stated `lanes` is honoured.
  [[nodiscard]] jit::Options effective_options() const noexcept {
    return jit::for_batch(options_, block_lanes);
  }

#ifdef DDX_HAS_JIT
  // A Kernel does not own its code, so the compiler outlives every kernel.
  // Null on a host with no JIT.
  [[nodiscard]] static jit::Compiler *compiler() {
    return rt::detail::shared_compiler();
  }
#endif

  // Decided in the *graph*, so every backend folds the same products.  On by
  // default in a JIT-less build as well -- DDX_JIT_DEFAULT_CONTRACT is the
  // JIT's -- so an answer does not depend on how the library was configured.
  [[nodiscard]] bool contracts() const noexcept {
    return options_.codegen.contract;
  }

  // The kernel where one compiled, else the sweep.
  void run(const Compiled &c, std::span<const T *const> xs,
           const Columns<T> &out, std::size_t n) const {
    [[maybe_unused]] const auto held = sweeping();
    if constexpr (jit::CKernelScalar<T>) {
      if (c.kernel) {
        const detail::TensorCall<T> call{.arena = arena_.get(),
                                         .graph = c.compile_graph.get(),
                                         .kernels = c.kernels};
        std::vector<T> &tapes = thread_tapes<T>().kernel_tapes;
        tapes.resize(c.kernel.shape().tapes);
        c.kernel(xs, out.values, out.jacobian, out.hessian, n,
                 {.call = &call, .tapes = tapes});
        return;
      }
    }
    interpret(*c.graph, *c.kernels, xs, out, n);
  }

  // The block sweep, within ~1.4x of a kernel and not a sad path.
  void interpret(const Graph<T> &graph, const TensorKernelTable<T> &kernels,
                 std::span<const T *const> xs, const Columns<T> &out,
                 std::size_t n) const {
    const auto blocks = graph.output_rows();
    const auto schedule = graph.schedule();
    // Both tapes are sized by the graph, whose rows are packed by liveness:
    // the schedule names rows, so a grown arena changes nothing here.
    const TensorLayout &rows = graph.tensors();
    const auto tensors = [&](std::vector<T> &vtape, std::size_t lanes) {
      vtape.resize(rows.extent * lanes);
      return detail::TapeView<T>{.scalar_rows = graph.rows(),
                                 .tensor_rows = rows.rows,
                                 .vtape = vtape.data(),
                                 .kernels = &kernels};
    };
    // The graph's width, not the arena's symbol count: `xs` carries one column
    // per input, and a pipeline may hold leaves that are not symbols.
    const std::size_t inputs = graph.arity();

    // Thread-kept (ThreadTapes).  resize() rather than assign(): every id
    // read is written first, so zeroing the arena per call is waste.
    //
    // A short batch sweeps one point at a time rather than padding out to
    // block_lanes, which on a batch of one is the whole cost.
    ThreadTapes<T> &kept = thread_tapes<T>();
    if (n < block_lanes) {
      const auto [at, tape, vtape] = std::tie(kept.at, kept.tape, kept.vtape);
      at.resize(inputs);
      tape.resize(graph.tape_rows());
      const auto t = tensors(vtape, 1);
      for (const std::size_t i : std::views::iota(0uz, n)) {
        std::ranges::transform(xs, at.begin(),
                               [i](const T *column) { return column[i]; });
        evaluate_into(*arena_, at, schedule, std::span<T>{tape}, t);
        scatter_blocks(blocks, std::span<const T>{tape},
                       std::span<const T>{vtape}, out, i, 1, 1);
      }
      return;
    }

    // block_lanes points per sweep: the switch is paid once per node per
    // block, and each operation becomes a lane loop wide enough to vectorise.
    // A short final block repeats its last point, and those lanes are never
    // read back.
    const auto [lanes, tape, vtape] =
        std::tie(kept.lanes, kept.lane_tape, kept.lane_vtape);
    lanes.resize(inputs * block_lanes);
    tape.resize(graph.tape_rows() * block_lanes);
    const auto t = tensors(vtape, block_lanes);

    for (const std::size_t base :
         std::views::iota(0uz, n) | std::views::stride(block_lanes)) {
      const std::size_t width = std::min(block_lanes, n - base);
      for (const auto [dst, column] :
           std::views::zip(lanes | std::views::chunk(block_lanes), xs)) {
        const auto tail =
            std::ranges::copy(std::span{column + base, width}, dst.begin()).out;
        std::ranges::fill(tail, dst.end(), column[base + width - 1]);
      }
      evaluate_block<block_lanes>(*arena_, std::span<const T>{lanes}, schedule,
                                  std::span<T>{tape}, t);
      scatter_blocks(blocks, std::span<const T>{tape},
                     std::span<const T>{vtape}, out, base, block_lanes, width);
    }
  }

  std::unique_ptr<Arena<T>> arena_;
  std::vector<NodeId> roots_;
  jit::Options options_{};
  // Null: the process default, read at each freeze.
  const TensorKernelTable<T> *kernels_ = nullptr;
  Jacobian jacobian_;
  // unique_ptr, not shared: its destructor is constexpr, which is what keeps a
  // constant-evaluated equation destructible.  Null there; a poisoned equation
  // reads neither, and settle() leaves it without both.
  std::unique_ptr<Pipelines> pipelines_;
  // Every other sweep, built once on first ask under the arena's write lock.
  std::unique_ptr<Sweeps> sweeps_;
  // Set exactly when why_not() refused, so poisoned() <=> arena_ == nullptr.
  std::optional<error> bad_{};
  // A reachable extern supplies first partials only; see orders().
  bool first_order_ = false;
};

// restrict()'s answer: the equation's Jacobian narrowed to some symbols'
// columns, in symbol order, with pipelines of its own -- frozen over the kept
// cells alone, so the partials of the others are never computed.  It reads the
// equation it came from and must not outlive it.
template <typename C>
class DDX_TEMPLATE_API Restricted
    : public JacobianSurface<Restricted<C>, typename C::value_type> {
  using T = typename C::value_type;
  using Want = rt::Want;
  friend JacobianSurface<Restricted, T>;

public:
  Restricted(const C &core, const std::vector<bool> &chosen)
      : core_(&core), pipelines_(std::make_unique<typename C::Pipelines>()) {
    const Jacobian &full = core.jacobian_;
    std::vector<std::uint32_t> column(chosen.size());
    std::uint32_t next = 0;
    for (const auto [j, keep] : chosen | std::views::enumerate) {
      column[static_cast<std::size_t>(j)] = next;
      if (keep) {
        names_.push_back(core.arena_->symbols()[static_cast<std::size_t>(j)]);
        ++next;
      }
    }
    jacobian_ = {.value = full.value,
                 .partial = {},
                 .pattern = {.rowptr = {0}, .col = {}, .columns = next},
                 .zero = full.zero};
    for (const Sparsity::Row &row : full.pattern) {
      for (const auto [i, j] : row.cols | std::views::enumerate) {
        if (chosen[j]) {
          const std::size_t k = row.offset + static_cast<std::size_t>(i);
          kept_.push_back(k);
          jacobian_.partial.push_back(full.partial[k]);
          jacobian_.pattern.col.push_back(column[j]);
        }
      }
      jacobian_.pattern.rowptr.push_back(jacobian_.pattern.col.size());
    }
  }

  // The symbols kept, in the order their columns are.
  [[nodiscard]] std::span<const std::string> symbols() const { return names_; }
  [[nodiscard]] std::size_t jacobian_columns() const { return kept_.size(); }
  [[nodiscard]] const Sparsity &jacobian_pattern() const {
    return jacobian_.pattern;
  }

private:
  [[nodiscard]] typename C::Cells cells() const {
    return {.jacobian = jacobian_, .kept = std::span{kept_}};
  }
  // `direction` is still one value per symbol of the equation.
  template <Want W>
  [[nodiscard]] result<PerBlock<std::vector<T>>>
  at(std::span<const T> along,
     const rt::detail::CPointArg<T> auto &...args) const {
    return core_->template at_in<W>(pipelines_.get(), cells(), along, args...);
  }
  [[nodiscard]] auto unit(std::string_view along) const {
    return core_->unit(along);
  }
  [[nodiscard]] const Sparsity &jacobian_sparsity() const {
    return jacobian_.pattern;
  }
  [[nodiscard]] result<void> dispatch(Want want, std::span<const T *const> xs,
                                      const Columns<T> &out,
                                      std::size_t n) const {
    return core_->dispatch_in(*pipelines_, cells(), want, xs, out, n);
  }

  const C *core_;
  std::vector<std::string> names_;
  Jacobian jacobian_;
  std::vector<std::size_t> kept_; // indices into the equation's cells
  std::unique_ptr<typename C::Pipelines> pipelines_;
};

// How many functions an equation has, as its type says: a count fixed there
// narrows one function's value to a scalar; a run-time count answers a
// vector.
template <std::size_t N> struct Fixed {
  static constexpr std::optional<std::size_t> expected = N;
  static constexpr bool scalar = N == 1;
};
struct Dynamic {
  static constexpr std::optional<std::size_t> expected = std::nullopt;
  static constexpr bool scalar = false;
};

// Run `assemble` inside a fresh arena, and hand both back.
template <impl::Numeric T>
[[nodiscard]] auto assemble(std::invocable auto &&assemble) {
  auto arena = std::make_unique<Arena<T>>();
  auto built = [&] {
    const auto scope = scoped_arena(*arena);
    return std::invoke(DDX_FWD(assemble));
  }();
  return std::pair{std::move(arena), std::move(built)};
}

} // namespace ddx::rt::detail

namespace ddx::rt {
// What Equation::restrict() and System::restrict() answer.
template <typename C> using Restricted = detail::Restricted<C>;

// Free this thread's sweep tapes: a pool worker that swept something large
// keeps its tapes at that size until it asks.  The next sweep on the thread
// grows them again.
template <impl::Numeric T = double> void release_tapes() noexcept {
  detail::thread_tapes<T>() = {};
}
} // namespace ddx::rt

// The scalars libddx compiles the engine for once (src/rt/instances.cpp); a TU
// here only instantiates what it inlines.  Any other T instantiates as before.
// thread_tapes() is on the list for its thread_local: one set of tapes per
// thread, not one per binary.  Arena and Graph declare theirs beside the class
// (DDX_EXTERN_INSTANCE).  The visibility is DDX_TEMPLATE_API's on GCC and
// Clang and DDX_INSTANCE_API's on MSVC, which refuses dllexport on an extern
// declaration (C4910): libddx's own TUs skip these there, and instances.cpp
// exports at the definition.  C4251 is the std:: members of an exported class:
// both sides of the boundary share one STL.
#define DDX_RT_INSTANCES(EXTERN, API, T)                                       \
  EXTERN template class API ::ddx::rt::detail::Pipeline<T>;                    \
  EXTERN template class API ::ddx::rt::detail::JacobianSurface<                \
      ::ddx::rt::detail::Core<T>, T>;                                          \
  EXTERN template class API ::ddx::rt::detail::Core<T>;                        \
  EXTERN template class API ::ddx::rt::detail::Restricted<                     \
      ::ddx::rt::detail::Core<T>>;                                             \
  EXTERN template API ::ddx::rt::detail::ThreadTapes<T>                        \
      & ::ddx::rt::detail::thread_tapes<T>();

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
#if !(defined(_MSC_VER) && defined(DDX_BUILDING))
DDX_RT_INSTANCES(extern, DDX_INSTANCE_API, double)
DDX_RT_INSTANCES(extern, DDX_INSTANCE_API, float)
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif
