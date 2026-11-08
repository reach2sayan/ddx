#pragma once

#include "dual/taylor_dual.hpp"
#include "md/md.hpp"
#include "ops/numeric.hpp"
#include "rt/coupling.hpp"
#include "rt/dense.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "rt/pipeline.hpp"
#include "rt/rebalance.hpp"
#include "symbolic/entry.hpp"
#include "util/config.hpp"
#include "util/export.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>

#include "jit/options.hpp"

#include <mutex>
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
#include <utility>
#include <vector>

namespace ddx::rt::detail {

// A point keyed by run-time names: any range of (name, value) pairs. Entry
// keys off a FixedString, so it cannot carry a name learned at run time.
template <typename R>
concept CNamedRange =
    std::ranges::input_range<R> &&
    requires(const std::ranges::range_value_t<R> &e) {
      { e.first } -> std::convertible_to<std::string_view>;
      requires impl::Numeric<std::remove_cvref_t<decltype(e.second)>>;
    };

template <typename A>
concept CPointArg =
    impl::Numeric<std::remove_cvref_t<A>> || impl::CEntry<A> ||
    CNamedRange<A> ||
    (std::ranges::input_range<A> &&
     impl::Numeric<std::remove_cvref_t<std::ranges::range_value_t<A>>>);

// CPointArg admits any input_range, so without this concept a span of columns
// would be read as a point.
template <typename R, typename Ptr>
concept CColumns = std::ranges::contiguous_range<R> &&
                   std::convertible_to<std::ranges::range_value_t<R>, Ptr>;

// The default Yield, which releases nothing. Exported so that Core<T>
// instantiations are visible too.
struct DDX_API Nothing {};

template <typename C> class DDX_TEMPLATE_API Restricted;

// The Jacobian spellings an equation and its restrict() view share. D supplies
// at<W>(), dispatch(), unit() and jacobian_sparsity().
template <typename D, impl::Numeric T> class DDX_TEMPLATE_API JacobianSurface {
  using Want = rt::Want;
  using Blocks = PerBlock<std::vector<T>>;

public:
  // Dense row-major m x n, or n long when m == 1. The graph stores only the
  // pattern's cells, so the zeros are put back here.
  [[nodiscard]] constexpr result<std::vector<T>>
  jacobian(const rt::detail::CPointArg auto &...args) const {
    return dense_at<Want::Jacobian, &Blocks::jacobian>({}, args...);
  }

  // The Jacobian alone, from a graph with no value block, so work that only the
  // value needs is never swept.
  [[nodiscard]] constexpr result<std::vector<T>>
  gradient(const rt::detail::CPointArg auto &...args) const {
    return dense_at<Want::Gradient, &Blocks::jacobian>({}, args...);
  }

  // dJ/dv as dense row-major m x n: every Jacobian cell differentiated along
  // `direction`, in one forward sweep.
  [[nodiscard]] result<std::vector<T>>
  jacobian_jvp(std::span<const T> direction,
               const rt::detail::CPointArg auto &...args) const {
    return dense_at<Want::JacobianJvp, &Blocks::hessian>(direction, args...);
  }
  // The whole Jacobian differentiated along one symbol's basis vector.
  [[nodiscard]] result<std::vector<T>>
  jacobian_jvp(std::string_view along,
               const rt::detail::CPointArg auto &...args) const {
    return self().unit(along).and_then([&](const auto &e) {
      return jacobian_jvp(std::span<const T>{e}, args...);
    });
  }

  // --- batch ---------------------------------------------------------------
  // The `g` and `dg` blocks have jacobian_columns() columns, laid out by
  // jacobian_pattern().
  [[nodiscard]] result<void>
  jacobian(const rt::detail::CColumns<const T *> auto &xs,
           const rt::detail::CColumns<T *> auto &f,
           const rt::detail::CColumns<T *> auto &g, std::size_t n) const {
    return batch<Want::Jacobian>(xs, n, f, g);
  }

  // Runs its own pipeline rather than jacobian()'s, because on a coupled model
  // the gradient is most of the graph.
  [[nodiscard]] result<void>
  gradient(const rt::detail::CColumns<const T *> auto &xs,
           const rt::detail::CColumns<T *> auto &g, std::size_t n) const {
    return batch<Want::Gradient>(xs, n, g);
  }

  // The Jacobian, and its derivative along `v`.
  [[nodiscard]] result<void>
  jacobian_jvp(const rt::detail::CColumns<const T *> auto &xs,
               const rt::detail::CColumns<const T *> auto &v,
               const rt::detail::CColumns<T *> auto &f,
               const rt::detail::CColumns<T *> auto &g,
               const rt::detail::CColumns<T *> auto &dg, std::size_t n) const {
    return batch<Want::JacobianJvp>(widen(xs, v), n, f, g, dg);
  }

protected:
  // A batch call for W, with `outs` given in the want table's block order.
  template <Want W>
  [[nodiscard]] result<void> batch(const auto &xs, std::size_t n,
                                   const auto &...outs) const {
    constexpr WantInfo w = want_info(W);
    const std::array<std::span<T *const>, sizeof...(outs)> given{
        as_columns(outs)...};
    auto next = given.begin();
    Columns<T> out;
    if constexpr (w.values) {
      out.values = *next++;
    }
    if constexpr (w.jacobian != BlockShape::None) {
      out.jacobian = *next++;
    }
    if constexpr (w.hessian != BlockShape::None) {
      out.hessian = *next++;
    }
    return self().dispatch(W, as_columns(xs), out, n);
  }

  [[nodiscard]] static auto
  as_columns(const std::ranges::contiguous_range auto &r) {
    using Ptr = std::ranges::range_value_t<std::remove_cvref_t<decltype(r)>>;
    return std::span<Ptr const>{std::ranges::data(r), std::ranges::size(r)};
  }

  // The point followed by the direction in one array, as values or columns.
  // Named inputs come first, as the ABI and input_column() expect.
  [[nodiscard]] static auto widen(const auto &xs, const auto &seeds) {
    boost::container::small_vector<std::ranges::range_value_t<decltype(xs)>, 64>
        out;
    out.reserve(std::ranges::size(xs) + std::ranges::size(seeds));
    std::ranges::copy(xs, std::back_inserter(out));
    std::ranges::copy(seeds, std::back_inserter(out));
    return out;
  }

  // Row-major rows x columns from the pattern's cells, with the zeros put back.
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
  // One block of W's answer, with the zeros put back.
  template <Want W, auto Block>
  [[nodiscard]] constexpr result<std::vector<T>>
  dense_at(std::span<const T> along, const auto &...args) const {
    return self()
        .template at<W>(along, args...)
        .transform([this](const auto &b) {
          return dense(self().jacobian_sparsity(), b.*Block);
        });
  }
};

// A sweep built on first ask and then kept, with its once_flag beside it.
template <typename V> class Lazy {
public:
  const V &get(auto &&build) {
    std::call_once(flag_, [&] { value_ = build(); });
    return value_;
  }

private:
  std::once_flag flag_;
  V value_;
};

// The roots and Jacobian a kernel is lowered from; see rt/rebalance.hpp.
struct Rebalanced {
  std::vector<NodeId> roots;
  Jacobian derivative;
};

// The arena's lock and every lazily built sweep, kept on the heap because a
// mutex cannot move and constant evaluation never needs them.
// Lock order: a pipeline's lock, then the arena's. Nothing under the arena's
// lock touches a pipeline, and nothing reacquires what a Yield released.
struct Sweeps {
  std::shared_mutex arena_mutex;
  Lazy<std::vector<Hessian>> hessians;
  Lazy<DenseHessian> dense;
  // Indexed by Want: the extra block of Hvp, Vjp, Jvp and JacobianJvp.
  std::array<Lazy<std::vector<NodeId>>, want_count> products;
  Lazy<ArenaWalk> hessian_walk; // a system's hessian() at a point
  Lazy<ArenaWalk> taylor_walk;  // root 0's univariate_derivative
  Lazy<bool> taylor_computes;   // taylor_walk at TaylorDual, any order
#ifdef DDX_HAS_JIT
  Lazy<Rebalanced> rebalanced;
#endif
};

// Per-thread sweep buffers, since two threads may be inside one equation at
// once. Kept between calls, so resize() allocates only when a sweep grows.
template <impl::Numeric T> struct TapeSet {
  std::vector<T> point, tape, vtape;
};
template <impl::Numeric T> struct ThreadTapes {
  TapeSet<T> values;           // For a system's hessian() at a point.
  TapeSet<T> one;              // For a batch swept one point at a time.
  TapeSet<T> block;            // For a batch swept block_lanes points at once.
  std::vector<T> kernel_tapes; // For a kernel's tensor steps.
  // A point call's output addresses, kept because a dense block is n^2 of them
  // and a fresh array that size page-faults on every call.
  std::vector<T *> addresses;
};
// Block scope, because GCC 15 never registers the destructor of a thread_local
// variable template named from a template, which leaked tapes on thread exit.
template <impl::Numeric T>
[[nodiscard]] DDX_TEMPLATE_API ThreadTapes<T> &thread_tapes() {
  thread_local ThreadTapes<T> kept;
  return kept;
}

// Lends a thread's tapes to one sweep. A sweep that may call an extern takes
// them away meanwhile, so a nested sweep on this thread grows its own set.
template <impl::Numeric T> class TapeLease {
public:
  explicit TapeLease(bool calls_out) : thread_(thread_tapes<T>()) {
    if (calls_out) {
      taken_.emplace(std::move(thread_));
    }
  }
  ~TapeLease() {
    if (taken_) {
      thread_ = std::move(*taken_);
    }
  }
  TapeLease(const TapeLease &) = delete;
  TapeLease &operator=(const TapeLease &) = delete;
  [[nodiscard]] ThreadTapes<T> &tapes() noexcept {
    return taken_ ? *taken_ : thread_;
  }

private:
  ThreadTapes<T> &thread_;
  std::optional<ThreadTapes<T>> taken_;
};

// An equation's engine: its own arena and roots, every sweep, and a pipeline
// per Want. `Yield` is what a sweep releases while it runs, e.g. Python's GIL.
template <impl::Numeric T, std::default_initializable Yield = Nothing>
class DDX_TEMPLATE_API Core : public JacobianSurface<Core<T, Yield>, T> {
  // Restricted runs its own pipelines through dispatch_in() and at_in().
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

  // Optional rather than 0 on poison, since a literal-only graph has arity 0.
  [[nodiscard]] constexpr std::optional<std::size_t> arity() const {
    return unless_poisoned([this] { return symbol_count(); });
  }
  [[nodiscard]] constexpr std::optional<std::size_t> outputs() const {
    return unless_poisoned([this] { return roots_.size(); });
  }
  // In declaration order, which is the order of a positional point and of
  // every derivative's columns.
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
  // Everything a point supplies, in positional order: the symbols, then the
  // parameters.
  [[nodiscard]] constexpr std::optional<std::size_t> input_count() const {
    return unless_poisoned([this] { return arena_->inputs().named(); });
  }
  [[nodiscard]] std::optional<std::vector<std::string>> inputs() const {
    return unless_poisoned([this] { return arena_->names(); });
  }

  // The input list is known only at run time, so every spelling can fail.
  template <rt::detail::CPointArg... Args>
  [[nodiscard]] constexpr result<std::vector<T>>
  point(const Args &...args) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    std::vector<T> at(arena_->inputs().named(), T{});
    result<void> ok{};
    if constexpr ((impl::CEntry<Args> && ...) && sizeof...(Args) > 0) {
      ok = assign_named(
          at, std::array{std::pair{std::remove_cvref_t<Args>::symbol.view(),
                                   static_cast<T>(args.value)}...});
    } else if constexpr (sizeof...(Args) == 1 &&
                         (rt::detail::CNamedRange<Args> && ...)) {
      // Checked before the plain range, since a map is also an input_range but
      // its elements are pairs.
      ok = assign_named(at, args...);
    } else if constexpr (sizeof...(Args) == 1 &&
                         (std::ranges::input_range<Args> && ...)) {
      ok = assign_range(at, args...);
    } else {
      const std::array<T, sizeof...(Args)> positional{static_cast<T>(args)...};
      ok = assign_range(at, positional);
    }
    return ok.transform([&at] { return std::move(at); });
  }

  // H(x)v, n long. The direction comes first as a span, so it can never be
  // read as part of the point.
  [[nodiscard]] result<std::vector<T>>
  hvp(std::span<const T> direction,
      const rt::detail::CPointArg auto &...args) const {
    return block_at<Want::Hvp, &Blocks::hessian>(direction, args...);
  }

  // The j-th Hessian column, chosen by name since the symbol list exists only
  // at run time.
  [[nodiscard]] result<std::vector<T>>
  hvp(std::string_view along, const rt::detail::CPointArg auto &...args) const {
    return unit(along).and_then(
        [&](const auto &e) { return hvp(std::span<const T>{e}, args...); });
  }

  // w'J, n long, with one weight per function.
  [[nodiscard]] result<std::vector<T>>
  vjp(std::span<const T> weights,
      const rt::detail::CPointArg auto &...args) const {
    return block_at<Want::Vjp, &Blocks::jacobian>(weights, args...);
  }

  // Jv, m long, the mirror of vjp().
  [[nodiscard]] result<std::vector<T>>
  jvp(std::span<const T> direction,
      const rt::detail::CPointArg auto &...args) const {
    return block_at<Want::Jvp, &Blocks::jacobian>(direction, args...);
  }

  // A view keeping only the named symbols' Jacobian columns, so the other cells
  // are never swept. It reads this equation and must not outlive it.
  template <std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_reference_t<R>,
                                 std::string_view>
  [[nodiscard]] result<Restricted<Core>> restrict(const R &names) const {
    DDX_CHECK(alive());
    std::vector<bool> chosen(symbol_count(), false);
    for (const std::string_view name : names) {
      DDX_TRY(slot, symbol(name));
      chosen[*slot] = true;
    }
    return Restricted<Core>{*this, chosen};
  }
  [[nodiscard]] result<Restricted<Core>> restrict(
      std::initializer_list<std::string_view> names) const {
    return restrict(std::span{names});
  }

  // F(z, x) for each implicit's closure, which is zero to rounding where the
  // caller's z is a root. ddx never solves for z; this only checks it.
  [[nodiscard]] result<std::vector<T>>
  residual(const rt::detail::CPointArg auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      [[maybe_unused]] const auto held = reading();
      const auto closures = arena_->closures();
      const auto values = evaluate_reachable(*arena_, closures, at,
                                             tensor_kernels(), contraction());
      return closures |
             std::views::transform([&values](NodeId f) { return values[f]; }) |
             impl::to<std::vector<T>>();
    });
  }

  // Dense row-major m x n x n; the graph holds it compressed by colour.
  [[nodiscard]] result<std::vector<T>>
  hessian(const rt::detail::CPointArg auto &...args) const {
    if (!poisoned() && is_system()) {
      // A system is walked off the arena one output at a time, since no
      // pipeline holds m colourings; orders() applies the pipeline's gate.
      DDX_TRY(at, point(args...));
      DDX_CHECK(orders(Want::Hessian));
      const std::size_t m = roots_.size();
      const std::size_t n = symbol_count();
      std::vector<T> f(m);
      std::vector<T> g(m * n);
      std::vector<T> out(m * n * n);
      walk(*at, f, g, out);
      return out;
    }
    return at<Want::Hessian>({}, args...).transform([this](const auto &b) {
      // Scatter only the colouring's stored cells; the rest of `out` stays 0.
      const std::size_t n = symbol_count();
      std::vector<T> out(n * n);
      scatter(
          hessians().front().colouring,
          [&b](std::size_t k) { return &b.hessian[k]; },
          [&out](std::size_t c) { return &out[c]; }, n, 1);
      return out;
    });
  }

  // The same n x n Hessian from one O(n^2) sweep with matrix tangents, where
  // hessian() colours and sweeps n + 1 times. One output only.
  [[nodiscard]] result<std::vector<T>>
  dense_hessian(const rt::detail::CPointArg auto &...args) const {
    return block_at<Want::DenseHessian, &Blocks::hessian>({}, args...);
  }
  // Terms the dense sweep kept O(n^2) and terms that fell back to a dense
  // product, which is still correct. Refuses exactly as dense_hessian() would.
  struct DenseCost {
    std::size_t n2_terms = 0;
    std::size_t n3_terms = 0;
  };
  [[nodiscard]] result<DenseCost> dense_hessian_cost() const {
    return ready(Want::DenseHessian).transform([this] {
      const DenseHessian &d = dense_sweep();
      return DenseCost{.n2_terms = d.n2_terms, .n3_terms = d.n3_terms};
    });
  }

  // Whether and how to compile, discarding anything already compiled. This is
  // the one non-const member, so a call overlapping it is the caller's race.
  auto &options(this auto &self, const jit::Options &opt) {
    return replaced(self, self.options_, opt);
  }
  [[nodiscard]] const jit::Options &options() const noexcept {
    return options_;
  }

  // The table computing tensor nodes, rt::use_tensor_kernels by default.
  // Setting one drops the pipelines, since a table decides the last bits.
  auto &tensor_kernels(this auto &self, const TensorKernelTable<T> &table) {
    return replaced(self, self.kernels_, &table);
  }
  [[nodiscard]] constexpr const TensorKernelTable<T> &
  tensor_kernels() const noexcept {
    if (kernels_ != nullptr) {
      return *kernels_;
    }
    // The process default lives in libddx, out of constant evaluation's reach,
    // and a constant-evaluated equation computes no tensor row anyway.
    if consteval {
      return reference_table<T>;
    }
    return rt::tensor_kernels<T>();
  }

  // Sweeps and freezes `wants` now, so later calls from any number of threads
  // find their pipelines built and never take a write lock.
  [[nodiscard]] result<void> prepare(std::initializer_list<Want> wants) const {
    for (const Want want : wants) {
      DDX_CHECK(ready(want));
      (void)pipeline(want);
    }
    return {};
  }

  // Whether a batch call for `want` runs compiled code, building its pipeline
  // now. False with no backend, on a refused compile, and under Interpret.
  [[nodiscard]] bool uses_kernel(Want want = Want::Jacobian) const {
    return compile_status(want).has_value();
  }

  // What one call for `want` steps through, counted on its frozen graph. Builds
  // the pipeline as prepare() does but evaluates nothing.
  [[nodiscard]] result<Census> census(Want want = Want::Jacobian) const {
    return ready(want).transform(
        [&] { return detail::census_of(*pipeline(want)->graph); });
  }

  // uses_kernel() with the reason when it is false: the equation's own error,
  // not_compiling under Interpret, or the JIT's refusal and its detail.
  [[nodiscard]] jit::compile_result<void>
  compile_status(Want want = Want::Jacobian) const {
    return ready(want)
        .transform_error([](error e) {
          return jit::compile_error{.code = e.code, .detail = {}};
        })
        .and_then([&] { return pipeline(want)->status; });
  }

  // One Taylor sweep seeded with c[0] = x0 and c[1] = 1; the derivative is
  // c[Order] times Order!.
  template <std::size_t Order>
    requires(Order > 0)
  [[nodiscard]] result<T> univariate_derivative(T x0) const {
    T d{};
    return univariate_derivative<Order>(std::span{&x0, 1}, std::span{&d, 1})
        .transform([&] { return d; });
  }

  // out[i] is the Order-th derivative at xs[i]. From block_lanes points on,
  // each sweep carries block_lanes of them, so dispatch is paid once a block.
  template <std::size_t Order>
    requires(Order > 0)
  [[nodiscard]] result<void> univariate_derivative(std::span<const T> xs,
                                                   std::span<T> out) const {
    DDX_CHECK(single());
    if (arena_->inputs().named() != 1 || symbol_count() != 1) {
      return fail(errc::not_univariate);
    }
    if (out.size() != xs.size()) {
      return fail(errc::wrong_arity);
    }
    using Taylor = impl::TaylorDual<T, Order>;
    // Only root 0's cone is walked, since the arena may also hold a gradient
    // and a Hessian and a Taylor node costs Order + 1 coefficients.
    const ArenaWalk &walk = point_walk(
        sweeps_->taylor_walk, [this] { return std::array{roots_[0]}; });
    // Refuse rather than sweep, since an op the Taylor scalar lacks would
    // evaluate to a zero node and give a silently wrong number. The order
    // never changes what is supported, so the verdict is kept at order 1.
    using Order1 = impl::TaylorDual<T, 1>;
    static_assert(
        std::ranges::all_of(std::views::iota(0uz, op_count), [](auto i) {
          const auto op = static_cast<OpCode>(i);
          return supports<Order1>(op) == supports<Taylor>(op);
        }));
    if (!sweeps_->taylor_computes.get([&] {
          [[maybe_unused]] const auto held = reading();
          return walk.template computes<Order1>(*arena_);
        })) {
      return fail(errc::unsupported_scalar);
    }
    // The tape is thread-kept and not cleared; every id read below is
    // scheduled, so each sweep writes it first.
    TapeLease<Taylor> lease{calls_out_};
    TapeSet<Taylor> &kept = lease.tapes().values;
    [[maybe_unused]] const auto held = sweeping();
    const auto sweep = [&](auto lanes) {
      constexpr std::size_t W = lanes;
      kept.tape.resize(arena_->size() * W);
      kept.vtape.resize(walk.extent() * W);
      std::array<Taylor, W> at;
      for (const std::size_t base :
           std::views::iota(0uz, xs.size()) | std::views::stride(W)) {
        // A short block repeats its last point.
        const std::size_t width = std::min(W, xs.size() - base);
        for (const auto [k, a] : at | std::views::enumerate) {
          a = Taylor::variable(xs[base + std::min<std::size_t>(k, width - 1)]);
        }
        walk.template operator()<W>(*arena_, at, std::span<Taylor>{kept.tape},
                                    std::span<Taylor>{kept.vtape});
        std::ranges::transform(
            std::span{kept.tape}.subspan(roots_[0] * W, width),
            out.begin() + base, [](const Taylor &t) {
              return t.c[Order] *
                     static_cast<T>(
                         impl::detail::compile_time_factorial(Order));
            });
      }
    };
    xs.size() < block_lanes
        ? sweep(std::integral_constant<std::size_t, 1>{})
        : sweep(std::integral_constant<std::size_t, block_lanes>{});
    return {};
  }

  // --- batch ---------------------------------------------------------------
  // `xs[j]` is symbol j's column; each output span holds one pointer per output
  // column, and every column is n long.

  [[nodiscard]] result<void>
  evaluate(const rt::detail::CColumns<const T *> auto &xs,
           const rt::detail::CColumns<T *> auto &f, std::size_t n) const {
    return this->template batch<Want::Value>(xs, n, f);
  }

  [[nodiscard]] result<void>
  hessian(const rt::detail::CColumns<const T *> auto &xs,
          const rt::detail::CColumns<T *> auto &f,
          const rt::detail::CColumns<T *> auto &g,
          const rt::detail::CColumns<T *> auto &h, std::size_t n) const {
    return this->template batch<Want::Hessian>(xs, n, f, g, h);
  }

  // `v` is a second block of input columns, one per symbol and n long, spliced
  // after `xs`. `g` comes out of the same sweep.
  [[nodiscard]] result<void> hvp(const rt::detail::CColumns<const T *> auto &xs,
                                 const rt::detail::CColumns<const T *> auto &v,
                                 const rt::detail::CColumns<T *> auto &f,
                                 const rt::detail::CColumns<T *> auto &g,
                                 const rt::detail::CColumns<T *> auto &hv,
                                 std::size_t n) const {
    return this->template batch<Want::Hvp>(widen(xs, v), n, f, g, hv);
  }

  // `h` is n^2 columns, row-major: H(i, j) is column i * n + j.
  [[nodiscard]] result<void>
  dense_hessian(const rt::detail::CColumns<const T *> auto &xs,
                const rt::detail::CColumns<T *> auto &f,
                const rt::detail::CColumns<T *> auto &g,
                const rt::detail::CColumns<T *> auto &h, std::size_t n) const {
    return this->template batch<Want::DenseHessian>(xs, n, f, g, h);
  }

  [[nodiscard]] result<void> vjp(const rt::detail::CColumns<const T *> auto &xs,
                                 const rt::detail::CColumns<const T *> auto &w,
                                 const rt::detail::CColumns<T *> auto &f,
                                 const rt::detail::CColumns<T *> auto &out,
                                 std::size_t n) const {
    return this->template batch<Want::Vjp>(widen(xs, w), n, f, out);
  }

  [[nodiscard]] result<void> jvp(const rt::detail::CColumns<const T *> auto &xs,
                                 const rt::detail::CColumns<const T *> auto &v,
                                 const rt::detail::CColumns<T *> auto &f,
                                 const rt::detail::CColumns<T *> auto &out,
                                 std::size_t n) const {
    return this->template batch<Want::Jvp>(widen(xs, v), n, f, out);
  }

  // The column counts a caller sizes buffers by. None builds a pipeline; only
  // the Hessian's counts sweep, since its colouring fixes them.
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

  // The compressed blocks' layouts, which batch callers need because only the
  // dense spellings undo the compression.

  // at(i, j) is nullopt where df[i]/dx[j] is structurally zero.
  [[nodiscard]] constexpr std::optional<std::reference_wrapper<const Sparsity>>
  jacobian_pattern() const {
    return unless_poisoned([this] { return std::cref(jacobian_.pattern); });
  }

  // The (row, column) cell each Hessian column holds; see Colouring::entries().
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
  using Surface::widen;
  using Compiled = detail::Compiled<T>;
  using Pipeline = detail::Pipeline<T>;
  using Pipelines = std::array<Pipeline, want_count>;

  // Poisoned: no arena, no roots, no pipelines or sweeps, and every accessor
  // returns early.
  constexpr explicit Core(error why) : bad_(why) { settle(); }

  // The equation owns its arena from here; a lent one was copied, so nothing
  // outside can move a node under it.
  constexpr Core(std::unique_ptr<Arena<T>> arena, std::vector<NodeId> roots)
      : arena_(std::move(arena)), roots_(std::move(roots)) {
    // An implicit no derivative can carry, or a vector output where a scalar is
    // required, poisons the whole equation.
    if (const auto why = unclosed(*arena_)) {
      poison(error{*why});
    } else if (std::ranges::any_of(roots_, [this](NodeId r) {
                 return !detail::is_scalar((*arena_)[r]);
               })) {
      poison(error{errc::tensor_shape});
    } else if (no_kernel_for_tensors()) {
      poison(error{errc::unsupported_scalar});
    } else {
      // One reverse sweep per output is cheap, and the column counts, the
      // pattern and constant evaluation all read it.
      jacobian_ = derive_jacobian(*arena_, roots_);
    }
    settle();
  }

  // A tensor row is computed by a table of kernels, at floating point only.
  [[nodiscard]] constexpr bool no_kernel_for_tensors() const {
    if constexpr (std::floating_point<T>) {
      return false;
    } else {
      const auto live = detail::reachable(*arena_, roots_);
      return std::ranges::any_of(detail::live_ids(live), [this](NodeId v) {
        return detail::on_tensors((*arena_)[v]);
      });
    }
  }

  // Refuses with `why` and drops everything else, so from here on poisoned()
  // holds exactly when arena_ is null.
  constexpr void poison(error why) {
    bad_ = why;
    arena_.reset();
    roots_.clear();
  }
  // Every constructor's tail: the gates' flags, then the pipelines and sweeps,
  // which constant evaluation and a poisoned equation never reach.
  constexpr void settle() {
    if (!arena_) {
      return;
    }
    first_order_ = first_order_only(*arena_, roots_);
    calls_out_ = !arena_->externs().empty();
    if !consteval {
      pipelines_ = std::make_unique<Pipelines>();
      sweeps_ = std::make_unique<Sweeps>();
    }
  }

  // The first poison reason, by code: a clashing name or a missing arena is a
  // different mistake from a literal that reached no graph.
  [[nodiscard]] static constexpr std::optional<error>
  why_not(std::span<const Expression<T>> outputs) noexcept {
    for (const Expression<T> &e : outputs) {
      if (const auto why = Router::why(e)) {
        return error{*why};
      }
    }
    if (Router::first_arena(outputs) == nullptr) {
      return error{errc::no_graph};
    }
    return std::nullopt;
  }

  // The arena the outputs are read in (`owned`, or a copy of the one they were
  // built in), and their roots there.
  [[nodiscard]] static constexpr std::pair<std::unique_ptr<Arena<T>>,
                                           std::vector<NodeId>>
  rooted(std::span<const Expression<T>> outputs,
         std::unique_ptr<Arena<T>> owned) {
    auto arena =
        owned ? std::move(owned)
              : std::make_unique<Arena<T>>(*Router::first_arena(outputs));
    auto roots = outputs | std::views::transform([&arena](const auto &e) {
                   return e.id(*arena);
                 }) |
                 impl::to<std::vector<NodeId>>();
    return {std::move(arena), std::move(roots)};
  }

  // Builds a poisoned or a live D from `outputs`, for whichever class wraps
  // Core.
  template <typename D>
  [[nodiscard]] static constexpr D make(std::span<const Expression<T>> outputs,
                                        std::unique_ptr<Arena<T>> owned) {
    if (const auto why = why_not(outputs)) {
      return D{*why};
    }
    auto [arena, roots] = rooted(outputs, std::move(owned));
    return D{std::move(arena), std::move(roots)};
  }

  // Assumes a live arena; arity() is the checked spelling.
  [[nodiscard]] constexpr std::size_t symbol_count() const {
    return arena_->symbols().size();
  }
  [[nodiscard]] constexpr std::span<const NodeId> roots() const {
    return roots_;
  }
  [[nodiscard]] constexpr const Jacobian &derivative() const {
    return jacobian_;
  }

  // The Jacobian cells a set of pipelines is frozen over: all of them for the
  // equation's own, or the indices a restrict() kept.
  struct Cells {
    const Jacobian &jacobian;
    // Unset means all cells; a restriction may keep none.
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
        compiling(), [&] { return frozen_for(want, cells); });
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
    DDX_CHECK(ready(want));
    auto c = pipeline_in(pipelines, cells, want);
    // Count columns, not nodes, since a vector output is one column per
    // element.
    const bool fits = xs.size() == c->graph->arity() &&
                      std::ranges::none_of(zip_blocks(out, c->graph->layout()),
                                           [](const auto &pair) {
                                             const auto &[columns, count] =
                                                 pair;
                                             return columns.size() != count;
                                           });
    if (!fits) {
      return fail(errc::wrong_column_count);
    }
    return c;
  }

  // A system's Hessians read off the arena one point at a time, since a frozen
  // graph carries one colouring. `f` is m long, `g` m x n and `h` m x n x n.
  void walk(std::span<const T> at, std::span<T> f, std::span<T> g,
            std::span<T> h) const {
    const auto &blocks = hessians();
    // The nodes the three blocks read, planned once. Appended per block rather
    // than via views::join, which GCC 15 flags maybe-uninitialized under -Wall.
    const ArenaWalk &walk = point_walk(sweeps_->hessian_walk, [&] {
      std::vector<NodeId> wanted{roots_.begin(), roots_.end()};
      std::ranges::for_each(blocks, [&wanted](const Hessian &b) {
        impl::append(wanted, b.compressed);
        impl::append(wanted, b.partial);
        wanted.push_back(b.zero);
      });
      return wanted;
    });
    const std::size_t n = symbol_count();
    // The tape is thread-kept and not cleared; every id read below is
    // scheduled, so this call writes it first.
    TapeLease<T> lease{calls_out_};
    ThreadTapes<T> &kept = lease.tapes();
    std::vector<T> &values = kept.values.tape;
    [[maybe_unused]] const auto held = sweeping();
    values.resize(arena_->size());
    kept.values.vtape.resize(walk.extent());
    walk(*arena_, at, std::span<T>{values}, std::span<T>{kept.values.vtape},
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

  // Nullopt on a poisoned equation, `value()` otherwise.
  [[nodiscard]] constexpr auto unless_poisoned(auto &&value) const {
    using V = std::remove_cvref_t<std::invoke_result_t<decltype(value)>>;
    return poisoned() ? std::optional<V>{} : std::optional<V>{value()};
  }

  // As unless_poisoned, but also nullopt for a system, since the answer is
  // about a single output's Hessian.
  [[nodiscard]] auto unless_system(auto &&value) const {
    return is_system() ? decltype(unless_poisoned(value)){}
                       : unless_poisoned(value);
  }

  // m long, always a vector: what System answers and Equation narrows.
  [[nodiscard]] constexpr result<std::vector<T>>
  values(const rt::detail::CPointArg auto &...args) const {
    return block_at<Want::Value, &Blocks::values>({}, args...);
  }

private:
  // The walk over the cone `wanted()` names, planned once under the read lock.
  const ArenaWalk &point_walk(Lazy<ArenaWalk> &slot, auto &&wanted) const {
    return slot.get([&] {
      const auto ids = wanted();
      [[maybe_unused]] const auto held = reading();
      return ArenaWalk{*arena_, live_ids(reachable(*arena_, ids)),
                       contraction()};
    });
  }

  // Runs `build` once under the arena's write lock, since every sweep appends.
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
  [[nodiscard]] const DenseHessian &dense_sweep() const {
    return sweep(sweeps_->dense, [this] {
      return derive_dense_hessian(*arena_, roots_.front());
    });
  }
  // The product a directional want adds. Hvp is root 0's only, since a system
  // refuses it earlier.
  [[nodiscard]] const std::vector<NodeId> &product(Want want) const {
    return sweep(sweeps_->products[std::to_underlying(want)], [&] {
      switch (want) {
      case Want::Hvp:
        return derive_hvp(*arena_, roots_.front()).product;
      case Want::Vjp:
        return derive_vjp(*arena_, roots_);
      case Want::Jvp:
        return derive_jvp(*arena_, roots_);
      default:
        return derive_jvp(*arena_, jacobian_.partial);
      }
    });
  }

  [[nodiscard]] result<std::uint32_t> symbol(std::string_view name) const {
    return arena_->symbol(name);
  }
  [[nodiscard]] constexpr const Sparsity &jacobian_sparsity() const {
    return jacobian_.pattern;
  }
  // e_j for the symbol named `along`, or unknown_symbol.
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
  // The gates, in the order a call meets them; ready() is the one every batch
  // and point call passes.
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
  // Second order is refused where a reachable extern supplies only first
  // partials, rather than answered with zeros.
  [[nodiscard]] constexpr result<void> orders(Want want) const {
    return want_info(want).second_order && first_order_
               ? fail(errc::extern_order)
               : result<void>{};
  }
  // The dense Hessian has no block where a matrix the symbols move is
  // multiplied (see rt/dense.hpp).
  [[nodiscard]] constexpr result<void> blocked(Want want) const {
    return want == Want::DenseHessian && symbol_count() != 0 &&
                   dense_sweep().block == no_node
               ? fail(errc::dense_refused)
               : result<void>{};
  }
  [[nodiscard]] constexpr result<void> ready(Want want) const {
    return (want_info(want).single ? single() : alive())
        .and_then([&] { return orders(want); })
        .and_then([&] { return blocked(want); });
  }

private:
  using Blocks = PerBlock<std::vector<T>>;
  // One block of W's answer.
  template <Want W, auto Block>
  [[nodiscard]] constexpr result<std::vector<T>>
  block_at(std::span<const T> along, const auto &...args) const {
    return at<W>(along, args...).transform([](auto b) {
      return std::move(b.*Block);
    });
  }
  // Sets `member` to `value` and drops whatever was compiled for the old one.
  static auto &replaced(auto &self, auto &member, const auto &value) {
    if (!(member == value)) {
      member = value;
      self.pipelines_ = std::make_unique<Pipelines>();
    }
    return self;
  }
  [[nodiscard]] auto reading() const {
    return std::shared_lock{sweeps_->arena_mutex};
  }
  [[nodiscard]] auto writing() const {
    return std::unique_lock{sweeps_->arena_mutex};
  }
  // What a sweep holds. Yield is released first, so the lock is never waited
  // for or held while reacquiring what Yield let go.
  struct Sweeping {
    [[no_unique_address]] Yield yielded{};
    std::shared_lock<std::shared_mutex> held;
  };
  [[nodiscard]] Sweeping sweeping() const {
    return Sweeping{.held = reading()};
  }

  // The three blocks' column counts for `want`, taking the Jacobian cells from
  // a restrict() view where it kept some.
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
        return extent(Direction::Symbols);
      case BlockShape::Functions:
        return extent(Direction::Functions);
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

  // How many values a direction holds: one per symbol or per function.
  [[nodiscard]] constexpr std::size_t extent(Direction d) const noexcept {
    return d == Direction::Symbols     ? symbol_count()
           : d == Direction::Functions ? roots_.size()
                                       : 0;
  }

  // One point of `W`, widened by `along` where the want takes a direction.
  // Constant evaluation takes gather()'s arena walk instead of a pipeline.
  template <Want W>
  [[nodiscard]] constexpr result<PerBlock<std::vector<T>>>
  at_in(Pipelines *pipelines, const Cells &cells, std::span<const T> along,
        const rt::detail::CPointArg auto &...args) const {
    DDX_TRY(pt, point(args...));
    constexpr Direction direction = want_info(W).direction;
    if (direction != Direction::None && along.size() != extent(direction)) {
      return fail(errc::wrong_direction);
    }
    const Layout size = sizes(W, cells);
    PerBlock<std::vector<T>> out{.values = std::vector<T>(size.values),
                                 .jacobian = std::vector<T>(size.jacobian),
                                 .hessian = std::vector<T>(size.hessian)};
    if constexpr (direction == Direction::None) {
      DDX_CHECK(gather(pipelines, cells, W, *pt, out.values, out.jacobian,
                       out.hessian));
    } else {
      DDX_CHECK(gather(pipelines, cells, W, widen(*pt, along), out.values,
                       out.jacobian, out.hessian));
    }
    return out;
  }
  template <Want W>
  [[nodiscard]] constexpr result<PerBlock<std::vector<T>>>
  at(std::span<const T> along,
     const rt::detail::CPointArg auto &...args) const {
    return at_in<W>(pipelines_.get(), every_cell(), along, args...);
  }

  // Assigns name by name, refusing a point that leaves an input unnamed rather
  // than reading it as zero.
  [[nodiscard]] constexpr result<void>
  assign_named(std::vector<T> &at,
               const rt::detail::CNamedRange auto &entries) const {
    std::vector<bool> seen(at.size(), false);
    for (const auto &[name, value] : entries) {
      const auto input = arena_->lookup(std::string_view{name});
      if (!input) {
        return fail(errc::unknown_symbol);
      }
      const auto column =
          input_column(arena_->inputs(), input->op, input->index);
      at[column] = static_cast<T>(value);
      seen[column] = true;
    }
    return std::ranges::contains(seen, false) ? fail(errc::short_point)
                                              : result<void>{};
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

  // One point through the batch path at n = 1, so the batch gate is the point
  // gate. Constant evaluation, having no pipelines, takes the arena walk.
  // The walk is a tail, not an `else`, which GCC would escalate under P2564.
  // MSVC calls it dead (C4702, read at the brace), so the pragma sits outside.
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
      // At one point, every column is a single value's address.
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
    const auto values = evaluate_reachable(*arena_, wanted, at,
                                           tensor_kernels(), contraction());
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

  // The options a pipeline compiles under, and none under Interpret.
  [[nodiscard]] std::optional<jit::Options> compiling() const {
    if (jit::CKernelScalar<T> && options_.backend == jit::Backend::Compile) {
      return effective_options();
    }
    return std::nullopt;
  }

  // The graphs a pipeline computes.
  [[nodiscard]] Compiled frozen_for(Want want, const Cells &cells) const {
    // The blocks the want adds, swept before the freeze takes the lock. Vjp
    // and Jvp replace the Jacobian block rather than add to it.
    OutputBlocks added;
    Colouring colouring;
    switch (want) {
    case Want::Vjp:
    case Want::Jvp:
      added.jacobian = product(want);
      break;
    case Want::Hessian:
      added.hessian = hessians().front().compressed;
      colouring = hessians().front().colouring;
      break;
    case Want::Hvp:
      added.hessian = product(want);
      break;
    case Want::JacobianJvp: {
      const auto &tangent = product(want);
      added.hessian = cells.kept
                          ? *cells.kept | std::views::transform([&](auto k) {
                              return tangent[k];
                            }) | impl::to<std::vector<NodeId>>()
                          : tangent;
      break;
    }
    case Want::DenseHessian:
      // With no symbols there is no dense block.
      if (dense_sweep().block != no_node) {
        added.hessian = {dense_sweep().block};
      }
      break;
    case Want::Value:
    case Want::Jacobian:
    case Want::Gradient:
      break;
    }
    // `roots` and `partials` differ between the swept graph and the rebalanced
    // kernel graph. Every direction column counts, even an unread one.
    const WantInfo &info = want_info(want);
    const auto freeze = [&](std::span<const NodeId> roots,
                            const Jacobian &partials, bool for_kernel) {
      OutputBlocks blocks = added;
      if (info.values) {
        blocks.values.assign(roots.begin(), roots.end());
      }
      const bool pattern = info.jacobian == BlockShape::Cells;
      if (pattern) {
        blocks.jacobian = partials.partial;
      }
      return std::make_shared<const Graph<T>>(
          Graph<T>::freeze(*arena_, std::move(blocks), colouring,
                           pattern ? partials.pattern : Sparsity{}, contracts(),
                           extent(info.direction), for_kernel));
    };
    [[maybe_unused]] const Rebalanced *over =
        cells.kept ? nullptr : rebalanced(want);
    [[maybe_unused]] const auto held = reading();
    auto swept = freeze(roots_, cells.jacobian, false);
    auto compiled = swept;
#ifdef DDX_HAS_JIT
    // The kernel gets its own graph where it differs: over rebalanced roots, or
    // with a tensor's elementwise rows fused into their readers.
    if (over || (swept->has_tensors() && compiling())) {
      auto own = freeze(over ? over->roots : roots_,
                        over ? over->derivative : cells.jacobian, true);
      // Calls are checked against the swept graph and a rebalanced sum may fold
      // differently, so the kernel graph is kept only if its layout matches.
      if (in_order(own->layout()) == in_order(swept->layout()) &&
          own->jacobian_pattern().rowptr == swept->jacobian_pattern().rowptr &&
          own->jacobian_pattern().col == swept->jacobian_pattern().col) {
        compiled = std::move(own);
      }
    }
#endif
    return {.graph = std::move(swept),
            .compile_graph = std::move(compiled),
            .kernels = &tensor_kernels()};
  }

  // The rebalanced roots the kernel for `want` is lowered from, swept on first
  // ask; null where the kernel is lowered from the equation's own roots.
  [[nodiscard]] const Rebalanced *rebalanced([[maybe_unused]] Want want) const {
#ifdef DDX_HAS_JIT
    if constexpr (jit::CKernelScalar<T>) {
      // A graph with tensor steps is never rebalanced. The roots' cone is read,
      // not the whole arena, and under the lock since another sweep may append.
      const auto tensors = [this] {
        [[maybe_unused]] const auto held = reading();
        const auto live = detail::reachable(*arena_, roots_);
        return std::ranges::any_of(detail::live_ids(live), [this](NodeId v) {
          return detail::on_tensors((*arena_)[v]);
        });
      };
      if (!rebalanceable(want) || !compiling() ||
          detail::shared_compiler() == nullptr || tensors()) {
        return nullptr;
      }
      return &sweep(sweeps_->rebalanced, [this] {
        Rebalanced over{.roots = detail::rebalance(*arena_, roots_),
                        .derivative = {}};
        over.derivative = derive_jacobian(*arena_, over.roots);
        return over;
      });
    }
#endif
    return nullptr;
  }

  // The emission width for the caller's batch: a short batch emits scalar, the
  // same threshold the sweep uses. An explicit `lanes` is honoured.
  [[nodiscard]] jit::Options effective_options() const noexcept {
    return jit::for_batch(options_, block_lanes);
  }

  // FMA contraction is decided in the graph so every backend folds the same
  // products; it defaults on even without a JIT, so the build never matters.
  [[nodiscard]] constexpr bool contracts() const noexcept {
    return options_.codegen.contract;
  }
  [[nodiscard]] constexpr Contract contraction() const noexcept {
    return contracts() ? Contract::on : Contract::off;
  }

  // The kernel where one compiled, else the sweep.
  void run(const Compiled &c, std::span<const T *const> xs,
           const Columns<T> &out, std::size_t n) const {
    [[maybe_unused]] const auto held = sweeping();
    TapeLease<T> lease{calls_out_};
    if constexpr (jit::CKernelScalar<T>) {
      if (c.kernel) {
        const detail::TensorCall<T> call{.arena = arena_.get(),
                                         .graph = c.compile_graph.get(),
                                         .kernels = c.kernels};
        std::vector<T> &tapes = lease.tapes().kernel_tapes;
        tapes.resize(c.kernel.shape().tapes);
        c.kernel(xs, out.values, out.jacobian, out.hessian, n,
                 {.call = &call, .tapes = tapes});
        return;
      }
    }
    interpret(lease.tapes(), *c.graph, *c.kernels, xs, out, n);
  }

  // The interpreter, within ~1.4x of a kernel on large graphs. A batch shorter
  // than block_lanes sweeps one point at a time rather than padding.
  void interpret(ThreadTapes<T> &kept, const Graph<T> &graph,
                 const TensorKernelTable<T> &kernels,
                 std::span<const T *const> xs, const Columns<T> &out,
                 std::size_t n) const {
    n < block_lanes
        ? sweep_points<1>(graph, kernels, kept.one, xs, out, n)
        : sweep_points<block_lanes>(graph, kernels, kept.block, xs, out, n);
  }

  // W points per sweep, so dispatch is paid once per node per block and each op
  // becomes a vectorisable lane loop. A short block repeats its last point.
  template <std::size_t W>
  void sweep_points(const Graph<T> &graph, const TensorKernelTable<T> &kernels,
                    TapeSet<T> &kept, std::span<const T *const> xs,
                    const Columns<T> &out, std::size_t n) const {
    auto &[point, tape, vtape] = kept;
    point.resize(graph.arity() * W);
    tape.resize(graph.tape_rows() * W);
    vtape.resize(graph.tensors().extent * W);
    const auto t = detail::TapeView<T>::of(graph, vtape.data(), &kernels);
    for (const std::size_t base :
         std::views::iota(0uz, n) | std::views::stride(W)) {
      const std::size_t width = std::min(W, n - base);
      // At W = 1 the gather is a fair part of the cost on a small graph.
      if constexpr (W == 1) {
        std::ranges::transform(xs, point.begin(), [base](const T *column) {
          return column[base];
        });
      } else {
        for (const auto [dst, column] :
             std::views::zip(point | std::views::chunk(W), xs)) {
          const auto tail =
              std::ranges::copy(std::span{column + base, width}, dst.begin())
                  .out;
          std::ranges::fill(tail, dst.end(), column[base + width - 1]);
        }
      }
      evaluate_block<W>(*arena_, std::span<const T>{point}, graph.schedule(),
                        std::span<T>{tape}, t);
      scatter_blocks(graph.output_rows(), std::span<const T>{tape},
                     std::span<const T>{vtape}, out, base, W, width);
    }
  }

  std::unique_ptr<Arena<T>> arena_;
  std::vector<NodeId> roots_;
  jit::Options options_{};
  // Null means the process default, read at each freeze.
  const TensorKernelTable<T> *kernels_ = nullptr;
  Jacobian jacobian_;
  // A unique_ptr, whose constexpr destructor keeps a constant-evaluated
  // equation destructible. Null there and when poisoned.
  std::unique_ptr<Pipelines> pipelines_;
  // Every other sweep, built once on first ask under the arena's write lock.
  std::unique_ptr<Sweeps> sweeps_;
  // Set exactly when the equation is poisoned, which is when arena_ is null.
  std::optional<error> bad_{};
  // Set when a reachable extern supplies only first partials; see orders().
  bool first_order_ = false;
  bool calls_out_ = false; // Set when the arena holds an extern; see TapeLease.
};

// restrict()'s answer: the Jacobian narrowed to some symbols' columns, with its
// own pipelines over the kept cells. It must not outlive its equation.
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
    for (const Sparsity::Row &row : full.pattern.rows()) {
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

  // The symbols kept, in column order.
  [[nodiscard]] std::span<const std::string> symbols() const { return names_; }
  [[nodiscard]] std::size_t jacobian_columns() const { return kept_.size(); }
  [[nodiscard]] const Sparsity &jacobian_pattern() const {
    return jacobian_.pattern;
  }
  [[nodiscard]] result<void> dispatch(Want want, std::span<const T *const> xs,
                                      const Columns<T> &out,
                                      std::size_t n) const {
    return core_->dispatch_in(*pipelines_, cells(), want, xs, out, n);
  }

private:
  [[nodiscard]] typename C::Cells cells() const {
    return {.jacobian = jacobian_, .kept = std::span{kept_}};
  }
  // `along` still holds one value per symbol of the whole equation.
  template <Want W>
  [[nodiscard]] result<PerBlock<std::vector<T>>>
  at(std::span<const T> along,
     const rt::detail::CPointArg auto &...args) const {
    return core_->template at_in<W>(pipelines_.get(), cells(), along, args...);
  }
  [[nodiscard]] auto unit(std::string_view along) const {
    return core_->unit(along);
  }
  [[nodiscard]] const Sparsity &jacobian_sparsity() const {
    return jacobian_.pattern;
  }

  const C *core_;
  std::vector<std::string> names_;
  Jacobian jacobian_;
  std::vector<std::size_t> kept_; // Indices into the equation's cells.
  std::unique_ptr<typename C::Pipelines> pipelines_;
};

// An equation's function count as its type states it: a fixed count of one
// narrows the value to a scalar, while a run-time count answers a vector.
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

// Frees this thread's sweep tapes, which otherwise stay at their largest size
// until the thread exits. The next sweep on the thread grows them again.
template <impl::Numeric T = double> void release_tapes() noexcept {
  detail::thread_tapes<T>() = {};
}
} // namespace ddx::rt

// The scalars libddx instantiates once (src/rt/instances.cpp); thread_tapes()
// is listed so there is one tape set per thread rather than per binary.
// MSVC refuses dllexport on an extern declaration (C4910), so libddx's own TUs
// skip these there and instances.cpp exports at the definition.
#define DDX_RT_INSTANCES(EXTERN, API, T)                                       \
  EXTERN template class API ::ddx::rt::detail::Pipeline<T>;                    \
  EXTERN template class API ::ddx::rt::detail::JacobianSurface<                \
      ::ddx::rt::detail::Core<T>, T>;                                          \
  EXTERN template class API ::ddx::rt::detail::Core<T>;                        \
  EXTERN template class API ::ddx::rt::detail::Restricted<                     \
      ::ddx::rt::detail::Core<T>>;                                             \
  EXTERN template API ::ddx::rt::detail::ThreadTapes<T>                        \
      & ::ddx::rt::detail::thread_tapes<T>();

// C4251 flags std:: members of exported classes; both sides share one STL.
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
