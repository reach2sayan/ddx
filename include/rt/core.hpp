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
#include "rt/lanes.hpp"
#include "rt/rebalance.hpp"
#include "symbolic/entry.hpp" // CEntry
#include "util/config.hpp"
#include "util/ranges.hpp" // to<C>() and append(), ours

#include <boost/container/small_vector.hpp>

#include "cl/device.hpp"
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
#include <utility>
#include <vector>

namespace ddx::impl::rt_detail {

// A point keyed by name at run time: any range of (name, value), which is the
// map spelling of the names rt::var(name) reads.  Entry keys off a FixedString
// and so cannot say a name the program only learns while it runs.
template <typename R, typename T>
concept CNamedRange =
    std::ranges::input_range<R> &&
    requires(const std::ranges::range_value_t<R> &e) {
      { e.first } -> std::convertible_to<std::string_view>;
      requires Numeric<std::remove_cvref_t<decltype(e.second)>>;
    };

template <typename A, typename T>
concept CPointArg =
    Numeric<std::remove_cvref_t<A>> || CEntry<A> || CNamedRange<A, T> ||
    (std::ranges::input_range<A> &&
     Numeric<std::remove_cvref_t<std::ranges::range_value_t<A>>>);

// CPointArg admits any input_range, so without this a span of columns would try
// to become a scalar.
template <typename R, typename Ptr>
concept CColumns = std::ranges::contiguous_range<R> &&
                   std::convertible_to<std::ranges::range_value_t<R>, Ptr>;

#ifdef DDX_HAS_JIT
// One LLJIT for the process -- a static in a class template would be per
// specialisation -- reached only from a compiling backend, so an interpret-only
// program never brings LLVM up.
[[nodiscard]] inline jit::Compiler *shared_compiler() {
  static jit::result<jit::Compiler> instance = jit::Compiler::create();
  return instance ? &*instance : nullptr;
}
#endif

#ifdef DDX_HAS_OPENCL
// One device per selector for the process, as shared_compiler() is one LLJIT;
// null where none answers the selector, and cl::Device::shared() says why.
[[nodiscard]] inline const cl::Device *
shared_device(std::string_view selector) {
  const auto &device = cl::Device::shared(selector);
  return device ? &*device : nullptr;
}
#endif

} // namespace ddx::impl::rt_detail

namespace ddx::rt::detail {

// What a sweep gives up around itself where the caller holds nothing.
struct Nothing {};

template <typename C> class Restricted;

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
// Lock order: a lane's lock, then the arena's.  A sweep appends to the arena,
// so it writes; an interpreter reads.  Nothing under the arena's lock touches a
// lane, and nothing under either reacquires what a Yield gave up.
struct Sweeps {
  std::shared_mutex arena;
  Lazy<std::vector<Hessian>> hessians;
  Lazy<HessianVector> hvp;
  Lazy<VectorJacobian> vjp;
  Lazy<Tangent> jvp;
  Lazy<DenseHessian> dense;
  Lazy<std::vector<NodeId>> tangent; // one forward sweep
  Lazy<std::vector<Step>> walk;      // a system's hessian() schedule
#ifdef DDX_HAS_JIT
  Lazy<Rebalanced> rebalanced;
#endif
};

// An equation: an arena of its own, its roots, the Jacobian swept up front and
// every other sweep on first ask, a lane per Want.  Equation, System and the
// Python Equation are this plus the spellings their callers want.  `Yield` is
// what a sweep gives up while it runs: the GIL, on the Python side.
template <impl::Numeric T, std::default_initializable Yield = Nothing>
class Core {
  // Freezes its own lanes through checked(), over cells of this Jacobian.
  friend class Restricted<Core>;

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
  template <impl::rt_detail::CPointArg<T>... Args>
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
                         (impl::rt_detail::CNamedRange<Args, T> && ...)) {
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

  // Dense row-major m x n; at m == 1 the leading axis goes and this is n long.
  // The graph holds only the cells the pattern names, so the zeros go back in
  // here and this spelling never meets the compression.
  [[nodiscard]] constexpr result<std::vector<T>>
  jacobian(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return at<Want::Jacobian>({}, args...).transform([this](const auto &b) {
      return dense(derivative_.pattern, b.jacobian);
    });
  }

  // The Jacobian alone, from a graph with no value block: whatever only the
  // value needs is not swept, which on a model whose derivative is simpler
  // than its function is most of the graph.
  [[nodiscard]] constexpr result<std::vector<T>>
  gradient(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return at<Want::Gradient>({}, args...).transform([this](const auto &b) {
      return dense(derivative_.pattern, b.jacobian);
    });
  }

  // H(x)v, n long.  The direction leads and is a span, so it cannot be read as
  // a point value however the point itself is spelled.
  [[nodiscard]] result<std::vector<T>>
  hvp(std::span<const T> direction,
      const impl::rt_detail::CPointArg<T> auto &...args) const {
    return at<Want::Hvp>(direction, args...).transform([](auto b) {
      return std::move(b.hessian);
    });
  }

  // The j-th Hessian column, named and not indexed: the symbol list exists
  // only at run time.
  [[nodiscard]] result<std::vector<T>>
  hvp(std::string_view along,
      const impl::rt_detail::CPointArg<T> auto &...args) const {
    return unit(along).and_then(
        [&](const auto &e) { return hvp(std::span<const T>{e}, args...); });
  }

  // w'J, n long: one weight per function.
  [[nodiscard]] result<std::vector<T>>
  vjp(std::span<const T> weights,
      const impl::rt_detail::CPointArg<T> auto &...args) const {
    return at<Want::Vjp>(weights, args...).transform([](auto b) {
      return std::move(b.jacobian);
    });
  }

  // J v, m long: the mirror of vjp().
  [[nodiscard]] result<std::vector<T>>
  jvp(std::span<const T> direction,
      const impl::rt_detail::CPointArg<T> auto &...args) const {
    return at<Want::Jvp>(direction, args...).transform([](auto b) {
      return std::move(b.jacobian);
    });
  }

  // dJ/dv, dense row-major m x n: the derivative of every Jacobian cell along
  // `direction`, one forward sweep however many functions there are.
  [[nodiscard]] result<std::vector<T>>
  jacobian_jvp(std::span<const T> direction,
               const impl::rt_detail::CPointArg<T> auto &...args) const {
    return at<Want::JacobianTangent>(direction, args...)
        .transform([this](const auto &b) {
          return dense(derivative_.pattern, b.hessian);
        });
  }
  // Along one symbol's basis vector: d/dx_j of the whole Jacobian.
  [[nodiscard]] result<std::vector<T>>
  jacobian_jvp(std::string_view along,
               const impl::rt_detail::CPointArg<T> auto &...args) const {
    return unit(along).and_then([&](const auto &e) {
      return jacobian_jvp(std::span<const T>{e}, args...);
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
  residual(const impl::rt_detail::CPointArg<T> auto &...args) const {
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
  hessian(const impl::rt_detail::CPointArg<T> auto &...args) const {
    if (!poisoned() && is_system()) {
      // Off the arena, one output's Hessian at a time: no lane holds m
      // colourings.  The gate is the lane's, asked without one.
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
          hessians().front().coloring,
          [&b](std::size_t k) { return &b.hessian[k]; },
          [&out](std::size_t c) { return &out[c]; }, n, 1);
      return out;
    });
  }

  // The same n x n, dense and row-major, from one O(n^2) sweep: the gradient's
  // cone walked with matrix tangents, where hessian() colours and sweeps n + 1
  // times over a dense term.  One output only.
  [[nodiscard]] result<std::vector<T>>
  dense_hessian(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return orders(Want::DenseHessian).and_then([&] {
      return at<Want::DenseHessian>({}, args...)
          .transform([this](const auto &b) {
            // The block is column-major, H(i, j) at i + j n.
            const std::size_t n = symbol_count();
            std::vector<T> out(n * n);
            for (std::size_t i = 0; i < n; ++i) {
              for (std::size_t j = 0; j < n; ++j) {
                out[i * n + j] = b.hessian[i + j * n];
              }
            }
            return out;
          });
    });
  }
  // How many terms the sweep kept quadratic, and how many fell back to a dense
  // product: a model's structure defeated the algebra where the second is not
  // zero -- still right, not O(n^2).
  struct DenseCost {
    std::size_t n2_terms = 0;
    std::size_t n3_terms = 0;
  };
  [[nodiscard]] std::optional<DenseCost> dense_hessian_cost() const {
    return unless_system([this] {
      const DenseHessian &d = dense_sweep();
      return DenseCost{.n2_terms = d.n2_terms, .n3_terms = d.n3_terms};
    });
  }

  // Whether to compile at all, and how; discards anything already compiled.
  // The one non-const member: a call overlapping it is the caller's race.
  auto &options(this auto &self, const jit::Options &opt) {
    if (opt == self.options_) {
      return self;
    }
    self.options_ = opt;
    self.settle_device();
    self.lanes_ = std::make_unique<Lanes>();
    // Choosing a backend starts the build, so it overlaps whatever the caller
    // does before their first call.
    if (opt.backend != jit::Backend::Interpret && !self.poisoned()) {
      (void)self.lane(Want::Jacobian);
    }
    return self;
  }
  [[nodiscard]] const jit::Options &options() const noexcept {
    return options_;
  }

  // What computes the tensor nodes: the process default (rt::use_kernels)
  // unless this equation is handed a table of its own.  A table decides the
  // last bits, so the lanes go with the old one.
  auto &kernels(this auto &self, const KernelTable &table) {
    if (&table == self.kernels_) {
      return self;
    }
    self.kernels_ = &table;
    self.lanes_ = std::make_unique<Lanes>();
    return self;
  }
  [[nodiscard]] constexpr const KernelTable &tensor_kernels() const noexcept {
    if (kernels_ != nullptr) {
      return *kernels_;
    }
    // The process default lives in libddx, which constant evaluation cannot
    // reach; a constant-evaluated equation computes no tensor row anyway.
    if consteval {
      return reference_table;
    }
    return rt::kernels();
  }

  // Sweep and freeze `wants` now, so the calls that follow -- from any number
  // of threads -- find their lanes built and never take a write lock.
  [[nodiscard]] result<void> prepare(std::initializer_list<Want> wants) const {
    return std::ranges::fold_left(
        wants, result<void>{}, [this](result<void> ok, Want want) {
          return ok.and_then([&] { return ready(want); }).transform([&] {
            (void)lane(want);
          });
        });
  }

  // Whether a batch call goes through compiled code: false with no backend,
  // on a refused compile, and under Interpret.
  [[nodiscard]] bool uses_kernel() const {
    return !poisoned() && static_cast<bool>(lane(Want::Jacobian)->kernel);
  }

  // Which rung is answering, as its codegen level, and nothing where the sweep
  // or a device is.  Every rung gives the same bits, so this is the only way
  // to see it climb.  Does not wait.
  [[nodiscard]] std::optional<jit::Level> kernel_level() const {
    if (poisoned()) {
      return std::nullopt;
    }
    if (const auto l = lane(Want::Jacobian);
        l->kernel.jit_kernel() != nullptr) {
      return l->level;
    } else {
      return std::nullopt;
    }
  }

  // Block until the *first* rung of `want`'s lane lands -- rungs[0], the cheap
  // compile -- and answer whether a kernel answers that lane.  Nothing else
  // waits.
  //
  // Under Adapt it waits for a rung already bought and buys none: a call that
  // quietly overrode the counter would make every measurement of the policy a
  // lie.  A caller who wants a kernel now asks for Backend::Compile.
  [[nodiscard]] bool wait_for_kernel(Want want = Want::Jacobian) const {
    if (poisoned()) {
      return false;
    }
    (void)lane(want); // launches it, if nothing has yet
    if (const auto first = lane_for(want).pending(); first.valid()) {
      [[maybe_unused]] const Yield yielded{};
      first.wait();
    }
    return static_cast<bool>(lane(want)->kernel);
  }

  // How far the Jacobian lane is toward its next rung, and nothing under any
  // other backend or once there is none left.  Without it a lane that is still
  // counting and one whose compile was refused both read as no kernel.
  struct Warmup {
    std::size_t points;    // batch points this lane has been asked for
    std::size_t threshold; // where the next rung is bought
  };
  [[nodiscard]] std::optional<Warmup> warming() const {
    if (poisoned() || options_.backend != jit::Backend::Adapt) {
      return std::nullopt;
    }
    const Lane &l = lane_for(Want::Jacobian);
    const unsigned asked = l.asked();
    return asked >= Lane::ladder()
               ? std::nullopt
               : std::optional{Warmup{
                     .points = l.points(),
                     .threshold = Lane::rung_at(asked, effective_options())}};
  }

  // Under Backend::Device: the device answering, or why none is -- a selector
  // nothing matched, or a kernel its compiler refused, with the build log.
  // Nothing under any other backend.  Does not wait.
  [[nodiscard]] std::optional<jit::result<std::string_view>>
  device_status() const {
    if (options_.backend != jit::Backend::Device) {
      return std::nullopt;
    }
    if (poisoned()) {
      return std::unexpected{jit::error{bad_->code, {}}};
    }
#ifdef DDX_HAS_OPENCL
    const auto &device = cl::Device::shared(options_.device);
    if (!device) {
      return std::unexpected{device.error()};
    }
    (void)lane(Want::Jacobian); // adopts a compile that has landed
    if (auto why = lane_for(Want::Jacobian).refused()) {
      return std::unexpected{*std::move(why)};
    }
    return device->identity();
#else
    return std::unexpected{
        jit::error{errc::no_device, "built without DDX_BUILD_OPENCL"}};
#endif
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
      Taylor seed;
      seed.c[0] = x0;
      seed.c[1] = T{1};

      const std::array<Taylor, 1> at{seed};
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

  // Its own lane, not jacobian()'s with the partials dropped: on a coupled
  // model the gradient is most of the graph.
  [[nodiscard]] result<void>
  evaluate(const impl::rt_detail::CColumns<const T *> auto &xs,
           const impl::rt_detail::CColumns<T *> auto &f, std::size_t n) const {
    return dispatch(Want::Value, as_columns(xs), {.values = as_columns(f)}, n);
  }

  [[nodiscard]] result<void>
  jacobian(const impl::rt_detail::CColumns<const T *> auto &xs,
           const impl::rt_detail::CColumns<T *> auto &f,
           const impl::rt_detail::CColumns<T *> auto &g, std::size_t n) const {
    return dispatch(Want::Jacobian, as_columns(xs),
                    {.values = as_columns(f), .jacobian = as_columns(g)}, n);
  }

  [[nodiscard]] result<void>
  gradient(const impl::rt_detail::CColumns<const T *> auto &xs,
           const impl::rt_detail::CColumns<T *> auto &g, std::size_t n) const {
    return dispatch(Want::Gradient, as_columns(xs), {.jacobian = as_columns(g)},
                    n);
  }

  [[nodiscard]] result<void>
  hessian(const impl::rt_detail::CColumns<const T *> auto &xs,
          const impl::rt_detail::CColumns<T *> auto &f,
          const impl::rt_detail::CColumns<T *> auto &g,
          const impl::rt_detail::CColumns<T *> auto &h, std::size_t n) const {
    return dispatch(Want::Hessian, as_columns(xs),
                    {.values = as_columns(f),
                     .jacobian = as_columns(g),
                     .hessian = as_columns(h)},
                    n);
  }

  // `v` is a second block of input columns, not a second point: one column per
  // symbol, each n long, spliced behind `xs`.  `g` comes out of the same sweep.
  [[nodiscard]] result<void>
  hvp(const impl::rt_detail::CColumns<const T *> auto &xs,
      const impl::rt_detail::CColumns<const T *> auto &v,
      const impl::rt_detail::CColumns<T *> auto &f,
      const impl::rt_detail::CColumns<T *> auto &g,
      const impl::rt_detail::CColumns<T *> auto &hv, std::size_t n) const {
    return dispatch(Want::Hvp, as_columns(widen(xs, v)),
                    {.values = as_columns(f),
                     .jacobian = as_columns(g),
                     .hessian = as_columns(hv)},
                    n);
  }

  // `h` is n^2 columns, column-major: H(i, j) is column i + j n.
  [[nodiscard]] result<void>
  dense_hessian(const impl::rt_detail::CColumns<const T *> auto &xs,
                const impl::rt_detail::CColumns<T *> auto &f,
                const impl::rt_detail::CColumns<T *> auto &g,
                const impl::rt_detail::CColumns<T *> auto &h,
                std::size_t n) const {
    return dispatch(Want::DenseHessian, as_columns(xs),
                    {.values = as_columns(f),
                     .jacobian = as_columns(g),
                     .hessian = as_columns(h)},
                    n);
  }

  [[nodiscard]] result<void>
  vjp(const impl::rt_detail::CColumns<const T *> auto &xs,
      const impl::rt_detail::CColumns<const T *> auto &w,
      const impl::rt_detail::CColumns<T *> auto &f,
      const impl::rt_detail::CColumns<T *> auto &out, std::size_t n) const {
    return dispatch(Want::Vjp, as_columns(widen(xs, w)),
                    {.values = as_columns(f), .jacobian = as_columns(out)}, n);
  }

  [[nodiscard]] result<void>
  jvp(const impl::rt_detail::CColumns<const T *> auto &xs,
      const impl::rt_detail::CColumns<const T *> auto &v,
      const impl::rt_detail::CColumns<T *> auto &f,
      const impl::rt_detail::CColumns<T *> auto &out, std::size_t n) const {
    return dispatch(Want::Jvp, as_columns(widen(xs, v)),
                    {.values = as_columns(f), .jacobian = as_columns(out)}, n);
  }

  // `g` and `dg` both have jacobian_columns() columns and read with
  // jacobian_pattern(): the Jacobian, and its derivative along `v`.
  [[nodiscard]] result<void>
  jacobian_jvp(const impl::rt_detail::CColumns<const T *> auto &xs,
               const impl::rt_detail::CColumns<const T *> auto &v,
               const impl::rt_detail::CColumns<T *> auto &f,
               const impl::rt_detail::CColumns<T *> auto &g,
               const impl::rt_detail::CColumns<T *> auto &dg,
               std::size_t n) const {
    return dispatch(Want::JacobianTangent, as_columns(widen(xs, v)),
                    {.values = as_columns(f),
                     .jacobian = as_columns(g),
                     .hessian = as_columns(dg)},
                    n);
  }

  // What a caller sizes buffers by.  None builds a lane, and only the
  // Hessian's sweeps anything: its colouring is what fixes the count.
  [[nodiscard]] constexpr std::optional<std::size_t> value_columns() const {
    return outputs();
  }
  [[nodiscard]] constexpr std::optional<std::size_t> jacobian_columns() const {
    return unless_poisoned([this] { return derivative_.partial.size(); });
  }
  [[nodiscard]] std::optional<std::size_t> hessian_columns() const {
    return unless_system(
        [this] { return hessians().front().compressed.size(); });
  }
  [[nodiscard]] std::optional<std::size_t> hessian_colors() const {
    return unless_system([this] { return hessians().front().colors(); });
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
    return unless_poisoned([this] { return std::cref(derivative_.pattern); });
  }

  // Which (row, column) cell each Hessian column holds: Coloring::entries().
  // One output only, and it sweeps the Hessian on first ask.
  [[nodiscard]] std::optional<std::reference_wrapper<const Coloring>>
  hessian_pattern() const {
    return unless_system(
        [this] { return std::cref(hessians().front().coloring); });
  }

  // Where H(i, j) sits, or nullopt for a cell the colouring calls zero.
  [[nodiscard]] std::optional<std::size_t> hessian_cell(std::size_t i,
                                                        std::size_t j) const {
    if (poisoned() || is_system() || i >= symbol_count() ||
        j >= symbol_count()) {
      return std::nullopt;
    }
    return hessians().front().coloring.cell_of(i, j);
  }

protected:
  using Compiled = detail::Compiled<T>;
  using Lane = detail::Lane<T>;
  using Lanes = std::array<Lane, want_count>;

  // Poisoned: arena_ null, roots_ empty, every accessor short-circuiting first.
  constexpr explicit Core(error why) : bad_(why) { settle(); }

  // The arena is the equation's own from here: one it was lent is copied, so
  // nothing another equation or the caller builds can move a node under it.
  constexpr Core(std::unique_ptr<Builder<T>> arena, std::vector<NodeId> roots)
      : arena_(std::move(arena)), roots_(std::move(roots)) {
    // An implicit no derivative could carry refuses the whole equation, as a
    // poisoned output would; so does a vector where an output must be a
    // scalar -- spell a system over its elements.
    if (const auto why = unclosed(*arena_)) {
      poison(error{*why});
    } else if (std::ranges::any_of(roots_, [this](NodeId r) {
                 return tensor_valued(arena_->op_of(r));
               })) {
      poison(error{errc::tensor_shape});
    } else {
      // One reverse sweep per output is cheap, and it is what the column
      // counts, the pattern and constant evaluation read.
      derivative_ = build_jacobian_impl(*arena_, roots_);
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
  // Every equation's tail: what the gates read of the arena, and the lanes and
  // sweeps, which constant evaluation never reaches.
  constexpr void settle() {
    if (arena_) {
      first_order_ = first_order_only(*arena_, roots_);
    }
    if !consteval {
      lanes_ = std::make_unique<Lanes>();
      sweeps_ = std::make_unique<Sweeps>();
      settle_device();
    }
  }

  // Poison first, and by the code it carries: a clashing name or an absent
  // arena is a different mistake from a literal that reached no graph.
  [[nodiscard]] static constexpr std::optional<error>
  why_not(std::span<const RTExpression<T>> outputs) noexcept {
    for (const RTExpression<T> &e : outputs) {
      if (const auto why = Rooting::why(e)) {
        return error{*why};
      }
    }
    if (outputs.empty() || Rooting::builder(outputs.front()) == nullptr) {
      return error{errc::no_graph};
    }
    return std::nullopt;
  }

  // The arena the outputs will be read in -- `owned`, or a copy of the one
  // they were built in -- and their roots there.
  [[nodiscard]] static constexpr std::pair<std::unique_ptr<Builder<T>>,
                                           std::vector<NodeId>>
  rooted(std::span<const RTExpression<T>> outputs,
         std::unique_ptr<Builder<T>> owned) {
    auto arena =
        owned
            ? std::move(owned)
            : std::make_unique<Builder<T>>(*Rooting::builder(outputs.front()));
    auto roots = outputs | std::views::transform([&arena](const auto &e) {
                   return e.id(*arena);
                 }) |
                 impl::to<std::vector<NodeId>>();
    return {std::move(arena), std::move(roots)};
  }

  // The three ways an equation comes to be, for whichever class wraps this.
  template <typename D>
  [[nodiscard]] static constexpr D
  make(std::span<const RTExpression<T>> outputs,
       std::unique_ptr<Builder<T>> owned) {
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
    return derivative_;
  }

  // Which Jacobian cells a set of lanes is frozen over: every one, for the
  // equation's own, or the cells a restrict() kept, as indices into them.
  struct Cells {
    const Jacobian &jacobian;
    // Unset is all of them; a restriction can keep none.
    std::optional<std::span<const std::size_t>> kept{};
  };
  [[nodiscard]] constexpr Cells every_cell() const {
    return {.jacobian = derivative_};
  }

  // The lane for `want`, frozen on first ask.  `n` is the batch about to run
  // and is what Adapt counts; an observer asks without buying anything.
  [[nodiscard]] std::shared_ptr<const Compiled> lane(Want want,
                                                     std::size_t n = 0) const {
    return lane_in(*lanes_, every_cell(), want, n);
  }
  [[nodiscard]] std::shared_ptr<const Compiled>
  lane_in(Lanes &lanes, const Cells &cells, Want want, std::size_t n) const {
    return lanes[std::to_underlying(want)].snapshot(
        want, n, setting(), [&] { return frozen_for(want, cells); });
  }

  [[nodiscard]] result<void> dispatch(Want want, std::span<const T *const> xs,
                                      const Columns<T> &out,
                                      std::size_t n) const {
    return dispatch_in(*lanes_, every_cell(), want, xs, out, n);
  }
  // Through a restrict() view's own lanes and cells, or the equation's.
  [[nodiscard]] result<void> dispatch_in(Lanes &lanes, const Cells &cells,
                                         Want want,
                                         std::span<const T *const> xs,
                                         const Columns<T> &out,
                                         std::size_t n) const {
    return checked(lanes, cells, want, xs, out, n)
        .transform([&](const std::shared_ptr<const Compiled> &c) {
          run(*c, xs, out, n);
        });
  }

  // The lane a call runs on, once the call is known to fit it.
  [[nodiscard]] result<std::shared_ptr<const Compiled>>
  checked(Lanes &lanes, const Cells &cells, Want want,
          std::span<const T *const> xs, const Columns<T> &out,
          std::size_t n) const {
    return ready(want).and_then(
        [&]() -> result<std::shared_ptr<const Compiled>> {
          auto c = lane_in(lanes, cells, want, n);
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
  // carries one colouring, so no lane holds m of them.  `f` is m long, `g` the
  // dense m x n and `h` the dense m x n x n.
  void walk(std::span<const T> at, std::span<T> f, std::span<T> g,
            std::span<T> h) const {
    const auto &blocks = hessians();
    // What the three blocks read -- values, partials and Hessian::at's
    // compressed cells -- settled on first ask, since it depends only on the
    // sweeps.
    const auto &walk = sweeps_->walk.get([&] {
      auto wanted = blocks | std::views::transform(&Hessian::compressed) |
                    std::views::join | impl::to<std::vector<NodeId>>();
      impl::append(wanted, blocks | std::views::transform(&Hessian::partial) |
                               std::views::join);
      impl::append(wanted, blocks | std::views::transform(&Hessian::zero));
      impl::append(wanted, roots_);
      [[maybe_unused]] const auto held = reading();
      return schedule_of(*arena_, live_ids(reachable(*arena_, wanted)));
    });
    const std::size_t n = symbol_count();
    // The tape is thread-kept; every id read below is scheduled, so it is
    // written this call.
    static thread_local std::vector<T> values;
    [[maybe_unused]] const auto held = sweeping();
    values.resize(arena_->size());
    detail::TensorScratch<T> scratch{*arena_, walk, tensor_kernels()};
    evaluate_into(*arena_, at, walk, std::span<T>{values}, scratch.view());
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
  values(const impl::rt_detail::CPointArg<T> auto &...args) const {
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
               return build_hessian_impl(*arena_, r);
             }) |
             impl::to<std::vector<Hessian>>();
    });
  }
  // Root 0's only: a system refuses hvp before it gets here.
  [[nodiscard]] const HessianVector &hvp_sweep() const {
    return sweep(sweeps_->hvp,
                 [this] { return build_hvp_impl(*arena_, roots_.front()); });
  }
  [[nodiscard]] const DenseHessian &dense_sweep() const {
    return sweep(sweeps_->dense, [this] {
      return build_dense_hessian_impl(*arena_, roots_.front());
    });
  }
  [[nodiscard]] const VectorJacobian &vjp_sweep() const {
    return sweep(sweeps_->vjp,
                 [this] { return build_vjp_impl(*arena_, roots_); });
  }
  [[nodiscard]] const Tangent &jvp_sweep() const {
    return sweep(sweeps_->jvp,
                 [this] { return build_jvp_impl(*arena_, roots_); });
  }
  [[nodiscard]] const std::vector<NodeId> &tangent_sweep() const {
    return sweep(sweeps_->tangent, [this] {
      return build_jacobian_tangent_impl(*arena_, derivative_);
    });
  }

  // The symbol named `name`, by slot: a parameter or an implicit is an input,
  // not a symbol.
  [[nodiscard]] result<std::uint32_t> symbol(std::string_view name) const {
    const auto input = arena_->lookup(name);
    return input && input->op == OpCode::Var
               ? result<std::uint32_t>{input->slot}
               : fail(errc::unknown_symbol);
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

  // The sweep `want`'s freeze reads, built before the freeze takes the lock.
  void ensure(Want want) const {
    switch (want) {
    case Want::Hessian:
      (void)hessians();
      break;
    case Want::Hvp:
      (void)hvp_sweep();
      break;
    case Want::Vjp:
      (void)vjp_sweep();
      break;
    case Want::Jvp:
      (void)jvp_sweep();
      break;
    case Want::JacobianTangent:
      (void)tangent_sweep();
      break;
    case Want::DenseHessian:
      (void)dense_sweep();
      break;
    case Want::Value:
    case Want::Jacobian:
    case Want::Gradient:
      break;
    }
  }

protected:
  // The gates, in the order a call meets them.  alive(): not poisoned.
  // single(): one function's, which a system refuses.  orders(): second order
  // is refused where a reachable extern supplies first partials only, rather
  // than answered with the zeros a missing component would read.  ready() is
  // the one gate every batch and point call passes, off the want's row.
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
  [[nodiscard]] constexpr result<void> ready(Want want) const {
    return (want_info(want).single ? single() : alive()).and_then([&] {
      return orders(want);
    });
  }

private:
  [[nodiscard]] auto reading() const {
    return std::shared_lock{sweeps_->arena};
  }
  [[nodiscard]] auto writing() const {
    return std::unique_lock{sweeps_->arena};
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
    const auto count = [&](Block shape) -> std::size_t {
      switch (shape) {
      case Block::None:
        return 0;
      case Block::Cells:
        return cells.kept ? cells.kept->size() : cells.jacobian.partial.size();
      case Block::Symbols:
        return n;
      case Block::Functions:
        return m;
      case Block::Colours:
        return hessians().front().compressed.size();
      case Block::Dense:
        return n * n;
      }
      std::unreachable();
    };
    const WantInfo &w = want_info(want);
    return {.values = w.values ? m : 0,
            .jacobian = count(w.jacobian),
            .hessian = count(w.hessian)};
  }

  // One point of `W`: the blocks sized by the table, and where the lane takes
  // a direction -- one value per symbol or per function, as the seeds say --
  // the point widened by it.  `lanes` and `cells` are the equation's or a
  // restrict() view's; constant evaluation takes gather()'s arena walk.
  template <Want W>
  [[nodiscard]] constexpr result<Blocks<std::vector<T>>>
  at_in(Lanes *lanes, const Cells &cells, std::span<const T> along,
        const impl::rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).and_then(
        [&](const std::vector<T> &pt) -> result<Blocks<std::vector<T>>> {
          constexpr Seeds seeds = want_info(W).seeds;
          if constexpr (seeds != Seeds::None) {
            if (along.size() !=
                (seeds == Seeds::Functions ? roots_.size() : symbol_count())) {
              return fail(errc::wrong_direction);
            }
          }
          const Layout size = sizes(W, cells);
          Blocks<std::vector<T>> out{.values = std::vector<T>(size.values),
                                     .jacobian = std::vector<T>(size.jacobian),
                                     .hessian = std::vector<T>(size.hessian)};
          const auto swept = [&] {
            if constexpr (seeds == Seeds::None) {
              return gather(lanes, cells, W, pt, out.values, out.jacobian,
                            out.hessian);
            } else {
              return gather(lanes, cells, W, widen(pt, along), out.values,
                            out.jacobian, out.hessian);
            }
          }();
          return swept.transform([&] { return std::move(out); });
        });
  }
  template <Want W>
  [[nodiscard]] constexpr result<Blocks<std::vector<T>>>
  at(std::span<const T> along,
     const impl::rt_detail::CPointArg<T> auto &...args) const {
    return at_in<W>(lanes_.get(), every_cell(), along, args...);
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
    const auto column = input_column(arena_->inputs(), input->op, input->slot);
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
                     const impl::rt_detail::CNamedRange<T> auto &vm) const {
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

  // At one point every column is one value, so a column pointer is its address.
  template <typename U>
  [[nodiscard]] static auto single_columns(std::span<U> values) {
    auto addresses =
        values | std::views::transform([](U &v) { return std::addressof(v); });
    return boost::container::small_vector<U *, 32>(addresses.begin(),
                                                   addresses.end());
  }

  // One point through the batch path at n = 1, not the arena walk, so the
  // batch gate is the point gate.  Constant evaluation keeps it, having no
  // lane and nothing to refuse.
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
  [[nodiscard]] constexpr result<void> gather(Lanes *lanes, const Cells &cells,
                                              Want want, std::span<const T> at,
                                              std::span<T> f, std::span<T> g,
                                              std::span<T> h) const {
    if !consteval {
      const auto xs = single_columns(at);
      const auto fs = single_columns(f);
      const auto gs = single_columns(g);
      const auto hs = single_columns(h);
      return dispatch_in(*lanes, cells, want, as_columns(xs),
                         {.values = as_columns(fs),
                          .jacobian = as_columns(gs),
                          .hessian = as_columns(hs)},
                         1);
    }
    auto wanted = roots_;
    impl::append(wanted, derivative_.partial);
    const auto values =
        evaluate_reachable(*arena_, wanted, at, tensor_kernels());
    const auto pick = [&values](std::span<T> out, const auto &nodes) {
      std::ranges::transform(nodes | std::views::take(out.size()), out.begin(),
                             [&values](NodeId v) { return values[v]; });
    };
    pick(f, roots_);
    pick(g, derivative_.partial);
    return {};
  }
#ifdef _MSC_VER
#pragma warning(pop)
#endif

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

  [[nodiscard]] Lane &lane_for(Want want) const {
    return (*lanes_)[std::to_underlying(want)];
  }

  // Looked up where the options change -- options() -- so a batch call never
  // resolves a selector.
  void settle_device() {
#ifdef DDX_HAS_OPENCL
    device_ = options_.backend == jit::Backend::Device
                  ? impl::rt_detail::shared_device(options_.device)
                  : nullptr;
#endif
  }

  // What a lane needs of the equation.  No compiler is a lane that will never
  // compile: a poisoned equation and an interpreting backend both arrive here
  // as one, and so does a host with no JIT.
  [[nodiscard]] Setting setting() const {
    Setting out;
    if constexpr (std::same_as<T, double>) {
      out.options = effective_options();
      if (!bad_) {
#ifdef DDX_HAS_JIT
        if (options_.backend == jit::Backend::Compile ||
            options_.backend == jit::Backend::Adapt) {
          out.compiler = compiler();
        }
#endif
#ifdef DDX_HAS_OPENCL
        out.device = device_;
#endif
      }
    }
    return out;
  }

  // What a freeze asks for.  `ids` and `partials` differ between the swept
  // graph and the kernel's rebalanced one; the rest cannot be rebalanced.  Read
  // only after ensure() has built what the want names.
  struct Frozen {
    const Core &core;
    std::span<const NodeId> ids;
    const Jacobian &partials;
    std::span<const NodeId> product{}; // dJ/dv over `partials`' cells

    [[nodiscard]] std::span<const NodeId> roots() const { return ids; }
    [[nodiscard]] const Jacobian &jacobian() const { return partials; }
    [[nodiscard]] const Hessian &hessian() const {
      return core.sweeps_->hessians.peek().front();
    }
    [[nodiscard]] const HessianVector &hessian_vector() const {
      return core.sweeps_->hvp.peek();
    }
    [[nodiscard]] const VectorJacobian &vector_jacobian() const {
      return core.sweeps_->vjp.peek();
    }
    [[nodiscard]] const Tangent &tangent() const {
      return core.sweeps_->jvp.peek();
    }
    [[nodiscard]] std::span<const NodeId> jacobian_tangent() const {
      return product;
    }
    [[nodiscard]] const DenseHessian &dense_hessian() const {
      return core.sweeps_->dense.peek();
    }
  };

  // The graphs a lane computes.  A poisoned equation freezes an empty one:
  // every accessor short-circuits before it, and dispatch() refuses on `bad_`.
  [[nodiscard]] Compiled frozen_for(Want want, const Cells &cells) const {
    if (bad_) {
      return {.graph = std::make_shared<const Graph<T>>()};
    }
    ensure(want);
    // A restriction keeps the cells it chose, of the product too; only the
    // whole Jacobian has rebalanced roots to compile from.
    std::vector<NodeId> product;
    if (want == Want::JacobianTangent) {
      const auto &tangent = sweeps_->tangent.peek();
      product =
          cells.kept
              ? *cells.kept | std::views::transform([&tangent](std::size_t k) {
                  return tangent[k];
                }) | impl::to<std::vector<NodeId>>()
              : tangent;
    }
    [[maybe_unused]] const bool rebalance = !cells.kept && rebalances(want);
    [[maybe_unused]] const auto held = reading();
    Frozen from{*this, roots_, cells.jacobian, product};
    auto swept = std::make_shared<const Graph<T>>(
        freeze_for(want, *arena_, from, contracts()));
    auto compiled = swept;
#ifdef DDX_HAS_JIT
    if (rebalance) {
      const Rebalanced &over = sweeps_->rebalanced.peek();
      Frozen rebalanced{*this, over.roots, over.derivative};
      compiled = std::make_shared<const Graph<T>>(
          freeze_for(want, *arena_, rebalanced, contracts()));
    }
#endif
    return {.graph = std::move(swept),
            .compile_graph = std::move(compiled),
            .kernels = &tensor_kernels()};
  }

  // Whether the kernel for `want` is lowered from rebalanced roots, which are
  // swept here on first ask: only a lane that compiles reads them.
  [[nodiscard]] bool rebalances([[maybe_unused]] Want want) const {
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      // No kernel is lowered over tensor rows, so no rebalanced copy either.
      if (!rebalanceable(want) || setting().compiler == nullptr ||
          std::ranges::any_of(arena_->nodes(), [](const Node<T> &n) {
            return kind_of(n.op) == Kind::Tensor;
          })) {
        return false;
      }
      (void)sweep(sweeps_->rebalanced, [this] {
        Rebalanced over{.roots = detail::rebalance(*arena_, roots_),
                        .derivative = {}};
        over.derivative = build_jacobian_impl(*arena_, over.roots);
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
    return impl::rt_detail::shared_compiler();
  }
#endif

  // Decided in the *graph*, so every backend folds the same products.  On by
  // default in a JIT-less build as well -- DDX_JIT_DEFAULT_CONTRACT is the
  // JIT's -- so an answer does not depend on how the library was configured.
  [[nodiscard]] bool contracts() const noexcept {
    return options_.codegen.contract;
  }

  // A kernel that could not run this call -- a device refusing a launch --
  // hands it back, and the sweep answers.
  void run(const Compiled &c, std::span<const T *const> xs,
           const Columns<T> &out, std::size_t n) const {
    [[maybe_unused]] const auto held = sweeping();
    if constexpr (std::same_as<T, double>) {
      if (c.kernel(xs, out.values, out.jacobian, out.hessian, n)) {
        return;
      }
    }
    interpret(*c.graph, *c.kernels, xs, out, n);
  }

  // The block sweep, within ~1.4x of a kernel and not a sad path.
  void interpret(const Graph<T> &graph, const KernelTable &kernels,
                 std::span<const T *const> xs, const Columns<T> &out,
                 std::size_t n) const {
    const auto blocks = graph.output_rows();
    const auto schedule = graph.schedule();
    // Both tapes are sized by the graph, whose rows are packed by liveness:
    // the schedule names rows, so a grown arena changes nothing here.
    const TensorLayout &rows = graph.tensors();
    const auto tensors = [&](std::vector<T> &vtape, std::size_t lanes) {
      vtape.resize(rows.extent * lanes);
      return detail::Scratch<T>{.slots = graph.slots(),
                                .rows = rows.rows,
                                .vtape = vtape.data(),
                                .kernels = &kernels};
    };
    // The graph's width, not the arena's symbol count: `xs` carries one column
    // per input, and a lane may hold leaves that are not symbols.
    const std::size_t inputs = graph.arity();

    // thread_local rather than a member: two threads may be inside one
    // equation at once.  resize() rather than assign(): every id read is
    // written first, so zeroing the arena per call is waste.
    //
    // A short batch sweeps one point at a time rather than padding out to
    // block_lanes, which on a batch of one is the whole cost.
    if (n < block_lanes) {
      static thread_local std::vector<T> at;
      static thread_local std::vector<T> tape;
      static thread_local std::vector<T> vtape;
      at.resize(inputs);
      tape.resize(graph.tape_slots());
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
    static thread_local std::vector<T> lanes;
    static thread_local std::vector<T> tape;
    static thread_local std::vector<T> vtape;
    lanes.resize(inputs * block_lanes);
    tape.resize(graph.tape_slots() * block_lanes);
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

  std::unique_ptr<Builder<T>> arena_;
  std::vector<NodeId> roots_;
  jit::Options options_{};
  // Null: the process default, read at each freeze.
  const KernelTable *kernels_ = nullptr;
#ifdef DDX_HAS_OPENCL
  // Settled with options_: null unless the backend is Device and one answered.
  const cl::Device *device_ = nullptr;
#endif
  Jacobian derivative_;
  // unique_ptr, not shared: its destructor is constexpr, which is what keeps a
  // constant-evaluated equation destructible.  Null only there.
  std::unique_ptr<Lanes> lanes_;
  // Every other sweep, built once on first ask under the arena's write lock.
  std::unique_ptr<Sweeps> sweeps_;
  // Set exactly when why_not() refused, so poisoned() <=> arena_ == nullptr.
  std::optional<error> bad_{};
  // A reachable extern supplies first partials only; see orders().
  bool first_order_ = false;
};

// restrict()'s answer: the equation's Jacobian narrowed to some symbols'
// columns, in symbol order, with lanes of its own -- frozen over the kept cells
// alone, so the partials of the others are never computed.  It reads the
// equation it came from and must not outlive it.
template <typename C> class Restricted {
  using T = typename C::value_type;
  using Want = rt::Want;

public:
  Restricted(const C &core, const std::vector<bool> &chosen)
      : core_(&core), lanes_(std::make_unique<typename C::Lanes>()) {
    const Jacobian &full = core.derivative_;
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

  // Dense row-major m x k.
  [[nodiscard]] result<std::vector<T>>
  jacobian(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return core_
        ->template at_in<Want::Jacobian>(lanes_.get(), cells(), {}, args...)
        .transform([&](const auto &b) {
          return C::dense(jacobian_.pattern, b.jacobian);
        });
  }

  // dJ/dv over the kept cells, dense row-major m x k; `direction` is still one
  // value per symbol of the equation.
  [[nodiscard]] result<std::vector<T>>
  jacobian_jvp(std::span<const T> direction,
               const impl::rt_detail::CPointArg<T> auto &...args) const {
    return core_
        ->template at_in<Want::JacobianTangent>(lanes_.get(), cells(),
                                                direction, args...)
        .transform([&](const auto &b) {
          return C::dense(jacobian_.pattern, b.hessian);
        });
  }
  [[nodiscard]] result<std::vector<T>>
  jacobian_jvp(std::string_view along,
               const impl::rt_detail::CPointArg<T> auto &...args) const {
    return core_->unit(along).and_then([&](const auto &e) {
      return jacobian_jvp(std::span<const T>{e}, args...);
    });
  }

  // The batch calls, laid out as the equation's are with `g` and `dg` sized
  // by jacobian_columns().
  [[nodiscard]] result<void>
  jacobian(const impl::rt_detail::CColumns<const T *> auto &xs,
           const impl::rt_detail::CColumns<T *> auto &f,
           const impl::rt_detail::CColumns<T *> auto &g, std::size_t n) const {
    return dispatch(Want::Jacobian, C::as_columns(xs),
                    {.values = C::as_columns(f), .jacobian = C::as_columns(g)},
                    n);
  }
  [[nodiscard]] result<void>
  gradient(const impl::rt_detail::CColumns<const T *> auto &xs,
           const impl::rt_detail::CColumns<T *> auto &g, std::size_t n) const {
    return dispatch(Want::Gradient, C::as_columns(xs),
                    {.jacobian = C::as_columns(g)}, n);
  }
  [[nodiscard]] result<void>
  jacobian_jvp(const impl::rt_detail::CColumns<const T *> auto &xs,
               const impl::rt_detail::CColumns<const T *> auto &v,
               const impl::rt_detail::CColumns<T *> auto &f,
               const impl::rt_detail::CColumns<T *> auto &g,
               const impl::rt_detail::CColumns<T *> auto &dg,
               std::size_t n) const {
    return dispatch(Want::JacobianTangent, C::as_columns(C::widen(xs, v)),
                    {.values = C::as_columns(f),
                     .jacobian = C::as_columns(g),
                     .hessian = C::as_columns(dg)},
                    n);
  }

private:
  [[nodiscard]] typename C::Cells cells() const {
    return {.jacobian = jacobian_, .kept = std::span{kept_}};
  }
  [[nodiscard]] result<void> dispatch(Want want, std::span<const T *const> xs,
                                      const Columns<T> &out,
                                      std::size_t n) const {
    return core_->dispatch_in(*lanes_, cells(), want, xs, out, n);
  }

  const C *core_;
  std::vector<std::string> names_;
  Jacobian jacobian_;
  std::vector<std::size_t> kept_; // indices into the equation's cells
  std::unique_ptr<typename C::Lanes> lanes_;
};

// How many functions an equation has, as its type says: a count fixed there
// refuses a file holding another and narrows one function's value to a
// scalar; a run-time count takes any file and answers a vector.
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
  auto arena = std::make_unique<Builder<T>>();
  auto built = [&] {
    const auto scope = scoped_arena(*arena);
    return std::invoke(DDX_FWD(assemble));
  }();
  return std::pair{std::move(arena), std::move(built)};
}

// What Equation and System are over Core: the three ways one comes to be, and
// evaluate(), with `Outputs` the one place the count's policy acts.
template <typename D, impl::Numeric T, typename Outputs>
class Facade : public Core<T> {
  using Base = Core<T>;

public:
  // A refusal rides on the equation rather than a result at every call site;
  // see poisoned().  `owned` is null where the outputs' arena is the caller's,
  // which is then copied.
  [[nodiscard]] static constexpr D
  create(std::unique_ptr<Builder<T>> owned,
         std::span<const RTExpression<T>> outputs) {
    return Base::template make<D>(outputs, std::move(owned));
  }
  [[nodiscard]] constexpr auto
  evaluate(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return this->values(args...).transform([](std::vector<T> f) {
      if constexpr (Outputs::scalar) {
        return f.front();
      } else {
        return f;
      }
    });
  }
  using Base::evaluate;

protected:
  using Base::Base;
};

} // namespace ddx::rt::detail

namespace ddx::rt {
// What Equation::restrict() and System::restrict() answer.
template <typename C> using Restricted = detail::Restricted<C>;
} // namespace ddx::rt
