#pragma once

#include "dual/taylor_dual.hpp" // the univariate sweep's scalar
#include "md/md.hpp"
#include "ops/numeric.hpp" // compile_time_factorial
#include "rt/archive/archive.hpp"
#include "rt/cache.hpp"
#include "rt/coupling.hpp"
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

#include <filesystem>
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

// The arena's lock and one flag per sweep built on first ask.  On the heap:
// neither moves, and constant evaluation never reaches them.
//
// Lock order: a lane's lock, then the arena's.  A sweep appends to the arena,
// so it writes; an interpreter reads.  Nothing under the arena's lock touches a
// lane, and nothing under either reacquires what a Yield gave up.
struct Latches {
  std::shared_mutex arena;
  std::once_flag hessians, hvp, vjp, jvp, tangent, rebalanced, walk;
};

// An equation: an arena of its own, its roots, the Jacobian swept up front and
// every other sweep on first ask, a lane per Want.  Equation, System and the
// Python Equation are this plus the spellings their callers want.  `Yield` is
// what a sweep gives up while it runs: the GIL, on the Python side.
template <impl::Numeric T, CValueCache<T> Cache,
          std::default_initializable Yield = Nothing>
class Core : public Caching<Core<T, Cache, Yield>, T, Cache> {
  using Memo = Caching<Core, T, Cache>;
  // Reaches arena(), arena_size() and sweeping().
  friend Memo;
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
    if (poisoned()) {
      return std::nullopt;
    }
    // Not std::optional{...}: symbols() answers with a const vector&, so CTAD
    // would deduce optional<vector> and span the copy as it died.
    return std::span<const std::string>{arena_->symbols()};
  }
  [[nodiscard]] constexpr std::optional<std::span<const std::string>>
  params() const {
    if (poisoned()) {
      return std::nullopt;
    }
    return std::span<const std::string>{arena_->params()};
  }
  // Everything a point supplies, in the order a positional one is read: the
  // symbols, then the parameters.
  [[nodiscard]] constexpr std::optional<std::size_t> input_count() const {
    return unless_poisoned([this] { return arena_->inputs().named(); });
  }
  [[nodiscard]] std::optional<std::vector<std::string>> inputs() const {
    return unless_poisoned([this] {
      auto names = arena_->symbols();
      impl::append(names, arena_->params());
      impl::append(names, arena_->implicits());
      return names;
    });
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
      const std::array positional{static_cast<T>(args)...};
      ok = assign_range(at, positional);
    }
    return ok.transform([&at] { return std::move(at); });
  }

  // Dense row-major m x n; at m == 1 the leading axis goes and this is n long.
  // The graph holds only the cells the pattern names, so the zeros go back in
  // here and this spelling never meets the compression.
  [[nodiscard]] constexpr result<std::vector<T>>
  jacobian(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      std::vector<T> f(roots_.size());
      std::vector<T> cells(derivative_.partial.size());
      gather(Want::Jacobian, at, f, cells);
      return dense(derivative_.pattern, cells);
    });
  }

  // The Jacobian alone, from a graph with no value block: whatever only the
  // value needs is not swept, which on a model whose derivative is simpler
  // than its function is most of the graph.
  [[nodiscard]] constexpr result<std::vector<T>>
  gradient(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([&](const auto &at) {
      std::vector<T> cells(derivative_.partial.size());
      gather(Want::Gradient, at, {}, cells);
      return dense(derivative_.pattern, cells);
    });
  }

  // H(x)v, n long.  The direction leads and is a span, so it cannot be read as
  // a point value however the point itself is spelled.
  [[nodiscard]] result<std::vector<T>>
  hvp(std::span<const T> direction,
      const impl::rt_detail::CPointArg<T> auto &...args) const {
    return single().and_then(
        [&] { return seeded<Want::Hvp>(direction, args...); });
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
    return seeded<Want::Vjp>(weights, args...);
  }

  // J v, m long: the mirror of vjp().
  [[nodiscard]] result<std::vector<T>>
  jvp(std::span<const T> direction,
      const impl::rt_detail::CPointArg<T> auto &...args) const {
    return seeded<Want::Jvp>(direction, args...);
  }

  // dJ/dv, dense row-major m x n: the derivative of every Jacobian cell along
  // `direction`, one forward sweep however many functions there are.
  [[nodiscard]] result<std::vector<T>>
  jacobian_jvp(std::span<const T> direction,
               const impl::rt_detail::CPointArg<T> auto &...args) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    if (direction.size() != symbol_count()) {
      return fail(errc::wrong_direction);
    }
    return point(args...).transform([&](const auto &at) {
      boost::container::small_vector<T, 64> widened;
      widened.reserve(at.size() + direction.size());
      impl::append(widened, at);
      impl::append(widened, direction);
      std::vector<T> f(roots_.size());
      std::vector<T> g(derivative_.partial.size());
      std::vector<T> dg(derivative_.partial.size());
      gather(Want::JacobianTangent, widened, f, g, dg);
      return dense(derivative_.pattern, dg);
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
    if (bad_) {
      return std::unexpected{*bad_};
    }
    std::vector<bool> chosen(symbol_count(), false);
    for (const std::string_view name : names) {
      const auto input = arena_->lookup(name);
      if (!input || input->op != OpCode::Var) {
        return fail(errc::unknown_symbol);
      }
      chosen[input->slot] = true;
    }
    return Restricted<Core>{*this, chosen};
  }
  [[nodiscard]] result<Restricted<Core>>
  restrict(std::initializer_list<std::string_view> names) const {
    return restrict(std::span{names});
  }

  // F(z, x) for each implicit's closure, at a point: zero, to rounding, where
  // the caller's root is one.  ddx never solves for z; this is the check.
  [[nodiscard]] result<std::vector<T>>
  residual(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      [[maybe_unused]] const auto held = reading();
      const auto closures = arena_->closures();
      const auto values = evaluate_reachable(*arena_, closures, at);
      return closures |
             std::views::transform([&values](NodeId f) { return values[f]; }) |
             impl::to<std::vector<T>>();
    });
  }

  // Dense row-major m x n x n; the graph holds it compressed by colour.
  [[nodiscard]] result<std::vector<T>>
  hessian(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      const std::size_t m = roots_.size();
      const std::size_t n = symbol_count();
      std::vector<T> out(m * n * n);
      if (m == 1) {
        const Coloring &coloring = hessians().front().coloring;
        std::array<T, 1> f{};
        std::vector<T> g(derivative_.partial.size());
        std::vector<T> h(hessians().front().compressed.size());
        gather(Want::Hessian, at, f, g, h);
        // Only the cells the colouring stores; `out` is already zero, which is
        // what every other column of that colour is.
        for (const Cell &cell : coloring.entries()) {
          out[cell.row * n + cell.column] = h[cell.slot];
        }
      } else {
        std::vector<T> f(m);
        std::vector<T> g(m * n);
        walk(at, f, g, out);
      }
      return out;
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
    // codegen decides the arithmetic, so what was remembered was answered by
    // another graph.
    self.cache().clear();
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

  // Sweep and freeze `wants` now, so the calls that follow -- from any number
  // of threads -- find their lanes built and never take a write lock.
  [[nodiscard]] result<void> prepare(std::initializer_list<Want> wants) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    for (const Want want : wants) {
      if (needs_single(want) && roots_.size() != 1) {
        return fail(errc::not_single_output);
      }
      (void)lane(want);
    }
    return {};
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
    return asked >= Lane::ladder() ? std::nullopt : std::optional{Warmup {
      .points = l.points(),
      .threshold = Lane::rung_at(asked, effective_options())
    }};
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
    return dispatch(Want::Gradient, as_columns(xs),
                    {.jacobian = as_columns(g)}, n);
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

  // Where H(i, j) sits, or nullopt for a cell the colouring calls zero.
  [[nodiscard]] std::optional<std::size_t> hessian_cell(std::size_t i,
                                                        std::size_t j) const {
    if (poisoned() || roots_.size() != 1 || i >= symbol_count() ||
        j >= symbol_count()) {
      return std::nullopt;
    }
    return hessians().front().coloring.cell_of(i, j);
  }

  // --- the equation on disk --------------------------------------------------
  // The arena and the sweeps, which is what building spends its time on.  No
  // lane's Graph: freezing one is a linear pass over an arena the file has.

  [[nodiscard]] result<void> save(const std::filesystem::path &path) const
    requires std::floating_point<T>
  {
    return poisoned() ? std::unexpected{*bad_} : rt::save(snapshot(), path);
  }

  // Whether `path` holds *this* equation.  rt::verify() answers the other
  // question, whether a file is readable by this build at all.
  [[nodiscard]] result<void> verify(const std::filesystem::path &path) const
    requires std::floating_point<T>
  {
    if (poisoned()) {
      return std::unexpected{*bad_};
    }
    return load_snapshot<T>(path).and_then([this](const Verified<T> &snap) {
      return describes(*snap) ? result<void>{} : fail(errc::archive_mismatch);
    });
  }

  // The only way to tell whether rt::equation(path, model) built or loaded.
  [[nodiscard]] constexpr bool loaded() const noexcept { return loaded_; }

protected:
  using Compiled = detail::Compiled<T>;
  using Lane = detail::Lane<T>;
  using Lanes = std::array<Lane, want_count>;

  // Poisoned: arena_ null, roots_ empty, every accessor short-circuiting first.
  constexpr explicit Core(error why) : bad_(why) {
    if !consteval {
      lanes_ = std::make_unique<Lanes>();
      latches_ = std::make_unique<Latches>();
    }
  }

  // The arena is the equation's own from here: one it was lent is copied, so
  // nothing another equation or the caller builds can move a node under it.
  constexpr Core(std::unique_ptr<Builder<T>> arena, std::vector<NodeId> roots)
      : arena_(std::move(arena)), roots_(std::move(roots)),
        // Before any sweep appends: what a saved file is keyed on.
        model_nodes_(static_cast<std::uint32_t>(arena_->size())) {
    // An implicit no derivative could carry refuses the whole equation, as a
    // poisoned output would.
    if (const auto why = unclosed(*arena_)) {
      bad_ = error{*why};
      arena_.reset();
      roots_.clear();
    } else {
      // One reverse sweep per output is cheap, and it is what the column
      // counts, the pattern and constant evaluation read.
      derivative_ = build_jacobian_impl(*arena_, roots_);
    }
    if !consteval {
      lanes_ = std::make_unique<Lanes>();
      latches_ = std::make_unique<Latches>();
    }
  }

  // The snapshot's node array is installed verbatim, not replayed: make() folds
  // and would renumber the ids the saved sweeps name.  A block the file carries
  // counts as swept.
  explicit Core(Rebuilt<T> r)
      : arena_(std::move(r.arena)), roots_(std::move(r.rest.roots)),
        model_nodes_(r.rest.model_nodes), options_(r.rest.options),
        objects_(std::move(r.rest.objects)),
        derivative_(std::move(r.rest.jacobian)),
        hessians_(std::move(r.rest.hessians)), hvp_(std::move(r.rest.hvp)),
        vjp_(std::move(r.rest.vjp)), jvp_(std::move(r.rest.jvp)),
        lanes_(std::make_unique<Lanes>()),
        latches_(std::make_unique<Latches>()), loaded_(true) {
    const auto swept = [](std::once_flag &flag, bool carried) {
      if (carried) {
        std::call_once(flag, [] {});
      }
    };
    swept(latches_->hessians, hessians_.size() == roots_.size());
    swept(latches_->hvp, hvp_.value != no_node);
    swept(latches_->vjp, vjp_.value.size() == roots_.size());
    swept(latches_->jvp, jvp_.value.size() == roots_.size());
    settle_device();
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
    auto arena = owned ? std::move(owned)
                       : std::make_unique<Builder<T>>(
                             *Rooting::builder(outputs.front()));
    auto roots = outputs | std::views::transform([&arena](const auto &e) {
                   return e.id(*arena);
                 }) |
                 impl::to<std::vector<NodeId>>();
    return {std::move(arena), std::move(roots)};
  }

  // The three ways an equation comes to be, for whichever class wraps this.
  template <typename D>
  [[nodiscard]] static constexpr D
  make(Cache memo, std::span<const RTExpression<T>> outputs,
       std::unique_ptr<Builder<T>> owned) {
    if (const auto why = why_not(outputs)) {
      return D{*why};
    }
    auto [arena, roots] = rooted(outputs, std::move(owned));
    D eq{std::move(arena), std::move(roots)};
    eq.take_cache(std::move(memo));
    return eq;
  }

  // Build the model, then take the sweeps off disk if the file still describes
  // it.  An absent or stale file rebuilds rather than refusing; loaded() tells.
  template <typename D>
  [[nodiscard]] static D cached(Cache memo, const std::filesystem::path &path,
                                std::span<const RTExpression<T>> outputs,
                                std::unique_ptr<Builder<T>> owned)
    requires std::floating_point<T>
  {
    if (const auto why = why_not(outputs)) {
      return D{*why};
    }
    auto [arena, roots] = rooted(outputs, std::move(owned));
    if (auto snap = load_snapshot<T>(path);
        snap && describes(**snap, *arena, roots, arena->size())) {
      D eq{std::move(*snap).rebuild()};
      eq.take_cache(std::move(memo));
      return eq;
    }
    D eq{std::move(arena), std::move(roots)};
    eq.take_cache(std::move(memo));
    // A file that cannot be written is still a working equation.
    (void)eq.save(path);
    return eq;
  }

  // Nothing built and nothing swept; the model need not exist in this program.
  // `outputs`, where the caller's type fixes the count, refuses any other.
  template <typename D>
  [[nodiscard]] static result<D> load(const std::filesystem::path &path,
                                      Cache memo,
                                      std::optional<std::size_t> outputs)
    requires std::floating_point<T>
  {
    return load_snapshot<T>(path).and_then(
        [&](Verified<T> snap) -> result<D> {
          if (outputs && snap->roots.size() != *outputs) {
            return fail(errc::archive_mismatch);
          }
          D eq{std::move(snap).rebuild()};
          eq.take_cache(std::move(memo));
          return eq;
        });
  }

  // The roots as well as the digest: two equations over one arena share every
  // node and differ only in which ids they call outputs.
  [[nodiscard]] static bool describes(const Snapshot<T> &snap,
                                      const Builder<T> &arena,
                                      std::span<const NodeId> roots,
                                      std::size_t model_nodes) {
    return std::ranges::equal(snap.roots, roots) &&
           digest(snap) == digest(arena, model_nodes);
  }
  [[nodiscard]] bool describes(const Snapshot<T> &snap) const {
    [[maybe_unused]] const auto held = reading();
    return describes(snap, *arena_, roots_, model_nodes_);
  }

  // Every sweep first, then the arena: they append, so a node array taken
  // before they run lacks the ids they hand back.
  [[nodiscard]] Snapshot<T> snapshot() const {
    Snapshot<T> snap;
    snap.hessians = hessians();
    if (roots_.size() == 1) {
      snap.hvp = hvp_sweep();
    }
    snap.vjp = vjp_sweep();
    snap.jvp = jvp_sweep();
    snap.jacobian = derivative_;
    snap.roots = roots_;
    snap.model_nodes = model_nodes_;
    snap.options = options_;
    snap.objects = objects();
    [[maybe_unused]] const auto held = reading();
    snap.symbols = arena_->symbols();
    snap.params = arena_->params();
    snap.implicits = arena_->implicits();
    snap.closures.assign(arena_->closures().begin(), arena_->closures().end());
    snap.nodes.assign(arena_->nodes().begin(), arena_->nodes().end());
    return snap;
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
    std::span<const std::size_t> kept{}; // empty: all of them
  };
  [[nodiscard]] Cells every_cell() const { return {.jacobian = derivative_}; }

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
    return checked(*lanes_, every_cell(), want, xs, out, n)
        .transform([&](const std::shared_ptr<const Compiled> &c) {
          this->through(want, *c->graph, answered(*c), xs, out, n,
                        [&] { run(*c, xs, out, n); });
        });
  }

  // The lane a call runs on, once the call is known to fit it.
  [[nodiscard]] result<std::shared_ptr<const Compiled>>
  checked(Lanes &lanes, const Cells &cells, Want want,
          std::span<const T *const> xs, const Columns<T> &out,
          std::size_t n) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    if (needs_single(want) && roots_.size() != 1) {
      return fail(errc::not_single_output);
    }
    auto c = lane_in(lanes, cells, want, n);
    if (xs.size() != c->graph->arity() ||
        std::ranges::any_of(zip_blocks(out, c->graph->output_blocks()),
                            [](const auto &pair) {
                              const auto &[columns, block] = pair;
                              return columns.size() != block.size();
                            })) {
      return fail(errc::wrong_column_count);
    }
    return c;
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
    std::call_once(latches_->walk, [&] {
      auto wanted = blocks | std::views::transform(&Hessian::compressed) |
                    std::views::join | impl::to<std::vector<NodeId>>();
      impl::append(wanted, blocks | std::views::transform(&Hessian::partial) |
                               std::views::join);
      impl::append(wanted, blocks | std::views::transform(&Hessian::zero));
      impl::append(wanted, roots_);
      [[maybe_unused]] const auto held = reading();
      walk_ = schedule_of(*arena_, live_ids(reachable(*arena_, wanted)));
    });
    const std::size_t n = symbol_count();
    // The tape is thread-kept; every id read below is scheduled, so it is
    // written this call.
    static thread_local std::vector<T> values;
    [[maybe_unused]] const auto held = sweeping();
    values.resize(arena_->size());
    evaluate_into(*arena_, at, walk_, std::span<T>{values});
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

  // The point columns and the direction's, in one array: the named inputs
  // first, then the seeds, which is the order the ABI takes and input_column()
  // states.
  [[nodiscard]] static auto widen(const auto &xs, const auto &seeds) {
    boost::container::small_vector<const T *, 64> out;
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
    return poisoned() || roots_.size() != 1 ? std::optional<V>{}
                                            : std::optional<V>{value()};
  }

  // m long, always a vector: what System answers and Equation narrows.
  [[nodiscard]] constexpr result<std::vector<T>>
  values(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      std::vector<T> f(roots_.size());
      gather(Want::Value, at, f);
      return f;
    });
  }

private:
  // Run `build` once, under the arena's write lock: every sweep appends.
  template <typename V>
  const V &once(std::once_flag &flag, V &slot, auto &&build) const {
    std::call_once(flag, [&] {
      [[maybe_unused]] const auto held = writing();
      slot = build();
    });
    return slot;
  }

  [[nodiscard]] const std::vector<Hessian> &hessians() const {
    return once(latches_->hessians, hessians_, [this] {
      return roots_ | std::views::transform([this](NodeId r) {
               return build_hessian_impl(*arena_, r);
             }) |
             impl::to<std::vector<Hessian>>();
    });
  }
  // Root 0's only: a system refuses hvp before it gets here.
  [[nodiscard]] const HessianVector &hvp_sweep() const {
    return once(latches_->hvp, hvp_,
                [this] { return build_hvp_impl(*arena_, roots_.front()); });
  }
  [[nodiscard]] const VectorJacobian &vjp_sweep() const {
    return once(latches_->vjp, vjp_,
                [this] { return build_vjp_impl(*arena_, roots_); });
  }
  [[nodiscard]] const Tangent &jvp_sweep() const {
    return once(latches_->jvp, jvp_,
                [this] { return build_jvp_impl(*arena_, roots_); });
  }
  // Not saved: one forward sweep, re-derived on first ask.
  [[nodiscard]] const std::vector<NodeId> &tangent_sweep() const {
    return once(latches_->tangent, jacobian_tangent_, [this] {
      return build_jacobian_tangent_impl(*arena_, derivative_);
    });
  }

  // e_j for a symbol named `along`, or unknown_symbol.
  [[nodiscard]] result<boost::container::small_vector<T, 64>>
  unit(std::string_view along) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    const auto input = arena_->lookup(along);
    if (!input || input->op != OpCode::Var) {
      return fail(errc::unknown_symbol);
    }
    boost::container::small_vector<T, 64> e(symbol_count(), T{0});
    e[input->slot] = T{1};
    return e;
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
    case Want::Value:
    case Want::Jacobian:
    case Want::Gradient:
      break;
    }
  }

  [[nodiscard]] static constexpr bool needs_single(Want want) noexcept {
    return want == Want::Hessian || want == Want::Hvp;
  }
  [[nodiscard]] result<void> single() const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    return roots_.size() == 1 ? result<void>{} : fail(errc::not_single_output);
  }

  [[nodiscard]] auto reading() const {
    return std::shared_lock{latches_->arena};
  }
  [[nodiscard]] auto writing() const {
    return std::unique_lock{latches_->arena};
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

  // What Caching walks; its callers hold sweeping() around every read.
  [[nodiscard]] const Builder<T> &arena() const noexcept { return *arena_; }
  [[nodiscard]] std::size_t arena_size() const {
    [[maybe_unused]] const auto held = reading();
    return arena_->size();
  }

  // The three seeded products: the point widened by `along` -- the named
  // inputs first, then the seeds, which is the order the ABI takes and
  // input_column() states.
  template <Want W>
  [[nodiscard]] result<std::vector<T>>
  seeded(std::span<const T> along,
         const impl::rt_detail::CPointArg<T> auto &...args) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    const std::size_t m = roots_.size();
    if (along.size() != (W == Want::Vjp ? m : symbol_count())) {
      return fail(errc::wrong_direction);
    }
    return point(args...).transform([this, along, m](const auto &at) {
      boost::container::small_vector<T, 64> widened;
      widened.reserve(at.size() + along.size());
      impl::append(widened, at);
      impl::append(widened, along);
      std::vector<T> f(m);
      boost::container::small_vector<T, 64> g(
          W == Want::Hvp ? derivative_.partial.size() : 0);
      std::vector<T> out(W == Want::Jvp ? m : symbol_count());
      if constexpr (W == Want::Hvp) {
        gather(W, widened, f, g, out);
      } else {
        gather(W, widened, f, out);
      }
      return out;
    });
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

  // One point through the batch path at n = 1, not the arena walk.  Constant
  // evaluation keeps it, having no lane.
  constexpr void gather(Want want, std::span<const T> at, std::span<T> f,
                        std::span<T> g = {}, std::span<T> h = {}) const {
    if !consteval {
      const auto xs = single_columns(at);
      const auto fs = single_columns(f);
      const auto gs = single_columns(g);
      const auto hs = single_columns(h);
      (void)dispatch(want, as_columns(xs),
                     {.values = as_columns(fs),
                      .jacobian = as_columns(gs),
                      .hessian = as_columns(hs)},
                     1);
      return;
    }
    auto wanted = roots_;
    impl::append(wanted, derivative_.partial);
    const auto values = evaluate_reachable(*arena_, wanted, at);
    const auto pick = [&values](std::span<T> out, const auto &nodes) {
      std::ranges::transform(nodes | std::views::take(out.size()), out.begin(),
                             [&values](NodeId v) { return values[v]; });
    };
    pick(f, roots_);
    pick(g, derivative_.partial);
  }

  // Row-major rows x columns from the cells the pattern names, the zeros put
  // back.
  [[nodiscard]] static constexpr std::vector<T>
  dense(const Sparsity &pattern, std::span<const T> cells) {
    const std::size_t n = pattern.columns;
    std::vector<T> out(pattern.size() * n, T{0});
    for (const auto [i, j, v] : pattern.cells(cells)) {
      out[i * n + j] = v;
    }
    return out;
  }

  [[nodiscard]] Lane &lane_for(Want want) const {
    return (*lanes_)[std::to_underlying(want)];
  }

  // Looked up where the options change -- options() and the loaded
  // constructor -- so a batch call never resolves a selector.
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
      out.objects = objects_;
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

  // What the lanes are holding, for the file to carry.  Only kernels that kept
  // their bytes, so without retain_object an equation saves its graph and no
  // code.
  [[nodiscard]] std::vector<Object> objects() const {
#ifdef DDX_HAS_JIT
    if (poisoned() || compiler() == nullptr) {
      return {};
    }
    std::vector<Object> out;
    out.reserve(want_count);
    for (const auto [want, l] : std::views::zip(want_values, *lanes_)) {
      if (auto kept = l.object(want, setting())) {
        out.push_back(*std::move(kept));
      }
    }
    return out;
#else
    return {};
#endif
  }

  // What a freeze asks for.  `ids` and `partials` differ between the swept graph
  // and the kernel's rebalanced one; the rest cannot be rebalanced.  Read only
  // after ensure() has built what the want names.
  struct Frozen {
    const Core &core;
    std::span<const NodeId> ids;
    const Jacobian &partials;
    std::span<const NodeId> product{}; // dJ/dv over `partials`' cells

    [[nodiscard]] std::span<const NodeId> roots() const { return ids; }
    [[nodiscard]] const Jacobian &jacobian() const { return partials; }
    [[nodiscard]] const Hessian &hessian() const {
      return core.hessians_.front();
    }
    [[nodiscard]] const HessianVector &hessian_vector() const {
      return core.hvp_;
    }
    [[nodiscard]] const VectorJacobian &vector_jacobian() const {
      return core.vjp_;
    }
    [[nodiscard]] const Tangent &tangent() const { return core.jvp_; }
    [[nodiscard]] std::span<const NodeId> jacobian_tangent() const {
      return product;
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
      product = cells.kept.empty()
                    ? jacobian_tangent_
                    : cells.kept | std::views::transform([this](std::size_t k) {
                        return jacobian_tangent_[k];
                      }) | impl::to<std::vector<NodeId>>();
    }
    [[maybe_unused]] const bool rebalance =
        cells.kept.empty() && rebalances(want);
    [[maybe_unused]] const auto held = reading();
    Frozen from{*this, roots_, cells.jacobian, product};
    auto swept = std::make_shared<const Graph<T>>(
        freeze_for(want, *arena_, from, contracts()));
    auto compiled = swept;
#ifdef DDX_HAS_JIT
    if (rebalance) {
      Frozen rebalanced{*this, compile_roots_, compile_derivative_};
      compiled = std::make_shared<const Graph<T>>(
          freeze_for(want, *arena_, rebalanced, contracts()));
    }
#endif
    return {.graph = std::move(swept), .compile_graph = std::move(compiled)};
  }

  // Whether the kernel for `want` is lowered from rebalanced roots, which are
  // swept here on first ask: only a lane that compiles reads them.
  [[nodiscard]] bool rebalances([[maybe_unused]] Want want) const {
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      if (!rebalanceable(want) || setting().compiler == nullptr) {
        return false;
      }
      std::call_once(latches_->rebalanced, [this] {
        [[maybe_unused]] const auto held = writing();
        compile_roots_ = detail::rebalance(*arena_, roots_);
        compile_derivative_ = build_jacobian_impl(*arena_, compile_roots_);
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

  [[nodiscard]] static Answered answered(const Compiled &c) noexcept {
    return c.kernel ? Answered::ByKernel : Answered::BySweep;
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
    interpret(*c.graph, xs, out, n);
  }

  // The block sweep, within ~1.4x of a kernel and not a sad path.
  void interpret(const Graph<T> &graph, std::span<const T *const> xs,
                 const Columns<T> &out, std::size_t n) const {
    const auto blocks = graph.output_blocks();
    const auto schedule = graph.schedule();
    // The graph's width, not the arena's symbol count: `xs` carries one column
    // per input, and a lane may hold leaves that are not symbols.
    const std::size_t inputs = graph.arity();

    // Tapes are sized by the arena rather than the graph: it may have grown
    // since the freeze, and live ids index below that.
    //
    // thread_local rather than a member: two threads may be inside one
    // equation at once.  resize() rather than assign(): every id read is
    // written first, so zeroing the arena per call is waste.
    //
    // A short batch sweeps one point at a time rather than padding out to
    // block_lanes, which on a batch of one is the whole cost.
    if (n < block_lanes) {
      static thread_local std::vector<T> at;
      static thread_local std::vector<T> tape;
      at.resize(inputs);
      tape.resize(arena_->size());
      for (const std::size_t i : std::views::iota(0uz, n)) {
        std::ranges::transform(xs, at.begin(),
                               [i](const T *column) { return column[i]; });
        evaluate_into(*arena_, at, schedule, std::span<T>{tape});
        scatter_blocks(blocks, std::span<const T>{tape}, out, i, 1, 1);
      }
      return;
    }

    // block_lanes points per sweep: the switch is paid once per node per
    // block, and each operation becomes a lane loop wide enough to vectorise.
    // A short final block repeats its last point, and those lanes are never
    // read back.
    static thread_local std::vector<T> lanes;
    static thread_local std::vector<T> tape;
    lanes.resize(inputs * block_lanes);
    tape.resize(arena_->size() * block_lanes);

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
                                  std::span<T>{tape});
      scatter_blocks(blocks, std::span<const T>{tape}, out, base, block_lanes,
                     width);
    }
  }

  std::unique_ptr<Builder<T>> arena_;
  std::vector<NodeId> roots_;
  // Where the model ends and the sweeps begin.  A file is keyed on the arena up
  // to here, so rebuilding the model reproduces the key.
  std::uint32_t model_nodes_ = 0;

  jit::Options options_{};
  // Machine code a file handed over, consulted once per lane by prepare().
  std::vector<Object> objects_{};
#ifdef DDX_HAS_OPENCL
  // Settled with options_: null unless the backend is Device and one answered.
  const cl::Device *device_ = nullptr;
#endif
  Jacobian derivative_;
  // Each built once, on first ask, under the arena's write lock -- the sweeps
  // append to it -- and never touched again.  Latches says which are built.
  mutable std::vector<Hessian> hessians_;
  mutable HessianVector hvp_;
  mutable VectorJacobian vjp_;
  mutable Tangent jvp_;
  mutable std::vector<NodeId> jacobian_tangent_;
  // A system's hessian() walk.
  mutable std::vector<Step> walk_;
#ifdef DDX_HAS_JIT
  mutable std::vector<NodeId> compile_roots_;
  mutable Jacobian compile_derivative_;
#endif
  // unique_ptr, not shared: its destructor is constexpr, which is what keeps a
  // constant-evaluated equation destructible.  Null only there.
  std::unique_ptr<Lanes> lanes_;
  std::unique_ptr<Latches> latches_;
  // Set exactly when why_not() refused, so poisoned() <=> arena_ == nullptr.
  std::optional<error> bad_{};
  bool loaded_ = false;
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
  [[nodiscard]] std::span<const std::string> symbols() const {
    return names_;
  }
  [[nodiscard]] std::size_t jacobian_columns() const { return kept_.size(); }
  [[nodiscard]] const Sparsity &jacobian_pattern() const {
    return jacobian_.pattern;
  }

  // Dense row-major m x k.
  [[nodiscard]] result<std::vector<T>>
  jacobian(const impl::rt_detail::CPointArg<T> auto &...args) const {
    return core_->point(args...).and_then([&](const auto &at) {
      std::vector<T> f(core_->roots_.size());
      std::vector<T> g(kept_.size());
      return gather(Want::Jacobian, at, f, g).transform(
          [&] { return C::dense(jacobian_.pattern, g); });
    });
  }

  // dJ/dv over the kept cells, dense row-major m x k; `direction` is still one
  // value per symbol of the equation.
  [[nodiscard]] result<std::vector<T>>
  jacobian_jvp(std::span<const T> direction,
               const impl::rt_detail::CPointArg<T> auto &...args) const {
    if (direction.size() != core_->arity().value_or(0)) {
      return fail(errc::wrong_direction);
    }
    return core_->point(args...).and_then([&](const auto &at) {
      boost::container::small_vector<T, 64> widened;
      widened.reserve(at.size() + direction.size());
      impl::append(widened, at);
      impl::append(widened, direction);
      std::vector<T> f(core_->roots_.size());
      std::vector<T> g(kept_.size());
      std::vector<T> dg(kept_.size());
      return gather(Want::JacobianTangent, widened, f, g, dg).transform(
          [&] { return C::dense(jacobian_.pattern, dg); });
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
  [[nodiscard]] result<void> dispatch(Want want, std::span<const T *const> xs,
                                      const Columns<T> &out,
                                      std::size_t n) const {
    return core_
        ->checked(*lanes_, {.jacobian = jacobian_, .kept = kept_}, want, xs,
                  out, n)
        .transform([&](const auto &c) { core_->run(*c, xs, out, n); });
  }

  [[nodiscard]] result<void> gather(Want want, std::span<const T> at,
                                    std::span<T> f, std::span<T> g,
                                    std::span<T> dg = {}) const {
    const auto xs = C::single_columns(at);
    const auto fs = C::single_columns(f);
    const auto gs = C::single_columns(g);
    const auto dgs = C::single_columns(dg);
    return dispatch(want, C::as_columns(xs),
                    {.values = C::as_columns(fs),
                     .jacobian = C::as_columns(gs),
                     .hessian = C::as_columns(dgs)},
                    1);
  }

  const C *core_;
  std::vector<std::string> names_;
  Jacobian jacobian_;
  std::vector<std::size_t> kept_; // indices into the equation's cells
  std::unique_ptr<typename C::Lanes> lanes_;
};

} // namespace ddx::rt::detail

namespace ddx::rt {
// What Equation::restrict() and System::restrict() answer.
template <typename C> using Restricted = detail::Restricted<C>;
} // namespace ddx::rt
