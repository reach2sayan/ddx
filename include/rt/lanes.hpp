#pragma once

#include "cl/device.hpp"
#include "cl/kernel.hpp"
#include "jit/kernel.hpp"
#include "rt/archive/archive.hpp" // object_of, adopt_stored
#include "rt/graph.hpp"
#include "util/config.hpp" // DDX_FWD

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <future>
#include <limits>
#include <memory>
#include <mutex> // unique_lock, which <shared_mutex> does not carry
#include <optional>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <utility>
#include <variant>

// One shape of graph per Want, and the compile ladder over it.  Both equations
// keep an array of these and differ only in the two parameters.  NOTES.md,
// "One lane, two parameters".
namespace ddx::rt::detail {

// A counter for a lane whose caller already serialises everything.
template <typename U> class Plain {
public:
  [[nodiscard]] U load(std::memory_order = std::memory_order_relaxed) const
      noexcept {
    return value_;
  }
  void store(U v, std::memory_order = std::memory_order_relaxed) noexcept {
    value_ = v;
  }
  U fetch_add(U v, std::memory_order = std::memory_order_relaxed) noexcept {
    const U was = value_;
    value_ += v;
    return was;
  }

private:
  U value_{};
};

// Counters sit outside the mutex because every batch call touches them and only
// a crossing needs the write lock; relaxed because the count orders nothing --
// climb() re-reads under the lock, which is what makes a launch happen once.
struct Guarded {
  template <typename U> using Counter = std::atomic<U>;
  [[nodiscard]] auto read() const { return std::shared_lock{mutex}; }
  [[nodiscard]] auto write() const { return std::unique_lock{mutex}; }
  mutable std::shared_mutex mutex;
};

// For a caller already holding a lock of its own: Python, which holds the GIL
// for every lane touch and gives it up only inside a sweep.
struct Serialised {
  template <typename U> using Counter = Plain<U>;
  struct Held {};
  [[nodiscard]] static constexpr Held read() noexcept { return {}; }
  [[nodiscard]] static constexpr Held write() noexcept { return {}; }
};

// The kernel a lane publishes, whichever backend built it.  A variant, so the
// alternative a build may lack is spelled here and nowhere else.
class AnyKernel {
public:
  AnyKernel() = default;
  explicit AnyKernel(jit::Kernel k) noexcept : held_{std::move(k)} {}
#ifdef DDX_HAS_OPENCL
  explicit AnyKernel(cl::Kernel k) noexcept : held_{std::move(k)} {}
#endif

  // Whether anything but the sweep answers.
  [[nodiscard]] explicit operator bool() const noexcept {
    return std::visit(
        []<typename K>(const K &k) {
          if constexpr (std::same_as<K, std::monostate>) {
            return false;
          } else {
            return static_cast<bool>(k);
          }
        },
        held_);
  }

  // Whether the columns were written.  False is the sweep's cue: no kernel, or
  // a device that could not run this call -- a kernel is never a correctness
  // dependency.
  [[nodiscard]] bool operator()(std::span<const double *const> xs,
                                std::span<double *const> f,
                                std::span<double *const> g,
                                std::span<double *const> h,
                                std::size_t n) const {
    return std::visit(
        [&]<typename K>(const K &k) {
          if constexpr (std::same_as<K, std::monostate>) {
            return false;
          } else if constexpr (std::same_as<K, jit::Kernel>) {
            if (k) {
              k(xs, f, g, h, n);
            }
            return static_cast<bool>(k);
          } else {
            return k && k(xs, f, g, h, n).has_value();
          }
        },
        held_);
  }

  // The LLVM kernel, for what only it has: stored bytes and a codegen level.
  [[nodiscard]] const jit::Kernel *jit_kernel() const noexcept {
    return std::get_if<jit::Kernel>(&held_);
  }

private:
#ifdef DDX_HAS_OPENCL
  std::variant<std::monostate, jit::Kernel, cl::Kernel> held_;
#else
  std::variant<std::monostate, jit::Kernel> held_;
#endif
};

// One rung's compile, whichever backend is running it.  Copied out by
// pending(), so a caller waits holding no lock of ours.
class Pending {
public:
  Pending() = default;
  explicit Pending(std::shared_future<jit::result<jit::Kernel>> f)
      : future_{std::move(f)} {}
#ifdef DDX_HAS_OPENCL
  explicit Pending(std::shared_future<cl::result<cl::Kernel>> f)
      : future_{std::move(f)} {}
#endif

  [[nodiscard]] bool valid() const {
    return std::visit([](const auto &f) { return f.valid(); }, future_);
  }
  // Polling a shared_future from many threads is well-defined.
  [[nodiscard]] bool landed() const {
    using namespace std::chrono_literals;
    return std::visit(
        [](const auto &f) {
          return f.valid() && f.wait_for(0s) == std::future_status::ready;
        },
        future_);
  }
  void wait() const {
    std::visit([](const auto &f) { f.wait(); }, future_);
  }
  // The kernel, or why there is none.  Only once landed().
  [[nodiscard]] jit::result<AnyKernel> take() const {
    return std::visit(
        [](const auto &f) {
          return f.get().transform([](const auto &k) { return AnyKernel{k}; });
        },
        future_);
  }

private:
#ifdef DDX_HAS_OPENCL
  std::variant<std::shared_future<jit::result<jit::Kernel>>,
               std::shared_future<cl::result<cl::Kernel>>>
      future_;
#else
  std::variant<std::shared_future<jit::result<jit::Kernel>>> future_;
#endif
};

// Published, never written: a kernel arrives as a *new* Compiled.
template <impl::Numeric T> struct Compiled {
  // Shared, so the graph outlives an equation that went away mid-compile.
  std::shared_ptr<const Graph<T>> graph;
  // Not what the sweep walks: spines are blocked for the kernel, where the
  // chain is latency, and left alone for the sweep, where the same rewrite
  // costs tape locality.  Aliases `graph` where they agree.
  std::shared_ptr<const Graph<T>> compile_graph{};
  AnyKernel kernel{};
  // Rungs share one pool and need not land in the order they were asked for,
  // so this is what refuses a late one.
  jit::Level level = jit::Level::O0;

  [[nodiscard]] std::shared_ptr<const Compiled> with(AnyKernel k,
                                                     jit::Level l) const {
    return std::make_shared<const Compiled>(Compiled{.graph = graph,
                                                     .compile_graph =
                                                         compile_graph,
                                                     .kernel = std::move(k),
                                                     .level = l});
  }
};

// What a lane needs from the equation that owns it.  Both null is a lane that
// will never compile: an interpreting backend, a poisoned equation and a host
// with neither backend all arrive here as one.  At most one is set.
struct Setting {
  jit::Options options{};
  std::span<const Object> objects{};
  jit::Compiler *compiler = nullptr;  // Compile and Adapt
  const cl::Device *device = nullptr; // Device
};

template <impl::Numeric T, typename Lock, std::size_t Rungs>
  requires(Rungs >= 1)
class Lane {
public:
  using Held = Compiled<T>;

  // Ownership, not a reference: a reader holds its share for the whole call, so
  // publishing cannot pull the graph out from under one in flight.  `n` is the
  // batch about to run and is what Adapt counts; `freeze` is called at most
  // once, under the write lock.
  [[nodiscard]] std::shared_ptr<const Held>
  snapshot(Want want, std::size_t n, const Setting &setting, auto &&freeze) {
    const bool earned = count(n, setting);
    {
      [[maybe_unused]] const auto read = lock_.read();
      if (ready_ && !earned && !arrived()) {
        return ready_;
      }
    }
    [[maybe_unused]] const auto fill = lock_.write(); // another thread may have won
    if (!ready_) {
      prepare(want, setting, DDX_FWD(freeze));
    }
    // No call waits for a compile: until one lands the graph is swept, and the
    // kernel replaces the sweep the moment it arrives.
    if (arrived()) {
      adopt();
    }
    climb(setting);
    return ready_;
  }

  // Only an LLVM kernel that kept its bytes, so without retain_object an
  // equation saves its graph and no code.
  [[nodiscard]] std::optional<Object> object([[maybe_unused]] Want want,
                                             const Setting &setting) const {
    [[maybe_unused]] const auto read = lock_.read();
    const jit::Kernel *const kernel =
        ready_ ? ready_->kernel.jit_kernel() : nullptr;
    if (setting.compiler == nullptr || kernel == nullptr ||
        kernel->object().empty()) {
      return std::nullopt;
    }
#ifdef DDX_HAS_JIT
    return object_of(want, *ready_->graph, *kernel, *setting.compiler,
                     setting.options.codegen);
#else
    return std::nullopt;
#endif
  }

  // The cheapest rung, so a caller waits on it holding no lock of ours -- and,
  // on the Python side, none of its own either.
  [[nodiscard]] Pending pending() const {
    [[maybe_unused]] const auto read = lock_.read();
    return rungs_.front().pending;
  }

  // Why the last rung to land brought no kernel, if one did not.
  [[nodiscard]] std::optional<jit::error> refused() const {
    [[maybe_unused]] const auto read = lock_.read();
    return refused_;
  }

  [[nodiscard]] std::size_t points() const {
    return points_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] unsigned asked() const {
    return asked_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] static constexpr unsigned ladder() noexcept { return Rungs; }

  // hot_points is the batch the *cheap* rung has to earn, so it is added to
  // what came before.  Saturating: "never" is spelled as a huge threshold.
  [[nodiscard]] static std::size_t rung_at(unsigned asked,
                                           const jit::Options &options) {
    constexpr auto ceiling = std::numeric_limits<std::size_t>::max();
    const std::size_t warm = options.warm_points;
    const std::size_t hot = options.hot_points;
    return asked == 0 ? warm : (warm > ceiling - hot ? ceiling : warm + hot);
  }

private:
  // `level` is also the rank: a higher rung may replace a lower one, never the
  // other way about.
  struct Rung {
    Pending pending; // set once, then read
    jit::Level level = jit::Level::O0;
  };

  // Whether the batch bought a rung.  False once every rung is spoken for, so
  // a settled lane never writes to the counter at all.
  [[nodiscard]] bool count(std::size_t n, const Setting &setting) {
    if constexpr (std::same_as<T, double>) {
      if (n == 0 || setting.compiler == nullptr ||
          setting.options.backend != jit::Backend::Adapt) {
        return false;
      }
      const unsigned asked = asked_.load(std::memory_order_relaxed);
      if (asked >= Rungs) {
        return false;
      }
      return points_.fetch_add(n, std::memory_order_relaxed) + n >=
             rung_at(asked, setting.options);
    }
    return false;
  }

  // Freezing and launching are one step deliberately: split, a lane could be
  // ready having compiled nothing, and the next caller would interpret forever
  // with every answer still right.
  void prepare([[maybe_unused]] Want want, const Setting &setting,
               auto &&freeze) {
    ready_ = std::make_shared<const Held>(freeze());
    if constexpr (std::same_as<T, double>) {
#ifdef DDX_HAS_OPENCL
      // One rung, over the swept graph rather than the rebalanced one: a
      // device hides a spine's latency with occupancy, and leaving the spine
      // alone is what keeps it on the sweep's bits.
      if (setting.device != nullptr) {
        rungs_[0] = {Pending{setting.device->compile_async(ready_->graph)},
                     jit::Level::O0};
        return;
      }
#endif
      if (setting.compiler == nullptr) {
        return;
      }
#ifdef DDX_HAS_JIT
      // A rung already climbed: published at its own level, with the ladder
      // launched as always and adopt()'s rank dropping what cannot beat it.
      if (auto have =
              adopt_stored(setting.objects, want, *ready_->compile_graph,
                           *setting.compiler, setting.options.codegen)) {
        const bool topped =
            have.level >= setting.options.codegen.codegen_level;
        ready_ = ready_->with(AnyKernel{std::move(have.kernel)}, have.level);
        // Nothing left to climb to, so nothing is launched and, under Adapt,
        // nothing more is counted.
        if (Rungs == 1 || topped) {
          asked_.store(Rungs, std::memory_order_relaxed);
          return;
        }
        asked_.store(1, std::memory_order_relaxed);
      }
      // Adapt buys its rungs in climb(); Compile asks for them outright.
      if (setting.options.backend == jit::Backend::Compile) {
        launch(setting);
      }
#endif
    }
  }

  // Under the write lock and idempotent: two threads may both arrive owing a
  // rung, and the second re-reads an `asked` the first has already moved.
  void climb(const Setting &setting) {
    if constexpr (std::same_as<T, double>) {
      if (setting.options.backend != jit::Backend::Adapt) {
        return;
      }
      const unsigned asked = asked_.load(std::memory_order_relaxed);
      if (asked >= Rungs || points_.load(std::memory_order_relaxed) <
                                rung_at(asked, setting.options)) {
        return;
      }
      if (setting.compiler == nullptr) {
        // No ladder on this host, and the sweep answers everything: stop
        // counting rather than take the write lock on every call from here on.
        asked_.store(Rungs, std::memory_order_relaxed);
        return;
      }
#ifdef DDX_HAS_JIT
      // The swept graph, not the rebalanced one a Compile backend takes: a
      // lane that climbs mid-run must not change its answer under a caller, and
      // rebalancing moves the last bits.
      if (Rungs > 1 && asked == 0 &&
          setting.options.codegen.codegen_level > jit::Level::O0) {
        rungs_[0] = rung(setting, jit::Level::O0, ready_->graph);
        asked_.store(1, std::memory_order_relaxed);
        return;
      }
      // Either the top rung over a cheap one, or -- at codegen 0, or on a
      // one-rung ladder -- the only rung there is.
      rungs_[asked == 0 ? 0 : Rungs - 1] =
          rung(setting, setting.options.codegen.codegen_level, ready_->graph);
      asked_.store(Rungs, std::memory_order_relaxed);
#endif
    }
  }

  // Cheapest rung first, so it is also first in the pool's queue: codegen 0
  // lands sooner for a slower kernel, and every level agrees to the bit.
  void launch(const Setting &setting) {
    std::size_t next = 0;
    if (Rungs > 1 && setting.options.codegen.codegen_level > jit::Level::O0) {
      rungs_[next++] = rung(setting, jit::Level::O0, ready_->compile_graph);
    }
    rungs_[next] = rung(setting, setting.options.codegen.codegen_level,
                        ready_->compile_graph);
  }

  [[nodiscard]] Rung
  rung([[maybe_unused]] const Setting &setting, jit::Level level,
       [[maybe_unused]] const std::shared_ptr<const Graph<T>> &from) {
#ifdef DDX_HAS_JIT
    jit::Options opt = setting.options;
    opt.codegen.codegen_level = level;
    return {Pending{setting.compiler->compile_async(from, opt)}, level};
#else
    return {Pending{}, level};
#endif
  }

  // Republish rather than write into what a reader holds; a rung that lost the
  // race is dropped rather than allowed to demote a live kernel.
  void adopt() {
    AnyKernel best;
    jit::Level level = jit::Level::O0;
    for (Rung &rung : rungs_) {
      if (!rung.pending.landed()) {
        continue;
      }
      // A refused compile leaves the kernel empty, and the sweep stays.
      if (auto came = rung.pending.take(); !came) {
        refused_ = std::move(came).error();
      } else if (!best || rung.level > level) {
        best = *std::move(came);
        level = rung.level;
      }
      rung.pending = {};
    }
    if (best && (!ready_->kernel || level > ready_->level)) {
      ready_ = ready_->with(std::move(best), level);
    }
  }

  // Whether the lane owes a reader anything: only ever a kernel to publish.
  [[nodiscard]] bool arrived() const {
    return std::ranges::any_of(
        rungs_, [](const Rung &rung) { return rung.pending.landed(); });
  }

  [[no_unique_address]] Lock lock_;
  std::shared_ptr<const Held> ready_;
  // Submission order, so rungs_[0] answers soonest.
  std::array<Rung, Rungs> rungs_;
  std::optional<jit::error> refused_;
  mutable typename Lock::template Counter<std::size_t> points_{};
  mutable typename Lock::template Counter<unsigned> asked_{};
};

} // namespace ddx::rt::detail
