#pragma once

#include "jit/kernel.hpp"
#include "rt/graph.hpp"
#include "rt/kernels.hpp"
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
#include <ranges>
#include <shared_mutex>
#include <span>
#include <utility>

// One shape of graph per Want, and the compile ladder over it.  An equation
// keeps an array of these.
namespace ddx::rt::detail {

// Counters sit outside the mutex because every batch call touches them and only
// a crossing needs the write lock; relaxed because the count orders nothing --
// climb() re-reads under the lock, which is what makes a launch happen once.
struct Guarded {
  [[nodiscard]] auto read() const { return std::shared_lock{mutex}; }
  [[nodiscard]] auto write() const { return std::unique_lock{mutex}; }
  mutable std::shared_mutex mutex;
};

// One rung's compile.  Copied out by pending(), so a caller waits holding no
// lock of ours.
using Pending = std::shared_future<jit::compile_result<jit::Kernel>>;
// Polling a shared_future from many threads is well-defined.
[[nodiscard]] inline bool landed(const Pending &p) {
  using namespace std::chrono_literals;
  return p.valid() && p.wait_for(0s) == std::future_status::ready;
}

// Published, never written: a kernel arrives as a *new* Compiled.
template <impl::Numeric T> struct Compiled {
  // Shared, so the graph outlives an equation that went away mid-compile.
  std::shared_ptr<const Graph<T>> graph;
  // Not what the sweep walks: spines are blocked for the kernel, where the
  // chain is latency, and left alone for the sweep, where the same rewrite
  // costs tape locality.  Aliases `graph` where they agree.
  std::shared_ptr<const Graph<T>> compile_graph{};
  jit::Kernel kernel{};
  // Rungs share one pool and need not land in the order they were asked for,
  // so this is what refuses a late one.
  jit::OptLevel level = jit::OptLevel::O0;
  // What the sweep computes tensor rows with: resolved when the pipeline froze,
  // so a table the equation is handed later means a new pipeline.
  const TensorKernelTable *kernels = nullptr;

  [[nodiscard]] std::shared_ptr<const Compiled> with(jit::Kernel k,
                                                     jit::OptLevel l) const {
    return std::make_shared<const Compiled>(
        Compiled{.graph = graph,
                 .compile_graph = compile_graph,
                 .kernel = std::move(k),
                 .level = l,
                 .kernels = kernels});
  }
};

// What a pipeline needs from the equation that owns it.  No compiler is a
// pipeline that will never compile: an interpreting backend, a poisoned
// equation and a host with no JIT all arrive here as one.
struct Setting {
  jit::Options options{};
  jit::Compiler *compiler = nullptr; // Compile and Adapt
};

// Two rungs: a cheap compile that lands soon, then the requested level.
template <impl::Numeric T> class Pipeline {
public:
  using Held = Compiled<T>;
  static constexpr std::size_t Rungs = 2;

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
    [[maybe_unused]] const auto fill =
        lock_.write(); // another thread may have won
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

  // The cheapest rung, so a caller waits on it holding no lock of ours -- and,
  // on the Python side, none of its own either.
  [[nodiscard]] Pending pending() const {
    [[maybe_unused]] const auto read = lock_.read();
    return rungs_.front().pending;
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
    jit::OptLevel level = jit::OptLevel::O0;
  };

  // Whether the batch bought a rung.  False once every rung is spoken for, so
  // a settled pipeline never writes to the counter at all.
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
    } else {
      return false;
    }
  }

  // Freezing and launching are one step deliberately: split, a pipeline could
  // be ready having compiled nothing, and the next caller would interpret
  // forever with every answer still right.
  void prepare([[maybe_unused]] Want want, const Setting &setting,
               auto &&freeze) {
    ready_ = std::make_shared<const Held>(freeze());
    if constexpr (std::same_as<T, double>) {
      if (setting.compiler == nullptr) {
        return;
      }
#ifdef DDX_HAS_JIT
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
      // pipeline that climbs mid-run must not change its answer under a caller,
      // and rebalancing moves the last bits.
      if (asked == 0 &&
          setting.options.codegen.codegen_level > jit::OptLevel::O0) {
        rungs_[0] = rung(setting, jit::OptLevel::O0, ready_->graph);
        asked_.store(1, std::memory_order_relaxed);
        return;
      }
      // Either the top rung over a cheap one, or -- at codegen 0 -- the only
      // rung there is.
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
    if (setting.options.codegen.codegen_level > jit::OptLevel::O0) {
      rungs_[next++] = rung(setting, jit::OptLevel::O0, ready_->compile_graph);
    }
    rungs_[next] = rung(setting, setting.options.codegen.codegen_level,
                        ready_->compile_graph);
  }

  [[nodiscard]] Rung
  rung([[maybe_unused]] const Setting &setting, jit::OptLevel level,
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
    jit::Kernel best;
    jit::OptLevel level = jit::OptLevel::O0;
    for (Rung &rung : rungs_) {
      if (!landed(rung.pending)) {
        continue;
      }
      // A refused compile leaves the kernel empty, and the sweep stays.
      if (auto came = rung.pending.get(); came && (!best || rung.level > level)) {
        best = *std::move(came);
        level = rung.level;
      }
      rung.pending = {};
    }
    if (best && (!ready_->kernel || level > ready_->level)) {
      ready_ = ready_->with(std::move(best), level);
    }
  }

  // Whether the pipeline owes a reader anything: only ever a kernel to publish.
  [[nodiscard]] bool arrived() const {
    return std::ranges::any_of(
        rungs_, [](const Rung &rung) { return landed(rung.pending); });
  }

  Guarded lock_;
  std::shared_ptr<const Held> ready_;
  // Submission order, so rungs_[0] answers soonest.
  std::array<Rung, Rungs> rungs_;
  mutable std::atomic<std::size_t> points_{};
  mutable std::atomic<unsigned> asked_{};
};

} // namespace ddx::rt::detail
