#pragma once

#include "jit/kernel.hpp"
#include "rt/graph.hpp"
#include "rt/kernels.hpp"
#include "util/config.hpp" // DDX_FWD
#include "util/export.hpp"

#include <concepts>
#include <memory>
#include <mutex> // unique_lock, which <shared_mutex> does not carry
#include <shared_mutex>

// One shape of graph per Want, and the kernel compiled from it.  An equation
// keeps an array of these.
namespace ddx::rt::detail {

// A reader that finds the lock taken goes the slow way, so it waits only after
// giving up what the writer may be waiting to reacquire.
struct Guarded {
  [[nodiscard]] auto try_read() const {
    return std::shared_lock{mutex, std::try_to_lock};
  }
  [[nodiscard]] auto write() const { return std::unique_lock{mutex}; }
  mutable std::shared_mutex mutex;
};

// Published, never written.
template <impl::Numeric T> struct Compiled {
  // Shared, so a reader holds the graph for the whole call.
  std::shared_ptr<const Graph<T>> graph;
  // Not what the sweep walks: spines are blocked for the kernel, where the
  // chain is latency, and left alone for the sweep, where the same rewrite
  // costs tape locality.  Aliases `graph` where they agree.
  std::shared_ptr<const Graph<T>> compile_graph{};
  jit::Kernel<T> kernel{};
  jit::OptLevel level = jit::OptLevel::O0; // what `kernel` was compiled at
  // What the sweep computes tensor rows with: resolved when the pipeline froze,
  // so a table the equation is handed later means a new pipeline.
  const TensorKernelTable<T> *kernels = nullptr;
};

// What a pipeline needs from the equation that owns it.  No compiler is a
// pipeline that will never compile: an interpreting backend, a poisoned
// equation and a host with no JIT all arrive here as one.
struct Setting {
  jit::Options options{};
  jit::Compiler *compiler = nullptr;
};

// Frozen and compiled on first ask, by whichever caller asks first; every
// caller after it runs what that one built.
template <impl::Numeric T> class DDX_TEMPLATE_API Pipeline {
public:
  using Held = Compiled<T>;

  // Ownership, not a reference: a reader holds its share for the whole call.
  // `Yield` is let go before the write lock is waited for, so a compile never
  // holds it; `freeze` is called at most once, under the write lock.
  template <std::default_initializable Yield>
  [[nodiscard]] std::shared_ptr<const Held> snapshot(const Setting &setting,
                                                     auto &&freeze) {
    if (const auto read = lock_.try_read(); read && ready_) {
      return ready_;
    }
    [[maybe_unused]] const Yield yielded{};
    [[maybe_unused]] const auto fill =
        lock_.write(); // another thread may have won
    if (!ready_) {
      ready_ = std::make_shared<const Held>(
          compiled(setting, DDX_FWD(freeze)()));
    }
    return ready_;
  }

private:
  // A refused compile leaves the kernel empty, and the sweep stays.
  [[nodiscard]] static Held compiled([[maybe_unused]] const Setting &setting,
                                     Held held) {
#ifdef DDX_HAS_JIT
    if constexpr (jit::CKernelScalar<T>) {
      if (setting.compiler == nullptr) {
        return held;
      }
      if (auto kernel =
              setting.compiler->compile(*held.compile_graph, setting.options)) {
        held.kernel = *std::move(kernel);
        held.level = setting.options.codegen.codegen_level;
      }
    }
#endif
    return held;
  }

  Guarded lock_;
  std::shared_ptr<const Held> ready_;
};

} // namespace ddx::rt::detail
