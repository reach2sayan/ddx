#pragma once

#include "jit/kernel.hpp"
#include "rt/graph.hpp"
#include "rt/kernels.hpp"
#include "util/config.hpp" // DDX_FWD
#include "util/export.hpp"

#include <concepts>
#include <memory>
#include <mutex> // unique_lock, which <shared_mutex> does not carry
#include <optional>
#include <shared_mutex>

// One shape of graph per Want, and the kernel compiled from it.  An equation
// keeps an array of these.
namespace ddx::rt::detail {

#ifdef DDX_HAS_JIT
// One LLJIT for the process -- a static in a class template would be per
// specialisation -- reached only from a compiling backend, so an interpret-only
// program never brings LLVM up.  Defined in libddx, so a program and the
// library share it.  Null on a host with no JIT.
[[nodiscard]] DDX_API jit::Compiler *shared_compiler();
#endif

// What a kernel's tensor steps compute with: the sweep's own arena, the graph
// the kernel was lowered from, and the table.  Read under the arena's lock.
template <impl::Numeric T> struct TensorCall {
  const Arena<T> *arena = nullptr;
  const Graph<T> *graph = nullptr;
  const TensorKernelTable<T> *kernels = nullptr;
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
  // What the sweep computes tensor rows with: resolved when the pipeline froze,
  // so a table the equation is handed later means a new pipeline.
  const TensorKernelTable<T> *kernels = nullptr;
};

// Frozen and compiled on first ask, by whichever caller asks first; every
// caller after it runs what that one built.
template <impl::Numeric T> class DDX_TEMPLATE_API Pipeline {
public:
  using Held = Compiled<T>;

  // Ownership, not a reference: a reader holds its share for the whole call.
  // No options is a pipeline that never compiles.  A reader that finds the
  // lock taken lets `Yield` go before it waits, so a compile never holds it;
  // `freeze` is called at most once, under the write lock.
  template <std::default_initializable Yield>
  [[nodiscard]] std::shared_ptr<const Held>
  snapshot(const std::optional<jit::Options> &compiling, auto &&freeze) {
    if (const std::shared_lock read{mutex_, std::try_to_lock}; read && ready_) {
      return ready_;
    }
    [[maybe_unused]] const Yield yielded{};
    const std::unique_lock fill{mutex_}; // another thread may have won
    if (!ready_) {
      ready_ =
          std::make_shared<const Held>(compiled(compiling, DDX_FWD(freeze)()));
    }
    return ready_;
  }

private:
  // A refused compile leaves the kernel empty, and the sweep stays.
  [[nodiscard]] static Held
  compiled([[maybe_unused]] const std::optional<jit::Options> &compiling,
           Held held) {
#ifdef DDX_HAS_JIT
    if constexpr (jit::CKernelScalar<T>) {
      if (jit::Compiler *const c = shared_compiler(); compiling && c) {
        if (auto kernel = c->compile(*held.compile_graph, *compiling)) {
          held.kernel = *std::move(kernel);
        }
      }
    }
#endif
    return held;
  }

  std::shared_mutex mutex_;
  std::shared_ptr<const Held> ready_;
};

} // namespace ddx::rt::detail
