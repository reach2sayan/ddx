#pragma once

#include "jit/kernel.hpp"
#include "rt/graph.hpp"
#include "rt/kernels.hpp"
#include "util/config.hpp" // DDX_FWD
#include "util/export.hpp"

#include <concepts>
#include <memory>
#include <mutex> // unique_lock, which <shared_mutex> does not provide
#include <optional>
#include <shared_mutex>

// One graph per Want and the kernel compiled from it. An equation keeps an
// array of these.
namespace ddx::rt::detail {

#ifdef DDX_HAS_JIT
// The process-wide LLJIT, created on first use so interpret-only programs
// never start LLVM. Defined in libddx; null on a host with no JIT.
[[nodiscard]] DDX_API jit::Compiler *shared_compiler();
#endif

// The context a kernel's tensor steps run with. Read under the arena's lock.
template <impl::Numeric T> struct TensorCall {
  const Arena<T> *arena = nullptr;
  const Graph<T> *graph = nullptr;
  const TensorKernelTable<T> *kernels = nullptr;
};

// Immutable once published.
template <impl::Numeric T> struct Compiled {
  std::shared_ptr<const Graph<T>> graph;
  // The kernel's graph, with reduction spines blocked; it aliases graph when
  // the two agree. The sweep skips blocking because it costs tape locality.
  std::shared_ptr<const Graph<T>> compile_graph{};
  jit::Kernel<T> kernel{};
  // The tensor kernel table, fixed when the pipeline froze; a new table needs a
  // new pipeline.
  const TensorKernelTable<T> *kernels = nullptr;
  // The reason kernel is empty, or success when it is compiled.
  jit::compile_result<void> status = std::unexpected{
      jit::compile_error{.code = errc::not_compiling, .detail = {}}};
};

// Frozen and compiled by the first caller of snapshot(); later callers reuse
// that result.
template <impl::Numeric T> class DDX_TEMPLATE_API Pipeline {
public:
  // Without options the pipeline never compiles. A blocked reader constructs
  // Yield before waiting; freeze runs at most once, under the write lock.
  template <std::default_initializable Yield>
  [[nodiscard]] std::shared_ptr<const Compiled<T>>
  snapshot(const std::optional<jit::Options> &compiling, auto &&freeze) {
    if (const std::shared_lock read{mutex_, std::try_to_lock}; read && ready_) {
      return ready_;
    }
    [[maybe_unused]] const Yield yielded{};
    const std::unique_lock fill{mutex_}; // another thread may have won
    if (!ready_) {
      ready_ = std::make_shared<const Compiled<T>>(
          compiled(compiling, DDX_FWD(freeze)()));
    }
    return ready_;
  }

private:
  // A refused compile leaves the kernel empty, so the sweep runs instead, and
  // records the reason in status.
  [[nodiscard]] static Compiled<T>
  compiled(const std::optional<jit::Options> &compiling, Compiled<T> held) {
    if (!compiling) {
      return held;
    }
#ifdef DDX_HAS_JIT
    if constexpr (jit::CKernelScalar<T>) {
      jit::Compiler *const c = shared_compiler();
      held.status =
          c ? c->compile(*held.compile_graph, *compiling)
                  .transform(
                      [&held](jit::Kernel<T> k) { held.kernel = std::move(k); })
            : std::unexpected{jit::compile_error{
                  .code = errc::jit_target, .detail = "no JIT on this host"}};
    }
#else
    held.status = std::unexpected{jit::compile_error{
        .code = errc::jit_target, .detail = "this build has no JIT"}};
#endif
    return held;
  }

  std::shared_mutex mutex_;
  std::shared_ptr<const Compiled<T>> ready_;
};

} // namespace ddx::rt::detail
