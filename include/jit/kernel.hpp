#pragma once

#include "jit/options.hpp"
#include "util/export.hpp"
#include "util/pinned.hpp"

#include <cassert>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <future>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// The LLVM side: what compiles and what it hands back.  Names no LLVM type;
// jit/options.hpp holds what a caller says, in every build.
namespace ddx::jit {

// The scalars a kernel is emitted over; any other sweeps.
template <typename T>
concept CKernelScalar = std::same_as<T, double> || std::same_as<T, float>;

// `codegen` brackets the symbol lookup: LLJIT compiles lazily.
struct CompileReport {
  std::size_t nodes = 0;        // live nodes in the graph
  std::size_t instructions = 0; // IR instructions after emission
  std::chrono::nanoseconds emit{};
  std::chrono::nanoseconds optimize{};
  std::chrono::nanoseconds codegen{};
};

// The functions a kernel calls back into, by the index its extern nodes
// carry: held, so a kernel outlives the graph it was lowered from, and tabled,
// so the object holds no address and any process may run it.
struct Externs {
  std::vector<std::shared_ptr<const void>> held;
  std::vector<const void *> table;

  template <impl::Numeric T>
  [[nodiscard]] static Externs of(const rt::Graph<T> &g) {
    Externs out;
    for (const auto &e : g.externs()) {
      out.held.push_back(e.fn);
      out.table.push_back(e.fn.get());
    }
    return out;
  }
};

// A copy is one atomic increment; the last Kernel to go frees the code.  Any
// scalar names one, so a pipeline holds its slot whether or not it compiles;
// only a CKernelScalar is ever filled.
template <impl::Numeric T> class Kernel {
public:
  using function_type = void (*)(const T *const *, T *const *, T *const *,
                                 T *const *, std::size_t, const void *const *);

  Kernel() = default;
  Kernel(function_type fn, KernelShape shape, Externs externs,
         std::shared_ptr<void> code) noexcept
      : fn_{fn}, code_{std::move(code)}, externs_{std::move(externs)},
        shape_{shape} {}

  // xs[j] is the column for symbol j, g[j] the partial in it, each n long; an
  // unrequested block is `{}`.  noexcept: codegen marks the kernel nounwind.
  void operator()(std::span<const T *const> xs, std::span<T *const> f,
                  std::span<T *const> g, std::span<T *const> h,
                  std::size_t n) const noexcept {
    // A mismatch past here is silent memory corruption.
    assert(xs.size() == shape_.arity && f.size() == shape_.values &&
           g.size() == shape_.jacobian && h.size() == shape_.hessian);
    fn_(xs.data(), f.data(), g.data(), h.data(), n, externs_.table.data());
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return fn_ != nullptr;
  }
  [[nodiscard]] const KernelShape &shape() const noexcept { return shape_; }

  [[deprecated("no object is kept")]] [[nodiscard]] std::span<const std::byte>
  object() const noexcept {
    return {};
  }
  [[deprecated("no object is kept")]] [[nodiscard]] std::string_view
  symbol() const noexcept {
    return {};
  }

private:
  function_type fn_ = nullptr;
  std::shared_ptr<void> code_; // Held, never read
  Externs externs_;
  KernelShape shape_{};
};

// The LLJIT itself, shared by every Kernel it made.
class Compiler : private impl::noncopyable {
public:
  [[nodiscard]] static DDX_JIT_API compile_result<Compiler> create();

  // What this host compiles for, as one line:
  // `x86_64-pc-linux-gnu/znver3/feat-<hash>/llvm-20.1`.
  [[nodiscard]] DDX_JIT_API std::string_view host_identity() const noexcept;

  DDX_JIT_API ~Compiler();
  DDX_JIT_API Compiler(Compiler &&) noexcept;
  DDX_JIT_API Compiler &operator=(Compiler &&) noexcept;

  // Instantiated in the library for each CKernelScalar.
  template <CKernelScalar T>
  [[nodiscard]] DDX_JIT_API compile_result<Kernel<T>>
  compile(const rt::Graph<T> &g, const Options &opt = {},
          CompileReport *report = nullptr);

  // Always errc::jit_object: no object is kept to link.
  template <CKernelScalar T>
  [[deprecated("no object is kept")]] [[nodiscard]] DDX_JIT_API
  compile_result<Kernel<T>>
  adopt(std::span<const std::byte> object, std::string_view symbol,
        const rt::Graph<T> &g);

private:
  struct Impl;
  DDX_JIT_API explicit Compiler(std::shared_ptr<Impl> impl) noexcept;

  // Private, but Ir::str() calls it from a consumer's TU, so it is exported.
  template <CKernelScalar> friend class Ir;
  template <CKernelScalar T>
  [[nodiscard]] DDX_JIT_API compile_result<std::string>
  render_ir(const rt::Graph<T> &g, const Options &opt) const;

  std::shared_ptr<Impl> impl_;
};

// Borrowing, so the pipeline runs only if something reads it; the deleted
// overloads refuse temporaries.
template <CKernelScalar T> class Ir {
public:
  Ir(const Compiler &c, const rt::Graph<T> &g, Options opt = {}) noexcept
      : compiler_{c}, graph_{g}, options_{opt} {}
  Ir(const Compiler &&, const rt::Graph<T> &, Options = {}) = delete;
  Ir(const Compiler &, const rt::Graph<T> &&, Options = {}) = delete;

  [[nodiscard]] compile_result<std::string> str() const {
    return compiler_.render_ir(graph_, options_);
  }

private:
  const Compiler &compiler_;
  const rt::Graph<T> &graph_;
  Options options_;
};

} // namespace ddx::jit
