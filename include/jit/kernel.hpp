#pragma once

#include "jit/options.hpp"
#include "util/export.hpp"
#include "util/pinned.hpp"

#include <cassert>
#include <chrono>
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

// A copy is one atomic increment; the last Kernel to go frees the code.
class Kernel {
public:
  using function_type = void (*)(const double *const *, double *const *,
                                 double *const *, double *const *, std::size_t,
                                 const void *const *);

  using object_type = std::shared_ptr<const std::vector<std::byte>>;
  using symbol_type = std::shared_ptr<const std::string>;

  Kernel() = default;
  Kernel(function_type fn, KernelShape shape, Externs externs,
         std::shared_ptr<void> code, symbol_type symbol = {},
         object_type object = {}) noexcept
      : fn_{fn}, code_{std::move(code)}, externs_{std::move(externs)},
        symbol_{std::move(symbol)}, object_{std::move(object)}, shape_{shape} {}

  // xs[j] is the column for symbol j, g[j] the partial in it, each n long; an
  // unrequested block is `{}`.  noexcept: codegen marks the kernel nounwind.
  void operator()(std::span<const double *const> xs, std::span<double *const> f,
                  std::span<double *const> g, std::span<double *const> h,
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

  // The bytes the compile produced, which the object cache stores.
  [[nodiscard]] std::span<const std::byte> object() const noexcept {
    return object_ ? std::span<const std::byte>{*object_}
                   : std::span<const std::byte>{};
  }

  // Chosen by the compile: whoever stores the object stores this beside it.
  [[nodiscard]] std::string_view symbol() const noexcept {
    return symbol_ ? std::string_view{*symbol_} : std::string_view{};
  }

private:
  function_type fn_ = nullptr;
  std::shared_ptr<void> code_; // Held, never read
  Externs externs_;
  symbol_type symbol_;
  object_type object_;
  KernelShape shape_{};
};

// The LLJIT itself, shared by every Kernel it made.
class Compiler : private impl::noncopyable {
public:
  [[nodiscard]] static DDX_JIT_API result<Compiler> create();

  // `x86_64-pc-linux-gnu/znver3/+avx2,+fma/llvm-20.1`: a string, not a folded
  // key, because this answers *why* a stored kernel was passed over.
  [[nodiscard]] DDX_JIT_API std::string_view host_identity() const noexcept;

  DDX_JIT_API ~Compiler();
  DDX_JIT_API Compiler(Compiler &&) noexcept;
  DDX_JIT_API Compiler &operator=(Compiler &&) noexcept;

  [[nodiscard]] DDX_JIT_API result<Kernel>
  compile(const rt::Graph<double> &g, const Options &opt = {},
          CompileReport *report = nullptr);

  // The future's destructor does not join: dropping it abandons the result.
  [[nodiscard]] DDX_JIT_API std::shared_future<result<Kernel>>
  compile_async(std::shared_ptr<const rt::Graph<double>> g, Options opt = {});

  // Links an object compiled earlier.  Nothing beyond the link is verified --
  // the cache key carries the host, options and graph digest for that -- and a
  // corrupt object is an error, not an abort.  The graph is what the object
  // was lowered from: its shape and its extern functions are read off it, never
  // claimed by the bytes.
  [[nodiscard]] DDX_JIT_API result<Kernel>
  adopt(std::span<const std::byte> object, std::string_view symbol,
        const rt::Graph<double> &g);

private:
  struct Impl;
  DDX_JIT_API explicit Compiler(std::shared_ptr<Impl> impl) noexcept;

  // Private, but Ir::str() calls it from a consumer's TU, so it is exported.
  friend class Ir;
  [[nodiscard]] DDX_JIT_API result<std::string>
  render_ir(const rt::Graph<double> &g, const Options &opt) const;

  std::shared_ptr<Impl> impl_;
};

// Borrowing, so the pipeline runs only if something reads it; the deleted
// overloads refuse temporaries.
class Ir {
public:
  Ir(const Compiler &c, const rt::Graph<double> &g, Options opt = {}) noexcept
      : compiler_{c}, graph_{g}, options_{opt} {}
  Ir(const Compiler &&, const rt::Graph<double> &, Options = {}) = delete;
  Ir(const Compiler &, const rt::Graph<double> &&, Options = {}) = delete;

  [[nodiscard]] result<std::string> str() const {
    return compiler_.render_ir(graph_, options_);
  }

private:
  const Compiler &compiler_;
  const rt::Graph<double> &graph_;
  Options options_;
};

} // namespace ddx::jit
