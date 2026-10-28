#pragma once

#include "jit/options.hpp"
#include "util/export.hpp"
#include "util/pinned.hpp"
#include "util/ranges.hpp"

#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// The LLVM side: what compiles and what it hands back.  Names no LLVM type;
// jit/options.hpp holds what a caller says, in every build.
namespace ddx::jit {

// The scalars a kernel is emitted over; any other sweeps.
template <typename T>
concept CKernelScalar = std::same_as<T, double> || std::same_as<T, float>;

// The functions a kernel calls back into, whose addresses it was compiled
// with: held, so a kernel outlives the graph it was lowered from.
using Externs = std::vector<std::shared_ptr<const void>>;

template <impl::Numeric T>
[[nodiscard]] Externs externs_of(const rt::Graph<T> &g) {
  return g.externs() |
         std::views::transform(
             [](const auto &e) -> std::shared_ptr<const void> {
               return e.fn;
             }) |
         impl::to<Externs>();
}

// A copy is one atomic increment; the last Kernel to go frees the code.  Any
// scalar names one, so a pipeline holds its slot whether or not it compiles;
// only a CKernelScalar is ever filled.
template <impl::Numeric T> class Kernel {
public:
  using function_type = void (*)(const T *const *, T *const *, T *const *,
                                 T *const *, std::size_t,
                                 const rt::detail::TensorCall<T> *, T *);

  // What a kernel's tensor steps read besides the columns: the sweep's arena,
  // graph and table, and shape().tapes elements to sweep through.
  struct Tensors {
    const rt::detail::TensorCall<T> *call = nullptr;
    std::span<T> tapes{};
  };

  Kernel() = default;
  Kernel(function_type fn, KernelShape shape, Externs externs,
         std::shared_ptr<void> code) noexcept
      : fn_{fn}, code_{std::move(code)}, externs_{std::move(externs)},
        shape_{shape} {}

  // xs[j] is the column for symbol j, g[j] the partial in it, each n long; an
  // unrequested block is `{}`.  noexcept: codegen marks the kernel nounwind.
  void operator()(std::span<const T *const> xs, std::span<T *const> f,
                  std::span<T *const> g, std::span<T *const> h, std::size_t n,
                  Tensors t = {}) const noexcept {
    // A mismatch past here is silent memory corruption.
    assert(xs.size() == shape_.arity && f.size() == shape_.values &&
           g.size() == shape_.jacobian && h.size() == shape_.hessian &&
           t.tapes.size() >= shape_.tapes &&
           (shape_.tapes == 0 || t.call != nullptr));
    fn_(xs.data(), f.data(), g.data(), h.data(), n, t.call, t.tapes.data());
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return fn_ != nullptr;
  }
  [[nodiscard]] const KernelShape &shape() const noexcept { return shape_; }

private:
  function_type fn_ = nullptr;
  std::shared_ptr<void> code_; // Held, never read
  Externs externs_;            // Held, never read
  KernelShape shape_{};
};

// The LLJIT itself, shared by every Kernel it made.
class Compiler : private impl::noncopyable {
public:
  [[nodiscard]] static DDX_JIT_API compile_result<Compiler> create();

  // Instantiated in the library for each CKernelScalar.
  template <CKernelScalar T>
  [[nodiscard]] DDX_JIT_API compile_result<Kernel<T>>
  compile(const rt::Graph<T> &g, const Options &opt = {});

  // The module compile() would hand the backend, as text.
  template <CKernelScalar T>
  [[nodiscard]] DDX_JIT_API compile_result<std::string>
  ir(const rt::Graph<T> &g, const Options &opt = {}) const;

private:
  struct Impl;
  DDX_JIT_API explicit Compiler(std::shared_ptr<Impl> impl) noexcept;
  std::shared_ptr<Impl> impl_;
};

} // namespace ddx::jit
