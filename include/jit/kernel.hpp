#pragma once

#include "jit/options.hpp"
#include "util/export.hpp"
#include "util/ranges.hpp"

#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// The JIT compiler and the kernels it returns. No LLVM type appears here; the
// caller's options are in jit/options.hpp.
namespace ddx::jit {

// The scalars the JIT compiles for; any other scalar is interpreted.
template <typename T>
concept CKernelScalar = std::same_as<T, double> || std::same_as<T, float>;

// The extern functions whose addresses a kernel was compiled with. Holding
// them lets the kernel outlive its graph.
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

// A shared handle to compiled code, freed with the last copy. Any scalar can
// name a Kernel, but only a CKernelScalar one is ever filled.
template <impl::Numeric T> class Kernel {
public:
  using function_type = void (*)(const T *const *, T *const *, T *const *,
                                 T *const *, std::size_t,
                                 const rt::detail::TensorCall<T> *, T *);

  // Extra inputs for tensor steps: the sweep's TensorCall and shape().tapes
  // elements of scratch tape.
  struct Tensors {
    const rt::detail::TensorCall<T> *call = nullptr;
    std::span<T> tapes{};
  };

  Kernel() = default;
  Kernel(function_type fn, KernelShape shape, Externs externs,
         std::shared_ptr<void> code) noexcept
      : fn_{fn}, code_{std::move(code)}, externs_{std::move(externs)},
        shape_{shape} {}

  // xs[j] is the column for symbol j and g[j] its partial, each n long; an
  // unrequested block is empty. The kernel is compiled nounwind.
  void operator()(std::span<const T *const> xs, std::span<T *const> f,
                  std::span<T *const> g, std::span<T *const> h, std::size_t n,
                  Tensors t = {}) const noexcept {
    // The kernel trusts these sizes, so a mismatch would corrupt memory.
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
  std::shared_ptr<void> code_; // kept alive, never read
  Externs externs_;            // kept alive, never read
  KernelShape shape_{};
};

// Owns the LLVM JIT that every Kernel it made shares.
class Compiler {
public:
  Compiler(const Compiler &) = delete;
  Compiler &operator=(const Compiler &) = delete;
  Compiler(Compiler &&) = default;
  Compiler &operator=(Compiler &&) = default;
  ~Compiler() = default;

  [[nodiscard]] static DDX_API compile_result<Compiler> create();

  // Instantiated in the library for each CKernelScalar.
  template <CKernelScalar T>
  [[nodiscard]] DDX_API compile_result<Kernel<T>>
  compile(const rt::Graph<T> &g, const Options &opt = {});

  // The LLVM IR that compile() would hand to the backend, as text.
  template <CKernelScalar T>
  [[nodiscard]] DDX_API compile_result<std::string>
  ir(const rt::Graph<T> &g, const Options &opt = {}) const;

private:
  struct Impl;
  DDX_API explicit Compiler(std::shared_ptr<Impl> impl) noexcept;
  std::shared_ptr<Impl> impl_;
};

} // namespace ddx::jit
