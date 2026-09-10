#pragma once

#include "jit/options.hpp"
#include "util/export.hpp"

#include <chrono>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>

// The OpenCL side: a graph built for a device and run on it.  Names no OpenCL
// type, so these are header types in every build, defined in libddx only where
// DDX_HAS_OPENCL is.
namespace ddx::cl {

using jit::error;
template <typename T> using result = jit::result<T>;

// `build` is the driver compiling the emitted source, which is the cost.
struct CompileReport {
  std::size_t nodes = 0;        // schedule steps emitted
  std::size_t source_bytes = 0; // OpenCL C handed to the driver
  std::chrono::nanoseconds emit{};
  std::chrono::nanoseconds build{};
};

class Device;

// A program built for one device, with its queue and buffers.  A copy shares
// them; the last copy to go releases them.
class Kernel {
public:
  Kernel() = default;

  // xs[j] is the column for input j, each n long; the columns go up, the
  // outputs come back.  One call at a time per kernel: an OpenCL kernel's
  // arguments are shared state.  A refusal may leave columns part-written.
  [[nodiscard]] DDX_API result<void>
  operator()(std::span<const double *const> xs, std::span<double *const> f,
             std::span<double *const> g, std::span<double *const> h,
             std::size_t n) const;

  [[nodiscard]] explicit operator bool() const noexcept {
    return impl_ != nullptr;
  }
  [[nodiscard]] const jit::KernelShape &shape() const noexcept {
    return shape_;
  }

private:
  friend class Device;
  struct Impl;
  Kernel(std::shared_ptr<Impl> impl, jit::KernelShape shape) noexcept
      : impl_{std::move(impl)}, shape_{shape} {}

  std::shared_ptr<Impl> impl_;
  jit::KernelShape shape_{};
};

// The OpenCL C a graph lowers to.  Needs no device, as jit::Ir needs no kernel.
[[nodiscard]] DDX_API std::string source_of(const rt::Graph<double> &g);

} // namespace ddx::cl
