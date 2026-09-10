#pragma once

#include "cl/kernel.hpp"
#include "jit/options.hpp"
#include "util/export.hpp"

#include <future>
#include <memory>
#include <string_view>
#include <utility>

namespace ddx::cl {

// One OpenCL device with double precision, and the context kernels are built
// in.  A copy is a handle to the same one.
class Device {
public:
  // `selector` is a case-insensitive substring of "<platform> / <device>".
  // Empty takes the first GPU with double precision, else the first device of
  // any type that has it.  errc::no_device where nothing answers; the detail
  // lists what was seen.
  [[nodiscard]] static DDX_API result<Device>
  create(std::string_view selector = {});

  // One per selector for the process, created on first ask and never torn
  // down: vendor runtimes do not take kindly to a context released while the
  // program exits.  The reference outlives every caller.
  [[nodiscard]] static DDX_API const result<Device> &
  shared(std::string_view selector = {});

  // "<platform> / <device> / <driver> / <OpenCL version>".
  [[nodiscard]] DDX_API std::string_view identity() const noexcept;

  // The graph as OpenCL C, built by this device's compiler.
  // errc::device_compile carries the build log.
  [[nodiscard]] DDX_API result<Kernel>
  compile(const rt::Graph<double> &g, CompileReport *report = nullptr) const;

  // The future's destructor does not join: dropping it abandons the result.
  [[nodiscard]] DDX_API std::shared_future<result<Kernel>>
  compile_async(std::shared_ptr<const rt::Graph<double>> g) const;

private:
  struct Impl;
  explicit Device(std::shared_ptr<Impl> impl) noexcept
      : impl_{std::move(impl)} {}

  std::shared_ptr<Impl> impl_;
};

} // namespace ddx::cl
