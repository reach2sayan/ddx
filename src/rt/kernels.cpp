#include "rt/kernels.hpp"

#include <atomic>

namespace ddx::rt {
namespace {
std::atomic<const TensorKernelTable *> current{&reference_table};
} // namespace

void use_tensor_kernels(const TensorKernelTable &table) noexcept {
  current.store(&table, std::memory_order_release);
}

const TensorKernelTable &tensor_kernels() noexcept {
  return *current.load(std::memory_order_acquire);
}

} // namespace ddx::rt
