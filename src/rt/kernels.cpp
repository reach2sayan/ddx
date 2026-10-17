#include "rt/kernels.hpp"

#include <atomic>

namespace ddx::rt {
namespace {
std::atomic<const KernelTable *> current{&reference_table};
} // namespace

void use_kernels(const KernelTable &table) noexcept {
  current.store(&table, std::memory_order_release);
}

const KernelTable &kernels() noexcept {
  return *current.load(std::memory_order_acquire);
}

} // namespace ddx::rt
