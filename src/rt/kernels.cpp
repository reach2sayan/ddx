#include "rt/kernels.hpp"

template DDX_INSTANCE_API
    std::atomic<const ddx::rt::TensorKernelTable<double> *> &
    ddx::rt::detail::default_tensor_kernels<double>() noexcept;
template DDX_INSTANCE_API
    std::atomic<const ddx::rt::TensorKernelTable<float> *> &
    ddx::rt::detail::default_tensor_kernels<float>() noexcept;
