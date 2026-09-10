#pragma once

#include "rt/graph.hpp"

#include <string>
#include <string_view>

namespace ddx::cl::detail {

inline constexpr std::string_view kernel_name = "ddx_kernel";

// All the driver is told, and nothing that licenses it to move a bit: no
// -cl-mad-enable, -cl-fast-relaxed-math or -cl-denorms-are-zero.
inline constexpr std::string_view build_options = "-cl-std=CL1.2";

// One work-item per point.  Every block's columns lie end to end, n apart, so
// the four arrays the host ABI takes become four buffers.
[[nodiscard]] std::string emit_source(const rt::Graph<double> &g);

} // namespace ddx::cl::detail
