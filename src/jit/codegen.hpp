#pragma once

#include "jit/kernel.hpp"
#include "rt/graph.hpp"

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Module.h>

#include <string>
#include <string_view>

namespace ddx::jit::detail {

// The C name of a libm function for T, such as sin for double, sinf for float.
template <CKernelScalar T>
[[nodiscard]] std::string libm_name(std::string_view name) {
  return std::string{name} + (std::same_as<T, float> ? "f" : "");
}

// Calls a user-supplied extern function:
// ddx_call_extern(fn, in, arity, answer, components).
inline constexpr std::string_view trampoline_symbol = "ddx_call_extern";
// Runs the tensor step for one node, ddx_tensor_step(call, node, lanes, tape,
// vtape), and scatters the outputs, ddx_tensor_store(call, f, g, h, i, live,
// lanes, tape, vtape).
inline constexpr std::string_view tensor_step_symbol = "ddx_tensor_step";
inline constexpr std::string_view tensor_store_symbol = "ddx_tensor_store";
// Returns a constant table's elements: ddx_tensor_table(call, ref).
inline constexpr std::string_view tensor_table_symbol = "ddx_tensor_table";

// Emits a graph as one loop over n points, `lanes` points per iteration, with
// the body over <lanes x T>, or over T when lanes is 1.
template <CKernelScalar T>
[[nodiscard]] std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<T> &g, llvm::StringRef name,
            unsigned lanes, const llvm::DataLayout &layout,
            llvm::StringRef triple);

} // namespace ddx::jit::detail
