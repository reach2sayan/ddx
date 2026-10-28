#pragma once

#include "jit/kernel.hpp"
#include "rt/graph.hpp"

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Module.h>

#include <string>
#include <string_view>

namespace ddx::jit::detail {

// C's spelling of a libm function at T: sin for double, sinf for float.
template <CKernelScalar T>
[[nodiscard]] std::string libm_name(std::string_view name) {
  return std::string{name} + (std::same_as<T, float> ? "f" : "");
}

// The runtime's way into a function the program supplied, defined beside libm
// and suffixed as it is: ddx_call_extern(fn, in, arity, answer, components).
inline constexpr std::string_view trampoline_symbol = "ddx_call_extern";
// The interpreter's tensor step for one node, all lanes:
// ddx_tensor_step(call, node, lanes, tape, vtape).  And its scatter of every
// output block: ddx_tensor_store(call, f, g, h, i, live, lanes, tape, vtape).
inline constexpr std::string_view tensor_step_symbol = "ddx_tensor_step";
inline constexpr std::string_view tensor_store_symbol = "ddx_tensor_store";

// One graph as a single counted loop over n points, `lanes` points per
// iteration: the body is emitted over <lanes x T>, or over T when lanes is 1.
template <CKernelScalar T>
[[nodiscard]] std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<T> &g,
            llvm::StringRef name, unsigned lanes,
            const llvm::DataLayout &layout, llvm::StringRef triple);

} // namespace ddx::jit::detail
