#pragma once

#include "jit/kernel.hpp"
#include "rt/graph.hpp"

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Module.h>

#include <string>
#include <string_view>

namespace ddx::jit::detail {

// C's spelling of a libm function at T: sin for double, sinf for float.
template <CKernelScalar T> inline constexpr std::string_view libm_suffix = "";
template <> inline constexpr std::string_view libm_suffix<float> = "f";

template <CKernelScalar T>
[[nodiscard]] std::string libm_name(std::string_view name) {
  return std::string{name}.append(libm_suffix<T>);
}

// The runtime's way into a function the program supplied, defined beside
// libm: ddx_call_extern(fn, in, arity, answer, components), suffixed as libm.
template <CKernelScalar T>
[[nodiscard]] std::string trampoline_symbol() {
  return libm_name<T>("ddx_call_extern");
}

// One graph as a single counted loop over n points, `lanes` points per
// iteration: the body is emitted over <lanes x T>, or over T when lanes is 1.
template <CKernelScalar T>
[[nodiscard]] std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<T> &g,
            llvm::StringRef name, unsigned lanes,
            const llvm::DataLayout &layout, llvm::StringRef triple);

} // namespace ddx::jit::detail
