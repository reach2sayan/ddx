#include "rt/core.hpp"

DDX_RT_INSTANCES(, , double)
DDX_RT_INSTANCES(, , float)
DDX_RT_INSTANCES(, , long double)
DDX_RT_INSTANCES(, , int)

#ifdef DDX_HAS_JIT
namespace ddx::rt::detail {
jit::Compiler *shared_compiler() {
  static jit::compile_result<jit::Compiler> instance = jit::Compiler::create();
  return instance ? &*instance : nullptr;
}
} // namespace ddx::rt::detail
#endif
