#include "rt/core.hpp"

#ifdef _MSC_VER
#pragma warning(disable : 4251) // see rt/core.hpp
#endif

template class DDX_INSTANCE_API ddx::rt::Arena<double>;
template class DDX_INSTANCE_API ddx::rt::Arena<float>;
template class DDX_INSTANCE_API ddx::rt::Graph<double>;
template class DDX_INSTANCE_API ddx::rt::Graph<float>;
DDX_RT_INSTANCES(, DDX_INSTANCE_API, double)
DDX_RT_INSTANCES(, DDX_INSTANCE_API, float)

#ifdef DDX_HAS_JIT
namespace ddx::rt::detail {
jit::Compiler *shared_compiler() {
  static jit::compile_result<jit::Compiler> instance = jit::Compiler::create();
  return instance ? &*instance : nullptr;
}
} // namespace ddx::rt::detail
#endif
