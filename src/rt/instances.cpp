#include "rt/core.hpp"

#ifdef _MSC_VER
#pragma warning(disable : 4251) // see rt/core.hpp
#endif

template class DDX_INSTANCE_API ddx::rt::detail::Arena<double>;
template class DDX_INSTANCE_API ddx::rt::detail::Arena<float>;
template class DDX_INSTANCE_API ddx::rt::detail::Graph<double>;
template class DDX_INSTANCE_API ddx::rt::detail::Graph<float>;
DDX_RT_INSTANCES(, DDX_INSTANCE_API, double)
DDX_RT_INSTANCES(, DDX_INSTANCE_API, float)

#ifdef DDX_HAS_JIT
namespace ddx::rt::detail {
jit::detail::Compiler *shared_compiler() {
  static jit::compile_result<jit::detail::Compiler> instance =
      jit::detail::Compiler::create();
  return instance ? &*instance : nullptr;
}
} // namespace ddx::rt::detail
#endif
