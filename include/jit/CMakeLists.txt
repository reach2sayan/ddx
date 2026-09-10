# options.hpp is a header type in every build; kernel.hpp and format.hpp mean
# something only where DDX_HAS_JIT is defined, which ddx_rt carries from src/jit.
target_sources(ddx_rt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/options.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/kernel.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/format.hpp>
)
