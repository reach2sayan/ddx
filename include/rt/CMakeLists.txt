target_sources(ddx_rt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/opcode.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/apply.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/builder.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/expressions.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/interpret.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/kernels.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/tensor.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/eigen_kernels.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/derivative.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/dense.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/coupling.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/bridge.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/graph.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/core.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/equation.hpp>
)


add_subdirectory(text)
