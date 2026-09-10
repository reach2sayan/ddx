# Header types in every build; defined in libddx only where DDX_HAS_OPENCL is.
target_sources(ddx_rt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/device.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/kernel.hpp>
)
