target_sources(ddx_symbolic INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/symbol.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/expressions.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/simplify.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/traits.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/values.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/entry.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/format.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/bound.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/equation.hpp>
)
