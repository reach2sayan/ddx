# libddx: the runtime graph's scalar-free part, and the engine for the scalars
# rt/core.hpp names, compiled once.
add_library(ddx_rt SHARED coupling.cpp instances.cpp kernels.cpp)
add_library(ddx::rt ALIAS ddx_rt)

set_target_properties(ddx_rt PROPERTIES
        OUTPUT_NAME ddx
        EXPORT_NAME rt
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})

target_compile_definitions(ddx_rt PRIVATE DDX_BUILDING)
target_link_libraries(ddx_rt PUBLIC ddx Boost::headers Threads::Threads)
ddx_target_flags(ddx_rt)

# boost::throw_exception, which Boost leaves to the program under -fno-exceptions.
target_sources(ddx_rt PRIVATE boost_no_exceptions.cpp)

add_subdirectory(text)
