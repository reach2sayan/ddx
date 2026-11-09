# libddx, the runtime graph and its engine for the scalars rt/core.hpp names.
add_library(ddx_rt SHARED coupling.cpp instances.cpp kernels.cpp)
add_library(ddx::rt ALIAS ddx_rt)

set_target_properties(ddx_rt PROPERTIES
        OUTPUT_NAME ddx
        EXPORT_NAME rt
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})

target_compile_definitions(ddx_rt PRIVATE DDX_BUILDING)
target_link_libraries(ddx_rt PUBLIC ddx Boost::headers Threads::Threads)
ddx_target_flags(ddx_rt LIBRARY)

# Defines boost::throw_exception, which Boost requires under -fno-exceptions.
target_sources(ddx_rt PRIVATE boost_no_exceptions.cpp)

add_subdirectory(text)
