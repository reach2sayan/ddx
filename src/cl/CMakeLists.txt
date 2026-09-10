# The only OpenCL-aware code, and the second object in libddx built with
# exceptions: Boost.Compute throws, and device.cpp catches at the boundary.
add_library(ddx_cl_objs OBJECT codegen.cpp device.cpp)

set_target_properties(ddx_cl_objs PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

# 1.2 throughout: the version NVIDIA, AMD, Intel and pocl all serve.
target_compile_definitions(ddx_cl_objs PRIVATE DDX_BUILDING
        CL_TARGET_OPENCL_VERSION=120 BOOST_COMPUTE_MAX_CL_VERSION=120)
target_include_directories(ddx_cl_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ddx_cl_objs PRIVATE ddx Boost::headers OpenCL::Headers Threads::Threads)
ddx_target_flags(ddx_cl_objs EXCEPTIONS)

target_sources(ddx_rt PRIVATE $<TARGET_OBJECTS:ddx_cl_objs>)
add_library(ddx::cl ALIAS ddx_rt)

# PUBLIC: consumers' own TUs read it.
target_compile_definitions(ddx_rt PUBLIC DDX_HAS_OPENCL)

# The loader swallowed and hidden: it dlopens the vendor's driver, so libddx
# needs nothing new to load -- tests/check_exports.cmake holds that line.
target_link_libraries(ddx_rt PRIVATE OpenCL)
if (NOT MSVC AND NOT APPLE)
    target_link_options(ddx_rt PRIVATE -Wl,--exclude-libs,ALL -Wl,--gc-sections)
endif ()
