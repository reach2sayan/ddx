cmake_minimum_required(VERSION 3.26)

# Newest GCC on PATH, unless the caller chose a compiler.  Must precede project().
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
        AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(DDX_PREFERRED_CXX NAMES g++-15 g++-14)
    if (DDX_PREFERRED_CXX)
        set(CMAKE_CXX_COMPILER "${DDX_PREFERRED_CXX}")
    endif ()
endif ()

project(ddx VERSION 1.3.2 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(DdxToolchainProbes)
include(DdxCompileOptions)
include(DdxDependencies)

# --- the library ------------------------------------------------------------
# One INTERFACE target per header directory; ddx_util owns the include path.
# ops/ and symbolic/ are fully constexpr: no run-time dependency in their path.
ddx_use_boost()
find_package(Threads REQUIRED)

configure_file(include/util/version.hpp.in
        "${CMAKE_CURRENT_BINARY_DIR}/include/util/version.hpp" @ONLY)

add_library(ddx_util INTERFACE)
add_library(ddx::util ALIAS ddx_util)
set_target_properties(ddx_util PROPERTIES EXPORT_NAME util)
target_include_directories(ddx_util INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(ddx_util INTERFACE Boost::headers)

add_library(ddx_ops INTERFACE)
add_library(ddx::ops ALIAS ddx_ops)
set_target_properties(ddx_ops PROPERTIES EXPORT_NAME ops)
target_link_libraries(ddx_ops INTERFACE ddx_util)

add_library(ddx_md INTERFACE)
add_library(ddx::md ALIAS ddx_md)
set_target_properties(ddx_md PROPERTIES EXPORT_NAME md)
target_link_libraries(ddx_md INTERFACE ddx_util ddx_ops)

add_library(ddx_symbolic INTERFACE)
add_library(ddx::symbolic ALIAS ddx_symbolic)
set_target_properties(ddx_symbolic PROPERTIES EXPORT_NAME symbolic)
target_link_libraries(ddx_symbolic INTERFACE ddx_ops ddx_md)

add_library(ddx_dual INTERFACE)
add_library(ddx::dual ALIAS ddx_dual)
set_target_properties(ddx_dual PROPERTIES EXPORT_NAME dual)
target_link_libraries(ddx_dual INTERFACE ddx_symbolic)

add_library(ddx INTERFACE)
add_library(ddx::ddx ALIAS ddx)
target_link_libraries(ddx INTERFACE ddx_util ddx_ops ddx_md ddx_symbolic ddx_dual)

# Header lists for the IDE; nothing is compiled from there.
add_subdirectory(include)

# --- libddx -----------------------------------------------------------------
option(DDX_BUILD_JIT "Build the LLVM JIT backend into libddx" OFF)
set(DDX_BUILD_OPENCL AUTO CACHE STRING
        "Build the OpenCL device backend into libddx: AUTO (where this host has an OpenCL runtime), ON or OFF")
set_property(CACHE DDX_BUILD_OPENCL PROPERTY STRINGS AUTO ON OFF)
ddx_resolve_opencl(DDX_OPENCL)

add_subdirectory(src/rt)
add_subdirectory(include/rt)
# Header types in every build, whichever backends are compiled in.
add_subdirectory(include/jit)
add_subdirectory(include/cl)

if (DDX_BUILD_JIT)
    # The JIT's defaults follow how the project itself was built.
    if (DDX_FP_FLAGS)
        set(DDX_JIT_DEFAULT_CONTRACT 1)
    else ()
        set(DDX_JIT_DEFAULT_CONTRACT 0)
    endif ()

    # No predefined macro carries the optimisation level, so CMake says.  Never 0.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" DDX_BUILD_TYPE_UPPER)
    if (DDX_BUILD_TYPE_UPPER STREQUAL "DEBUG")
        set(DDX_JIT_DEFAULT_OPT 1)
    elseif (DDX_BUILD_TYPE_UPPER STREQUAL "RELEASE")
        set(DDX_JIT_DEFAULT_OPT 3)
    else ()
        set(DDX_JIT_DEFAULT_OPT 2)
    endif ()

    ddx_use_llvm()
    add_subdirectory(src/jit)
endif ()

if (DDX_OPENCL)
    # Here, not in ddx_use_opencl(): enable_language() is file scope only, and
    # libddx links the loader, which is C.
    enable_language(C)
    ddx_use_opencl()
    add_subdirectory(src/cl)
endif ()

option(DDX_INSTALL "Generate install rules" ${PROJECT_IS_TOP_LEVEL})
if (DDX_INSTALL)
    include(DdxInstall)
endif ()

# A consumer that add_subdirectory()s us gets the library and nothing else.
if (NOT PROJECT_IS_TOP_LEVEL)
    return ()
endif ()

enable_testing()
option(DDX_BUILD_TESTS "Build GoogleTest tests" ON)
if (DDX_BUILD_TESTS)
    add_subdirectory(tests)
endif ()

# After enable_testing(): python/ registers a pytest run with it.
option(DDX_BUILD_PYTHON "Build the pybind11 extension module" OFF)
if (DDX_BUILD_PYTHON)
    ddx_use_pybind11()
    add_subdirectory(python)
endif ()
