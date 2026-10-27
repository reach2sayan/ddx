cmake_minimum_required(VERSION 3.26)

# Newest GCC on PATH, unless the caller chose a compiler.  Must precede project().
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
        AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(DDX_PREFERRED_CXX NAMES g++-15 g++-14)
    if (DDX_PREFERRED_CXX)
        set(CMAKE_CXX_COMPILER "${DDX_PREFERRED_CXX}")
    endif ()
endif ()

project(ddx VERSION 4.4.0 LANGUAGES CXX)

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
# One INTERFACE target per header directory; ddx_util carries the headers.
ddx_use_boost()
find_package(Threads REQUIRED)

configure_file(include/util/version.hpp.in
        "${CMAKE_CURRENT_BINARY_DIR}/include/util/version.hpp" @ONLY)

function(ddx_interface name)
    add_library(ddx_${name} INTERFACE)
    add_library(ddx::${name} ALIAS ddx_${name})
    set_target_properties(ddx_${name} PROPERTIES EXPORT_NAME ${name})
    target_link_libraries(ddx_${name} INTERFACE ${ARGN})
endfunction()

ddx_interface(util Boost::headers)
ddx_interface(ops ddx_util)
ddx_interface(md ddx_util ddx_ops)
ddx_interface(symbolic ddx_ops ddx_md)
ddx_interface(dual ddx_symbolic)

file(GLOB_RECURSE ddx_headers CONFIGURE_DEPENDS include/*.hpp)
target_sources(ddx_util INTERFACE FILE_SET HEADERS
        BASE_DIRS include "${CMAKE_CURRENT_BINARY_DIR}/include"
        FILES ${ddx_headers} "${CMAKE_CURRENT_BINARY_DIR}/include/util/version.hpp")

# Every header compiles on its own: target all_verify_interface_header_sets.
option(DDX_VERIFY_HEADERS "Check that every public header is self-contained" OFF)
set_property(TARGET ddx_util PROPERTY VERIFY_INTERFACE_HEADER_SETS ${DDX_VERIFY_HEADERS})

add_library(ddx INTERFACE)
add_library(ddx::ddx ALIAS ddx)
target_link_libraries(ddx INTERFACE ddx_util ddx_ops ddx_md ddx_symbolic ddx_dual)

# --- libddx -----------------------------------------------------------------
option(DDX_BUILD_JIT "Build the LLVM JIT backend into libddx" OFF)

add_subdirectory(src/rt)

if (DDX_BUILD_JIT)
    ddx_use_llvm()
    add_subdirectory(src/jit)
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
