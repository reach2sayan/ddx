# The floor the toolchain must clear: <expected>.
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

set(CMAKE_REQUIRED_FLAGS "-std=c++23")
check_cxx_source_compiles(
        "#include <expected>
         int main() { return std::expected<int, int>{0}.value(); }"
        DDX_TOOLCHAIN_OK)
unset(CMAKE_REQUIRED_FLAGS)

if (NOT MSVC AND NOT DDX_TOOLCHAIN_OK)
    message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} cannot build ddx, which "
            "needs <expected>.  Build with GCC 14+ or Clang 20+.  "
            "The compiler's own diagnostic is in the CMakeConfigureLog.")
endif ()

# DDX_BUILD_OPENCL is AUTO, ON or OFF; AUTO is on where this host has an
# OpenCL runtime -- an ICD the loader would dlopen -- and off elsewhere.  A
# configure-time guess only: which devices exist is asked again at run time.
function(ddx_resolve_opencl out)
    string(TOUPPER "${DDX_BUILD_OPENCL}" choice)
    if (NOT choice STREQUAL "AUTO")
        if (DDX_BUILD_OPENCL)
            set(${out} ON PARENT_SCOPE)
        else ()
            set(${out} OFF PARENT_SCOPE)
        endif ()
        return()
    endif ()
    set(found "")
    if (NOT WIN32 AND NOT APPLE)
        set(dirs /etc/OpenCL/vendors)
        if (DEFINED ENV{OCL_ICD_VENDORS} AND IS_DIRECTORY "$ENV{OCL_ICD_VENDORS}")
            list(PREPEND dirs "$ENV{OCL_ICD_VENDORS}")
        endif ()
        foreach (dir IN LISTS dirs)
            file(GLOB icds LIST_DIRECTORIES false "${dir}/*.icd")
            if (icds)
                list(TRANSFORM icds REPLACE ".*/" "")
                string(JOIN ", " icds ${icds})
                set(found "${icds} in ${dir}")
                break()
            endif ()
        endforeach ()
    endif ()
    if (found)
        message(STATUS "OpenCL device backend: on (AUTO found ${found})")
        set(${out} ON PARENT_SCOPE)
    else ()
        message(STATUS "OpenCL device backend: off (AUTO found no OpenCL runtime; -DDDX_BUILD_OPENCL=ON builds it anyway)")
        set(${out} OFF PARENT_SCOPE)
    endif ()
endfunction()
