# Every third-party dependency.  LLVM and pybind11 are found; the rest are
# fetched against a pin, and nothing downloads until its macro asks.
#
#   ddx_use_boost()            Boost, header-only, fetched         always
#   ddx_use_llvm()             LLVM 20, found                      DDX_BUILD_JIT
#   ddx_use_opencl()           OpenCL headers + ICD loader, fetched DDX_BUILD_OPENCL
#   ddx_use_googletest()       GoogleTest, fetched                 top-level only
#   ddx_use_pybind11()         pybind11, found in the build env    DDX_BUILD_PYTHON
include_guard(GLOBAL)

include(FetchContent)

# --- versions ---------------------------------------------------------------
set(DDX_GOOGLETEST_VERSION "1.18.0" CACHE STRING "GoogleTest release to fetch")

# Exact, not a floor: the ORC C++ API is not stable across releases.
set(DDX_LLVM_VERSION 20)

# --- declarations -----------------------------------------------------------
FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v${DDX_GOOGLETEST_VERSION}.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
)

# --- Boost -------------------------------------------------------------------
# Fetched and pinned, never found; DDX_BOOST_INCLUDEDIR is the only override.
set(DDX_BOOST_INCLUDEDIR "" CACHE PATH "A Boost include root to use instead of the pinned fetch; must hold boost/version.hpp")
set(DDX_BOOST_VERSION "1.92.0" CACHE STRING "Boost release to fetch")
set(DDX_BOOST_SHA256 "ea7b982002cc9dfbe59b0b217b206f470dc75f3de0bb2973d844118934d82411"
        CACHE STRING "SHA256 of the archive DDX_BOOST_VERSION names")
cmake_path(SET _ddx_deps_default NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../.deps")
set(DDX_DEPS_DIR "${_ddx_deps_default}" CACHE PATH "Where fetched Boost is unpacked; shared by every build tree")
unset(_ddx_deps_default)

# SOURCE_SUBDIR names nothing, so MakeAvailable unpacks and stops.
FetchContent_Declare(boost
        URL https://github.com/boostorg/boost/releases/download/boost-${DDX_BOOST_VERSION}/boost-${DDX_BOOST_VERSION}-b2-nodocs.tar.xz
        URL_HASH SHA256=${DDX_BOOST_SHA256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR ddx-does-not-build-boost
        SOURCE_DIR "${DDX_DEPS_DIR}/boost-${DDX_BOOST_VERSION}"
        SUBBUILD_DIR "${DDX_DEPS_DIR}/boost-${DDX_BOOST_VERSION}-subbuild"
        BINARY_DIR "${DDX_DEPS_DIR}/boost-${DDX_BOOST_VERSION}-build"
)

macro(ddx_use_boost)
    if (NOT TARGET Boost::headers)
        if (DDX_BOOST_INCLUDEDIR)
            _ddx_adopt_boost("${DDX_BOOST_INCLUDEDIR}")
        else ()
            FetchContent_MakeAvailable(boost)
            _ddx_adopt_boost("${boost_SOURCE_DIR}")
        endif ()
    endif ()
endmacro()

macro(_ddx_adopt_boost root)
    if (NOT EXISTS "${root}/boost/version.hpp")
        message(FATAL_ERROR
                "No Boost at ${root}: it has no boost/version.hpp.  "
                "DDX_BOOST_INCLUDEDIR must name the directory *containing* boost/, "
                "and setting it turns the pinned fetch off entirely.  Unset it to "
                "build against the fetched Boost ${DDX_BOOST_VERSION} instead.")
    endif ()
    add_library(Boost::headers INTERFACE IMPORTED GLOBAL)
    set_target_properties(Boost::headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${root}")
    file(STRINGS "${root}/boost/version.hpp" _ddx_boost_version_line REGEX "^#define BOOST_LIB_VERSION ")
    string(REGEX REPLACE ".*\"([0-9_]+)\".*" "\\1" Boost_VERSION "${_ddx_boost_version_line}")
    string(REPLACE "_" "." Boost_VERSION "${Boost_VERSION}")
    unset(_ddx_boost_version_line)
    set(DDX_BOOST_ROOT "${root}")
    message(STATUS "Boost ${Boost_VERSION} (${root})")
endmacro()

# --- LLVM -------------------------------------------------------------------
# Found, then linked as archives, zlib and zstd included.  Sets
# LLVM_DEFINITIONS_LIST and DDX_LLVM_LIBS for src/jit.
macro(ddx_use_llvm)
    if (NOT LLVM_FOUND)
        # A libz.so cached by an earlier configure would win over ZLIB_USE_STATIC_LIBS.
        foreach (_ddx_zlib_var ZLIB_LIBRARY_RELEASE ZLIB_LIBRARY_DEBUG)
            if (${_ddx_zlib_var} AND NOT ${_ddx_zlib_var} MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
                unset(${_ddx_zlib_var} CACHE)
            endif ()
        endforeach ()
        unset(_ddx_zlib_var)
        set(ZLIB_USE_STATIC_LIBS ON)
        # Components ddx never links; FindLibEdit's C try_compile fails a C++-only tree.
        foreach (_ddx_llvm_extra FFI LibEdit LibXml2 CURL)
            set(CMAKE_DISABLE_FIND_PACKAGE_${_ddx_llvm_extra} ON)
        endforeach ()
        find_package(LLVM REQUIRED CONFIG)
        foreach (_ddx_llvm_extra FFI LibEdit LibXml2 CURL)
            unset(CMAKE_DISABLE_FIND_PACKAGE_${_ddx_llvm_extra})
        endforeach ()
        unset(_ddx_llvm_extra)
        unset(ZLIB_USE_STATIC_LIBS)
        # LLVMConfig matches only an exact version request, so check the major here.
        if (NOT LLVM_VERSION_MAJOR EQUAL DDX_LLVM_VERSION)
            message(FATAL_ERROR
                    "ddx::jit is built against LLVM ${DDX_LLVM_VERSION}; found "
                    "${LLVM_PACKAGE_VERSION}.  Point CMake at it with "
                    "-DLLVM_DIR=/usr/lib/llvm-${DDX_LLVM_VERSION}/lib/cmake/llvm.")
        endif ()
        message(STATUS "LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_INCLUDE_DIRS})")
        separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
        llvm_map_components_to_libnames(DDX_LLVM_LIBS
                core orcjit native passes support analysis target)
    endif ()
    _ddx_llvm_archives()
endmacro()

# Refuse anything but archives, and retarget LLVMSupport's zstd to the static
# one Findzstd defines beside it.  Idempotent; runs even when a parent found LLVM.
function(_ddx_llvm_archives)
    set(missing "")
    if (NOT TARGET LLVMCore)
        string(APPEND missing "  LLVM's component archives (llvm-${DDX_LLVM_VERSION}-dev)\n")
    endif ()
    if (NOT TARGET zstd::libzstd_static)
        string(APPEND missing "  libzstd.a (libzstd-dev)\n")
    endif ()
    set(zlib "")
    if (TARGET ZLIB::ZLIB)
        get_target_property(zlib ZLIB::ZLIB IMPORTED_LOCATION_RELEASE)
        if (NOT zlib)
            get_target_property(zlib ZLIB::ZLIB IMPORTED_LOCATION)
        endif ()
    endif ()
    if (NOT zlib MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
        string(APPEND missing "  libz.a (zlib1g-dev)\n")
    endif ()
    if (missing)
        message(FATAL_ERROR
                "libddx links LLVM and what it needs as archives, and these were "
                "not found:\n${missing}")
    endif ()
    get_target_property(deps LLVMSupport INTERFACE_LINK_LIBRARIES)
    string(REPLACE "zstd::libzstd_shared" "zstd::libzstd_static" deps "${deps}")
    set_property(TARGET LLVMSupport PROPERTY INTERFACE_LINK_LIBRARIES "${deps}")
endfunction()

# --- OpenCL -------------------------------------------------------------------
# The Khronos headers and ICD loader, one tag for both.  The loader is built
# static and swallowed into libddx: it dlopens a vendor's driver at run time,
# so libddx needs no libOpenCL to load.
set(DDX_OPENCL_VERSION "v2026.05.29" CACHE STRING
        "KhronosGroup OpenCL-Headers and OpenCL-ICD-Loader tag to fetch")
set(DDX_OPENCL_HEADERS_SHA256 "d9e6c48357de5002da11ce45de600e0c3ffe6ab4f628a3b9fe2b38603161658a"
        CACHE STRING "SHA256 of the OpenCL-Headers archive DDX_OPENCL_VERSION names")
set(DDX_OPENCL_LOADER_SHA256 "48fd0c5181db7cd046f4f731d5955694892e10998d49d09ee0d997e7e04fd939"
        CACHE STRING "SHA256 of the OpenCL-ICD-Loader archive DDX_OPENCL_VERSION names")

# Unpacked only: the loader is added below, with its options and out of ALL.
foreach (_ddx_cl IN ITEMS Headers ICD-Loader)
    string(TOLOWER "opencl-${_ddx_cl}" _ddx_cl_dir)
    string(REPLACE "-" "_" _ddx_cl_name "${_ddx_cl_dir}")
    if (_ddx_cl STREQUAL "Headers")
        set(_ddx_cl_hash ${DDX_OPENCL_HEADERS_SHA256})
    else ()
        set(_ddx_cl_hash ${DDX_OPENCL_LOADER_SHA256})
    endif ()
    FetchContent_Declare(${_ddx_cl_name}
            URL https://github.com/KhronosGroup/OpenCL-${_ddx_cl}/archive/refs/tags/${DDX_OPENCL_VERSION}.tar.gz
            URL_HASH SHA256=${_ddx_cl_hash}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            SOURCE_SUBDIR ddx-does-not-build-this
            SOURCE_DIR "${DDX_DEPS_DIR}/${_ddx_cl_dir}-${DDX_OPENCL_VERSION}"
            SUBBUILD_DIR "${DDX_DEPS_DIR}/${_ddx_cl_dir}-${DDX_OPENCL_VERSION}-subbuild"
            BINARY_DIR "${DDX_DEPS_DIR}/${_ddx_cl_dir}-${DDX_OPENCL_VERSION}-build")
endforeach ()
unset(_ddx_cl)
unset(_ddx_cl_dir)
unset(_ddx_cl_name)
unset(_ddx_cl_hash)

# A function, so the loader's options stay in its scope.  The caller enables C,
# which has to happen at file scope.  Defines OpenCL (the loader) and
# OpenCL::Headers.
function(ddx_use_opencl)
    if (TARGET OpenCL)
        return()
    endif ()
    FetchContent_MakeAvailable(opencl_headers opencl_icd_loader)
    set(OPENCL_ICD_LOADER_HEADERS_DIR "${opencl_headers_SOURCE_DIR}" CACHE PATH
            "The OpenCL headers the ICD loader builds against" FORCE)
    set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS OFF)
    set(OPENCL_ICD_LOADER_PIC ON)
    set(OPENCL_ICD_LOADER_BUILD_TESTING OFF)
    set(ENABLE_OPENCL_LAYERS OFF)
    add_subdirectory("${opencl_icd_loader_SOURCE_DIR}"
            "${CMAKE_BINARY_DIR}/_deps/opencl-icd-loader-build" EXCLUDE_FROM_ALL SYSTEM)
    message(STATUS "OpenCL ${DDX_OPENCL_VERSION}: headers and a static ICD loader")
endfunction()

# --- pybind11 ---------------------------------------------------------------
# The interpreter's own pybind11, never the machine's.  3.0 for native_enum.
macro(ddx_use_pybind11)
    if (NOT TARGET pybind11::module)
        find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
        if (NOT DEFINED pybind11_DIR)
            execute_process(
                    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE pybind11_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
            if (NOT pybind11_DIR)
                message(FATAL_ERROR
                        "${Python_EXECUTABLE} has no pybind11.  The `python` preset builds "
                        "against .venv, so install it there: `uv sync`.  Left to search the "
                        "machine, CMake finds whatever the distro packaged -- which is how a "
                        "2.x shows up against a 3.0 request.")
            endif ()
        endif ()
        find_package(pybind11 3.0 CONFIG REQUIRED)
    endif ()
endmacro()

# --- GoogleTest --------------------------------------------------------------
macro(ddx_use_googletest)
    if (NOT TARGET gtest_main)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        _ddx_silence_dependency(gtest gtest_main gmock gmock_main)
    endif ()
endmacro()

function(_ddx_silence_dependency)
    if (NOT MSVC)
        return ()
    endif ()
    foreach (target IN LISTS ARGV)
        if (TARGET ${target})
            target_compile_options(${target} PRIVATE /W0 /wd4668)
            set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
        endif ()
    endforeach ()
endfunction()
