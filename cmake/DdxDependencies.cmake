# Every third-party dependency.  Nothing downloads until its macro asks.
#
#   ddx_use_boost()            Boost, header-only, found or fetched  always
#   ddx_use_llvm()             LLVM 20, found (or Conan's)           DDX_BUILD_JIT
#   ddx_use_googletest()       GoogleTest, fetched                   top-level only
#   ddx_use_pybind11()         pybind11, found in the build env      DDX_BUILD_PYTHON
include_guard(GLOBAL)

include(FetchContent)

# The package registry records build trees; never find another project's.
set(DDX_FIND_NO_REGISTRY NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)

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
# In order: a Boost::headers a parent already defined, find_package, the fetch.
# DDX_BOOST_INCLUDEDIR covers what find_package cannot -- an unpacked tarball,
# which carries no CMake config until it is built.
set(DDX_BOOST_INCLUDEDIR "" CACHE PATH "A raw Boost header tree to use (must hold boost/version.hpp); overrides both find_package and the fetch")
set(DDX_BOOST_VERSION "1.92.0" CACHE STRING "Boost release to fetch when none is supplied")
# A distro Boost older than this falls through to the fetch.
set(DDX_BOOST_MINIMUM "1.88" CACHE STRING "Oldest Boost a supplied one may be")
set(DDX_BOOST_SHA256 "ea7b982002cc9dfbe59b0b217b206f470dc75f3de0bb2973d844118934d82411"
        CACHE STRING "SHA256 of the archive DDX_BOOST_VERSION names")
cmake_path(SET _ddx_deps_default NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../.deps")
set(DDX_DEPS_DIR "${_ddx_deps_default}" CACHE PATH "Where fetched Boost is unpacked; shared by every build tree")
unset(_ddx_deps_default)

# Unpacked only (SOURCE_SUBDIR names nothing), once for every build tree; the
# subbuild caches its generator, so it stays per tree.
FetchContent_Declare(boost
        URL https://github.com/boostorg/boost/releases/download/boost-${DDX_BOOST_VERSION}/boost-${DDX_BOOST_VERSION}-b2-nodocs.tar.xz
        URL_HASH SHA256=${DDX_BOOST_SHA256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR ddx-does-not-build-boost
        SOURCE_DIR "${DDX_DEPS_DIR}/boost-${DDX_BOOST_VERSION}"
        SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/boost-${DDX_BOOST_VERSION}-subbuild"
        BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/boost-${DDX_BOOST_VERSION}-build"
)

# DDX_BOOST_VENDORED: only a Boost ddx fetched is installed with it.
macro(ddx_use_boost)
    if (NOT TARGET Boost::headers)
        set(DDX_BOOST_VENDORED FALSE CACHE INTERNAL "ddx fetched Boost itself")
        if (DDX_BOOST_INCLUDEDIR)
            _ddx_adopt_boost("${DDX_BOOST_INCLUDEDIR}")
        else ()
            find_package(Boost ${DDX_BOOST_MINIMUM} QUIET CONFIG ${DDX_FIND_NO_REGISTRY})
            if (Boost_FOUND AND TARGET Boost::headers)
                get_target_property(_ddx_boost_inc Boost::headers INTERFACE_INCLUDE_DIRECTORIES)
                list(GET _ddx_boost_inc 0 DDX_BOOST_ROOT)
                unset(_ddx_boost_inc)
                message(STATUS "Boost ${Boost_VERSION} (found: ${Boost_DIR})")
            else ()
                FetchContent_MakeAvailable(boost)
                _ddx_adopt_boost("${boost_SOURCE_DIR}")
                set(DDX_BOOST_VENDORED TRUE CACHE INTERNAL "ddx fetched Boost itself")
            endif ()
        endif ()
    endif ()
endmacro()

macro(_ddx_adopt_boost root)
    if (NOT EXISTS "${root}/boost/version.hpp")
        message(FATAL_ERROR
                "No Boost at ${root}: it has no boost/version.hpp.  "
                "DDX_BOOST_INCLUDEDIR must name the directory *containing* boost/, "
                "and setting it turns off both find_package and the pinned fetch.  "
                "Unset it to let ddx find an installed Boost, or to fall back to "
                "fetching ${DDX_BOOST_VERSION}.")
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
# Found, never fetched: an installed LLVMConfig or Conan's llvm-core, whose
# config has no LLVM_VERSION_MAJOR or llvm_map_components_to_libnames.
# Sets LLVM_DEFINITIONS_LIST and DDX_LLVM_LIBS for src/jit.
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
        find_package(LLVM CONFIG)
        foreach (_ddx_llvm_extra FFI LibEdit LibXml2 CURL)
            unset(CMAKE_DISABLE_FIND_PACKAGE_${_ddx_llvm_extra})
        endforeach ()
        unset(_ddx_llvm_extra)
        unset(ZLIB_USE_STATIC_LIBS)
        if (NOT LLVM_FOUND)
            message(FATAL_ERROR
                    "DDX_BUILD_JIT is on and no LLVM ${DDX_LLVM_VERSION} was found; ddx "
                    "never downloads it.  Either point CMake at an installed one "
                    "(-DLLVM_DIR=<prefix>/lib/cmake/llvm, e.g. apt's "
                    "/usr/lib/llvm-${DDX_LLVM_VERSION}/lib/cmake/llvm) or have Conan "
                    "build it: `conan install . -o \"&:jit=True\" --build=missing` and "
                    "configure with the conan_toolchain.cmake it writes.")
        endif ()
        # LLVMConfig matches only an exact version request, so check the major here.
        string(REGEX MATCH "^[0-9]+" _ddx_llvm_major "${LLVM_PACKAGE_VERSION}")
        if (NOT _ddx_llvm_major EQUAL DDX_LLVM_VERSION)
            message(FATAL_ERROR
                    "ddx::jit is built against LLVM ${DDX_LLVM_VERSION}; found "
                    "${LLVM_PACKAGE_VERSION}.  Point CMake at it with "
                    "-DLLVM_DIR=<prefix>/lib/cmake/llvm.")
        endif ()
        unset(_ddx_llvm_major)
        message(STATUS "LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")
        separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
        # core orcjit native passes support analysis target
        set(DDX_LLVM_LIBS LLVMCore LLVMOrcJIT LLVMPasses LLVMSupport LLVMAnalysis LLVMTarget)
        foreach (_ddx_llvm_part CodeGen AsmParser Desc Disassembler Info)
            list(APPEND DDX_LLVM_LIBS LLVM${LLVM_NATIVE_ARCH}${_ddx_llvm_part})
        endforeach ()
        unset(_ddx_llvm_part)
    endif ()
    _ddx_llvm_archives()
endmacro()

# Archives only.  Conan's llvm-core always is; an LLVMConfig (LLVM_VERSION_MAJOR
# set) is checked, and its zstd retargeted to the static one.
function(_ddx_llvm_archives)
    set(missing "")
    foreach (lib IN LISTS DDX_LLVM_LIBS)
        if (NOT TARGET ${lib})
            string(APPEND missing "  ${lib}, one of LLVM's component archives\n")
        endif ()
    endforeach ()
    if (DEFINED LLVM_VERSION_MAJOR)
        if (NOT TARGET zstd::libzstd_static)
            string(APPEND missing "  libzstd's archive (libzstd-dev on Debian/Ubuntu)\n")
        endif ()
        set(zlib "")
        if (TARGET ZLIB::ZLIB)
            get_target_property(zlib ZLIB::ZLIB IMPORTED_LOCATION_RELEASE)
            if (NOT zlib)
                get_target_property(zlib ZLIB::ZLIB IMPORTED_LOCATION)
            endif ()
        endif ()
        if (NOT zlib MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
            string(APPEND missing "  zlib's archive (zlib1g-dev on Debian/Ubuntu)\n")
        endif ()
    endif ()
    if (missing)
        message(FATAL_ERROR
                "libddx links LLVM and what it needs as archives, and these were "
                "not found:\n${missing}"
                "Conan's llvm-core has all of them: `conan install . -o \"&:jit=True\"`.")
    endif ()
    if (DEFINED LLVM_VERSION_MAJOR)
        get_target_property(deps LLVMSupport INTERFACE_LINK_LIBRARIES)
        string(REPLACE "zstd::libzstd_shared" "zstd::libzstd_static" deps "${deps}")
        set_property(TARGET LLVMSupport PROPERTY INTERFACE_LINK_LIBRARIES "${deps}")
    endif ()
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
                        "${Python_EXECUTABLE} has no pybind11; `uv sync` installs it in .venv.")
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
    endif ()
endmacro()
