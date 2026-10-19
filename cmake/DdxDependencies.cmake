# Every third-party dependency.  Nothing downloads until its macro asks.
#
#   ddx_use_boost()            Boost, header-only, found or fetched  always
#   ddx_use_eigen()            Eigen, header-only, found or fetched  tests_eigen
#   ddx_use_llvm()             LLVM 20, found                        DDX_BUILD_JIT
#   ddx_use_googletest()       GoogleTest, fetched                   top-level only
#   ddx_use_pybind11()         pybind11, found in the build env      DDX_BUILD_PYTHON
#
# Found or fetched: use the Boost or Eigen you were given, download one only if
# you were not.  -DBoost_ROOT=, CMAKE_PREFIX_PATH and toolchain files all work.
include_guard(GLOBAL)

include(FetchContent)

# Skip CMake's user package registry: it records build trees, so a bare configure
# once found C:/vcpkg/buildtrees/eigen3/x64-windows-rel instead of fetching.
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

# --- Eigen, for the interop test only ------------------------------------------
# Headers only, never built, never a dependency of ddx itself.
set(DDX_EIGEN_INCLUDEDIR "" CACHE PATH "A raw Eigen header tree to use (must hold Eigen/Core); overrides both find_package and the fetch")
set(DDX_EIGEN_VERSION "3.4.0" CACHE STRING "Eigen release to fetch when none is supplied")
FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/${DDX_EIGEN_VERSION}/eigen-${DDX_EIGEN_VERSION}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR ddx-does-not-build-eigen
)

macro(ddx_use_eigen)
    if (NOT TARGET ddx_eigen)
        add_library(ddx_eigen INTERFACE)
        if (DDX_EIGEN_INCLUDEDIR)
            _ddx_adopt_eigen("${DDX_EIGEN_INCLUDEDIR}")
        else ()
            find_package(Eigen3 QUIET CONFIG ${DDX_FIND_NO_REGISTRY})
            if (Eigen3_FOUND AND TARGET Eigen3::Eigen)
                target_link_libraries(ddx_eigen INTERFACE Eigen3::Eigen)
                message(STATUS "Eigen ${Eigen3_VERSION} (found: ${Eigen3_DIR})")
            else ()
                FetchContent_MakeAvailable(eigen)
                _ddx_adopt_eigen("${eigen_SOURCE_DIR}")
            endif ()
        endif ()
    endif ()
endmacro()

macro(_ddx_adopt_eigen root)
    if (NOT EXISTS "${root}/Eigen/Core")
        message(FATAL_ERROR
                "No Eigen at ${root}: it has no Eigen/Core.  DDX_EIGEN_INCLUDEDIR "
                "must name the directory *containing* Eigen/, and setting it turns "
                "off both find_package and the pinned fetch.")
    endif ()
    target_include_directories(ddx_eigen SYSTEM INTERFACE "${root}")
    message(STATUS "Eigen (${root})")
endmacro()

# --- Boost -------------------------------------------------------------------
# In order: a Boost::headers a parent already defined, find_package, the fetch.
# DDX_BOOST_INCLUDEDIR covers what find_package cannot -- an unpacked tarball,
# which carries no CMake config until it is built.
set(DDX_BOOST_INCLUDEDIR "" CACHE PATH "A raw Boost header tree to use (must hold boost/version.hpp); overrides both find_package and the fetch")
set(DDX_BOOST_VERSION "1.92.0" CACHE STRING "Boost release to fetch when none is supplied")
# A floor on what find_package may accept, so a distro Boost too old for ddx
# falls through to the fetch instead of silently replacing it.
set(DDX_BOOST_MINIMUM "1.88" CACHE STRING "Oldest Boost a supplied one may be")
set(DDX_BOOST_SHA256 "ea7b982002cc9dfbe59b0b217b206f470dc75f3de0bb2973d844118934d82411"
        CACHE STRING "SHA256 of the archive DDX_BOOST_VERSION names")
cmake_path(SET _ddx_deps_default NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../.deps")
set(DDX_DEPS_DIR "${_ddx_deps_default}" CACHE PATH "Where fetched Boost is unpacked; shared by every build tree")
unset(_ddx_deps_default)

# SOURCE_SUBDIR names nothing, so MakeAvailable unpacks and stops.
# SOURCE_DIR is shared between trees (one download); SUBBUILD_DIR and BINARY_DIR
# cannot be -- a subbuild caches its generator, so sharing it fails the moment
# two trees use different ones.
FetchContent_Declare(boost
        URL https://github.com/boostorg/boost/releases/download/boost-${DDX_BOOST_VERSION}/boost-${DDX_BOOST_VERSION}-b2-nodocs.tar.xz
        URL_HASH SHA256=${DDX_BOOST_SHA256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR ddx-does-not-build-boost
        SOURCE_DIR "${DDX_DEPS_DIR}/boost-${DDX_BOOST_VERSION}"
        SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/boost-${DDX_BOOST_VERSION}-subbuild"
        BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/boost-${DDX_BOOST_VERSION}-build"
)

# DDX_BOOST_VENDORED is read by DdxInstall: only a Boost ddx fetched is carried
# into the package.
macro(ddx_use_boost)
    if (NOT TARGET Boost::headers)
        if (DDX_BOOST_INCLUDEDIR)
            _ddx_adopt_boost("${DDX_BOOST_INCLUDEDIR}")
            set(DDX_BOOST_VENDORED FALSE CACHE INTERNAL "ddx fetched Boost itself")
        else ()
            find_package(Boost ${DDX_BOOST_MINIMUM} QUIET CONFIG ${DDX_FIND_NO_REGISTRY})
            if (Boost_FOUND AND TARGET Boost::headers)
                set(DDX_BOOST_VENDORED FALSE CACHE INTERNAL "ddx fetched Boost itself")
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
            # Not /W0: GoogleTest puts /W4 in its own CXX flags, and cl reports
            # the override as D9025.  Its sources are clean at the level it
            # picked, so take that level and only drop the warning-as-error.
            set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
        endif ()
    endforeach ()
endfunction()
