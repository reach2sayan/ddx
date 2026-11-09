# One target per dependency tier: `tests` and `tests_dual` link libddx for the
# Hessian colouring, `tests_rt` the runtime graph, `tests_jit` the backend.
ddx_use_googletest()
include(GoogleTest)

# Every test TU parses the same ~150k lines, so one PCH (on `tests`) serves all
# four targets; unity batches share one instantiation of the runtime core.
option(DDX_TEST_PCH "Precompile the headers every test includes" ON)
option(DDX_TEST_UNITY "Compile each test target in unity batches" ON)
set(DDX_TEST_UNITY_BATCH 4 CACHE STRING "Test sources per unity batch")
set(DDX_TEST_ENV "")
if (DDX_SANITIZE STREQUAL "thread")
    set(DDX_TEST_ENV PROPERTIES ENVIRONMENT
        "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tsan.supp:ignore_noninstrumented_modules=1")
endif ()

# ddx_test(<name> SOURCES <glob|file>... [LIBS <target>...])
function(ddx_test name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;LIBS")
    list(TRANSFORM arg_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
    file(GLOB sources CONFIGURE_DEPENDS ${arg_SOURCES})
    add_executable(${name} ${sources})
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${name} PRIVATE gtest_main ddx::rt ${arg_LIBS})
    ddx_target_flags(${name})
    ddx_runtime_deps(${name})
    if (DDX_TEST_PCH AND name STREQUAL "tests")
        target_precompile_headers(tests PRIVATE <gtest/gtest.h> <ddx.hpp> <rt/equation.hpp>)
    elseif (DDX_TEST_PCH)
        target_precompile_headers(${name} REUSE_FROM tests)
    endif ()
    set_target_properties(${name} PROPERTIES UNITY_BUILD ${DDX_TEST_UNITY}
            UNITY_BUILD_BATCH_SIZE ${DDX_TEST_UNITY_BATCH})
    gtest_discover_tests(${name} ${DDX_TEST_ENV})
endfunction()

ddx_test(tests SOURCES "*.cpp")
ddx_test(tests_dual SOURCES "dual/*.cpp")
ddx_test(tests_rt SOURCES "rt/*.cpp")

if (DDX_BUILD_JIT)
    ddx_test(tests_jit SOURCES "jit/*.cpp")
    if (DDX_SANITIZE STREQUAL "off")
        if (MSVC)
            set(exports_tool "${CMAKE_LINKER}")
        else ()
            set(exports_tool "${CMAKE_READELF}")
        endif ()
        add_test(NAME jit.exports
                COMMAND ${CMAKE_COMMAND} -DNM=${CMAKE_NM} -DDEPS=${exports_tool}
                -DLIB=$<TARGET_FILE:ddx_rt> -P "${CMAKE_CURRENT_SOURCE_DIR}/check_exports.cmake")
    endif ()
endif ()

# The out-of-tree package test, plus optional consumer glue for Eigen and CBLAS,
# each built only when that library is found.
find_package(Eigen3 3.3 QUIET CONFIG ${DDX_FIND_NO_REGISTRY})
if (TARGET Eigen3::Eigen)
    ddx_test(tests_eigen SOURCES eigen/tests_eigen.cpp LIBS Eigen3::Eigen)
endif ()

# Any CBLAS as a kernel table.
find_package(BLAS QUIET)
find_path(DDX_CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
if (TARGET BLAS::BLAS AND DDX_CBLAS_INCLUDE_DIR)
    ddx_test(tests_cblas SOURCES cblas/tests_cblas.cpp LIBS BLAS::BLAS)
    target_include_directories(tests_cblas SYSTEM PRIVATE "${DDX_CBLAS_INCLUDE_DIR}")
endif ()

option(DDX_TEST_PACKAGE "Build the out-of-tree consumer tests" ${PROJECT_IS_TOP_LEVEL})
if (DDX_TEST_PACKAGE AND DDX_INSTALL AND DDX_SANITIZE STREQUAL "off")
    set(DDX_PACKAGE_COMPONENTS rt dual)
    if (DDX_BUILD_JIT)
        list(APPEND DDX_PACKAGE_COMPONENTS jit)
    endif ()
    # $<SEMICOLON> is used because an escaped ; does not survive add_test().
    string(JOIN "$<SEMICOLON>" DDX_PACKAGE_COMPONENTS_ARG ${DDX_PACKAGE_COMPONENTS})

    set(DDX_PACKAGE_COMMON_OPTIONS
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DDDX_CONSUMER_COMPONENTS=${DDX_PACKAGE_COMPONENTS_ARG})

    # A multi-config generator's configuration never reaches a nested cmake.
    get_property(DDX_PACKAGE_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (DDX_PACKAGE_MULTI_CONFIG)
        set(DDX_PACKAGE_CONFIG -C $<CONFIG>)
        set(DDX_PACKAGE_INSTALL_CONFIG --config $<CONFIG>)
    endif ()

    # ddx_package_test(<name> <cmake options>...): configure, build and test
    # tests/package against this ddx.
    function(ddx_package_test name)
        add_test(NAME package.${name}
                COMMAND ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG}
                --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                                 "${CMAKE_CURRENT_BINARY_DIR}/package/${name}"
                --build-generator ${CMAKE_GENERATOR}
                --build-project ddx_package_consumer
                --build-options ${ARGN}
                --test-command ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG} --output-on-failure)
    endfunction()

    ddx_package_test(build_tree -Dddx_DIR=${PROJECT_BINARY_DIR} ${DDX_PACKAGE_COMMON_OPTIONS})

    # Emptied first, because an install never removes files and all_headers
    # globs the prefix.
    set(DDX_PACKAGE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package/prefix")
    add_test(NAME package.install_clean
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${DDX_PACKAGE_PREFIX})
    add_test(NAME package.install
            COMMAND ${CMAKE_COMMAND} --install ${PROJECT_BINARY_DIR}
                    ${DDX_PACKAGE_INSTALL_CONFIG} --prefix ${DDX_PACKAGE_PREFIX})
    set_tests_properties(package.install_clean PROPERTIES FIXTURES_SETUP ddx_installed)
    set_tests_properties(package.install PROPERTIES FIXTURES_SETUP ddx_installed
            DEPENDS package.install_clean)

    ddx_package_test(install_tree -DCMAKE_PREFIX_PATH=${DDX_PACKAGE_PREFIX} ${DDX_PACKAGE_COMMON_OPTIONS})
    set_tests_properties(package.install_tree PROPERTIES FIXTURES_REQUIRED ddx_installed)

    # A component this build lacks must fail at configure time.
    if (NOT DDX_BUILD_JIT)
        ddx_package_test(missing_component -Dddx_DIR=${PROJECT_BINARY_DIR}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DBoost_DIR=${Boost_DIR}
                -DDDX_CONSUMER_COMPONENTS=jit)
        set_tests_properties(package.missing_component PROPERTIES WILL_FAIL TRUE)
    endif ()
endif ()
