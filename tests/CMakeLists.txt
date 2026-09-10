# One target per dependency tier: `tests` links GoogleTest alone, `tests_rt`
# adds libddx, `tests_jit` adds the backend.
ddx_use_googletest()
include(GoogleTest)

file(GLOB DDX_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(tests ${DDX_TEST_SOURCES})
target_include_directories(tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(tests PRIVATE gtest_main ddx)
ddx_target_flags(tests)
set(DDX_TEST_ENV "")
if (DDX_SANITIZE STREQUAL "thread")
    set(DDX_TEST_ENV PROPERTIES ENVIRONMENT
        "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tsan.supp:ignore_noninstrumented_modules=1")
endif ()

gtest_discover_tests(tests ${DDX_TEST_ENV})

file(GLOB DDX_DUAL_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/dual/*.cpp")
add_executable(tests_dual ${DDX_DUAL_TEST_SOURCES})
target_include_directories(tests_dual PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(tests_dual PRIVATE gtest_main ddx)
ddx_target_flags(tests_dual)
gtest_discover_tests(tests_dual ${DDX_TEST_ENV})

file(GLOB DDX_RT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/rt/*.cpp")
add_executable(tests_rt ${DDX_RT_TEST_SOURCES})
target_link_libraries(tests_rt PRIVATE gtest_main ddx::rt)
ddx_target_flags(tests_rt)
ddx_runtime_deps(tests_rt)
gtest_discover_tests(tests_rt ${DDX_TEST_ENV})

if (DDX_BUILD_JIT)
    file(GLOB DDX_JIT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/jit/*.cpp")
    add_executable(tests_jit ${DDX_JIT_TEST_SOURCES})
    target_link_libraries(tests_jit PRIVATE gtest_main ddx::jit)
    ddx_target_flags(tests_jit)
    ddx_runtime_deps(tests_jit)
    gtest_discover_tests(tests_jit ${DDX_TEST_ENV})

    # Its own process: the assertion is the exit status.
    add_executable(tests_jit_exit_stress jit/stress/exit_stress.cpp)
    target_link_libraries(tests_jit_exit_stress PRIVATE ddx::jit)
    ddx_target_flags(tests_jit_exit_stress)
    ddx_runtime_deps(tests_jit_exit_stress)
    add_test(NAME jit.exit_stress COMMAND tests_jit_exit_stress)
    if (DDX_TEST_ENV)
        set_tests_properties(jit.exit_stress ${DDX_TEST_ENV})
    endif ()

    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND DDX_SANITIZE STREQUAL "off")
        add_test(NAME jit.exports
                COMMAND ${CMAKE_COMMAND} -DNM=${CMAKE_NM} -DREADELF=${CMAKE_READELF}
                -DLIB=$<TARGET_FILE:ddx_rt> -DWHAT=LLVM -DFORBID=[Ll][Ll][Vv][Mm]
                -P "${CMAKE_CURRENT_SOURCE_DIR}/check_exports.cmake")
    endif ()
endif ()

if (DDX_OPENCL)
    file(GLOB DDX_CL_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cl/*.cpp")
    add_executable(tests_cl ${DDX_CL_TEST_SOURCES})
    # tests/rt/energy_models.hpp, the models the Equation-level checks run.
    target_include_directories(tests_cl PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(tests_cl PRIVATE gtest_main ddx::cl)
    ddx_target_flags(tests_cl)
    ddx_runtime_deps(tests_cl)
    gtest_discover_tests(tests_cl ${DDX_TEST_ENV})

    # The loader's entry points and Boost.Compute's inlines, by name.
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND DDX_SANITIZE STREQUAL "off")
        add_test(NAME cl.exports
                COMMAND ${CMAKE_COMMAND} -DNM=${CMAKE_NM} -DREADELF=${CMAKE_READELF}
                -DLIB=$<TARGET_FILE:ddx_rt> -DWHAT=OpenCL "-DFORBID=[ ]cl[A-Z]|5boost7compute"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/check_exports.cmake")
    endif ()
endif ()

# --- packaging ---------------------------------------------------------------
# tests/package consumed out of tree, against the build tree and an install prefix.
option(DDX_TEST_PACKAGE "Build the out-of-tree consumer tests" ${PROJECT_IS_TOP_LEVEL})
if (DDX_TEST_PACKAGE AND DDX_INSTALL AND DDX_SANITIZE STREQUAL "off")
    set(DDX_PACKAGE_COMPONENTS rt dual)
    if (DDX_BUILD_JIT)
        list(APPEND DDX_PACKAGE_COMPONENTS jit)
    endif ()
    if (DDX_OPENCL)
        list(APPEND DDX_PACKAGE_COMPONENTS cl)
    endif ()
    # $<SEMICOLON>: an escaped separator does not survive add_test()'s expansion.
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
    set(DDX_PACKAGE_PROJECT --build-project ddx_package_consumer)

    add_test(NAME package.build_tree
            COMMAND ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG}
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                             "${CMAKE_CURRENT_BINARY_DIR}/package/build_tree"
            --build-generator ${CMAKE_GENERATOR} ${DDX_PACKAGE_PROJECT}
            --build-options -Dddx_DIR=${PROJECT_BINARY_DIR} ${DDX_PACKAGE_COMMON_OPTIONS}
            --test-command ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG} --output-on-failure)

    # Emptied first: an install takes nothing out of a prefix, and all_headers globs it.
    set(DDX_PACKAGE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package/prefix")
    add_test(NAME package.install_clean
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${DDX_PACKAGE_PREFIX})
    add_test(NAME package.install
            COMMAND ${CMAKE_COMMAND} --install ${PROJECT_BINARY_DIR}
                    ${DDX_PACKAGE_INSTALL_CONFIG} --prefix ${DDX_PACKAGE_PREFIX})
    set_tests_properties(package.install_clean PROPERTIES FIXTURES_SETUP ddx_installed)
    set_tests_properties(package.install PROPERTIES FIXTURES_SETUP ddx_installed
            DEPENDS package.install_clean)

    add_test(NAME package.install_tree
            COMMAND ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG}
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                             "${CMAKE_CURRENT_BINARY_DIR}/package/install_tree"
            --build-generator ${CMAKE_GENERATOR} ${DDX_PACKAGE_PROJECT}
            --build-options -DCMAKE_PREFIX_PATH=${DDX_PACKAGE_PREFIX} ${DDX_PACKAGE_COMMON_OPTIONS}
            --test-command ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG} --output-on-failure)
    set_tests_properties(package.install_tree PROPERTIES FIXTURES_REQUIRED ddx_installed)

    # A component this build lacks must fail at configure time.
    if (NOT DDX_BUILD_JIT)
        add_test(NAME package.missing_component
                COMMAND ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG}
                --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                                 "${CMAKE_CURRENT_BINARY_DIR}/package/missing_component"
                --build-generator ${CMAKE_GENERATOR} ${DDX_PACKAGE_PROJECT}
                --build-options -Dddx_DIR=${PROJECT_BINARY_DIR}
                                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                                -DBoost_DIR=${Boost_DIR}
                                -DDDX_CONSUMER_COMPONENTS=jit)
        set_tests_properties(package.missing_component PROPERTIES WILL_FAIL TRUE)
    endif ()
    if (NOT DDX_OPENCL)
        add_test(NAME package.missing_component_cl
                COMMAND ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG}
                --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                                 "${CMAKE_CURRENT_BINARY_DIR}/package/missing_component_cl"
                --build-generator ${CMAKE_GENERATOR} ${DDX_PACKAGE_PROJECT}
                --build-options -Dddx_DIR=${PROJECT_BINARY_DIR}
                                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                                -DDDX_CONSUMER_COMPONENTS=cl)
        set_tests_properties(package.missing_component_cl PROPERTIES WILL_FAIL TRUE)
    endif ()
endif ()
