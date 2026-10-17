# The extension module, a consumer of libddx.  The wheel is these two files:
#   ddx/_ddx.cpython-*.so   these bindings
#   ddx/libddx.so.1         the library they link

# A Windows wheel is MSVC-built: anything else links a runtime the wheel does
# not carry, and the import fails on every machine but the builder's.
if (WIN32 AND NOT MSVC)
    message(FATAL_ERROR "the Python module on Windows needs MSVC, not "
                        "${CMAKE_CXX_COMPILER_ID}; run from a developer prompt")
endif ()

pybind11_add_module(ddx_python src/module.cpp)
set_target_properties(ddx_python PROPERTIES OUTPUT_NAME _ddx)
target_link_libraries(ddx_python PRIVATE ddx::rt)
if (DDX_BUILD_JIT)
    target_link_libraries(ddx_python PRIVATE ddx::jit)
endif ()

ddx_target_flags(ddx_python EXCEPTIONS)
ddx_runtime_deps(ddx_python)

set_target_properties(ddx_python PROPERTIES
        INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>")
install(TARGETS ddx_python LIBRARY DESTINATION ddx)
install(TARGETS ddx_rt
        LIBRARY DESTINATION ddx NAMELINK_SKIP
        RUNTIME DESTINATION ddx
        ARCHIVE DESTINATION ddx EXCLUDE_FROM_ALL)
# A wheel holds no symlinks: the real file takes the soname.
if (NOT WIN32)
    install(CODE "file(RENAME
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/ddx/$<TARGET_FILE_NAME:ddx_rt>\"
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/ddx/$<TARGET_SONAME_FILE_NAME:ddx_rt>\")")
endif ()

# --- the package, in the build tree ------------------------------------------
# $<1:...> keeps a multi-config generator from appending Release/.
set(DDX_PYTHON_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/ddx")
set_target_properties(ddx_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "$<1:${DDX_PYTHON_PACKAGE}>")

file(GLOB DDX_PYTHON_SOURCES CONFIGURE_DEPENDS "ddx/*.py" "ddx/py.typed" "ddx/*.pyi")
add_custom_target(ddx_python_package ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DDX_PYTHON_PACKAGE}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DDX_PYTHON_SOURCES} "${DDX_PYTHON_PACKAGE}"
        DEPENDS ${DDX_PYTHON_SOURCES}
        COMMENT "Staging the ddx package beside _ddx")
add_dependencies(ddx_python_package ddx_python)

# --- pytest, as a ctest -------------------------------------------------------
add_test(NAME python.pytest
        COMMAND "${Python_EXECUTABLE}" -m pytest -q
        "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set_tests_properties(python.pytest PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};PYTHONDONTWRITEBYTECODE=1;DDX_PACKAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}")

# --- stubs --------------------------------------------------------------------
# On request; the result is checked in.  Against the bare extension, not the package.
add_custom_target(ddx_python_stubs
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${DDX_PYTHON_PACKAGE}"
        "${Python_EXECUTABLE}" -m pybind11_stubgen _ddx
        -o "${CMAKE_CURRENT_BINARY_DIR}/stubs"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/stubs/_ddx.pyi"
        "${CMAKE_CURRENT_SOURCE_DIR}/ddx/_ddx.pyi"
        DEPENDS ddx_python
        COMMENT "Regenerating python/ddx/_ddx.pyi")
