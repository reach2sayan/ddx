include_guard(GLOBAL)

include(CMakePackageConfigHelpers)
include(DdxBoostHeaders)

# ddx_rt is libddx; the JIT objects are inside it, so ddx::jit exports nothing.
set(DDX_EXPORT_TARGETS ddx ddx_util ddx_ops ddx_md ddx_symbolic ddx_dual ddx_rt)

install(TARGETS ${DDX_EXPORT_TARGETS}
        EXPORT ddxTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILE_SET HEADERS)

# The Boost subset the installed headers reach -- only when ddx fetched it.  A
# supplied Boost is the consumer's; the config below asks for it again.
if (DDX_BOOST_VENDORED)
    foreach (dir IN LISTS DDX_BOOST_HEADER_DIRS)
        file(GLOB headers LIST_DIRECTORIES false "${DDX_BOOST_ROOT}/${dir}/*")
        if (NOT headers)
            message(FATAL_ERROR
                    "DdxBoostHeaders names ${dir}, which holds no files under "
                    "${DDX_BOOST_ROOT}.  The list and the tree disagree; "
                    "regenerate it with `python3 scripts/boost_headers.py`.")
        endif ()
        install(FILES ${headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${dir}")
    endforeach ()
endif ()

install(EXPORT ddxTargets
        NAMESPACE ddx::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)

install(FILES "${PROJECT_SOURCE_DIR}/LICENSE.txt" "${PROJECT_SOURCE_DIR}/THIRD-PARTY-NOTICES.txt"
        DESTINATION ${CMAKE_INSTALL_DOCDIR})

# Two configs from one template: the build tree and the prefix hold a fetched
# Boost in different places.
set(DDX_BOOST_CONFIG_ROOT "${DDX_BOOST_ROOT}")
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/ddx-config.cmake.in"
        "${PROJECT_BINARY_DIR}/ddx-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)

if (DDX_BOOST_VENDORED)
    set(DDX_BOOST_CONFIG_ROOT "\${PACKAGE_PREFIX_DIR}/${CMAKE_INSTALL_INCLUDEDIR}")
endif ()
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/ddx-config.cmake.in"
        "${PROJECT_BINARY_DIR}/install/ddx-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)
write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/ddx-config-version.cmake"
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)
install(FILES
        "${PROJECT_BINARY_DIR}/install/ddx-config.cmake"
        "${PROJECT_BINARY_DIR}/ddx-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)

export(EXPORT ddxTargets NAMESPACE ddx:: FILE "${PROJECT_BINARY_DIR}/ddxTargets.cmake")
