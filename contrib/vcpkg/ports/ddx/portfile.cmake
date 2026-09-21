# libddx is a shared library on every platform.
vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO reach2sayan/ddx
    REF "v${VERSION}"
    SHA512 22060159825396d19e15ca806c131ff1a7ef80f49c77bd16ac9c308e52898ca6f777be5f74cceab528dcd95df3d486ff0750b75ca7030fe8380b30d5f1b838d5
    HEAD_REF main
)

# DDX_BOOST_INCLUDEDIR replaces the pinned Boost fetch with vcpkg's headers and
# keeps them out of the install; ENABLE_NATIVE_ARCH=OFF pins x86-64-v3, which a
# binary cache requires.  No JIT: ddx wants exactly LLVM 20 and vcpkg's llvm
# port is older.
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        "-DDDX_BOOST_INCLUDEDIR=${CURRENT_INSTALLED_DIR}/include"
        -DDDX_BUILD_TESTS=OFF
        -DDDX_BUILD_BENCHMARKS=OFF
        -DENABLE_NATIVE_ARCH=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/ddx)

# ddx-config.cmake bakes DDX_BOOST_INCLUDEDIR in absolute; retarget it to the
# consumer's prefix, where the same Boost headers live.
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/ddx/ddx-config.cmake"
    "${CURRENT_INSTALLED_DIR}/include"
    [[${PACKAGE_PREFIX_DIR}/include]]
    IGNORE_UNCHANGED
)

vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt" "${SOURCE_PATH}/THIRD-PARTY-NOTICES.txt")
