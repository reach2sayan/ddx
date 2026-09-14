# libddx is a shared library on every platform.
vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO reach2sayan/ddx
    REF "v${VERSION}"
    SHA512 c4a723e99476e6bd2115bc9f8c31f5311ca3f7971994e220701bb6889715f7daf7aaef1f104c0f549714767c52e671636be47c5b0e9ee57670213c36d81d1e79
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
