import os
import re

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import load

# `conan create .` packages this tree; `conan editable add .` consumes it in place.
# libddx is shared on every platform, and the JIT stays off: it wants exactly
# LLVM 20, which ConanCenter does not carry -- the same build as the vcpkg port.
required_conan_version = ">=2.0"


class DdxConan(ConanFile):
    name = "ddx"
    description = "Symbolic and automatic differentiation for C++23 over a runtime expression graph"
    license = "BSL-1.0"
    url = "https://github.com/reach2sayan/ddx"
    homepage = "https://github.com/reach2sayan/ddx"
    topics = ("automatic-differentiation", "autodiff", "symbolic", "jacobian", "hessian")
    package_type = "shared-library"
    settings = "os", "arch", "compiler", "build_type"
    default_options = {"boost/*:header_only": True}
    test_package_folder = "tests/conan"
    exports_sources = ("CMakeLists.txt", "LICENSE.txt", "THIRD-PARTY-NOTICES.txt",
                       "cmake/*", "include/*", "src/*")

    def set_version(self):
        cmake = load(self, os.path.join(self.recipe_folder, "CMakeLists.txt"))
        self.version = re.search(r"project\(ddx VERSION (\S+)", cmake).group(1)

    def validate(self):
        check_min_cppstd(self, 23)

    def requirements(self):
        # The installed headers include Boost's, so a consumer needs it too.
        self.requires("boost/1.91.0", transitive_headers=True)

    def layout(self):
        cmake_layout(self)
        # Editable: the build tree's own ddx-config.cmake, which export() writes there.
        self.cpp.source.includedirs = ["include"]
        self.cpp.build.builddirs = ["."]

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["DDX_BUILD_TESTS"] = False
        tc.cache_variables["DDX_BUILD_JIT"] = False
        # A binary package cannot assume the build machine's CPU.
        tc.cache_variables["ENABLE_NATIVE_ARCH"] = False
        tc.generate()
        # find_package(Boost CONFIG) in DdxDependencies finds this one and fetches nothing.
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        CMake(self).install()

    def package_info(self):
        # Consumers get ddx's own ddx-config.cmake: its components, its flags.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = [os.path.join("lib", "cmake", "ddx")]
        self.cpp_info.libs = ["ddx"]
        if self.settings.os in ("Linux", "FreeBSD"):
            self.cpp_info.system_libs = ["pthread"]
