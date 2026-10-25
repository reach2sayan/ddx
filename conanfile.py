import os
import re

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import load

# `conan create .` packages this tree; `conan build .` is how CI builds it.
# jit=True needs LLVM 20, which ConanCenter lacks:
#   conan remote add ddx contrib/conan --type=local-recipes-index
# boost=False leaves Boost to CMake, which finds or fetches it as for any user.
required_conan_version = ">=2.2"


class DdxConan(ConanFile):
    name = "ddx"
    description = "Symbolic and automatic differentiation for C++23 over a runtime expression graph"
    license = "BSL-1.0"
    url = "https://github.com/reach2sayan/ddx"
    homepage = "https://github.com/reach2sayan/ddx"
    topics = ("automatic-differentiation", "autodiff", "symbolic", "jacobian", "hessian")
    package_type = "shared-library"
    settings = "os", "arch", "compiler", "build_type"
    options = {"jit": [True, False], "boost": [True, False],
               "tests": [True, False], "python": [True, False]}
    default_options = {
        "jit": False,
        "boost": True,
        "tests": False,
        "python": False,
        "boost/*:header_only": True,
        # What libddx links: static, no RTTI or EH, nothing past zlib and zstd.
        "llvm-core/*:shared": False,
        "llvm-core/*:rtti": False,
        "llvm-core/*:exceptions": False,
        "llvm-core/*:with_ffi": False,
        "llvm-core/*:with_libedit": False,
        "llvm-core/*:with_xml2": False,
        "llvm-core/*:with_z3": False,
    }
    test_package_folder = "tests/conan"
    exports_sources = ("CMakeLists.txt", "LICENSE.txt", "THIRD-PARTY-NOTICES.txt",
                       "cmake/*", "include/*", "src/*")

    def set_version(self):
        cmake = load(self, os.path.join(self.recipe_folder, "CMakeLists.txt"))
        self.version = re.search(r"project\(ddx VERSION (\S+)", cmake).group(1)

    def configure(self):
        if self.options.jit:
            native = {"x86_64": "X86", "armv8": "AArch64"}.get(str(self.settings.arch))
            if native:
                self.options["llvm-core"].targets = native

    def package_id(self):
        del self.info.options.tests
        del self.info.options.python

    def validate(self):
        check_min_cppstd(self, 23)

    def requirements(self):
        if self.options.boost:
            # The installed headers include Boost's, so a consumer needs it too.
            self.requires("boost/1.91.0", transitive_headers=True)
        if self.options.jit:
            self.requires("llvm-core/20.1.8")

    def layout(self):
        cmake_layout(self)
        # Editable: the build tree's own ddx-config.cmake, which export() writes there.
        self.cpp.source.includedirs = ["include"]
        self.cpp.build.builddirs = ["."]

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["DDX_BUILD_TESTS"] = bool(self.options.tests)
        tc.cache_variables["DDX_VERIFY_HEADERS"] = bool(self.options.tests)
        tc.cache_variables["DDX_BUILD_JIT"] = bool(self.options.jit)
        tc.cache_variables["DDX_BUILD_PYTHON"] = bool(self.options.python)
        # A binary package cannot assume the build machine's CPU.
        tc.cache_variables["ENABLE_NATIVE_ARCH"] = False
        tc.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        if self.options.tests:
            cmake.build(target="all_verify_interface_header_sets")
            cmake.build(target="tests")
            cmake.ctest(cli_args=["--output-on-failure"])

    def package(self):
        CMake(self).install()

    def package_info(self):
        # Consumers get ddx's own ddx-config.cmake: its components, its flags.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = [os.path.join("lib", "cmake", "ddx")]
        self.cpp_info.libs = ["ddx"]
        if self.settings.os in ("Linux", "FreeBSD"):
            self.cpp_info.system_libs = ["pthread"]
