# The floor the toolchain must clear: <expected>.
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

set(CMAKE_REQUIRED_FLAGS "-std=c++23")
check_cxx_source_compiles(
        "#include <expected>
         int main() { return std::expected<int, int>{0}.value(); }"
        DDX_TOOLCHAIN_OK)
unset(CMAKE_REQUIRED_FLAGS)

if (NOT MSVC AND NOT DDX_TOOLCHAIN_OK)
    message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} cannot build ddx, which "
            "needs <expected>.  Build with GCC 14+ or Clang 20+.  "
            "The compiler's own diagnostic is in the CMakeConfigureLog.")
endif ()
