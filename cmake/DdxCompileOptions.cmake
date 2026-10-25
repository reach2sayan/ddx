# The flags our own targets are built with, applied by ddx_target_flags().
include_guard(GLOBAL)

option(ENABLE_NATIVE_ARCH "Build optimized for this machine" ON)
option(DDX_FP_FLAGS "Pin FP contraction and drop errno on libm calls" ON)

if (MSVC)
    set(DDX_CODEGEN_FLAGS /arch:AVX2 /bigobj)
    set(DDX_WARNINGS /Wall /wd4061 /wd4623 /wd4625 /wd4626 /wd5026 /wd5027
                     /wd4710 /wd4711 /wd4868 /wd4820 /wd5045 /wd5246 /wd4514
                     /wd4324 /wd5266 /wd4866 /wd4371 /wd4686)
    # False positives: C4548 on C++23 `m[i, j]`, C5264 on `named<"x">`.
    list(APPEND DDX_WARNINGS /wd4548 /wd5264)
    list(APPEND DDX_WARNINGS /external:anglebrackets /external:W0)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    list(APPEND DDX_CODEGEN_FLAGS /EHs-c- /D_HAS_EXCEPTIONS=0 /wd4577)
    # Overriding CMake's default /EHsc per target is warning D9025; drop it here.
    string(REGEX REPLACE "[/-]EHsc *" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
else ()
    set(DDX_CODEGEN_FLAGS "")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        if (ENABLE_NATIVE_ARCH)
            set(DDX_CODEGEN_FLAGS -march=native)
        else ()
            set(DDX_CODEGEN_FLAGS -march=x86-64-v3)
        endif ()
    endif ()
    if (DDX_FP_FLAGS)
        list(APPEND DDX_CODEGEN_FLAGS -ffp-contract=fast -fno-math-errno)
    endif ()
    list(APPEND DDX_CODEGEN_FLAGS -fno-exceptions -funwind-tables)
    set(DDX_WARNINGS -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()

# Top level only.  The sloppiness lets ccache hit on TUs using the tests' PCH.
option(DDX_CCACHE "Compile through ccache when it is found" ON)
if (DDX_CCACHE AND PROJECT_IS_TOP_LEVEL AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(DDX_CCACHE_PROGRAM ccache)
    if (DDX_CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER ${DDX_CCACHE_PROGRAM}
            sloppiness=pch_defines,time_macros,include_file_mtime,include_file_ctime)
    endif ()
endif ()

# Global: a sanitizer has to instrument everything, gtest included.
set(DDX_SANITIZE "off" CACHE STRING "off | thread | address | undefined")
set_property(CACHE DDX_SANITIZE PROPERTY STRINGS off thread address undefined)
if (NOT DDX_SANITIZE STREQUAL "off")
    if (MSVC)
        message(WARNING "DDX_SANITIZE=${DDX_SANITIZE} ignored: MSVC has no such sanitizer")
    else ()
        set(sanitize -fsanitize=${DDX_SANITIZE})
        # The null checks stop GCC folding `&member == 0` in constant expressions.
        if (DDX_SANITIZE STREQUAL "undefined")
            list(APPEND sanitize -fno-sanitize=null,nonnull-attribute,returns-nonnull-attribute)
        endif ()
        add_compile_options(${sanitize} -fno-omit-frame-pointer -g)
        add_link_options(${sanitize})
    endif ()
endif ()

# ddx_target_flags(<target> [EXCEPTIONS])
# EXCEPTIONS keeps exceptions on for one target (pybind11, Boost.Parser).
function(ddx_target_flags target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "EXCEPTIONS" "" "")
    set(flags ${DDX_CODEGEN_FLAGS})
    if (arg_EXCEPTIONS)
        list(REMOVE_ITEM flags -fno-exceptions /EHs-c- /D_HAS_EXCEPTIONS=0)
        if (MSVC)
            list(APPEND flags /EHsc)
        endif ()
    endif ()
    target_compile_options(${target} PRIVATE ${flags} ${DDX_WARNINGS})
    set_property(TARGET ${target} PROPERTY COMPILE_WARNING_AS_ERROR ON)
endfunction()

# A shared libddx has to be findable at run time by every executable linking it.
function(ddx_runtime_deps target)
    set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "$ORIGIN")
    if (WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
                COMMAND_EXPAND_LISTS)
    endif ()
endfunction()
